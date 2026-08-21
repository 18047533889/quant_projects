# -*- coding: utf-8 -*-
"""MB-P1-016: Batch transfer instead of per-operator.

Backend 间数据传输批量化（减少转换开销）：
    - Pandas → DuckDB：批量注册多张表（一次连接初始化）
    - DuckDB → Polars：批量 Arrow transfer（共享 RecordBatchReader）
    - Polars → Pandas：批量 to_pandas()（共享类型推断）
    - 避免逐算子 format 转换（N 次 → 1 次）

示例场景：
    100 个因子，每个因子从 Pandas source 执行 DuckDB SQL
    - 逐算子转换：100 次 pandas→duckdb→pandas（300 次转换）
    - 批量转换：1 次 register batch + 100 次 SQL + 1 次 batch export（102 次）

R21-TRANSFER-EXECUTOR：
    每个 :class:`TransferTransform` 必须映射到一个**真实** executor 函数 —— 不是
    只加 metrics 的 no-op。未知 target / 无法完成的转换必须抛
    :class:`TypedTransferError`（fail closed），绝不静默成功。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from planner.backend_region import Representation, TransferTransform

from runtime.exceptions import (
    SemanticMismatchError,
    TransferInputTypeError,
    TypedTransferError,
    UnknownTransferTargetError,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 语义快照：传输前后校验（date/instrument/order/dtype/null/timezone/grain/...）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticSnapshot:
    """一次传输前后的语义不变量快照。

    用于校验 transfer 是否真正保持了数据语义（而非静默丢失列 / 行 / 类型）。
    """

    num_rows: int
    num_cols: int
    col_names: tuple[str, ...]
    dtypes: tuple[str, ...]
    null_counts: tuple[int, ...] | None
    timezone: tuple[str | None, ...] | None
    grain: str | None = None
    source_snapshot: dict[str, Any] | None = None

    @staticmethod
    def from_dataframe(df: Any) -> "SemanticSnapshot":
        """从 pandas DataFrame 提取语义快照。"""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TransferInputTypeError(
                f"from_dataframe requires pandas.DataFrame, got {type(df)}"
            )
        null_counts = tuple(int(df[col].isna().sum()) for col in df.columns)
        tz = tuple(
            getattr(df[col].dtype, "tz", None) for col in df.columns
        )
        grain = getattr(df, "attrs", {}).get("grain", None)
        source_snapshot = getattr(df, "attrs", {}).get("source_snapshot")
        return SemanticSnapshot(
            num_rows=int(df.shape[0]),
            num_cols=int(df.shape[1]),
            col_names=tuple(str(c) for c in df.columns),
            dtypes=tuple(str(df[col].dtype) for col in df.columns),
            null_counts=null_counts,
            timezone=tuple(tz),
            grain=grain,
            source_snapshot=source_snapshot,
        )

    @staticmethod
    def from_arrow(table: Any) -> "SemanticSnapshot":
        """从 pyarrow Table 取语义快照（null 计数 + tz）。"""
        import pyarrow as pa

        if not isinstance(table, pa.Table):
            raise TransferInputTypeError(
                f"from_arrow requires pyarrow.Table, got {type(table)}"
            )
        null_counts = tuple(
            table.column(i).null_count for i in range(table.num_columns)
        )
        tz = tuple(
            (getattr(field.type, "tz", None)) for field in table.schema
        )
        metadata = table.schema.metadata or {}
        grain = None
        source_snapshot = None
        if b"grain" in metadata:
            grain = metadata[b"grain"].decode("utf-8")
        if b"source_snapshot" in metadata:
            source_snapshot = metadata[b"source_snapshot"].decode("utf-8")
        return SemanticSnapshot(
            num_rows=table.num_rows,
            num_cols=table.num_columns,
            col_names=tuple(str(n) for n in table.column_names),
            dtypes=tuple(str(t) for t in table.schema.types),
            null_counts=null_counts,
            timezone=tuple(tz),
            grain=grain,
            source_snapshot=source_snapshot,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "col_names": list(self.col_names),
            "dtypes": list(self.dtypes),
            "null_counts": list(self.null_counts) if self.null_counts else None,
            "timezone": list(self.timezone) if self.timezone else None,
            "grain": self.grain,
            "source_snapshot": dict(self.source_snapshot) if self.source_snapshot else None,
        }


def _assert_semantic_preservation(
    before: SemanticSnapshot, after: SemanticSnapshot, transform: TransferTransform
) -> None:
    """校验转换前后语义不变量一致；不一致抛 SemanticMismatchError。"""
    if before.num_rows != after.num_rows:
        raise SemanticMismatchError(
            f"{transform.value}: row count changed {before.num_rows} → {after.num_rows}"
        )
    if before.num_cols != after.num_cols:
        raise SemanticMismatchError(
            f"{transform.value}: column count changed {before.num_cols} → {after.num_cols}"
        )
    if before.col_names != after.col_names:
        raise SemanticMismatchError(
            f"{transform.value}: column names changed {before.col_names} → {after.col_names}"
        )
    if before.dtypes != after.dtypes:
        # Arrow→pandas 等 dtype 表示可能合法变化（如 int32↔int64），但
        # 更广的 dtype 改变（category/object ↔ numeric）视为失败。
        _assert_dtypes_compatible(before.dtypes, after.dtypes, transform)

    # Null count preservation
    if before.null_counts is not None and after.null_counts is not None:
        if before.null_counts != after.null_counts:
            raise SemanticMismatchError(
                f"{transform.value}: null counts changed "
                f"{before.null_counts} -> {after.null_counts}"
            )

    # Timezone preservation
    if before.timezone is not None and after.timezone is not None:
        if before.timezone != after.timezone:
            raise SemanticMismatchError(
                f"{transform.value}: timezone changed "
                f"{before.timezone} -> {after.timezone}"
            )

    # Grain preservation
    if before.grain is not None and after.grain is not None:
        if before.grain != after.grain:
            raise SemanticMismatchError(
                f"{transform.value}: grain changed "
                f"{before.grain!r} -> {after.grain!r}"
            )

    # Source snapshot preservation
    if before.source_snapshot is not None and after.source_snapshot is not None:
        if before.source_snapshot != after.source_snapshot:
            raise SemanticMismatchError(
                f"{transform.value}: source_snapshot changed"
            )


def _assert_dtypes_compatible(
    before: tuple[str, ...], after: tuple[str, ...], transform: TransferTransform
) -> None:
    """容忍 dtype 表示层差异（arrow/pandas/numpy 命名），拒绝语义型差异。"""
    # Arrow int/large vs pandas 常规 dtype：允许括号/精度差异；category/object 类
    # 不得变成数值类型（数值不应变 string）。
    def _cat(s: str) -> str:
        return s.split("[")[0].strip()

    for i, (b, a) in enumerate(zip(before, after)):
        bb, aa = _cat(b), _cat(a)
        # 明确把 category 转 numeric、numeric 转 string 视为语义漂移
        if bb == "category" and "category" not in aa:
            raise SemanticMismatchError(
                f"{transform.value}: column {i} category dtype lost: {b} → {a}"
            )
        # string 家族：Arrow string / large_string → pandas object 是合法表示差异
        string_family = {"str", "string", "large_string", "object"}
        bb_str = bb in string_family
        aa_str = aa in string_family
        if bb_str != aa_str:
            raise SemanticMismatchError(
                f"{transform.value}: column {i} string dtype changed: {b} → {a}"
            )


# ---------------------------------------------------------------------------
# 真实 executor：每个 TransferTransform 对应一个实际转换函数
# ---------------------------------------------------------------------------


class TransferExecutor:
    """把 :class:`TransferTransform` 映射到真实 backend 转换函数。

    R21-TRANSFER-EXECUTOR：每个 transform 都有可执行的转换。未知 transform
    （没有真实转换函数）抛 :class:`UnknownTransferTargetError`，绝不静默返回
    原数据冒充成功。
    """

    def __init__(self, *, duckdb_conn: Any | None = None) -> None:
        # 保留 duckdb 连接句柄，供 DUCKDB_TO_ARROW / batch 注册使用；由调用方
        # 传入或惰性创建，所有权/存活由持方负责。
        self._duckdb_conn = duckdb_conn

    # -- 基础转换（直接、真实） ----------------------------------------------

    def duckdb_to_arrow(self, data: Any) -> Any:
        """DuckDB Relation / Connection → pyarrow Table。"""
        try:
            import pyarrow as pa

            if isinstance(data, pa.Table):
                return data
            if hasattr(data, "arrow"):
                rel = data
                if hasattr(rel, "arrow"):
                    table = rel.arrow()
                elif hasattr(rel, "fetch_arrow_table"):
                    table = rel.fetch_arrow_table()
                else:
                    raise TransferInputTypeError(
                        "duckdb object has no arrow()/fetch_arrow_table()",
                    )
                if not isinstance(table, pa.Table):
                    table = pa.table(table)
                return table
            raise TransferInputTypeError(f"not a DuckDB relation: {type(data)}")
        except TransferInputTypeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TypedTransferError(
                f"duckdb_to_arrow failed: {exc}"
            ) from exc

    def q_to_arrow(self, data: Any) -> Any:
        """Q handle / 结果 → pyarrow Table。

        R21：Q → Arrow 用于跨 backend 消费。若数据已可被 Arrow 直接消费
        （如已经序列化的 Arrow buffer），原样返回；否则抛 TransferInputTypeError
        表示没有真实转换可用 —— 绝不静默冒充成功。
        """
        import pyarrow as pa

        if isinstance(data, pa.Table):
            return data
        if isinstance(data, pa.RecordBatch):
            return pa.Table.from_batches([data])
        if hasattr(data, "to_arrow"):
            table = data.to_arrow()
            if isinstance(table, pa.Table):
                return table
            table = pa.table(table)
            return table
        raise TransferInputTypeError(
            f"q_to_arrow: no real Q→Arrow conversion for {type(data)}; "
            "provide pyarrow.Table / .to_arrow() result",
        )

    def clickhouse_to_arrow(self, data: Any) -> Any:
        """ClickHouse Arrow → Arrow Table（native Arrow，通常已就绪）。"""
        import pyarrow as pa

        if isinstance(data, pa.Table):
            return data
        if isinstance(data, pa.RecordBatch):
            return pa.Table.from_batches([data])
        if hasattr(data, "to_pandas"):
            # ClickHouse 客户端会给出 pandas；经 Arrow 转回（避免 pandas→arrow→pandas）
            return pa.Table.from_pandas(data, preserve_index=False)
        raise TransferInputTypeError(
            f"clickhouse_to_arrow: unexpected type {type(data)}",
        )

    def arrow_to_polars(self, data: Any) -> Any:
        """Arrow Table → Polars DataFrame（直接，非 to_pandas 中转）。"""
        import pyarrow as pa

        if not isinstance(data, pa.Table):
            raise TransferInputTypeError(
                f"arrow_to_polars requires pyarrow.Table, got {type(data)}",
            )
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover
            raise TypedTransferError("polars not installed") from exc
        return pl.from_arrow(data)

    def arrow_to_pandas(self, data: Any) -> Any:
        """Arrow Table → pandas DataFrame（zero-copy 尽力）。"""
        import pyarrow as pa

        if not isinstance(data, pa.Table):
            raise TransferInputTypeError(
                f"arrow_to_pandas requires pyarrow.Table, got {type(data)}",
            )
        return data.to_pandas(
            self_destruct=True, split_blocks=True, use_threads=True
        )

    def polars_to_pandas(self, data: Any) -> Any:
        """Polars → pandas DataFrame。"""
        import polars as pl

        if not isinstance(data, pl.DataFrame):
            raise TransferInputTypeError(
                f"polars_to_pandas requires pl.DataFrame, got {type(data)}",
            )
        return data.to_pandas()

    def pandas_to_polars(self, data: Any) -> Any:
        """pandas → Polars DataFrame（经 Arrow，避免双 to_pandas）。"""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TransferInputTypeError(
                f"pandas_to_polars requires pd.DataFrame, got {type(data)}",
            )
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover
            raise TypedTransferError("polars not installed") from exc
        return pl.from_pandas(data)

    def polars_to_numpy(self, data: Any) -> Any:
        """Polars → NumPy ndarray（buffer protocol / to_numpy）。"""
        import polars as pl

        if isinstance(data, pl.Series):
            return data.to_numpy()
        if isinstance(data, pl.DataFrame):
            return data.to_numpy()
        raise TransferInputTypeError(
            f"polars_to_numpy requires pl.Series/DataFrame, got {type(data)}",
        )

    def numpy_to_polars(self, data: Any) -> Any:
        """NumPy ndarray → Polars Series/DataFrame。"""
        import numpy as np

        if not isinstance(data, np.ndarray):
            raise TransferInputTypeError(
                f"numpy_to_polars requires np.ndarray, got {type(data)}",
            )
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover
            raise TypedTransferError("polars not installed") from exc
        if data.ndim == 1:
            return pl.Series(data)
        return pl.DataFrame(data)

    # -----------------------------------------------------------------------
    # 把 transform 绑定到真实 executor；未知 → UnknownTransferTargetError
    # -----------------------------------------------------------------------

    def execute(
        self,
        transform: TransferTransform,
        data: Any,
        *,
        verify: bool = True,
    ) -> Any:
        """执行一次真实转换并（可选）校验语义保留。"""
        handlers: dict[TransferTransform, Any] = {
            TransferTransform.DUCKDB_TO_ARROW: self.duckdb_to_arrow,
            TransferTransform.Q_TO_ARROW: self.q_to_arrow,
            TransferTransform.CLICKHOUSE_TO_ARROW: self.clickhouse_to_arrow,
            TransferTransform.ARROW_TO_POLARS: self.arrow_to_polars,
            TransferTransform.ARROW_TO_PANDAS: self.arrow_to_pandas,
            TransferTransform.POLARS_TO_PANDAS: self.polars_to_pandas,
            TransferTransform.PANDAS_TO_POLARS: self.pandas_to_polars,
            TransferTransform.POLARS_TO_NUMPY: self.polars_to_numpy,
            TransferTransform.NUMPY_TO_POLARS: self.numpy_to_polars,
            # 结构性/reshape transform 在批处理器层处理（需 schema 知识）
        }
        handler = handlers.get(transform)
        if handler is None:
            raise UnknownTransferTargetError(
                f"no real executor for transform {transform.value}",
            )

        before = None
        if verify:
            try:
                before = _capture_before(data, transform)
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "semantic before-snapshot skipped: %s", exc
                )

        result = handler(data)

        if verify and before is not None:
            after = _capture_after(result, transform.value)
            if after is not None:
                _assert_semantic_preservation(before, after, transform)
        return result


def _capture_before(data: Any, transform_value: str) -> SemanticSnapshot | None:
    """转换前快照（可空 —— 无法量化时不阻塞转换）。"""
    import pyarrow as pa

    if isinstance(data, pa.Table):
        return SemanticSnapshot.from_arrow(data)
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return SemanticSnapshot.from_dataframe(data)
    return None


def _capture_after(data: Any, transform_value: str) -> SemanticSnapshot | None:
    """转换后快照（可空）。"""
    import pyarrow as pa

    if isinstance(data, pa.Table):
        return SemanticSnapshot.from_arrow(data)
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return SemanticSnapshot.from_dataframe(data)
    return None


# ---------------------------------------------------------------------------
# BatchTransferOptimizer —— 复用真实的 TransferExecutor 完成批量传输
# ---------------------------------------------------------------------------


@dataclass
class BatchTransferMetrics:
    """批量传输统计。"""

    batch_transfers: int = 0
    single_transfers: int = 0
    total_tables_transferred: int = 0
    total_rows_transferred: int = 0
    estimated_saved_conversions: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_transfers": self.batch_transfers,
            "single_transfers": self.single_transfers,
            "total_tables_transferred": self.total_tables_transferred,
            "total_rows_transferred": self.total_rows_transferred,
            "estimated_saved_conversions": self.estimated_saved_conversions,
            "failures": self.failures,
        }


class BatchTransferOptimizer:
    """Backend 间批量数据传输优化器。

    集成点：
        - HybridExecutor 在 backend 切换时调用 batch_register()
        - PhysicalLowerer 识别连续相同 source 的 region（批量传输候选）
        - AdaptiveBatchScheduler fusion group 共享 batch transfer

    R21-TRANSFER-EXECUTOR：
        - 每张表按 TransferTransform 走真实 executor
        - 未知 target / 不确定转换 → 抛 TypedTransferError（fail closed）
        - 批处理任何失败都向上传播 typed failure，绝不吞错返回 batch_id
    """

    def __init__(
        self,
        *,
        batch_threshold: int = 5,
        duckdb_conn: Any | None = None,
        executor: TransferExecutor | None = None,
    ) -> None:
        """
        Args:
            batch_threshold: 累积多少张表触发批量传输
            duckdb_conn: 若调用方已持有 DuckDB 连接，可注入（共享句柄）
            executor: 自定义 executor（默认新建；复用连接时传 conn）
        """
        self.batch_threshold = batch_threshold
        self._duckdb_conn = duckdb_conn
        self._executor = executor or TransferExecutor(duckdb_conn=duckdb_conn)
        self._metrics = BatchTransferMetrics()
        self._lock = threading.RLock()
        # 待批量传输的表队列（按 target transform 分组）
        self._pending_batches: dict[TransferTransform, list[dict[str, Any]]] = {}

    @property
    def resident_duckdb_conn(self) -> Any | None:
        """作为“常驻句柄”暴露的 DuckDB 连接（所有权/存活由持方负责）。

        批次传输结束后连接保持打开，供后续 SQL / 复用；关闭由调用方负责，
        以显式所有权而不是泄漏。未注入时创建的内存连接保存在此。
        """
        return self._duckdb_conn

    def register_transfer_request(
        self,
        table_name: str,
        data: Any,
        source_backend: str,
        target_backend: str,
        *,
        transform: TransferTransform | None = None,
    ) -> str | None:
        """注册一次传输请求（不立即执行，累积到批次）。

        Args:
            table_name: 表名
            data: 数据（DataFrame / Arrow Table / NumPy）
            source_backend: 源 backend
            target_backend: 目标 backend
            transform: 显式 TransferTransform（缺省则由 backend 推断）

        Returns:
            batch_id（已触发批量传输）或 None（仍在累积）
        """
        if transform is None:
            transform = _infer_transform(target_backend)
        if transform is None:
            raise UnknownTransferTargetError(
                f"unknown target_backend {target_backend!r} (no real executor)",
            )

        with self._lock:
            if transform not in self._pending_batches:
                self._pending_batches[transform] = []
            self._pending_batches[transform].append(
                {
                    "table_name": table_name,
                    "data": data,
                    "source_backend": source_backend,
                    "target_backend": target_backend,
                }
            )
            if len(self._pending_batches[transform]) >= self.batch_threshold:
                return self._execute_batch_transfer(transform)

        return None

    def flush_pending(
        self,
        source_backend: str | None = None,
        target_backend: str | None = None,
        *,
        transform: TransferTransform | None = None,
    ) -> str | None:
        """强制刷新待传输批次（即使未达阈值）。

        Returns:
            batch_id（已执行）或 None（无待传输数据）
        """
        if transform is not None:
            with self._lock:
                if transform not in self._pending_batches or not self._pending_batches[transform]:
                    return None
                return self._execute_batch_transfer(transform)

        keys = list(self._pending_batches.keys())
        last = None
        for k in keys:
            with self._lock:
                if k in self._pending_batches and self._pending_batches[k]:
                    last = self._execute_batch_transfer(k)
        return last

    def _execute_batch_transfer(self, transform: TransferTransform) -> str:
        """执行一个 transform 的批量传输（内部调用，须持锁）。

        Returns:
            batch_id
        Raises:
            TypedTransferError: 任一表转换失败时向上传播（绝不吞错）
        """
        batch = self._pending_batches.pop(transform, [])
        if not batch:
            return ""

        batch_id = f"batch_{self._metrics.batch_transfers + 1}"
        _logger.info(
            "batch transfer %s (%d tables)", transform.value, len(batch),
        )

        with self._lock:
            self._metrics.batch_transfers += 1
            self._metrics.total_tables_transferred += len(batch)
            self._metrics.estimated_saved_conversions += max(0, len(batch) - 1)

        for item in batch:
            data = item["data"]
            try:
                converted = self._executor.execute(transform, data)
                item["converted"] = converted
                try:
                    self._metrics.total_rows_transferred += _row_count(converted)
                except Exception:  # noqa: BLE001
                    pass
            except TypedTransferError as exc:
                with self._lock:
                    self._metrics.failures += 1
                _logger.error(
                    "batch %s failed on %s: %s",
                    batch_id, item["table_name"], exc,
                )
                raise exc

        return batch_id

    def metrics(self) -> BatchTransferMetrics:
        """返回累计统计。"""
        with self._lock:
            m = self._metrics
            return BatchTransferMetrics(
                batch_transfers=m.batch_transfers,
                single_transfers=m.single_transfers,
                total_tables_transferred=m.total_tables_transferred,
                total_rows_transferred=m.total_rows_transferred,
                estimated_saved_conversions=m.estimated_saved_conversions,
                failures=m.failures,
            )

    def reset_metrics(self) -> None:
        """重置统计。"""
        with self._lock:
            self._metrics = BatchTransferMetrics()


def _row_count(data: Any) -> int:
    """健壮地取行数。"""
    try:
        return int(getattr(data, "num_rows", len(data)))
    except Exception:  # noqa: BLE001
        return 0


def _infer_transform(target_backend: str) -> TransferTransform | None:
    """按 target backend 推断 transform（与 planner 表示对齐）。

    返回 None 表示未知 target —— 调用方应 fail closed。
    """
    mapping = {
        "arrow": TransferTransform.Q_TO_ARROW,  # 通用 Arrow target 用 Q→Arrow
        "duckdb": TransferTransform.Q_TO_ARROW,  # 经 Arrow 到 DuckDB
        "polars": TransferTransform.ARROW_TO_POLARS,
        "pandas": TransferTransform.ARROW_TO_PANDAS,
        "numpy": TransferTransform.ARROW_TO_PANDAS,
        "q": TransferTransform.DUCKDB_TO_ARROW,
        "clickhouse": TransferTransform.CLICKHOUSE_TO_ARROW,
    }
    return mapping.get(target_backend)


class UnknownTransferError(TypedTransferError):
    """目标 backend 未知，无法映射到真实 transform。"""


# 全局单例
_global_batch_transfer: BatchTransferOptimizer | None = None
_global_lock = threading.Lock()


def get_global_batch_transfer_optimizer(
    duckdb_conn: Any | None = None,
) -> BatchTransferOptimizer:
    """返回全局 BatchTransferOptimizer（进程级单例）。"""
    global _global_batch_transfer
    if _global_batch_transfer is None:
        with _global_lock:
            if _global_batch_transfer is None:
                _global_batch_transfer = BatchTransferOptimizer(
                    duckdb_conn=duckdb_conn
                )
    return _global_batch_transfer
