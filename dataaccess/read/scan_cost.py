"""
data_access.read.scan_cost —— 读路径成本估算与执行路由

职责
    1. 估算一次读的扫描成本：file_count / total_bytes / estimated_rows /
       projected_columns / total_columns / remote / selectivity / 引擎启动成本
    2. 给出 (engine, result_mode) 路由建议——不只按行数，而是综合成本

设计要点
    1. estimated_rows 优先来自 manifest（不扫 footer）；否则 fallback
       ``dataset_read_stats`` 的 footer 估算。
    2. result_mode：
       - 小（行数少 + 本地）→ arrow（一次 materialize）
       - 大（行数多 / 结果字节大）→ stream
       - 宽表 + prefer_polars → lazy（Polars pushdown）
    3. engine：
       - 格式 arrow/feather → pyarrow
       - prefer_polars 且体量大 → polars
       - 默认 duckdb（共享连接，footer cache 复用）

非职责
    不做真正的文件 IO；不替换 QueryBudget（那是硬拦截，这里是软路由）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from data_access.core.storage import is_remote_storage
from data_access.read.formats import format_adapter_for_dataset
from data_access.read.query_budget import QueryBudget

if TYPE_CHECKING:
    from .store import DataAccessStore

logger = logging.getLogger("data_access.scan_cost")

_ARROW_MAX_ROWS = 5_000_000
_STREAM_MIN_ROWS = 1_000_000
_ENGINE_STARTUP_MS = {"duckdb": 5.0, "polars": 15.0, "pyarrow": 3.0}


@dataclass(frozen=True)
class ScanCost:
    dataset: str
    file_count: int
    total_bytes: int | None
    estimated_rows: int
    projected_columns: int
    total_columns: int | None
    remote: bool
    selectivity: float | None = None      # time_range/instrument 估算选择率
    instrument_count: int = 0
    engine_startup_ms: float = 5.0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "estimated_rows": self.estimated_rows,
            "projected_columns": self.projected_columns,
            "total_columns": self.total_columns,
            "remote": self.remote,
            "selectivity": self.selectivity,
            "instrument_count": self.instrument_count,
            "engine_startup_ms": self.engine_startup_ms,
            "score": self.score,
        }


def estimate_scan_cost(
    store: "DataAccessStore",
    dataset: str,
    *,
    columns: Sequence[str] | None = None,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    prefer_polars: bool = False,
    **params: Any,
) -> ScanCost:
    """估算一次读的扫描成本并计算路由分。"""
    ds = store._registry.get(dataset)
    try:
        stats = store.dataset_read_stats(
            dataset,
            columns=list(columns) if columns else None,
            time_range=time_range,
            prefer_polars=prefer_polars,
            **params,
        )
        estimated_rows = stats.estimated_rows
        file_count = stats.parquet_files
    except Exception:
        estimated_rows = 0
        file_count = 0

    # manifest 有更准的字节数
    total_bytes: int | None = None
    try:
        from data_access.read.manifest import load_manifest_for_dataset

        manifest = load_manifest_for_dataset(store, dataset, **params)
        if manifest is not None:
            total_bytes = manifest.total_bytes
    except Exception:
        pass

    schema = getattr(ds, "schema", None) or {}
    total_columns = len(schema) if schema else None
    projected = len(columns) if columns else 0

    remote = is_remote_storage(ds)
    startup = _ENGINE_STARTUP_MS.get("polars" if prefer_polars else "duckdb", 5.0)

    selectivity = 1.0
    if time_range is not None:
        start, end = time_range
        if start is not None and end is not None:
            selectivity = 0.3  # 经验值：闭区间时间窗典型截掉 ~70%
        elif start is not None or end is not None:
            selectivity = 0.5

    score = 0.0
    # 成本分：行数 * 投影比例 * 远程惩罚 * 选择率
    rows_factor = max(estimated_rows, 0)
    col_factor = (projected / total_columns) if (projected and total_columns) else 1.0
    remote_factor = 3.0 if remote else 1.0
    score = rows_factor * col_factor * remote_factor * (selectivity or 1.0)

    return ScanCost(
        dataset=dataset,
        file_count=file_count,
        total_bytes=total_bytes,
        estimated_rows=estimated_rows,
        projected_columns=projected,
        total_columns=total_columns,
        remote=remote,
        selectivity=selectivity,
        instrument_count=len(instrument_filter) if instrument_filter else 0,
        engine_startup_ms=startup,
        score=score,
    )


def suggest_read_strategy(
    cost: ScanCost,
    *,
    prefer_polars: bool = False,
    engine: str = "auto",
    result: str = "auto",
) -> tuple[str, str]:
    """根据 ScanCost 建议 (engine, result_mode)。

    engine ∈ {duckdb, polars, pyarrow}；result_mode ∈ {arrow, pandas, polars, lazy, stream}。
    显式传入的 engine/result 直接返回（auto 才路由）。
    """
    if engine not in {"auto", "duckdb", "polars", "pyarrow"}:
        raise ValueError(f"engine 必须是 auto|duckdb|polars|pyarrow，收到 {engine!r}")
    if result not in {"auto", "arrow", "pandas", "polars", "lazy", "stream"}:
        raise ValueError(
            f"result 必须是 auto|arrow|pandas|polars|lazy|stream，收到 {result!r}"
        )

    if engine == "auto":
        if cost.file_count == 0 and cost.estimated_rows == 0:
            engine = "duckdb"
        elif prefer_polars and cost.estimated_rows >= _STREAM_MIN_ROWS:
            engine = "polars"
        else:
            engine = "duckdb"

    if result == "auto":
        if engine == "polars":
            result = "lazy"
        elif cost.estimated_rows >= _STREAM_MIN_ROWS or (
            cost.total_bytes is not None and cost.total_bytes >= 512 * 1024 * 1024
        ):
            result = "stream"
        elif cost.estimated_rows <= _ARROW_MAX_ROWS:
            result = "arrow"
        else:
            result = "stream"
    return engine, result


def log_routing(cost: ScanCost, engine: str, result: str) -> None:
    logger.info(
        "read_auto dataset=%s rows=%d files=%d bytes=%s engine=%s result=%s score=%.0f",
        cost.dataset,
        cost.estimated_rows,
        cost.file_count,
        cost.total_bytes,
        engine,
        result,
        cost.score,
    )
