# -*- coding: utf-8 -*-
"""MB-P1-012: DuckDB Arrow boundary optimization.

DuckDB ↔ Arrow zero-copy 边界优化：
    - DuckDB 原生支持 Arrow 作为输入/输出（`arrow()` / `from_arrow()`）
    - 避免 Arrow → Pandas → DuckDB 双次转换（内存拷贝 + 类型转换开销）
    - Streaming Arrow batches（大数据集分批传输，不全部常驻内存）
    - Arrow Flight RPC（分布式场景）

本模块提供：
    1. Arrow RecordBatch 流式输入 DuckDB（分批 scan，不全部物化）
    2. DuckDB query 直接输出 Arrow Table（zero-copy，避免 Pandas 中转）
    3. Arrow schema 与 DuckDB schema 双向映射
    4. 批次大小自适应（内存压力高时减小 batch_size）
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterator

_logger = logging.getLogger(__name__)

# 默认 Arrow batch 行数（64K rows = 单批 ~8-32MB）
DEFAULT_ARROW_BATCH_SIZE = 65536


@dataclass
class ArrowBoundaryMetrics:
    """Arrow ↔ DuckDB 边界传输统计。"""

    batches_to_duckdb: int = 0
    batches_from_duckdb: int = 0
    total_rows_in: int = 0
    total_rows_out: int = 0
    total_bytes_in: int = 0
    total_bytes_out: int = 0
    zero_copy_transfers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batches_to_duckdb": self.batches_to_duckdb,
            "batches_from_duckdb": self.batches_from_duckdb,
            "total_rows_in": self.total_rows_in,
            "total_rows_out": self.total_rows_out,
            "total_bytes_in": self.total_bytes_in,
            "total_bytes_out": self.total_bytes_out,
            "zero_copy_transfers": self.zero_copy_transfers,
        }


class DuckDBArrowBoundaryOptimizer:
    """DuckDB ↔ Arrow zero-copy 传输控制器。

    用法（输入）：
        >>> optimizer = DuckDBArrowBoundaryOptimizer()
        >>> conn = duckdb.connect()
        >>> arrow_table = pa.table({"a": [1, 2, 3]})
        >>> rel = optimizer.register_arrow_table(conn, "my_table", arrow_table)
        >>> result = conn.execute("SELECT * FROM my_table").arrow()

    用法（输出）：
        >>> result_arrow = optimizer.query_to_arrow(conn, "SELECT * FROM data")
    """

    def __init__(
        self,
        *,
        batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
        streaming_threshold_rows: int = 1_000_000,
    ) -> None:
        """
        Args:
            batch_size: Arrow RecordBatch 行数（自适应调整）
            streaming_threshold_rows: 超过此行数启用 streaming scan
        """
        self.batch_size = batch_size
        self.streaming_threshold_rows = streaming_threshold_rows
        self._metrics = ArrowBoundaryMetrics()
        self._lock = threading.RLock()

    def register_arrow_table(
        self,
        conn: Any,
        table_name: str,
        arrow_table: Any,
        *,
        streaming: bool | None = None,
    ) -> Any:
        """将 Arrow Table 注册到 DuckDB（zero-copy，DuckDB 直接读 Arrow 内存）。

        Args:
            conn: DuckDB connection
            table_name: 表名（DuckDB 内可见）
            arrow_table: pyarrow.Table 或 RecordBatch
            streaming: True 强制 streaming；None 自动判断

        Returns:
            DuckDB Relation（可继续 SQL 操作）
        """
        try:
            import pyarrow as pa
        except ImportError:
            _logger.warning("pyarrow not installed; arrow boundary disabled")
            # fallback: 转 Pandas 再注册（有拷贝开销）
            try:
                df = arrow_table.to_pandas()
                return conn.register(table_name, df)
            except Exception as exc:
                _logger.error("fallback to pandas failed: %s", exc)
                raise

        if not isinstance(arrow_table, (pa.Table, pa.RecordBatch)):
            raise TypeError(
                f"expected pyarrow.Table or RecordBatch, got {type(arrow_table)}"
            )

        # 统一转 Table（RecordBatch → Table）
        if isinstance(arrow_table, pa.RecordBatch):
            arrow_table = pa.Table.from_batches([arrow_table])

        n_rows = arrow_table.num_rows
        with self._lock:
            self._metrics.batches_to_duckdb += 1
            self._metrics.total_rows_in += n_rows
            self._metrics.zero_copy_transfers += 1

        # 自动判断 streaming（大表分批传输，避免峰值内存）
        use_streaming = streaming
        if streaming is None:
            use_streaming = n_rows >= self.streaming_threshold_rows

        try:
            if use_streaming and n_rows > self.batch_size:
                # streaming scan：逐 batch 传输（DuckDB 内部 scan Arrow 不全物化）
                return self._register_streaming(conn, table_name, arrow_table)
            # 小表直接注册（DuckDB from_arrow() zero-copy）
            rel = conn.from_arrow(arrow_table)
            conn.register(table_name, rel)
            return rel
        except Exception as exc:
            _logger.error("register_arrow_table failed: %s", exc)
            raise

    def _register_streaming(
        self, conn: Any, table_name: str, arrow_table: Any
    ) -> Any:
        """Streaming 注册（逐 batch scan，不全部常驻内存）。

        DuckDB 支持 Arrow Dataset API（分区扫描）；本版本简化为单表分批。
        """
        try:
            # DuckDB 0.9+ 支持 Arrow Dataset
            import pyarrow.dataset as ds

            dataset = ds.dataset(arrow_table)
            rel = conn.from_arrow(dataset)
            conn.register(table_name, rel)
            return rel
        except Exception:
            # fallback：直接注册整表（DuckDB 内部 scan 也是惰性的）
            rel = conn.from_arrow(arrow_table)
            conn.register(table_name, rel)
            return rel

    def query_to_arrow(
        self,
        conn: Any,
        query: str,
        *,
        batch_size: int | None = None,
    ) -> Any:
        """执行 DuckDB query，结果直接返回 Arrow Table（zero-copy，避免 Pandas）。

        Args:
            conn: DuckDB connection
            query: SQL query
            batch_size: Arrow batch 行数（None 使用实例默认）

        Returns:
            pyarrow.Table
        """
        try:
            import pyarrow as pa
        except ImportError:
            _logger.warning("pyarrow not installed; fallback to pandas")
            return conn.execute(query).df()

        batch_size = batch_size or self.batch_size

        try:
            # DuckDB .arrow() 返回 Arrow Table（zero-copy）
            result = conn.execute(query).arrow()
            if not isinstance(result, pa.Table):
                result = pa.table(result)

            with self._lock:
                self._metrics.batches_from_duckdb += 1
                self._metrics.total_rows_out += result.num_rows
                self._metrics.zero_copy_transfers += 1

            return result
        except Exception as exc:
            _logger.error("query_to_arrow failed: %s; fallback to fetchdf", exc)
            try:
                df = conn.execute(query).fetchdf()
                return pa.Table.from_pandas(df)
            except Exception:
                raise

    def query_to_arrow_batches(
        self,
        conn: Any,
        query: str,
        *,
        batch_size: int | None = None,
    ) -> Iterator[Any]:
        """执行 DuckDB query，结果以 Arrow RecordBatch 流式返回（分批，不全物化）。

        Args:
            conn: DuckDB connection
            query: SQL query
            batch_size: 每批行数

        Yields:
            pyarrow.RecordBatch（streaming，一次只持有一批）
        """
        try:
            import pyarrow as pa
        except ImportError:
            _logger.warning("pyarrow not installed; batching disabled")
            yield conn.execute(query).df()
            return

        batch_size = batch_size or self.batch_size

        try:
            # DuckDB fetch_arrow_reader() 返回 streaming reader
            reader = conn.execute(query).fetch_arrow_reader(batch_size)
            for batch in reader:
                with self._lock:
                    self._metrics.batches_from_duckdb += 1
                    self._metrics.total_rows_out += batch.num_rows
                yield batch
        except Exception as exc:
            _logger.error("query_to_arrow_batches failed: %s", exc)
            # fallback：单批返回全部结果
            try:
                table = conn.execute(query).arrow()
                for batch in table.to_batches(max_chunksize=batch_size):
                    yield batch
            except Exception:
                raise

    def pandas_to_arrow_to_duckdb(
        self,
        conn: Any,
        table_name: str,
        df: Any,
    ) -> Any:
        """Pandas DataFrame → Arrow → DuckDB（减少一次拷贝）。

        Pandas → DuckDB 直接路径也是 zero-copy（DuckDB 0.9+），但显式经 Arrow
        可统一监控 arrow boundary metrics。
        """
        try:
            import pyarrow as pa
        except ImportError:
            return conn.register(table_name, df)

        try:
            arrow_table = pa.Table.from_pandas(df, preserve_index=False)
            return self.register_arrow_table(conn, table_name, arrow_table)
        except Exception as exc:
            _logger.warning("pandas_to_arrow failed: %s; direct register", exc)
            return conn.register(table_name, df)

    def arrow_to_pandas_zero_copy(self, arrow_table: Any) -> Any:
        """Arrow Table → Pandas DataFrame（尽可能 zero-copy）。

        pyarrow.Table.to_pandas(zero_copy_only=False, self_destruct=True) 可减少拷贝。
        """
        try:
            import pyarrow as pa
        except ImportError:
            return arrow_table

        if not isinstance(arrow_table, pa.Table):
            return arrow_table

        try:
            # zero_copy_only=False + self_destruct=True：尽量 zero-copy，
            # 不可避免的类型转换才拷贝（如 Arrow dictionary → Pandas categorical）
            return arrow_table.to_pandas(
                self_destruct=True,
                split_blocks=True,
                use_threads=True,
            )
        except Exception as exc:
            _logger.debug("zero_copy to_pandas failed: %s; standard path", exc)
            return arrow_table.to_pandas()

    def adjust_batch_size(self, pressure_stage: str) -> None:
        """根据内存压力自适应调整 batch_size。

        Args:
            pressure_stage: "NORMAL" | "PRESSURE_1" | ... | "CRITICAL"
        """
        with self._lock:
            if pressure_stage == "CRITICAL":
                self.batch_size = max(4096, self.batch_size // 4)
            elif pressure_stage in {"PRESSURE_3", "PRESSURE_4"}:
                self.batch_size = max(8192, self.batch_size // 2)
            elif pressure_stage == "NORMAL":
                # 恢复到默认
                self.batch_size = min(DEFAULT_ARROW_BATCH_SIZE, self.batch_size * 2)

    def metrics(self) -> ArrowBoundaryMetrics:
        """返回累计传输统计。"""
        with self._lock:
            return ArrowBoundaryMetrics(
                batches_to_duckdb=self._metrics.batches_to_duckdb,
                batches_from_duckdb=self._metrics.batches_from_duckdb,
                total_rows_in=self._metrics.total_rows_in,
                total_rows_out=self._metrics.total_rows_out,
                total_bytes_in=self._metrics.total_bytes_in,
                total_bytes_out=self._metrics.total_bytes_out,
                zero_copy_transfers=self._metrics.zero_copy_transfers,
            )

    def reset_metrics(self) -> None:
        """重置统计。"""
        with self._lock:
            self._metrics = ArrowBoundaryMetrics()


# 全局单例
_global_arrow_optimizer: DuckDBArrowBoundaryOptimizer | None = None
_global_lock = threading.Lock()


def get_global_arrow_optimizer() -> DuckDBArrowBoundaryOptimizer:
    """返回全局 DuckDB Arrow boundary optimizer（进程级单例）。"""
    global _global_arrow_optimizer
    if _global_arrow_optimizer is None:
        with _global_lock:
            if _global_arrow_optimizer is None:
                _global_arrow_optimizer = DuckDBArrowBoundaryOptimizer()
    return _global_arrow_optimizer
