# -*- coding: utf-8
"""统一读结果句柄：一次读取，多种下游形态（arrow/pandas/polars/lazy/stream）。"""
from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa


class ReadHandle:
    """包装一次读的结果，按需转换到调用方想要的形态。

    ``store.read()`` / ``store.read_uri()`` 返回本句柄。转换惰性：只有真正
    调用对应方法时才物化（``to_polars`` 首次调用才 import polars）。

    数据源可以是三种之一（互斥）：
        - table：已物化的 Arrow Table
        - stream：RecordBatch 迭代器（低内存峰值）
        - lazy：Polars LazyFrame（未物化）

    示例：
        >>> handle = store.read("factor_lake", factor_id="mom_3d")
        >>> tbl = handle.to_arrow()
        >>> df  = handle.to_pandas()
        >>> plf = handle.to_lazy()          # 需要 polars
        >>> for batch in handle.stream():   # RecordBatch 流
        ...     ...
    """

    def __init__(
        self,
        table: pa.Table | None = None,
        *,
        snapshot: Any = None,
        stats: Any = None,
        lineage: Any = None,
        stream: Iterator[pa.RecordBatch] | None = None,
        lazy: Any = None,
        batch_size: int = 100_000,
        budget: Any = None,
        govern_lazy: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.stats = stats
        self.lineage = lineage
        self._batch_size = batch_size
        # #P0-21 governed lazy：production 不暴露 raw LazyFrame，collect 必须走
        # 预算/deadline 治理；``to_lazy()`` 在 governed 时拒绝裸 LazyFrame。
        self._budget = budget
        self._govern_lazy = bool(govern_lazy)
        self._polars_df: Any = None
        self._pandas_df: Any = None
        if table is not None:
            self._source: Any = table
            self._kind = "table"
        elif stream is not None:
            self._source = stream
            self._kind = "stream"
        elif lazy is not None:
            self._source = lazy
            self._kind = "lazy"
        else:
            self._source = None
            self._kind = "none"

    def _collect_lazy(self) -> Any:
        """#P0-21 governed lazy collect：带 QueryBudget 的受控终点。"""
        from data_access.read.query_budget import collect_polars_with_budget

        return collect_polars_with_budget(self._source, query_budget=self._budget)

    # ---- 基本信息 ----

    @property
    def rows(self) -> int | None:
        if self._kind == "table":
            return self._source.num_rows
        # lazy / stream 形态不触发 collect（#28：rows 只是元信息，物化由
        # to_arrow/to_pandas/to_polars 等显式方法负责）。
        return None

    @property
    def columns(self) -> list[str] | None:
        if self._kind == "table":
            return list(self._source.column_names)
        if self._kind == "lazy":
            return list(self._source.columns)
        return None

    @property
    def nbytes(self) -> int | None:
        return self._source.nbytes if self._kind == "table" else None

    # ---- 转换 ----

    def to_arrow(self) -> pa.Table:
        if self._kind == "table":
            return self._source
        if self._kind == "stream":
            batches = list(self._source)
            self._source = pa.Table.from_batches(batches) if batches else pa.table({})
            self._kind = "table"
            return self._source
        if self._kind == "lazy":
            if self._govern_lazy:
                table = self._collect_lazy().to_arrow()
            else:
                table = self._source.collect().to_arrow()
            self._source = table
            self._kind = "table"
            return table
        raise RuntimeError("ReadHandle 没有可读数据")

    def to_pandas(self):
        import pandas as pd  # noqa: F401

        if self._pandas_df is None:
            # 不能 self_destruct：同一 handle 可能继续 to_polars/to_lazy/stream，
            # 释放共享 Arrow buffer 会导致 use-after-free（段错误）。
            self._pandas_df = self.to_arrow().to_pandas(split_blocks=True)
        return self._pandas_df

    def to_polars(self):
        import polars as pl

        if self._polars_df is None:
            if self._kind == "lazy":
                if self._govern_lazy:
                    self._polars_df = self._collect_lazy()
                else:
                    self._polars_df = self._source.collect()
            else:
                self._polars_df = pl.from_arrow(self.to_arrow())
        return self._polars_df

    def to_lazy(self):
        import polars as pl

        if self._govern_lazy:
            # #P0-21 production 不暴露 raw LazyFrame：collect 会绕过 budget/
            # deadline 治理。要裸 LazyFrame 请显式 unsafe_scan_polars()。
            raise RuntimeError(
                "governed lazy handle 不暴露 raw LazyFrame（production fail-closed）；"
                "请用 to_arrow()/to_polars()/stream() 受控终点。"
            )
        if self._kind == "lazy":
            return self._source
        return self.to_polars().lazy()

    def stream(self, batch_size: int | None = None) -> Iterator[pa.RecordBatch]:
        """RecordBatch 流。有底层流时直接透传；否则把已物化表切 batch。"""
        if self._kind == "stream":
            yield from self._source
            return
        if self._kind == "lazy":
            if self._govern_lazy:
                for batch in self._collect_lazy().to_batches(
                    max_chunksize=batch_size or self._batch_size
                ):
                    yield batch
                return
            for batch in self._source.collect_stream():
                yield batch
            return
        table = self.to_arrow()
        bs = batch_size or self._batch_size
        if bs is None or bs <= 0:
            bs = 100_000
        for batch in table.to_batches(max_chunksize=bs):
            yield batch

    def __repr__(self) -> str:
        # #28：repr 不触发 lazy collect——rows 对非物化形态返回 None
        return (
            f"ReadHandle(kind={self._kind}, rows={self.rows}, "
            f"columns={self.columns}, snapshot={getattr(self.snapshot, 'snapshot_id', None)})"
        )
