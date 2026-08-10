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
        normalize: Any = None,
        _reservation: Any = None,
        _reservation_release_fn: Any = None,
    ) -> None:
        self.snapshot = snapshot
        self.stats = stats
        self.lineage = lineage
        self._batch_size = batch_size
        # R26-P0-017：governed lazy ReadHandle 持有 governor reservation，物化/
        # stream 结束释放（release 幂等，防重复）。
        self._reservation = _reservation
        self._reservation_release_fn = _reservation_release_fn
        # #P0-21 governed lazy：production 不暴露 raw LazyFrame，collect 必须走
        # 预算/deadline 治理；``to_lazy()`` 在 governed 时拒绝裸 LazyFrame。
        self._budget = budget
        self._govern_lazy = bool(govern_lazy)
        # #收官轮 P0：lazy 结果形态的输出层单位归一化钩子。``normalize_units=True``
        # 在 eager table 路径由调用方直接做；lazy/polars result 走统一物化终点
        # （``_materialize_arrow``），必须在这里补上——否则 ``engine='auto'`` 路由
        # 到 polars+lazy 时 ``normalize_units=True`` 静默失效，A股 Return 保持 raw
        # BP（该值与 eager 路径不一致）。
        self._normalize = normalize if callable(normalize) else None
        self._polars_df: Any = None
        self._pandas_df: Any = None
        # #P1-9 / #P0-C4 stream 消费状态：one-shot 流一旦开始消费，任何第二终点
        # 都 fail-closed（除非 stream(buffer=True) 显式把结果固化进句柄）。这覆盖
        # 部分消费（break 后 to_arrow 得到截断 batch）**和**完整消费（迭代器耗尽后
        # to_arrow/list() 静默变空表 / 第二次 stream() 静默无数据）两种情形。
        self._stream_started = False
        self._stream_completed = False
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

    def _collect_lazy_arrow(self) -> pa.Table:
        """#P0-1 governed lazy collect 的统一 Arrow 终点。

        ``collect_polars_with_budget`` 返回的是 ``pa.Table``（预算强制在 Arrow
        物化之后）。所有受控终点（to_arrow / to_polars / stream）都必须从这一个
        Arrow 源派生，避免把 Arrow Table 误当 Polars DataFrame / 调用不存在的
        ``.to_arrow()``。
        """
        from data_access.read.query_budget import collect_polars_with_budget

        try:
            table = collect_polars_with_budget(
                self._source, query_budget=self._budget
            )
        finally:
            # R26-P0-017：物化终点释放 governor reservation（幂等）。
            self._release_reservation()
        if not isinstance(table, pa.Table):
            # 防御：budget 层语义回归（返回非 Arrow）直接 fail，不静默透传。
            raise TypeError(
                f"governed lazy collect 必须返回 pyarrow.Table，收到 "
                f"{type(table).__name__}"
            )
        return table

    def _release_reservation(self) -> None:
        if self._reservation is not None and self._reservation_release_fn is not None:
            fn = self._reservation_release_fn
            self._reservation_release_fn = None
            fn(self._reservation)
            self._reservation = None

    # R29-P0 #206：governed handle 的显式资源生命周期——lazy reservation 没有天然
    # release point（``with store.read(...) as h`` 后不用 h 就返回），必须支持
    # close()/with/GC 兜底。release 幂等（``_reservation_release_fn=None``）。
    def close(self) -> None:
        """显式释放 governor reservation（不消费结果）。幂等。"""
        self._release_reservation()

    def __enter__(self) -> "ReadHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC 兜底
        try:
            self._release_reservation()
        except Exception:
            pass

    def _ensure_not_consumed(self, action: str) -> None:
        """#P0-C4 one-shot 流一旦开始消费，任何后续物化/迭代都 fail-closed。

        已用 ``stream(buffer=True)`` 或 ``to_arrow()`` 物化进 ``table`` 形态的
        句柄不受限（可任意复用）。其余形态被消费过（无论完整/部分）都拒绝——
        完整消费后 ``list(iterator)`` 只会得到空表，部分消费后得到截断结果，
        都不允许静默发生。
        """
        if not self._stream_started or self._kind == "table":
            return
        state = "完整消费" if self._stream_completed else "部分消费（中途 break）"
        raise RuntimeError(
            f"ReadHandle 的 one-shot 流已被{state}，无法{action}——迭代器已耗尽，"
            "继续会得到空表/截断/交叉数据。请重新 read()；若想消费后再复用结果，"
            "请首次调用 stream(buffer=True) 显式缓存。"
        )

    def _materialize_arrow(self) -> pa.Table:
        """#P0-C5 canonical materialization：第一次 terminal collect 后把 Arrow
        Table 缓存为该句柄的 ``_source``，后续 pandas/polars/stream 全部从这一份
        派生——同一 ReadHandle 绝不重复执行底层 LazyFrame / 流（中间数据变化时
        两个终点也会因此保持一致）。
        """
        if self._kind == "table":
            return self._source
        if self._kind == "lazy":
            if self._govern_lazy:
                table = self._collect_lazy_arrow()
            else:
                table = self._source.collect().to_arrow()
            if self._normalize is not None:
                # 收官轮 P0：lazy 结果形态补输出层单位归一化（normalize_units=True
                # 与 eager table 路径一致）。
                table = self._normalize(table)
            self._source = table
            self._kind = "table"
            return table
        if self._kind == "stream":
            # 调用方已过 _ensure_not_consumed → 这里必是未消费的流，一次性物化。
            batches = list(self._source)
            self._source = (
                pa.Table.from_batches(batches) if batches else pa.table({})
            )
            if self._normalize is not None:
                # 收官轮 P0：stream 结果形态同样补输出层单位归一化。
                self._source = self._normalize(self._source)
            self._kind = "table"
            self._stream_completed = True
            return self._source
        raise RuntimeError("ReadHandle 没有可读数据")

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
        # #P0-C4/#P0-C5 统一走 canonical materialization；one-shot 流已消费过则
        # fail-closed（不能静默返回空表/截断 batch）。
        self._ensure_not_consumed("to_arrow()")
        return self._materialize_arrow()

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
            # #P0-C5 统一从 canonical Arrow 派生：第一次 terminal collect 后
            # 缓存 Arrow Table，后续 to_arrow/to_pandas/stream 复用同一份，绝不
            # 重复执行 LazyFrame（旧实现 to_polars 后 to_arrow 会再次 collect）。
            self._polars_df = pl.from_arrow(self.to_arrow())
        return self._polars_df

    def to_lazy(self):
        import polars as pl

        # #P0-C4 已消费过的流不允许再取 lazy（重新 collect 会重复扫描底层数据）。
        self._ensure_not_consumed("to_lazy()")
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

    def stream(
        self,
        batch_size: int | None = None,
        *,
        buffer: bool = False,
    ) -> Iterator[pa.RecordBatch]:
        """RecordBatch 流。有底层流时直接透传；否则把已物化表切 batch。

        #P1-9/#P0-C4 one-shot 语义：流形态（stream / 未物化 lazy）一旦开始消费，
        任何第二终点（再次 stream() / to_arrow() / to_polars() / to_lazy()）都
        fail-closed——迭代器耗尽后继续 list() 只会得到空表，部分消费后得到截断
        结果。**唯一例外**：首次 ``stream(buffer=True)`` 把结果固化进句柄（转成
        table 形态），之后可任意复用。

        #P0-C5 canonical：``to_arrow()`` 先跑的句柄，``stream()`` 直接从此缓存
        slice，不重复执行底层 LazyFrame；governed lazy 没有真正流式（collect 全量
        物化），stream() 也从统一 Arrow 源派生。
        """
        self._ensure_not_consumed("再次 stream()")
        bs = batch_size or self._batch_size
        if bs is None or bs <= 0:
            bs = 100_000
        if self._kind == "stream":
            self._stream_started = True
            if buffer:
                batches = list(self._source)
                self._source = (
                    pa.Table.from_batches(batches) if batches else pa.table({})
                )
                self._kind = "table"
                self._stream_completed = True
                for batch in self._source.to_batches(max_chunksize=bs):
                    yield batch
                return
            for batch in self._source:
                yield batch
            self._stream_completed = True
            return
        if self._kind == "lazy":
            self._stream_started = True
            if buffer or self._govern_lazy:
                # governed lazy 本就没有流式（_collect_lazy_arrow 全量物化）；
                # buffer=True 也主动固化 → 都走 canonical Arrow。
                table = self._materialize_arrow()
                self._stream_completed = True
                for batch in table.to_batches(max_chunksize=bs):
                    yield batch
                return
            # 非 governed lazy 且未 buffer：真正的 collect_stream（低内存），
            # one-shot——完成后迭代器耗尽，后续终点 fail-closed。
            try:
                for batch in self._source.collect_stream():
                    yield batch
            finally:
                # R26-P0-017：流式断开/耗尽 → release reservation（幂等）。
                self._release_reservation()
            self._stream_completed = True
            return
        # table 形态：从 canonical Arrow 切片（可任意次复用）。
        for batch in self._source.to_batches(max_chunksize=bs):
            yield batch

    def __repr__(self) -> str:
        # #28：repr 不触发 lazy collect——rows 对非物化形态返回 None
        return (
            f"ReadHandle(kind={self._kind}, rows={self.rows}, "
            f"columns={self.columns}, snapshot={getattr(self.snapshot, 'snapshot_id', None)})"
        )
