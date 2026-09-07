# -*- coding: utf-8
"""统一读结果句柄：一次读取，多种下游形态（arrow/pandas/polars/lazy/stream）。

R39 P0 #38-#42：
    - #40 显式状态机：OPEN → CONSUMING → MATERIALIZED → CLOSED/FAILED
      （不再用 ``_kind + _stream_started + _stream_completed`` 三个碎状态）。
      CLOSED 硬拒绝一切读；FAILED 不可重消费。
    - #39 ``close()`` 是真正 owner：关闭底层 stream/generator + source + 释放
      reservation + 跑 cleanup 回调。
    - #38 create-then-close（从不迭代）也释放 reservation / anchor（cleanup
      回调挂在句柄上，而不是只挂在生成器 finally 里）。
    - #41 所有 terminal 消费路径（to_arrow / stream / 迭代耗尽）共享**一个**
      ``_terminal_finalize()``：释放 reservation + 关闭 source + 跑回调。
    - #42 stats 在 terminal 消费期间累计（rows/bytes/elapsed），消费完写回
      ``handle.stats``。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterator

import pyarrow as pa

#: 状态机（R39 P0 #40）。
STATE_OPEN = "OPEN"
STATE_CONSUMING = "CONSUMING"
STATE_MATERIALIZED = "MATERIALIZED"
STATE_CLOSED = "CLOSED"
STATE_FAILED = "FAILED"


class ReadHandle:
    """包装一次读的结果，按需转换到调用方想要的形态。

    ``store.read()`` / ``store.read_uri()`` 返回本句柄。转换惰性：只有真正
    调用对应方法时才物化（``to_polars`` 首次调用才 import polars）。

    数据源可以是三种之一（互斥）：
        - table：已物化的 Arrow Table（初始状态 = MATERIALIZED）
        - stream：RecordBatch 迭代器（低内存峰值）
        - lazy：Polars LazyFrame（未物化）
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
        _cleanup_callbacks: list[Callable[[], None]] | None = None,
        _deadline: Any = None,
        read_identity: Any = None,
    ) -> None:
        self.snapshot = snapshot
        self.stats = stats
        self.lineage = lineage
        # R21 DataReadIdentity：绑定本次读取的完整上下文（dataset / revision /
        # availability / calendar / universe / source snapshot / grain / session /
        # decision clock）。每次 read 记录其 DataReadIdentity；无身份时为 None。
        self.read_identity = read_identity
        self._batch_size = batch_size
        # R26-P0-017：governed lazy ReadHandle 持有 governor reservation，物化/
        # stream 结束释放（release 幂等，防重复）。
        self._reservation = _reservation
        self._reservation_release_fn = _reservation_release_fn
        # R39 P0 #38：挂在句柄上的 cleanup 回调（reservation 释放 / anchor 清理）。
        # 只在 ``_terminal_finalize`` / ``close`` / ``__del__`` 跑**一次**——
        # create-then-close（从不迭代）也释放。
        self._cleanup_callbacks = list(_cleanup_callbacks) if _cleanup_callbacks else []
        # R39 P0 #37：执行期绝对 deadline（流式逐 batch 检查）。
        self._deadline = _deadline
        # #P0-21 governed lazy：production 不暴露 raw LazyFrame，collect 必须走
        # 预算/deadline 治理；``to_lazy()`` 在 governed 时拒绝裸 LazyFrame。
        self._budget = budget
        self._govern_lazy = bool(govern_lazy)
        self._normalize = normalize if callable(normalize) else None
        self._polars_df: Any = None
        self._pandas_df: Any = None
        # ---- R39 P0 #40 状态机 ----
        if table is not None:
            self._source: Any = table
            self._kind = "table"
            self._state = STATE_MATERIALIZED
        elif stream is not None:
            self._source = stream
            self._kind = "stream"
            self._state = STATE_OPEN
        elif lazy is not None:
            self._source = lazy
            self._kind = "lazy"
            self._state = STATE_OPEN
        else:
            self._source = None
            self._kind = "none"
            self._state = STATE_OPEN
        # Keep resource ownership separate from the canonical materialized
        # result.  Buffering/materialization replaces ``_source`` with a Table;
        # terminal cleanup must still close the original iterator/lazy source.
        self._owned_source = self._source
        self._source_closed = False
        # ---- R39 P0 #42 stats 累计器（初始值来自传入 stats）----
        self._stats_rows = getattr(stats, "rows", 0) or 0
        self._stats_bytes = getattr(stats, "bytes", 0) or 0
        self._stats_elapsed_ms = getattr(stats, "elapsed_ms", 0.0) or 0.0
        self._stats_paths = tuple(getattr(stats, "paths", ()) or ())
        self._consume_started: float | None = None

    # ---- 状态机 ----

    @property
    def state(self) -> str:
        return self._state

    def _ensure_open(self, action: str) -> None:
        """CLOSED 硬拒绝一切读；FAILED 不可重消费。"""
        if self._state == STATE_CLOSED:
            raise RuntimeError(
                f"ReadHandle 已 CLOSED，无法{action}（R39 P0 #40：one-shot 流已被消费"
                "或句柄已关闭——closed 硬拒绝所有读；请重新 read()）。"
            )
        if self._state == STATE_FAILED:
            raise RuntimeError(
                f"ReadHandle 处于 FAILED 终态，无法{action}（R39 P0 #40：failed 不可重消费）。"
            )
        if self._state == STATE_CONSUMING:
            raise RuntimeError(
                f"ReadHandle 的流正在 CONSUMING 中，无法{action}（one-shot 流已被消费）。"
            )

    def _start_consume(self) -> None:
        if self._consume_started is None:
            self._consume_started = time.perf_counter()

    def _acc_batch(self, batch: Any) -> None:
        """累计一个 batch 的 rows/bytes（#42）。

        兼容 Arrow RecordBatch（num_rows/nbytes）与 Polars DataFrame
        （height/estimated_size）。
        """
        rows = getattr(batch, "num_rows", None)
        if rows is None:
            rows = getattr(batch, "height", 0) or 0
        self._stats_rows += int(rows or 0)
        b = getattr(batch, "nbytes", None)
        if b is None:
            est = getattr(batch, "estimated_size", None)
            b = int(est()) if callable(est) else 0
        self._stats_bytes += int(b or 0)

    def _collect_stream_batches(self) -> list[Any]:
        """Collect the owned stream with the same per-batch governance as stream()."""
        batches: list[Any] = []
        for batch in self._source:
            self._deadline_check()
            self._acc_batch(batch)
            batches.append(batch)
        return batches

    def _finalize_stats(self) -> None:
        """terminal 消费后把累计值写回 ``self.stats``（#42）。"""
        if self._consume_started is not None:
            self._stats_elapsed_ms = (
                self._stats_elapsed_ms
                + (time.perf_counter() - self._consume_started) * 1000.0
            )
        if self.stats is None:
            from data_access.read.read_contract import ReadStats

            self.stats = ReadStats(
                rows=self._stats_rows,
                bytes=self._stats_bytes,
                elapsed_ms=self._stats_elapsed_ms,
                paths=self._stats_paths,
            )
            return
        try:
            self.stats = type(self.stats)(
                rows=self._stats_rows,
                bytes=self._stats_bytes,
                elapsed_ms=self._stats_elapsed_ms,
                paths=self._stats_paths,
            )
        except TypeError:
            # 自定义 stats 类型不接受 paths → 退回构造只读字段。
            from data_access.read.read_contract import ReadStats

            self.stats = ReadStats(
                rows=self._stats_rows,
                bytes=self._stats_bytes,
                elapsed_ms=self._stats_elapsed_ms,
                paths=self._stats_paths,
            )

    def _run_cleanup_callbacks(self) -> None:
        cbs = self._cleanup_callbacks
        self._cleanup_callbacks = []
        for cb in cbs:
            try:
                cb()
            except Exception:
                pass

    def _close_source(self) -> None:
        """关闭底层 source（stream/generator/reader）。#39/#41。"""
        if self._source_closed:
            return
        self._source_closed = True
        src = self._owned_source
        self._owned_source = None
        if src is None:
            return
        close = getattr(src, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _terminal_finalize(self) -> None:
        """**唯一** terminal-finalize：释放 reservation + 关闭 source + 跑回调。"""
        self._close_source()
        self._run_cleanup_callbacks()
        self._release_reservation()
        if self._consume_started is not None:
            self._finalize_stats()

    def _collect_lazy_arrow(self) -> pa.Table:
        """#P0-1 governed lazy collect 的统一 Arrow 终点。

        ``collect_polars_with_budget`` 返回的是 ``pa.Table``（预算强制在 Arrow
        物化期间逐 chunk 执行，R39 P0 #36/#37）。所有受控终点（to_arrow /
        to_polars / stream）都必须从这一个 Arrow 源派生，避免把 Arrow Table 误当
        Polars DataFrame / 调用不存在的 ``.to_arrow()``。
        """
        from data_access.read.query_budget import collect_polars_with_budget

        # deadline_at 不显式传：collect_polars_with_budget 内部会从 current_deadline()
        # / budget 派生，保持兼容（旧 mock 只接受 (lf, query_budget)）。
        table = collect_polars_with_budget(
            self._source,
            query_budget=self._budget,
        )
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
        """R39 P0 #39/#40：真正的 owner 关闭——终止底层流 + 释放 reservation +
        cleanup 回调 + 进入 CLOSED。幂等。"""
        if self._state == STATE_CLOSED:
            return
        self._state = STATE_CLOSED
        self._terminal_finalize()

    def fail(self, exc: BaseException | None = None) -> None:
        """R39 P0 #40：进入 FAILED 终态（不可重消费），并做一次 terminal 清理。"""
        if self._state in (STATE_CLOSED, STATE_FAILED):
            return
        self._state = STATE_FAILED
        self._terminal_finalize()

    def __enter__(self) -> "ReadHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC 兜底
        try:
            self.close()
        except Exception:
            pass

    def _ensure_not_consumed(self, action: str) -> None:
        """R39 P0 #40：one-shot 流一旦开始消费，任何后续物化/迭代都 fail-closed。"""
        self._ensure_open(action)

    def _materialize_arrow(self) -> pa.Table:
        """#P0-C5 canonical materialization：第一次 terminal collect 后把 Arrow
        Table 缓存为该句柄的 ``_source``，后续 pandas/polars/stream 全部从这一份
        派生——同一 ReadHandle 绝不重复执行底层 LazyFrame / 流。
        """
        self._ensure_open("to_arrow()")
        if self._kind == "table":
            return self._source
        self._start_consume()
        if self._kind == "lazy":
            try:
                if self._govern_lazy:
                    table = self._collect_lazy_arrow()
                else:
                    table = self._source.collect().to_arrow()
                if self._normalize is not None:
                    table = self._normalize(table)
                self._source = table
                self._kind = "table"
                self._state = STATE_MATERIALIZED
                return table
            except BaseException:
                self._state = STATE_FAILED
                raise
            finally:
                self._terminal_finalize()
        if self._kind == "stream":
            # OPEN 流一次性物化（_ensure_open 已拒绝 CONSUMING/CLOSED/FAILED）。
            try:
                batches = self._collect_stream_batches()
                self._source = (
                    pa.Table.from_batches(batches) if batches else pa.table({})
                )
                if self._normalize is not None:
                    self._source = self._normalize(self._source)
                self._kind = "table"
                self._state = STATE_MATERIALIZED
                return self._source
            except BaseException:
                self._state = STATE_FAILED
                raise
            finally:
                self._terminal_finalize()
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

        R39 P0 #40/#41/#42：显式状态机 + 共享 terminal-finalize + stats 累计。
        """
        self._ensure_open("再次 stream()")
        bs = batch_size or self._batch_size
        if bs is None or bs <= 0:
            bs = 100_000
        if self._kind == "table":
            # 已物化表：可任意次复用（状态保持 MATERIALIZED）。
            for batch in self._source.to_batches(max_chunksize=bs):
                yield batch
            return
        # stream / lazy 形态：one-shot 消费。
        self._start_consume()
        if self._kind == "stream":
            if buffer:
                try:
                    batches = self._collect_stream_batches()
                    self._source = (
                        pa.Table.from_batches(batches) if batches else pa.table({})
                    )
                    if self._normalize is not None:
                        self._source = self._normalize(self._source)
                    self._kind = "table"
                    self._state = STATE_MATERIALIZED
                except BaseException:
                    self._state = STATE_FAILED
                    self._terminal_finalize()
                    raise
                self._terminal_finalize()
                for batch in self._source.to_batches(max_chunksize=bs):
                    yield batch
                return
            # one-shot 流：完整/部分消费都进终态（CLOSED），共享 terminal-finalize。
            self._state = STATE_CONSUMING
            try:
                for batch in self._source:
                    self._deadline_check()
                    self._acc_batch(batch)
                    yield batch
            finally:
                self._state = STATE_CLOSED
                self._terminal_finalize()
            return
        if self._kind == "lazy":
            if buffer or self._govern_lazy:
                # governed lazy 本就没有流式（_collect_lazy_arrow 全量物化）；
                # buffer=True 也主动固化 → 都走 canonical Arrow。
                table = self._materialize_arrow()
                for batch in table.to_batches(max_chunksize=bs):
                    yield batch
                return
            # 非 governed lazy 且未 buffer：分块 collect_batches（低内存），
            # one-shot——完成后迭代器耗尽，后续终点 fail-closed。
            self._state = STATE_CONSUMING
            try:
                for df_batch in self._source.collect_batches(chunk_size=bs):
                    self._deadline_check()
                    self._acc_batch(df_batch)
                    yield df_batch.to_arrow()
            finally:
                self._state = STATE_CLOSED
                self._terminal_finalize()
            return
        raise RuntimeError("ReadHandle 没有可读数据")

    def _deadline_check(self) -> None:
        """R39 P0 #37：流式消费逐 batch 检查绝对 deadline（超时立即中断）。"""
        if self._deadline is not None:
            self._deadline.check(context="ReadHandle 流式消费")

    def __repr__(self) -> str:
        # #28：repr 不触发 lazy collect——rows 对非物化形态返回 None
        return (
            f"ReadHandle(state={self._state}, kind={self._kind}, rows={self.rows}, "
            f"columns={self.columns}, snapshot={getattr(self.snapshot, 'snapshot_id', None)})"
        )
