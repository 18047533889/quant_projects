"""Polars LazyFrame 读路径：store.scan → 单次 collect → MultiIndex Series。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any
import weakref


def _collect_arrow_table(lf: Any, *, select_cols: list[str]) -> Any:
    """ScanHandle / LazyFrame 统一 collect → Arrow Table。"""
    from data_access.read.query_budget import collect_polars_with_budget
    from data_access.read.scan_handle import ScanHandle

    selected = lf.select(select_cols)
    if isinstance(selected, ScanHandle):
        return selected.collect().table
    if isinstance(lf, ScanHandle):
        return lf.select(select_cols).collect().table
    return collect_polars_with_budget(selected)


def _store_scan(store: Any, dataset: str, read_kwargs: dict[str, Any]) -> Any:
    """优先 ``store.scan()``（budget + snapshot）；回退 ``scan_polars``。"""
    scan_fn = getattr(store, "scan", None)
    if callable(scan_fn):
        return scan_fn(dataset, **read_kwargs)
    return store.scan_polars(dataset, **read_kwargs)


def _ensure_polars_lazyframe(lf: Any) -> Any:
    """Polars 原生 long 路径需要裸 ``LazyFrame``，不能链式 ``.collect()`` 得到 ``ReadResult``。

    Phase 5 R13：通过 data_access 正式 ``ScanHandle.native_lazyframe()`` 获取
    composition-only 句柄，不再访问私有 ``._lf``。duck-type ``native_lazyframe()``
    以便 ScanHandle-like 对象/测试 fake 同样走 composition-only 语义。
    """
    from data_access.read.scan_handle import ScanHandle

    if isinstance(lf, ScanHandle):
        return lf.native_lazyframe()
    native = getattr(lf, "native_lazyframe", None)
    if callable(native):
        return native()
    return lf


def _store_scan_polars_native(store: Any, dataset: str, read_kwargs: dict[str, Any]) -> Any:
    """Polars long 原生路径与 governed-lazy **统一**：走 ``store.scan()``（ScanHandle，
    budget + snapshot 绑定），再取 composition-only LazyFrame。

    #收官轮 P0（Integration，incomplete-fix bypass）：旧实现优先
    ``store.scan_polars()`` 拿裸 LazyFrame，绕过 ScanHandle 的 collect-time
    snapshot revalidation / terminal QueryBudget / audit——两个 helper（
    ``scan_dataset_columns`` vs ``build_scan_polars_long``）各走一条路。现在
    native-long 与 lazy bundle 同源：compile 链拿到真实 LazyFrame（join 参数需要），
    而 scan 时刻的 snapshot 由 ScanHandle 绑定，collect 前由
    ``revalidate_for_long_collect`` + budget 受控。strict 模式下
    ``native_lazyframe()`` 抛错 = 要求走受控 collect 的 fail-closed 语义。
    """
    return _ensure_polars_lazyframe(_store_scan(store, dataset, read_kwargs))


def _snapshot_id_from_scan(scan_obj: Any) -> str | None:
    snap = getattr(scan_obj, "snapshot", None)
    if snap is not None:
        return getattr(snap, "snapshot_id", None)
    return None


def enforce_source_ordering(
    lf_or_df: Any,
    *,
    instrument_col: str,
    time_col: str,
    session_col: str | None = None,
    frequency: str | None = None,
) -> Any:
    """Enforce the source long-table ordering contract (P0-10).

    Ordering guarantee: every source scan returns a long table sorted ascending
    by ``(instrument, time)`` — the daily-panel contract — so engine window /
    forward iterators that step instrument-by-instrument see a stable order.
    Minute/intraday data additionally sorts by ``(instrument, time, session)``
    when ``session_col`` is provided.  ``frequency=None`` defaults to the daily
    ``(instrument, time)`` contract.

    Accepts a polars ``LazyFrame``/``DataFrame`` (returned unchanged in type) or
    a pandas ``DataFrame``.  This is a pure helper (no ``DataAccessSource``
    dependency) so it can be unit-tested in isolation.
    """
    import polars as pl

    sort_cols = [instrument_col, time_col]
    if frequency == "minute" and session_col:
        sort_cols = [instrument_col, time_col, session_col]
    if isinstance(lf_or_df, pl.LazyFrame) or isinstance(lf_or_df, pl.DataFrame):
        return lf_or_df.sort(sort_cols)
    from .pandas_compat import pd

    if isinstance(lf_or_df, pd.DataFrame):
        return lf_or_df.sort_values(sort_cols)
    raise TypeError(
        "enforce_source_ordering expects a polars LazyFrame/DataFrame or pandas "
        f"DataFrame, got {type(lf_or_df)!r}"
    )


@dataclass
class LazyColumnBundle:
    """共享 LazyFrame 扫描图；``materialize_columns`` 只做一次 collect。"""

    lf: Any
    time_column: str
    instrument_column: str
    physical_columns: tuple[str, ...]
    output_names: dict[str, str]
    normalize_timestamp: bool
    timestamp_unit: str | None
    snapshot_id: str | None = None
    _materialized: dict[str, Any] = field(default_factory=dict)
    _materialized_budget: int | None = None
    _cache_broker: Any | None = field(default=None, repr=False, compare=False)
    _cache_leases: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _live_cache_leases: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _cache_context: tuple | None = field(default=None, init=False, repr=False)
    _materialize_lock: Any = field(default_factory=RLock, init=False, repr=False, compare=False)

    @property
    def materialized_budget(self) -> int:
        """Shared-broker residency cap; unknown/zero authority disables caching."""
        explicit = None if self._materialized_budget is None else max(
            0, int(self._materialized_budget)
        )
        if explicit == 0:
            return 0
        try:
            if self._cache_broker is None:
                from factor_engine.storage.sources.data_access_source import _data_cache_authority
                self._cache_broker = _data_cache_authority()
            shared = max(0, int(self._cache_broker.current_read_budget()))
            return shared if explicit is None else min(explicit, shared)
        except Exception:
            return 0

    def _cache_bytes(self) -> int:
        from factor_engine.runtime.resource_governor import estimate_object_bytes

        return sum(estimate_object_bytes(v) for v in self._materialized.values())

    def _evict_to(self, target: int) -> None:
        """有界 LRU：超出预算时逐出最早物化的列。"""
        from factor_engine.runtime.resource_governor import estimate_object_bytes

        if not isinstance(self._materialized, OrderedDict):
            self._materialized = OrderedDict(self._materialized)
        while self._materialized and self._cache_bytes() > target:
            name, value = self._materialized.popitem(last=False)
            self._cache_leases.pop(name, None)
            # Removing residency does not release its lease while a caller still
            # owns the physical buffers; owner finalizers perform that release.
            _ = estimate_object_bytes(value)

    def release_cached(self, physical_columns: list[str]) -> None:
        """Transfer residency away from the bundle without touching request refs."""
        with self._materialize_lock:
            for name in physical_columns:
                self._materialized.pop(name, None)
                self._cache_leases.pop(name, None)

    def close(self) -> None:
        """Drop residency; physical-owner finalizers release outstanding leases."""
        with self._materialize_lock:
            self.release_cached(list(self._materialized))

    def missing_physical(self, physical: list[str]) -> list[str]:
        """返回 bundle 中缺失的物理列名列表。"""
        have = set(self.physical_columns)
        return [c for c in physical if c not in have]

    def materialize_columns(
        self, physical_columns: list[str], *,
        output_names: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Cache physical columns; own request references survive LRU eviction."""
        from data_access.read.adapters import arrow_table_to_multiindex_columns

        names = list(dict.fromkeys(physical_columns))
        missing = self.missing_physical(names)
        if missing:
            raise KeyError(f"columns outside bundle: {missing}")
        effective_out = dict(self.output_names if output_names is None else output_names)
        logical_names = [effective_out.get(src, src) for src in names]
        if len(set(logical_names)) != len(logical_names):
            raise ValueError("conflicting physical-to-logical column mapping")
        context = (id(self.lf), self.snapshot_id, self.time_column,
                   self.instrument_column, self.normalize_timestamp, self.timestamp_unit)
        with self._materialize_lock:
            if self._cache_context != context:
                self.release_cached(list(self._materialized))
                self._cache_context = context
            if not isinstance(self._materialized, OrderedDict):
                self._materialized = OrderedDict(self._materialized)
            # Strong request-owned references MUST precede inserting new entries.
            request = {}
            for src in names:
                if src in self._materialized:
                    request[src] = self._materialized[src]
                    self._materialized.move_to_end(src)
            pending = [src for src in names if src not in request]
            if pending:
                select_cols = list(dict.fromkeys(
                    [self.time_column, self.instrument_column, *pending]))
                table = _collect_arrow_table(self.lf, select_cols=select_cols)
                fetched = arrow_table_to_multiindex_columns(
                    table, timestamp_column=self.time_column,
                    instrument_column=self.instrument_column,
                    value_columns=pending, output_names=None,
                    normalize_timestamp=self.normalize_timestamp,
                    timestamp_unit=self.timestamp_unit,
                )
                if set(fetched) != set(pending):
                    raise RuntimeError("materialized physical columns do not match request")
                request.update(fetched)
                budget = self.materialized_budget
                if budget > 0 and self._cache_broker is not None:
                    from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
                    from factor_engine.runtime.resource_governor import estimate_object_bytes
                    for src in pending:
                        value = fetched[src]
                        nbytes = estimate_object_bytes(value)
                        if nbytes <= 0 or nbytes > budget:
                            continue
                        from factor_engine.storage.sources.data_access_source import DataAccessSource
                        owners = DataAccessSource._physical_cache_owners(value)
                        if not owners:
                            continue
                        lease = self._cache_broker.acquire_memory(
                            MemoryLeaseKind.SOURCE_READ, nbytes,
                            lease_id=(f"lazy-column-bundle:{id(self)}:{src}:"
                                      f"{time.monotonic_ns()}"),
                        )
                        if lease is None:
                            continue
                        self._materialized[src] = value
                        token = id(lease)
                        state = {"remaining": len(owners), "lease": lease,
                                 "lock": RLock(), "finalizers": ()}
                        self._live_cache_leases[token] = state
                        bundle_ref = weakref.ref(self)
                        def release_owner(tok=token, owner_state=state, ref=bundle_ref):
                            with owner_state["lock"]:
                                owner_state["remaining"] -= 1
                                if owner_state["remaining"] != 0:
                                    return
                                owner_state["lease"].release()
                            bundle = ref()
                            if bundle is not None:
                                with bundle._materialize_lock:
                                    bundle._live_cache_leases.pop(tok, None)
                        registered = []
                        try:
                            for owner in owners:
                                registered.append(weakref.finalize(owner, release_owner))
                            state["finalizers"] = tuple(registered)
                        except Exception:
                            for finalizer in registered:
                                finalizer.detach()
                            self._materialized.pop(src, None)
                            self._live_cache_leases.pop(token, None)
                            lease.release()
                            continue
                        self._cache_leases[src] = token
            # Budget may shrink between waves even when every column is a hit.
            # Request references already own all results before cache eviction.
            self._evict_to(self.materialized_budget)
            if set(request) != set(names):
                raise RuntimeError("incomplete materialization")
            # Rename only at the public boundary, never in the cache namespace.
            return {logical: request[src].rename(logical, copy=False)
                    for src, logical in zip(names, logical_names)}


def build_lazy_column_bundle(
    store: Any,
    dataset: str,
    *,
    physical_columns: list[str],
    time_column: str,
    instrument_column: str,
    time_range: tuple[Any, Any] | None,
    instrument_filter: list[str] | None,
    output_names: dict[str, str] | None,
    normalize_timestamp: bool,
    timestamp_unit: str | None,
    params: dict[str, Any] | None,
    mode: str = "auto",
    filters: Any = None,
    materialized_budget: int | None = None,
) -> LazyColumnBundle:
    """构建 LazyFrame bundle（不 collect）。

    ``mode``/``filters``（#收官轮 P0）：FE 的 ``read_mode`` 与 ``semantic_filters``
    贯穿到 DataAccess ``scan``——语义模式（panel/event/pit）与行级过滤不再在
    lazy 路径丢失。
    """
    all_cols = list(
        dict.fromkeys([time_column, instrument_column, *physical_columns])
    )
    read_kwargs: dict[str, Any] = {
        "columns": all_cols,
        "time_range": time_range,
        "instrument_filter": instrument_filter,
    }
    if params:
        read_kwargs.update(params)
    read_kwargs["mode"] = mode
    if filters is not None:
        read_kwargs["filters"] = filters
    scan_obj = _store_scan(store, dataset, read_kwargs)
    return LazyColumnBundle(
        lf=scan_obj,
        time_column=time_column,
        instrument_column=instrument_column,
        physical_columns=tuple(physical_columns),
        output_names=dict(output_names or {}),
        normalize_timestamp=normalize_timestamp,
        timestamp_unit=timestamp_unit,
        snapshot_id=_snapshot_id_from_scan(scan_obj),
        _materialized_budget=materialized_budget,
    )


def scan_dataset_columns(
    store: Any,
    dataset: str,
    *,
    physical_columns: list[str],
    time_column: str,
    instrument_column: str,
    time_range: tuple[Any, Any] | None,
    instrument_filter: list[str] | None,
    output_names: dict[str, str] | None,
    normalize_timestamp: bool,
    timestamp_unit: str | None,
    params: dict[str, Any] | None,
    bundle: LazyColumnBundle | None = None,
    mode: str = "auto",
    filters: Any = None,
) -> dict[str, Any]:
    """``store.scan_polars`` 读列并转为逻辑列名 → MultiIndex Series。

    ``mode``/``filters``（#收官轮 P0）：同 ``build_lazy_column_bundle``——
    FE read_mode / semantic_filters 贯穿到 DataAccess scan。
    """
    if bundle is not None:
        missing = bundle.missing_physical(list(physical_columns))
        if missing:
            raise ValueError(
                f"LazyColumnBundle 缺少物理列 {missing!r}，请 prefetch 时包含全部依赖列"
            )
        return bundle.materialize_columns(
            list(physical_columns),
            output_names=output_names,
        )

    from data_access.read.adapters import arrow_table_to_multiindex_columns

    all_cols = list(
        dict.fromkeys([time_column, instrument_column, *physical_columns])
    )
    read_kwargs: dict[str, Any] = {
        "columns": all_cols,
        "time_range": time_range,
        "instrument_filter": instrument_filter,
    }
    if params:
        read_kwargs.update(params)
    read_kwargs["mode"] = mode
    if filters is not None:
        read_kwargs["filters"] = filters

    scan_obj = _store_scan(store, dataset, read_kwargs)
    table = _collect_arrow_table(scan_obj, select_cols=all_cols)
    reverse_names = {src: tgt for src, tgt in (output_names or {}).items()}
    fetched = arrow_table_to_multiindex_columns(
        table,
        timestamp_column=time_column,
        instrument_column=instrument_column,
        value_columns=list(physical_columns),
        output_names=reverse_names or None,
        normalize_timestamp=normalize_timestamp,
        timestamp_unit=timestamp_unit,
    )
    if output_names:
        return {
            output_names.get(src, src): fetched[output_names.get(src, src)]
            for src in physical_columns
        }
    return fetched


def build_scan_polars_long(
    store: Any,
    dataset: str,
    *,
    logical_columns: list[str],
    physical_columns: list[str],
    output_names: dict[str, str],
    time_column: str,
    instrument_column: str,
    time_range: tuple[Any, Any] | None,
    instrument_filter: list[str] | None,
    params: dict[str, Any] | None = None,
    frequency: str | None = None,
    session_column: str | None = None,
    mode: str = "auto",
    filters: Any = None,
) -> Any:
    """``store.scan_polars`` → long LazyFrame（``ts / inst / <logical cols>``），无 pandas 往返。

    Ordering guarantee (P0-10): the returned LazyFrame is sorted ascending by
    ``(instrument, time)`` — ``(instrument, time, session)`` for minute data when
    ``session_column`` is provided.  ``frequency`` is detected by the caller from
    the source's grain metadata; ``None`` defaults to the daily contract.

    ``mode``/``filters``（#收官轮 P0）：FE read_mode / semantic_filters 贯穿到
    DataAccess scan_polars。
    """
    all_cols = list(dict.fromkeys([time_column, instrument_column, *physical_columns]))
    if frequency == "minute" and session_column and session_column not in all_cols:
        all_cols.append(session_column)
    read_kwargs: dict[str, Any] = {
        "columns": all_cols,
        "time_range": time_range,
        "instrument_filter": instrument_filter,
    }
    if params:
        read_kwargs.update(params)
    read_kwargs["mode"] = mode
    if filters is not None:
        read_kwargs["filters"] = filters
    lf = _store_scan_polars_native(store, dataset, read_kwargs)
    lf = enforce_source_ordering(
        lf,
        instrument_col=instrument_column,
        time_col=time_column,
        session_col=session_column,
        frequency=frequency,
    )
    rename: dict[str, str] = {time_column: "ts", instrument_column: "inst"}
    if frequency == "minute" and session_column:
        rename[session_column] = "session"
    for phys in physical_columns:
        logical = output_names.get(phys, phys)
        if phys != logical:
            rename[phys] = logical
    if rename:
        lf = lf.rename(rename)
    return lf


def build_scan_index_long(
    store: Any,
    dataset: str,
    *,
    time_column: str,
    instrument_column: str,
    time_range: tuple[Any, Any] | None,
    instrument_filter: list[str] | None,
    params: dict[str, Any] | None = None,
    mode: str = "auto",
    filters: Any = None,
) -> Any:
    """仅 scan ts / inst 轴（universe 对齐，不读因子列）。"""
    read_kwargs: dict[str, Any] = {
        "columns": [time_column, instrument_column],
        "time_range": time_range,
        "instrument_filter": instrument_filter,
    }
    if params:
        read_kwargs.update(params)
    read_kwargs["mode"] = mode
    if filters is not None:
        read_kwargs["filters"] = filters
    lf = _store_scan_polars_native(store, dataset, read_kwargs)
    rename = {time_column: "ts", instrument_column: "inst"}
    return lf.rename(rename).select(["ts", "inst"]).unique()


def quote_ch_ident(name: str) -> str:
    """ClickHouse 标识符转义（反引号包裹）。"""
    if not isinstance(name, str) or not name:
        raise ValueError("invalid ClickHouse identifier")
    if "`" in name or "\x00" in name:
        raise ValueError(f"unsafe ClickHouse identifier: {name!r}")
    return f"`{name}`"


def quote_ch_literal(val: Any) -> str:
    """ClickHouse SQL 字面量（字符串 / 日期等）。"""
    if val is None:
        return "NULL"
    s = str(val).replace("'", "''")
    return f"'{s}'"


def build_clickhouse_scan_sql(
    table: str,
    *,
    timestamp_column: str,
    instrument_column: str,
    physical_columns: list[str],
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: list[str] | None = None,
    frequency: str | None = None,
    session_column: str | None = None,
) -> str:
    """构造 ClickHouse long-table scan SQL（含强制 ORDER BY 排序契约，P0-10）。"""
    cols = [timestamp_column, instrument_column]
    if frequency == "minute" and session_column:
        cols.append(session_column)
    cols += list(physical_columns)
    quoted = ", ".join(quote_ch_ident(c) for c in cols)
    sql = f"SELECT {quoted} FROM {quote_ch_ident(table)}"
    clauses: list[str] = []
    if time_range is not None:
        start, end = time_range
        if start is not None:
            clauses.append(f"{quote_ch_ident(timestamp_column)} >= {quote_ch_literal(start)}")
        if end is not None:
            clauses.append(f"{quote_ch_ident(timestamp_column)} <= {quote_ch_literal(end)}")
    if instrument_filter:
        insts = ", ".join(quote_ch_literal(str(s)) for s in instrument_filter)
        clauses.append(f"{quote_ch_ident(instrument_column)} IN ({insts})")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    order_cols = [instrument_column, timestamp_column]
    if frequency == "minute" and session_column:
        order_cols = [instrument_column, timestamp_column, session_column]
    sql += " ORDER BY " + ", ".join(quote_ch_ident(c) for c in order_cols)
    return sql


def scan_clickhouse_long(
    config: Any,
    *,
    table: str,
    timestamp_column: str,
    instrument_column: str,
    physical_columns: list[str],
    output_names: dict[str, str],
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: list[str] | None = None,
    frequency: str | None = None,
    session_column: str | None = None,
) -> Any:
    """ClickHouse SQL → Polars LazyFrame（``ts / inst / logical cols``）。

    Ordering guarantee (P0-10): the SQL carries ``ORDER BY <instrument>, <time>``
    and the post-query result is re-sorted via ``enforce_source_ordering`` so the
    contract holds for every path (``(instrument, time, session)`` for minute).
    """
    import polars as pl

    from data_access.clickhouse.panel import execute_query

    sql = build_clickhouse_scan_sql(
        table,
        timestamp_column=timestamp_column,
        instrument_column=instrument_column,
        physical_columns=physical_columns,
        time_range=time_range,
        instrument_filter=instrument_filter,
        frequency=frequency,
        session_column=session_column,
    )
    table_arrow = execute_query(config, sql)
    lf = pl.from_arrow(table_arrow)
    rename: dict[str, str] = {timestamp_column: "ts", instrument_column: "inst"}
    if frequency == "minute" and session_column:
        rename[session_column] = "session"
    for phys in physical_columns:
        logical = output_names.get(phys, phys)
        if phys != logical:
            rename[phys] = logical
    if rename:
        lf = lf.rename(rename)
    lf = lf.lazy()
    return enforce_source_ordering(
        lf,
        instrument_col="inst",
        time_col="ts",
        session_col="session" if (frequency == "minute" and session_column) else None,
        frequency=frequency,
    )
