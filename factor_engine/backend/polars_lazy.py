"""Polars LazyFrame 读路径：scan_polars → 单次 collect → MultiIndex Series。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    _materialized: dict[str, Any] = field(default_factory=dict)

    def missing_physical(self, physical: list[str]) -> list[str]:
        have = set(self.physical_columns)
        return [c for c in physical if c not in have]

    def materialize_columns(
        self,
        physical_columns: list[str],
        *,
        output_names: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """从 bundle 物化列（已 collect 的列走缓存）。"""
        from data_access.adapters import arrow_table_to_multiindex_columns

        names = list(dict.fromkeys(physical_columns))
        pending = [c for c in names if c not in self._materialized]
        if pending:
            select_cols = list(
                dict.fromkeys(
                    [self.time_column, self.instrument_column, *pending]
                )
            )
            table = (
                self.lf.select(select_cols).collect().to_arrow()
            )
            reverse = {src: tgt for src, tgt in (output_names or self.output_names).items()}
            fetched = arrow_table_to_multiindex_columns(
                table,
                timestamp_column=self.time_column,
                instrument_column=self.instrument_column,
                value_columns=pending,
                output_names=reverse or None,
                normalize_timestamp=self.normalize_timestamp,
                timestamp_unit=self.timestamp_unit,
            )
            effective_out = output_names or self.output_names
            if effective_out:
                for src in pending:
                    logical = effective_out.get(src, src)
                    self._materialized[logical] = fetched[logical]
            else:
                self._materialized.update(fetched)

        effective_out = output_names or self.output_names
        if effective_out:
            return {
                effective_out.get(src, src): self._materialized[effective_out.get(src, src)]
                for src in names
                if effective_out.get(src, src) in self._materialized
            }
        return {src: self._materialized[src] for src in names if src in self._materialized}


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
) -> LazyColumnBundle:
    """构建 LazyFrame bundle（不 collect）。"""
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
    lf = store.scan_polars(dataset, **read_kwargs)
    return LazyColumnBundle(
        lf=lf,
        time_column=time_column,
        instrument_column=instrument_column,
        physical_columns=tuple(physical_columns),
        output_names=dict(output_names or {}),
        normalize_timestamp=normalize_timestamp,
        timestamp_unit=timestamp_unit,
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
) -> dict[str, Any]:
    """``store.scan_polars`` 读列并转为逻辑列名 → MultiIndex Series。"""
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

    from data_access.adapters import arrow_table_to_multiindex_columns

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

    lf = store.scan_polars(dataset, **read_kwargs)
    table = lf.collect().to_arrow()
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
) -> Any:
    """``store.scan_polars`` → long LazyFrame（``ts / inst / <logical cols>``），无 pandas 往返。"""
    all_cols = list(dict.fromkeys([time_column, instrument_column, *physical_columns]))
    read_kwargs: dict[str, Any] = {
        "columns": all_cols,
        "time_range": time_range,
        "instrument_filter": instrument_filter,
    }
    if params:
        read_kwargs.update(params)
    lf = store.scan_polars(dataset, **read_kwargs)
    rename: dict[str, str] = {time_column: "ts", instrument_column: "inst"}
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
) -> Any:
    """仅 scan ts / inst 轴（universe 对齐，不读因子列）。"""
    read_kwargs: dict[str, Any] = {
        "columns": [time_column, instrument_column],
        "time_range": time_range,
        "instrument_filter": instrument_filter,
    }
    if params:
        read_kwargs.update(params)
    lf = store.scan_polars(dataset, **read_kwargs)
    rename = {time_column: "ts", instrument_column: "inst"}
    return lf.rename(rename).select(["ts", "inst"]).unique()


def quote_ch_ident(name: str) -> str:
    """ClickHouse 标识符转义（反引号包裹）。"""
    if not isinstance(name, str) or not name:
        raise ValueError("invalid ClickHouse identifier")
    if "`" in name or "\x00" in name:
        raise ValueError(f"unsafe ClickHouse identifier: {name!r}")
    return f"`{name}`"


def build_clickhouse_scan_sql(
    table: str,
    *,
    timestamp_column: str,
    instrument_column: str,
    physical_columns: list[str],
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: list[str] | None = None,
) -> str:
    """构造 ClickHouse long-table scan SQL。"""
    cols = [timestamp_column, instrument_column, *physical_columns]
    quoted = ", ".join(quote_ch_ident(c) for c in cols)
    sql = f"SELECT {quoted} FROM {quote_ch_ident(table)}"
    clauses: list[str] = []
    if time_range is not None:
        start, end = time_range
        if start is not None:
            clauses.append(f"{quote_ch_ident(timestamp_column)} >= '{start}'")
        if end is not None:
            clauses.append(f"{quote_ch_ident(timestamp_column)} <= '{end}'")
    if instrument_filter:
        insts = ", ".join(f"'{str(s).replace(chr(39), chr(39)+chr(39))}'" for s in instrument_filter)
        clauses.append(f"{quote_ch_ident(instrument_column)} IN ({insts})")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
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
) -> Any:
    """ClickHouse SQL → Polars LazyFrame（``ts / inst / logical cols``）。"""
    import polars as pl

    from data_access.clickhouse_panel import execute_query

    sql = build_clickhouse_scan_sql(
        table,
        timestamp_column=timestamp_column,
        instrument_column=instrument_column,
        physical_columns=physical_columns,
        time_range=time_range,
        instrument_filter=instrument_filter,
    )
    table_arrow = execute_query(config, sql)
    lf = pl.from_arrow(table_arrow)
    rename: dict[str, str] = {timestamp_column: "ts", instrument_column: "inst"}
    for phys in physical_columns:
        logical = output_names.get(phys, phys)
        if phys != logical:
            rename[phys] = logical
    if rename:
        lf = lf.rename(rename)
    return lf.lazy()
