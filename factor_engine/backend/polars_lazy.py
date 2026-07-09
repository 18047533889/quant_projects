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
