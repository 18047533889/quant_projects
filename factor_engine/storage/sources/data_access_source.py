"""通过 ``data_access.get_store()`` 读已登记数据集（因子引擎统一读通道）。"""

from __future__ import annotations

import sys
from typing import Any

from logging_utils import get_logger
from workspace_paths import quant_projects_root

from .datasource import DataSource

logger = get_logger("storage.data_access_source")


def _ensure_data_access_importable() -> None:
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_store():
    _ensure_data_access_importable()
    from data_access import get_store

    return get_store()


class DataAccessSource(DataSource):
    """``datasets.yaml`` 登记的数据集 → ``(timestamp, instrument)`` MultiIndex Series。"""

    def __init__(
        self,
        *,
        dataset: str,
        fields: dict[str, str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        instrument_filter: list[str] | None = None,
        normalize_timestamp: bool | None = None,
        timestamp_unit: str | None = None,
        read_auto: bool | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.fields = dict(fields or {})
        self.start_date = start_date
        self.end_date = end_date
        self.instrument_filter = list(instrument_filter) if instrument_filter else None
        self.normalize_timestamp = normalize_timestamp
        self.timestamp_unit = timestamp_unit
        self.params = dict(params or {})
        if read_auto is not None:
            self.read_auto = bool(read_auto)
        else:
            self.read_auto = bool(self.params.pop("read_auto", False))
        self._lazy_scan = bool(self.params.pop("lazy_scan", False))
        self._column_cache: dict[str, Any] = {}
        self._panel_cache: dict[str, Any] = {}
        self._lazy_bundle: Any | None = None

    def column_cache_stats(self) -> dict[str, int]:
        return {
            "cached_columns": len(self._column_cache),
            "cached_panels": len(self._panel_cache),
        }

    def enable_lazy_scan(self, enabled: bool = True) -> None:
        """启用 ``scan_polars`` 读路径（配合 ``PolarsBackend.use_lazy``）。"""
        self._lazy_scan = bool(enabled)
        if enabled:
            self.read_auto = True

    @property
    def lazy_scan(self) -> bool:
        return bool(self._lazy_scan)

    def read_session(self) -> "DataSourceReadSession":
        from .read_session import DataSourceReadSession

        return DataSourceReadSession(self)

    def _time_range(self) -> tuple[Any, Any] | None:
        if self.start_date is None and self.end_date is None:
            return None
        return (self.start_date, self.end_date)

    def _resolve_columns(self, names: list[str]) -> tuple[list[str], dict[str, str]]:
        """逻辑列名 → (物理列名列表, output_names 映射)。"""
        physical: list[str] = []
        output_names: dict[str, str] = {}
        for name in names:
            src = self.fields.get(name, name)
            physical.append(src)
            if src != name:
                output_names[src] = name
        return physical, output_names

    def load_column(self, name: str):
        if name in self._column_cache:
            return self._column_cache[name]
        result = self.load_columns([name])
        return result[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        needed = [n for n in names if n not in self._column_cache]
        if not needed:
            return {n: self._column_cache[n] for n in names}

        physical, output_names = self._resolve_columns(needed)
        store = _get_store()
        logger.info(
            "data_access 读数据集 '%s' 列 %s (time_range=%s, read_auto=%s, lazy_scan=%s)",
            self.dataset,
            needed,
            self._time_range(),
            self.read_auto,
            self._lazy_scan,
        )
        if self.read_auto and self._lazy_scan:
            from backend.polars_lazy import scan_dataset_columns

            ds = store._registry.get(self.dataset)
            from data_access.store import adapter_options_for_dataset

            adapter_opts = adapter_options_for_dataset(ds)
            norm_ts = (
                self.normalize_timestamp
                if self.normalize_timestamp is not None
                else bool(adapter_opts.get("normalize_timestamp", False))
            )
            ts_unit = self.timestamp_unit or adapter_opts.get("timestamp_unit")
            fetched = scan_dataset_columns(
                store,
                self.dataset,
                physical_columns=list(physical),
                time_column=ds.time_column,
                instrument_column=ds.instrument_column,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                output_names=output_names or None,
                normalize_timestamp=norm_ts,
                timestamp_unit=ts_unit,
                params=dict(self.params),
                bundle=self._lazy_bundle,
            )
        elif self.read_auto:
            from data_access.adapters import arrow_table_to_multiindex_columns
            from data_access.store import adapter_options_for_dataset

            ds = store._registry.get(self.dataset)
            adapter_opts = adapter_options_for_dataset(ds)
            norm_ts = (
                self.normalize_timestamp
                if self.normalize_timestamp is not None
                else bool(adapter_opts.get("normalize_timestamp", False))
            )
            ts_unit = self.timestamp_unit or adapter_opts.get("timestamp_unit")
            all_cols = list(dict.fromkeys([ds.time_column, ds.instrument_column, *physical]))
            table = store.read_auto(
                self.dataset,
                columns=all_cols,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                prefer_polars=True,
                **self.params,
            )
            reverse_names = {src: tgt for src, tgt in output_names.items()}
            fetched = arrow_table_to_multiindex_columns(
                table,
                timestamp_column=ds.time_column,
                instrument_column=ds.instrument_column,
                value_columns=list(physical),
                output_names=reverse_names or None,
                normalize_timestamp=norm_ts,
                timestamp_unit=ts_unit,
            )
            if output_names:
                fetched = {
                    output_names.get(src, src): fetched[output_names.get(src, src)]
                    for src in physical
                }
        else:
            fetched = store.load_columns(
                self.dataset,
                columns=physical,
                output_names=output_names or None,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                normalize_timestamp=self.normalize_timestamp,
                timestamp_unit=self.timestamp_unit,
                **self.params,
            )
        for n in needed:
            self._column_cache[n] = fetched[n]
        return {n: self._column_cache[n] for n in names}

    def prefetch_columns(self, names: list[str]) -> None:
        """批量预加载列（lazy_scan 时构建共享 LazyColumnBundle，单次 collect）。"""
        if not names:
            return
        if self.read_auto and self._lazy_scan:
            self._prefetch_lazy_bundle(names)
            return
        self.load_columns(names)

    def _prefetch_lazy_bundle(self, names: list[str]) -> None:
        needed = [n for n in names if n not in self._column_cache]
        if not needed:
            return
        physical, output_names = self._resolve_columns(needed)
        store = _get_store()
        ds = store._registry.get(self.dataset)
        from data_access.store import adapter_options_for_dataset
        from backend.polars_lazy import build_lazy_column_bundle

        adapter_opts = adapter_options_for_dataset(ds)
        norm_ts = (
            self.normalize_timestamp
            if self.normalize_timestamp is not None
            else bool(adapter_opts.get("normalize_timestamp", False))
        )
        ts_unit = self.timestamp_unit or adapter_opts.get("timestamp_unit")

        if self._lazy_bundle is None:
            self._lazy_bundle = build_lazy_column_bundle(
                store,
                self.dataset,
                physical_columns=list(physical),
                time_column=ds.time_column,
                instrument_column=ds.instrument_column,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                output_names=output_names or None,
                normalize_timestamp=norm_ts,
                timestamp_unit=ts_unit,
                params=dict(self.params),
            )
        else:
            missing = self._lazy_bundle.missing_physical(list(physical))
            if missing:
                merged_physical = list(
                    dict.fromkeys([*self._lazy_bundle.physical_columns, *physical])
                )
                merged_out = dict(self._lazy_bundle.output_names)
                merged_out.update(output_names)
                self._lazy_bundle = build_lazy_column_bundle(
                    store,
                    self.dataset,
                    physical_columns=merged_physical,
                    time_column=ds.time_column,
                    instrument_column=ds.instrument_column,
                    time_range=self._time_range(),
                    instrument_filter=self.instrument_filter,
                    output_names=merged_out or None,
                    normalize_timestamp=norm_ts,
                    timestamp_unit=ts_unit,
                    params=dict(self.params),
                )

        fetched = self._lazy_bundle.materialize_columns(
            list(physical),
            output_names=output_names or None,
        )
        for n in needed:
            self._column_cache[n] = fetched[n]

    def prefetch_panels(self, names: list[str]) -> None:
        """一次 SQL 批量读列并 unstack 为宽表 panel（panel-native 热路径）。"""
        needed = [n for n in names if n not in self._panel_cache]
        if not needed:
            return
        batch = self.load_columns(needed)
        for name, series in batch.items():
            if name not in self._panel_cache:
                level = series.index.names[-1] if series.index.names[-1] is not None else "instrument"
                self._panel_cache[name] = series.unstack(level=level)

    def dataset_axis_columns(self) -> tuple[str, str]:
        """registry 中该数据集的时间列 / 标的列（SQL 下推用）。"""
        store = _get_store()
        return store.dataset_axis_columns(self.dataset)

    def load_column_panel(self, name: str):
        """加载单列为宽表 panel（index=时间, columns=标的），供 panel-native 热路径。"""
        if name in self._panel_cache:
            return self._panel_cache[name]
        series = self.load_column(name)
        level = series.index.names[-1] if series.index.names[-1] is not None else "instrument"
        panel = series.unstack(level=level)
        self._panel_cache[name] = panel
        return panel

    def scan_polars_long(self, columns: list[str]):
        """Parquet/DuckDB scan → ``ts / inst / <cols>`` LazyFrame（PolarsLongBackend 用）。"""
        physical, output_names = self._resolve_columns(columns)
        store = _get_store()
        ds = store._registry.get(self.dataset)
        from backend.polars_lazy import build_scan_polars_long

        return build_scan_polars_long(
            store,
            self.dataset,
            logical_columns=columns,
            physical_columns=physical,
            output_names=output_names,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            params=dict(self.params),
        )

    def scan_index_long(self):
        """仅 scan ``ts / inst`` 轴（universe 对齐，不读因子列）。"""
        store = _get_store()
        ds = store._registry.get(self.dataset)
        from backend.polars_lazy import build_scan_index_long

        return build_scan_index_long(
            store,
            self.dataset,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            params=dict(self.params),
        )
