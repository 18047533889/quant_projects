"""通过 ``data_access.get_store()`` 读已登记数据集（因子引擎统一读通道）。"""

from __future__ import annotations

import sys
from typing import Any

from logging_utils import get_logger
from workspace_paths import quant_projects_root

from .datasource import DataSource

logger = get_logger("storage.data_access_source")


def _ensure_data_access_importable() -> None:
    """将 quant_projects 根目录加入 sys.path。
    
    参数:
        无
    
    返回:
        无
    """
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_store():
    """获取 data_access 全局 store 实例。
    
    参数:
        无
    
    返回:
        无
    """
    _ensure_data_access_importable()
    from data_access import get_store

    return get_store()


class DataAccessSource(DataSource):
    """通过 data_access 读取已登记数据集列。
    
    参数:
        dataset: data_access 数据集名称（可选）
        fields: 逻辑列到物理列映射（可选）
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        instrument_filter: 见函数签名（可选）
        normalize_timestamp: 见函数签名（可选）
        timestamp_unit: 见函数签名（可选）
        read_auto: 见函数签名（可选）
        params: 见函数签名（可选）
    """

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
        """初始化实例。
        
        参数:
            dataset: data_access 数据集名称（可选）
            fields: 逻辑列到物理列映射（可选）
            start_date: 起始日期（可选）
            end_date: 结束日期（可选）
            instrument_filter: 见函数签名（可选）
            normalize_timestamp: 见函数签名（可选）
            timestamp_unit: 见函数签名（可选）
            read_auto: 见函数签名（可选）
            params: 见函数签名（可选）
        
        返回:
            无
        """
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
        self._data_snapshot_id: str | None = None

    @property
    def data_snapshot_id(self) -> str | None:
        """最近一次读路径返回的 ``DataSnapshot.snapshot_id``（若有）。"""
        return self._data_snapshot_id

    def _record_read_snapshot(self, snapshot_id: str | None) -> None:
        if snapshot_id:
            self._data_snapshot_id = snapshot_id

    def column_cache_stats(self) -> dict[str, int]:
        """column_cache_stats。
        
        参数:
            无
        
        返回:
            dict[str, int]
        """
        return {
            "cached_columns": len(self._column_cache),
            "cached_panels": len(self._panel_cache),
        }

    def enable_lazy_scan(self, enabled: bool = True) -> None:
        """启用 ``scan_polars`` 读路径（配合 ``PolarsBackend.use_lazy``）。
        
        参数:
            enabled: 见函数签名（可选）
        
        返回:
            无
        """
        self._lazy_scan = bool(enabled)
        if enabled:
            self.read_auto = True

    @property
    def lazy_scan(self) -> bool:
        """lazy_scan。
        
        参数:
            无
        
        返回:
            bool
        """
        return bool(self._lazy_scan)

    def read_session(self) -> "DataSourceReadSession":
        """read_session。
        
        参数:
            无
        
        返回:
            'DataSourceReadSession'
        """
        from .read_session import DataSourceReadSession

        return DataSourceReadSession(self)

    def _time_range(self) -> tuple[Any, Any] | None:
        """_time_range。
        
        参数:
            无
        
        返回:
            tuple[Any, Any] | None
        """
        if self.start_date is None and self.end_date is None:
            return None
        return (self.start_date, self.end_date)

    def _resolve_columns(self, names: list[str]) -> tuple[list[str], dict[str, str]]:
        """逻辑列名 → (物理列名列表, output_names 映射)。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            tuple[list[str], dict[str, str]]
        """
        physical: list[str] = []
        output_names: dict[str, str] = {}
        for name in names:
            src = self.fields.get(name, name)
            physical.append(src)
            if src != name:
                output_names[src] = name
        return physical, output_names

    def load_column(self, name: str):
        """load_column。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
        if name in self._column_cache:
            return self._column_cache[name]
        result = self.load_columns([name])
        return result[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """load_columns。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            dict[str, Any]
        """
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

            ds = store.get_dataset(self.dataset)
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
            if self._lazy_bundle and self._lazy_bundle.snapshot_id:
                self._record_read_snapshot(self._lazy_bundle.snapshot_id)
        elif self.read_auto:
            from data_access.adapters import arrow_table_to_multiindex_columns
            from data_access.store import adapter_options_for_dataset

            ds = store.get_dataset(self.dataset)
            adapter_opts = adapter_options_for_dataset(ds)
            norm_ts = (
                self.normalize_timestamp
                if self.normalize_timestamp is not None
                else bool(adapter_opts.get("normalize_timestamp", False))
            )
            ts_unit = self.timestamp_unit or adapter_opts.get("timestamp_unit")
            all_cols = list(dict.fromkeys([ds.time_column, ds.instrument_column, *physical]))
            read_result = store.read_result(
                self.dataset,
                columns=all_cols,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                **self.params,
            )
            self._record_read_snapshot(read_result.snapshot.snapshot_id)
            table = read_result.table
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
            from data_access.adapters import arrow_table_to_multiindex_columns
            from data_access.store import adapter_options_for_dataset

            ds = store.get_dataset(self.dataset)
            adapter_opts = adapter_options_for_dataset(ds)
            norm_ts = (
                self.normalize_timestamp
                if self.normalize_timestamp is not None
                else bool(adapter_opts.get("normalize_timestamp", False))
            )
            ts_unit = self.timestamp_unit or adapter_opts.get("timestamp_unit")
            all_cols = list(dict.fromkeys([ds.time_column, ds.instrument_column, *physical]))
            read_result = store.read_result(
                self.dataset,
                columns=all_cols,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                **self.params,
            )
            self._record_read_snapshot(read_result.snapshot.snapshot_id)
            reverse_names = {src: tgt for src, tgt in output_names.items()}
            fetched = arrow_table_to_multiindex_columns(
                read_result.table,
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
        for n in needed:
            self._column_cache[n] = fetched[n]
        return {n: self._column_cache[n] for n in names}

    def prefetch_columns(self, names: list[str]) -> None:
        """批量预加载列（lazy_scan 时构建共享 LazyColumnBundle，单次 collect）。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        if not names:
            return
        if self.read_auto and self._lazy_scan:
            self._prefetch_lazy_bundle(names)
            return
        self.load_columns(names)

    def _prefetch_lazy_bundle(self, names: list[str]) -> None:
        """_prefetch_lazy_bundle。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        needed = [n for n in names if n not in self._column_cache]
        if not needed:
            return
        physical, output_names = self._resolve_columns(needed)
        store = _get_store()
        ds = store.get_dataset(self.dataset)
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
            if self._lazy_bundle.snapshot_id:
                self._record_read_snapshot(self._lazy_bundle.snapshot_id)
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
        """一次 SQL 批量读列并 unstack 为宽表 panel（panel-native 热路径）。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        needed = [n for n in names if n not in self._panel_cache]
        if not needed:
            return
        batch = self.load_columns(needed)
        for name, series in batch.items():
            if name not in self._panel_cache:
                level = series.index.names[-1] if series.index.names[-1] is not None else "instrument"
                self._panel_cache[name] = series.unstack(level=level)

    def dataset_axis_columns(self) -> tuple[str, str]:
        """registry 中该数据集的时间列 / 标的列（SQL 下推用）。
        
        参数:
            无
        
        返回:
            tuple[str, str]
        """
        store = _get_store()
        return store.dataset_axis_columns(self.dataset)

    def load_column_panel(self, name: str):
        """加载单列为宽表 panel（index=时间, columns=标的），供 panel-native 热路径。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
        if name in self._panel_cache:
            return self._panel_cache[name]
        series = self.load_column(name)
        level = series.index.names[-1] if series.index.names[-1] is not None else "instrument"
        panel = series.unstack(level=level)
        self._panel_cache[name] = panel
        return panel

    def scan_polars_long(self, columns: list[str]):
        """Parquet/DuckDB scan → ``ts / inst / <cols>`` LazyFrame（PolarsLongBackend 用）。
        
        参数:
            columns: 列名列表
        
        返回:
            无
        """
        physical, output_names = self._resolve_columns(columns)
        store = _get_store()
        ds = store.get_dataset(self.dataset)
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
        """仅 scan ``ts / inst`` 轴（universe 对齐，不读因子列）。
        
        参数:
            无
        
        返回:
            无
        """
        store = _get_store()
        ds = store.get_dataset(self.dataset)
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
