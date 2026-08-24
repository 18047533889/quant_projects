# -*- coding: utf-8
"""长表 DataSource：只走 MultiIndex Series，不做 unstack/stack。"""
from __future__ import annotations

from typing import Any

from .datasource import DataSource


class LongTableDataSource(DataSource):
    """长表模式包装器，列读走 Series、panel 按需 unstack。
    
    参数:
        inner: 内层数据源
    """

    def __init__(self, inner: DataSource) -> None:
        """初始化实例。
        
        参数:
            inner: 内层数据源
        
        返回:
            无
        """
        self._inner = inner
        self._panel_cache: dict[str, Any] = {}
        for attr in (
            "dataset",
            "table",
            "start_date",
            "end_date",
            "instrument_filter",
            "fields",
            "timestamp_column",
            "instrument_column",
        ):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    def execution_spec(self) -> dict[str, Any] | None:
        """长表包装：透传内层 canonical 执行规格（#收官轮 P0）。"""
        fn = getattr(self._inner, "execution_spec", None)
        if not callable(fn):
            return None
        spec = fn()
        if not isinstance(spec, dict):
            return None
        return dict(spec)

    def load_column(self, name: str):
        """load_column。

        参数:
            name: 逻辑列名

        返回:
            无
        """
        return self._inner.load_column(name)

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """load_columns。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            dict[str, Any]
        """
        loader = getattr(self._inner, "load_columns", None)
        if callable(loader):
            return loader(names)
        return {n: self.load_column(n) for n in names}

    def prefetch_columns(self, names: list[str]) -> None:
        """prefetch_columns。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        fn = getattr(self._inner, "prefetch_columns", None)
        if callable(fn):
            fn(names)

    def prefetch_panels(self, names: list[str]) -> None:
        """长表模式：批量预加载列但不 unstack。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        self.prefetch_columns(names)

    def scan_polars_long(self, columns: list[str]):
        """优先 inner.scan_polars_long；否则从 Series 拼 long LazyFrame。

        Ordering guarantee (P0-10): the assembled long LazyFrame is sorted
        ascending by ``(instrument, time)`` via ``enforce_source_ordering``.

        参数:
            columns: 列名列表

        返回:
            无
        """
        inner_scan = getattr(self._inner, "scan_polars_long", None)
        if callable(inner_scan):
            return inner_scan(columns)
        import polars as pl

        from factor_engine.storage.factor_format import series_to_long_table

        ts, inst = self.dataset_axis_columns()
        merged: Any = None
        for name in sorted(columns):
            series = self.load_column(name)
            part = series_to_long_table(
                series,
                timestamp_col=ts,
                asset_col=inst,
                value_col=name,
            )
            if merged is None:
                merged = part
            else:
                merged = merged.merge(part, on=[ts, inst], how="outer")
            merged[name] = merged[name].astype("float64")
        if merged is None:
            raise ValueError("scan_polars_long: no columns")
        renamed = merged.rename(columns={ts: "ts", inst: "inst"})
        from factor_engine.backend.long_frame import long_table_to_polars_lazy
        from factor_engine.backend.polars_lazy import enforce_source_ordering

        lf = long_table_to_polars_lazy(renamed, float_cols=columns)
        frequency = getattr(self._inner, "frequency", None)
        return enforce_source_ordering(
            lf,
            instrument_col="inst",
            time_col="ts",
            frequency=frequency,
        )

    def scan_index_long(self):
        """代理 inner.scan_index_long（universe 对齐用，不落列缓存）。"""
        fn = getattr(self._inner, "scan_index_long", None)
        if not callable(fn):
            raise NotImplementedError
        return fn()

    def load_column_panel(self, name: str):
        """按需 unstack 一次并缓存（panel-native / 宽表算子热路径）。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
        if name in self._panel_cache:
            return self._panel_cache[name]
        series = self.load_column(name)
        from factor_engine.backend.cleaned_bridge import series_to_panel
        from factor_engine.backend.context import ExecutionContext

        ctx = ExecutionContext(
            data_source=self,
            panel_cache={},
            timestamp_col=getattr(self, "timestamp_column", "timestamp"),
            instrument_col=getattr(self, "instrument_column", "instrument"),
            prefer_long_table=True,
        )
        panel = series_to_panel(series, ctx)
        self._panel_cache[name] = panel
        return panel

    def dataset_axis_columns(self) -> tuple[str, str]:
        """dataset_axis_columns。
        
        参数:
            无
        
        返回:
            tuple[str, str]
        """
        fn = getattr(self._inner, "dataset_axis_columns", None)
        if callable(fn):
            return fn()
        ts = getattr(self._inner, "timestamp_column", "timestamp")
        inst = getattr(self._inner, "instrument_column", "instrument")
        return str(ts), str(inst)
