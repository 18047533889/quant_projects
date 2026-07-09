# -*- coding: utf-8
"""长表 DataSource：只走 MultiIndex Series，不做 unstack/stack。"""
from __future__ import annotations

from typing import Any

from .datasource import DataSource


class LongTableDataSource(DataSource):
    """包装现有数据源：列读取走 MultiIndex Series；panel 按需 lazy unstack 并缓存。"""

    def __init__(self, inner: DataSource) -> None:
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

    def load_column(self, name: str):
        return self._inner.load_column(name)

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        loader = getattr(self._inner, "load_columns", None)
        if callable(loader):
            return loader(names)
        return {n: self.load_column(n) for n in names}

    def prefetch_columns(self, names: list[str]) -> None:
        fn = getattr(self._inner, "prefetch_columns", None)
        if callable(fn):
            fn(names)

    def prefetch_panels(self, names: list[str]) -> None:
        """长表模式：批量预加载列但不 unstack。"""
        self.prefetch_columns(names)

    def scan_polars_long(self, columns: list[str]):
        """优先 inner.scan_polars_long；否则从 Series 拼 long LazyFrame。"""
        inner_scan = getattr(self._inner, "scan_polars_long", None)
        if callable(inner_scan):
            return inner_scan(columns)
        import polars as pl

        from storage.factor_format import series_to_long_table

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
        return pl.from_pandas(renamed).lazy()

    def load_column_panel(self, name: str):
        """按需 unstack 一次并缓存（panel-native / 宽表算子热路径）。"""
        if name in self._panel_cache:
            return self._panel_cache[name]
        series = self.load_column(name)
        from backend.cleaned_bridge import series_to_panel
        from backend.context import ExecutionContext

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
        fn = getattr(self._inner, "dataset_axis_columns", None)
        if callable(fn):
            return fn()
        ts = getattr(self._inner, "timestamp_column", "timestamp")
        inst = getattr(self._inner, "instrument_column", "instrument")
        return str(ts), str(inst)
