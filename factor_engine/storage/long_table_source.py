# -*- coding: utf-8
"""长表 DataSource：只走 MultiIndex Series，不做 unstack/stack。"""
from __future__ import annotations

from typing import Any

from .datasource import DataSource


class LongTableDataSource(DataSource):
    """包装现有数据源，禁用 panel 宽表路径（SQL 下推 / 长表算子热路径）。"""

    def __init__(self, inner: DataSource) -> None:
        self._inner = inner
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

    def load_column_panel(self, name: str):
        raise NotImplementedError(
            "LongTableDataSource 不支持宽表 panel；请使用 load_column() 或 SQL 下推。"
        )

    def dataset_axis_columns(self) -> tuple[str, str]:
        fn = getattr(self._inner, "dataset_axis_columns", None)
        if callable(fn):
            return fn()
        ts = getattr(self._inner, "timestamp_column", "timestamp")
        inst = getattr(self._inner, "instrument_column", "instrument")
        return str(ts), str(inst)
