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
        self._column_cache: dict[str, Any] = {}
        self._panel_cache: dict[str, Any] = {}

    def column_cache_stats(self) -> dict[str, int]:
        return {
            "cached_columns": len(self._column_cache),
            "cached_panels": len(self._panel_cache),
        }

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
            "data_access 读数据集 '%s' 列 %s (time_range=%s)",
            self.dataset,
            needed,
            self._time_range(),
        )
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
        """批量预加载列（走 store.load_columns，含 COS mirror + 日期窗口下推）。"""
        if not names:
            return
        self.load_columns(names)

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
