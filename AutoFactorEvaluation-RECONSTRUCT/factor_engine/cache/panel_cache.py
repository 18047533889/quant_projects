"""L1 Panel 缓存：Series → unstack 宽表复用。"""

from __future__ import annotations

from typing import Any


def series_panel_cache_key(series: Any) -> tuple[Any, ...]:
    """稳定 cache key：Series 身份 + index + 列名 + 长度。"""
    name = getattr(series, "name", None)
    return (id(series), id(series.index), str(name) if name is not None else "", len(series))


class PanelCache:
    """``ExecutionContext.panel_cache`` 的薄封装。"""

    def __init__(self, store: dict[Any, Any] | None = None) -> None:
        self._store: dict[Any, Any] = store if store is not None else {}

    @property
    def store(self) -> dict[Any, Any]:
        return self._store

    def get(self, key: Any) -> Any | None:
        return self._store.get(key)

    def get_for_series(self, series: Any) -> Any | None:
        """用 Series 身份键查 panel 缓存。"""
        return self.get(series_panel_cache_key(series))

    def set(self, key: Any, panel: Any) -> None:
        self._store[key] = panel

    def set_for_series(self, series: Any, panel: Any) -> None:
        """缓存某 Series unstack 后的宽表 panel。"""
        self.set(series_panel_cache_key(series), panel)

    def __len__(self) -> int:
        return len(self._store)
