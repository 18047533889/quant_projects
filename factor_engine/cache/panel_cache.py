"""L1 Panel 缓存：Series → unstack 宽表复用（字节感知 LRU，受 MemoryGovernor 治理）。

Phase 5 R6：宽表 panel 往往是最大内存占用（5000 列 × 数千行），同样必须受
``panel_budget`` 约束，逐出后由 DataAccessSource 侧按需重算。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any


def _governor():
    from runtime.resource_governor import global_memory_governor

    return global_memory_governor()


def _estimate(value: Any) -> int:
    from runtime.resource_governor import estimate_object_bytes

    return estimate_object_bytes(value)


def series_panel_cache_key(series: Any) -> tuple[Any, ...]:
    """稳定 cache key：Series 身份 + index + 列名 + 长度。"""
    name = getattr(series, "name", None)
    return (id(series), id(series.index), str(name) if name is not None else "", len(series))


class PanelCache:
    """``ExecutionContext.panel_cache`` 的字节感知封装。"""

    def __init__(
        self,
        store: dict[Any, Any] | None = None,
        *,
        budget_bytes: int | None = None,
    ) -> None:
        self._store: OrderedDict[Any, Any] = (
            store if isinstance(store, OrderedDict) else OrderedDict(store or {})
        )
        self._budget_bytes = budget_bytes
        self._bytes = 0
        for value in self._store.values():
            self._bytes += _estimate(value)

    @property
    def store(self) -> dict[Any, Any]:
        return self._store

    @property
    def budget_bytes(self) -> int:
        if self._budget_bytes is not None and self._budget_bytes > 0:
            return self._budget_bytes
        return int(_governor().process_budget_bytes * 0.25)

    def _evict_to(self, target: int) -> int:
        freed = 0
        while self._store and self._bytes > target:
            _, value = self._store.popitem(last=False)
            size = _estimate(value)
            self._bytes = max(0, self._bytes - size)
            freed += size
        return freed

    def get(self, key: Any) -> Any | None:
        if key in self._store:
            value = self._store.pop(key)
            self._store[key] = value
            return value
        return None

    def get_for_series(self, series: Any) -> Any | None:
        """用 Series 身份键查 panel 缓存。"""
        return self.get(series_panel_cache_key(series))

    def set(self, key: Any, panel: Any) -> None:
        size = _estimate(panel)
        if size > self.budget_bytes:
            return
        if key in self._store:
            self._bytes -= _estimate(self._store[key])
        self._bytes += size
        self._store[key] = panel
        self._evict_to(self.budget_bytes)

    def set_for_series(self, series: Any, panel: Any) -> None:
        """缓存某 Series unstack 后的宽表 panel。"""
        self.set(series_panel_cache_key(series), panel)

    def evict_if_over_budget(self) -> int:
        return self._evict_to(self.budget_bytes)

    def __len__(self) -> int:
        return len(self._store)
