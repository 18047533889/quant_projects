"""L1 Panel 缓存：Series → unstack 宽表复用（字节感知 LRU，受 MemoryGovernor 治理）。

Phase 5 R6：宽表 panel 往往是最大内存占用（5000 列 × 数千行），同样必须受
``panel_budget`` 约束，逐出后由 DataAccessSource 侧按需重算。
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np


def _governor():
    from runtime.resource_governor import global_memory_governor

    return global_memory_governor()


def _estimate(value: Any) -> int:
    from runtime.resource_governor import estimate_object_bytes

    return estimate_object_bytes(value)


def _content_hash(arr: Any) -> str:
    """对数组 / Index 做内容哈希（sha256 hexdigest）。

    审计 #331：
        - DatetimeIndex → ``asi8``（int64 纳秒）字节
        - MultiIndex → codes（int64）字节 + levels 字符串
        - 其余 Index / ndarray → ``tobytes()``；object dtype 逐元素字符串化
    """
    if arr is None:
        return ""
    # DatetimeIndex：int64 纳秒视图（C 速度，一次 tobytes）
    asi8 = getattr(arr, "asi8", None)
    if asi8 is not None:
        data = np.ascontiguousarray(np.asarray(asi8, dtype=np.int64))
        return hashlib.sha256(data.tobytes()).hexdigest()
    # MultiIndex：codes（int64）+ levels 字符串
    codes = getattr(arr, "codes", None)
    if codes is not None:
        h = hashlib.sha256()
        for c in codes:
            h.update(np.ascontiguousarray(np.asarray(c, dtype=np.int64)).tobytes())
        for lev in getattr(arr, "levels", ()):
            h.update("|".join(map(str, lev)).encode("utf-8"))
        return h.hexdigest()
    try:
        data = np.ascontiguousarray(np.asarray(arr))
        if data.dtype == object:
            return hashlib.sha256(
                "|".join(repr(x) for x in data.tolist()).encode("utf-8")
            ).hexdigest()
        return hashlib.sha256(data.tobytes()).hexdigest()
    except Exception:
        return hashlib.sha256(repr(arr).encode("utf-8")).hexdigest()


def series_panel_cache_key(series: Any) -> str:
    """稳定 cache key：基于内容的 sha256（值 buffer + index + name + length）。

    审计 #331：不再用 ``id(series)``——原地 ``series.iloc[:] = new_values`` 会
    改变 content 从而得到新键（同一 Series 对象也可命中更新后的 panel）。
    单次 ``tobytes()`` 是 C 速度，可接受。
    """
    name = getattr(series, "name", None)
    to_numpy = getattr(series, "to_numpy", None)
    if callable(to_numpy):
        values_bytes = _content_hash(np.ascontiguousarray(to_numpy()))
    else:
        values_bytes = _content_hash(getattr(series, "values", None))
    index_hash = _content_hash(getattr(series, "index", None))
    h = hashlib.sha256()
    h.update(values_bytes.encode("utf-8"))
    h.update(index_hash.encode("utf-8"))
    h.update(str(name if name is not None else "").encode("utf-8"))
    h.update(str(len(series)).encode("utf-8"))
    return h.hexdigest()


class PanelCache:
    """``ExecutionContext.panel_cache`` 的字节感知封装。"""

    def __init__(
        self,
        store: dict[Any, Any] | None = None,
        *,
        budget_bytes: int | None = None,
        layer_name: str = "l1_panel",
    ) -> None:
        self._store: OrderedDict[Any, Any] = (
            store if isinstance(store, OrderedDict) else OrderedDict(store or {})
        )
        self._budget_bytes = budget_bytes
        self.layer_name = layer_name
        self._bytes = 0
        self._lock = threading.RLock()
        for value in self._store.values():
            size = _estimate(value)
            self._bytes += size
            _governor().reserve_accounting(self.layer_name, size)

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
        with self._lock:
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
        gov = _governor()
        # R20-119..124：dict mutation / byte counter / governor accounting 原子。
        with gov.lock:
            with self._lock:
                if key in self._store:
                    old_size = _estimate(self._store[key])
                    self._bytes -= old_size
                    gov.release_accounting(self.layer_name, old_size)
                    # R20-146..152：overwrite → MRU（OrderedDict 对已有 key
                    # 重新赋值不改变插入位置）。
                    del self._store[key]
                self._bytes += size
                gov.reserve_accounting(self.layer_name, size)
                self._store[key] = panel
                freed = self._evict_to(self.budget_bytes)
                if freed > 0:
                    gov.release_accounting(self.layer_name, freed)

    def set_for_series(self, series: Any, panel: Any) -> None:
        """缓存某 Series unstack 后的宽表 panel。"""
        self.set(series_panel_cache_key(series), panel)

    def evict_if_over_budget(self, target: int = 0) -> int:
        """触发式逐出（MemoryGovernor evict hook）；返回释放字节数。

        审计 #332：``target > 0`` 用 ``target``，否则逐出到自身 ``budget_bytes``。
        记账由 ``MemoryGovernor._evict_for`` 统一扣减，这里不重复 release。
        """
        gov = _governor()
        with gov.lock:
            with self._lock:
                return self._evict_to(target if target > 0 else self.budget_bytes)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
