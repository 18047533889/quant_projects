"""L0 表达式缓存：CSE ``plan_ref`` / rolling 共享子树结果（字节感知 LRU，受 MemoryGovernor 治理）。

Phase 5 R6：CSE 共享结果同样归入全局内存预算，超过 ``cse_budget`` 时按 LRU 逐出，
避免几十棵大 panel CSE（10年×5000标的）常驻导致 OOM。
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from cache.layers import CacheHitStats, CacheLayer


def _governor():
    """惰性导入，避免 cache → runtime → storage → cache 循环导入。"""
    from runtime.resource_governor import global_memory_governor

    return global_memory_governor()


def _estimate(value: Any) -> int:
    from runtime.resource_governor import estimate_object_bytes

    return estimate_object_bytes(value)


class ExpressionCache:
    """``ExecutionContext.shared_result_cache`` 的字节感知封装。"""

    def __init__(
        self,
        store: dict[str, Any] | None = None,
        *,
        stats: CacheHitStats | None = None,
        budget_bytes: int | None = None,
        layer_name: str = "l0_cse",
    ) -> None:
        self._store: OrderedDict[str, Any] = (
            store if isinstance(store, OrderedDict) else OrderedDict(store or {})
        )
        self._stats = stats
        self._budget_bytes = budget_bytes
        self.layer_name = layer_name
        self._bytes = 0
        self._lock = threading.RLock()
        for value in self._store.values():
            size = _estimate(value)
            self._bytes += size
            _governor().reserve_accounting(self.layer_name, size)

    @property
    def store(self) -> dict[str, Any]:
        return self._store

    @property
    def budget_bytes(self) -> int:
        if self._budget_bytes is not None and self._budget_bytes > 0:
            return self._budget_bytes
        return int(_governor().process_budget_bytes * 0.30)

    def _evict_to(self, target: int) -> int:
        """LRU 逐出直到 ``_bytes <= target``，返回释放字节数。"""
        freed = 0
        while self._store and self._bytes > target:
            _, value = self._store.popitem(last=False)
            freed += _estimate(value)
            self._bytes = max(0, self._bytes - _estimate(value))
        return freed

    def get(self, sid: str) -> Any | None:
        """按 CSE 子树 id 取共享结果；命中/未命中更新 stats，命中时 bump LRU。"""
        with self._lock:
            if sid in self._store:
                if self._stats is not None:
                    self._stats.record_hit(CacheLayer.L0_CSE)
                value = self._store.pop(sid)
                self._store[sid] = value
                return value
            if self._stats is not None:
                self._stats.record_miss(CacheLayer.L0_CSE)
            return None

    def set(self, sid: str, value: Any) -> None:
        """写入 CSE 共享子树结果（Series 或 LazyFrame）；受字节预算约束。"""
        size = _estimate(value)
        if size > self.budget_bytes:
            # 单对象就超预算：宁可让调用方按需重算，也不要它撑爆内存
            return
        gov = _governor()
        # R20-119..124：dict mutation / byte counter / governor accounting 原子。
        with gov.lock:
            with self._lock:
                if sid in self._store:
                    old_size = _estimate(self._store[sid])
                    self._bytes -= old_size
                    gov.release_accounting(self.layer_name, old_size)
                    # R20-146..152：overwrite → MRU。
                    del self._store[sid]
                self._bytes += size
                gov.reserve_accounting(self.layer_name, size)
                self._store[sid] = value
                freed = self._evict_to(self.budget_bytes)
                if freed > 0:
                    gov.release_accounting(self.layer_name, freed)

    def release(self, sid: str) -> None:
        """显式释放某 sid（CSE 引用计数归零时立即回收，P0-5）。"""
        # 锁顺序恒为 governor → cache（与 set / evict hook 一致，避免锁反转死锁）。
        gov = _governor()
        with gov.lock:
            with self._lock:
                if sid in self._store:
                    size = _estimate(self._store[sid])
                    self._bytes -= size
                    del self._store[sid]
                    gov.release_accounting(self.layer_name, size)

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
