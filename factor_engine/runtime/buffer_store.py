# -*- coding: utf-8 -*-
"""R36 §104/105/106：GovernedBufferStore —— CSE 共享缓冲的 governed 写入通道。

R36 P0-021（§104/106）：当前 CSE shared materialize 走 ``ctx.shared_result_cache[sid]
= value`` 的 **raw dict 写入**，绕过带 budget/LRU/governor 的 ExpressionCache。
修复：ExecutionContext 不暴露可直接写 raw dict 作为权威，改经 :class:`GovernedBufferStore`
的 ``put/get/release/spill`` 统一通道——byte 记账 + 预算拒绝 + 冷淘汰 + reconciliation。

R36 P0-023（§108/109）：``release(key)`` 必须**同时**移除 backing ref 与 accounting，
否则「实际内存仍在、账面已经没了」。``reconciliation()`` 定期对比 accounted vs
sampled actual bytes，漂移超阈值可检测（§109）。
"""
from __future__ import annotations

import threading
from typing import Any


def _estimate_bytes(value: Any) -> int:
    try:
        from runtime.resource_governor import estimate_object_bytes

        return max(0, int(estimate_object_bytes(value)))
    except Exception:
        return 0


class GovernedBufferStore:
    """带 byte 预算的 CSE 共享缓冲存储（§105 put/get/pin/release/spill）。"""

    def __init__(
        self,
        backing: dict[str, Any] | None = None,
        *,
        budget_bytes: int | None = None,
    ) -> None:
        self._backing: dict[str, Any] = backing if backing is not None else {}
        self._budget = budget_bytes if budget_bytes is not None and budget_bytes > 0 else None
        self._accounted: dict[str, int] = {}
        self._pinned: set[str] = set()
        self._lock = threading.RLock()
        self._writes = 0
        self._refused = 0
        self._releases = 0
        self._reconciliations = 0

    # -- 基础 --

    @property
    def budget_bytes(self) -> int | None:
        return self._budget

    def _total_accounted(self) -> int:
        return sum(self._accounted.values())

    def put(self, key: str, value: Any, *, bytes_: int | None = None) -> bool:
        """写入（§104）：经 budget 校验 + 记账。超预算先冷淘汰，仍不够则拒绝。

        拒绝不静默：记 ``refused``（hard gate R36_ZERO_RAW_CSE_CACHE_BYPASS 消费）。
        """
        size = max(0, int(bytes_)) if bytes_ is not None else _estimate_bytes(value)
        with self._lock:
            old = self._accounted.get(key, 0)
            if self._budget is not None:
                needed = self._total_accounted() - old + size
                if needed > self._budget:
                    # 冷淘汰（未 pinned）腾空间。
                    self._evict_for(needed - self._budget)
                    if self._total_accounted() - old + size > self._budget:
                        self._refused += 1
                        return False
            self._backing[key] = value
            self._accounted[key] = size
            self._writes += 1
            return True

    def get(self, key: str) -> Any | None:
        return self._backing.get(key)

    def pin(self, key: str) -> bool:
        with self._lock:
            if key not in self._backing:
                return False
            self._pinned.add(key)
            return True

    def release(self, key: str) -> None:
        """§108/109：backing ref 与 accounting **同时**释放（不留 ghost 账面）。"""
        with self._lock:
            existed = key in self._backing
            self._backing.pop(key, None)
            self._accounted.pop(key, None)
            self._pinned.discard(key)
            if existed:
                self._releases += 1

    def spill(self, key: str) -> None:
        """R36 §82/84：标记为 spill 候选（第一版：不物理落盘，仅释放记账保持正确）。"""
        self.release(key)

    def _evict_for(self, needed: int) -> int:
        """淘汰未 pinned 对象腾出空间（简单的 cold-first：先移除无 pin 的最小值）。"""
        freed = 0
        for key in [k for k in self._accounted if k not in self._pinned]:
            if freed >= needed:
                break
            size = self._accounted.pop(key, 0)
            self._backing.pop(key, None)
            freed += size
        return freed

    # -- reconciliation（§109） --

    def reconciliation(self, *, sample_limit: int = 512) -> dict[str, Any]:
        """对比 accounted bytes 与实际估算 bytes；漂移超阈值可 detect。"""
        with self._lock:
            accounted = self._total_accounted()
            keys = list(self._backing.keys())[:sample_limit]
            sampled_actual = sum(_estimate_bytes(self._backing[k]) for k in keys)
            drift = abs(accounted - sampled_actual)
            self._reconciliations += 1
            return {
                "accounted_bytes": accounted,
                "sampled_actual_bytes": sampled_actual,
                "sampled_keys": len(keys),
                "drift_bytes": drift,
                "reconciliations": self._reconciliations,
            }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "budget_bytes": self._budget,
                "accounted_bytes": self._total_accounted(),
                "keys": len(self._backing),
                "writes": self._writes,
                "refused": self._refused,
                "releases": self._releases,
                "pinned": len(self._pinned),
                "reconciliation": self.reconciliation(),
            }
