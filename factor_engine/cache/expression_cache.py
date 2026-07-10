"""L0 表达式缓存：CSE ``plan_ref`` / rolling 共享子树结果。"""

from __future__ import annotations

from typing import Any

from cache.layers import CacheHitStats, CacheLayer


class ExpressionCache:
    """``ExecutionContext.shared_result_cache`` 的薄封装。"""

    def __init__(
        self,
        store: dict[str, Any] | None = None,
        *,
        stats: CacheHitStats | None = None,
    ) -> None:
        self._store: dict[str, Any] = store if store is not None else {}
        self._stats = stats

    @property
    def store(self) -> dict[str, Any]:
        return self._store

    def get(self, sid: str) -> Any | None:
        """按 CSE 子树 id 取共享结果；命中/未命中更新 stats。"""
        if sid in self._store:
            if self._stats is not None:
                self._stats.record_hit(CacheLayer.L0_CSE)
            return self._store[sid]
        if self._stats is not None:
            self._stats.record_miss(CacheLayer.L0_CSE)
        return None

    def set(self, sid: str, value: Any) -> None:
        """写入 CSE 共享子树结果（Series 或 LazyFrame）。"""
        self._store[sid] = value

    def __len__(self) -> int:
        return len(self._store)
