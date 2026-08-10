"""R25 §28 —— CacheManager：统一 cache lifecycle（pin / GC / quota）。

对象状态：
    - unpinned       未引用
    - pinned_by_queries  被 in-flight query 引用（refcount>0）
    - evictable      可逐出（LRU）
    - stale          过期（TTL 或 scope 不匹配）
    - quarantined    隔离（损坏 / 权限不符，不允许自动恢复）

GC 原则（§68/§69）：
    - **never delete pinned generation**（refcount>0 只删 refcount=0）；
    - 磁盘达到 high_watermark → evict LRU 到 low_watermark；
    - 无法释放 → new admission fail（不等磁盘 100%）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("data_access.cache_manager")


class CacheEntryState(Enum):
    UNPINNED = "unpinned"
    PINNED = "pinned_by_queries"
    EVICTABLE = "evictable"
    STALE = "stale"
    QUARANTINED = "quarantined"


@dataclass
class CacheEntry:
    key: str
    path: str
    size_bytes: int = 0
    state: CacheEntryState = CacheEntryState.UNPINNED
    refcount: int = 0
    last_access: float = field(default_factory=time.time)


class CacheManager:
    """统一 cache lifecycle（R25 §28）。

    - ``max_bytes``      磁盘配额上限（None = 不限制）
    - ``high_watermark`` 触发 GC 的占用比例（默认 0.85）
    - ``low_watermark``  GC 后的目标占用比例（默认 0.6）
    - ``ttl``            条目 TTL 秒（None = 不过期）
    - ``per_principal_quota`` 每 principal 配额（可选）
    """

    def __init__(
        self,
        *,
        max_bytes: int | None = None,
        high_watermark: float = 0.85,
        low_watermark: float = 0.6,
        ttl: float | None = None,
        per_principal_quota: int | None = None,
    ) -> None:
        self._max_bytes = max_bytes
        self._high = high_watermark
        self._low = low_watermark
        self._ttl = ttl
        self._per_principal_quota = per_principal_quota
        self._lock = threading.Lock()
        self._entries: dict[str, CacheEntry] = {}

    # ---- pin/unpin（§68）----

    def pin(self, key: str, *, path: str | None = None, size_bytes: int = 0) -> None:
        """Query resolve source snapshot 时 pin（refcount+1）。

        新条目 admission（§28）：加入后若超 max_bytes → 先尝试 GC（evict unpinned
        LRU 到 low_watermark）；无法释放 → 拒绝新条目（不等磁盘 100%）。
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                new_size = size_bytes or 0
                if self._max_bytes is not None:
                    if self._total_bytes_locked() + new_size > self._max_bytes:
                        self._gc_locked()
                        if self._total_bytes_locked() + new_size > self._max_bytes:
                            raise MemoryError(
                                "cache admission: 磁盘达 high_watermark 且无法释放（pinned），"
                                "拒绝新 cache 条目（R25 §28，不等磁盘 100%）"
                            )
                entry = CacheEntry(
                    key=key,
                    path=path or "",
                    size_bytes=new_size,
                    state=CacheEntryState.PINNED,
                    refcount=1,
                )
                self._entries[key] = entry
            else:
                entry.refcount += 1
                entry.last_access = time.time()
                entry.state = CacheEntryState.PINNED
            logger.debug("cache pin %s refcount=%d", key, entry.refcount)

    def unpin(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.refcount = max(0, entry.refcount - 1)
            entry.last_access = time.time()
            if entry.refcount == 0:
                entry.state = CacheEntryState.EVICTABLE

    def is_pinned(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and entry.refcount > 0

    # ---- 状态 ----

    def mark_stale(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.refcount == 0:
                entry.state = CacheEntryState.STALE

    def mark_quarantined(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.state = CacheEntryState.QUARANTINED

    def _total_bytes_locked(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    # ---- GC（§28/§69）----

    def _gc_locked(self) -> bool:
        """LRU evict 到 low_watermark；返回是否成功腾出空间。"""
        assert self._max_bytes is not None
        target = int(self._max_bytes * self._low)
        evictable = sorted(
            [e for e in self._entries.values() if e.refcount == 0],
            key=lambda e: e.last_access,
        )
        freed = 0
        for e in evictable:
            if self._total_bytes_locked() - freed <= target:
                break
            self._entries.pop(e.key, None)
            freed += e.size_bytes
            # 物理删除缓存文件（pinned 绝不删）。
            try:
                p = Path(e.path)
                if p.exists():
                    p.unlink()
            except OSError:
                pass
            logger.info("cache evict %s size=%d", e.key, e.size_bytes)
        return (self._total_bytes_locked() <= target) or (freed > 0)

    def run_gc(self) -> int:
        """显式 GC（high_watermark 触发）；返回逐出字节数。"""
        if self._max_bytes is None:
            return 0
        with self._lock:
            if self._total_bytes_locked() < int(self._max_bytes * self._high):
                return 0
            before = self._total_bytes_locked()
            self._gc_locked()
            return before - self._total_bytes_locked()

    # ---- 统计 ----

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes_locked()

    def pinned_bytes(self) -> int:
        with self._lock:
            return sum(
                e.size_bytes for e in self._entries.values() if e.refcount > 0
            )

    def eviction_count(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if e.state == CacheEntryState.EVICTABLE)


_cache_manager: CacheManager | None = None
_cache_manager_lock = threading.Lock()


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        with _cache_manager_lock:
            if _cache_manager is None:
                _cache_manager = CacheManager()
    return _cache_manager


def reset_cache_manager() -> None:
    global _cache_manager
    with _cache_manager_lock:
        _cache_manager = None
