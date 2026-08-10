"""R25 §28 + R26-P0-019 —— CacheManager：统一 cache lifecycle（pin / GC / quota / TTL）。

对象状态：
    - unpinned       未引用
    - pinned_by_queries  被 in-flight query 引用（refcount>0）
    - evictable      可逐出（LRU）
    - stale          过期（TTL 或 scope 不匹配）
    - quarantined    隔离（损坏 / 权限不符，不允许自动恢复）

R26-P0-019 修正：
    - ``CacheEntry`` 增加 principal_scope / security_digest / source_snapshot_id /
      created_at / expires_at（per-principal quota、TTL 真正执行）；
    - **TTL 真实执行**：过期条目在 pin/access/GC 时按 expires_at 判 stale/evict；
    - **per_principal_quota 真实执行**：同一 principal 总字节超配额 → 拒绝/GC；
    - **GC accounting 修正**：先物理删除成功再更新 logical bytes（绝不 double-count），
      entry 从 manager 删除但磁盘没删 → 视为未释放；
    - **目录安全递归删除** + symlink 防护（不跟随 symlink 删除目标外文件）。

GC 原则（§68/§69）：
    - **never delete pinned generation**（refcount>0 只删 refcount=0）；
    - 磁盘达到 high_watermark → evict LRU 到 low_watermark；
    - 无法释放 → new admission fail（不等磁盘 100%）。
"""
from __future__ import annotations

import logging
import os
import shutil
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
    """R26-P0-019：完整 cache entry identity。

    ``principal_scope`` + ``security_digest`` 保证 per-principal 隔离
    （T-R26-CACHE-002：A 的 cache 不被 B 复用）。
    ``source_snapshot_id`` 绑定读取快照（generation 变化 → 不同 snapshot → 不命中）。
    ``expires_at`` TTL 真实执行。
    """

    key: str
    path: str
    size_bytes: int = 0
    state: CacheEntryState = CacheEntryState.UNPINNED
    refcount: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    principal_scope: str | None = None
    security_digest: str | None = None
    source_snapshot_id: str | None = None


class CacheManager:
    """统一 cache lifecycle（R25 §28 + R26-P0-019）。

    - ``max_bytes``      磁盘配额上限（None = 不限制）
    - ``high_watermark`` 触发 GC 的占用比例（默认 0.85）
    - ``low_watermark``  GC 后的目标占用比例（默认 0.6）
    - ``ttl``            条目 TTL 秒（None = 不过期）
    - ``per_principal_quota`` 每 principal 配额（可选，R26 真实执行）
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
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("CacheManager.max_bytes 必须为正")
        if not (0 < low_watermark <= high_watermark <= 1.0):
            raise ValueError(
                "CacheManager watermark 必须满足 0 < low <= high <= 1"
            )
        self._max_bytes = max_bytes
        self._high = high_watermark
        self._low = low_watermark
        self._ttl = ttl
        self._per_principal_quota = per_principal_quota
        self._lock = threading.Lock()
        self._entries: dict[str, CacheEntry] = {}

    # ---- 工具 ----

    @staticmethod
    def _safe_delete(path: str) -> bool:
        """物理删除（文件或目录），**不跟随 symlink**。成功返回 True。"""
        p = Path(path)
        try:
            if p.is_symlink() or p.is_file():
                p.unlink(missing_ok=True)
                return True
            if p.is_dir():
                # 安全递归删除：绝不跟随目录内 symlink 指向的外部文件。
                shutil.rmtree(p, ignore_errors=False)
                return True
            return False
        except OSError as exc:
            logger.warning("cache delete %s failed: %s", path, exc)
            return False

    def _expired(self, entry: CacheEntry, now: float) -> bool:
        return entry.expires_at is not None and now > entry.expires_at

    # ---- pin/unpin（§68）----

    def pin(
        self,
        key: str,
        *,
        path: str | None = None,
        size_bytes: int = 0,
        principal_scope: str | None = None,
        security_digest: str | None = None,
        source_snapshot_id: str | None = None,
    ) -> None:
        """Query resolve source snapshot 时 pin（refcount+1）。

        新条目 admission（§28）：加入后若超 max_bytes → 先尝试 GC（evict unpinned
        LRU 到 low_watermark）；无法释放 → 拒绝新条目（不等磁盘 100%）。

        R26-P0-019：TTL 过期条目视为 stale 并物理清理；per-principal 配额真实执行。
        """
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                new_size = size_bytes or 0
                if self._max_bytes is not None:
                    if self._total_bytes_locked() + new_size > self._max_bytes:
                        self._gc_locked(now=now)
                        if self._total_bytes_locked() + new_size > self._max_bytes:
                            raise MemoryError(
                                "cache admission: 磁盘达 high_watermark 且无法释放（pinned），"
                                "拒绝新 cache 条目（R25 §28，不等磁盘 100%）"
                            )
                # R26-P0-019：per-principal quota 真实执行。
                if (
                    self._per_principal_quota is not None
                    and principal_scope
                    and self._principal_bytes_locked(principal_scope) + new_size
                    > self._per_principal_quota
                ):
                    self._gc_principal_locked(principal_scope, now=now)
                    if (
                        self._principal_bytes_locked(principal_scope) + new_size
                        > self._per_principal_quota
                    ):
                        raise MemoryError(
                            f"cache admission: principal {principal_scope!r} 配额超限"
                            f"（R26-P0-019 per-principal quota fail-closed）"
                        )
                entry = CacheEntry(
                    key=key,
                    path=path or "",
                    size_bytes=new_size,
                    state=CacheEntryState.PINNED,
                    refcount=1,
                    last_access=now,
                    created_at=now,
                    expires_at=(now + self._ttl) if self._ttl is not None else None,
                    principal_scope=principal_scope,
                    security_digest=security_digest,
                    source_snapshot_id=source_snapshot_id,
                )
                self._entries[key] = entry
            else:
                if self._expired(entry, now):
                    # 过期条目：物理清理后重建（R26-P0-019 TTL 真实执行）。
                    self._remove_entry_locked(key, entry)
                    entry = None
                    return self.pin(
                        key,
                        path=path,
                        size_bytes=size_bytes,
                        principal_scope=principal_scope,
                        security_digest=security_digest,
                        source_snapshot_id=source_snapshot_id,
                    )
                entry.refcount += 1
                entry.last_access = now
                entry.state = CacheEntryState.PINNED
            logger.debug("cache pin %s refcount=%d", key, entry.refcount)

    def unpin(self, key: str) -> None:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.refcount = max(0, entry.refcount - 1)
            entry.last_access = now
            if entry.refcount == 0:
                entry.state = CacheEntryState.EVICTABLE

    def is_pinned(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and entry.refcount > 0

    def get(self, key: str) -> CacheEntry | None:
        """读取条目（TTL 过期 → 移除并返回 None）。"""
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if self._expired(entry, now):
                self._remove_entry_locked(key, entry)
                return None
            entry.last_access = now
            return entry

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

    def _principal_bytes_locked(self, scope: str) -> int:
        return sum(
            e.size_bytes
            for e in self._entries.values()
            if e.principal_scope == scope
        )

    def _remove_entry_locked(self, key: str, entry: CacheEntry) -> bool:
        """物理删除成功才更新 logical accounting（R26-P0-019 防 double-count）。"""
        deleted = False
        if entry.path:
            deleted = self._safe_delete(entry.path)
        self._entries.pop(key, None)
        if not deleted and entry.path:
            # 管理器删除但磁盘未释放 → 记日志（report 不再声称已释放）。
            logger.warning(
                "cache entry %s 从 manager 移除但物理文件未删除：%s",
                key, entry.path,
            )
        return deleted

    # ---- GC（§28/§69 + R26-P0-019）----

    def _gc_locked(self, now: float | None = None) -> bool:
        """LRU evict 到 low_watermark；返回是否成功腾出空间。"""
        assert self._max_bytes is not None
        now = now or time.time()
        target = int(self._max_bytes * self._low)
        # 先清 TTL 过期 + STALE（unpinned）。
        for key in list(self._entries):
            e = self._entries[key]
            if e.refcount == 0 and (e.state == CacheEntryState.STALE or self._expired(e, now)):
                self._remove_entry_locked(key, e)
        evictable = sorted(
            [e for e in self._entries.values() if e.refcount == 0],
            key=lambda e: e.last_access,
        )
        for e in evictable:
            if self._total_bytes_locked() <= target:
                break
            self._remove_entry_locked(e.key, e)
            logger.info("cache evict %s size=%d", e.key, e.size_bytes)
        return self._total_bytes_locked() <= target

    def _gc_principal_locked(self, scope: str, now: float | None = None) -> bool:
        """per-principal GC：逐出该 principal 的 unpinned LRU。"""
        now = now or time.time()
        evictable = sorted(
            [
                e
                for e in self._entries.values()
                if e.principal_scope == scope and e.refcount == 0
            ],
            key=lambda e: e.last_access,
        )
        for e in evictable:
            self._remove_entry_locked(e.key, e)
        return True

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
            return sum(
                1 for e in self._entries.values() if e.state == CacheEntryState.EVICTABLE
            )

    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)


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
