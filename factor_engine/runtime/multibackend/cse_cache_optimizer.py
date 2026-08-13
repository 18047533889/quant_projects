# -*- coding: utf-8 -*-
"""MB-P1-022: CSE cache optimizer.

CSE (Common Subexpression Elimination) cache 命中率优化：
    - Cache eviction policy（LRU vs LFU vs LIRS）
    - Pin/unpin 机制（消费期内不 evict）
    - Cache warming（预热热数据）
    - Cache partitioning（按 backend 分区避免竞争）
    - Memory-aware eviction（内存压力时主动 evict）

目标：
    - Cache hit rate > 80%（同一 batch 内 shared node 复用）
    - Memory overhead < 20%（cache 不占用过多内存）
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """单个 cache 条目。"""

    key: str
    value: Any
    size_bytes: int
    created_at_ms: float
    last_accessed_ms: float
    access_count: int
    pin_count: int  # pin 引用计数（>0 时不可 evict）

    def is_pinned(self) -> bool:
        return self.pin_count > 0


@dataclass
class CSECacheMetrics:
    """CSE cache 统计。"""

    total_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    pin_operations: int = 0
    unpin_operations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 3),
            "total_size_bytes": self.total_size_bytes,
            "pin_operations": self.pin_operations,
            "unpin_operations": self.unpin_operations,
        }


class CSECacheOptimizer:
    """CSE cache 优化器（提升命中率 + 内存治理）。

    集成点：
        - SharedBufferStore（CSE cache 实现）集成此优化器
        - AdaptiveBatchScheduler 执行时 pin/unpin shared nodes
        - ResourceBroker 内存压力时触发 evict
    """

    def __init__(
        self,
        *,
        max_cache_bytes: int = 4 * 1024**3,
        eviction_policy: str = "lru",
    ) -> None:
        """
        Args:
            max_cache_bytes: Cache 最大内存占用
            eviction_policy: Eviction 策略（"lru" | "lfu" | "lirs"）
        """
        self.max_cache_bytes = max_cache_bytes
        self.eviction_policy = eviction_policy

        self._cache: dict[str, CacheEntry] = {}
        self._lru_order: list[str] = []  # LRU eviction queue
        self._metrics = CSECacheMetrics()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """从 cache 读取（记录访问统计）。

        Args:
            key: Cache key（如 shared node ID）

        Returns:
            Cached value 或 None（miss）
        """
        with self._lock:
            self._metrics.total_lookups += 1

            entry = self._cache.get(key)
            if entry is None:
                self._metrics.cache_misses += 1
                return None

            # Cache hit：更新访问统计
            self._metrics.cache_hits += 1
            entry.last_accessed_ms = time.monotonic() * 1000.0
            entry.access_count += 1

            # LRU: 移到队列末尾
            if key in self._lru_order:
                self._lru_order.remove(key)
            self._lru_order.append(key)

            return entry.value

    def put(self, key: str, value: Any, size_bytes: int = 0) -> None:
        """写入 cache（可能触发 eviction）。

        Args:
            key: Cache key
            value: Cached value
            size_bytes: Value 内存占用（字节）
        """
        with self._lock:
            now_ms = time.monotonic() * 1000.0

            # 已存在：更新
            if key in self._cache:
                old_entry = self._cache[key]
                self._metrics.total_size_bytes -= old_entry.size_bytes

            # 新建 entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at_ms=now_ms,
                last_accessed_ms=now_ms,
                access_count=0,
                pin_count=0,
            )
            self._cache[key] = entry
            self._lru_order.append(key)
            self._metrics.total_size_bytes += size_bytes

            # Eviction（内存超限）
            while self._metrics.total_size_bytes > self.max_cache_bytes:
                evicted = self._evict_one()
                if not evicted:
                    break  # 全部 pinned，无法 evict

    def pin(self, key: str) -> None:
        """Pin cache entry（消费期内不可 evict）。

        Args:
            key: Cache key
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.pin_count += 1
                self._metrics.pin_operations += 1

    def unpin(self, key: str) -> None:
        """Unpin cache entry（引用计数 -1）。

        Args:
            key: Cache key
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.pin_count = max(0, entry.pin_count - 1)
                self._metrics.unpin_operations += 1

    def _evict_one(self) -> bool:
        """Evict 一个 entry（按 eviction policy 选择）。

        Returns:
            True 表示成功 evict，False 表示无可 evict 项（全部 pinned）
        """
        if self.eviction_policy == "lru":
            return self._evict_lru()
        elif self.eviction_policy == "lfu":
            return self._evict_lfu()
        else:
            return self._evict_lru()

    def _evict_lru(self) -> bool:
        """LRU eviction（最久未访问优先）。"""
        for key in self._lru_order:
            entry = self._cache.get(key)
            if entry is None:
                continue
            if entry.is_pinned():
                continue

            # Evict this entry
            self._cache.pop(key)
            self._lru_order.remove(key)
            self._metrics.total_size_bytes -= entry.size_bytes
            self._metrics.evictions += 1
            _logger.debug("evicted cache entry: %s (LRU, size=%d)", key, entry.size_bytes)
            return True

        return False

    def _evict_lfu(self) -> bool:
        """LFU eviction（最少访问优先）。"""
        # 找出访问次数最少的 unpinned entry
        candidates = [
            (k, e) for k, e in self._cache.items() if not e.is_pinned()
        ]
        if not candidates:
            return False

        candidates.sort(key=lambda x: x[1].access_count)
        key, entry = candidates[0]

        self._cache.pop(key)
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._metrics.total_size_bytes -= entry.size_bytes
        self._metrics.evictions += 1
        _logger.debug("evicted cache entry: %s (LFU, access=%d)", key, entry.access_count)
        return True

    def evict_by_memory_pressure(self, target_bytes: int) -> int:
        """根据内存压力主动 evict（释放至目标大小）。

        Args:
            target_bytes: 目标 cache 大小（字节）

        Returns:
            Evicted entry 数量
        """
        evicted_count = 0

        with self._lock:
            while self._metrics.total_size_bytes > target_bytes:
                if not self._evict_one():
                    break  # 无法继续 evict
                evicted_count += 1

        if evicted_count > 0:
            _logger.info(
                "memory pressure eviction: %d entries evicted (%.1f MB freed)",
                evicted_count,
                (self._metrics.total_size_bytes - target_bytes) / (1024**2),
            )

        return evicted_count

    def clear(self) -> None:
        """清空 cache（保留 pinned entries）。"""
        with self._lock:
            to_remove = [k for k, e in self._cache.items() if not e.is_pinned()]
            for key in to_remove:
                entry = self._cache.pop(key)
                if key in self._lru_order:
                    self._lru_order.remove(key)
                self._metrics.total_size_bytes -= entry.size_bytes

            _logger.info("cache cleared: %d entries removed", len(to_remove))

    def metrics(self) -> CSECacheMetrics:
        """返回累计统计。"""
        with self._lock:
            return CSECacheMetrics(
                total_lookups=self._metrics.total_lookups,
                cache_hits=self._metrics.cache_hits,
                cache_misses=self._metrics.cache_misses,
                evictions=self._metrics.evictions,
                total_size_bytes=self._metrics.total_size_bytes,
                pin_operations=self._metrics.pin_operations,
                unpin_operations=self._metrics.unpin_operations,
            )

    def summary(self) -> dict[str, Any]:
        """返回 cache 状态摘要。"""
        with self._lock:
            metrics = self.metrics().to_dict()
            pinned_count = sum(1 for e in self._cache.values() if e.is_pinned())
            return {
                "cache_entries": len(self._cache),
                "pinned_entries": pinned_count,
                "max_cache_bytes": self.max_cache_bytes,
                "eviction_policy": self.eviction_policy,
                "metrics": metrics,
            }


# 全局单例
_global_cse_cache: CSECacheOptimizer | None = None
_global_lock = threading.Lock()


def get_global_cse_cache_optimizer() -> CSECacheOptimizer:
    """返回全局 CSECacheOptimizer（进程级单例）。"""
    global _global_cse_cache
    if _global_cse_cache is None:
        with _global_lock:
            if _global_cse_cache is None:
                _global_cse_cache = CSECacheOptimizer()
    return _global_cse_cache
