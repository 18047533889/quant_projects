"""分层缓存：L1(内存快) → L2(Redis中) → L3(磁盘慢)。

R43 分层缓存策略：
    - L1: 内存（快，小）—— 热数据
    - L2: Redis（中等，中等）—— 共享缓存
    - L3: 磁盘（慢，大）—— 持久化缓存
    - 自动提升（L3 hit → 写入 L2，L2 hit → 写入 L1）
    - 级联写入（可选）
    - 级联失效
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Generic, TypeVar

from .unified_cache import UnifiedCache, CacheStats, EvictionPolicy

T = TypeVar("T")


class TieredCache(Generic[T]):
    """分层缓存（L1/L2/L3 级联）。

    特性：
        - 多层级联（L1 miss → L2 → L3）
        - 自动提升（下层 hit 自动写入上层）
        - 级联写入（写入时同时写 L1/L2/L3）
        - 级联失效（invalidate 所有层）
        - 统一统计（合并各层指标）
    """

    def __init__(
        self,
        *,
        l1: UnifiedCache[T] | None = None,
        l2: UnifiedCache[T] | None = None,
        l3: UnifiedCache[T] | None = None,
        auto_promote: bool = True,
        cascade_write: bool = False,
    ) -> None:
        """初始化分层缓存。

        Args:
            l1: L1 缓存（内存，快）
            l2: L2 缓存（Redis，中等）
            l3: L3 缓存（磁盘，慢）
            auto_promote: 是否自动提升（下层 hit → 写入上层）
            cascade_write: 是否级联写入（写入所有层）
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.auto_promote = auto_promote
        self.cascade_write = cascade_write
        self._lock = threading.RLock()

    def get(
        self,
        key: str,
        factory: Callable[[], T] | None = None,
        ttl: float | None = None,
    ) -> T | None:
        """从缓存读取（级联查找 L1 → L2 → L3）。

        Args:
            key: 缓存键
            factory: 缺失时的工厂函数
            ttl: 写入时的 TTL

        Returns:
            缓存值或 None
        """
        with self._lock:
            # L1 查找
            if self.l1 is not None:
                value = self.l1.get(key)
                if value is not None:
                    return value

            # L2 查找
            if self.l2 is not None:
                value = self.l2.get(key)
                if value is not None:
                    # 自动提升到 L1
                    if self.auto_promote and self.l1 is not None:
                        self.l1.set(key, value, ttl=ttl)
                    return value

            # L3 查找
            if self.l3 is not None:
                value = self.l3.get(key)
                if value is not None:
                    # 自动提升到 L2 和 L1
                    if self.auto_promote:
                        if self.l2 is not None:
                            self.l2.set(key, value, ttl=ttl)
                        if self.l1 is not None:
                            self.l1.set(key, value, ttl=ttl)
                    return value

            # 所有层都 miss：调用 factory
            if factory is not None:
                value = factory()
                self.set(key, value, ttl=ttl)
                return value

            return None

    def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """写入缓存。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: TTL（秒）
        """
        with self._lock:
            if self.cascade_write:
                # 级联写入所有层
                if self.l1 is not None:
                    self.l1.set(key, value, ttl=ttl)
                if self.l2 is not None:
                    self.l2.set(key, value, ttl=ttl)
                if self.l3 is not None:
                    self.l3.set(key, value, ttl=ttl)
            else:
                # 只写入 L1（或第一个可用层）
                if self.l1 is not None:
                    self.l1.set(key, value, ttl=ttl)
                elif self.l2 is not None:
                    self.l2.set(key, value, ttl=ttl)
                elif self.l3 is not None:
                    self.l3.set(key, value, ttl=ttl)

    def invalidate(self, key: str) -> int:
        """使某个键在所有层失效。

        Args:
            key: 缓存键

        Returns:
            失效的层数
        """
        with self._lock:
            count = 0
            if self.l1 is not None and self.l1.invalidate(key):
                count += 1
            if self.l2 is not None and self.l2.invalidate(key):
                count += 1
            if self.l3 is not None and self.l3.invalidate(key):
                count += 1
            return count

    def clear(self, pattern: str | None = None) -> dict[str, int]:
        """清空所有层（支持模式匹配）。

        Args:
            pattern: 匹配模式（None = 清空全部）

        Returns:
            各层清除数量
        """
        with self._lock:
            result = {}
            if self.l1 is not None:
                result["l1"] = self.l1.clear(pattern=pattern)
            if self.l2 is not None:
                result["l2"] = self.l2.clear(pattern=pattern)
            if self.l3 is not None:
                result["l3"] = self.l3.clear(pattern=pattern)
            return result

    def stats(self) -> dict[str, Any]:
        """返回所有层的统计指标。

        Returns:
            包含各层统计的字典
        """
        with self._lock:
            result: dict[str, Any] = {}
            if self.l1 is not None:
                result["l1"] = self.l1.stats().to_dict()
            if self.l2 is not None:
                result["l2"] = self.l2.stats().to_dict()
            if self.l3 is not None:
                result["l3"] = self.l3.stats().to_dict()

            # 合并总计
            total = CacheStats()
            for layer_stats in [self.l1, self.l2, self.l3]:
                if layer_stats is not None:
                    s = layer_stats.stats()
                    total.hits += s.hits
                    total.misses += s.misses
                    total.evictions += s.evictions
                    total.expirations += s.expirations
                    total.sets += s.sets
                    total.deletes += s.deletes
                    total.size_bytes += s.size_bytes
                    total.entry_count += s.entry_count

            result["total"] = total.to_dict()
            return result

    def pin(self, key: str) -> int:
        """Pin 某个键在所有层。

        Args:
            key: 缓存键

        Returns:
            成功 pin 的层数
        """
        with self._lock:
            count = 0
            if self.l1 is not None and self.l1.pin(key):
                count += 1
            if self.l2 is not None and self.l2.pin(key):
                count += 1
            if self.l3 is not None and self.l3.pin(key):
                count += 1
            return count

    def unpin(self, key: str) -> int:
        """Unpin 某个键在所有层。

        Args:
            key: 缓存键

        Returns:
            成功 unpin 的层数
        """
        with self._lock:
            count = 0
            if self.l1 is not None and self.l1.unpin(key):
                count += 1
            if self.l2 is not None and self.l2.unpin(key):
                count += 1
            if self.l3 is not None and self.l3.unpin(key):
                count += 1
            return count


def create_standard_tiered_cache(
    *,
    l1_max_bytes: int = 256 * 1024**2,  # 256 MB
    l2_max_bytes: int = 1024 * 1024**2,  # 1 GB
    l3_max_bytes: int = 10 * 1024**3,  # 10 GB
    default_ttl: float | None = None,
    auto_promote: bool = True,
    cascade_write: bool = False,
) -> TieredCache:
    """创建标准三层缓存。

    Args:
        l1_max_bytes: L1 最大字节数（内存）
        l2_max_bytes: L2 最大字节数（Redis/内存）
        l3_max_bytes: L3 最大字节数（磁盘）
        default_ttl: 默认 TTL（秒）
        auto_promote: 是否自动提升
        cascade_write: 是否级联写入

    Returns:
        配置好的分层缓存
    """
    l1 = UnifiedCache(
        max_bytes=l1_max_bytes,
        eviction_policy=EvictionPolicy.LRU,
        default_ttl=default_ttl,
        name="l1_memory",
    )

    l2 = UnifiedCache(
        max_bytes=l2_max_bytes,
        eviction_policy=EvictionPolicy.LRU,
        default_ttl=default_ttl,
        name="l2_shared",
    )

    l3 = UnifiedCache(
        max_bytes=l3_max_bytes,
        eviction_policy=EvictionPolicy.LRU,
        default_ttl=default_ttl,
        name="l3_disk",
    )

    return TieredCache(
        l1=l1,
        l2=l2,
        l3=l3,
        auto_promote=auto_promote,
        cascade_write=cascade_write,
    )
