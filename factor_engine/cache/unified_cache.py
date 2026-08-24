"""统一缓存层：提供一致的接口、TTL、LRU 逐出和并发安全。

R43 缓存策略优化：
    - 统一接口（get/set/invalidate/clear/stats）
    - TTL 过期机制
    - LRU/LFU/FIFO 逐出策略
    - 并发安全（threading.RLock）
    - 分层支持（L1/L2/L3）
    - 智能失效（数据更新联动）
    - 内存治理集成
    - 指标统计（hits/misses/evictions）
"""
from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class EvictionPolicy(str, Enum):
    """缓存逐出策略。"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL_ONLY = "ttl_only"  # 仅 TTL 过期，不主动逐出


@dataclass
class CacheEntry(Generic[T]):
    """缓存条目（带 TTL、访问统计、依赖关系）。"""
    key: str
    value: T
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    expires_at: float | None = None  # None = 永不过期
    pin_count: int = 0  # 引用计数（>0 时不可逐出）
    dependencies: set[str] = field(default_factory=set)  # 依赖的其他缓存键
    dependents: set[str] = field(default_factory=set)  # 依赖此键的其他缓存键

    def is_expired(self, now: float | None = None) -> bool:
        """是否已过期。"""
        if self.expires_at is None:
            return False
        if now is None:
            now = time.monotonic()
        return now >= self.expires_at

    def is_pinned(self) -> bool:
        """是否被 pin（不可逐出）。"""
        return self.pin_count > 0


@dataclass
class CacheStats:
    """缓存统计指标。"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    sets: int = 0
    deletes: int = 0
    size_bytes: int = 0
    entry_count: int = 0

    def hit_rate(self) -> float:
        """命中率。"""
        total = self.hits + self.misses
        return self.hits / max(1, total)

    def to_dict(self) -> dict[str, Any]:
        """导出为字典。"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "sets": self.sets,
            "deletes": self.deletes,
            "hit_rate": round(self.hit_rate(), 3),
            "size_bytes": self.size_bytes,
            "entry_count": self.entry_count,
        }


class CacheBackend(ABC, Generic[T]):
    """缓存后端抽象接口（支持多种存储）。"""

    @abstractmethod
    def get(self, key: str) -> CacheEntry[T] | None:
        """读取条目。"""
        pass

    @abstractmethod
    def set(self, entry: CacheEntry[T]) -> None:
        """写入条目。"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除条目，返回是否存在。"""
        pass

    @abstractmethod
    def clear(self) -> int:
        """清空所有条目，返回删除数量。"""
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        """返回所有键。"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """返回条目数量。"""
        pass


class MemoryCacheBackend(CacheBackend[T]):
    """内存缓存后端（OrderedDict 实现 LRU）。"""

    def __init__(self) -> None:
        self._store: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> CacheEntry[T] | None:
        with self._lock:
            return self._store.get(key)

    def set(self, entry: CacheEntry[T]) -> None:
        with self._lock:
            # 移除旧条目（如存在）并添加到末尾（MRU）
            if entry.key in self._store:
                del self._store[entry.key]
            self._store[entry.key] = entry

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def move_to_end(self, key: str) -> None:
        """将键移到末尾（标记为最近使用）。"""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)

    def popitem_lru(self) -> tuple[str, CacheEntry[T]] | None:
        """弹出最久未使用的条目。"""
        with self._lock:
            if not self._store:
                return None
            key, entry = self._store.popitem(last=False)
            return key, entry

    def items(self) -> list[tuple[str, CacheEntry[T]]]:
        """返回所有条目（复制）。"""
        with self._lock:
            return list(self._store.items())


class UnifiedCache(Generic[T]):
    """统一缓存接口（集成 TTL、LRU、并发安全、内存治理）。

    特性：
        - 支持 factory 函数（get 时自动填充）
        - TTL 自动过期
        - 多种逐出策略（LRU/LFU/FIFO）
        - Pin/unpin 机制（防止关键数据被逐出）
        - 模式匹配清除（通配符/正则）
        - 内存预算管理
        - 统计指标
    """

    def __init__(
        self,
        *,
        max_size: int | None = None,
        max_bytes: int | None = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        default_ttl: float | None = None,
        backend: CacheBackend[T] | None = None,
        size_estimator: Callable[[T], int] | None = None,
        name: str = "cache",
    ) -> None:
        """初始化统一缓存。

        Args:
            max_size: 最大条目数（None = 无限制）
            max_bytes: 最大字节数（None = 无限制）
            eviction_policy: 逐出策略
            default_ttl: 默认 TTL（秒，None = 永不过期）
            backend: 缓存后端（None = 使用内存后端）
            size_estimator: 大小估算函数（None = 使用默认）
            name: 缓存名称（用于日志/统计）
        """
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.name = name

        self._backend = backend or MemoryCacheBackend[T]()
        self._size_estimator = size_estimator or self._default_size_estimator
        self._stats = CacheStats()
        self._lock = threading.RLock()

    @staticmethod
    def _default_size_estimator(value: Any) -> int:
        """默认大小估算（惰性导入 resource_governor）。"""
        try:
            from factor_engine.runtime.resource_governor import estimate_object_bytes
            return estimate_object_bytes(value)
        except Exception:
            # Fallback: 简单估算
            import sys
            return sys.getsizeof(value)

    def _is_over_budget(self) -> bool:
        """是否超出预算。"""
        if self.max_size and self._stats.entry_count > self.max_size:
            return True
        if self.max_bytes and self._stats.size_bytes > self.max_bytes:
            return True
        return False

    def _evict_expired(self) -> int:
        """清除所有过期条目，返回清除数量。"""
        now = time.monotonic()
        expired_keys = []

        # 查找过期条目
        if isinstance(self._backend, MemoryCacheBackend):
            with self._lock:
                for key, entry in self._backend.items():
                    if entry.is_expired(now):
                        expired_keys.append(key)
        else:
            # 通用后端：遍历所有键
            for key in self._backend.keys():
                entry = self._backend.get(key)
                if entry and entry.is_expired(now):
                    expired_keys.append(key)

        # 删除过期条目
        count = 0
        for key in expired_keys:
            entry = self._backend.get(key)
            if entry:
                self._backend.delete(key)
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                self._stats.expirations += 1
                count += 1

        return count

    def _evict_one(self) -> bool:
        """逐出一个条目（按策略选择），返回是否成功。"""
        if self.eviction_policy == EvictionPolicy.LRU:
            return self._evict_lru()
        elif self.eviction_policy == EvictionPolicy.LFU:
            return self._evict_lfu()
        elif self.eviction_policy == EvictionPolicy.FIFO:
            return self._evict_fifo()
        else:
            return False

    def _evict_lru(self) -> bool:
        """LRU 逐出（最久未访问）。"""
        if not isinstance(self._backend, MemoryCacheBackend):
            # 通用后端：查找 last_accessed 最小的
            candidates = []
            for key in self._backend.keys():
                entry = self._backend.get(key)
                if entry and not entry.is_pinned():
                    candidates.append((key, entry))
            if not candidates:
                return False
            candidates.sort(key=lambda x: x[1].last_accessed)
            key, entry = candidates[0]
            # 删除条目
            self._backend.delete(key)
        else:
            # MemoryCacheBackend：popitem(last=False) 直接取 LRU
            with self._lock:
                result = self._backend.popitem_lru()
                if not result:
                    return False
                key, entry = result
                # 如果 pinned，跳过并继续查找
                if entry.is_pinned():
                    # 放回末尾
                    self._backend.set(entry)
                    return False
                # popitem_lru 已经删除了，不需要再 delete

        # 更新统计
        self._stats.size_bytes -= entry.size_bytes
        self._stats.entry_count -= 1
        self._stats.evictions += 1
        return True

    def _evict_lfu(self) -> bool:
        """LFU 逐出（最少访问）。"""
        candidates = []
        for key in self._backend.keys():
            entry = self._backend.get(key)
            if entry and not entry.is_pinned():
                candidates.append((key, entry))

        if not candidates:
            return False

        # 按 access_count 排序
        candidates.sort(key=lambda x: x[1].access_count)
        key, entry = candidates[0]

        self._backend.delete(key)
        self._stats.size_bytes -= entry.size_bytes
        self._stats.entry_count -= 1
        self._stats.evictions += 1
        return True

    def _evict_fifo(self) -> bool:
        """FIFO 逐出（最早创建）。"""
        candidates = []
        for key in self._backend.keys():
            entry = self._backend.get(key)
            if entry and not entry.is_pinned():
                candidates.append((key, entry))

        if not candidates:
            return False

        # 按 created_at 排序
        candidates.sort(key=lambda x: x[1].created_at)
        key, entry = candidates[0]

        self._backend.delete(key)
        self._stats.size_bytes -= entry.size_bytes
        self._stats.entry_count -= 1
        self._stats.evictions += 1
        return True

    def _evict_to_budget(self) -> int:
        """逐出条目直到满足预算，返回逐出数量。"""
        count = 0
        while self._is_over_budget():
            if not self._evict_one():
                break  # 无法继续逐出（全部 pinned）
            count += 1
        return count

    def get(
        self,
        key: str,
        factory: Callable[[], T] | None = None,
        ttl: float | None = None,
    ) -> T | None:
        """读取缓存（支持 factory 自动填充）。

        Args:
            key: 缓存键
            factory: 缺失时的工厂函数（None = 返回 None）
            ttl: 写入时的 TTL（秒，None = 使用 default_ttl）

        Returns:
            缓存值或 None
        """
        with self._lock:
            # 先清理过期条目
            self._evict_expired()

            entry = self._backend.get(key)
            now = time.monotonic()

            if entry is None or entry.is_expired(now):
                self._stats.misses += 1

                # 过期条目删除
                if entry is not None:
                    self._backend.delete(key)
                    self._stats.size_bytes -= entry.size_bytes
                    self._stats.entry_count -= 1
                    self._stats.expirations += 1

                # factory 填充
                if factory is not None:
                    value = factory()
                    self.set(key, value, ttl=ttl)
                    return value

                return None

            # 命中：更新访问统计
            self._stats.hits += 1
            entry.last_accessed = now
            entry.access_count += 1

            # LRU: 移到末尾
            if self.eviction_policy == EvictionPolicy.LRU:
                if isinstance(self._backend, MemoryCacheBackend):
                    self._backend.move_to_end(key)

            return entry.value

    def set(self, key: str, value: T, ttl: float | None = None, absolute_ttl: float | None = None) -> None:
        """写入缓存。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 相对 TTL（秒，None = 使用 default_ttl）
            absolute_ttl: 绝对 TTL（时间戳，None = 使用 ttl 或 default_ttl）
        """
        with self._lock:
            now = time.monotonic()
            size = self._size_estimator(value)

            # 确定过期时间：优先使用 absolute_ttl，其次 ttl，最后 default_ttl
            if absolute_ttl is not None:
                expires_at = absolute_ttl
            elif ttl is not None:
                expires_at = now + ttl
            elif self.default_ttl is not None:
                expires_at = now + self.default_ttl
            else:
                expires_at = None

            # 删除旧条目（如存在）
            old_entry = self._backend.get(key)
            if old_entry:
                self._stats.size_bytes -= old_entry.size_bytes
                self._stats.entry_count -= 1

            # 创建新条目
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size,
                created_at=now,
                last_accessed=now,
                access_count=0,
                expires_at=expires_at,
                pin_count=0,
            )

            self._backend.set(entry)
            self._stats.size_bytes += size
            self._stats.entry_count += 1
            self._stats.sets += 1

            # 检查预算并逐出
            self._evict_to_budget()

    def invalidate(self, key: str) -> bool:
        """使某个键失效（删除），并级联失效依赖此键的所有缓存。

        Args:
            key: 缓存键

        Returns:
            是否存在该键
        """
        with self._lock:
            entry = self._backend.get(key)
            if entry:
                # 级联失效依赖此键的缓存
                for dependent_key in entry.dependents.copy():
                    self.invalidate(dependent_key)

                # 从依赖项中移除自身
                for dep_key in entry.dependencies:
                    dep_entry = self._backend.get(dep_key)
                    if dep_entry:
                        dep_entry.dependents.discard(key)

                self._backend.delete(key)
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                self._stats.deletes += 1
                return True
            return False

    def add_dependency(self, key: str, depends_on: str) -> bool:
        """添加依赖关系：key 依赖 depends_on。

        Args:
            key: 缓存键
            depends_on: 依赖的缓存键

        Returns:
            是否成功添加依赖
        """
        with self._lock:
            entry = self._backend.get(key)
            dep_entry = self._backend.get(depends_on)

            if entry and dep_entry:
                entry.dependencies.add(depends_on)
                dep_entry.dependents.add(key)
                # 更新后端
                self._backend.set(entry)
                self._backend.set(dep_entry)
                return True
            return False

    def clear(self, pattern: str | None = None) -> int:
        """清空缓存（支持模式匹配）。

        Args:
            pattern: 匹配模式（None = 清空全部，否则清空匹配的键）
                     支持通配符：'prefix:*', '*:suffix', '*substring*'

        Returns:
            清除数量
        """
        with self._lock:
            if pattern is None:
                count = self._backend.clear()
                self._stats.size_bytes = 0
                self._stats.entry_count = 0
                return count

            # 模式匹配清除
            to_delete = []
            for key in self._backend.keys():
                if self._match_pattern(key, pattern):
                    to_delete.append(key)

            count = 0
            for key in to_delete:
                if self.invalidate(key):
                    count += 1

            return count

    @staticmethod
    def _match_pattern(key: str, pattern: str) -> bool:
        """简单通配符匹配。"""
        if "*" not in pattern:
            return key == pattern

        if pattern == "*":
            return True

        if pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in key
        elif pattern.startswith("*"):
            return key.endswith(pattern[1:])
        elif pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        else:
            # 通用通配符匹配
            import re
            regex = pattern.replace("*", ".*")
            return bool(re.match(f"^{regex}$", key))

    def pin(self, key: str) -> bool:
        """Pin 某个键（防止被逐出）。

        Args:
            key: 缓存键

        Returns:
            是否存在该键
        """
        with self._lock:
            entry = self._backend.get(key)
            if entry:
                entry.pin_count += 1
                # 更新后端（确保 pin_count 持久化）
                self._backend.set(entry)
                return True
            return False

    def unpin(self, key: str) -> bool:
        """Unpin 某个键（允许被逐出）。

        Args:
            key: 缓存键

        Returns:
            是否存在该键
        """
        with self._lock:
            entry = self._backend.get(key)
            if entry:
                entry.pin_count = max(0, entry.pin_count - 1)
                self._backend.set(entry)
                return True
            return False

    def stats(self) -> CacheStats:
        """返回统计指标（副本）。"""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                sets=self._stats.sets,
                deletes=self._stats.deletes,
                size_bytes=self._stats.size_bytes,
                entry_count=self._stats.entry_count,
            )

    def __len__(self) -> int:
        """返回条目数量。"""
        return len(self._backend)

    def __contains__(self, key: str) -> bool:
        """检查键是否存在（不更新访问统计）。"""
        entry = self._backend.get(key)
        if entry and not entry.is_expired():
            return True
        return False
