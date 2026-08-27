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

Fail-closed 治理（R21-CSE-CACHE-IDENTITY）：
    1. Cache key 绑定完整语义身份，绝不能用裸 shared-node 字符串当 key ——
       否则两个不同 snapshot/universe/params 的同一 node 会错误复用。
    2. ``size_bytes`` 未知/<=0 一律拒绝（CacheCapacityExceeded），绝不静默
       无上限缓存。
    3. 更新已有 key 时先把旧 key 从 ``_lru_order`` 移除再重排，避免重复 LRU
       项导致内存计量失真。
    4. 全部 pinned + 超容量时抛 CacheCapacityExceeded（backpressure），绝不静默
       ``break`` 突破硬上限。
    5. LIRS 未实现则 fail-closed（UnsupportedEvictionPolicy），绝不静默回落到 LRU。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from factor_engine.runtime.resource_errors import ResourceGovernanceError

_logger = logging.getLogger(__name__)


class CacheCapacityExceeded(ResourceGovernanceError):
    """CSE cache 超出硬性内存上限且无法释放（全部 pinned / 单条超限）。

    fail-closed：禁止 fallback，调用方必须处理 backpressure（spill / 拒绝）。
    """


class UnsupportedEvictionPolicy(ValueError):
    """请求的 eviction 策略未实现。

    fail-closed：不静默回退到 LRU —— LIRS 声称支持但未实现时构造即失败。
    """


@dataclass(frozen=True)
class CSECacheKey:
    """CSE cache 条目绑定的完整语义身份。

    任意一个维度变化都产生不同 key —— 同一 shared node id 但不同
    DataSnapshot / Universe / Market / DecisionClock / params / 物理实现 /
    semantic contract 不能互相复用缓存。

    字段（与任务指定维度一一对应）：
        logical_node:      LogicalNodeIdentity
        bound_params:      BoundParameterIdentity
        data_source:       DataReadIdentity
        universe:          UniverseSnapshotIdentity
        market:            Market
        decision_clock:    DecisionClock
        physical_impl:     PhysicalImplementationID
        semantic_contract: SemanticContractIdentity

    调用方负责把各权威 identity 对象折叠成稳定字符串（如 ``DecisionClock`` 的
    ``semantic_hash``、``PhysicalImplementationID.value``、universe 的
    ``membership_hash``）。本类只负责组合 + 稳定哈希。
    """

    logical_node: str = ""
    bound_params: str = ""
    data_source: str = ""
    universe: str = ""
    market: str = ""
    decision_clock: str = ""
    physical_impl: str = ""
    semantic_contract: str = ""

    def to_key(self) -> str:
        """稳定 canonical JSON —— 可直接作 dict key / checkpoint 后缀。"""
        return _canonical_json(asdict(self))

    def digest(self) -> str:
        """SHA-256 hex（完整 256-bit）—— 避免 64-bit 截断碰撞。"""
        return hashlib.sha256(self.to_key().encode("utf-8")).hexdigest()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CSECacheKey) and self.to_key() == other.to_key()

    def __hash__(self) -> int:
        return hash(self.to_key())


def _canonical_json(payload: dict[str, Any]) -> str:
    """字段排序 + 紧凑分隔的 canonical JSON（跨进程稳定）。"""
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class CacheEntry:
    """单个 cache 条目。"""

    key: CSECacheKey
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
    """CSE cache 优化器（提升命中率 + 内存治理，fail-closed）。

    集成点：
        - SharedBufferStore（CSE cache 实现）集成此优化器
        - AdaptiveBatchScheduler 执行时 pin/unpin shared nodes
        - ResourceBroker 内存压力时触发 evict
    """

    def __init__(
        self,
        *,
        # P3/P4: ``None`` → 调用方从 broker live headroom 派生 CSE 预算（不再固定
        # 4GiB）。本优化器只做账目，不自行探测内存。
        max_cache_bytes: int | None = None,
        eviction_policy: str = "lru",
    ) -> None:
        """
        Args:
            max_cache_bytes: Cache 最大内存占用（None → 绝对上限回退 4GiB，
                但单权威由 ResourceBroker 的 ``current_cse_budget()`` 提供）
            eviction_policy: Eviction 策略（"lru" | "lfu"）。"lirs" 未实现，
                fail-closed 抛 ``UnsupportedEvictionPolicy``，绝不静默回退 LRU。
        """
        if eviction_policy not in {"lru", "lfu"}:
            raise UnsupportedEvictionPolicy(
                f"eviction_policy={eviction_policy!r} is not implemented. "
                f"Supported: 'lru', 'lfu'. LIRS is claimed but not implemented — "
                f"refusing to silently fall back to LRU."
            )
        if max_cache_bytes is None:
            max_cache_bytes = 4 * 1024**3  # 绝对上限回退（单权威在 broker）
        self.max_cache_bytes = max_cache_bytes
        self.eviction_policy = eviction_policy

        self._cache: dict[CSECacheKey, CacheEntry] = {}
        self._lru_order: list[CSECacheKey] = []  # LRU eviction queue
        self._metrics = CSECacheMetrics()
        self._lock = threading.RLock()

    def _normalize_key(self, key: Any) -> CSECacheKey:
        """把用户 key 归一为完整语义 key。

        ``str`` 是向后兼容的裸 node id —— 仅作为 ``logical_node`` 维度，其余
        语义维度为空；生产应传完整 ``CSECacheKey``。
        """
        if isinstance(key, CSECacheKey):
            return key
        if isinstance(key, str):
            return CSECacheKey(logical_node=key)
        raise TypeError(
            f"cache key must be CSECacheKey or str, got {type(key).__module__}."
            f"{type(key).__qualname__}"
        )

    def get(self, key: Any) -> Any | None:
        """从 cache 读取（记录访问统计）。

        Args:
            key: Cache key（完整 ``CSECacheKey`` 或向后兼容的 str）

        Returns:
            Cached value 或 None（miss）
        """
        with self._lock:
            self._metrics.total_lookups += 1
            ckey = self._normalize_key(key)

            entry = self._cache.get(ckey)
            if entry is None:
                self._metrics.cache_misses += 1
                return None

            # Cache hit：更新访问统计
            self._metrics.cache_hits += 1
            entry.last_accessed_ms = time.monotonic() * 1000.0
            entry.access_count += 1

            # LRU: 移到队列末尾（先移除旧位置，避免重复项）
            if ckey in self._lru_order:
                self._lru_order.remove(ckey)
            self._lru_order.append(ckey)

            return entry.value

    def put(self, key: Any, value: Any, size_bytes: int | None = None) -> None:
        """写入 cache（可能触发 eviction）。

        Args:
            key: Cache key（完整 ``CSECacheKey``）
            value: Cached value
            size_bytes: Value 内存占用（字节）。未知/非正数一律拒绝
                （CacheCapacityExceeded）—— 绝不静默无限缓存。

        Raises:
            CacheCapacityExceeded: 若 size 未知/非正，或全 pinned 仍超上限。
        """
        with self._lock:
            ckey = self._normalize_key(key)
            now_ms = time.monotonic() * 1000.0

            # size_bytes 未知 → fail-closed，不是 0、不是无限。
            if size_bytes is None or size_bytes <= 0:
                raise CacheCapacityExceeded(
                    f"CSE cache put rejected: size_bytes must be a positive int "
                    f"(got {size_bytes!r}). Unknown sizes must not silently bypass "
                    f"the memory cap."
                )

            # 已存在：更新 —— 先扣除旧 size，并从 _lru_order 移除旧位置，避免重复。
            if ckey in self._cache:
                old_entry = self._cache[ckey]
                self._metrics.total_size_bytes -= old_entry.size_bytes
                if ckey in self._lru_order:
                    self._lru_order.remove(ckey)

            # 新建 entry
            entry = CacheEntry(
                key=ckey,
                value=value,
                size_bytes=size_bytes,
                created_at_ms=now_ms,
                last_accessed_ms=now_ms,
                access_count=0,
                pin_count=0,
            )
            self._cache[ckey] = entry
            self._lru_order.append(ckey)
            self._metrics.total_size_bytes += size_bytes

            # Eviction（内存超限）—— 无法通过逐出已有 unpinned 条目腾出空间，
            # 就必须 fail-closed，绝不静默 break 突破硬上限。刚插入的 ``ckey`` 不
            # 作 evict 候选：把一个放不下的条目立刻逐出来"满足"容量是静默丢弃，
            # 不是治理。
            while self._metrics.total_size_bytes > self.max_cache_bytes:
                evicted = self._evict_one(protected=ckey)
                if not evicted:
                    raise CacheCapacityExceeded(
                        f"CSE cache capacity exceeded: current={self._metrics.total_size_bytes}"
                        f" bytes, max={self.max_cache_bytes} bytes, and no evictable "
                        f"(unpinned, pre-existing) entry available. All existing entries "
                        f"may be pinned, or the new entry alone exceeds the cap. "
                        f"Apply backpressure / spill."
                    )

    def pin(self, key: Any) -> None:
        """Pin cache entry（消费期内不可 evict）。

        Args:
            key: Cache key
        """
        with self._lock:
            ckey = self._normalize_key(key)
            entry = self._cache.get(ckey)
            if entry is not None:
                entry.pin_count += 1
                self._metrics.pin_operations += 1

    def unpin(self, key: Any) -> None:
        """Unpin cache entry（引用计数 -1）。

        Args:
            key: Cache key
        """
        with self._lock:
            ckey = self._normalize_key(key)
            entry = self._cache.get(ckey)
            if entry is not None:
                entry.pin_count = max(0, entry.pin_count - 1)
                self._metrics.unpin_operations += 1

    def _evict_one(self, protected: Any = None) -> bool:
        """Evict 一个 entry（按 eviction policy 选择）。

        Args:
            protected: 本次不参与逐出的 key（调用方刚写入的 entry）。

        Returns:
            True 表示成功 evict，False 表示无可 evict 项（全部 pinned）。
        """
        if self.eviction_policy == "lru":
            return self._evict_lru(protected=protected)
        if self.eviction_policy == "lfu":
            return self._evict_lfu(protected=protected)
        # 构造期已拦截未知策略；若被绕过，仍 fail-closed。
        raise UnsupportedEvictionPolicy(
            f"eviction_policy={self.eviction_policy!r} is not implemented. "
            f"Refusing to silently fall back to LRU."
        )

    def _evict_lru(self, protected: Any = None) -> bool:
        """LRU eviction（最久未访问优先）。"""
        for key in self._lru_order:
            if key == protected:
                continue
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
            _logger.debug("evicted cache entry: %s (LRU, size=%d)", key.to_key(), entry.size_bytes)
            return True

        return False

    def _evict_lfu(self, protected: Any = None) -> bool:
        """LFU eviction（最少访问优先）。"""
        # 找出访问次数最少的 unpinned entry（排除本次刚写入的 protected）
        candidates = [
            (k, e) for k, e in self._cache.items()
            if not e.is_pinned() and k != protected
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
        _logger.debug("evicted cache entry: %s (LFU, access=%d)", key.to_key(), entry.access_count)
        return True

    def evict_by_memory_pressure(self, target_bytes: int) -> int:
        """根据内存压力主动 evict（释放至目标大小）。

        Args:
            target_bytes: 目标 cache 大小（字节）

        Returns:
            Evicted entry 数量

        Raises:
            CacheCapacityExceeded: 无法释放至 target（全部 pinned / 无可 evict）。
        """
        evicted_count = 0

        with self._lock:
            while self._metrics.total_size_bytes > target_bytes:
                if not self._evict_one():
                    raise CacheCapacityExceeded(
                        f"CSE cache memory pressure eviction failed: current="
                        f"{self._metrics.total_size_bytes} bytes, target={target_bytes} "
                        f"bytes, but no evictable (unpinned) entry remains. All entries "
                        f"may be pinned — apply backpressure / spill."
                    )
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
    """返回全局 CSECacheOptimizer（进程级单例）。

    P3/P4: 从 ResourceBroker（单权威）的 ``current_cse_budget()`` 派生 cache 预算，
    不再固定 4GiB；broker 无法给出真实预算时回退绝对上限。
    """
    global _global_cse_cache
    if _global_cse_cache is None:
        with _global_lock:
            if _global_cse_cache is None:
                max_cache = None
                try:
                    from factor_engine.runtime.resource_broker import ResourceBroker

                    broker = ResourceBroker()
                    max_cache = broker.current_cse_budget()
                except Exception:
                    max_cache = None
                if not max_cache:
                    max_cache = 4 * 1024**3
                _global_cse_cache = CSECacheOptimizer(max_cache_bytes=max_cache)
    return _global_cse_cache
