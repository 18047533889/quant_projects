"""
cache_v2-backed implementation of the intermediate cache protocol.

Wraps :class:`quant_evaluator.runtime.cache_v2.MultiLevelCache` so metric
intermediates get TTL + dependency invalidation, compression, and the
production-grade layer coordination instead of the legacy size-only LRU.
"""

from typing import Any, Dict, Optional

from quant_evaluator.runtime.cache_v2 import (
    CacheV2Config,
    MultiLevelCache,
    create_cache,
)
from quant_evaluator.runtime.intermediates import CacheKey


class V2IntermediateCache:
    """Intermediate cache backed by the cache_v2 ``MultiLevelCache``.

    Implements the :class:`~quant_evaluator.runtime.cache_protocol.CacheProtocol`
    surface (``get``/``put``/``has``/``invalidate``/``invalidate_metric``/
    ``clear``/``get_stats``) over string cache_v2 keys.  The full ``CacheKey``
    identity (metric, chunk, input hash, version) is folded into the physical
    key so two metrics sharing an input hash can never collide.
    """

    def __init__(
        self,
        *,
        cache: Optional[MultiLevelCache] = None,
        config: Optional[CacheV2Config] = None,
        max_size_mb: float = 1024.0,
        enable: bool = True,
    ):
        """
        Initialize the cache.

        Args:
            cache: An existing MultiLevelCache to wrap (takes precedence).
            config: A CacheV2Config to build the wrapped cache from.
            max_size_mb: L1 memory budget when neither ``cache`` nor ``config``
                is supplied (matches the legacy constructor's knob).
            enable: Master switch; when False every operation is a no-op miss.
        """
        if cache is not None and config is not None:
            raise ValueError("pass either cache or config, not both")
        if cache is None:
            resolved = config or CacheV2Config(memory_size_mb=max_size_mb)
            cache = create_cache(resolved)
        self._cache = cache
        self.enable = enable

    @staticmethod
    def _physical_key(key: CacheKey) -> str:
        """Fold the full CacheKey identity into one cache_v2 string key."""
        return (
            f"intermediates:v1|metric={key.metric_id}"
            f"|chunk={key.chunk_id}"
            f"|hash={key.input_hash}"
            f"|ver={key.version}"
        )

    def get(self, key: CacheKey) -> Optional[Any]:
        """Return the cached value for ``key`` or None on miss."""
        if not self.enable:
            return None
        return self._cache.get(self._physical_key(key))

    def put(self, key: CacheKey, value: Any) -> bool:
        """Store ``value`` under ``key``; return True when stored."""
        if not self.enable:
            return False
        return self._cache.put(self._physical_key(key), value)

    def has(self, key: CacheKey) -> bool:
        """Return whether ``key`` is present (without promoting it)."""
        if not self.enable:
            return False
        return self._cache.get(self._physical_key(key)) is not None

    def invalidate(self, key: CacheKey) -> bool:
        """Remove one entry; return True when something was removed."""
        return self._cache.invalidate(self._physical_key(key))

    def invalidate_metric(self, metric_id: str) -> int:
        """Remove every entry for ``metric_id``; return the count removed."""
        removed = 0
        for key in self._metric_keys(metric_id):
            if self._cache.invalidate(key):
                removed += 1
        return removed

    def _metric_keys(self, metric_id: str) -> list:
        """Enumerate physical keys currently held for ``metric_id``.

        cache_v2 exposes no key-scan API, so the L1 entries are enumerated via
        the coordinator statistics' backing store.  This is best-effort: keys
        that already evicted from every layer simply count as removed.
        """
        l1 = getattr(self._cache, "l1", None)
        if l1 is None:
            return []
        prefix = f"intermediates:v1|metric={metric_id}|"
        with l1._lock:
            return [key for key in l1._cache if key.startswith(prefix)]

    @property
    def num_entries(self) -> int:
        """Number of entries currently held in L1."""
        l1 = getattr(self._cache, "l1", None)
        if l1 is None:
            return 0
        return l1.get_stats().get("num_entries", 0)

    @property
    def size_mb(self) -> float:
        """Current L1 size in megabytes."""
        l1 = getattr(self._cache, "l1", None)
        if l1 is None:
            return 0.0
        return l1.get_stats().get("current_size_bytes", 0) / (1024 * 1024)

    def clear(self) -> None:
        """Remove every entry."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics including ``num_entries``/``size_mb``."""
        stats: Dict[str, Any] = dict(self._cache.get_stats())
        l1_stats = stats.get("l1") or {}
        stats.setdefault(
            "num_entries",
            l1_stats.get("num_entries", 0),
        )
        stats.setdefault(
            "size_mb",
            l1_stats.get("current_size_bytes", 0) / (1024 * 1024),
        )
        return stats


__all__ = ["V2IntermediateCache"]
