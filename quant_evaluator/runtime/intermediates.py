"""
Intermediate result caching for metric evaluation.

Stores intermediate computations to avoid redundant work when multiple
metrics depend on the same base computations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import hashlib
import pickle
import time


@dataclass(frozen=True)
class CacheKey:
    """
    Key for identifying cached intermediate results.

    Combines metric ID, chunk ID, and input hash to uniquely identify
    a computation.
    """
    metric_id: str
    chunk_id: Optional[int] = None
    input_hash: Optional[str] = None
    version: str = "v1"

    def __hash__(self):
        return hash((self.metric_id, self.chunk_id, self.input_hash, self.version))

    def __str__(self):
        parts = [self.metric_id]
        if self.chunk_id is not None:
            parts.append(f"chunk_{self.chunk_id}")
        if self.input_hash:
            parts.append(self.input_hash[:8])
        return "_".join(parts)


@dataclass
class CacheEntry:
    """
    Entry in the intermediate cache.

    Stores the result along with metadata about creation time and size.
    """
    key: CacheKey
    value: Any
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.size_bytes == 0:
            try:
                self.size_bytes = len(pickle.dumps(self.value))
            except Exception:
                self.size_bytes = -1  # Unknown size

    def mark_accessed(self):
        """Mark this entry as accessed (updates counters)."""
        object.__setattr__(self, "access_count", self.access_count + 1)
        object.__setattr__(self, "last_accessed", time.time())


class IntermediateCache:
    """
    Cache for intermediate metric computation results.

    Implements size-based eviction (LRU) when cache exceeds max size.
    """

    def __init__(self, max_size_mb: float = 1024.0, enable: bool = True):
        """
        Initialize cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
            enable: Whether caching is enabled
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.enable = enable
        self._cache: Dict[CacheKey, CacheEntry] = {}
        self._current_size = 0

    def get(self, key: CacheKey) -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enable:
            return None

        entry = self._cache.get(key)
        if entry is not None:
            entry.mark_accessed()
            return entry.value

        return None

    def put(self, key: CacheKey, value: Any) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to store

        Returns:
            True if stored, False if caching disabled or eviction failed
        """
        if not self.enable:
            return False

        # Create entry
        entry = CacheEntry(key=key, value=value)

        # Check if this would exceed cache size
        if entry.size_bytes > self.max_size_bytes:
            # Value too large for cache
            return False

        # Evict if necessary
        while self._current_size + entry.size_bytes > self.max_size_bytes:
            if not self._evict_lru():
                return False

        # Store entry
        if key in self._cache:
            # Update existing entry
            old_entry = self._cache[key]
            self._current_size -= old_entry.size_bytes

        self._cache[key] = entry
        self._current_size += entry.size_bytes

        return True

    def has(self, key: CacheKey) -> bool:
        """Check if key exists in cache."""
        return self.enable and key in self._cache

    def invalidate(self, key: CacheKey) -> bool:
        """
        Remove entry from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if entry was removed, False if not found
        """
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_size -= entry.size_bytes
            return True
        return False

    def invalidate_metric(self, metric_id: str) -> int:
        """
        Invalidate all entries for a metric.

        Args:
            metric_id: Metric ID to invalidate

        Returns:
            Number of entries removed
        """
        keys_to_remove = [k for k in self._cache if k.metric_id == metric_id]
        for key in keys_to_remove:
            self.invalidate(key)
        return len(keys_to_remove)

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._current_size = 0

    def _evict_lru(self) -> bool:
        """
        Evict least recently used entry.

        Returns:
            True if entry was evicted, False if cache is empty
        """
        if not self._cache:
            return False

        # Find LRU entry
        lru_key = min(self._cache, key=lambda k: self._cache[k].last_accessed)
        self.invalidate(lru_key)
        return True

    @property
    def size_mb(self) -> float:
        """Current cache size in megabytes."""
        return self._current_size / (1024 * 1024)

    @property
    def num_entries(self) -> int:
        """Number of entries in cache."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_accesses = sum(e.access_count for e in self._cache.values())
        avg_size = self._current_size / len(self._cache) if self._cache else 0

        return {
            "num_entries": self.num_entries,
            "size_mb": self.size_mb,
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": self._current_size / self.max_size_bytes if self.max_size_bytes > 0 else 0,
            "total_accesses": total_accesses,
            "avg_entry_size_kb": avg_size / 1024,
        }


def compute_input_hash(
    factor_ids: Tuple[str, ...],
    time_slice: Optional[Tuple[int, int]] = None,
    asset_slice: Optional[Tuple[int, int]] = None,
    **kwargs
) -> str:
    """
    Compute hash of input parameters for cache key.

    Args:
        factor_ids: Factor identifiers
        time_slice: Time range (start, end)
        asset_slice: Asset range (start, end)
        **kwargs: Additional parameters to hash

    Returns:
        Hex digest of input hash
    """
    hasher = hashlib.sha256()

    # Hash factor IDs
    for fid in sorted(factor_ids):
        hasher.update(fid.encode('utf-8'))

    # Hash slices
    if time_slice:
        hasher.update(str(time_slice).encode('utf-8'))
    if asset_slice:
        hasher.update(str(asset_slice).encode('utf-8'))

    # Hash additional kwargs
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        hasher.update(f"{key}={value}".encode('utf-8'))

    return hasher.hexdigest()
