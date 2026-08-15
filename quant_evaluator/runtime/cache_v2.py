"""
Advanced multi-level cache system with compression, warming, and distributed support.

Provides L1 (memory) + L2 (disk) + optional L3 (Redis) caching with:
- Automatic compression for memory reduction (50-80% target)
- Cache warming strategies
- Automatic invalidation based on TTL and dependencies
- Distributed cache support via Redis (optional)
- Thread-safe operations with fine-grained locking
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from quant_evaluator.contracts.errors import DurableCacheCapabilityError

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

logger = logging.getLogger(__name__)


class CacheConfigurationError(DurableCacheCapabilityError, ValueError):
    """Raised when cache layers violate runtime policy or are unavailable."""


# ============================================================================
# Compression utilities
# ============================================================================

@dataclass
class CompressionStats:
    """Statistics for compression operations."""
    original_bytes: int
    compressed_bytes: int
    compression_time_ms: float
    decompression_time_ms: float = 0.0

    @property
    def ratio(self) -> float:
        """Compression ratio (original / compressed)."""
        return self.original_bytes / max(1, self.compressed_bytes)

    @property
    def savings_pct(self) -> float:
        """Space savings percentage."""
        return (1.0 - self.compressed_bytes / max(1, self.original_bytes)) * 100


class Compressor(ABC):
    """Abstract compressor interface."""

    @abstractmethod
    def compress(self, data: bytes) -> Tuple[bytes, CompressionStats]:
        """Compress data and return compressed bytes + stats."""
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass


class ZlibCompressor(Compressor):
    """Zlib compression (good balance of speed and ratio)."""

    def __init__(self, level: int = 6):
        self.level = level

    def compress(self, data: bytes) -> Tuple[bytes, CompressionStats]:
        start = time.perf_counter()
        compressed = zlib.compress(data, level=self.level)
        elapsed_ms = (time.perf_counter() - start) * 1000

        stats = CompressionStats(
            original_bytes=len(data),
            compressed_bytes=len(compressed),
            compression_time_ms=elapsed_ms,
        )
        return compressed, stats

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class LZ4Compressor(Compressor):
    """LZ4 compression (faster, slightly lower ratio)."""

    def __init__(self):
        if not HAS_LZ4:
            raise ImportError("lz4 not available")

    def compress(self, data: bytes) -> Tuple[bytes, CompressionStats]:
        start = time.perf_counter()
        compressed = lz4.frame.compress(data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        stats = CompressionStats(
            original_bytes=len(data),
            compressed_bytes=len(compressed),
            compression_time_ms=elapsed_ms,
        )
        return compressed, stats

    def decompress(self, data: bytes) -> bytes:
        return lz4.frame.decompress(data)


class NoCompressor(Compressor):
    """No-op compressor for testing or when compression is disabled."""

    def compress(self, data: bytes) -> Tuple[bytes, CompressionStats]:
        stats = CompressionStats(
            original_bytes=len(data),
            compressed_bytes=len(data),
            compression_time_ms=0.0,
        )
        return data, stats

    def decompress(self, data: bytes) -> bytes:
        return data


def create_compressor(method: str = "auto", level: int = 6) -> Compressor:
    """
    Create compressor instance.

    Args:
        method: "lz4", "zlib", "none", or "auto" (lz4 if available, else zlib)
        level: Compression level for zlib (1-9)

    Returns:
        Compressor instance
    """
    if method == "auto":
        method = "lz4" if HAS_LZ4 else "zlib"

    if method == "lz4":
        return LZ4Compressor()
    elif method == "zlib":
        return ZlibCompressor(level=level)
    elif method == "none":
        return NoCompressor()
    else:
        raise ValueError(f"Unknown compression method: {method}")


# ============================================================================
# Cache entry and metadata
# ============================================================================

@dataclass
class CacheMetadata:
    """Metadata for cache entries."""
    key: str
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    compressed_size: int
    ttl_seconds: Optional[float]
    dependencies: Set[str]
    version: str = "v2"
    compression_method: str = "none"

    def is_expired(self, now: float) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (now - self.created_at) > self.ttl_seconds

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "compressed_size": self.compressed_size,
            "ttl_seconds": self.ttl_seconds,
            "dependencies": list(self.dependencies),
            "version": self.version,
            "compression_method": self.compression_method,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> CacheMetadata:
        """Deserialize from dictionary."""
        return cls(
            key=data["key"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            access_count=data["access_count"],
            size_bytes=data["size_bytes"],
            compressed_size=data["compressed_size"],
            ttl_seconds=data.get("ttl_seconds"),
            dependencies=set(data.get("dependencies", [])),
            version=data.get("version", "v2"),
            compression_method=data.get("compression_method", "none"),
        )


@dataclass
class CacheEntry:
    """Cache entry with compressed value and metadata."""
    metadata: CacheMetadata
    compressed_value: bytes

    def mark_accessed(self):
        """Update access statistics."""
        self.metadata.last_accessed = time.time()
        self.metadata.access_count += 1


# ============================================================================
# L1: Memory cache layer
# ============================================================================

class MemoryCacheLayer:
    """L1 memory cache with compression and LRU eviction."""

    def __init__(
        self,
        max_size_bytes: int,
        compressor: Compressor,
        enable_compression: bool = True,
    ):
        """
        Initialize memory cache layer.

        Args:
            max_size_bytes: Maximum memory usage in bytes
            compressor: Compressor instance
            enable_compression: Whether to compress values in memory
        """
        self.max_size_bytes = max_size_bytes
        self.compressor = compressor
        self.enable_compression = enable_compression
        self._cache: Dict[str, CacheEntry] = {}
        self._current_size = 0
        self._lock = threading.RLock()
        self._compression_stats: List[CompressionStats] = []

    def get(self, key: str) -> Optional[Tuple[Any, CacheMetadata]]:
        """Get value from memory cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            # Check expiration
            if entry.metadata.is_expired(time.time()):
                self._remove_entry(key)
                return None

            # Decompress if needed
            try:
                if entry.metadata.compression_method != "none":
                    data = self.compressor.decompress(entry.compressed_value)
                else:
                    data = entry.compressed_value
                value = pickle.loads(data)
            except Exception as e:
                logger.warning(f"Failed to deserialize cache entry {key}: {e}")
                self._remove_entry(key)
                return None

            # Update access stats and move to end (LRU)
            entry.mark_accessed()
            self._cache.pop(key)
            self._cache[key] = entry

            return value, entry.metadata

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        dependencies: Optional[Set[str]] = None,
        created_at: Optional[float] = None,
    ) -> bool:
        """
        Put value into memory cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            dependencies: Set of dependency keys

        Returns:
            True if stored successfully
        """
        with self._lock:
            try:
                # Serialize
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                original_size = len(data)

                # Compress if enabled
                if self.enable_compression:
                    compressed, stats = self.compressor.compress(data)
                    self._compression_stats.append(stats)
                    compression_method = "lz4" if HAS_LZ4 else "zlib"
                else:
                    compressed = data
                    compression_method = "none"

                compressed_size = len(compressed)

                # Check if value is too large
                if compressed_size > self.max_size_bytes:
                    logger.debug(f"Value too large for cache: {compressed_size} bytes")
                    return False

                # Preserve the original creation time when promoting between layers so
                # an entry's absolute expiry is never extended by a cache hit.
                now = time.time()
                entry_created_at = now if created_at is None else created_at
                metadata = CacheMetadata(
                    key=key,
                    created_at=entry_created_at,
                    last_accessed=now,
                    access_count=0,
                    size_bytes=original_size,
                    compressed_size=compressed_size,
                    ttl_seconds=ttl_seconds,
                    dependencies=dependencies or set(),
                    compression_method=compression_method,
                )

                # Evict if necessary
                while self._current_size + compressed_size > self.max_size_bytes:
                    if not self._evict_lru():
                        return False

                # Remove old entry if exists
                if key in self._cache:
                    old_entry = self._cache[key]
                    self._current_size -= old_entry.metadata.compressed_size

                # Store entry
                entry = CacheEntry(metadata=metadata, compressed_value=compressed)
                self._cache[key] = entry
                self._current_size += compressed_size

                return True

            except Exception as e:
                logger.error(f"Failed to cache value for key {key}: {e}")
                return False

    def invalidate(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            return self._remove_entry(key)

    def invalidate_dependencies(self, dependency: str) -> int:
        """Invalidate all entries that depend on the given dependency."""
        with self._lock:
            keys_to_remove = [
                k for k, e in self._cache.items()
                if dependency in e.metadata.dependencies
            ]
            for key in keys_to_remove:
                self._remove_entry(key)
            return len(keys_to_remove)

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    def _remove_entry(self, key: str) -> bool:
        """Remove entry and update size tracking."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size -= entry.metadata.compressed_size
            return True
        return False

    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False

        # Find LRU entry (first in insertion order)
        lru_key = next(iter(self._cache))
        self._remove_entry(lru_key)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_original = sum(e.metadata.size_bytes for e in self._cache.values())
            total_compressed = sum(
                e.metadata.compressed_size for e in self._cache.values()
            )
            avg_compression_ratio = 1.0
            if self._compression_stats:
                avg_compression_ratio = sum(
                    s.ratio for s in self._compression_stats[-1000:]
                ) / min(1000, len(self._compression_stats))

            return {
                "num_entries": len(self._cache),
                "current_size_bytes": self._current_size,
                "max_size_bytes": self.max_size_bytes,
                "utilization": self._current_size / max(1, self.max_size_bytes),
                "total_original_bytes": total_original,
                "total_compressed_bytes": total_compressed,
                "memory_savings_pct": (
                    (1.0 - total_compressed / max(1, total_original)) * 100
                ),
                "avg_compression_ratio": avg_compression_ratio,
            }


# ============================================================================
# L2: Disk cache layer
# ============================================================================

class DiskCacheLayer:
    """L2 disk cache with atomic writes and checksum verification."""

    def __init__(self, root_dir: Path, compressor: Compressor):
        """
        Initialize disk cache layer.

        Args:
            root_dir: Root directory for cache files
            compressor: Compressor instance
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.compressor = compressor
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _get_lock(self, key: str) -> threading.Lock:
        """Get per-key lock for atomic writes."""
        with self._locks_guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _key_path(self, key: str) -> Path:
        """Get file path for key."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root_dir / digest[:2] / f"{digest}.cache"

    def _meta_path(self, cache_path: Path) -> Path:
        """Get metadata path for cache file."""
        return cache_path.with_suffix(".meta.json")

    def get(self, key: str) -> Optional[Tuple[Any, CacheMetadata]]:
        """Get value from disk cache."""
        cache_path = self._key_path(key)
        meta_path = self._meta_path(cache_path)

        if not cache_path.exists() or not meta_path.exists():
            return None

        try:
            # Load metadata
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata = CacheMetadata.from_dict(meta_data)

            # Check expiration
            if metadata.is_expired(time.time()):
                self.invalidate(key)
                return None

            # Load and verify compressed data
            compressed_data = cache_path.read_bytes()
            checksum = hashlib.sha256(compressed_data).hexdigest()
            expected_checksum = meta_data.get("checksum")

            if expected_checksum and checksum != expected_checksum:
                logger.warning(f"Checksum mismatch for cache key {key}")
                self.invalidate(key)
                return None

            # Decompress and deserialize
            if metadata.compression_method != "none":
                data = self.compressor.decompress(compressed_data)
            else:
                data = compressed_data

            value = pickle.loads(data)
            return value, metadata

        except Exception as e:
            logger.warning(f"Failed to load cache from disk for key {key}: {e}")
            return None

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        dependencies: Optional[Set[str]] = None,
        created_at: Optional[float] = None,
    ) -> bool:
        """
        Put value into disk cache with atomic write.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            dependencies: Set of dependency keys

        Returns:
            True if stored successfully
        """
        lock = self._get_lock(key)
        with lock:
            try:
                # Serialize and compress
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                compressed, stats = self.compressor.compress(data)

                # Preserve the original creation time when promoting between layers so
                # an entry's absolute expiry is never extended by a cache hit.
                now = time.time()
                entry_created_at = now if created_at is None else created_at
                metadata = CacheMetadata(
                    key=key,
                    created_at=entry_created_at,
                    last_accessed=now,
                    access_count=0,
                    size_bytes=len(data),
                    compressed_size=len(compressed),
                    ttl_seconds=ttl_seconds,
                    dependencies=dependencies or set(),
                    compression_method="lz4" if HAS_LZ4 else "zlib",
                )

                # Compute checksum
                checksum = hashlib.sha256(compressed).hexdigest()

                # Write atomically using temp file
                cache_path = self._key_path(key)
                meta_path = self._meta_path(cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                import tempfile
                import uuid

                tmp_cache = cache_path.parent / f".tmp.{uuid.uuid4().hex}.cache"
                tmp_meta = cache_path.parent / f".tmp.{uuid.uuid4().hex}.meta.json"

                try:
                    # Open follow-ups intentionally outside this safety repair:
                    # values still use pickle, and the data/metadata pair can be
                    # observed between its two atomic renames (torn publish).
                    # Write temp files
                    tmp_cache.write_bytes(compressed)

                    meta_dict = metadata.to_dict()
                    meta_dict["checksum"] = checksum
                    tmp_meta.write_text(json.dumps(meta_dict), encoding="utf-8")

                    # Atomic rename
                    tmp_cache.replace(cache_path)
                    tmp_meta.replace(meta_path)

                    return True

                except Exception as e:
                    # Clean up temp files
                    tmp_cache.unlink(missing_ok=True)
                    tmp_meta.unlink(missing_ok=True)
                    raise

            except Exception as e:
                logger.error(f"Failed to write cache to disk for key {key}: {e}")
                return False

    def invalidate(self, key: str) -> bool:
        """Remove entry from disk cache."""
        cache_path = self._key_path(key)
        meta_path = self._meta_path(cache_path)

        removed = False
        if cache_path.exists():
            cache_path.unlink()
            removed = True
        if meta_path.exists():
            meta_path.unlink()
            removed = True

        return removed

    def invalidate_dependencies(self, dependency: str) -> int:
        """Invalidate disk entries bound to the given dependency."""
        keys_to_remove = []
        for meta_path in self.root_dir.rglob("*.meta.json"):
            try:
                meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                metadata = CacheMetadata.from_dict(meta_data)
                if dependency in metadata.dependencies:
                    keys_to_remove.append(metadata.key)
            except Exception as e:
                logger.warning(f"Failed to inspect disk cache metadata {meta_path}: {e}")

        return sum(1 for key in keys_to_remove if self.invalidate(key))

    def clear(self):
        """Clear all cache files."""
        import shutil
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir)
            self.root_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# L3: Distributed cache layer (Redis)
# ============================================================================

class RedisCacheLayer:
    """L3 distributed cache using Redis."""

    def __init__(
        self,
        redis_url: str,
        compressor: Compressor,
        key_prefix: str = "cache:",
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache layer.

        Args:
            redis_url: Redis connection URL
            compressor: Compressor instance
            key_prefix: Prefix for all keys
            default_ttl: Default TTL in seconds
        """
        if not HAS_REDIS:
            raise ImportError("redis-py not available")

        self.client = redis.from_url(redis_url, decode_responses=False)
        self.compressor = compressor
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

    def _prefixed_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Tuple[Any, CacheMetadata]]:
        """Get value from Redis cache."""
        prefixed = self._prefixed_key(key)

        try:
            # Get value and metadata
            pipe = self.client.pipeline()
            pipe.get(prefixed)
            pipe.hgetall(f"{prefixed}:meta")
            results = pipe.execute()

            compressed_data = results[0]
            meta_dict = results[1]

            if not compressed_data or not meta_dict:
                return None

            # Decode metadata
            meta_decoded = {
                k.decode("utf-8"): v.decode("utf-8")
                for k, v in meta_dict.items()
            }
            metadata = CacheMetadata.from_dict(json.loads(meta_decoded["data"]))

            # Check expiration
            if metadata.is_expired(time.time()):
                self.invalidate(key)
                return None

            # Decompress and deserialize
            if metadata.compression_method != "none":
                data = self.compressor.decompress(compressed_data)
            else:
                data = compressed_data

            value = pickle.loads(data)
            return value, metadata

        except Exception as e:
            logger.warning(f"Failed to get from Redis cache for key {key}: {e}")
            return None

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        dependencies: Optional[Set[str]] = None,
        created_at: Optional[float] = None,
    ) -> bool:
        """
        Put value into Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            dependencies: Set of dependency keys

        Returns:
            True if stored successfully
        """
        prefixed = self._prefixed_key(key)
        ttl = int(ttl_seconds) if ttl_seconds else self.default_ttl

        try:
            # Serialize and compress
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            compressed, stats = self.compressor.compress(data)

            # Preserve the original creation time when writing a promoted entry.
            now = time.time()
            entry_created_at = now if created_at is None else created_at
            metadata = CacheMetadata(
                key=key,
                created_at=entry_created_at,
                last_accessed=now,
                access_count=0,
                size_bytes=len(data),
                compressed_size=len(compressed),
                ttl_seconds=ttl_seconds,
                dependencies=dependencies or set(),
                compression_method="lz4" if HAS_LZ4 else "zlib",
            )

            # Store with pipeline
            pipe = self.client.pipeline()
            pipe.setex(prefixed, ttl, compressed)
            pipe.hset(
                f"{prefixed}:meta",
                mapping={"data": json.dumps(metadata.to_dict())}
            )
            pipe.expire(f"{prefixed}:meta", ttl)
            pipe.execute()

            return True

        except Exception as e:
            logger.error(f"Failed to write to Redis cache for key {key}: {e}")
            return False

    def invalidate(self, key: str) -> bool:
        """Remove entry from Redis cache."""
        prefixed = self._prefixed_key(key)
        try:
            pipe = self.client.pipeline()
            pipe.delete(prefixed)
            pipe.delete(f"{prefixed}:meta")
            results = pipe.execute()
            return any(results)
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache for key {key}: {e}")
            return False

    def invalidate_dependencies(self, dependency: str) -> int:
        """Invalidate Redis entries bound to the given dependency."""
        try:
            cursor = 0
            keys_to_remove = []
            pattern = f"{self.key_prefix}*:meta"
            while True:
                cursor, meta_keys = self.client.scan(cursor, match=pattern, count=100)
                for meta_key in meta_keys:
                    try:
                        meta_dict = self.client.hgetall(meta_key)
                        raw_data = meta_dict.get(b"data") or meta_dict.get("data")
                        if not raw_data:
                            continue
                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode("utf-8")
                        metadata = CacheMetadata.from_dict(json.loads(raw_data))
                        if dependency in metadata.dependencies:
                            keys_to_remove.append(metadata.key)
                    except Exception as e:
                        # One malformed record must not prevent later dependency-bound
                        # entries from being invalidated. Quarantine the unusable pair
                        # so it cannot remain as a promotion candidate.
                        value_key = (
                            meta_key[:-5]
                            if meta_key.endswith(b":meta")
                            else meta_key[:-5]
                            if meta_key.endswith(":meta")
                            else None
                        )
                        logger.warning(
                            f"Failed to inspect Redis cache metadata {meta_key!r}: {e}"
                        )
                        if value_key is not None:
                            self.client.delete(meta_key, value_key)
                        else:
                            self.client.delete(meta_key)
                if cursor == 0:
                    break
            return sum(1 for key in keys_to_remove if self.invalidate(key))
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis dependency {dependency}: {e}")
            return 0

    def clear(self):
        """Clear all cache entries with this prefix."""
        try:
            pattern = f"{self.key_prefix}*"
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")


# ============================================================================
# Multi-level cache coordinator
# ============================================================================

@dataclass(frozen=True)
class CacheV2Config:
    """Production-facing cache factory configuration.

    L2/L3 are advisory, trusted-environment accelerators in research/test modes;
    they are not authenticated durable stores. Production therefore supports L1
    only until the durable layers use a single authenticated atomic envelope.
    """

    runtime_mode: Literal["research", "test", "production"] = "research"
    memory_size_mb: float = 512.0
    disk_root: Optional[Path] = None
    redis_url: Optional[str] = None
    compression: str = "auto"
    compression_level: int = 6
    enable_l1: bool = True
    enable_l2: bool = False
    enable_l3: bool = False


def create_cache(config: CacheV2Config) -> "MultiLevelCache":
    """Create a cache from validated public configuration."""
    return MultiLevelCache(
        memory_size_mb=config.memory_size_mb,
        disk_root=config.disk_root,
        redis_url=config.redis_url,
        compression=config.compression,
        compression_level=config.compression_level,
        enable_l1=config.enable_l1,
        enable_l2=config.enable_l2,
        enable_l3=config.enable_l3,
        runtime_mode=config.runtime_mode,
    )


@dataclass
class CacheStats:
    """Statistics for multi-level cache."""
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0
    l1_stats: Dict[str, Any] = field(default_factory=dict)
    evictions: int = 0
    invalidations: int = 0


class MultiLevelCache:
    """
    Multi-level cache with L1 (memory) + L2 (disk) + optional L3 (Redis).

    Features:
    - Automatic compression (50-80% memory reduction)
    - TTL-based expiration
    - Dependency tracking and invalidation
    - Cache warming strategies
    - Thread-safe operations
    """

    def __init__(
        self,
        memory_size_mb: float = 512.0,
        disk_root: Optional[Path] = None,
        redis_url: Optional[str] = None,
        compression: str = "auto",
        compression_level: int = 6,
        enable_l1: bool = True,
        enable_l2: bool = False,
        enable_l3: bool = False,
        runtime_mode: Literal["research", "test", "production"] = "research",
        production_mode: Optional[bool] = None,
    ):
        """
        Initialize multi-level cache.

        Args:
            memory_size_mb: L1 memory cache size in MB
            disk_root: L2 disk cache root directory
            redis_url: L3 Redis connection URL
            compression: Compression method ("auto", "lz4", "zlib", "none")
            compression_level: Compression level for zlib
            enable_l1: Enable L1 memory cache
            enable_l2: Enable L2 disk cache. Research/test use must opt in.
            enable_l3: Enable L3 Redis cache. Research/test use must opt in.
            runtime_mode: Runtime trust policy (research, test, or production).
            production_mode: Deprecated boolean alias for runtime_mode.
        """
        if production_mode is not None:
            alias_mode = "production" if production_mode else "research"
            if runtime_mode != "research" and runtime_mode != alias_mode:
                raise CacheConfigurationError(
                    "runtime_mode conflicts with production_mode alias"
                )
            runtime_mode = alias_mode
        if runtime_mode not in {"research", "test", "production"}:
            raise CacheConfigurationError(f"Unknown cache runtime mode: {runtime_mode!r}")
        if runtime_mode == "production" and (enable_l2 or enable_l3):
            requested_layers = []
            if enable_l2:
                requested_layers.append("L2 disk")
            if enable_l3:
                requested_layers.append("L3 Redis")
            raise CacheConfigurationError(
                "Durable cache layers are disabled in production mode: "
                f"requested {', '.join(requested_layers)}. "
                "Use L1 only in production; L2/L3 are trusted, advisory "
                "research/test accelerators until authenticated atomic persistence exists."
            )

        self.runtime_mode = runtime_mode
        self.production_mode = runtime_mode == "production"
        self.compressor = create_compressor(compression, compression_level)
        self.enable_l1 = enable_l1
        self.enable_l2 = enable_l2
        self.enable_l3 = enable_l3

        # Initialize layers
        self.l1: Optional[MemoryCacheLayer] = None
        self.l2: Optional[DiskCacheLayer] = None
        self.l3: Optional[RedisCacheLayer] = None

        if enable_l1:
            memory_bytes = int(memory_size_mb * 1024 * 1024)
            self.l1 = MemoryCacheLayer(
                max_size_bytes=memory_bytes,
                compressor=self.compressor,
                enable_compression=True,
            )

        if enable_l2:
            if disk_root is None:
                raise CacheConfigurationError(
                    "L2 disk cache was requested but disk_root is unavailable"
                )
            self.l2 = DiskCacheLayer(
                root_dir=Path(disk_root),
                compressor=self.compressor,
            )

        if enable_l3:
            if redis_url is None:
                raise CacheConfigurationError(
                    "L3 Redis cache was requested but redis_url is unavailable"
                )
            if not HAS_REDIS:
                raise CacheConfigurationError(
                    "L3 Redis cache was requested but redis-py is unavailable"
                )
            self.l3 = RedisCacheLayer(
                redis_url=redis_url,
                compressor=self.compressor,
            )

        # Statistics
        self.stats = CacheStats()
        self._stats_lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (checks L1 → L2 → L3).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Try L1 (memory)
        if self.l1:
            result = self.l1.get(key)
            if result is not None:
                with self._stats_lock:
                    self.stats.l1_hits += 1
                return result[0]

        # Try L2 (disk)
        if self.l2:
            result = self.l2.get(key)
            if result is not None:
                value, metadata = result
                with self._stats_lock:
                    self.stats.l2_hits += 1

                # Promote to L1
                if self.l1:
                    self.l1.put(
                        key,
                        value,
                        ttl_seconds=metadata.ttl_seconds,
                        dependencies=metadata.dependencies,
                        created_at=metadata.created_at,
                    )

                return value

        # Try L3 (Redis)
        if self.l3:
            result = self.l3.get(key)
            if result is not None:
                value, metadata = result
                with self._stats_lock:
                    self.stats.l3_hits += 1

                # Promote to L2 and L1
                if self.l2:
                    self.l2.put(
                        key,
                        value,
                        ttl_seconds=metadata.ttl_seconds,
                        dependencies=metadata.dependencies,
                        created_at=metadata.created_at,
                    )
                if self.l1:
                    self.l1.put(
                        key,
                        value,
                        ttl_seconds=metadata.ttl_seconds,
                        dependencies=metadata.dependencies,
                        created_at=metadata.created_at,
                    )

                return value

        # Cache miss
        with self._stats_lock:
            self.stats.misses += 1
        return None

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        dependencies: Optional[Set[str]] = None,
        write_through: bool = True,
    ) -> bool:
        """
        Put value into cache (writes to all enabled layers).

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            dependencies: Set of dependency keys
            write_through: Write to all layers (True) or just L1 (False)

        Returns:
            True if stored in at least one layer
        """
        success = False

        # Write to L1
        if self.l1:
            if self.l1.put(key, value, ttl_seconds, dependencies):
                success = True

        if write_through:
            # Write to L2
            if self.l2:
                if self.l2.put(key, value, ttl_seconds, dependencies):
                    success = True

            # Write to L3
            if self.l3:
                if self.l3.put(key, value, ttl_seconds, dependencies):
                    success = True

        return success

    def invalidate(self, key: str) -> bool:
        """
        Invalidate key across all cache layers.

        Args:
            key: Cache key

        Returns:
            True if removed from at least one layer
        """
        removed = False

        if self.l1 and self.l1.invalidate(key):
            removed = True
        if self.l2 and self.l2.invalidate(key):
            removed = True
        if self.l3 and self.l3.invalidate(key):
            removed = True

        if removed:
            with self._stats_lock:
                self.stats.invalidations += 1

        return removed

    def invalidate_dependency(self, dependency: str) -> int:
        """
        Invalidate all entries that depend on the given dependency.

        Args:
            dependency: Dependency key

        Returns:
            Number of entries invalidated
        """
        count = 0

        if self.l1:
            count += self.l1.invalidate_dependencies(dependency)
        if self.l2:
            count += self.l2.invalidate_dependencies(dependency)
        if self.l3:
            count += self.l3.invalidate_dependencies(dependency)

        with self._stats_lock:
            self.stats.invalidations += count

        return count

    def clear(self):
        """Clear all cache layers."""
        if self.l1:
            self.l1.clear()
        if self.l2:
            self.l2.clear()
        if self.l3:
            self.l3.clear()

    def warm(
        self,
        keys: List[str],
        loader: Callable[[str], Any],
        ttl_seconds: Optional[float] = None,
        dependencies: Optional[Set[str]] = None,
    ):
        """
        Warm cache with pre-computed values.

        Args:
            keys: List of keys to warm
            loader: Function that loads value for a key
            ttl_seconds: TTL for cached values
            dependencies: Dependencies for cached values
        """
        for key in keys:
            if self.get(key) is None:
                try:
                    value = loader(key)
                    self.put(key, value, ttl_seconds, dependencies)
                except Exception as e:
                    logger.warning(f"Failed to warm cache for key {key}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._stats_lock:
            stats_dict = {
                "l1_hits": self.stats.l1_hits,
                "l2_hits": self.stats.l2_hits,
                "l3_hits": self.stats.l3_hits,
                "misses": self.stats.misses,
                "evictions": self.stats.evictions,
                "invalidations": self.stats.invalidations,
            }

            total_requests = (
                self.stats.l1_hits
                + self.stats.l2_hits
                + self.stats.l3_hits
                + self.stats.misses
            )
            if total_requests > 0:
                stats_dict["hit_rate"] = (
                    (self.stats.l1_hits + self.stats.l2_hits + self.stats.l3_hits)
                    / total_requests
                )
            else:
                stats_dict["hit_rate"] = 0.0

            if self.l1:
                stats_dict["l1"] = self.l1.get_stats()

            return stats_dict


# ============================================================================
# Cache warming strategies
# ============================================================================

class CacheWarmingStrategy(ABC):
    """Abstract cache warming strategy."""

    @abstractmethod
    def get_keys_to_warm(self) -> List[str]:
        """Get list of keys that should be warmed."""
        pass


class MostRecentKeysStrategy(CacheWarmingStrategy):
    """Warm cache with most recently used keys."""

    def __init__(self, history: List[str], top_n: int = 100):
        self.history = history
        self.top_n = top_n

    def get_keys_to_warm(self) -> List[str]:
        """Return most recent N unique keys."""
        seen = set()
        result = []
        for key in reversed(self.history):
            if key not in seen:
                result.append(key)
                seen.add(key)
                if len(result) >= self.top_n:
                    break
        return result


class PredictiveStrategy(CacheWarmingStrategy):
    """Warm cache based on access patterns and predictions."""

    def __init__(self, access_log: List[Tuple[str, float]], lookahead_seconds: float = 300):
        self.access_log = access_log
        self.lookahead_seconds = lookahead_seconds

    def get_keys_to_warm(self) -> List[str]:
        """Predict keys likely to be accessed soon."""
        # Simple implementation: keys accessed in recent time window
        if not self.access_log:
            return []

        now = time.time()
        recent_keys = [
            key for key, timestamp in self.access_log
            if now - timestamp < self.lookahead_seconds
        ]

        # Return unique keys, most recent first
        seen = set()
        result = []
        for key in reversed(recent_keys):
            if key not in seen:
                result.append(key)
                seen.add(key)

        return result
