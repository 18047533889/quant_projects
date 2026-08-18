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
import base64
import json
import logging
import math
import os
import pickle
import threading
import time
import uuid
import zlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Set, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no advisory flock
    fcntl = None

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

_DISK_LOCKS: Dict[str, threading.Lock] = {}
_DISK_ROOT_LOCKS: Dict[str, threading.RLock] = {}
_DISK_LOCKS_GUARD = threading.Lock()
_DISK_PROCESS_LOCK_STATE = threading.local()


def _is_valid_ttl(ttl_seconds: Optional[float]) -> bool:
    """Return whether a TTL is absent or a finite positive duration."""
    if ttl_seconds is None:
        return True
    if isinstance(ttl_seconds, bool):
        return False
    try:
        return math.isfinite(ttl_seconds) and ttl_seconds > 0
    except TypeError:
        return False


def _compression_method(compressor: Compressor) -> str:
    """Return the codec represented by a compressor instance."""
    if isinstance(compressor, NoCompressor):
        return "none"
    if isinstance(compressor, LZ4Compressor):
        return "lz4"
    if isinstance(compressor, ZlibCompressor):
        return "zlib"
    raise TypeError(f"unsupported compressor type: {type(compressor)!r}")


def _codec_compatible(metadata_method: str, compressor: Compressor) -> bool:
    """Reject persisted bytes whose declared codec differs from this layer."""
    try:
        return metadata_method == _compression_method(compressor)
    except (TypeError, ValueError):
        return False


def _valid_origin(created_at: Optional[float], now: float, ttl_seconds: Optional[float]) -> bool:
    """Reject unusable absolute origins before writing a cache entry."""
    if created_at is None:
        return True
    if isinstance(created_at, bool) or isinstance(ttl_seconds, bool):
        return False
    try:
        if not math.isfinite(created_at) or created_at > now:
            return False
        return ttl_seconds is None or now - created_at < ttl_seconds
    except (TypeError, ValueError):
        return False


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
    generation: str = ""

    def is_expired(self, now: float) -> bool:
        """Check if entry has expired based on TTL."""
        try:
            if not math.isfinite(self.created_at) or not math.isfinite(now):
                return True
        except TypeError:
            return True
        if self.created_at > now:
            return True
        if self.ttl_seconds is None:
            return False
        if not _is_valid_ttl(self.ttl_seconds):
            return True
        return (now - self.created_at) >= self.ttl_seconds

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
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> CacheMetadata:
        """Deserialize and validate a durable metadata record."""
        if not isinstance(data, dict):
            raise ValueError("cache metadata must be an object")
        required = {
            "key", "created_at", "last_accessed", "access_count",
            "size_bytes", "compressed_size", "ttl_seconds", "dependencies",
            "version", "compression_method",
        }
        if not required.issubset(data) or set(data) - required - {"checksum", "generation"}:
            raise ValueError("cache metadata schema mismatch")
        if not isinstance(data["key"], str):
            raise ValueError("cache metadata key must be a string")
        if data["version"] != "v2":
            raise ValueError("unsupported cache metadata version")
        if data["compression_method"] not in {"none", "zlib", "lz4"}:
            raise ValueError("unsupported cache compression method")
        for name in ("created_at", "last_accessed"):
            if isinstance(data[name], bool) or not isinstance(data[name], (int, float)):
                raise ValueError(f"invalid cache metadata {name}")
            if not math.isfinite(data[name]):
                raise ValueError(f"invalid cache metadata {name}")
        if data["last_accessed"] < data["created_at"]:
            raise ValueError("cache metadata last_accessed precedes created_at")
        for name in ("access_count", "size_bytes", "compressed_size"):
            if isinstance(data[name], bool) or not isinstance(data[name], int) or data[name] < 0:
                raise ValueError(f"invalid cache metadata {name}")
        ttl = data["ttl_seconds"]
        if not _is_valid_ttl(ttl):
            raise ValueError("invalid cache metadata ttl_seconds")
        dependencies = data["dependencies"]
        if not isinstance(dependencies, (list, tuple, set)) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise ValueError("invalid cache metadata dependencies")
        generation = data.get("generation", "")
        if not isinstance(generation, str):
            raise ValueError("invalid cache metadata generation")
        return cls(
            key=data["key"],
            created_at=float(data["created_at"]),
            last_accessed=float(data["last_accessed"]),
            access_count=data["access_count"],
            size_bytes=data["size_bytes"],
            compressed_size=data["compressed_size"],
            ttl_seconds=ttl,
            dependencies=set(dependencies),
            version=data["version"],
            compression_method=data["compression_method"],
            generation=data.get("generation", ""),
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
                if not _codec_compatible(
                    entry.metadata.compression_method, self.compressor
                ):
                    raise ValueError("cache codec does not match configured compressor")
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
            if not _is_valid_ttl(ttl_seconds):
                return False
            try:
                # Capture the origin before potentially slow serialization or
                # compression so direct writes cannot extend absolute TTLs.
                now = time.time()
                entry_created_at = now if created_at is None else created_at
                if not _valid_origin(entry_created_at, now, ttl_seconds):
                    return False

                # Serialize
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                original_size = len(data)

                # Compress if enabled
                if self.enable_compression:
                    compressed, stats = self.compressor.compress(data)
                    self._compression_stats.append(stats)
                    compression_method = _compression_method(self.compressor)
                else:
                    compressed = data
                    compression_method = "none"

                compressed_size = len(compressed)
                publication_now = time.time()
                if not _valid_origin(entry_created_at, publication_now, ttl_seconds):
                    return False

                # Check if value is too large
                if compressed_size > self.max_size_bytes:
                    logger.debug(f"Value too large for cache: {compressed_size} bytes")
                    return False

                metadata = CacheMetadata(
                    key=key,
                    created_at=entry_created_at,
                    last_accessed=publication_now,
                    access_count=0,
                    size_bytes=original_size,
                    compressed_size=compressed_size,
                    ttl_seconds=ttl_seconds,
                    dependencies=set(dependencies) if dependencies is not None else set(),
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

    def invalidate_dependencies(self, dependency: str, *, return_keys: bool = False):
        """Invalidate entries that depend on the given dependency.

        ``return_keys`` lets the coordinator count logical entries once when
        the same key is present in multiple physical layers.
        """
        with self._lock:
            keys_to_remove = [
                k for k, e in self._cache.items()
                if dependency in e.metadata.dependencies
            ]
            for key in keys_to_remove:
                self._remove_entry(key)
            return set(keys_to_remove) if return_keys else len(keys_to_remove)

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
        self._remove_abandoned_record_temps()

    def _remove_abandoned_record_temps(self) -> None:
        """Remove unpublished record files left by an abruptly terminated writer."""
        with self._process_lock(), self._root_lock():
            for temporary in self.root_dir.rglob(".tmp.*.cache"):
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove abandoned disk cache temporary %s: %s",
                        temporary,
                        exc,
                    )

    def _root_lock(self) -> threading.RLock:
        """Get the process-wide lock that fences whole-cache operations."""
        root = str(self.root_dir.resolve(strict=False))
        with _DISK_LOCKS_GUARD:
            if root not in _DISK_ROOT_LOCKS:
                _DISK_ROOT_LOCKS[root] = threading.RLock()
            return _DISK_ROOT_LOCKS[root]

    def _get_lock(self, key: str) -> threading.Lock:
        """Get a process-wide lock for one physical cache entry."""
        with _DISK_LOCKS_GUARD:
            if key not in _DISK_LOCKS:
                _DISK_LOCKS[key] = threading.Lock()
            return _DISK_LOCKS[key]

    def _entry_lock(self, cache_path: Path) -> threading.Lock:
        """Use the canonical physical path as the shared lock identity."""
        return self._get_lock(str(cache_path.resolve(strict=False)))

    @contextmanager
    def _process_lock(self) -> Iterator[None]:
        """Serialize cache operations across processes sharing this root.

        POSIX ``flock`` is process-scoped rather than safely recursive across
        separately opened file descriptions. Track ownership per thread/root so
        coordinator operations can call layer APIs under one outer disk fence.
        """
        if fcntl is None:
            yield
            return
        root = str(self.root_dir.resolve(strict=False))
        state = getattr(_DISK_PROCESS_LOCK_STATE, "roots", None)
        if state is None:
            state = {}
            _DISK_PROCESS_LOCK_STATE.roots = state
        held = state.get(root)
        if held is not None:
            held[1] += 1
            try:
                yield
            finally:
                held[1] -= 1
            return

        lock_path = self.root_dir / ".cache.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            state[root] = [handle, 1]
            try:
                yield
            finally:
                del state[root]
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _key_path(self, key: str) -> Path:
        """Get file path for key."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root_dir / digest[:2] / f"{digest}.cache"

    def _meta_path(self, cache_path: Path) -> Path:
        """Get metadata path for cache file."""
        return cache_path.with_suffix(".meta.json")

    @staticmethod
    def _decode_record(raw: bytes) -> Tuple[CacheMetadata, bytes]:
        """Decode one immutable JSON record containing metadata and payload."""
        envelope = json.loads(raw.decode("utf-8"))
        if not isinstance(envelope, dict) or envelope.get("schema") != "cache-record-v1":
            raise ValueError("cache record schema mismatch")
        metadata_dict = envelope.get("metadata")
        if not isinstance(metadata_dict, dict):
            raise ValueError("cache record metadata mismatch")
        key_digest = envelope.get("key_digest")
        if (
            not isinstance(key_digest, str)
            or hashlib.sha256(metadata_dict.get("key", "").encode("utf-8")).hexdigest()
            != key_digest
        ):
            raise ValueError("cache record key mismatch")
        compressed_data = base64.b64decode(envelope.get("compressed_data", ""), validate=True)
        checksum = envelope.get("checksum")
        if not isinstance(checksum, str) or hashlib.sha256(compressed_data).hexdigest() != checksum:
            raise ValueError("cache record checksum mismatch")
        return CacheMetadata.from_dict(metadata_dict), compressed_data

    @staticmethod
    def _encode_record(metadata: CacheMetadata, compressed_data: bytes) -> bytes:
        """Encode metadata and payload into one publishable immutable record."""
        return json.dumps(
            {
                "schema": "cache-record-v1",
                "metadata": metadata.to_dict(),
                "key_digest": hashlib.sha256(metadata.key.encode("utf-8")).hexdigest(),
                "compressed_data": base64.b64encode(compressed_data).decode("ascii"),
                "checksum": hashlib.sha256(compressed_data).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _invalidation_epoch_path(self) -> Path:
        """Return the durable namespace invalidation marker path."""
        return self.root_dir / ".invalidation.epoch"

    def _read_invalidation_epoch_unlocked(self) -> Optional[str]:
        """Read the current namespace invalidation epoch."""
        try:
            return self._invalidation_epoch_path().read_text(encoding="ascii")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            logger.warning("Failed to read disk invalidation epoch: %s", exc)
            return None

    def invalidation_epoch(self) -> Optional[str]:
        """Read the namespace epoch under the shared disk-root fence."""
        with self._process_lock(), self._root_lock():
            return self._read_invalidation_epoch_unlocked()

    def _bump_invalidation_epoch_unlocked(self) -> str:
        """Atomically publish a new namespace invalidation epoch."""
        epoch = uuid.uuid4().hex
        epoch_path = self._invalidation_epoch_path()
        temporary = self.root_dir / f".tmp.{uuid.uuid4().hex}.epoch"
        temporary.write_text(epoch, encoding="ascii")
        temporary.replace(epoch_path)
        return epoch

    def get(self, key: str) -> Optional[Tuple[Any, CacheMetadata]]:
        """Get value from disk cache."""
        cache_path = self._key_path(key)
        lock = self._entry_lock(cache_path)
        with self._process_lock(), self._root_lock(), lock:
            return self._get_unlocked(key)

    def _get_unlocked(self, key: str) -> Optional[Tuple[Any, CacheMetadata]]:
        """Read one disk record while holding its per-key lock."""
        cache_path = self._key_path(key)
        meta_path = self._meta_path(cache_path)

        if cache_path.exists():
            try:
                metadata, compressed_data = self._decode_record(cache_path.read_bytes())
                if metadata.key != key or not _codec_compatible(
                    metadata.compression_method, self.compressor
                ) or metadata.is_expired(time.time()):
                    cache_path.unlink(missing_ok=True)
                    return None
                if metadata.compression_method != "none":
                    data = self.compressor.decompress(compressed_data)
                else:
                    data = compressed_data
                return pickle.loads(data), metadata
            except Exception as e:
                logger.warning(f"Failed to load cache record for key {key}: {e}")
                cache_path.unlink(missing_ok=True)
                return None

        # Read legacy two-file entries only as a fail-closed compatibility path.
        if not meta_path.exists():
            return None
        try:
            metadata_text = meta_path.read_text(encoding="utf-8")
            compressed_data = cache_path.read_bytes()
            meta_data = json.loads(metadata_text)
            metadata = CacheMetadata.from_dict(meta_data)
            if metadata.is_expired(time.time()) or metadata.key != key or not _codec_compatible(
                metadata.compression_method, self.compressor
            ):
                self._invalidate_if_unchanged_unlocked(key, metadata_text, compressed_data)
                return None
            checksum = hashlib.sha256(compressed_data).hexdigest()
            if checksum != meta_data.get("checksum"):
                self._invalidate_if_unchanged_unlocked(key, metadata_text, compressed_data)
                return None
            data = self.compressor.decompress(compressed_data) if metadata.compression_method != "none" else compressed_data
            return pickle.loads(data), metadata
        except Exception as e:
            logger.warning(f"Failed to load legacy cache pair for key {key}: {e}")
            metadata_text = locals().get("metadata_text")
            compressed_data = locals().get("compressed_data")
            if metadata_text is not None:
                self._invalidate_if_unchanged_unlocked(key, metadata_text, compressed_data)
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
        cache_path = self._key_path(key)
        lock = self._entry_lock(cache_path)
        with self._process_lock(), self._root_lock(), lock:
            if not _is_valid_ttl(ttl_seconds):
                return False
            try:
                # Capture the origin before potentially slow serialization or
                # compression so direct writes cannot extend absolute TTLs.
                now = time.time()
                entry_created_at = now if created_at is None else created_at
                if not _valid_origin(entry_created_at, now, ttl_seconds):
                    return False

                # Serialize and compress
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                compressed, stats = self.compressor.compress(data)
                publication_now = time.time()
                if not _valid_origin(entry_created_at, publication_now, ttl_seconds):
                    return False

                metadata = CacheMetadata(
                    key=key,
                    created_at=entry_created_at,
                    last_accessed=publication_now,
                    access_count=0,
                    size_bytes=len(data),
                    compressed_size=len(compressed),
                    ttl_seconds=ttl_seconds,
                    dependencies=set(dependencies) if dependencies is not None else set(),
                    compression_method=_compression_method(self.compressor),
                    generation=uuid.uuid4().hex,
                )

                # Publish one immutable record so value and metadata cannot tear.
                record = self._encode_record(metadata, compressed)
                # Encoding can be slow too; do not publish a record whose absolute
                # TTL expired after the earlier preprocessing validation.
                if not _valid_origin(entry_created_at, time.time(), ttl_seconds):
                    return False
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_cache = cache_path.parent / f".tmp.{uuid.uuid4().hex}.cache"
                try:
                    with tmp_cache.open("wb") as handle:
                        handle.write(record)
                        handle.flush()
                        os.fsync(handle.fileno())
                    tmp_cache.replace(cache_path)
                    try:
                        directory_fd = os.open(cache_path.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    except OSError:
                        # Directory fsync is unavailable on some filesystems.
                        pass
                    return True
                except Exception:
                    tmp_cache.unlink(missing_ok=True)
                    raise

            except Exception as e:
                logger.error(f"Failed to write cache to disk for key {key}: {e}")
                return False

    def invalidate(self, key: str) -> bool:
        """Remove entry from disk cache."""
        cache_path = self._key_path(key)
        with self._process_lock(), self._root_lock(), self._entry_lock(cache_path):
            return self._invalidate_unlocked(key)

    def _invalidate_unlocked(self, key: str) -> bool:
        """Remove a disk entry while holding its per-key lock."""
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

    def _invalidate_if_unchanged_unlocked(
        self,
        key: str,
        expected_metadata: str,
        expected_value: Optional[bytes],
    ) -> bool:
        """Remove a pair only when both physical files match the read snapshot."""
        cache_path = self._key_path(key)
        meta_path = self._meta_path(cache_path)
        try:
            if meta_path.read_text(encoding="utf-8") != expected_metadata:
                return False
            if expected_value is None:
                # A missing value is a stable legacy-pair identity only while
                # the value file remains absent. Remove metadata alone so a
                # concurrently published value cannot be deleted.
                if cache_path.exists():
                    return False
                meta_path.unlink(missing_ok=True)
                return True
            if cache_path.read_bytes() != expected_value:
                return False
        except (FileNotFoundError, OSError):
            return False
        return self._invalidate_unlocked(key)

    def _invalidate_scanned_pair(
        self,
        meta_path: Path,
        value_path: Path,
        expected_text: str,
        *,
        expected_value: Optional[bytes],
        dependency: Optional[str] = None,
        generation: Optional[str] = None,
    ) -> bool:
        """Delete a scanned pair only if both files remain unchanged."""
        if expected_value is None:
            return False
        with self._root_lock(), self._entry_lock(value_path):
            try:
                current_text = meta_path.read_text(encoding="utf-8")
                current_value = value_path.read_bytes()
            except (FileNotFoundError, OSError):
                return False
            if current_text != expected_text or current_value != expected_value:
                return False
            if generation is not None:
                try:
                    current = CacheMetadata.from_dict(json.loads(current_text))
                except Exception:
                    return False
                if current.generation != generation:
                    return False
                if dependency not in current.dependencies:
                    return False
            meta_path.unlink(missing_ok=True)
            value_path.unlink(missing_ok=True)
            return True

    def invalidate_dependencies(self, dependency: str, *, return_keys: bool = False):
        removed_keys = set()
        removed_count = 0
        with self._process_lock(), self._root_lock():
            return self._invalidate_dependencies_unlocked(
                dependency, return_keys=return_keys
            )

    def _invalidate_dependencies_unlocked(
        self, dependency: str, *, return_keys: bool = False
    ):
        removed_keys = set()
        removed_count = 0
        for cache_path in self.root_dir.rglob("*.cache"):
            if cache_path.name.startswith(".tmp."):
                continue
            try:
                raw = cache_path.read_bytes()
                metadata, _ = self._decode_record(raw)
                if dependency not in metadata.dependencies:
                    continue
                with self._entry_lock(cache_path):
                    if cache_path.read_bytes() != raw:
                        continue
                    cache_path.unlink(missing_ok=True)
                    removed_count += 1
                    if cache_path == self._key_path(metadata.key):
                        removed_keys.add(metadata.key)
            except Exception as e:
                logger.warning(f"Failed to inspect disk cache record {cache_path}: {e}")
                try:
                    with self._entry_lock(cache_path):
                        if cache_path.read_bytes() == raw:
                            cache_path.unlink(missing_ok=True)
                except (FileNotFoundError, OSError, UnboundLocalError):
                    pass
        for meta_path in self.root_dir.rglob("*.meta.json"):
            value_path = meta_path.with_name(meta_path.name[:-len(".meta.json")] + ".cache")
            scanned_text = None
            scanned_value = None
            try:
                scanned_text = meta_path.read_text(encoding="utf-8")
                scanned_value = value_path.read_bytes()
                metadata = CacheMetadata.from_dict(json.loads(scanned_text))
                if dependency not in metadata.dependencies:
                    continue
                if self._invalidate_scanned_pair(meta_path, value_path, scanned_text,
                    expected_value=scanned_value, dependency=dependency,
                    generation=metadata.generation):
                    removed_count += 1
                    if value_path == self._key_path(metadata.key):
                        removed_keys.add(metadata.key)
            except Exception:
                if scanned_text is not None:
                    self._invalidate_scanned_pair(meta_path, value_path, scanned_text,
                        expected_value=scanned_value)

        return removed_keys if return_keys else removed_count

    def clear(self):
        """Clear all cache files while preserving the process-lock inode."""
        import shutil
        with self._process_lock(), self._root_lock():
            self.root_dir.mkdir(parents=True, exist_ok=True)
            for child in self.root_dir.iterdir():
                if child.name in {".cache.lock", ".invalidation.epoch"}:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)


# ============================================================================
# L3: Distributed cache layer (Redis)
# ============================================================================

class RedisCacheLayer:
    """L3 distributed cache using Redis."""

    _COMPARE_DELETE_SCRIPT = """
    local payload = redis.call('HGET', KEYS[1], 'data')
    if not payload then return 0 end
    local decoded = cjson.decode(payload)
    if (decoded.generation or '') ~= ARGV[1] then return 0 end
    return redis.call('DEL', KEYS[1], KEYS[2])
    """

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

    def _invalidation_epoch_key(self) -> str:
        """Return the shared Redis key used for L1 invalidation epochs."""
        return f"{self.key_prefix}:invalidation_epoch"

    def invalidation_epoch(self) -> Optional[str]:
        """Read the shared Redis invalidation epoch, or None on failure."""
        try:
            value = self.client.get(self._invalidation_epoch_key())
        except Exception as exc:
            logger.warning("Failed to read Redis invalidation epoch: %s", exc)
            return None
        if isinstance(value, bytes):
            return value.decode("ascii", errors="replace")
        return str(value or "")

    def bump_invalidation_epoch(self) -> Optional[str]:
        """Publish a new shared Redis invalidation epoch, or None on failure."""
        epoch = uuid.uuid4().hex
        try:
            self.client.set(self._invalidation_epoch_key(), epoch)
        except Exception as exc:
            logger.warning("Failed to publish Redis invalidation epoch: %s", exc)
            return None
        return epoch

    def _logical_key_matches_physical(self, logical_key: str, value_key: Any) -> bool:
        """Return whether metadata identity matches the scanned Redis value key."""
        expected = self._prefixed_key(logical_key)
        if isinstance(value_key, bytes):
            try:
                value_key = value_key.decode("utf-8")
            except UnicodeDecodeError:
                return False
        return value_key == expected

    @staticmethod
    def _value_key_for_meta(meta_key: Any) -> Optional[Any]:
        """Return a metadata key's value key without mixing bytes and text."""
        suffix = b":meta" if isinstance(meta_key, bytes) else ":meta"
        return meta_key[:-len(suffix)] if meta_key.endswith(suffix) else None

    def _invalidate_physical(self, value_key: Any, meta_key: Any) -> bool:
        """Delete one Redis value/metadata pair using physical keys."""
        try:
            results = self.client.delete(value_key, meta_key)
            return bool(results)
        except Exception as e:
            logger.warning("Failed to quarantine Redis cache pair %r/%r: %s", value_key, meta_key, e)
            return False

    _COMPARE_QUARANTINE_SCRIPT = """
    local payload = redis.call('HGET', KEYS[1], 'data')
    if ARGV[1] == 'missing' then
        if payload ~= false then return 0 end
    elseif payload ~= ARGV[2] then
        return 0
    end
    return redis.call('DEL', KEYS[1], KEYS[2])
    """

    _COMPARE_DELETE_VALUE_ORPHAN_SCRIPT = """
    local payload = redis.call('GET', KEYS[1])
    if payload == false or payload ~= ARGV[1] then return 0 end
    if redis.call('EXISTS', KEYS[2]) ~= 0 then return 0 end
    return redis.call('DEL', KEYS[1])
    """

    def _compare_delete_physical(
        self, meta_key: Any, value_key: Any, generation: str
    ) -> bool:
        """Delete a scanned pair only while its generation is unchanged."""
        try:
            result = self.client.eval(
                self._COMPARE_DELETE_SCRIPT,
                2,
                meta_key,
                value_key,
                generation,
            )
            return bool(result)
        except Exception as e:
            logger.warning(
                "Failed generation-checked Redis invalidation for %r/%r: %s",
                meta_key,
                value_key,
                e,
            )
            return False

    def _compare_quarantine_physical(
        self, meta_key: Any, value_key: Any, raw_data: Optional[Any]
    ) -> bool:
        """Quarantine a malformed pair only if its scanned payload persists."""
        try:
            if raw_data is None:
                result = self.client.eval(
                    self._COMPARE_QUARANTINE_SCRIPT,
                    2,
                    meta_key,
                    value_key,
                    "missing",
                    "",
                )
            else:
                result = self.client.eval(
                    self._COMPARE_QUARANTINE_SCRIPT,
                    2,
                    meta_key,
                    value_key,
                    "present",
                    raw_data,
                )
            return bool(result)
        except Exception as e:
            logger.warning(
                "Failed generation-independent Redis quarantine for %r/%r: %s",
                meta_key,
                value_key,
                e,
            )
            return False

    def _compare_delete_value_orphan(
        self, value_key: Any, meta_key: Any, raw_value: Any
    ) -> bool:
        """Delete a value-only record if no replacement pair was published."""
        try:
            result = self.client.eval(
                self._COMPARE_DELETE_VALUE_ORPHAN_SCRIPT,
                2,
                value_key,
                meta_key,
                raw_value,
            )
            return bool(result)
        except Exception as e:
            logger.warning(
                "Failed compare-checked Redis orphan cleanup for %r: %s",
                value_key,
                e,
            )
            return False

    def _cleanup_get_pair(
        self, value_key: Any, meta_key: Any, meta_dict: Optional[dict]
    ) -> None:
        """Remove a bad read pair only if its scanned metadata still matches."""
        raw_data = None
        if meta_dict:
            raw_data = meta_dict.get(b"data")
            if raw_data is None:
                raw_data = meta_dict.get("data")
        if raw_data is None:
            self._compare_quarantine_physical(meta_key, value_key, None)
            return
        try:
            raw_text = raw_data.decode("utf-8") if isinstance(raw_data, bytes) else raw_data
            generation = CacheMetadata.from_dict(json.loads(raw_text)).generation
        except Exception:
            self._compare_quarantine_physical(meta_key, value_key, raw_data)
            return
        self._compare_delete_physical(meta_key, value_key, generation)

    def get(self, key: str) -> Optional[Tuple[Any, CacheMetadata]]:
        prefixed = self._prefixed_key(key)
        meta_dict = None

        try:
            # Get value and metadata
            pipe = self.client.pipeline()
            pipe.get(prefixed)
            pipe.hgetall(f"{prefixed}:meta")
            results = pipe.execute()

            compressed_data = results[0]
            meta_dict = results[1]

            if not compressed_data or not meta_dict:
                if compressed_data or meta_dict:
                    self._cleanup_get_pair(prefixed, f"{prefixed}:meta", meta_dict)
                return None

            # Decode metadata
            meta_decoded = {
                k.decode("utf-8"): v.decode("utf-8")
                for k, v in meta_dict.items()
            }
            metadata = CacheMetadata.from_dict(json.loads(meta_decoded["data"]))

            # Check expiration
            if metadata.is_expired(time.time()):
                self._cleanup_get_pair(prefixed, f"{prefixed}:meta", meta_dict)
                return None
            if metadata.key != key or not _codec_compatible(
                metadata.compression_method, self.compressor
            ):
                self._cleanup_get_pair(prefixed, f"{prefixed}:meta", meta_dict)
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
            self._cleanup_get_pair(prefixed, f"{prefixed}:meta", meta_dict)
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

        try:
            if not _is_valid_ttl(ttl_seconds):
                return False
            # Capture the origin before potentially slow serialization or
            # compression so direct writes cannot extend absolute TTLs.
            now = time.time()
            entry_created_at = now if created_at is None else created_at
            if not _valid_origin(entry_created_at, now, ttl_seconds):
                return False

            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            compressed, stats = self.compressor.compress(data)
            publication_now = time.time()

            if ttl_seconds is None:
                redis_ttl = None
            else:
                # Physical expiry must cover only the remaining absolute TTL
                # after preprocessing, while metadata keeps the original origin.
                remaining_ttl = ttl_seconds - max(0.0, publication_now - entry_created_at)
                if not math.isfinite(remaining_ttl) or remaining_ttl <= 0:
                    return False
                # Redis' second-granularity expiry must not truncate a live
                # fractional TTL to zero or expire before the metadata does.
                redis_ttl = math.ceil(remaining_ttl)
            metadata = CacheMetadata(
                key=key,
                created_at=entry_created_at,
                last_accessed=publication_now,
                access_count=0,
                size_bytes=len(data),
                compressed_size=len(compressed),
                ttl_seconds=ttl_seconds,
                dependencies=set(dependencies) if dependencies is not None else set(),
                compression_method=_compression_method(self.compressor),
                generation=uuid.uuid4().hex,
            )

            # Store with pipeline
            pipe = self.client.pipeline()
            if redis_ttl is None:
                pipe.set(prefixed, compressed)
            else:
                pipe.setex(prefixed, redis_ttl, compressed)
            pipe.hset(
                f"{prefixed}:meta",
                mapping={"data": json.dumps(metadata.to_dict())}
            )
            if redis_ttl is not None:
                pipe.expire(f"{prefixed}:meta", redis_ttl)
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

    def invalidate_dependencies(self, dependency: str, *, return_keys: bool = False):
        """Invalidate Redis entries bound to the given dependency."""
        try:
            cursor = 0
            records_to_remove = []
            quarantine_records = []
            claimed_value_keys = set()
            pattern = f"{self.key_prefix}*:meta"
            while True:
                cursor, meta_keys = self.client.scan(cursor, match=pattern, count=100)
                for meta_key in meta_keys:
                    try:
                        meta_dict = self.client.hgetall(meta_key)
                        raw_data = meta_dict.get(b"data")
                        if raw_data is None:
                            raw_data = meta_dict.get("data")
                        if raw_data is None:
                            value_key = self._value_key_for_meta(meta_key)
                            if value_key is not None:
                                quarantine_records.append((meta_key, value_key, None))
                            continue
                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode("utf-8")
                        metadata = CacheMetadata.from_dict(json.loads(raw_data))
                        claimed_value_keys.add(
                            self._prefixed_key(metadata.key).encode("utf-8")
                        )
                        if dependency in metadata.dependencies:
                            value_key = self._value_key_for_meta(meta_key)
                            if value_key is not None:
                                records_to_remove.append(
                                    (meta_key, value_key, metadata.key, metadata.generation)
                                )
                    except Exception as e:
                        # One malformed record must not prevent later dependency-bound
                        # entries from being invalidated. Quarantine the unusable pair
                        # so it cannot remain as a promotion candidate.
                        value_key = self._value_key_for_meta(meta_key)
                        logger.warning(
                            f"Failed to inspect Redis cache metadata {meta_key!r}: {e}"
                        )
                        if value_key is not None:
                            quarantine_records.append((meta_key, value_key, raw_data))
                        else:
                            quarantine_records.append((meta_key, meta_key, raw_data))
                if cursor == 0:
                    break

            value_orphans = []
            cursor = 0
            epoch_key = self._invalidation_epoch_key()
            pattern = f"{self.key_prefix}*"
            while True:
                cursor, value_keys = self.client.scan(cursor, match=pattern, count=100)
                for value_key in value_keys:
                    text_key = (
                        value_key.decode("utf-8", errors="replace")
                        if isinstance(value_key, bytes)
                        else value_key
                    )
                    if text_key == epoch_key or text_key.endswith(":meta"):
                        continue
                    if value_key in claimed_value_keys:
                        continue
                    try:
                        raw_value = self.client.get(value_key)
                        meta_key = f"{text_key}:meta"
                        if raw_value is not None and not self.client.exists(meta_key):
                            value_orphans.append((value_key, meta_key, raw_value))
                    except Exception as e:
                        logger.warning(
                            "Failed to inspect Redis cache value %r: %s", value_key, e
                        )
                if cursor == 0:
                    break
            removed_keys = set()
            removed_count = 0
            for meta_key, value_key, raw_data in quarantine_records:
                self._compare_quarantine_physical(meta_key, value_key, raw_data)
            for value_key, meta_key, raw_value in value_orphans:
                self._compare_delete_value_orphan(value_key, meta_key, raw_value)
            for meta_key, value_key, logical_key, generation in records_to_remove:
                try:
                    # Delete only if the scanned generation is still current;
                    # replacements written after the scan must survive.
                    if self._compare_delete_physical(meta_key, value_key, generation):
                        removed_count += 1
                        if self._logical_key_matches_physical(logical_key, value_key):
                            removed_keys.add(logical_key)
                except Exception as e:
                    logger.warning(
                        "Failed to invalidate Redis pair %r/%r: %s",
                        meta_key,
                        value_key,
                        e,
                    )
            return removed_keys if return_keys else removed_count
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis dependency {dependency}: {e}")
            return set() if return_keys else 0

    def clear(self):
        """Clear all cache entries with this prefix."""
        try:
            pattern = f"{self.key_prefix}*"
            epoch_key = self._invalidation_epoch_key()
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                cache_keys = [key for key in keys if key != epoch_key]
                if cache_keys:
                    self.client.delete(*cache_keys)
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
        self._coordination_lock = threading.RLock()
        self._l1_invalidation_epoch: Optional[str] = (
            self.l2.invalidation_epoch()
            if self.l1 and self.l2
            else self.l3.invalidation_epoch()
            if self.l1 and self.l3
            else ""
        )

    def _shared_invalidation_epoch_unlocked(self) -> Optional[str]:
        """Read the epoch from the configured shared backing layer."""
        if self.l2:
            return self.l2._read_invalidation_epoch_unlocked()
        if self.l3:
            read_epoch = getattr(self.l3, "invalidation_epoch", None)
            if read_epoch is None:
                return None
            try:
                return read_epoch()
            except Exception as exc:
                logger.warning("Failed to read shared invalidation epoch: %s", exc)
                return None
        return ""

    def _bump_shared_invalidation_epoch_unlocked(self) -> Optional[str]:
        """Bump the epoch in the configured shared backing layer."""
        if self.l2:
            return self.l2._bump_invalidation_epoch_unlocked()
        if self.l3:
            bump_epoch = getattr(self.l3, "bump_invalidation_epoch", None)
            if bump_epoch is None:
                return None
            try:
                return bump_epoch()
            except Exception as exc:
                logger.warning("Failed to publish shared invalidation epoch: %s", exc)
                return None
        return ""

    def _record_shared_epoch_unlocked(self, epoch: Optional[str]) -> bool:
        """Record a successful fence; mark L1 stale when publication fails."""
        if epoch is None:
            # Do not destroy unrelated local entries merely because a shared
            # fence could not be published.  The unknown marker forces the
            # next coordinated read to validate the shared epoch before L1 use.
            self._l1_invalidation_epoch = None
            return False
        self._l1_invalidation_epoch = epoch
        return True

    def _refresh_l1_epoch_unlocked(self) -> bool:
        """Refresh the L1 epoch, clearing and bypassing L1 on read failure."""
        if not self.l1 or not (self.l2 or self.l3):
            return True
        current_epoch = self._shared_invalidation_epoch_unlocked()
        if current_epoch is None:
            self.l1.clear()
            self._l1_invalidation_epoch = None
            return False
        if current_epoch != self._l1_invalidation_epoch:
            self.l1.clear()
            self._l1_invalidation_epoch = current_epoch
        return True

    def _shared_epoch_unchanged_unlocked(self, expected_epoch: Optional[str]) -> bool:
        """Reject lower-layer reads that crossed a sibling invalidation fence."""
        if not self.l1 or not (self.l2 or self.l3):
            return True
        if self.l3 and not self.l2 and not callable(
            getattr(self.l3, "invalidation_epoch", None)
        ):
            return True
        current_epoch = self._shared_invalidation_epoch_unlocked()
        if current_epoch is None or current_epoch != expected_epoch:
            self.l1.clear()
            self._l1_invalidation_epoch = current_epoch
            return False
        return True

    def get(self, key: str) -> Optional[Any]:
        """Get a value while serializing promotion with dependency invalidation."""
        with self._coordination_lock:
            if self.l2:
                with self.l2._process_lock(), self.l2._root_lock():
                    return self._get_unlocked(key)
            return self._get_unlocked(key)

    def _get_unlocked(self, key: str) -> Optional[Any]:
        """
        Get value from cache (checks L1 → L2 → L3).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Try L1 (memory)
        if self.l1:
            l1_epoch_available = self._refresh_l1_epoch_unlocked()
            if l1_epoch_available:
                result = self.l1.get(key)
                if result is not None:
                    with self._stats_lock:
                        self.stats.l1_hits += 1
                    return result[0]

        # Try L2 (disk)
        if self.l2:
            read_epoch = self._shared_invalidation_epoch_unlocked()
            result = self.l2.get(key)
            if result is not None:
                value, metadata = result

                if not self._shared_epoch_unchanged_unlocked(read_epoch):
                    with self._stats_lock:
                        self.stats.misses += 1
                    return None

                # Promote to L1
                if self.l1:
                    self.l1.put(
                        key,
                        value,
                        ttl_seconds=metadata.ttl_seconds,
                        dependencies=metadata.dependencies,
                        created_at=metadata.created_at,
                    )

                # Promotion can consume the remaining absolute TTL. Never
                # return a value that expired while an upper layer serialized it.
                if metadata.is_expired(time.time()):
                    self._invalidate_unlocked(key)
                    with self._stats_lock:
                        self.stats.misses += 1
                    return None

                with self._stats_lock:
                    self.stats.l2_hits += 1
                return value

        # Try L3 (Redis)
        if self.l3:
            read_epoch = self._shared_invalidation_epoch_unlocked()
            result = self.l3.get(key)
            if result is not None:
                value, metadata = result

                if not self._shared_epoch_unchanged_unlocked(read_epoch):
                    with self._stats_lock:
                        self.stats.misses += 1
                    return None

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

                if metadata.is_expired(time.time()):
                    self._invalidate_unlocked(key)
                    with self._stats_lock:
                        self.stats.misses += 1
                    return None

                with self._stats_lock:
                    self.stats.l3_hits += 1
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
        """Store a value while serializing writes with invalidation."""
        with self._coordination_lock:
            return self._put_unlocked(
                key,
                value,
                ttl_seconds,
                dependencies,
                write_through,
            )

    def _put_unlocked(
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
        if not _is_valid_ttl(ttl_seconds):
            return False

        success = False
        # Capture one origin for every write-through layer. Each layer's TTL is
        # absolute from this instant, not from its individual serialization time.
        created_at = time.time()

        # Write to L1
        if self.l1:
            if self.l1.put(key, value, ttl_seconds, dependencies, created_at=created_at):
                success = True

        if write_through:
            # Write to L2
            if self.l2:
                if self.l2.put(
                    key, value, ttl_seconds, dependencies, created_at=created_at
                ):
                    success = True

            # Write to L3
            if self.l3:
                if self.l3.put(
                    key, value, ttl_seconds, dependencies, created_at=created_at
                ):
                    success = True

        return success

    def invalidate(self, key: str) -> bool:
        """Invalidate a key without racing with reads or promotion."""
        with self._coordination_lock:
            if self.l2:
                # Hold the process fence before the root lock for the entire
                # start-epoch, mutation, and completion-epoch transaction.
                with self.l2._process_lock(), self.l2._root_lock():
                    return self._invalidate_with_epoch_fence_unlocked(key)
            return self._invalidate_with_epoch_fence_unlocked(key)

    def _invalidate_with_epoch_fence_unlocked(self, key: str) -> bool:
        """Invalidate one key under the shared epoch protocol."""
        if self.l2 or self.l3:
            self._record_shared_epoch_unlocked(
                self._bump_shared_invalidation_epoch_unlocked()
            )
        removed = self._invalidate_unlocked(key)
        if removed and (self.l2 or self.l3):
            # Publish a completion fence so sibling coordinators cannot
            # retain a value promoted while physical invalidation ran.
            self._record_shared_epoch_unlocked(
                self._bump_shared_invalidation_epoch_unlocked()
            )
        return removed

    def _invalidate_unlocked(self, key: str) -> bool:
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
        """Invalidate a dependency without racing with reads or promotion."""
        with self._coordination_lock:
            if self.l2:
                with self.l2._process_lock(), self.l2._root_lock():
                    if self.l2 or self.l3:
                        self._record_shared_epoch_unlocked(
                            self._bump_shared_invalidation_epoch_unlocked()
                        )
                    invalidated = self._invalidate_dependency_unlocked(dependency)
            else:
                if self.l2 or self.l3:
                    self._record_shared_epoch_unlocked(
                        self._bump_shared_invalidation_epoch_unlocked()
                    )
                invalidated = self._invalidate_dependency_unlocked(dependency)

            if self.l2 or self.l3:
                # Publish a completion fence after physical removal.  A sibling
                # may otherwise observe the start fence, promote the stale value,
                # and retain it in L1 after this invalidation returns.
                self._record_shared_epoch_unlocked(
                    self._bump_shared_invalidation_epoch_unlocked()
                )
            return invalidated

    def _invalidate_dependency_unlocked(self, dependency: str) -> int:
        """
        Invalidate all entries that depend on the given dependency.

        Args:
            dependency: Dependency key

        Returns:
            Number of entries invalidated
        """
        invalidated_keys = set()
        pending_dependencies = [dependency]
        visited_dependencies = set()

        # Dependency edges may point at another cache key.  Walk the resulting
        # key graph so invalidating a raw input also removes derived entries
        # that depend on an invalidated intermediate result.
        while pending_dependencies:
            current_dependency = pending_dependencies.pop()
            if current_dependency in visited_dependencies:
                continue
            visited_dependencies.add(current_dependency)

            removed_this_round = set()
            if self.l1:
                removed_this_round.update(
                    self.l1.invalidate_dependencies(
                        current_dependency, return_keys=True
                    )
                )
            if self.l2:
                removed_this_round.update(
                    self.l2.invalidate_dependencies(
                        current_dependency, return_keys=True
                    )
                )
            if self.l3:
                removed_this_round.update(
                    self.l3.invalidate_dependencies(
                        current_dependency, return_keys=True
                    )
                )

            new_keys = removed_this_round - invalidated_keys
            invalidated_keys.update(new_keys)
            pending_dependencies.extend(new_keys)

        count = len(invalidated_keys)
        with self._stats_lock:
            self.stats.invalidations += count

        return count

    def clear(self):
        """Clear all cache layers without racing with reads or promotion."""
        with self._coordination_lock:
            if self.l2:
                # Keep both locks held from the start epoch through all layer
                # mutations and the completion epoch, matching invalidation.
                with self.l2._process_lock(), self.l2._root_lock():
                    self._clear_with_epoch_fence_unlocked()
                return
            self._clear_with_epoch_fence_unlocked()

    def _clear_with_epoch_fence_unlocked(self) -> None:
        """Clear all layers under the shared epoch protocol."""
        if self.l2 or self.l3:
            self._record_shared_epoch_unlocked(
                self._bump_shared_invalidation_epoch_unlocked()
            )
        if self.l1:
            self.l1.clear()
        if self.l2:
            self.l2.clear()
        if self.l3:
            self.l3.clear()
        if self.l2 or self.l3:
            # Completion fence prevents a sibling from retaining a value
            # promoted while the shared layers were being cleared.
            self._record_shared_epoch_unlocked(
                self._bump_shared_invalidation_epoch_unlocked()
            )

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
            with self._coordination_lock:
                if self._get_unlocked(key) is not None:
                    continue
                try:
                    value = loader(key)
                    self._put_unlocked(
                        key,
                        value,
                        ttl_seconds,
                        dependencies,
                    )
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
