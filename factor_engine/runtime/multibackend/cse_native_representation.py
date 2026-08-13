# -*- coding: utf-8 -*-
"""MB-P1-004: CSE using native representation to avoid conversions.

Common Subexpression Elimination that preserves native backend representation
to avoid unnecessary pandas ↔ polars ↔ arrow conversions.

Key principles:
- Track representation for each CSE node
- Reuse in native form when possible
- Only convert when consumer requires different backend
- Minimize conversion overhead in shared computation paths
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any
from weakref import WeakValueDictionary

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeCSEKey:
    """Key for CSE cache with representation tracking.

    Attributes:
        structural_hash: Structural hash of computation
        representation: Native representation (pandas/polars/arrow/duckdb)
        semantic_digest: Semantic identity digest
    """
    structural_hash: str
    representation: str
    semantic_digest: str | None = None

    def with_representation(self, new_repr: str) -> NativeCSEKey:
        """Create key variant with different representation."""
        return NativeCSEKey(
            structural_hash=self.structural_hash,
            representation=new_repr,
            semantic_digest=self.semantic_digest,
        )


@dataclass
class CSEEntry:
    """Cached CSE result with native representation.

    Attributes:
        result: Computed result (in native representation)
        representation: Representation format
        size_bytes: Memory footprint estimate
        reuse_count: Number of times reused
        conversion_cache: Converted versions (other representations)
    """
    result: Any
    representation: str
    size_bytes: int
    reuse_count: int = 0
    conversion_cache: dict[str, Any] | None = None


class NativeCSECache:
    """CSE cache preserving native representation.

    Avoids unnecessary conversions by tracking representation for each
    cached result and serving in native form when possible.
    """

    def __init__(self, max_size_bytes: int = 2 * 1024**3):  # 2 GB
        """Initialize native CSE cache.

        Args:
            max_size_bytes: Maximum cache size (bytes)
        """
        self._cache: dict[NativeCSEKey, CSEEntry] = {}
        self._max_size_bytes = max_size_bytes
        self._current_size_bytes = 0
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        self._conversion_avoided_count = 0
        self._conversion_performed_count = 0

    def get(
        self,
        key: NativeCSEKey,
        *,
        accept_conversion: bool = True,
    ) -> CSEEntry | None:
        """Get cached result, optionally accepting converted version.

        Args:
            key: CSE key with desired representation
            accept_conversion: If True, accept cached result in different
                             representation and convert

        Returns:
            CSEEntry if found, None otherwise
        """
        with self._lock:
            # Try exact match (native representation)
            if key in self._cache:
                entry = self._cache[key]
                entry.reuse_count += 1
                self._hit_count += 1
                self._conversion_avoided_count += 1
                _logger.debug(
                    f"CSE hit (native {key.representation}): "
                    f"{key.structural_hash[:8]}"
                )
                return entry

            # Try to find in different representation
            if accept_conversion:
                for cached_key, entry in self._cache.items():
                    if (cached_key.structural_hash == key.structural_hash
                        and cached_key.semantic_digest == key.semantic_digest):
                        # Found in different representation
                        converted = self._convert_entry(
                            entry, key.representation
                        )
                        if converted is not None:
                            entry.reuse_count += 1
                            self._hit_count += 1
                            self._conversion_performed_count += 1
                            _logger.debug(
                                f"CSE hit (converted {entry.representation} → "
                                f"{key.representation}): {key.structural_hash[:8]}"
                            )
                            return converted

            self._miss_count += 1
            return None

    def put(
        self,
        key: NativeCSEKey,
        result: Any,
        size_bytes: int,
    ) -> None:
        """Store result in cache with native representation.

        Args:
            key: CSE key with native representation
            result: Computed result
            size_bytes: Memory footprint estimate
        """
        with self._lock:
            # Evict if necessary
            while (self._current_size_bytes + size_bytes > self._max_size_bytes
                   and self._cache):
                self._evict_lru()

            # Store entry
            entry = CSEEntry(
                result=result,
                representation=key.representation,
                size_bytes=size_bytes,
                reuse_count=0,
                conversion_cache={},
            )
            self._cache[key] = entry
            self._current_size_bytes += size_bytes

            _logger.debug(
                f"CSE store ({key.representation}): {key.structural_hash[:8]}, "
                f"{size_bytes // 1024} KB"
            )

    def invalidate(self, structural_hash: str) -> None:
        """Invalidate all entries matching structural hash."""
        with self._lock:
            to_remove = [
                k for k in self._cache
                if k.structural_hash == structural_hash
            ]
            for key in to_remove:
                entry = self._cache.pop(key)
                self._current_size_bytes -= entry.size_bytes
                _logger.debug(f"CSE invalidate: {key.structural_hash[:8]}")

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            _logger.info("CSE cache cleared")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = (
                self._hit_count / total_requests if total_requests > 0 else 0.0
            )
            conversion_avoided_rate = (
                self._conversion_avoided_count / self._hit_count
                if self._hit_count > 0 else 0.0
            )

            return {
                "entries": len(self._cache),
                "size_bytes": self._current_size_bytes,
                "max_size_bytes": self._max_size_bytes,
                "utilization": self._current_size_bytes / self._max_size_bytes,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "conversion_avoided_count": self._conversion_avoided_count,
                "conversion_performed_count": self._conversion_performed_count,
                "conversion_avoided_rate": conversion_avoided_rate,
                "total_reuse_count": sum(
                    e.reuse_count for e in self._cache.values()
                ),
            }

    def _evict_lru(self) -> None:
        """Evict least-recently-used entry."""
        if not self._cache:
            return

        # Find entry with lowest reuse count (simple LRU approximation)
        victim_key = min(self._cache.keys(), key=lambda k: self._cache[k].reuse_count)
        victim_entry = self._cache.pop(victim_key)
        self._current_size_bytes -= victim_entry.size_bytes

        _logger.debug(
            f"CSE evict: {victim_key.structural_hash[:8]}, "
            f"reuse={victim_entry.reuse_count}"
        )

    def _convert_entry(
        self, entry: CSEEntry, target_repr: str
    ) -> CSEEntry | None:
        """Convert cached entry to target representation.

        Args:
            entry: Source entry
            target_repr: Target representation

        Returns:
            Converted entry or None if conversion fails
        """
        # Check conversion cache first
        if entry.conversion_cache and target_repr in entry.conversion_cache:
            return CSEEntry(
                result=entry.conversion_cache[target_repr],
                representation=target_repr,
                size_bytes=entry.size_bytes,
                reuse_count=entry.reuse_count,
            )

        # Perform conversion
        try:
            converted_result = self._convert_data(
                entry.result, entry.representation, target_repr
            )
            if converted_result is not None:
                # Cache conversion
                if entry.conversion_cache is None:
                    entry.conversion_cache = {}
                entry.conversion_cache[target_repr] = converted_result

                return CSEEntry(
                    result=converted_result,
                    representation=target_repr,
                    size_bytes=entry.size_bytes,
                    reuse_count=entry.reuse_count,
                )

        except Exception as exc:
            _logger.warning(
                f"Conversion failed {entry.representation} → {target_repr}: {exc}"
            )

        return None

    def _convert_data(
        self, data: Any, source_repr: str, target_repr: str
    ) -> Any | None:
        """Convert data between representations.

        Args:
            data: Source data
            source_repr: Source representation
            target_repr: Target representation

        Returns:
            Converted data or None if conversion not supported
        """
        if source_repr == target_repr:
            return data

        try:
            # pandas → polars
            if source_repr == "pandas" and target_repr == "polars":
                import polars as pl
                return pl.from_pandas(data)

            # polars → pandas
            if source_repr == "polars" and target_repr == "pandas":
                return data.to_pandas()

            # pandas → arrow
            if source_repr == "pandas" and target_repr == "arrow":
                import pyarrow as pa
                return pa.Table.from_pandas(data)

            # arrow → pandas
            if source_repr == "arrow" and target_repr == "pandas":
                return data.to_pandas()

            # polars → arrow
            if source_repr == "polars" and target_repr == "arrow":
                return data.to_arrow()

            # arrow → polars
            if source_repr == "arrow" and target_repr == "polars":
                import polars as pl
                return pl.from_arrow(data)

            _logger.warning(
                f"Conversion not implemented: {source_repr} → {target_repr}"
            )
            return None

        except Exception as exc:
            _logger.warning(
                f"Conversion error {source_repr} → {target_repr}: {exc}"
            )
            return None


class NativeCSEManager:
    """Manager for native representation CSE across batch execution.

    Provides high-level interface for CSE with representation tracking,
    integrated with adaptive_batch_scheduler.
    """

    def __init__(self, cache_size_bytes: int = 2 * 1024**3):
        """Initialize CSE manager.

        Args:
            cache_size_bytes: Maximum cache size
        """
        self._cache = NativeCSECache(max_size_bytes=cache_size_bytes)
        self._pending_computations: dict[str, threading.Event] = {}
        self._pending_lock = threading.Lock()

    def get_or_compute(
        self,
        key: NativeCSEKey,
        compute_fn: Any,
        *,
        size_estimate_bytes: int = 0,
        accept_conversion: bool = True,
    ) -> Any:
        """Get cached result or compute if missing.

        Handles concurrent requests for same computation (deduplicate).

        Args:
            key: CSE key with desired representation
            compute_fn: Callable to compute result if cache miss
            size_estimate_bytes: Memory size estimate for caching
            accept_conversion: Accept converted representation

        Returns:
            Computed or cached result
        """
        # Try cache first
        entry = self._cache.get(key, accept_conversion=accept_conversion)
        if entry is not None:
            return entry.result

        # Check if computation is pending (avoid duplicate work)
        with self._pending_lock:
            structural = key.structural_hash
            if structural in self._pending_computations:
                # Wait for pending computation
                event = self._pending_computations[structural]
                _logger.debug(f"Waiting for pending computation: {structural[:8]}")
            else:
                # Mark as pending
                event = threading.Event()
                self._pending_computations[structural] = event

        # Wait if another thread is computing
        if not event.is_set():
            if event is not self._pending_computations.get(structural):
                event.wait(timeout=300)  # 5 min max wait
                # Try cache again
                entry = self._cache.get(key, accept_conversion=accept_conversion)
                if entry is not None:
                    return entry.result

        # Compute
        try:
            result = compute_fn()

            # Store in cache
            if size_estimate_bytes == 0:
                size_estimate_bytes = self._estimate_size(result)

            self._cache.put(key, result, size_estimate_bytes)

            return result

        finally:
            # Release pending waiters
            with self._pending_lock:
                if structural in self._pending_computations:
                    self._pending_computations[structural].set()
                    del self._pending_computations[structural]

    def invalidate(self, structural_hash: str) -> None:
        """Invalidate cached computation."""
        self._cache.invalidate(structural_hash)

    def clear(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get CSE statistics."""
        return self._cache.stats()

    def _estimate_size(self, result: Any) -> int:
        """Estimate memory size of result."""
        try:
            # Try pandas
            if hasattr(result, "memory_usage"):
                return int(result.memory_usage(deep=True).sum())

            # Try polars
            if hasattr(result, "estimated_size"):
                return int(result.estimated_size())

            # Try arrow
            if hasattr(result, "nbytes"):
                return int(result.nbytes)

        except Exception:
            pass

        # Conservative fallback
        return 100 * 1024 * 1024  # 100 MB


# Global singleton
_GLOBAL_CSE_MANAGER: NativeCSEManager | None = None


def global_cse_manager() -> NativeCSEManager:
    """Get global native CSE manager singleton."""
    global _GLOBAL_CSE_MANAGER
    if _GLOBAL_CSE_MANAGER is None:
        _GLOBAL_CSE_MANAGER = NativeCSEManager()
    return _GLOBAL_CSE_MANAGER
