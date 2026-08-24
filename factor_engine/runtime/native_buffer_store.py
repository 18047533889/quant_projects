# -*- coding: utf-8 -*-
"""MB-P1-012/MB-P1-013: Representation-aware native buffer store.

Fixes:
- MB-P1-012: CSE cache distinguishes native representations
- MB-P1-013: Same-backend consumers share native buffer, no redundant conversion
- MB-P1-014: Liveness-driven early free via refcount tracking

A single semantic node can have multiple physical representations:
  - Polars LazyFrame (native)
  - Arrow Table
  - Pandas DataFrame (long)
  - Pandas DataFrame (wide)

When multiple consumers use the same backend, they share the native buffer.
Cross-backend conversion happens exactly once per target representation and is cached.

Example:
  100 Polars roots consume shared node X:
    - X materialized once as Polars native
    - All 100 roots reuse that native buffer
    - Zero conversions

  3 Pandas roots also need X:
    - Convert Polars->Pandas once
    - Cache second representation
    - All 3 Pandas roots reuse that conversion
    - Total: 1 conversion, not 103
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from factor_engine.planner.backend_region import Representation


class BufferState(str, Enum):
    """Buffer lifecycle state."""

    AVAILABLE = "available"
    CONVERTING = "converting"  # Conversion in progress
    EVICTED = "evicted"


@dataclass
class RepresentationBuffer:
    """A single physical representation of a semantic node.

    MB-P1-012: CSE nodes can have multiple representations, one per backend family.
    MB-P1-013: Same-backend consumers share this native buffer.
    """

    representation: Representation
    value: Any
    bytes: int
    created_at: float
    last_access: float
    access_count: int = 0
    refcount: int = 0  # MB-P1-014: Consumer count for liveness
    state: BufferState = BufferState.AVAILABLE


@dataclass
class NativeBuffer:
    """Multi-representation buffer for a semantic node.

    MB-P1-012: A single semantic node (e.g., shared CSE expression) can have
    multiple physical representations:
      - representations: Dict mapping Representation -> RepresentationBuffer
      - Each backend family gets its own native buffer
      - Conversions are cached and shared

    MB-P1-013: Same-backend consumers share native representation:
      - 100 Polars consumers all use Polars native buffer
      - 3 Pandas consumers trigger one conversion, then all share it
      - Total: 1 conversion, not 103

    MB-P1-014: Liveness tracking:
      - Each consumer acquires a reference (refcount++)
      - When consumer finishes, release reference (refcount--)
      - When refcount reaches 0, buffer becomes eligible for eviction
    """

    semantic_node_id: str
    source_snapshot: str
    scope: str
    representations: dict[Representation, RepresentationBuffer] = field(default_factory=dict)
    total_bytes: int = 0
    created_at: float = field(default_factory=time.monotonic)
    # MB-P1-014: Remaining consumer count
    remaining_consumers: int = 0
    next_use_distance: int = 0  # For Belady-like eviction


class NativeBufferStore:
    """MB-P1-012/MB-P1-013: Representation-aware buffer cache.

    Stores multiple physical representations of semantic nodes, allowing
    same-backend consumers to share native buffers without conversion.
    """

    def __init__(self, *, max_memory_bytes: int = 1_000_000_000) -> None:
        self.max_memory_bytes = max_memory_bytes
        self._buffers: dict[str, NativeBuffer] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "conversions": 0,
            "evictions": 0,
            "bytes_peak": 0,
        }

    def get_native(
        self,
        semantic_node_id: str,
        representation: Representation,
        *,
        source_snapshot: str = "",
        scope: str = "",
    ) -> Any | None:
        """MB-P1-013: Get native buffer for specific representation.

        Args:
            semantic_node_id: Semantic identity of the node
            representation: Desired physical representation
            source_snapshot: Source snapshot binding
            scope: Execution scope

        Returns:
            Native buffer if available, None if not cached
        """
        with self._lock:
            key = self._make_key(semantic_node_id, source_snapshot, scope)
            buffer = self._buffers.get(key)

            if buffer is None:
                self._stats["misses"] += 1
                return None

            rep_buffer = buffer.representations.get(representation)
            if rep_buffer is None:
                self._stats["misses"] += 1
                return None

            if rep_buffer.state != BufferState.AVAILABLE:
                return None

            # Update access tracking
            rep_buffer.last_access = time.monotonic()
            rep_buffer.access_count += 1
            self._stats["hits"] += 1

            return rep_buffer.value

    def put_native(
        self,
        semantic_node_id: str,
        representation: Representation,
        value: Any,
        *,
        source_snapshot: str = "",
        scope: str = "",
        estimated_bytes: int = 0,
        consumer_count: int = 1,
    ) -> bool:
        """MB-P1-012: Store native buffer for a specific representation.

        Args:
            semantic_node_id: Semantic identity
            representation: Physical representation
            value: Actual buffer (Polars LazyFrame, Pandas DataFrame, etc.)
            source_snapshot: Source binding
            scope: Execution scope
            estimated_bytes: Size estimate
            consumer_count: Number of consumers (for MB-P1-014 liveness)

        Returns:
            True if stored, False if evicted or refused
        """
        if estimated_bytes == 0:
            estimated_bytes = self._estimate_bytes(value)

        with self._lock:
            # Check memory budget
            current_bytes = sum(b.total_bytes for b in self._buffers.values())
            if current_bytes + estimated_bytes > self.max_memory_bytes:
                # Try eviction
                self._evict_to_fit(estimated_bytes)
                current_bytes = sum(b.total_bytes for b in self._buffers.values())
                if current_bytes + estimated_bytes > self.max_memory_bytes:
                    return False  # Can't fit

            key = self._make_key(semantic_node_id, source_snapshot, scope)
            buffer = self._buffers.get(key)

            now = time.monotonic()
            if buffer is None:
                # Create new multi-representation buffer
                buffer = NativeBuffer(
                    semantic_node_id=semantic_node_id,
                    source_snapshot=source_snapshot,
                    scope=scope,
                    created_at=now,
                    remaining_consumers=consumer_count,
                )
                self._buffers[key] = buffer

            # Add this representation
            # FIX: If overwriting existing representation, subtract old bytes first
            old_rep = buffer.representations.get(representation)
            if old_rep is not None:
                buffer.total_bytes -= old_rep.bytes

            rep_buffer = RepresentationBuffer(
                representation=representation,
                value=value,
                bytes=estimated_bytes,
                created_at=now,
                last_access=now,
                refcount=consumer_count,
            )
            buffer.representations[representation] = rep_buffer
            buffer.total_bytes += estimated_bytes

            # Update stats
            peak = sum(b.total_bytes for b in self._buffers.values())
            if peak > self._stats["bytes_peak"]:
                self._stats["bytes_peak"] = peak

            return True

    def convert_and_cache(
        self,
        semantic_node_id: str,
        source_representation: Representation,
        target_representation: Representation,
        converter: callable,
        *,
        source_snapshot: str = "",
        scope: str = "",
        consumer_count: int = 1,
    ) -> Any | None:
        """MB-P1-013: Convert between representations and cache result.

        Ensures conversion happens at most once per representation pair.
        Multiple consumers of the target representation share the cached result.

        Args:
            semantic_node_id: Semantic node ID
            source_representation: Source format
            target_representation: Target format
            converter: Function to convert source -> target
            source_snapshot: Source binding
            scope: Execution scope
            consumer_count: How many consumers need target representation

        Returns:
            Converted value, or None if conversion fails
        """
        with self._lock:
            # Check if target already exists
            existing = self.get_native(
                semantic_node_id, target_representation,
                source_snapshot=source_snapshot, scope=scope
            )
            if existing is not None:
                return existing

            # Get source
            source_value = self.get_native(
                semantic_node_id, source_representation,
                source_snapshot=source_snapshot, scope=scope
            )
            if source_value is None:
                return None

        # Convert outside lock (can be expensive)
        try:
            converted = converter(source_value)
        except Exception:
            return None

        with self._lock:
            self._stats["conversions"] += 1

            # Cache converted result
            estimated_bytes = self._estimate_bytes(converted)
            success = self.put_native(
                semantic_node_id,
                target_representation,
                converted,
                source_snapshot=source_snapshot,
                scope=scope,
                estimated_bytes=estimated_bytes,
                consumer_count=consumer_count,
            )

            if success:
                return converted
            return None

    def acquire_ref(
        self,
        semantic_node_id: str,
        representation: Representation,
        *,
        source_snapshot: str = "",
        scope: str = "",
    ) -> bool:
        """MB-P1-014: Acquire reference before consuming buffer.

        Returns:
            True if reference acquired, False if buffer not found
        """
        with self._lock:
            key = self._make_key(semantic_node_id, source_snapshot, scope)
            buffer = self._buffers.get(key)
            if buffer is None:
                return False

            rep_buffer = buffer.representations.get(representation)
            if rep_buffer is None:
                return False

            rep_buffer.refcount += 1
            return True

    def release_ref(
        self,
        semantic_node_id: str,
        representation: Representation,
        *,
        source_snapshot: str = "",
        scope: str = "",
    ) -> None:
        """MB-P1-014: Release reference after consuming buffer.

        When refcount reaches 0, buffer becomes eligible for early eviction.
        This implements liveness-driven memory management.
        """
        with self._lock:
            key = self._make_key(semantic_node_id, source_snapshot, scope)
            buffer = self._buffers.get(key)
            if buffer is None:
                return

            rep_buffer = buffer.representations.get(representation)
            if rep_buffer is None:
                return

            if rep_buffer.refcount > 0:
                rep_buffer.refcount -= 1

            # MB-P1-014: If refcount is 0 and we need space, this is now
            # eligible for immediate eviction (early free)
            if rep_buffer.refcount == 0:
                buffer.remaining_consumers = max(0, buffer.remaining_consumers - 1)

    def release_consumed_by_root(
        self,
        semantic_node_ids: list[str],
        representation: Representation,
        *,
        source_snapshot: str = "",
        scope: str = "",
    ) -> None:
        """MB-P1-014: Release all references consumed by a completed root.

        This implements the early free pattern: when a root completes,
        immediately release all its consumed CSE buffers, making them
        eligible for eviction without waiting for full batch completion.
        """
        for node_id in semantic_node_ids:
            self.release_ref(
                node_id, representation,
                source_snapshot=source_snapshot, scope=scope
            )

    def _evict_to_fit(self, needed_bytes: int) -> None:
        """Evict buffers to free space.

        MB-P1-014: Prioritizes evicting buffers with refcount=0 (no active consumers).
        Uses LRU within that set.
        """
        candidates: list[tuple[str, float]] = []
        now = time.monotonic()

        for key, buffer in self._buffers.items():
            # Only evict if no active consumers
            if all(rb.refcount == 0 for rb in buffer.representations.values()):
                # Score by staleness (LRU)
                last_access = max(
                    rb.last_access for rb in buffer.representations.values()
                    if rb.state == BufferState.AVAILABLE
                ) if buffer.representations else 0.0
                staleness = now - last_access
                candidates.append((key, staleness))

        # Evict stalest first
        candidates.sort(key=lambda x: x[1], reverse=True)

        freed = 0
        for key, _ in candidates:
            if freed >= needed_bytes:
                break

            buffer = self._buffers.get(key)
            if buffer:
                freed += buffer.total_bytes
                del self._buffers[key]
                self._stats["evictions"] += 1

    def _estimate_bytes(self, value: Any) -> int:
        """Estimate memory footprint of a value."""
        try:
            from factor_engine.runtime.resource_governor import estimate_object_bytes
            return max(0, int(estimate_object_bytes(value)))
        except Exception:
            return 1_000_000  # Conservative fallback

    def _make_key(
        self, semantic_node_id: str, source_snapshot: str, scope: str
    ) -> str:
        """Create cache key from semantic coordinates."""
        return f"{semantic_node_id}::{source_snapshot}::{scope}"

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_bytes = sum(b.total_bytes for b in self._buffers.values())
            total_reps = sum(
                len(b.representations) for b in self._buffers.values()
            )
            return {
                **self._stats,
                "buffers": len(self._buffers),
                "total_representations": total_reps,
                "bytes_current": total_bytes,
                "utilization": total_bytes / max(1, self.max_memory_bytes),
            }


def convert_polars_to_pandas(polars_value: Any) -> Any:
    """Example converter: Polars -> Pandas."""
    if hasattr(polars_value, "collect"):
        return polars_value.collect().to_pandas()
    if hasattr(polars_value, "to_pandas"):
        return polars_value.to_pandas()
    return polars_value


def convert_pandas_to_polars(pandas_value: Any) -> Any:
    """Example converter: Pandas -> Polars."""
    try:
        import polars as pl
        return pl.from_pandas(pandas_value).lazy()
    except Exception:
        return pandas_value
