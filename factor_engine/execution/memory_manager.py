# -*- coding: utf-8 -*-
"""MB-P1-014/015/016: Memory manager with liveness tracking and streaming.

Implements:
    - Liveness/refcount early free (MB-P1-014, §46)
    - Representation-aware buffer store (MB-P1-012/013)
    - Streaming-first for low memory (MB-P1-016, §49)
    - KEEP_NATIVE / RECOMPUTE / SPILL / DIRECT_SINK decisions (MB-P1-015, §48)

Design:
    - NativeBufferStore: multi-representation buffer cache
    - LivenessManager: refcount-driven early release
    - MaterializationPolicy: when to keep/recompute/spill
    - StreamingCoordinator: low-memory streaming decisions
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MaterializationPolicy(Enum):
    """MB-P1-015, §48: Materialization decisions."""

    KEEP_NATIVE_MEMORY = "keep_native"
    RECOMPUTE = "recompute"
    SPILL_ARROW = "spill_arrow"
    SPILL_PARQUET = "spill_parquet"
    DIRECT_SINK = "direct_sink"


@dataclass(frozen=True)
class BufferKey:
    """MB-P1-012, §45: Buffer identity key.

    Unique key for semantic node + representation + snapshot.
    """

    semantic_node_id: str
    source_snapshot: str
    scope: str
    representation: str
    dtype_schema: str
    backend_semantic_version: str

    def to_tuple(self) -> tuple[str, ...]:
        return (
            self.semantic_node_id,
            self.source_snapshot,
            self.scope,
            self.representation,
            self.dtype_schema,
            self.backend_semantic_version,
        )


@dataclass
class BufferMetadata:
    """Buffer metadata for liveness tracking."""

    key: BufferKey
    size_bytes: int
    remaining_consumers: int
    created_ms: float
    last_accessed_ms: float
    materialization_policy: MaterializationPolicy
    spill_path: str | None = None


class NativeBufferStore:
    """MB-P1-012/013, §45: Representation-aware buffer store.

    Same semantic node can have multiple representations:
        - Polars native
        - Arrow
        - Pandas
        - q

    Each representation converted at most once, shared by consumers.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, ...], Any] = {}
        self._metadata: dict[tuple[str, ...], BufferMetadata] = {}

    def store(
        self,
        key: BufferKey,
        buffer: Any,
        size_bytes: int,
        consumer_count: int,
        policy: MaterializationPolicy = MaterializationPolicy.KEEP_NATIVE_MEMORY,
    ) -> None:
        """Store a buffer with metadata."""
        import time

        key_tuple = key.to_tuple()
        self._buffers[key_tuple] = buffer
        self._metadata[key_tuple] = BufferMetadata(
            key=key,
            size_bytes=size_bytes,
            remaining_consumers=consumer_count,
            created_ms=time.time() * 1000.0,
            last_accessed_ms=time.time() * 1000.0,
            materialization_policy=policy,
        )

    def get(self, key: BufferKey) -> Any | None:
        """Retrieve buffer if exists."""
        import time

        key_tuple = key.to_tuple()
        if key_tuple not in self._buffers:
            return None

        # Update last accessed
        meta = self._metadata.get(key_tuple)
        if meta:
            meta.last_accessed_ms = time.time() * 1000.0

        return self._buffers.get(key_tuple)

    def mark_consumed(self, key: BufferKey) -> bool:
        """Mark one consumer complete. Returns True if buffer can be released."""
        key_tuple = key.to_tuple()
        meta = self._metadata.get(key_tuple)
        if not meta:
            return False

        meta.remaining_consumers -= 1
        if meta.remaining_consumers <= 0:
            return True
        return False

    def release(self, key: BufferKey) -> int:
        """Release buffer, return freed bytes."""
        key_tuple = key.to_tuple()
        meta = self._metadata.pop(key_tuple, None)
        self._buffers.pop(key_tuple, None)
        return meta.size_bytes if meta else 0

    def total_live_bytes(self) -> int:
        """Total memory in store."""
        return sum(m.size_bytes for m in self._metadata.values())

    def count(self) -> int:
        """Buffer count."""
        return len(self._buffers)

    def list_keys(self) -> list[BufferKey]:
        """List all buffer keys."""
        return [m.key for m in self._metadata.values()]

    def to_dict(self) -> dict[str, Any]:
        """Export state."""
        return {
            "buffer_count": self.count(),
            "total_live_bytes": self.total_live_bytes(),
            "buffers": [
                {
                    "semantic_node_id": m.key.semantic_node_id,
                    "representation": m.key.representation,
                    "size_bytes": m.size_bytes,
                    "remaining_consumers": m.remaining_consumers,
                    "policy": m.materialization_policy.value,
                }
                for m in self._metadata.values()
            ],
        }


class LivenessManager:
    """MB-P1-014, §46: Liveness and refcount manager.

    Coordinates with NativeBufferStore for early release.
    """

    def __init__(self, buffer_store: NativeBufferStore) -> None:
        self._store = buffer_store
        self._dependency_graph: dict[str, list[str]] = {}  # node -> consumers

    def register_dependencies(
        self,
        node_id: str,
        consumer_ids: list[str],
    ) -> None:
        """Register node dependencies."""
        self._dependency_graph[node_id] = consumer_ids

    def notify_complete(self, node_id: str) -> list[str]:
        """Notify node completion. Returns nodes that can be released.

        Args:
            node_id: Completed node ID

        Returns:
            List of parent node IDs that can now be released
        """
        releasable = []

        # Check all parents
        for parent_id, consumers in self._dependency_graph.items():
            if node_id in consumers:
                # This node was a consumer of parent
                # Check if we should release parent
                remaining = [c for c in consumers if c != node_id]
                self._dependency_graph[parent_id] = remaining

                if len(remaining) == 0:
                    releasable.append(parent_id)

        return releasable

    def release_node(self, node_id: str) -> int:
        """Release node buffer, return freed bytes."""
        # Find all buffer keys for this node
        freed = 0
        for key in self._store.list_keys():
            if key.semantic_node_id == node_id:
                freed += self._store.release(key)

        # Remove from dependency graph
        self._dependency_graph.pop(node_id, None)

        return freed

    def total_live_bytes(self) -> int:
        """Total live memory."""
        return self._store.total_live_bytes()


class MaterializationDecision:
    """MB-P1-015, §48: Decide materialization policy per buffer.

    CSE doesn't always mean materialize. Balance:
        - Reuse count
        - Recompute cost
        - Memory pressure
        - I/O cost
    """

    @staticmethod
    def decide(
        *,
        reuse_count: int,
        recompute_cost_ms: float,
        size_bytes: int,
        memory_pressure: float,
        is_final_sink: bool = False,
    ) -> MaterializationPolicy:
        """Decide materialization policy.

        Args:
            reuse_count: Number of times buffer will be reused
            recompute_cost_ms: Cost to recompute
            size_bytes: Buffer size
            memory_pressure: 0.0-1.0, current memory pressure
            is_final_sink: Is this a final output

        Returns:
            Materialization policy
        """
        if is_final_sink:
            return MaterializationPolicy.DIRECT_SINK

        # High memory pressure: prefer spill or recompute
        if memory_pressure > 0.8:
            if recompute_cost_ms < 10.0:  # Cheap to recompute
                return MaterializationPolicy.RECOMPUTE
            else:
                # Expensive to recompute: spill
                if size_bytes > 100 * 1024**2:  # >100MB
                    return MaterializationPolicy.SPILL_PARQUET
                else:
                    return MaterializationPolicy.SPILL_ARROW

        # Low reuse: recompute
        if reuse_count <= 1:
            return MaterializationPolicy.RECOMPUTE

        # High reuse: keep in memory
        if reuse_count >= 3:
            return MaterializationPolicy.KEEP_NATIVE_MEMORY

        # Medium reuse: balance memory vs cost
        memory_cost = size_bytes / (1024**3)  # GB
        time_cost = recompute_cost_ms * (reuse_count - 1)

        if time_cost > memory_cost * 100:  # Time expensive
            return MaterializationPolicy.KEEP_NATIVE_MEMORY
        else:
            return MaterializationPolicy.RECOMPUTE


class StreamingCoordinator:
    """MB-P1-016, §49: Streaming-first coordinator for low memory.

    Low-memory servers must:
        - Streaming-first
        - Projection pushdown
        - Early release
        - Adaptive batch size
        - Preemptive spill
        - Direct sink
        - Avoid giant dense factor cube
    """

    def __init__(
        self,
        *,
        memory_budget_bytes: int,
        safety_factor: float = 0.7,
    ) -> None:
        self._budget = memory_budget_bytes
        self._safety = safety_factor
        self._safe_budget = int(memory_budget_bytes * safety_factor)

    def should_stream(
        self,
        *,
        estimated_peak_bytes: int,
        operator: str,
    ) -> bool:
        """Decide if operation should stream.

        Args:
            estimated_peak_bytes: Estimated peak memory
            operator: Operation name

        Returns:
            True if should use streaming
        """
        # Always stream if exceeds safe budget
        if estimated_peak_bytes > self._safe_budget:
            return True

        # Stream for large aggregations
        if "group" in operator.lower() or "agg" in operator.lower():
            if estimated_peak_bytes > self._safe_budget * 0.5:
                return True

        return False

    def adaptive_batch_size(
        self,
        *,
        current_live_bytes: int,
        default_batch: int = 100,
    ) -> int:
        """Adaptive batch size based on memory pressure.

        Args:
            current_live_bytes: Current memory usage
            default_batch: Default batch size

        Returns:
            Adjusted batch size
        """
        available = self._safe_budget - current_live_bytes
        if available < 0:
            available = 0

        pressure = current_live_bytes / self._safe_budget

        if pressure > 0.9:
            return max(10, default_batch // 4)
        elif pressure > 0.7:
            return max(25, default_batch // 2)
        elif pressure > 0.5:
            return max(50, int(default_batch * 0.75))
        else:
            return default_batch

    def should_preemptive_spill(
        self,
        *,
        current_live_bytes: int,
        incoming_bytes: int,
    ) -> bool:
        """Decide if should preemptively spill.

        Args:
            current_live_bytes: Current memory usage
            incoming_bytes: Incoming allocation

        Returns:
            True if should spill before allocation
        """
        total = current_live_bytes + incoming_bytes
        return total > self._safe_budget

    def select_spill_candidates(
        self,
        *,
        buffers: list[BufferMetadata],
        target_free_bytes: int,
    ) -> list[BufferKey]:
        """Select buffers to spill to free target bytes.

        Strategy: LRU with reuse count consideration.

        Args:
            buffers: Available buffers
            target_free_bytes: Target bytes to free

        Returns:
            List of buffer keys to spill
        """
        # Sort by score: last_accessed (older first), remaining_consumers (fewer first)
        scored = []
        for b in buffers:
            if b.materialization_policy == MaterializationPolicy.KEEP_NATIVE_MEMORY:
                # Score: older + fewer consumers = higher score = spill first
                score = -b.last_accessed_ms + (10 - b.remaining_consumers) * 10000
                scored.append((score, b))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Select until target met
        selected = []
        freed = 0
        for _score, buf in scored:
            selected.append(buf.key)
            freed += buf.size_bytes
            if freed >= target_free_bytes:
                break

        return selected

    def to_dict(self) -> dict[str, Any]:
        """Export configuration."""
        return {
            "memory_budget_bytes": self._budget,
            "safety_factor": self._safety,
            "safe_budget_bytes": self._safe_budget,
        }


class MemoryManager:
    """MB-P1-014/015/016: Unified memory manager.

    Coordinates:
        - NativeBufferStore (representation-aware cache)
        - LivenessManager (refcount early release)
        - MaterializationDecision (policy)
        - StreamingCoordinator (low-memory streaming)
    """

    def __init__(
        self,
        *,
        memory_budget_bytes: int,
        safety_factor: float = 0.7,
    ) -> None:
        self._buffer_store = NativeBufferStore()
        self._liveness = LivenessManager(self._buffer_store)
        self._streaming = StreamingCoordinator(
            memory_budget_bytes=memory_budget_bytes,
            safety_factor=safety_factor,
        )
        self._budget = memory_budget_bytes

    def allocate_buffer(
        self,
        *,
        key: BufferKey,
        buffer: Any,
        size_bytes: int,
        consumer_count: int,
        reuse_count: int,
        recompute_cost_ms: float,
        is_final_sink: bool = False,
    ) -> MaterializationPolicy:
        """Allocate buffer with policy decision.

        Args:
            key: Buffer key
            buffer: Buffer data
            size_bytes: Size in bytes
            consumer_count: Number of consumers
            reuse_count: Total reuse count
            recompute_cost_ms: Recompute cost
            is_final_sink: Is final output

        Returns:
            Selected materialization policy
        """
        memory_pressure = self._buffer_store.total_live_bytes() / self._budget

        policy = MaterializationDecision.decide(
            reuse_count=reuse_count,
            recompute_cost_ms=recompute_cost_ms,
            size_bytes=size_bytes,
            memory_pressure=memory_pressure,
            is_final_sink=is_final_sink,
        )

        if policy == MaterializationPolicy.KEEP_NATIVE_MEMORY:
            # Check if need preemptive spill
            if self._streaming.should_preemptive_spill(
                current_live_bytes=self._buffer_store.total_live_bytes(),
                incoming_bytes=size_bytes,
            ):
                # Spill instead
                policy = MaterializationPolicy.SPILL_ARROW

        if policy == MaterializationPolicy.KEEP_NATIVE_MEMORY:
            self._buffer_store.store(key, buffer, size_bytes, consumer_count, policy)

        return policy

    def get_buffer(self, key: BufferKey) -> Any | None:
        """Retrieve buffer."""
        return self._buffer_store.get(key)

    def notify_consumed(self, key: BufferKey) -> int:
        """Notify buffer consumed. Returns freed bytes if released."""
        should_release = self._buffer_store.mark_consumed(key)
        if should_release:
            return self._buffer_store.release(key)
        return 0

    def register_dependencies(self, node_id: str, consumer_ids: list[str]) -> None:
        """Register node dependencies."""
        self._liveness.register_dependencies(node_id, consumer_ids)

    def notify_node_complete(self, node_id: str) -> int:
        """Notify node completion. Returns total freed bytes."""
        releasable = self._liveness.notify_complete(node_id)
        freed = 0
        for parent_id in releasable:
            freed += self._liveness.release_node(parent_id)
        return freed

    def should_stream(self, estimated_peak_bytes: int, operator: str) -> bool:
        """Check if should stream."""
        return self._streaming.should_stream(
            estimated_peak_bytes=estimated_peak_bytes,
            operator=operator,
        )

    def adaptive_batch_size(self, default_batch: int = 100) -> int:
        """Get adaptive batch size."""
        return self._streaming.adaptive_batch_size(
            current_live_bytes=self._buffer_store.total_live_bytes(),
            default_batch=default_batch,
        )

    def total_live_bytes(self) -> int:
        """Total live memory."""
        return self._buffer_store.total_live_bytes()

    def buffer_count(self) -> int:
        """Buffer count."""
        return self._buffer_store.count()

    def to_dict(self) -> dict[str, Any]:
        """Export state."""
        return {
            "memory_budget_bytes": self._budget,
            "total_live_bytes": self.total_live_bytes(),
            "buffer_count": self.buffer_count(),
            "buffer_store": self._buffer_store.to_dict(),
            "streaming_config": self._streaming.to_dict(),
        }
