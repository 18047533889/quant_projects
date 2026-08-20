# -*- coding: utf-8 -*-
"""MB-P1-015: Materialization decision policy.

Replaces binary "extract if benefit > 0" with runtime-adaptive policy that chooses:
  - KEEP_NATIVE_MEMORY: Materialize in memory, keep native representation
  - RECOMPUTE: Don't materialize, recompute on each use
  - SPILL_ARROW: Materialize and spill to disk as Arrow
  - SPILL_PARQUET: Materialize and spill to disk as Parquet
  - DIRECT_SINK: Stream directly to final sink, don't materialize

Decision considers:
  - Memory pressure
  - Recompute cost vs materialization cost
  - Consumer count
  - Buffer size
  - Spill disk bandwidth
  - Memory budget headroom

MB-P1-016 integration: In low-memory mode, prefer streaming/recompute over materialization.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MaterializationStrategy(str, Enum):
    """Materialization strategies for CSE nodes."""

    KEEP_NATIVE_MEMORY = "keep_native_memory"  # Default: keep in RAM
    RECOMPUTE = "recompute"  # Don't materialize, compute on demand
    SPILL_ARROW = "spill_arrow"  # Spill to disk as Arrow
    SPILL_PARQUET = "spill_parquet"  # Spill to disk as Parquet
    DIRECT_SINK = "direct_sink"  # Stream to final sink, no intermediate
    STREAMING = "streaming"  # MB-P1-016: Stream through, don't materialize


@dataclass(frozen=True)
class MaterializationDecision:
    """A materialization decision for a single node."""

    node_id: str
    strategy: MaterializationStrategy
    reason: str
    estimated_memory_bytes: int
    recompute_cost_ms: float
    materialize_cost_ms: float
    consumer_count: int
    spill_cost_ms: float = 0.0
    reload_cost_ms: float = 0.0


@dataclass(frozen=True)
class MemoryPressure:
    """Current memory pressure state."""

    current_bytes: int
    budget_bytes: int
    utilization: float  # 0.0 - 1.0
    pressure_stage: str  # "normal" | "elevated" | "critical"

    @property
    def has_headroom(self) -> bool:
        return self.utilization < 0.7

    @property
    def is_critical(self) -> bool:
        return self.utilization > 0.9


class MaterializationPolicyEngine:
    """MB-P1-015: Runtime materialization decision engine.

    Makes adaptive decisions based on:
      - Current memory pressure
      - Node characteristics (size, recompute cost, consumers)
      - Disk I/O capacity
      - Execution mode (streaming-first vs memory-first)
    """

    def __init__(
        self,
        *,
        memory_budget_bytes: int = 1_000_000_000,
        streaming_first: bool = False,  # MB-P1-016
        spill_bandwidth_mbps: float = 200.0,
        reload_bandwidth_mbps: float = 300.0,
    ) -> None:
        self.memory_budget_bytes = memory_budget_bytes
        self.streaming_first = streaming_first
        self.spill_bandwidth_mbps = spill_bandwidth_mbps
        self.reload_bandwidth_mbps = reload_bandwidth_mbps

        # Thresholds
        self.min_benefit_for_materialize_ms = 5.0
        self.min_consumers_for_spill = 2
        self.large_buffer_threshold_mb = 100.0

    def decide(
        self,
        node_id: str,
        estimated_bytes: int,
        recompute_cost_ms: float,
        consumer_count: int,
        current_pressure: MemoryPressure,
        is_final_sink: bool = False,
    ) -> MaterializationDecision:
        """Make materialization decision for a node.

        Args:
            node_id: Node identifier
            estimated_bytes: Estimated buffer size
            recompute_cost_ms: Cost to recompute this node
            consumer_count: Number of consumers
            current_pressure: Current memory pressure
            is_final_sink: Whether this goes directly to final sink

        Returns:
            MaterializationDecision
        """
        # MB-P1-016: Streaming-first mode for low-memory environments
        if self.streaming_first:
            return self._decide_streaming_first(
                node_id, estimated_bytes, recompute_cost_ms,
                consumer_count, current_pressure, is_final_sink
            )

        # Normal mode: memory-first with adaptive spill
        return self._decide_memory_first(
            node_id, estimated_bytes, recompute_cost_ms,
            consumer_count, current_pressure, is_final_sink
        )

    def _decide_streaming_first(
        self,
        node_id: str,
        estimated_bytes: int,
        recompute_cost_ms: float,
        consumer_count: int,
        current_pressure: MemoryPressure,
        is_final_sink: bool,
    ) -> MaterializationDecision:
        """MB-P1-016: Streaming-first strategy for low-memory environments.

        Prefers:
          1. Direct sink for single-consumer final nodes
          2. Streaming execution when possible
          3. Recompute for cheap operations
          4. Spill only for expensive multi-consumer nodes
          5. Memory as last resort
        """
        materialize_cost = self._estimate_materialize_cost(estimated_bytes)

        # Rule 1: Direct sink for final single-consumer nodes
        if is_final_sink and consumer_count == 1:
            return MaterializationDecision(
                node_id=node_id,
                strategy=MaterializationStrategy.DIRECT_SINK,
                reason="final sink, single consumer, streaming-first mode",
                estimated_memory_bytes=0,
                recompute_cost_ms=0.0,
                materialize_cost_ms=0.0,
                consumer_count=consumer_count,
            )

        # Rule 2: Single consumer -> prefer streaming/recompute
        if consumer_count == 1:
            if recompute_cost_ms < self.min_benefit_for_materialize_ms:
                return MaterializationDecision(
                    node_id=node_id,
                    strategy=MaterializationStrategy.STREAMING,
                    reason="single consumer, cheap recompute, streaming-first",
                    estimated_memory_bytes=0,
                    recompute_cost_ms=recompute_cost_ms,
                    materialize_cost_ms=0.0,
                    consumer_count=consumer_count,
                )

        # Rule 3: Multi-consumer, cheap recompute -> still recompute in streaming mode
        if consumer_count <= 3 and recompute_cost_ms < 10.0:
            return MaterializationDecision(
                node_id=node_id,
                strategy=MaterializationStrategy.RECOMPUTE,
                reason=f"{consumer_count} consumers, cheap recompute, streaming-first",
                estimated_memory_bytes=0,
                recompute_cost_ms=recompute_cost_ms * consumer_count,
                materialize_cost_ms=0.0,
                consumer_count=consumer_count,
            )

        # Rule 4: Expensive multi-consumer -> spill
        total_recompute_cost = recompute_cost_ms * consumer_count
        spill_cost = self._estimate_spill_cost(estimated_bytes)
        reload_cost = self._estimate_reload_cost(estimated_bytes) * consumer_count

        if spill_cost + reload_cost < total_recompute_cost:
            strategy = (
                MaterializationStrategy.SPILL_PARQUET
                if estimated_bytes > 10_000_000
                else MaterializationStrategy.SPILL_ARROW
            )
            return MaterializationDecision(
                node_id=node_id,
                strategy=strategy,
                reason=f"expensive recompute ({recompute_cost_ms:.1f}ms × {consumer_count}), "
                       f"spill cheaper, streaming-first",
                estimated_memory_bytes=0,  # Spilled, not in memory
                recompute_cost_ms=0.0,
                materialize_cost_ms=materialize_cost,
                consumer_count=consumer_count,
                spill_cost_ms=spill_cost,
                reload_cost_ms=reload_cost,
            )

        # Rule 5: If memory available and really expensive, keep in memory
        if not current_pressure.is_critical:
            total_benefit = total_recompute_cost - materialize_cost
            if total_benefit > 50.0:  # High threshold in streaming mode
                return MaterializationDecision(
                    node_id=node_id,
                    strategy=MaterializationStrategy.KEEP_NATIVE_MEMORY,
                    reason=f"very expensive recompute ({total_benefit:.1f}ms benefit), "
                           f"memory available, streaming-first exception",
                    estimated_memory_bytes=estimated_bytes,
                    recompute_cost_ms=0.0,
                    materialize_cost_ms=materialize_cost,
                    consumer_count=consumer_count,
                )

        # Default: recompute
        return MaterializationDecision(
            node_id=node_id,
            strategy=MaterializationStrategy.RECOMPUTE,
            reason="streaming-first default",
            estimated_memory_bytes=0,
            recompute_cost_ms=recompute_cost_ms * consumer_count,
            materialize_cost_ms=0.0,
            consumer_count=consumer_count,
        )

    def _decide_memory_first(
        self,
        node_id: str,
        estimated_bytes: int,
        recompute_cost_ms: float,
        consumer_count: int,
        current_pressure: MemoryPressure,
        is_final_sink: bool,
    ) -> MaterializationDecision:
        """Memory-first strategy: materialize in memory when beneficial, spill when necessary."""
        materialize_cost = self._estimate_materialize_cost(estimated_bytes)

        # Single consumer: usually not worth materializing
        if consumer_count == 1 and not is_final_sink:
            if recompute_cost_ms < materialize_cost:
                return MaterializationDecision(
                    node_id=node_id,
                    strategy=MaterializationStrategy.RECOMPUTE,
                    reason=f"single consumer, recompute cheaper ({recompute_cost_ms:.1f} < {materialize_cost:.1f})",
                    estimated_memory_bytes=0,
                    recompute_cost_ms=recompute_cost_ms,
                    materialize_cost_ms=materialize_cost,
                    consumer_count=consumer_count,
                )

        # Calculate benefit of materialization
        total_recompute_cost = recompute_cost_ms * consumer_count
        benefit = total_recompute_cost - materialize_cost

        # Not enough benefit -> recompute
        if benefit < self.min_benefit_for_materialize_ms:
            return MaterializationDecision(
                node_id=node_id,
                strategy=MaterializationStrategy.RECOMPUTE,
                reason=f"benefit too low ({benefit:.1f}ms < threshold {self.min_benefit_for_materialize_ms})",
                estimated_memory_bytes=0,
                recompute_cost_ms=total_recompute_cost,
                materialize_cost_ms=materialize_cost,
                consumer_count=consumer_count,
            )

        # Check if we can fit in memory
        would_fit = (current_pressure.current_bytes + estimated_bytes
                     <= self.memory_budget_bytes * 0.95)

        if would_fit and not current_pressure.is_critical:
            # Keep in memory
            return MaterializationDecision(
                node_id=node_id,
                strategy=MaterializationStrategy.KEEP_NATIVE_MEMORY,
                reason=f"beneficial ({benefit:.1f}ms), fits in memory",
                estimated_memory_bytes=estimated_bytes,
                recompute_cost_ms=0.0,
                materialize_cost_ms=materialize_cost,
                consumer_count=consumer_count,
            )

        # Memory pressure: consider spill vs recompute
        spill_cost = self._estimate_spill_cost(estimated_bytes)
        reload_cost = self._estimate_reload_cost(estimated_bytes) * consumer_count
        spill_total = materialize_cost + spill_cost + reload_cost

        if spill_total < total_recompute_cost and consumer_count >= self.min_consumers_for_spill:
            # Spill to disk
            strategy = (
                MaterializationStrategy.SPILL_PARQUET
                if estimated_bytes > 10_000_000
                else MaterializationStrategy.SPILL_ARROW
            )
            return MaterializationDecision(
                node_id=node_id,
                strategy=strategy,
                reason=f"memory pressure, spill cheaper than recompute "
                       f"({spill_total:.1f} < {total_recompute_cost:.1f})",
                estimated_memory_bytes=0,
                recompute_cost_ms=0.0,
                materialize_cost_ms=materialize_cost,
                consumer_count=consumer_count,
                spill_cost_ms=spill_cost,
                reload_cost_ms=reload_cost,
            )

        # Fallback: recompute
        return MaterializationDecision(
            node_id=node_id,
            strategy=MaterializationStrategy.RECOMPUTE,
            reason=f"memory pressure, spill not worthwhile",
            estimated_memory_bytes=0,
            recompute_cost_ms=total_recompute_cost,
            materialize_cost_ms=materialize_cost,
            consumer_count=consumer_count,
        )

    def _estimate_materialize_cost(self, bytes_: int) -> float:
        """Estimate cost to materialize a buffer in memory."""
        mb = bytes_ / 1_000_000.0
        # Rough model: 1ms per MB to materialize
        return max(1.0, mb * 1.0)

    def _estimate_spill_cost(self, bytes_: int) -> float:
        """Estimate cost to spill buffer to disk."""
        mb = bytes_ / 1_000_000.0
        return max(1.0, mb / self.spill_bandwidth_mbps * 1000.0)

    def _estimate_reload_cost(self, bytes_: int) -> float:
        """Estimate cost to reload buffer from disk."""
        mb = bytes_ / 1_000_000.0
        return max(1.0, mb / self.reload_bandwidth_mbps * 1000.0)


def decide_batch_materialization(
    nodes: dict[str, dict[str, Any]],
    current_memory_bytes: int,
    memory_budget_bytes: int,
    streaming_first: bool = False,
) -> dict[str, MaterializationDecision]:
    """MB-P1-015: Batch materialization policy for all CSE nodes.

    Args:
        nodes: Map of node_id -> node_info dict with keys:
            - estimated_bytes
            - recompute_cost_ms
            - consumer_count
            - is_final_sink (optional)
        current_memory_bytes: Current memory usage
        memory_budget_bytes: Total memory budget
        streaming_first: MB-P1-016 streaming-first mode

    Returns:
        Map of node_id -> MaterializationDecision
    """
    utilization = current_memory_bytes / max(1, memory_budget_bytes)
    if utilization < 0.7:
        pressure_stage = "normal"
    elif utilization < 0.9:
        pressure_stage = "elevated"
    else:
        pressure_stage = "critical"

    pressure = MemoryPressure(
        current_bytes=current_memory_bytes,
        budget_bytes=memory_budget_bytes,
        utilization=utilization,
        pressure_stage=pressure_stage,
    )

    engine = MaterializationPolicyEngine(
        memory_budget_bytes=memory_budget_bytes,
        streaming_first=streaming_first,
    )

    decisions: dict[str, MaterializationDecision] = {}
    for node_id, info in nodes.items():
        decision = engine.decide(
            node_id=node_id,
            estimated_bytes=info.get("estimated_bytes", 0),
            recompute_cost_ms=info.get("recompute_cost_ms", 0.0),
            consumer_count=info.get("consumer_count", 1),
            current_pressure=pressure,
            is_final_sink=info.get("is_final_sink", False),
        )
        decisions[node_id] = decision

    return decisions
