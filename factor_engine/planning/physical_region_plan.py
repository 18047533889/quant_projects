# -*- coding: utf-8 -*-
"""Physical region plan: Complete data structure for multi-backend execution.

Implements the PhysicalRegionPlan as specified in section 10 of the
remediation document, representing the complete executable plan with
explicit backend regions, transfer edges, and topological ordering.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from planning.backend_region import BackendRegion
from planning.transfer_edge import TransferEdge


@dataclass(frozen=True)
class PhysicalRegionPlan:
    """Complete physical execution plan with explicit backend regions (section 10).

    This replaces the single-backend PlanRoute with a true multi-region
    execution plan. Each logical node is assigned to exactly one region,
    and all cross-region data flows are explicit transfer edges.

    Attributes:
        regions: All backend regions in this plan
        edges: All transfer edges between regions
        topological_order: Region execution order (respects dependencies)
        peak_memory_estimate: Estimated peak memory usage (bytes)
        estimated_ttdc_ms: Estimated time-to-durable-commit (milliseconds)
        plan_hash: Stable hash of this physical plan
        logical_node_count: Number of logical plan nodes
        shared_node_count: Number of nodes used by multiple consumers
        metadata: Additional telemetry and explain metadata
    """

    regions: tuple[BackendRegion, ...]
    edges: tuple[TransferEdge, ...]
    topological_order: tuple[str, ...]
    peak_memory_estimate: int
    estimated_ttdc_ms: float
    plan_hash: str
    logical_node_count: int = 0
    shared_node_count: int = 0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate plan consistency."""
        # Every logical node assigned exactly once (MB-P0-004 resolution)
        region_map = {rid: r for r in self.regions for rid in (r.region_id,)}
        all_nodes: set[str] = set()
        for region in self.regions:
            for node_id in region.node_ids:
                if node_id in all_nodes:
                    raise ValueError(
                        f"Node {node_id} assigned to multiple regions "
                        "(violates MB-P0-004: each node must be assigned exactly once)"
                    )
                all_nodes.add(node_id)

        # All edges reference valid regions
        for edge in self.edges:
            if edge.producer_region not in region_map:
                raise ValueError(
                    f"Edge {edge.edge_id} references unknown producer {edge.producer_region}"
                )
            if edge.consumer_region not in region_map:
                raise ValueError(
                    f"Edge {edge.edge_id} references unknown consumer {edge.consumer_region}"
                )

        # Topological order includes all regions
        if set(self.topological_order) != set(r.region_id for r in self.regions):
            raise ValueError("Topological order must include all regions exactly once")

    @property
    def region_count(self) -> int:
        """Number of backend regions in this plan."""
        return len(self.regions)

    @property
    def backend_switch_count(self) -> int:
        """Number of backend switches (cross-backend edges)."""
        count = 0
        region_map = {r.region_id: r for r in self.regions}
        for edge in self.edges:
            producer = region_map.get(edge.producer_region)
            consumer = region_map.get(edge.consumer_region)
            if producer and consumer and producer.backend != consumer.backend:
                count += 1
        return count

    @property
    def total_transfer_cost_ms(self) -> float:
        """Total cost of all transfer edges in milliseconds."""
        return sum(edge.estimated_cost_ms for edge in self.edges)

    def get_region_by_id(self, region_id: str) -> BackendRegion | None:
        """Retrieve region by ID."""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None

    def get_edges_for_region(self, region_id: str, *, as_consumer: bool = True) -> list[TransferEdge]:
        """Get all edges where this region is consumer (incoming) or producer (outgoing)."""
        if as_consumer:
            return [e for e in self.edges if e.consumer_region == region_id]
        return [e for e in self.edges if e.producer_region == region_id]

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan to dictionary for telemetry/explain (section 71)."""
        return {
            "plan_hash": self.plan_hash,
            "region_count": self.region_count,
            "edge_count": len(self.edges),
            "backend_switch_count": self.backend_switch_count,
            "logical_node_count": self.logical_node_count,
            "shared_node_count": self.shared_node_count,
            "peak_memory_estimate_mb": round(self.peak_memory_estimate / 1024.0 / 1024.0, 2),
            "estimated_ttdc_ms": round(self.estimated_ttdc_ms, 3),
            "total_transfer_cost_ms": round(self.total_transfer_cost_ms, 3),
            "regions": [r.to_dict() for r in self.regions],
            "edges": [e.to_dict() for e in self.edges],
            "topological_order": list(self.topological_order),
            "metadata": self.metadata or {},
        }

    def explain(self) -> str:
        """Generate human-readable explanation of this plan (section 71).

        Answers key questions from section 117:
        - How many regions?
        - Which backend for each region, and why?
        - How many backend switches?
        - How much data transferred at each boundary?
        - Peak memory usage?
        """
        lines = [
            "=" * 70,
            "Physical Region Plan Explanation",
            "=" * 70,
            f"Plan Hash: {self.plan_hash[:16]}...",
            f"Logical Nodes: {self.logical_node_count} ({self.shared_node_count} shared)",
            f"Backend Regions: {self.region_count}",
            f"Backend Switches: {self.backend_switch_count}",
            f"Peak Memory: {self.peak_memory_estimate / 1024.0 / 1024.0:.1f} MB",
            f"Estimated TTDC: {self.estimated_ttdc_ms:.1f} ms",
            f"Total Transfer Cost: {self.total_transfer_cost_ms:.1f} ms",
            "",
            "Region Execution Order:",
            "-" * 70,
        ]

        region_map = {r.region_id: r for r in self.regions}
        for i, region_id in enumerate(self.topological_order, 1):
            region = region_map.get(region_id)
            if not region:
                continue
            lines.append(
                f"{i}. Region {region.region_id} → {region.backend.value}"
            )
            lines.append(f"   Nodes: {len(region.node_ids)}")
            lines.append(f"   Axis: {region.execution_axis.value}")
            lines.append(f"   Rows: ~{region.estimated_rows:,}")
            lines.append(f"   Memory: ~{region.estimated_bytes / 1024.0 / 1024.0:.1f} MB")
            if region.can_stream:
                lines.append("   Streaming: Yes")
            if region.required_properties.sorted_by:
                lines.append(f"   Sorted by: {', '.join(region.required_properties.sorted_by)}")
            lines.append("")

        if self.edges:
            lines.append("Transfer Edges:")
            lines.append("-" * 70)
            for edge in self.edges:
                producer = region_map.get(edge.producer_region)
                consumer = region_map.get(edge.consumer_region)
                if not producer or not consumer:
                    continue
                lines.append(
                    f"{edge.producer_region} → {edge.consumer_region}"
                )
                lines.append(f"   {edge.source_representation.value} → {edge.target_representation.value}")
                lines.append(f"   Rows: ~{edge.estimated_rows:,}")
                lines.append(f"   Bytes: ~{edge.estimated_bytes / 1024.0 / 1024.0:.1f} MB")
                lines.append(f"   Cost: {edge.estimated_cost_ms:.1f} ms")
                flags = []
                if edge.requires_sort:
                    flags.append("SORT")
                if edge.requires_repartition:
                    flags.append("REPARTITION")
                if edge.requires_reshape:
                    flags.append("RESHAPE")
                if flags:
                    lines.append(f"   Flags: {', '.join(flags)}")
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def compute_plan_hash(
    regions: tuple[BackendRegion, ...],
    edges: tuple[TransferEdge, ...],
    logical_hash: str = "",
    node_implementations: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Compute stable hash of physical plan (MB-P2-014, section 68).

    Hash includes:
    - Logical structural hash (if provided)
    - Region assignments (node_id -> backend)
    - Representation choices
    - Edge structure
    - Required properties
    - Per-node implementation details (PI, bound_params, accelerator, kernel_signature)

    Does NOT include:
    - Actual runtime measurements
    - Estimated costs (these are predictions, not plan identity)
    - Telemetry metadata
    """
    # Build region payload with node implementations
    region_payload = []
    for r in regions:
        region_dict = {
            "region_id": r.region_id,
            "backend": r.backend.value if hasattr(r.backend, "value") else str(r.backend),
            "representation": r.representation.value if hasattr(r.representation, "value") else str(r.representation),
            "node_ids": sorted(r.node_ids),
            "execution_axis": r.execution_axis.value if hasattr(r.execution_axis, "value") else str(r.execution_axis),
            "sorted_by": list(r.required_properties.sorted_by),
            "partitioned_by": list(r.required_properties.partitioned_by),
        }

        # Add per-node implementation details if provided
        if node_implementations:
            node_details = {}
            for node_id in r.node_ids:
                if node_id in node_implementations:
                    impl = node_implementations[node_id]
                    node_details[node_id] = {
                        "node_id": node_id,
                        "physical_implementation_id": impl.get("physical_implementation_id", ""),
                        "bound_params": impl.get("bound_params", {}),
                        "accelerator": impl.get("accelerator", "none"),
                        "kernel_signature": impl.get("kernel_signature", ""),
                    }
            region_dict["node_implementations"] = node_details

        region_payload.append(region_dict)

    payload: dict[str, Any] = {
        "logical_hash": logical_hash,
        "regions": region_payload,
        "edges": [
            {
                "producer": e.producer_region,
                "consumer": e.consumer_region,
                "source_repr": e.source_representation.value if hasattr(e.source_representation, "value") else str(e.source_representation),
                "target_repr": e.target_representation.value if hasattr(e.target_representation, "value") else str(e.target_representation),
                "requires_sort": e.requires_sort,
                "requires_repartition": e.requires_repartition,
            }
            for e in edges
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def estimate_plan_peak_memory(regions: tuple[BackendRegion, ...], edges: tuple[TransferEdge, ...]) -> int:
    """Estimate peak memory usage across all regions (MB-P1-004, section 47).

    Peak memory must account for:
    - Largest region's memory footprint
    - Transfer edge overlap (source + target + scratch buffers)
    - Multiple concurrent regions (if scheduler allows parallelism)

    Conservative estimate: max single region + max transfer edge overlap.
    """
    if not regions:
        return 0

    # Largest single region
    max_region_bytes = max(r.estimated_bytes for r in regions)

    # Largest transfer edge (source + target + 20% scratch)
    max_edge_bytes = 0
    if edges:
        for edge in edges:
            edge_overlap = int(edge.estimated_bytes * 2.2)  # source + target + scratch
            max_edge_bytes = max(max_edge_bytes, edge_overlap)

    # Conservative peak = max region + max edge
    return max_region_bytes + max_edge_bytes
