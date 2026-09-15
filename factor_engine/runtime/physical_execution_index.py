"""Run-local immutable metadata index for physical-plan execution."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from factor_engine.planner.backend_region import PhysicalBackend, Representation


@dataclass(frozen=True)
class PhysicalExecutionIndex:
    plan: Any
    regions: Mapping[str, Any]
    node_regions: Mapping[str, str]
    incoming_regions: Mapping[str, frozenset[str]]
    edges_by_pair: Mapping[tuple[str, str], Any]
    topological_order: tuple[str, ...]
    topological_positions: Mapping[str, int]


def build_physical_execution_index(plan: Any) -> PhysicalExecutionIndex:
    regions = {region.region_id: region for region in plan.regions}
    if not regions:
        raise ValueError("production batch requires a BackendRegion")
    if len(regions) != len(plan.regions) or tuple(regions) == ():
        raise ValueError("physical batch has invalid topology")
    if len(plan.topological_order) != len(regions) or set(plan.topological_order) != set(regions):
        raise ValueError("physical batch has invalid topology")
    if not plan.root_region_ids or not set(plan.root_region_ids).issubset(regions):
        raise ValueError("physical batch has invalid root regions")
    node_regions: dict[str, str] = {}
    for region in plan.regions:
        for node_id in region.node_ids:
            if node_id in node_regions:
                raise ValueError(f"logical node {node_id!r} belongs to multiple BackendRegions")
            node_regions[node_id] = region.region_id
    order = {region_id: i for i, region_id in enumerate(plan.topological_order)}
    edge_ids: set[str] = set()
    edges_by_pair: dict[tuple[str, str], Any] = {}
    incoming: dict[str, set[str]] = {}
    for edge in plan.edges:
        if edge.edge_id in edge_ids:
            raise ValueError(f"duplicate TransferEdge ID {edge.edge_id!r}")
        edge_ids.add(edge.edge_id)
        producer = regions.get(edge.producer_region)
        consumer = regions.get(edge.consumer_region)
        if producer is None or consumer is None:
            raise ValueError(f"TransferEdge {edge.edge_id!r} references an unknown region")
        pair = (edge.producer_region, edge.consumer_region)
        if pair in edges_by_pair:
            raise ValueError("multiple TransferEdges between one region pair are ambiguous")
        edges_by_pair[pair] = edge
        incoming.setdefault(edge.consumer_region, set()).add(edge.producer_region)
        if order[edge.producer_region] >= order[edge.consumer_region]:
            raise ValueError(f"TransferEdge {edge.edge_id!r} violates topological order")
        source_matches = edge.source_backend == producer.backend and edge.source_representation == producer.representation
        allowed_boundary = edge.source_backend == PhysicalBackend.DUCKDB_SQL and edge.source_representation == Representation.ARROW_TABLE
        if not (source_matches or allowed_boundary) or edge.target_backend != consumer.backend or edge.target_representation != consumer.representation:
            raise ValueError(f"TransferEdge {edge.edge_id!r} backend residency is malformed")
    return PhysicalExecutionIndex(
        plan, MappingProxyType(regions), MappingProxyType(node_regions),
        MappingProxyType({k: frozenset(v) for k, v in incoming.items()}),
        MappingProxyType(edges_by_pair), tuple(plan.topological_order),
        MappingProxyType(order),
    )


class PhysicalExecutionIndexCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._plan: Any = None
        self._index: PhysicalExecutionIndex | None = None
        self._orphan_validated_plan: Any = None
        self.build_count = 0

    def get(self, plan: Any) -> PhysicalExecutionIndex:
        with self._lock:
            if self._plan is not plan or self._index is None:
                index = build_physical_execution_index(plan)
                self._index = index
                self._plan = plan
                self.build_count += 1
            return self._index

    def validate_orphans(
        self, plan: Any, index: PhysicalExecutionIndex, logical_root_ids: tuple[str, ...]
    ) -> None:
        with self._lock:
            if self._orphan_validated_plan is plan:
                return
            reachable = set(plan.root_region_ids)
            pending = list(reachable)
            while pending:
                consumer = pending.pop()
                for producer in index.incoming_regions.get(consumer, ()):
                    if producer not in reachable:
                        reachable.add(producer)
                        pending.append(producer)
            orphan_regions = set(index.regions).difference(reachable)
            if orphan_regions:
                raise ValueError(
                    "BackendRegion must have exactly one materialized output reachable "
                    f"from a declared batch root; orphan regions={sorted(orphan_regions)!r}"
                )
            self._orphan_validated_plan = plan
