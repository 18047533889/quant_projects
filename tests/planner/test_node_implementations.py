# -*- coding: utf-8 -*-
"""R21-NODE-IMPL regression tests.

BackendRegion must carry per-node implementation IDs via node_implementations
dict, with automatic RegionImplementationManifestHash computation.  This module
verifies:
1. Default empty node_implementations produces empty implementation_id
2. node_implementations auto-populates implementation_id with manifest hash
3. Manifest hash is deterministic and order-independent
4. Consistency check between implementation_id and computed hash
"""
from __future__ import annotations

import hashlib
import json

import pytest

from planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalRegionPlan,
    Representation,
    StateContract,
    TransferEdge,
)


def _region(**overrides: object) -> BackendRegion:
    base: dict[str, object] = {
        "region_id": "r1",
        "backend": PhysicalBackend.POLARS_PANEL,
        "representation": Representation.POLARS_LONG,
        "node_ids": ("n1", "n2"),
        "execution_axis": ExecutionAxis.TIME_PER_INSTRUMENT,
        "estimated_rows": 1000,
        "estimated_compute_ms": 10.0,
        "estimated_memory_bytes": 8000,
    }
    base.update(overrides)
    return BackendRegion(**base)  # type: ignore[arg-type]


def _edge() -> TransferEdge:
    return TransferEdge(
        edge_id="e1",
        producer_region="r1",
        consumer_region="r2",
        source_backend=PhysicalBackend.POLARS_PANEL,
        target_backend=PhysicalBackend.DUCKDB_SQL,
        source_representation=Representation.POLARS_LONG,
        target_representation=Representation.DUCKDB_RELATION,
        estimated_rows=1000,
        estimated_bytes=16000,
        estimated_transfer_ms=5.0,
    )


def _plan(regions: tuple[BackendRegion, ...], edges: tuple[TransferEdge, ...]) -> PhysicalRegionPlan:
    return PhysicalRegionPlan(
        plan_id="test_plan",
        regions=regions,
        edges=edges,
        topological_order=tuple(r.region_id for r in regions),
        root_region_ids=(regions[-1].region_id,),
        total_compute_ms=10.0,
        total_transfer_ms=5.0 if edges else 0.0,
        backend_switch_count=sum(
            e.source_backend != e.target_backend for e in edges
        ),
    )


class TestNodeImplementationsBasic:
    """R21-NODE-IMPL: basic node_implementations functionality."""

    def test_default_empty_node_implementations(self) -> None:
        """Empty node_implementations produces empty implementation_id."""
        r = _region()
        assert r.node_implementations == {}
        assert r.implementation_id == ""

    def test_node_implementations_auto_populates_implementation_id(self) -> None:
        """When node_implementations provided, implementation_id is auto-set to manifest hash."""
        pi1 = "pi:v1:abc123"
        pi2 = "pi:v1:def456"
        r = _region(
            node_implementations={"n1": pi1, "n2": pi2},
        )
        # Implementation ID should be auto-populated
        assert r.implementation_id.startswith("rimh:v1:")
        # Should match manual computation
        expected = BackendRegion.compute_region_implementation_manifest_hash({"n1": pi1, "n2": pi2})
        assert r.implementation_id == expected

    def test_manifest_hash_deterministic(self) -> None:
        """Same node_implementations always produces same hash."""
        pi1 = "pi:v1:abc123"
        pi2 = "pi:v1:def456"
        h1 = BackendRegion.compute_region_implementation_manifest_hash({"n1": pi1, "n2": pi2})
        h2 = BackendRegion.compute_region_implementation_manifest_hash({"n1": pi1, "n2": pi2})
        assert h1 == h2

    def test_manifest_hash_order_independent(self) -> None:
        """Dict insertion order does not affect manifest hash."""
        pi1 = "pi:v1:abc123"
        pi2 = "pi:v1:def456"
        h1 = BackendRegion.compute_region_implementation_manifest_hash({"n1": pi1, "n2": pi2})
        h2 = BackendRegion.compute_region_implementation_manifest_hash({"n2": pi2, "n1": pi1})
        assert h1 == h2

    def test_manifest_hash_format(self) -> None:
        """Manifest hash has correct prefix format."""
        h = BackendRegion.compute_region_implementation_manifest_hash({"n1": "pi:v1:abc"})
        assert h.startswith("rimh:v1:")
        assert len(h) == len("rimh:v1:") + 64  # SHA256 hex digest

    def test_empty_node_implementations_hash(self) -> None:
        """Empty dict returns empty string."""
        h = BackendRegion.compute_region_implementation_manifest_hash({})
        assert h == ""


class TestNodeImplementationsConsistency:
    """R21-NODE-IMPL: consistency between implementation_id and computed hash."""

    def test_consistent_implementation_id_accepted(self) -> None:
        """Provided implementation_id matching computed hash is accepted."""
        pi1 = "pi:v1:abc123"
        pi2 = "pi:v1:def456"
        expected = BackendRegion.compute_region_implementation_manifest_hash({"n1": pi1, "n2": pi2})
        r = _region(
            node_implementations={"n1": pi1, "n2": pi2},
            implementation_id=expected,
        )
        assert r.implementation_id == expected

    def test_inconsistent_implementation_id_rejected(self) -> None:
        """Provided implementation_id not matching computed hash raises ValueError."""
        pi1 = "pi:v1:abc123"
        pi2 = "pi:v1:def456"
        with pytest.raises(ValueError, match="does not match"):
            _region(
                node_implementations={"n1": pi1, "n2": pi2},
                implementation_id="rimh:v1:bad_hash",
            )


class TestNodeImplementationsToDict:
    """R21-NODE-IMPL: to_dict includes node implementation fields."""

    def test_to_dict_includes_node_implementations(self) -> None:
        """to_dict() includes node_implementations and manifest hash."""
        pi1 = "pi:v1:abc123"
        pi2 = "pi:v1:def456"
        r = _region(node_implementations={"n1": pi1, "n2": pi2})
        d = r.to_dict()
        assert "node_implementations" in d
        assert d["node_implementations"] == {"n1": pi1, "n2": pi2}
        assert "region_implementation_manifest_hash" in d
        assert d["region_implementation_manifest_hash"].startswith("rimh:v1:")


class TestNodeImplementationsPlanHash:
    """R21-NODE-IMPL: plan hash includes node_implementations in identity."""

    def test_plan_hash_differs_with_different_node_implementations(self) -> None:
        """Plans with different node_implementations have different hashes."""
        pi1 = "pi:v1:abc123"
        pi2 = "pi:v1:def456"
        pi3 = "pi:v1:ghi789"

        r1_a = _region(region_id="r1", node_implementations={"n1": pi1})
        r1_b = _region(region_id="r1", node_implementations={"n1": pi2})
        r2 = _region(
            region_id="r2",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_implementations={"n3": pi3},
        )

        e1_a = TransferEdge(
            edge_id="e1", producer_region="r1", consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL, target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG, target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=1000, estimated_bytes=16000, estimated_transfer_ms=5.0,
        )

        plan_a = _plan((r1_a, r2), (e1_a,))
        plan_b = _plan((r1_b, r2), (e1_a,))

        h1 = PhysicalRegionPlan.compute_plan_hash(plan_a.regions, plan_a.edges, "logical_hash")
        h2 = PhysicalRegionPlan.compute_plan_hash(plan_b.regions, plan_b.edges, "logical_hash")
        assert h1 != h2

    def test_plan_hash_same_for_identical_node_implementations(self) -> None:
        """Plans with identical node_implementations have same hashes."""
        pi1 = "pi:v1:abc123"
        pi2 = "pi:v1:def456"

        r1_a = _region(region_id="r1", node_implementations={"n1": pi1, "n2": pi2})
        r1_b = _region(region_id="r1", node_implementations={"n1": pi1, "n2": pi2})
        r2 = _region(
            region_id="r2",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
        )

        e1 = TransferEdge(
            edge_id="e1", producer_region="r1", consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL, target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG, target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=1000, estimated_bytes=16000, estimated_transfer_ms=5.0,
        )

        plan_a = _plan((r1_a, r2), (e1,))
        plan_b = _plan((r1_b, r2), (e1,))

        h1 = PhysicalRegionPlan.compute_plan_hash(plan_a.regions, plan_a.edges, "logical_hash")
        h2 = PhysicalRegionPlan.compute_plan_hash(plan_b.regions, plan_b.edges, "logical_hash")
        assert h1 == h2
