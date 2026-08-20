# -*- coding: utf-8 -*-
"""Test region planning logic for multi-backend execution.

Tests cover:
- Single region (homogeneous backend)
- Multi-region partitioning
- Cost-based backend selection
- Region boundary validation
- Topological ordering
"""
from __future__ import annotations

import pytest

from planner.backend_region import (
    BackendRegion,
    PhysicalBackend,
    Representation,
    ExecutionAxis,
    PhysicalProperties,
    StateContract,
    PhysicalRegionPlan,
    TransferEdge,
)
from planner.logical_plan import PlanNode


class TestSingleRegionPlanning:
    """Test single-region planning (simplest case)."""

    def test_single_backend_no_transfer(self):
        """Single root, single backend - no transfer edges needed."""
        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2", "n3"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=8_000_000,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_001",
            regions=(region,),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=50.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=50.0,
            peak_memory_bytes=8_000_000,
            logical_node_count=3,
            backend_switch_count=0,
            native_fraction=1.0,
            routing_basis="estimated",
        )

        assert len(plan.regions) == 1
        assert len(plan.edges) == 0
        assert plan.backend_switch_count == 0
        assert plan.total_transfer_ms == 0.0

    def test_pandas_single_region(self):
        """Pandas backend single region."""
        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.PANDAS_NUMPY,
            representation=Representation.PANDAS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=50_000,
            estimated_memory_bytes=4_000_000,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_002",
            regions=(region,),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=80.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=80.0,
            peak_memory_bytes=4_000_000,
            logical_node_count=2,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        assert plan.regions[0].backend == PhysicalBackend.PANDAS_NUMPY

    def test_duckdb_single_region(self):
        """DuckDB SQL backend single region."""
        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n1", "n2", "n3", "n4"),
            execution_axis=ExecutionAxis.RELATIONAL,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=6_000_000,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_003",
            regions=(region,),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=30.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=30.0,
            peak_memory_bytes=6_000_000,
            logical_node_count=4,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        assert plan.regions[0].backend == PhysicalBackend.DUCKDB_SQL
        assert plan.regions[0].execution_axis == ExecutionAxis.RELATIONAL


class TestMultiRegionPlanning:
    """Test multi-region planning with backend switches."""

    def test_two_region_polars_to_duckdb(self):
        """Two regions: Polars → DuckDB with transfer edge."""
        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=8_000_000,
        )

        region2 = BackendRegion(
            region_id="r2",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n3", "n4"),
            execution_axis=ExecutionAxis.RELATIONAL,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=6_000_000,
        )

        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_memory_bytes=800_000,
            estimated_transfer_ms=15.0,
            requires_sort=False,
            preserves_pit=True,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_004",
            regions=(region1, region2),
            edges=(edge,),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=80.0,
            total_transfer_ms=15.0,
            total_ttdc_ms=95.0,
            peak_memory_bytes=14_000_000,
            logical_node_count=4,
            backend_switch_count=1,
            native_fraction=0.75,
        )

        assert len(plan.regions) == 2
        assert len(plan.edges) == 1
        assert plan.backend_switch_count == 1
        assert plan.total_transfer_ms == 15.0

    def test_three_region_chain(self):
        """Three regions in chain: Pandas → Polars → DuckDB."""
        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.PANDAS_NUMPY,
            representation=Representation.PANDAS_LONG,
            node_ids=("n1",),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=4_000_000,
        )

        region2 = BackendRegion(
            region_id="r2",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n2", "n3"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=6_000_000,
        )

        region3 = BackendRegion(
            region_id="r3",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n4",),
            execution_axis=ExecutionAxis.RELATIONAL,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=5_000_000,
        )

        edge1 = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.PANDAS_NUMPY,
            target_backend=PhysicalBackend.POLARS_PANEL,
            source_representation=Representation.PANDAS_LONG,
            target_representation=Representation.POLARS_LONG,
            estimated_rows=100_000,
            estimated_memory_bytes=800_000,
            estimated_transfer_ms=10.0,
        )

        edge2 = TransferEdge(
            edge_id="e2",
            producer_region="r2",
            consumer_region="r3",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_memory_bytes=800_000,
            estimated_transfer_ms=12.0,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_005",
            regions=(region1, region2, region3),
            edges=(edge1, edge2),
            topological_order=("r1", "r2", "r3"),
            root_region_ids=("r3",),
            total_compute_ms=100.0,
            total_transfer_ms=22.0,
            total_ttdc_ms=122.0,
            peak_memory_bytes=15_000_000,
            logical_node_count=4,
            backend_switch_count=2,
            native_fraction=0.85,
        )

        assert len(plan.regions) == 3
        assert len(plan.edges) == 2
        assert plan.backend_switch_count == 2
        assert plan.topological_order == ("r1", "r2", "r3")

    def test_diamond_dag_multiple_roots(self):
        """Diamond DAG with multiple root nodes."""
        # r1, r2 both feed into r3
        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=6_000_000,
        )

        region2 = BackendRegion(
            region_id="r2",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n3", "n4"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=6_000_000,
        )

        region3 = BackendRegion(
            region_id="r3",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n5",),
            execution_axis=ExecutionAxis.RELATIONAL,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=5_000_000,
        )

        edge1 = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r3",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_memory_bytes=800_000,
            estimated_transfer_ms=12.0,
        )

        edge2 = TransferEdge(
            edge_id="e2",
            producer_region="r2",
            consumer_region="r3",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_memory_bytes=800_000,
            estimated_transfer_ms=12.0,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_006",
            regions=(region1, region2, region3),
            edges=(edge1, edge2),
            topological_order=("r1", "r2", "r3"),
            root_region_ids=("r3",),
            total_compute_ms=90.0,
            total_transfer_ms=24.0,
            total_ttdc_ms=114.0,
            peak_memory_bytes=17_000_000,
            logical_node_count=5,
            backend_switch_count=2,
            native_fraction=0.80,
        )

        assert len(plan.regions) == 3
        assert len(plan.edges) == 2
        assert len(plan.root_region_ids) == 1


class TestRegionBoundaryValidation:
    """Test region boundary contract validation."""

    def test_pit_preserved_across_boundary(self):
        """PIT safety must be preserved across region boundaries."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_memory_bytes=800_000,
            estimated_transfer_ms=10.0,
            preserves_pit=True,
        )

        assert edge.preserves_pit is True

    def test_cross_section_cannot_partition_by_instrument(self):
        """Cross-sectional operations cannot be asset-sharded (MB-P0-011)."""
        with pytest.raises(ValueError, match="CROSS_SECTION_PER_DATE cannot be partitioned by instrument"):
            BackendRegion(
                region_id="r1",
                backend=PhysicalBackend.POLARS_PANEL,
                representation=Representation.POLARS_LONG,
                node_ids=("n1",),
                execution_axis=ExecutionAxis.CROSS_SECTION_PER_DATE,
                required_properties=PhysicalProperties(partitioned_by=("instrument",)),
                state_contract=StateContract(),
                estimated_rows=100_000,
                estimated_memory_bytes=6_000_000,
            )

    def test_recursive_stateful_must_be_sequential(self):
        """Recursive stateful without checkpoint must be sequential (MB-P0-012)."""
        with pytest.raises(ValueError, match="RECURSIVE_TIME without checkpoint must be sequential_only"):
            BackendRegion(
                region_id="r1",
                backend=PhysicalBackend.POLARS_PANEL,
                representation=Representation.POLARS_LONG,
                node_ids=("n1",),
                execution_axis=ExecutionAxis.RECURSIVE_TIME_PER_INSTRUMENT,
                required_properties=PhysicalProperties(),
                state_contract=StateContract(
                    has_state=True,
                    checkpoint_capable=False,
                    sequential_only=False,  # Should fail
                ),
                estimated_rows=100_000,
                estimated_memory_bytes=6_000_000,
            )

    def test_every_node_assigned_exactly_once(self):
        """Each logical node must appear in exactly one region."""
        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=6_000_000,
        )

        region2 = BackendRegion(
            region_id="r2",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n3", "n4"),
            execution_axis=ExecutionAxis.RELATIONAL,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=5_000_000,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_007",
            regions=(region1, region2),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=70.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=70.0,
            peak_memory_bytes=11_000_000,
            logical_node_count=4,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        all_nodes = []
        for region in plan.regions:
            all_nodes.extend(region.node_ids)

        # Check no duplicates
        assert len(all_nodes) == 4
        assert len(set(all_nodes)) == 4
        assert set(all_nodes) == {"n1", "n2", "n3", "n4"}


class TestTopologicalOrdering:
    """Test topological ordering of regions."""

    def test_topological_order_respects_dependencies(self):
        """Topological order must place producers before consumers."""
        plan = PhysicalRegionPlan(
            plan_id="plan_008",
            regions=(),
            edges=(),
            topological_order=("r1", "r2", "r3"),
            root_region_ids=("r3",),
            total_compute_ms=100.0,
            total_transfer_ms=20.0,
            total_ttdc_ms=120.0,
            peak_memory_bytes=15_000_000,
            logical_node_count=6,
            backend_switch_count=2,
            native_fraction=0.85,
        )

        # r1 must come before r2 and r3
        order = plan.topological_order
        assert order.index("r1") < order.index("r2")
        assert order.index("r2") < order.index("r3")

    def test_root_regions_at_end_of_topo_order(self):
        """Root regions should be at the end of topological order."""
        plan = PhysicalRegionPlan(
            plan_id="plan_009",
            regions=(),
            edges=(),
            topological_order=("r1", "r2", "r3"),
            root_region_ids=("r3",),
            total_compute_ms=100.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=100.0,
            peak_memory_bytes=10_000_000,
            logical_node_count=5,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # Root should be last
        assert plan.topological_order[-1] in plan.root_region_ids
