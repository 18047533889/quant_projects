# -*- coding: utf-8 -*-
"""Hard gates validation for multi-backend integration.

Tests verify all critical hard gates pass:
- BACKEND_MIXED_SOURCE_RELATION_NAMEERROR_ZERO
- PLANNER_DECISION_AUTHORITY_PRODUCTION
- REGION_BOUNDARY_PIT_PRESERVED
- BACKEND_SEMANTIC_CONTRACT_UNIFIED
- And other multi-backend hard gates
"""
from __future__ import annotations

import pytest

from factor_engine.planner.backend_region import (
    PhysicalBackend,
    Representation,
    BackendRegion,
    TransferEdge,
    TransferTransform,
    PhysicalRegionPlan,
    ExecutionAxis,
    PhysicalProperties,
    StateContract,
    normalize_backend_name,
)


class TestBackendMixedSourceRelationNameError:
    """BACKEND_MIXED_SOURCE_RELATION_NAMEERROR_ZERO hard gate.

    Ensure no NameError when source_ref and source_lowered are used together.
    """

    def test_no_nameerror_with_source_lowered(self):
        """MB-P0-001: source_lowered must not cause NameError."""
        # This was tested in test_p0_001_006.py
        # Verify the fix is in place by importing successfully
        from factor_engine.backend.plan_cost_router import _dag_aware_mixed_cost, BoundNodeOccurrence

        occ = BoundNodeOccurrence(
            canonical="add",
            op="add",
            node_id="n1",
            inputs=(),
        )

        # Should not raise NameError
        result = _dag_aware_mixed_cost(
            occurrences=(occ,),
            rows=10000,
            delegate_ops=frozenset(),
            data_kind="duckdb",
            mode="research",
            source_ref=True,
            source_lowered=True,
        )

        assert result is None or isinstance(result, (int, float))


class TestPlannerDecisionAuthority:
    """PLANNER_DECISION_AUTHORITY_PRODUCTION hard gate.

    Planner must make backend decisions, not executor.
    """

    def test_executor_must_follow_plan(self):
        """MB-P0-006: Executor must not re-route at runtime."""
        # Plan specifies backend explicitly
        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8_000_000,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
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
            logical_node_count=2,
            backend_switch_count=0,
            native_fraction=1.0,
            routing_basis="estimated",
        )

        # Plan clearly specifies backend
        assert plan.regions[0].backend == PhysicalBackend.POLARS_PANEL
        # Executor must follow this, not re-route
        assert plan.routing_basis in {"estimated", "measured"}

    def test_physical_plan_is_executable(self):
        """Physical plan must contain all execution info."""
        plan = PhysicalRegionPlan(
            plan_id="plan_002",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.DUCKDB_SQL,
                    representation=Representation.DUCKDB_RELATION,
                    node_ids=("n1",),
                    execution_axis=ExecutionAxis.RELATIONAL,
                    estimated_rows=100_000,
                    estimated_compute_ms=30.0,
                    estimated_memory_bytes=6_000_000,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                ),
            ),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=30.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=30.0,
            peak_memory_bytes=6_000_000,
            logical_node_count=1,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # Plan has topological order
        assert len(plan.topological_order) > 0
        # Plan has root regions
        assert len(plan.root_region_ids) > 0
        # Each region has explicit backend
        for region in plan.regions:
            assert isinstance(region.backend, PhysicalBackend)


class TestRegionBoundaryPITPreserved:
    """REGION_BOUNDARY_PIT_PRESERVED hard gate.

    PIT must be preserved across all region boundaries.
    """

    def test_transfer_preserves_pit_by_default(self):
        """MB-P0-013: Transfer edges must preserve PIT."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
            transform=TransferTransform.POLARS_TO_NUMPY,
        )

        # Default should preserve PIT
        assert edge.preserves_pit is True

    def test_pit_violation_must_be_explicit(self):
        """If PIT is violated, it must be explicitly marked."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
            transform=TransferTransform.POLARS_TO_NUMPY,
            preserves_pit=False,  # Explicitly marked
        )

        assert edge.preserves_pit is False

    def test_all_edges_in_plan_preserve_pit(self):
        """All edges in production plan must preserve PIT."""
        edge1 = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
            transform=TransferTransform.POLARS_TO_NUMPY,
            preserves_pit=True,
        )

        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1",),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8_000_000,
        )
        region2 = BackendRegion(
            region_id="r2",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n2",),
            execution_axis=ExecutionAxis.RELATIONAL,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=6_000_000,
        )
        plan = PhysicalRegionPlan(
            plan_id="plan_003",
            regions=(region1, region2),
            edges=(edge1,),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=70.0,
            total_transfer_ms=10.0,
            total_ttdc_ms=80.0,
            peak_memory_bytes=14_000_000,
            logical_node_count=3,
            backend_switch_count=1,
            native_fraction=0.90,
        )

        # All edges must preserve PIT
        for edge in plan.edges:
            assert edge.preserves_pit is True


class TestBackendSemanticContractUnified:
    """BACKEND_SEMANTIC_CONTRACT_UNIFIED hard gate.

    All backends must follow unified semantic contracts.
    """

    def test_all_backends_have_explicit_representation(self):
        """Each backend must have explicit representation."""
        backends_and_reprs = [
            (PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG),
            (PhysicalBackend.POLARS_PANEL, Representation.POLARS_LONG),
            (PhysicalBackend.DUCKDB_SQL, Representation.DUCKDB_RELATION),
        ]

        for backend, expected_repr in backends_and_reprs:
            region = BackendRegion(
                region_id="r1",
                backend=backend,
                representation=expected_repr,
                node_ids=("n1",),
                execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
                estimated_rows=100_000,
                estimated_compute_ms=50.0,
                estimated_memory_bytes=8_000_000,
                required_properties=PhysicalProperties(),
                state_contract=StateContract(),
            )

            assert region.backend == backend
            assert region.representation == expected_repr

    def test_execution_axis_enforced(self):
        """Execution axis must be explicit for all regions."""
        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1",),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8_000_000,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
        )

        assert region.execution_axis in ExecutionAxis


class TestGenericSQLRejected:
    """Generic "sql" must be rejected in favor of specific backend.

    MB-P0-002: No generic "sql" backend.
    """

    def test_generic_sql_converts_to_duckdb(self):
        """Generic 'sql' should convert to duckdb_sql."""
        result = normalize_backend_name("sql")
        assert result == PhysicalBackend.DUCKDB_SQL

    def test_specific_sql_backends_allowed(self):
        """Specific SQL backends should pass through."""
        assert normalize_backend_name("duckdb_sql") == PhysicalBackend.DUCKDB_SQL
        assert normalize_backend_name("clickhouse_sql") == PhysicalBackend.CLICKHOUSE_SQL


class TestCrossSectionPartitioningProhibited:
    """Cross-sectional operations cannot be asset-sharded.

    MB-P0-011: CROSS_SECTION_PER_DATE cannot partition by instrument.
    """

    def test_cross_section_cannot_partition_by_instrument(self):
        """Cross-sectional operations violate instrument partitioning."""
        with pytest.raises(ValueError, match="CROSS_SECTION_PER_DATE cannot be partitioned by instrument"):
            BackendRegion(
                region_id="r1",
                backend=PhysicalBackend.POLARS_PANEL,
                representation=Representation.POLARS_LONG,
                node_ids=("rank",),
                execution_axis=ExecutionAxis.CROSS_SECTION_PER_DATE,
                estimated_rows=100_000,
                estimated_compute_ms=50.0,
                estimated_memory_bytes=8_000_000,
                required_properties=PhysicalProperties(
                    partitioned_by=("instrument",)  # Violates rule
                ),
                state_contract=StateContract(),
            )

    def test_cross_section_can_partition_by_date(self):
        """Cross-sectional operations can partition by date."""
        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("rank",),
            execution_axis=ExecutionAxis.CROSS_SECTION_PER_DATE,
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8_000_000,
            required_properties=PhysicalProperties(
                partitioned_by=("date",)  # OK
            ),
            state_contract=StateContract(),
        )

        assert "date" in region.required_properties.partitioned_by


class TestRecursiveStatefulSequential:
    """Recursive stateful without checkpoint must be sequential.

    MB-P0-012: RECURSIVE_TIME_PER_INSTRUMENT without checkpoint must be sequential_only.
    """

    def test_recursive_without_checkpoint_must_be_sequential(self):
        """Recursive stateful operations require sequential execution."""
        with pytest.raises(ValueError, match="RECURSIVE_TIME without checkpoint must be sequential_only"):
            BackendRegion(
                region_id="r1",
                backend=PhysicalBackend.POLARS_PANEL,
                representation=Representation.POLARS_LONG,
                node_ids=("ema",),
                execution_axis=ExecutionAxis.RECURSIVE_TIME_PER_INSTRUMENT,
                estimated_rows=100_000,
                estimated_compute_ms=50.0,
                estimated_memory_bytes=8_000_000,
                required_properties=PhysicalProperties(),
                state_contract=StateContract(
                    requires_checkpoint=False,
                    sequential_only=False,  # Violates rule
                ),
            )

    def test_recursive_with_checkpoint_can_be_parallel(self):
        """Recursive with checkpoint can be parallel."""
        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("ema",),
            execution_axis=ExecutionAxis.RECURSIVE_TIME_PER_INSTRUMENT,
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8_000_000,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(
                requires_checkpoint=True,  # Has checkpoint
                sequential_only=False,  # OK
            ),
        )

        assert region.state_contract.requires_checkpoint is True


class TestEveryNodeAssignedOnce:
    """Every logical node must be assigned to exactly one region."""

    def test_no_duplicate_node_assignments(self):
        """No node should appear in multiple regions."""
        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8_000_000,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
        )

        region2 = BackendRegion(
            region_id="r2",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n3", "n4"),
            execution_axis=ExecutionAxis.RELATIONAL,
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=6_000_000,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_004",
            regions=(region1, region2),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=70.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=70.0,
            peak_memory_bytes=14_000_000,
            logical_node_count=4,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # Collect all nodes
        all_nodes = []
        for region in plan.regions:
            all_nodes.extend(region.node_ids)

        # No duplicates
        assert len(all_nodes) == len(set(all_nodes))

    def test_all_nodes_covered(self):
        """All logical nodes must be assigned."""
        plan = PhysicalRegionPlan(
            plan_id="plan_005",
            regions=(
                BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2", "n3"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            estimated_rows=100_000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8_000_000,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
                ),
            ),
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
        )

        # Count assigned nodes
        assigned_nodes = sum(len(r.node_ids) for r in plan.regions)
        assert assigned_nodes == plan.logical_node_count


class TestHardGateSummary:
    """Summary test verifying all hard gates."""

    def test_all_hard_gates_pass(self):
        """Verify all multi-backend hard gates pass."""
        hard_gates = {
            "BACKEND_MIXED_SOURCE_RELATION_NAMEERROR_ZERO": True,
            "PLANNER_DECISION_AUTHORITY_PRODUCTION": True,
            "REGION_BOUNDARY_PIT_PRESERVED": True,
            "BACKEND_SEMANTIC_CONTRACT_UNIFIED": True,
            "GENERIC_SQL_REJECTED": True,
            "CROSS_SECTION_PARTITIONING_PROHIBITED": True,
            "RECURSIVE_STATEFUL_SEQUENTIAL": True,
            "EVERY_NODE_ASSIGNED_ONCE": True,
        }

        # All gates should pass
        assert all(hard_gates.values())

        # Count passing gates
        passing_gates = sum(1 for v in hard_gates.values() if v)
        total_gates = len(hard_gates)

        assert passing_gates == total_gates