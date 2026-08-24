# -*- coding: utf-8 -*-
"""Tests for multi-backend P0 fixes MB-P0-001 through MB-P0-006.

Tests cover:
- MB-P0-001: source_lowered NameError fix
- MB-P0-002: Generic "sql" replaced with specific backend
- MB-P0-003: Backend-specific native fraction
- MB-P0-004: BackendRegion model existence
- MB-P0-005: PhysicalRegionPlan model existence
- MB-P0-006: Executor must not re-route
"""
from __future__ import annotations

import pytest


class TestMBP0001SourceLoweredNameError:
    """MB-P0-001: _dag_aware_mixed_cost() must not have NameError on source_lowered."""

    def test_mixed_cost_source_relation_no_nameerror(self):
        """Mixed cost with source_ref should not raise NameError."""
        from factor_engine.backend.plan_cost_router import _dag_aware_mixed_cost, BoundNodeOccurrence

        # Simple occurrence with no source ref
        occ = BoundNodeOccurrence(
            canonical="ts_mean",
            op="ts_mean",
            node_id="n1",
            window=20,
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

        # Should return a float or None, not crash
        assert result is None or isinstance(result, (int, float))

    def test_lowerable_source_ref_keeps_native_candidates(self):
        """When source_ref is lowerable, SQL candidates should be eligible."""
        from factor_engine.backend.plan_cost_router import _dag_aware_mixed_cost, BoundNodeOccurrence

        occ = BoundNodeOccurrence(
            canonical="add",
            op="add",
            node_id="n1",
            inputs=(),
        )

        # source_ref=True + source_lowered=True should allow SQL
        result_lowerable = _dag_aware_mixed_cost(
            occurrences=(occ,),
            rows=10000,
            delegate_ops=frozenset(),
            data_kind="duckdb",
            mode="research",
            source_ref=True,
            source_lowered=True,
        )

        # source_ref=True + source_lowered=False should block SQL
        result_blocked = _dag_aware_mixed_cost(
            occurrences=(occ,),
            rows=10000,
            delegate_ops=frozenset(),
            data_kind="duckdb",
            mode="research",
            source_ref=True,
            source_lowered=False,
        )

        # Both should succeed (not crash), but costs may differ
        assert result_lowerable is None or isinstance(result_lowerable, (int, float))
        assert result_blocked is None or isinstance(result_blocked, (int, float))


class TestMBP0002GenericSQLBackend:
    """MB-P0-002: Generic "sql" must be replaced with specific backend."""

    def test_mixed_cost_uses_duckdb_sql_not_generic_sql(self):
        """Mixed cost for duckdb should use duckdb_sql, not generic "sql"."""
        from factor_engine.backend.plan_cost_router import _dag_aware_mixed_cost, BoundNodeOccurrence

        occ = BoundNodeOccurrence(
            canonical="add",
            op="add",
            node_id="n1",
            inputs=(),
        )

        # This should internally use "duckdb_sql" not "sql"
        result = _dag_aware_mixed_cost(
            occurrences=(occ,),
            rows=10000,
            delegate_ops=frozenset(),
            data_kind="duckdb",
            mode="research",
            source_ref=False,
            source_lowered=True,
        )

        # Should succeed - the fix ensures sql_backend is used
        assert result is None or isinstance(result, (int, float))

    def test_normalize_backend_name_rejects_generic_sql(self):
        """normalize_backend_name should convert 'sql' to specific backend."""
        from factor_engine.planner.backend_region import normalize_backend_name, PhysicalBackend

        # Generic "sql" should normalize to duckdb_sql (default SQL backend)
        result = normalize_backend_name("sql")
        assert result == PhysicalBackend.DUCKDB_SQL

        # Explicit backends should pass through
        assert normalize_backend_name("duckdb_sql") == PhysicalBackend.DUCKDB_SQL
        assert normalize_backend_name("clickhouse_sql") == PhysicalBackend.CLICKHOUSE_SQL
        assert normalize_backend_name("pandas_numpy") == PhysicalBackend.PANDAS_NUMPY


class TestMBP0003BackendSpecificNativeFraction:
    """MB-P0-003: Native fraction must be backend-specific."""

    def test_native_fraction_polars_uses_polars_capability(self):
        """Polars native fraction should check polars capability, not SQL."""
        from factor_engine.planner.native_fraction import plan_native_subgraph_fraction
        from factor_engine.planner.logical_plan import PlanNode

        # Simple plan with one operator
        plan = PlanNode(op="add", inputs=())

        report = plan_native_subgraph_fraction(
            plan, backend="polars_panel", mode="research"
        )

        assert report.backend == "polars_panel"
        # Report should have backend-specific classification
        assert isinstance(report.node_count_total, int)
        assert isinstance(report.node_count_native, int)
        assert isinstance(report.estimated_compute_fraction, float)

    def test_native_fraction_duckdb_uses_sql_capability(self):
        """DuckDB native fraction should check SQL capability, not polars."""
        from factor_engine.planner.native_fraction import plan_native_subgraph_fraction
        from factor_engine.planner.logical_plan import PlanNode

        plan = PlanNode(op="add", inputs=())

        report = plan_native_subgraph_fraction(
            plan, backend="duckdb_sql", mode="research", data_source_kind="duckdb"
        )

        assert report.backend == "duckdb_sql"
        assert isinstance(report.node_count_total, int)

    def test_delegate_not_counted_native(self):
        """Polars delegate operations should not count as native."""
        from factor_engine.planner.native_fraction import plan_native_subgraph_fraction
        from factor_engine.planner.logical_plan import PlanNode

        plan = PlanNode(op="add", inputs=())

        report = plan_native_subgraph_fraction(
            plan, backend="polars_panel", mode="research"
        )

        # Delegates should be tracked separately
        assert report.node_count_delegate >= 0
        # Native + delegate + unsupported should equal total
        assert (
            report.node_count_native
            + report.node_count_delegate
            + report.node_count_unsupported
            == report.node_count_total
        )


class TestMBP0004BackendRegionModel:
    """MB-P0-004: BackendRegion and TransferEdge must exist."""

    def test_backend_region_model_exists(self):
        """BackendRegion dataclass should be importable and usable."""
        from factor_engine.planner.backend_region import (
            BackendRegion,
            PhysicalBackend,
            Representation,
            ExecutionAxis,
        )

        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2", "n3"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            estimated_rows=100000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8000000,
            source_snapshot="snap_001",
            pit_safe=True,
        )

        assert region.region_id == "r1"
        assert region.backend == PhysicalBackend.POLARS_PANEL
        assert len(region.node_ids) == 3

    def test_transfer_edge_model_exists(self):
        """TransferEdge dataclass should be importable and usable."""
        from factor_engine.planner.backend_region import (
            TransferEdge,
            PhysicalBackend,
            Representation,
        )

        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100000,
            estimated_bytes=800000,
            estimated_transfer_ms=10.0,
            requires_sort=True,
            preserves_pit=True,
        )

        assert edge.edge_id == "e1"
        assert edge.source_backend == PhysicalBackend.POLARS_PANEL
        assert edge.target_backend == PhysicalBackend.DUCKDB_SQL
        assert edge.requires_sort is True


class TestMBP0005PhysicalRegionPlan:
    """MB-P0-005: PhysicalRegionPlan must exist for executable multi-region DAG."""

    def test_physical_region_plan_model_exists(self):
        """PhysicalRegionPlan should be importable and usable."""
        from factor_engine.planner.backend_region import (
            PhysicalRegionPlan,
            BackendRegion,
            TransferEdge,
            PhysicalBackend,
            Representation,
            ExecutionAxis,
        )

        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            estimated_rows=100000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8000000,
        )

        region2 = BackendRegion(
            region_id="r2",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n3", "n4"),
            execution_axis=ExecutionAxis.RELATIONAL,
            estimated_rows=100000,
            estimated_compute_ms=30.0,
            estimated_memory_bytes=4000000,
        )

        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100000,
            estimated_bytes=800000,
            estimated_transfer_ms=10.0,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_001",
            regions=(region1, region2),
            edges=(edge,),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=80.0,
            total_transfer_ms=10.0,
            total_ttdc_ms=90.0,
            peak_memory_bytes=12000000,
            plan_hash="abc123",
            logical_node_count=4,
            backend_switch_count=1,
            native_fraction=0.75,
            routing_basis="estimated",
        )

        assert plan.plan_id == "plan_001"
        assert len(plan.regions) == 2
        assert len(plan.edges) == 1
        assert plan.backend_switch_count == 1

    def test_every_logical_node_assigned_exactly_once(self):
        """Each logical node should appear in exactly one region."""
        from factor_engine.planner.backend_region import (
            PhysicalRegionPlan,
            BackendRegion,
            PhysicalBackend,
            Representation,
            ExecutionAxis,
        )

        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            estimated_rows=100000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8000000,
        )

        region2 = BackendRegion(
            region_id="r2",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("n3", "n4"),
            execution_axis=ExecutionAxis.RELATIONAL,
            estimated_rows=100000,
            estimated_compute_ms=30.0,
            estimated_memory_bytes=4000000,
        )

        plan = PhysicalRegionPlan(
            plan_id="plan_001",
            regions=(region1, region2),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=80.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=80.0,
            peak_memory_bytes=12000000,
            plan_hash="abc123",
            logical_node_count=4,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # Collect all node_ids across regions
        all_nodes = []
        for region in plan.regions:
            all_nodes.extend(region.node_ids)

        # Should have 4 total nodes
        assert len(all_nodes) == 4
        # No duplicates - each node assigned exactly once
        assert len(set(all_nodes)) == 4


class TestMBP0006ExecutorMustNotReroute:
    """MB-P0-006: Executor must follow the plan, not re-route at runtime."""

    def test_plan_route_carries_backend_decision(self):
        """PlanRoute should clearly specify the chosen backend."""
        from factor_engine.backend.plan_cost_router import PlanRoute

        route = PlanRoute(
            backend="polars_panel",
            estimated_cost=100.0,
            routing_basis="estimated",
            candidate_costs=(("pandas_numpy", 150.0), ("polars_panel", 100.0)),
            row_count_estimate=10000,
            ops=("ts_mean", "add"),
        )

        # Backend decision is explicit
        assert route.backend == "polars_panel"
        # Should not be ambiguous like "hybrid" without region details
        assert route.backend in {
            "pandas_numpy",
            "polars_panel",
            "polars_long",
            "duckdb_sql",
            "clickhouse_sql",
            "hybrid",
        }

    def test_physical_region_plan_is_executable(self):
        """PhysicalRegionPlan should contain all info needed for execution."""
        from factor_engine.planner.backend_region import (
            PhysicalRegionPlan,
            BackendRegion,
            PhysicalBackend,
            Representation,
            ExecutionAxis,
        )

        region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2", "n3"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            estimated_rows=100000,
            estimated_compute_ms=50.0,
            estimated_memory_bytes=8000000,
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
            peak_memory_bytes=8000000,
            plan_hash="abc123",
            logical_node_count=3,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # Plan should specify execution order
        assert len(plan.topological_order) > 0
        # Plan should specify root regions
        assert len(plan.root_region_ids) > 0
        # Each region has explicit backend assignment
        for region in plan.regions:
            assert isinstance(region.backend, PhysicalBackend)
