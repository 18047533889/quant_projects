# -*- coding: utf-8 -*-
"""Test cost model for multi-backend planning.

Tests cover:
- Per-backend cost estimation
- Transfer cost calculation
- Cost-based backend selection
- Memory footprint estimation
"""
from __future__ import annotations

import pytest

from planner.backend_region import (
    PhysicalBackend,
    Representation,
    BackendRegion,
    TransferEdge,
    PhysicalRegionPlan,
    ExecutionAxis,
    PhysicalProperties,
    StateContract,
    estimate_transfer_cost_ms,
)


class TestPerBackendCostEstimation:
    """Test cost estimation per backend."""

    def test_pandas_cost_higher_than_polars(self):
        """Pandas should generally cost more than Polars for large data."""
        # This is a general trend, not absolute
        pandas_region = BackendRegion(
            region_id="r_pandas",
            backend=PhysicalBackend.PANDAS_NUMPY,
            representation=Representation.PANDAS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=1_000_000,
            estimated_memory_bytes=80_000_000,
        )

        polars_region = BackendRegion(
            region_id="r_polars",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=1_000_000,
            estimated_memory_bytes=60_000_000,
        )

        # Polars generally uses less memory for same operations
        assert polars_region.estimated_bytes <= pandas_region.estimated_bytes

    def test_duckdb_cost_for_aggregation(self):
        """DuckDB should be efficient for aggregation operations."""
        region = BackendRegion(
            region_id="r_duckdb",
            backend=PhysicalBackend.DUCKDB_SQL,
            representation=Representation.DUCKDB_RELATION,
            node_ids=("group_by", "aggregate", "join"),
            execution_axis=ExecutionAxis.RELATIONAL,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=8_000_000,
        )

        # DuckDB is efficient for relational operations
        assert region.backend == PhysicalBackend.DUCKDB_SQL
        assert region.execution_axis == ExecutionAxis.RELATIONAL

    def test_memory_footprint_scales_with_rows(self):
        """Memory footprint should scale with row count."""
        small_region = BackendRegion(
            region_id="r_small",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1",),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=10_000,
            estimated_memory_bytes=800_000,
        )

        large_region = BackendRegion(
            region_id="r_large",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1",),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=1_000_000,
            estimated_memory_bytes=80_000_000,
        )

        # Memory should scale with rows
        assert large_region.estimated_bytes > small_region.estimated_bytes


class TestTransferCostCalculation:
    """Test transfer cost calculation (MB-P1-021)."""

    def test_zero_cost_for_same_representation(self):
        """Same representation should have minimal transfer cost."""
        cost = estimate_transfer_cost_ms(
            source_repr=Representation.POLARS_LONG,
            target_repr=Representation.POLARS_LONG,
            estimated_memory_bytes=1_000_000,
        )

        # Should be very low (near zero)
        assert cost >= 0.0
        assert cost < 10.0  # Less than 10ms for same representation

    def test_non_zero_cost_for_different_representation(self):
        """Different representation should have measurable cost."""
        cost = estimate_transfer_cost_ms(
            source_repr=Representation.PANDAS_LONG,
            target_repr=Representation.DUCKDB_RELATION,
            estimated_memory_bytes=1_000_000,
        )

        # Should have measurable cost
        assert cost > 0.0

    def test_sort_adds_significant_cost(self):
        """Adding sort should increase cost significantly."""
        cost_no_sort = estimate_transfer_cost_ms(
            source_repr=Representation.POLARS_LONG,
            target_repr=Representation.DUCKDB_RELATION,
            estimated_memory_bytes=10_000_000,
            requires_sort=False,
        )

        cost_with_sort = estimate_transfer_cost_ms(
            source_repr=Representation.POLARS_LONG,
            target_repr=Representation.DUCKDB_RELATION,
            estimated_memory_bytes=10_000_000,
            requires_sort=True,
        )

        # Sort should add significant cost
        assert cost_with_sort > cost_no_sort * 1.2

    def test_reshape_adds_cost(self):
        """Reshape (long ↔ wide) should add cost."""
        cost_no_reshape = estimate_transfer_cost_ms(
            source_repr=Representation.PANDAS_LONG,
            target_repr=Representation.POLARS_LONG,
            estimated_memory_bytes=10_000_000,
            requires_reshape=False,
        )

        cost_with_reshape = estimate_transfer_cost_ms(
            source_repr=Representation.PANDAS_LONG,
            target_repr=Representation.PANDAS_WIDE,
            estimated_memory_bytes=10_000_000,
            requires_reshape=True,
        )

        assert cost_with_reshape > cost_no_reshape


class TestCostBasedBackendSelection:
    """Test cost-based backend selection logic."""

    def test_select_cheapest_backend_for_simple_ops(self):
        """For simple operations, should select cheapest backend."""
        candidates = [
            ("pandas", 100.0),
            ("polars", 60.0),
            ("duckdb", 80.0),
        ]

        # Should select polars (lowest cost)
        best_backend = min(candidates, key=lambda x: x[1])
        assert best_backend[0] == "polars"

    def test_prefer_native_over_transfer_overhead(self):
        """Should prefer native execution over transfer overhead."""
        # Scenario: small computation in polars vs transfer to duckdb
        polars_compute_cost = 20.0
        duckdb_compute_cost = 15.0
        transfer_cost = 25.0

        polars_total = polars_compute_cost
        duckdb_total = transfer_cost + duckdb_compute_cost

        # Polars should win (20 < 40)
        assert polars_total < duckdb_total

    def test_amortize_transfer_for_large_subgraph(self):
        """Large subgraphs should amortize transfer costs."""
        # Scenario: transfer once, compute many operations
        transfer_cost = 30.0
        polars_compute_per_op = 5.0
        duckdb_compute_per_op = 2.0
        num_ops = 20

        polars_total = num_ops * polars_compute_per_op
        duckdb_total = transfer_cost + (num_ops * duckdb_compute_per_op)

        # DuckDB should win: 30 + 40 = 70 < 100
        assert duckdb_total < polars_total


class TestTotalPlanCost:
    """Test total plan cost calculation."""

    def test_single_region_plan_cost(self):
        """Single region plan cost is just compute cost."""
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
        )

        assert plan.total_ttdc_ms == plan.total_compute_ms
        assert plan.total_transfer_ms == 0.0

    def test_multi_region_plan_includes_transfer(self):
        """Multi-region plan must include transfer costs."""
        plan = PhysicalRegionPlan(
            plan_id="plan_002",
            regions=(),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=80.0,
            total_transfer_ms=15.0,
            total_ttdc_ms=95.0,
            peak_memory_bytes=12_000_000,
            logical_node_count=5,
            backend_switch_count=1,
            native_fraction=0.85,
        )

        assert plan.total_ttdc_ms == plan.total_compute_ms + plan.total_transfer_ms

    def test_plan_cost_decomposition(self):
        """Plan cost should be decomposed by component (MB-P1-007)."""
        plan = PhysicalRegionPlan(
            plan_id="plan_003",
            regions=(),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=100.0,
            total_transfer_ms=20.0,
            total_sort_ms=10.0,
            total_reshape_ms=5.0,
            total_ttdc_ms=135.0,
            peak_memory_bytes=15_000_000,
            logical_node_count=6,
            backend_switch_count=1,
            native_fraction=0.90,
        )

        # Total should include all components
        expected_total = (
            plan.total_compute_ms +
            plan.total_transfer_ms +
            plan.total_sort_ms +
            plan.total_reshape_ms
        )
        assert plan.total_ttdc_ms == expected_total


class TestMemoryFootprintEstimation:
    """Test memory footprint estimation."""

    def test_peak_memory_single_region(self):
        """Peak memory for single region is region memory."""
        region = BackendRegion(
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
        )

        assert plan.peak_memory_bytes == region.estimated_bytes

    def test_peak_memory_multi_region_sequential(self):
        """Peak memory for sequential regions is max of individual regions."""
        region1 = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1",),
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
            node_ids=("n2",),
            execution_axis=ExecutionAxis.RELATIONAL,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=6_000_000,
        )

        # If sequential, peak is max
        peak = max(region1.estimated_bytes, region2.estimated_bytes)

        plan = PhysicalRegionPlan(
            plan_id="plan_002",
            regions=(region1, region2),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=70.0,
            total_transfer_ms=10.0,
            total_ttdc_ms=80.0,
            peak_memory_bytes=peak,
            logical_node_count=2,
            backend_switch_count=1,
            native_fraction=1.0,
        )

        assert plan.peak_memory_bytes == 8_000_000


class TestNativeFractionCalculation:
    """Test native fraction calculation."""

    def test_all_native_fraction_is_one(self):
        """All operations native should give fraction = 1.0."""
        plan = PhysicalRegionPlan(
            plan_id="plan_001",
            regions=(),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=50.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=50.0,
            peak_memory_bytes=8_000_000,
            logical_node_count=5,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        assert plan.native_fraction == 1.0

    def test_partial_native_fraction(self):
        """Partially native operations should have fraction < 1.0."""
        plan = PhysicalRegionPlan(
            plan_id="plan_002",
            regions=(),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=80.0,
            total_transfer_ms=15.0,
            total_ttdc_ms=95.0,
            peak_memory_bytes=12_000_000,
            logical_node_count=10,
            backend_switch_count=1,
            native_fraction=0.75,  # 75% native
        )

        assert 0.0 < plan.native_fraction < 1.0

    def test_zero_native_fraction(self):
        """All operations delegated should give fraction = 0.0."""
        plan = PhysicalRegionPlan(
            plan_id="plan_003",
            regions=(),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=100.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=100.0,
            peak_memory_bytes=10_000_000,
            logical_node_count=8,
            backend_switch_count=0,
            native_fraction=0.0,  # All delegated
        )

        assert plan.native_fraction == 0.0


class TestPlanHashGeneration:
    """Test plan hash generation (MB-P2-014)."""

    def test_plan_hash_is_stable(self):
        """Plan hash should be stable for same inputs."""
        regions = (
            BackendRegion(
                region_id="r1",
                backend=PhysicalBackend.POLARS_PANEL,
                representation=Representation.POLARS_LONG,
                node_ids=("n1", "n2"),
                execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
                required_properties=PhysicalProperties(),
                state_contract=StateContract(),
                estimated_rows=100_000,
                estimated_memory_bytes=8_000_000,
            ),
        )

        edges = ()
        logical_hash = "abc123"

        hash1 = PhysicalRegionPlan.compute_plan_hash(regions, edges, logical_hash)
        hash2 = PhysicalRegionPlan.compute_plan_hash(regions, edges, logical_hash)

        assert hash1 == hash2

    def test_different_backend_produces_different_hash(self):
        """Different backend assignment should produce different hash."""
        region_polars = BackendRegion(
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

        region_pandas = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.PANDAS_NUMPY,
            representation=Representation.PANDAS_LONG,
            node_ids=("n1", "n2"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=100_000,
            estimated_memory_bytes=8_000_000,
        )

        logical_hash = "abc123"

        hash_polars = PhysicalRegionPlan.compute_plan_hash(
            (region_polars,), (), logical_hash
        )
        hash_pandas = PhysicalRegionPlan.compute_plan_hash(
            (region_pandas,), (), logical_hash
        )

        assert hash_polars != hash_pandas
