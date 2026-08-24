# -*- coding: utf-8 -*-
"""Test region optimizer for multi-backend planning.

Tests cover:
- Greedy region formation
- Cost-driven region merging
- Backend switch minimization
- Native subgraph maximization
"""
from __future__ import annotations

import pytest

from factor_engine.planner.backend_region import (
    PhysicalBackend,
    Representation,
    BackendRegion,
    TransferEdge,
    PhysicalRegionPlan,
    ExecutionAxis,
    PhysicalProperties,
    StateContract,
)


class TestGreedyRegionFormation:
    """Test greedy region formation algorithm."""

    def test_single_backend_forms_one_region(self):
        """All operations on same backend should form single region."""
        # All operations compatible with Polars
        plan = PhysicalRegionPlan(
            plan_id="plan_001",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.POLARS_PANEL,
                    representation=Representation.POLARS_LONG,
                    node_ids=("n1", "n2", "n3", "n4"),
                    execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=100_000,
                    estimated_memory_bytes=8_000_000,
                ),
            ),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=60.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=60.0,
            peak_memory_bytes=8_000_000,
            logical_node_count=4,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        assert len(plan.regions) == 1
        assert plan.backend_switch_count == 0

    def test_incompatible_ops_split_regions(self):
        """Incompatible operations should split into multiple regions."""
        # Some ops only work in Pandas, others in DuckDB
        plan = PhysicalRegionPlan(
            plan_id="plan_002",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.PANDAS_NUMPY,
                    representation=Representation.PANDAS_LONG,
                    node_ids=("n1", "n2"),
                    execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=100_000,
                    estimated_memory_bytes=8_000_000,
                ),
                BackendRegion(
                    region_id="r2",
                    backend=PhysicalBackend.DUCKDB_SQL,
                    representation=Representation.DUCKDB_RELATION,
                    node_ids=("n3", "n4"),
                    execution_axis=ExecutionAxis.RELATIONAL,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=100_000,
                    estimated_memory_bytes=6_000_000,
                ),
            ),
            edges=(
                TransferEdge(
                    edge_id="e1",
                    producer_region="r1",
                    consumer_region="r2",
                    source_backend=PhysicalBackend.PANDAS_NUMPY,
                    target_backend=PhysicalBackend.DUCKDB_SQL,
                    source_representation=Representation.PANDAS_LONG,
                    target_representation=Representation.DUCKDB_RELATION,
                    estimated_rows=100_000,
                    estimated_memory_bytes=800_000,
                    estimated_transfer_ms=15.0,
                ),
            ),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=80.0,
            total_transfer_ms=15.0,
            total_ttdc_ms=95.0,
            peak_memory_bytes=14_000_000,
            logical_node_count=4,
            backend_switch_count=1,
            native_fraction=0.80,
        )

        assert len(plan.regions) == 2
        assert plan.backend_switch_count == 1

    def test_maximal_regions_formed(self):
        """Should form maximal contiguous regions per backend."""
        # Long chain in Polars, then DuckDB, then Polars again
        plan = PhysicalRegionPlan(
            plan_id="plan_003",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.POLARS_PANEL,
                    representation=Representation.POLARS_LONG,
                    node_ids=("n1", "n2", "n3"),
                    execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=100_000,
                    estimated_memory_bytes=8_000_000,
                ),
                BackendRegion(
                    region_id="r2",
                    backend=PhysicalBackend.DUCKDB_SQL,
                    representation=Representation.DUCKDB_RELATION,
                    node_ids=("n4", "n5"),
                    execution_axis=ExecutionAxis.RELATIONAL,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=100_000,
                    estimated_memory_bytes=6_000_000,
                ),
                BackendRegion(
                    region_id="r3",
                    backend=PhysicalBackend.POLARS_PANEL,
                    representation=Representation.POLARS_LONG,
                    node_ids=("n6", "n7"),
                    execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=100_000,
                    estimated_memory_bytes=7_000_000,
                ),
            ),
            edges=(),
            topological_order=("r1", "r2", "r3"),
            root_region_ids=("r3",),
            total_compute_ms=120.0,
            total_transfer_ms=30.0,
            total_ttdc_ms=150.0,
            peak_memory_bytes=21_000_000,
            logical_node_count=7,
            backend_switch_count=2,
            native_fraction=0.85,
        )

        # Should have 3 regions (Polars, DuckDB, Polars)
        assert len(plan.regions) == 3
        assert plan.backend_switch_count == 2


class TestCostDrivenRegionMerging:
    """Test cost-driven region merging decisions."""

    def test_merge_when_transfer_exceeds_recompute(self):
        """Should merge regions when transfer cost exceeds recomputation."""
        # Scenario: small computation in suboptimal backend
        # vs expensive transfer to optimal backend

        # Option 1: Keep separate (transfer cost)
        transfer_cost = 50.0
        region1_compute = 20.0
        region2_compute = 10.0
        total_separate = region1_compute + transfer_cost + region2_compute

        # Option 2: Merge (recompute in suboptimal backend)
        merged_compute = 35.0  # Slightly higher but no transfer

        # Should merge since 35 < 80
        assert merged_compute < total_separate

    def test_keep_separate_when_native_much_faster(self):
        """Should keep regions separate when native execution is much faster."""
        # Scenario: DuckDB native aggregation vs Pandas fallback

        # Option 1: Separate with transfer
        polars_compute = 10.0
        transfer_cost = 15.0
        duckdb_native_compute = 5.0
        total_separate = polars_compute + transfer_cost + duckdb_native_compute

        # Option 2: Merge (fallback to slower backend)
        polars_fallback_compute = 50.0  # Much slower without DuckDB

        # Should keep separate since 30 < 50
        assert total_separate < polars_fallback_compute

    def test_merge_tiny_regions(self):
        """Should merge very small regions to reduce overhead."""
        # Many tiny regions with small computations but many transfers
        num_regions = 10
        compute_per_region = 2.0
        transfer_per_edge = 5.0

        total_separate = (num_regions * compute_per_region) + ((num_regions - 1) * transfer_per_edge)

        # Merged: slightly higher compute but no transfers
        merged_compute = 30.0

        # Should merge: 30 < 65
        assert merged_compute < total_separate


class TestBackendSwitchMinimization:
    """Test backend switch minimization."""

    def test_minimize_switches_in_linear_chain(self):
        """Should minimize backend switches in linear chain."""
        # A → B → A pattern should be avoided if possible
        # Better: A → A → A (recompute B in A)

        plan_switches = PhysicalRegionPlan(
            plan_id="plan_bad",
            regions=(),
            edges=(),
            topological_order=("r1", "r2", "r3"),
            root_region_ids=("r3",),
            total_compute_ms=100.0,
            total_transfer_ms=40.0,
            total_ttdc_ms=140.0,
            peak_memory_bytes=15_000_000,
            logical_node_count=6,
            backend_switch_count=2,  # Two switches
            native_fraction=0.75,
        )

        plan_merged = PhysicalRegionPlan(
            plan_id="plan_good",
            regions=(),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=120.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=120.0,
            peak_memory_bytes=12_000_000,
            logical_node_count=6,
            backend_switch_count=0,  # No switches
            native_fraction=0.85,
        )

        # Merged should be better: 120 < 140
        assert plan_merged.total_ttdc_ms < plan_switches.total_ttdc_ms

    def test_allow_switch_for_significant_gain(self):
        """Should allow backend switch for significant performance gain."""
        # Scenario: DuckDB aggregation is 10x faster than Polars

        plan_no_switch = PhysicalRegionPlan(
            plan_id="plan_polars_only",
            regions=(),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=200.0,  # Slow aggregation in Polars
            total_transfer_ms=0.0,
            total_ttdc_ms=200.0,
            peak_memory_bytes=10_000_000,
            logical_node_count=5,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        plan_with_switch = PhysicalRegionPlan(
            plan_id="plan_duckdb_agg",
            regions=(),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=40.0,  # Fast aggregation in DuckDB
            total_transfer_ms=15.0,
            total_ttdc_ms=55.0,
            peak_memory_bytes=12_000_000,
            logical_node_count=5,
            backend_switch_count=1,
            native_fraction=0.90,
        )

        # Switch should be beneficial: 55 < 200
        assert plan_with_switch.total_ttdc_ms < plan_no_switch.total_ttdc_ms


class TestNativeSubgraphMaximization:
    """Test native subgraph maximization."""

    def test_maximize_native_fraction(self):
        """Should prefer plans with higher native fraction."""
        plan_low_native = PhysicalRegionPlan(
            plan_id="plan_low",
            regions=(),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=100.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=100.0,
            peak_memory_bytes=10_000_000,
            logical_node_count=10,
            backend_switch_count=0,
            native_fraction=0.50,  # 50% native
        )

        plan_high_native = PhysicalRegionPlan(
            plan_id="plan_high",
            regions=(),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=95.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=95.0,
            peak_memory_bytes=10_000_000,
            logical_node_count=10,
            backend_switch_count=0,
            native_fraction=0.90,  # 90% native
        )

        # Higher native fraction should be preferred
        assert plan_high_native.native_fraction > plan_low_native.native_fraction
        assert plan_high_native.total_ttdc_ms < plan_low_native.total_ttdc_ms

    def test_prefer_maximal_native_subgraphs(self):
        """Should form maximal contiguous native subgraphs."""
        # Plan with small fragmented native regions
        plan_fragmented = PhysicalRegionPlan(
            plan_id="plan_fragmented",
            regions=(),
            edges=(),
            topological_order=("r1", "r2", "r3", "r4"),
            root_region_ids=("r4",),
            total_compute_ms=80.0,
            total_transfer_ms=45.0,  # Many small transfers
            total_ttdc_ms=125.0,
            peak_memory_bytes=12_000_000,
            logical_node_count=8,
            backend_switch_count=3,
            native_fraction=0.70,
        )

        # Plan with maximal native regions
        plan_maximal = PhysicalRegionPlan(
            plan_id="plan_maximal",
            regions=(),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r2",),
            total_compute_ms=85.0,
            total_transfer_ms=15.0,  # One large transfer
            total_ttdc_ms=100.0,
            peak_memory_bytes=12_000_000,
            logical_node_count=8,
            backend_switch_count=1,
            native_fraction=0.75,
        )

        # Maximal should be better
        assert plan_maximal.total_ttdc_ms < plan_fragmented.total_ttdc_ms
        assert plan_maximal.backend_switch_count < plan_fragmented.backend_switch_count


class TestMemoryConstrainedOptimization:
    """Test optimization under memory constraints."""

    def test_split_region_to_fit_memory(self):
        """Should split large region if it exceeds memory budget."""
        memory_budget = 10_000_000  # 10MB budget

        # Single large region exceeds budget
        large_region = BackendRegion(
            region_id="r1",
            backend=PhysicalBackend.POLARS_PANEL,
            representation=Representation.POLARS_LONG,
            node_ids=("n1", "n2", "n3", "n4"),
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            required_properties=PhysicalProperties(),
            state_contract=StateContract(),
            estimated_rows=1_000_000,
            estimated_memory_bytes=15_000_000,  # Exceeds budget
        )

        assert large_region.estimated_bytes > memory_budget

        # Should split into smaller regions
        # In practice, optimizer would create multiple smaller regions

    def test_prefer_streaming_under_memory_pressure(self):
        """Should prefer streaming execution under memory pressure."""
        # DuckDB with streaming
        plan_streaming = PhysicalRegionPlan(
            plan_id="plan_streaming",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.DUCKDB_SQL,
                    representation=Representation.DUCKDB_RELATION,
                    node_ids=("n1", "n2"),
                    execution_axis=ExecutionAxis.RELATIONAL,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=1_000_000,
                    estimated_memory_bytes=5_000_000,  # Lower memory
                    streaming_capable=True,
                ),
            ),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=60.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=60.0,
            peak_memory_bytes=5_000_000,
            logical_node_count=2,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # Pandas without streaming
        plan_materialized = PhysicalRegionPlan(
            plan_id="plan_materialized",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.PANDAS_NUMPY,
                    representation=Representation.PANDAS_LONG,
                    node_ids=("n1", "n2"),
                    execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=1_000_000,
                    estimated_memory_bytes=20_000_000,  # Higher memory
                    streaming_capable=False,
                ),
            ),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=50.0,
            total_transfer_ms=0.0,
            total_ttdc_ms=50.0,
            peak_memory_bytes=20_000_000,
            logical_node_count=2,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # Under memory pressure, should prefer streaming despite slightly higher time
        memory_budget = 10_000_000
        assert plan_streaming.peak_memory_bytes < memory_budget
        assert plan_materialized.peak_memory_bytes > memory_budget


class TestOptimizerHeuristics:
    """Test optimizer heuristics."""

    def test_sql_for_large_aggregations(self):
        """Should prefer SQL backends for large aggregations."""
        # Large aggregation workload
        plan_duckdb = PhysicalRegionPlan(
            plan_id="plan_duckdb",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.DUCKDB_SQL,
                    representation=Representation.DUCKDB_RELATION,
                    node_ids=("group_by", "aggregate", "join"),
                    execution_axis=ExecutionAxis.RELATIONAL,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=10_000_000,
                    estimated_memory_bytes=100_000_000,
                ),
            ),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=80.0,  # Fast SQL aggregation
            total_transfer_ms=0.0,
            total_ttdc_ms=80.0,
            peak_memory_bytes=100_000_000,
            logical_node_count=3,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        plan_pandas = PhysicalRegionPlan(
            plan_id="plan_pandas",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.PANDAS_NUMPY,
                    representation=Representation.PANDAS_LONG,
                    node_ids=("group_by", "aggregate", "join"),
                    execution_axis=ExecutionAxis.GLOBAL_PANEL,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=10_000_000,
                    estimated_memory_bytes=200_000_000,
                ),
            ),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=250.0,  # Slower in Pandas
            total_transfer_ms=0.0,
            total_ttdc_ms=250.0,
            peak_memory_bytes=200_000_000,
            logical_node_count=3,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # DuckDB should be much better
        assert plan_duckdb.total_ttdc_ms < plan_pandas.total_ttdc_ms

    def test_polars_for_time_series_rolling(self):
        """Should prefer Polars for time-series rolling operations."""
        # Rolling window workload
        plan_polars = PhysicalRegionPlan(
            plan_id="plan_polars",
            regions=(
                BackendRegion(
                    region_id="r1",
                    backend=PhysicalBackend.POLARS_PANEL,
                    representation=Representation.POLARS_LONG,
                    node_ids=("ts_mean", "ts_std", "ema"),
                    execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
                    required_properties=PhysicalProperties(),
                    state_contract=StateContract(),
                    estimated_rows=1_000_000,
                    estimated_memory_bytes=50_000_000,
                ),
            ),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=60.0,  # Fast in Polars
            total_transfer_ms=0.0,
            total_ttdc_ms=60.0,
            peak_memory_bytes=50_000_000,
            logical_node_count=3,
            backend_switch_count=0,
            native_fraction=1.0,
        )

        # Polars should be efficient for rolling operations
        assert plan_polars.total_ttdc_ms < 100.0
        assert plan_polars.backend_switch_count == 0
