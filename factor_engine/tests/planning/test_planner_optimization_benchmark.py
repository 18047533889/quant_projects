# -*- coding: utf-8 -*-
"""Comprehensive planner optimization benchmarks.

Benchmarks the optimized planner layer across various workloads:
    1. Small data (1K rows): backend selection overhead
    2. Medium data (100K rows): mixed workload
    3. Large data (10M rows): throughput and memory
    4. Complex DAG (100 nodes, 20 shared): optimization time
    5. Before/after comparisons for each optimization

Run with: pytest tests/planning/test_planner_optimization_benchmark.py -v -s
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from factor_engine.planning.backend_region import ExecutionAxis, PhysicalBackend
from factor_engine.planning.backend_selector import (
    IntelligentBackendSelector,
    OperatorProfile,
    RoutingContext,
)
from factor_engine.planning.cost_model_v2 import (
    CostBreakdown,
    audit_double_count,
    build_cost_breakdown,
    compute_shared_benefit,
    estimate_compute_cost,
)
from factor_engine.planning.hybrid_execution_planner import (
    FactorSpec,
    HybridExecutionPlanner,
    NodeMetadata,
)
from factor_engine.planning.memory_model import DataShapeEstimate, estimate_shape_from_metadata
from factor_engine.planning.query_optimizer import QueryNode, QueryOptimizer
from factor_engine.planning.region_optimizer import NodeCost, RegionOptimizer


class BenchmarkResult:
    """Benchmark result container."""

    def __init__(self, name: str):
        self.name = name
        self.execution_time_ms: float = 0.0
        self.memory_estimate_mb: float = 0.0
        self.optimization_time_ms: float = 0.0
        self.metrics: dict[str, Any] = {}

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.name}: "
            f"exec={self.execution_time_ms:.2f}ms, "
            f"mem={self.memory_estimate_mb:.1f}MB, "
            f"opt={self.optimization_time_ms:.2f}ms)"
        )


@pytest.fixture
def small_data_shape() -> DataShapeEstimate:
    """Small data: 1K rows."""
    return estimate_shape_from_metadata(
        dates=10,
        instruments=100,
        columns=5,
        frequency="daily",
        density=1.0,
    )


@pytest.fixture
def medium_data_shape() -> DataShapeEstimate:
    """Medium data: 100K rows."""
    return estimate_shape_from_metadata(
        dates=100,
        instruments=1000,
        columns=10,
        frequency="daily",
        density=0.95,
    )


@pytest.fixture
def large_data_shape() -> DataShapeEstimate:
    """Large data: 10M rows."""
    return estimate_shape_from_metadata(
        dates=252,
        instruments=5000,
        columns=20,
        frequency="minute",
        bars_per_session=240,
        density=0.9,
    )


class TestBackendSelectorBenchmark:
    """Benchmark backend selector optimizations."""

    def test_small_data_backend_selection(self, small_data_shape):
        """Benchmark backend selection overhead on small data."""
        selector = IntelligentBackendSelector()

        ctx = RoutingContext(
            estimated_rows=small_data_shape.estimated_rows,
            estimated_columns=small_data_shape.estimated_columns,
            estimated_bytes=small_data_shape.estimated_bytes,
            available_memory_bytes=8 * 1024**3,  # 8GB
            operator_profile=OperatorProfile.ELEMENTWISE_HEAVY,
            execution_axis=ExecutionAxis.GLOBAL_PANEL,
            window_params=[],
            has_regression=False,
            parent_backend=None,
            allows_streaming=True,
            performance_priority="speed",
        )

        # Warmup
        selector.select_backend(ctx)

        # Benchmark
        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            decision = selector.select_backend(ctx)

        elapsed = (time.perf_counter() - start) * 1000  # ms
        avg_time = elapsed / iterations

        print(f"\nSmall data backend selection: {avg_time:.4f}ms per call")
        print(f"  Chosen: {decision.chosen_backend.value}")
        print(f"  Cost: {decision.estimated_cost_ms:.2f}ms")
        print(f"  Confidence: {decision.confidence_score:.2f}")

        # Should be fast (< 1ms per call)
        assert avg_time < 1.0, f"Selection too slow: {avg_time:.4f}ms"

    def test_large_data_backend_selection(self, large_data_shape):
        """Benchmark backend selection on large data."""
        selector = IntelligentBackendSelector()

        ctx = RoutingContext(
            estimated_rows=large_data_shape.estimated_rows,
            estimated_columns=large_data_shape.estimated_columns,
            estimated_bytes=large_data_shape.estimated_bytes,
            available_memory_bytes=16 * 1024**3,  # 16GB
            operator_profile=OperatorProfile.WINDOW_HEAVY,
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            window_params=[20, 60, 120],
            has_regression=False,
            parent_backend=None,
            allows_streaming=True,
            performance_priority="balanced",
        )

        start = time.perf_counter()
        decision = selector.select_backend(ctx)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nLarge data backend selection: {elapsed:.4f}ms")
        print(f"  Chosen: {decision.chosen_backend.value}")
        print(f"  Cost: {decision.estimated_cost_ms:.2f}ms")
        print(f"  Memory: {decision.memory_footprint_bytes / 1024**2:.1f}MB")

        # Should prefer streaming backends for large data
        assert decision.chosen_backend in (
            PhysicalBackend.POLARS_LAZY,
            PhysicalBackend.DUCKDB_SQL,
        )

    def test_conversion_cost_matrix(self):
        """Test conversion cost matrix optimization."""
        selector = IntelligentBackendSelector()

        # Test all backend pairs
        backends = [
            PhysicalBackend.PANDAS_NUMPY,
            PhysicalBackend.POLARS_EAGER,
            PhysicalBackend.POLARS_LAZY,
            PhysicalBackend.DUCKDB_SQL,
        ]

        print("\nConversion cost matrix (100MB data):")
        print("From/To     ", "  ".join([b.value[:8] for b in backends]))

        test_bytes = 100 * 1024 * 1024  # 100MB
        for source in backends:
            costs = []
            for target in backends:
                if source == target:
                    costs.append("    -   ")
                else:
                    cost = selector._get_conversion_cost(source, target, test_bytes)
                    costs.append(f"{cost:>7.1f}ms")
            print(f"{source.value[:12]:<12}", " ".join(costs))


class TestCostModelBenchmark:
    """Benchmark cost model optimizations."""

    def test_operator_specific_costs(self, medium_data_shape):
        """Test operator-specific cost functions."""
        operators = [
            ("ts_rank", 20),
            ("cs_zscore", None),
            ("rolling_mean", 60),
            ("regression", 5),
            ("correlation", 20),
            ("add", None),
        ]

        print("\nOperator-specific costs (100K rows):")
        for op, window in operators:
            cost = estimate_compute_cost(
                operator=op,
                backend="polars_lazy",
                shape=medium_data_shape,
                window=window,
                feature_dim=5 if "regress" in op else None,
            )
            print(f"  {op:20s}: {cost:>8.2f}ms")

    def test_hardware_aware_cost(self, large_data_shape):
        """Test hardware-aware cost estimation."""
        print("\nHardware-aware costs (10M rows, window=20):")
        for cores in [2, 4, 8, 16]:
            cost = estimate_compute_cost(
                operator="rolling_mean",
                backend="polars_lazy",
                shape=large_data_shape,
                window=20,
                cpu_cores=cores,
            )
            print(f"  {cores:2d} cores: {cost:>8.2f}ms")

    def test_cost_breakdown_with_confidence(self, medium_data_shape):
        """Test cost breakdown with confidence intervals."""
        breakdown = build_cost_breakdown(
            source_scan_ms=50.0,
            compute_ms=200.0,
            materialize_ms=30.0,
            schedule_overhead_ms=5.0,
        )

        audit = audit_double_count(breakdown)

        print("\nCost breakdown with confidence:")
        print(f"  Total: {audit['total_ms']:.2f}ms")
        print("  Confidence intervals:")
        for key, (low, high) in audit["confidence_intervals"].items():
            print(f"    {key}: [{low:.1f}, {high:.1f}]ms")


class TestRegionOptimizerBenchmark:
    """Benchmark region optimizer optimizations."""

    def _create_simple_dag(
        self, num_nodes: int, shared_ratio: float = 0.2
    ) -> tuple[dict[str, list[str]], dict[str, dict[PhysicalBackend, NodeCost]]]:
        """Create a simple DAG for testing."""
        node_graph: dict[str, list[str]] = {}
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]] = {}

        # Create linear chain with some shared nodes
        for i in range(num_nodes):
            node_id = f"n{i}"
            if i < num_nodes - 1:
                node_graph[node_id] = [f"n{i+1}"]
            else:
                node_graph[node_id] = []

            # Add costs for all backends
            node_costs[node_id] = {
                PhysicalBackend.PANDAS_NUMPY: NodeCost(
                    node_id, PhysicalBackend.PANDAS_NUMPY, 10.0 + i * 2, 1024 * 1024
                ),
                PhysicalBackend.POLARS_EAGER: NodeCost(
                    node_id, PhysicalBackend.POLARS_EAGER, 8.0 + i * 1.5, 1024 * 1024 * 1.5
                ),
                PhysicalBackend.DUCKDB_SQL: NodeCost(
                    node_id, PhysicalBackend.DUCKDB_SQL, 6.0 + i, 1024 * 1024 * 1.2
                ),
            }

        # Add shared nodes
        num_shared = int(num_nodes * shared_ratio)
        for i in range(num_shared):
            shared_id = f"s{i}"
            target_idx = i * (num_nodes // max(num_shared, 1))
            if target_idx < num_nodes:
                node_graph[f"n{target_idx}"].append(shared_id)
                node_graph[shared_id] = []
                node_costs[shared_id] = node_costs[f"n0"]  # Reuse costs

        return node_graph, node_costs

    def test_small_dag_optimization(self):
        """Benchmark optimization on small DAG (10 nodes)."""
        node_graph, node_costs = self._create_simple_dag(10, shared_ratio=0.1)

        optimizer = RegionOptimizer(
            memory_budget=1024 * 1024 * 1024,  # 1GB
            enable_cse_aware=True,
            enable_parallel_detection=True,
            enable_aggressive_fusion=True,
        )

        start = time.perf_counter()
        plan = optimizer.optimize(
            node_graph=node_graph,
            node_costs=node_costs,
            root_ids=["n0"],
            logical_hash="test_small",
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nSmall DAG optimization (10 nodes): {elapsed:.2f}ms")
        print(f"  Regions: {plan.region_count}")
        print(f"  Backend switches: {plan.backend_switch_count}")
        print(f"  Estimated TTDC: {plan.estimated_ttdc_ms:.2f}ms")

        assert elapsed < 200, f"Optimization too slow: {elapsed:.2f}ms"

    def test_medium_dag_optimization(self):
        """Benchmark optimization on medium DAG (50 nodes)."""
        node_graph, node_costs = self._create_simple_dag(50, shared_ratio=0.15)

        optimizer = RegionOptimizer(
            memory_budget=2 * 1024 * 1024 * 1024,  # 2GB
            enable_cse_aware=True,
            enable_parallel_detection=True,
            enable_aggressive_fusion=True,
        )

        start = time.perf_counter()
        plan = optimizer.optimize(
            node_graph=node_graph,
            node_costs=node_costs,
            root_ids=["n0"],
            logical_hash="test_medium",
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nMedium DAG optimization (50 nodes): {elapsed:.2f}ms")
        print(f"  Regions: {plan.region_count}")
        print(f"  Backend switches: {plan.backend_switch_count}")
        print(f"  Estimated TTDC: {plan.estimated_ttdc_ms:.2f}ms")

        assert elapsed < 1000, f"Optimization too slow: {elapsed:.2f}ms"

    def test_complex_dag_optimization(self):
        """Benchmark optimization on complex DAG (100 nodes, 20 shared)."""
        node_graph, node_costs = self._create_simple_dag(100, shared_ratio=0.2)

        optimizer = RegionOptimizer(
            memory_budget=4 * 1024 * 1024 * 1024,  # 4GB
            enable_cse_aware=True,
            enable_parallel_detection=True,
            enable_aggressive_fusion=True,
        )

        start = time.perf_counter()
        plan = optimizer.optimize(
            node_graph=node_graph,
            node_costs=node_costs,
            root_ids=["n0"],
            logical_hash="test_complex",
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nComplex DAG optimization (100 nodes, 20 shared): {elapsed:.2f}ms")
        print(f"  Regions: {plan.region_count}")
        print(f"  Backend switches: {plan.backend_switch_count}")
        print(f"  Shared nodes: {plan.shared_node_count}")
        print(f"  Estimated TTDC: {plan.estimated_ttdc_ms:.2f}ms")

        assert elapsed < 5000, f"Optimization too slow: {elapsed:.2f}ms"

    def test_dp_pruning_effectiveness(self):
        """Test DP pruning reduces state space."""
        node_graph, node_costs = self._create_simple_dag(30)

        # Without pruning
        optimizer_no_prune = RegionOptimizer(
            enable_cse_aware=False,
            enable_parallel_detection=False,
            enable_aggressive_fusion=False,
        )
        start = time.perf_counter()
        plan_no_prune = optimizer_no_prune.optimize(
            node_graph, node_costs, ["n0"], logical_hash="no_prune"
        )
        time_no_prune = (time.perf_counter() - start) * 1000

        # With pruning
        optimizer_with_prune = RegionOptimizer(
            enable_cse_aware=True,
            enable_parallel_detection=True,
            enable_aggressive_fusion=True,
        )
        start = time.perf_counter()
        plan_with_prune = optimizer_with_prune.optimize(
            node_graph, node_costs, ["n0"], logical_hash="with_prune"
        )
        time_with_prune = (time.perf_counter() - start) * 1000

        print(f"\nDP Pruning effectiveness (30 nodes):")
        print(f"  Without optimizations: {time_no_prune:.2f}ms")
        print(f"  With optimizations: {time_with_prune:.2f}ms")
        print(f"  Speedup: {time_no_prune / time_with_prune:.2f}x")

        # With optimizations should be faster
        assert time_with_prune <= time_no_prune * 1.1  # Allow 10% margin


class TestQueryOptimizerBenchmark:
    """Benchmark query optimizer."""

    def test_constant_folding(self):
        """Test constant folding optimization."""
        query_dag = {
            "n1": QueryNode("n1", "constant", {"value": 2}, (), ("x",), is_constant=True),
            "n2": QueryNode("n2", "constant", {"value": 3}, (), ("y",), is_constant=True),
            "n3": QueryNode("n3", "add", {}, ("n1", "n2"), ("z",)),
            "n4": QueryNode("n4", "multiply", {}, ("n3", "n1"), ("w",)),
        }

        optimizer = QueryOptimizer(enable_constant_folding=True)

        start = time.perf_counter()
        optimized = optimizer.optimize(query_dag, ["n4"])
        elapsed = (time.perf_counter() - start) * 1000

        stats = optimizer.get_optimization_stats()

        print(f"\nConstant folding: {elapsed:.4f}ms")
        print(f"  Folded: {stats['constant_folding']} expressions")
        print(f"  Nodes before: {len(query_dag)}")
        print(f"  Nodes after: {len(optimized)}")

    def test_cse_elimination(self):
        """Test CSE elimination."""
        # Create DAG with duplicate subexpressions
        query_dag = {
            "n1": QueryNode("n1", "scan", {"table": "data"}, (), ("x", "y")),
            "n2": QueryNode("n2", "filter", {"pred": "x>0"}, ("n1",), ("x", "y")),
            "n3": QueryNode("n3", "filter", {"pred": "x>0"}, ("n1",), ("x", "y")),  # Duplicate
            "n4": QueryNode("n4", "aggregate", {}, ("n2",), ("sum_y",)),
            "n5": QueryNode("n5", "aggregate", {}, ("n3",), ("sum_y",)),  # Uses duplicate
        }

        optimizer = QueryOptimizer(enable_cse=True)

        start = time.perf_counter()
        optimized = optimizer.optimize(query_dag, ["n4", "n5"])
        elapsed = (time.perf_counter() - start) * 1000

        stats = optimizer.get_optimization_stats()

        print(f"\nCSE elimination: {elapsed:.4f}ms")
        print(f"  Eliminated: {stats['cse_eliminations']} duplicates")

    def test_full_optimization_pipeline(self):
        """Test full optimization pipeline."""
        # Create complex query DAG
        query_dag = {}
        num_nodes = 50
        for i in range(num_nodes):
            query_dag[f"n{i}"] = QueryNode(
                f"n{i}",
                "operator",
                {},
                (f"n{i-1}",) if i > 0 else (),
                ("col",),
            )

        optimizer = QueryOptimizer(
            enable_constant_folding=True,
            enable_cse=True,
            enable_predicate_pushdown=True,
            enable_projection_pushdown=True,
            enable_fusion=True,
            enable_dead_code_elimination=True,
        )

        start = time.perf_counter()
        optimized = optimizer.optimize(query_dag, ["n49"])
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nFull optimization pipeline (50 nodes): {elapsed:.2f}ms")
        print(f"  Stats: {optimizer.get_optimization_stats()}")


class TestIntegratedBenchmark:
    """Integrated end-to-end benchmarks."""

    def test_end_to_end_small_workload(self):
        """End-to-end benchmark: small workload."""
        print("\n" + "=" * 60)
        print("END-TO-END BENCHMARK: Small Workload (1K rows)")
        print("=" * 60)

        # Setup
        planner = HybridExecutionPlanner(
            memory_budget=1024 * 1024 * 1024,  # 1GB
            enable_parallel_regions=True,
            enable_data_locality=True,
            enable_streaming_first=False,
        )

        factors = [
            FactorSpec(
                factor_id="f1",
                root_node_id="n5",
                estimated_rows=1000,
                estimated_columns=5,
                estimated_bytes=40000,
                operator_names=["add", "multiply", "ts_rank"],
                window_params=[20],
            )
        ]

        node_graph = {
            "n1": [],
            "n2": ["n1"],
            "n3": ["n1"],
            "n4": ["n2", "n3"],
            "n5": ["n4"],
        }

        node_metadata = {
            f"n{i}": NodeMetadata(
                node_id=f"n{i}",
                operator_name="test_op",
                estimated_rows=1000,
                estimated_bytes=8000,
                window=20,
                is_shared=False,
                consumer_count=1,
                depth=i,
            )
            for i in range(1, 6)
        }

        # Benchmark
        start = time.perf_counter()
        plan = planner.plan_hybrid_execution(
            factors, node_graph, node_metadata, available_memory_bytes=1024 * 1024 * 1024
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Planning time: {elapsed:.2f}ms")
        print(f"Estimated TTDC: {plan.estimated_ttdc_ms:.2f}ms")
        print(f"Regions: {plan.region_count}")
        print(f"Peak memory: {plan.peak_memory_estimate / 1024**2:.1f}MB")

    def test_end_to_end_large_workload(self):
        """End-to-end benchmark: large workload."""
        print("\n" + "=" * 60)
        print("END-TO-END BENCHMARK: Large Workload (10M rows)")
        print("=" * 60)

        planner = HybridExecutionPlanner(
            memory_budget=8 * 1024 * 1024 * 1024,  # 8GB
            enable_parallel_regions=True,
            enable_data_locality=True,
            enable_streaming_first=True,
        )

        factors = [
            FactorSpec(
                factor_id=f"f{i}",
                root_node_id=f"n{i*10}",
                estimated_rows=10_000_000,
                estimated_columns=20,
                estimated_bytes=1_600_000_000,
                operator_names=["rolling_mean", "cs_rank", "ts_zscore"],
                window_params=[20, 60],
            )
            for i in range(3)
        ]

        # Create larger graph
        node_graph = {}
        for i in range(30):
            if i > 0:
                node_graph[f"n{i}"] = [f"n{i-1}"]
            else:
                node_graph[f"n{i}"] = []

        node_metadata = {
            f"n{i}": NodeMetadata(
                node_id=f"n{i}",
                operator_name="rolling_mean" if i % 2 == 0 else "cs_rank",
                estimated_rows=10_000_000,
                estimated_bytes=160_000_000,
                window=60 if i % 2 == 0 else None,
                is_shared=i < 5,
                consumer_count=2 if i < 5 else 1,
                depth=i,
            )
            for i in range(30)
        }

        # Benchmark
        start = time.perf_counter()
        plan = planner.plan_hybrid_execution(
            factors, node_graph, node_metadata, available_memory_bytes=8 * 1024 * 1024 * 1024
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Planning time: {elapsed:.2f}ms")
        print(f"Estimated TTDC: {plan.estimated_ttdc_ms:.2f}ms")
        print(f"Regions: {plan.region_count}")
        print(f"Peak memory: {plan.peak_memory_estimate / 1024**2:.1f}MB")
        print(f"Backend switches: {plan.backend_switch_count}")


if __name__ == "__main__":
    # Run benchmarks directly
    pytest.main([__file__, "-v", "-s"])
