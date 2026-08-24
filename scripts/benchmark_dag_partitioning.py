#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark DAG partitioning optimization.

测试多后端 DAG 执行优化的效果：
    1. 对比优化前后的后端切换次数
    2. 对比转换开销占比
    3. 测试批量转换识别
    4. 验证 pipeline fusion 效果
"""
import time
from dataclasses import dataclass

from factor_engine.planning.dag_partition_optimizer import (
    DAGNode,
    DAGPartitionOptimizer,
    DataShape,
    batch_conversion_opportunities,
)


@dataclass
class BenchmarkResult:
    """Benchmark 结果。"""
    scenario: str
    num_nodes: int
    num_roots: int
    compute_cost_ms: float
    conversion_cost_ms: float
    total_cost_ms: float
    num_partitions: int
    num_switches: int
    conversion_overhead_pct: float
    fusion_enabled: bool
    optimization_time_ms: float


def create_simple_dag(num_factors: int = 10) -> tuple[dict[str, DAGNode], list[str]]:
    """创建简单 DAG：多个因子共享一些底层算子。

    结构：
        source_cols (shared) → ts_mean (per factor) → cs_rank (per factor) → root
    """
    nodes: dict[str, DAGNode] = {}
    root_ids = []

    # 共享的源列读取节点
    for i in range(3):
        node_id = f"column_{i}"
        nodes[node_id] = DAGNode(
            node_id=node_id,
            operator="column",
            backend_candidates=["pandas_numpy", "polars", "duckdb_sql"],
            compute_costs={
                "pandas_numpy": 5.0,
                "polars": 3.0,
                "duckdb_sql": 2.0,
            },
            shape=DataShape(rows=100_000, cols=1),
            is_shared=True,
            consumer_count=num_factors,
        )

    # 每个因子的计算链
    for f in range(num_factors):
        # ts_mean 节点
        ts_node_id = f"ts_mean_f{f}"
        nodes[ts_node_id] = DAGNode(
            node_id=ts_node_id,
            operator="ts_mean",
            backend_candidates=["pandas_numpy", "polars", "duckdb_sql"],
            compute_costs={
                "pandas_numpy": 15.0,
                "polars": 8.0,
                "duckdb_sql": 10.0,
            },
            shape=DataShape(rows=100_000, cols=1),
            children=["column_0"],
        )
        nodes["column_0"].parents.append(ts_node_id)

        # cs_rank 节点
        cs_node_id = f"cs_rank_f{f}"
        nodes[cs_node_id] = DAGNode(
            node_id=cs_node_id,
            operator="cs_rank",
            backend_candidates=["pandas_numpy", "polars", "duckdb_sql"],
            compute_costs={
                "pandas_numpy": 12.0,
                "polars": 10.0,
                "duckdb_sql": 7.0,
            },
            shape=DataShape(rows=100_000, cols=1),
            children=[ts_node_id],
        )
        nodes[ts_node_id].parents.append(cs_node_id)

        # root 节点
        root_id = f"root_f{f}"
        nodes[root_id] = DAGNode(
            node_id=root_id,
            operator="add",
            backend_candidates=["pandas_numpy", "polars", "duckdb_sql"],
            compute_costs={
                "pandas_numpy": 3.0,
                "polars": 2.0,
                "duckdb_sql": 2.5,
            },
            shape=DataShape(rows=100_000, cols=1),
            children=[cs_node_id],
        )
        nodes[cs_node_id].parents.append(root_id)
        root_ids.append(root_id)

    return nodes, root_ids


def create_complex_dag(num_factors: int = 20) -> tuple[dict[str, DAGNode], list[str]]:
    """创建复杂 DAG：多层共享 + 多种算子类型。"""
    nodes: dict[str, DAGNode] = {}
    root_ids = []

    # Layer 1: 源数据（高度共享）
    for i in range(5):
        node_id = f"source_{i}"
        nodes[node_id] = DAGNode(
            node_id=node_id,
            operator="column",
            backend_candidates=["pandas_numpy", "polars", "duckdb_sql"],
            compute_costs={
                "pandas_numpy": 3.0,
                "polars": 2.0,
                "duckdb_sql": 1.5,
            },
            shape=DataShape(rows=200_000, cols=1),
            is_shared=True,
            consumer_count=num_factors // 2,
        )

    # Layer 2: 时序算子（部分共享）
    ts_nodes = []
    for i in range(num_factors // 2):
        node_id = f"ts_rolling_{i}"
        source_id = f"source_{i % 5}"
        nodes[node_id] = DAGNode(
            node_id=node_id,
            operator="ts_mean",
            backend_candidates=["pandas_numpy", "polars", "duckdb_sql"],
            compute_costs={
                "pandas_numpy": 20.0,
                "polars": 12.0,
                "duckdb_sql": 15.0,
            },
            shape=DataShape(rows=200_000, cols=1),
            children=[source_id],
            is_shared=True,
            consumer_count=2,
        )
        nodes[source_id].parents.append(node_id)
        ts_nodes.append(node_id)

    # Layer 3: 截面算子
    cs_nodes = []
    for i in range(num_factors):
        node_id = f"cs_op_{i}"
        ts_id = ts_nodes[i % len(ts_nodes)]
        nodes[node_id] = DAGNode(
            node_id=node_id,
            operator="cs_rank" if i % 2 == 0 else "cs_zscore",
            backend_candidates=["pandas_numpy", "polars", "duckdb_sql"],
            compute_costs={
                "pandas_numpy": 15.0,
                "polars": 12.0,
                "duckdb_sql": 9.0,
            },
            shape=DataShape(rows=200_000, cols=1),
            children=[ts_id],
        )
        nodes[ts_id].parents.append(node_id)
        cs_nodes.append(node_id)

    # Layer 4: 组合算子
    for i in range(num_factors):
        node_id = f"combine_{i}"
        cs_id = cs_nodes[i]
        nodes[node_id] = DAGNode(
            node_id=node_id,
            operator="add" if i % 3 == 0 else "multiply",
            backend_candidates=["pandas_numpy", "polars", "duckdb_sql"],
            compute_costs={
                "pandas_numpy": 5.0,
                "polars": 3.0,
                "duckdb_sql": 4.0,
            },
            shape=DataShape(rows=200_000, cols=1),
            children=[cs_id],
        )
        nodes[cs_id].parents.append(node_id)
        root_ids.append(node_id)

    return nodes, root_ids


def run_benchmark(
    scenario: str,
    nodes: dict[str, DAGNode],
    root_ids: list[str],
    enable_fusion: bool = True,
    conversion_penalty: float = 1.0,
) -> BenchmarkResult:
    """运行单个 benchmark。"""
    optimizer = DAGPartitionOptimizer(
        conversion_penalty_multiplier=conversion_penalty,
        max_partition_size=50,
        enable_fusion=enable_fusion,
    )

    start = time.time()
    plan = optimizer.optimize(nodes, root_ids)
    elapsed = (time.time() - start) * 1000

    return BenchmarkResult(
        scenario=scenario,
        num_nodes=len(nodes),
        num_roots=len(root_ids),
        compute_cost_ms=plan.total_compute_cost,
        conversion_cost_ms=plan.total_conversion_cost,
        total_cost_ms=plan.total_cost,
        num_partitions=len(plan.partitions),
        num_switches=plan.backend_switches,
        conversion_overhead_pct=plan.total_conversion_cost / plan.total_cost * 100 if plan.total_cost > 0 else 0,
        fusion_enabled=enable_fusion,
        optimization_time_ms=elapsed,
    )


def print_result(result: BenchmarkResult) -> None:
    """打印结果。"""
    print(f"\n{'='*70}")
    print(f"Scenario: {result.scenario}")
    print(f"{'='*70}")
    print(f"  Nodes: {result.num_nodes}, Roots: {result.num_roots}")
    print(f"  Compute cost: {result.compute_cost_ms:.1f} ms")
    print(f"  Conversion cost: {result.conversion_cost_ms:.1f} ms")
    print(f"  Total cost: {result.total_cost_ms:.1f} ms")
    print(f"  Partitions: {result.num_partitions}")
    print(f"  Backend switches: {result.num_switches}")
    print(f"  Conversion overhead: {result.conversion_overhead_pct:.1f}%")
    print(f"  Fusion enabled: {result.fusion_enabled}")
    print(f"  Optimization time: {result.optimization_time_ms:.1f} ms")


def compare_results(baseline: BenchmarkResult, optimized: BenchmarkResult) -> None:
    """对比结果。"""
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    cost_improvement = (baseline.total_cost_ms - optimized.total_cost_ms) / baseline.total_cost_ms * 100
    print(f"  Total cost reduction: {cost_improvement:.1f}%")

    switches_reduction = baseline.num_switches - optimized.num_switches
    print(f"  Backend switches reduction: {switches_reduction} ({baseline.num_switches} → {optimized.num_switches})")

    conv_overhead_reduction = baseline.conversion_overhead_pct - optimized.conversion_overhead_pct
    print(f"  Conversion overhead reduction: {conv_overhead_reduction:.1f}% ({baseline.conversion_overhead_pct:.1f}% → {optimized.conversion_overhead_pct:.1f}%)")


def test_batch_conversion_detection() -> None:
    """测试批量转换识别。"""
    print(f"\n{'='*70}")
    print("BATCH CONVERSION DETECTION TEST")
    print(f"{'='*70}")

    nodes, root_ids = create_simple_dag(num_factors=15)
    optimizer = DAGPartitionOptimizer(
        conversion_penalty_multiplier=0.5,  # 低惩罚：更多转换
        enable_fusion=False,
    )

    plan = optimizer.optimize(nodes, root_ids)
    opportunities = batch_conversion_opportunities(plan.partitions, nodes)

    print(f"  Total conversions: {plan.backend_switches}")
    print(f"  Batch opportunities: {len(opportunities)}")
    for key, node_list in opportunities.items():
        print(f"    {key}: {len(node_list)} nodes can share conversion")


def main() -> None:
    """主函数。"""
    print("="*70)
    print("DAG PARTITIONING OPTIMIZATION BENCHMARK")
    print("="*70)

    # Test 1: Simple DAG with different penalty settings
    print("\n\n### TEST 1: Simple DAG (10 factors) ###")
    nodes, root_ids = create_simple_dag(num_factors=10)

    baseline = run_benchmark(
        "Simple DAG - Low penalty (more switches)",
        nodes, root_ids,
        enable_fusion=False,
        conversion_penalty=0.5,
    )
    print_result(baseline)

    optimized = run_benchmark(
        "Simple DAG - High penalty (fewer switches)",
        nodes, root_ids,
        enable_fusion=True,
        conversion_penalty=2.0,
    )
    print_result(optimized)

    compare_results(baseline, optimized)

    # Test 2: Complex DAG
    print("\n\n### TEST 2: Complex DAG (20 factors) ###")
    nodes, root_ids = create_complex_dag(num_factors=20)

    baseline = run_benchmark(
        "Complex DAG - No fusion",
        nodes, root_ids,
        enable_fusion=False,
        conversion_penalty=1.0,
    )
    print_result(baseline)

    optimized = run_benchmark(
        "Complex DAG - With fusion",
        nodes, root_ids,
        enable_fusion=True,
        conversion_penalty=1.5,
    )
    print_result(optimized)

    compare_results(baseline, optimized)

    # Test 3: Scaling test
    print("\n\n### TEST 3: Scaling (50 factors) ###")
    nodes, root_ids = create_complex_dag(num_factors=50)

    result = run_benchmark(
        "Complex DAG - 50 factors",
        nodes, root_ids,
        enable_fusion=True,
        conversion_penalty=1.5,
    )
    print_result(result)

    # Test 4: Batch conversion detection
    test_batch_conversion_detection()

    print("\n\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
