"""
Final performance benchmark comparing all quantile implementations.

Tests original vs optimized implementations to demonstrate 5-10x speedup.
"""

import time
import numpy as np
from typing import Dict

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.quantile import assign_quantiles, compute_quantile_returns
from quant_evaluator.metrics.quantile_optimized import (
    assign_quantiles_vectorized_v2,
    compute_quantile_returns_ultra_fast,
)


def create_test_data(T: int, N: int, F: int):
    """Create synthetic factor and label data."""
    np.random.seed(42)

    factor_values = np.random.randn(T, N, F)
    factor_values[np.random.rand(T, N, F) < 0.05] = np.nan

    labels = np.random.randn(T, N) * 0.01
    labels[np.random.rand(T, N) < 0.03] = np.nan

    factor_batch = FactorBatch(
        factor_ids=tuple(f"factor_{i}" for i in range(F)),
        time_axis=AxisRef(name="time", dtype="int64", size=T, values=np.arange(T)),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N, values=np.arange(N)),
        values=factor_values,
    )

    label_bundle = LabelBundle(
        target_id="forward_return_1d",
        values=labels,
        horizon=1,
        execution_delay=0,
        decision_time=tuple(range(T)),
        execution_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(T)),
    )

    return factor_batch, label_bundle


def benchmark_full_pipeline(T: int, N: int, F: int, n_runs: int = 3) -> Dict[str, float]:
    """
    Benchmark complete quantile analysis pipeline.

    Returns:
        Timing results and speedup metrics
    """
    factor_batch, label_bundle = create_test_data(T, N, F)

    results = {}

    # Benchmark original implementation
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        q_ret_orig, q_cnt_orig = compute_quantile_returns(
            factor_batch, label_bundle, n_quantiles=5
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    results['original_ms'] = np.median(times)

    # Benchmark optimized implementation
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        q_ret_opt, q_cnt_opt = compute_quantile_returns_ultra_fast(
            factor_batch, label_bundle, n_quantiles=5
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    results['optimized_ms'] = np.median(times)

    results['speedup'] = results['original_ms'] / results['optimized_ms']

    # Verify correctness
    q_ret_orig, q_cnt_orig = compute_quantile_returns(factor_batch, label_bundle, n_quantiles=5)
    q_ret_opt, q_cnt_opt = compute_quantile_returns_ultra_fast(factor_batch, label_bundle, n_quantiles=5)

    valid_mask = np.isfinite(q_ret_orig) & np.isfinite(q_ret_opt)
    if np.any(valid_mask):
        max_diff = np.max(np.abs(q_ret_orig[valid_mask] - q_ret_opt[valid_mask]))
        results['max_diff'] = max_diff
        results['count_match'] = np.mean(q_cnt_orig == q_cnt_opt)
    else:
        results['max_diff'] = 0.0
        results['count_match'] = 1.0

    return results


def benchmark_assignment_only(T: int, N: int, n_runs: int = 5) -> Dict[str, float]:
    """Benchmark just quantile assignment step."""
    np.random.seed(42)
    values = np.random.randn(T, N)
    values[np.random.rand(T, N) < 0.05] = np.nan

    results = {}

    # Original
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        q_orig = assign_quantiles(values, n_quantiles=5)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    results['original_ms'] = np.median(times)

    # Optimized
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        q_opt = assign_quantiles_vectorized_v2(values, n_quantiles=5)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    results['optimized_ms'] = np.median(times)

    results['speedup'] = results['original_ms'] / results['optimized_ms']

    # Check agreement
    q_orig = assign_quantiles(values, n_quantiles=5)
    q_opt = assign_quantiles_vectorized_v2(values, n_quantiles=5)
    valid = (q_orig >= 0) & (q_opt >= 0)
    results['agreement'] = np.mean(q_orig[valid] == q_opt[valid]) if np.any(valid) else 1.0

    return results


def run_comprehensive_benchmark():
    """Run full benchmark suite."""
    print("=" * 90)
    print("QUANTILE OPERATIONS PERFORMANCE BENCHMARK - FINAL RESULTS")
    print("=" * 90)

    # Test 1: Quantile assignment scaling
    print("\n1. Quantile Assignment (5 quantiles)")
    print("-" * 90)
    print(f"{'Config (T×N)':<25} {'Original (ms)':<18} {'Optimized (ms)':<18} {'Speedup':<12} {'Agreement':<12}")
    print("-" * 90)

    configs = [(50, 500), (100, 1000), (252, 2000), (500, 3000)]

    for T, N in configs:
        result = benchmark_assignment_only(T, N, n_runs=5)
        print(f"{T}×{N:<20} {result['original_ms']:>14.2f}    {result['optimized_ms']:>14.2f}    "
              f"{result['speedup']:>8.2f}x    {result['agreement']:>8.1%}")

    # Test 2: Full pipeline with varying factor counts
    print("\n2. Full Quantile Return Pipeline (100 days × 1000 assets)")
    print("-" * 90)
    print(f"{'Factors':<12} {'Original (ms)':<18} {'Optimized (ms)':<18} {'Speedup':<12} {'Max Diff':<12}")
    print("-" * 90)

    factor_counts = [5, 10, 25, 50, 100, 200]

    for F in factor_counts:
        result = benchmark_full_pipeline(T=100, N=1000, F=F, n_runs=3)
        print(f"{F:<12} {result['original_ms']:>14.2f}    {result['optimized_ms']:>14.2f}    "
              f"{result['speedup']:>8.2f}x    {result['max_diff']:>10.2e}")

    # Test 3: Large-scale scenario
    print("\n3. Large-Scale Production Scenario")
    print("-" * 90)

    large_configs = [
        (252, 2000, 100, "Year × Mid-cap × 100 factors"),
        (252, 3000, 200, "Year × Large-cap × 200 factors"),
        (500, 2000, 50, "2 Years × Mid-cap × 50 factors"),
    ]

    for T, N, F, desc in large_configs:
        result = benchmark_full_pipeline(T, N, F, n_runs=2)
        print(f"\n{desc}")
        print(f"  Configuration: {T} days × {N} assets × {F} factors")
        print(f"  Original time:  {result['original_ms']:>10.2f} ms ({result['original_ms']/1000:.2f}s)")
        print(f"  Optimized time: {result['optimized_ms']:>10.2f} ms ({result['optimized_ms']/1000:.2f}s)")
        print(f"  Speedup:        {result['speedup']:>10.2f}x")
        print(f"  Time saved:     {result['original_ms'] - result['optimized_ms']:>10.2f} ms")
        print(f"  Max difference: {result['max_diff']:>10.2e}")

    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    run_comprehensive_benchmark()
