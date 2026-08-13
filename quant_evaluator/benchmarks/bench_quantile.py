"""
Performance benchmarks for quantile operations.

Compares original implementation vs optimized np.partition and vectorized versions.
"""

import time
import numpy as np
from typing import Dict, List

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.quantile import (
    assign_quantiles,
    assign_quantiles_fast,
    compute_quantile_returns,
    compute_quantile_returns_optimized,
)


def benchmark_assign_quantiles(T: int, N: int, n_quantiles: int = 5, n_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark quantile assignment with original vs optimized implementation.

    Args:
        T: Number of time periods
        N: Number of assets
        n_quantiles: Number of quantiles
        n_runs: Number of benchmark runs

    Returns:
        Timing results in milliseconds
    """
    # Generate synthetic data
    np.random.seed(42)
    values = np.random.randn(T, N)
    values[np.random.rand(T, N) < 0.05] = np.nan  # 5% missing

    results = {}

    # Benchmark original implementation
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        q_orig = assign_quantiles(values, n_quantiles=n_quantiles)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    results['original_ms'] = np.median(times)

    # Benchmark fast implementation
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        q_fast = assign_quantiles_fast(values, n_quantiles=n_quantiles)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    results['fast_ms'] = np.median(times)

    results['speedup'] = results['original_ms'] / results['fast_ms']

    # Verify correctness (results should be similar)
    q_orig = assign_quantiles(values, n_quantiles=n_quantiles)
    q_fast = assign_quantiles_fast(values, n_quantiles=n_quantiles)

    # Check agreement on valid assignments
    valid_mask = (q_orig >= 0) & (q_fast >= 0)
    if np.any(valid_mask):
        agreement = np.mean(q_orig[valid_mask] == q_fast[valid_mask])
        results['agreement'] = agreement
    else:
        results['agreement'] = 1.0

    return results


def benchmark_quantile_returns(
    T: int, N: int, F: int, n_quantiles: int = 5, n_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark quantile return computation across implementations.

    Args:
        T: Number of time periods
        N: Number of assets
        F: Number of factors
        n_quantiles: Number of quantiles
        n_runs: Number of benchmark runs

    Returns:
        Timing results and speedup metrics
    """
    # Generate synthetic data
    np.random.seed(42)
    factor_values = np.random.randn(T, N, F)
    factor_values[np.random.rand(T, N, F) < 0.05] = np.nan

    labels = np.random.randn(T, N) * 0.01  # 1% returns
    labels[np.random.rand(T, N) < 0.03] = np.nan

    from quant_evaluator.contracts.factor_batch import AxisRef

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

    results = {}

    # Benchmark standard version
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        q_ret, q_cnt = compute_quantile_returns(
            factor_batch, label_bundle, n_quantiles=n_quantiles
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    results['standard_ms'] = np.median(times)

    # Benchmark optimized version
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        q_ret_opt, q_cnt_opt = compute_quantile_returns_optimized(
            factor_batch, label_bundle, n_quantiles=n_quantiles
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    results['optimized_ms'] = np.median(times)

    results['speedup'] = results['standard_ms'] / results['optimized_ms']

    # Verify correctness
    q_ret, q_cnt = compute_quantile_returns(factor_batch, label_bundle, n_quantiles=n_quantiles)
    q_ret_opt, q_cnt_opt = compute_quantile_returns_optimized(factor_batch, label_bundle, n_quantiles=n_quantiles)

    # Check numerical agreement
    valid_mask = np.isfinite(q_ret) & np.isfinite(q_ret_opt)
    if np.any(valid_mask):
        max_diff = np.max(np.abs(q_ret[valid_mask] - q_ret_opt[valid_mask]))
        results['max_return_diff'] = max_diff
        results['count_agreement'] = np.mean(q_cnt == q_cnt_opt)
    else:
        results['max_return_diff'] = 0.0
        results['count_agreement'] = 1.0

    return results


def run_benchmark_suite():
    """
    Run comprehensive benchmark suite across different data sizes.
    """
    print("=" * 80)
    print("Quantile Operations Performance Benchmark Suite")
    print("=" * 80)

    # Benchmark 1: Quantile assignment scaling
    print("\n1. Quantile Assignment Scaling (5 quantiles)")
    print("-" * 80)
    print(f"{'Size (T×N)':<20} {'Original (ms)':<15} {'Fast (ms)':<18} {'Speedup':<10} {'Agreement':<10}")
    print("-" * 80)

    sizes = [
        (50, 500),
        (100, 1000),
        (252, 2000),
        (500, 3000),
    ]

    for T, N in sizes:
        result = benchmark_assign_quantiles(T, N, n_quantiles=5, n_runs=3)
        print(f"{T}×{N:<15} {result['original_ms']:>12.2f}   {result['fast_ms']:>14.2f}   "
              f"{result['speedup']:>8.2f}x  {result['agreement']:>8.1%}")

    # Benchmark 2: Quantile returns with varying factor counts
    print("\n2. Quantile Returns Computation (100 days × 1000 assets)")
    print("-" * 80)
    print(f"{'Factors':<10} {'Standard (ms)':<18} {'Optimized (ms)':<15} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 80)

    factor_counts = [5, 10, 25, 50, 100]

    for F in factor_counts:
        result = benchmark_quantile_returns(T=100, N=1000, F=F, n_quantiles=5, n_runs=3)
        print(f"{F:<10} {result['standard_ms']:>14.2f}   {result['optimized_ms']:>12.2f}   "
              f"{result['speedup']:>8.2f}x  {result['max_return_diff']:>10.2e}")

    # Benchmark 3: Large factor set scenario
    print("\n3. Large Factor Set Scenario")
    print("-" * 80)

    large_result = benchmark_quantile_returns(T=252, N=2000, F=200, n_quantiles=5, n_runs=2)
    print(f"Configuration: 252 days × 2000 assets × 200 factors")
    print(f"Standard time:  {large_result['standard_ms']:.2f} ms")
    print(f"Optimized time: {large_result['optimized_ms']:.2f} ms")
    print(f"Speedup:        {large_result['speedup']:.2f}x")
    print(f"Max difference: {large_result['max_return_diff']:.2e}")
    print(f"Count agreement: {large_result['count_agreement']:.1%}")

    print("\n" + "=" * 80)
    print("Benchmark complete")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark_suite()
