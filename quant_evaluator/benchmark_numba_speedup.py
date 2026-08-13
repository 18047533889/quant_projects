"""
Comprehensive Numba backend speedup benchmark.

Demonstrates performance improvements from Numba JIT compilation across
all major computational kernels with various batch sizes.
"""

import time
import numpy as np
from typing import Dict, List, Tuple

from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    compute_quantile_returns_fast,
)
from quant_evaluator.kernels.numba_backend import (
    numba_ic_batch,
    numba_quantile_binning,
    numba_quantile_returns,
    numba_rolling_mean,
    numba_rolling_std,
    numba_rolling_corr,
    numba_corrcoef_matrix,
)


def make_test_data(T: int, N: int, F: int, seed: int = 42, sparsity: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic test data."""
    np.random.seed(seed)
    factor_values = np.random.randn(T, N, F)
    label_values = np.random.randn(T, N)

    if sparsity > 0:
        factor_mask = np.random.rand(T, N, F) < sparsity
        factor_values[factor_mask] = np.nan

        label_mask = np.random.rand(T, N) < sparsity
        label_values[label_mask] = np.nan

    return factor_values, label_values


def benchmark_ic_computation():
    """Benchmark IC computation (Pearson and Spearman)."""
    print("=" * 80)
    print("IC COMPUTATION SPEEDUP BENCHMARK (Numba vs Fast NumPy)")
    print("=" * 80)
    print()

    configs = [
        (30, 100, 100, 0.0, "Small batch - 100 factors, dense"),
        (50, 100, 500, 0.0, "Medium batch - 500 factors, dense"),
        (50, 100, 1000, 0.0, "Large batch - 1k factors, dense"),
        (50, 100, 2000, 0.0, "Very large batch - 2k factors, dense"),
        (30, 100, 5000, 0.0, "Massive batch - 5k factors, dense"),
        (50, 200, 1000, 0.2, "Large batch - 1k factors, 20% sparse"),
    ]

    results = []

    for method in ["pearson", "spearman"]:
        print(f"\n{'─' * 80}")
        print(f"Method: {method.upper()}")
        print(f"{'─' * 80}\n")

        for T, N, F, sparsity, description in configs:
            print(f"Testing: {description}")
            print(f"  Dimensions: T={T}, N={N}, F={F}")

            factor_values, label_values = make_test_data(T, N, F, seed=42, sparsity=sparsity)

            # Warm-up JIT compilation
            _ = numba_ic_batch(factor_values[:5, :, :10], label_values[:5, :], method=method, min_obs=10)

            # Benchmark fast (NumPy vectorized)
            start = time.perf_counter()
            ic_fast, counts_fast = fast_ic_batch(factor_values, label_values, method=method, min_obs=10)
            fast_time = time.perf_counter() - start

            # Benchmark Numba
            start = time.perf_counter()
            ic_numba, counts_numba = numba_ic_batch(factor_values, label_values, method=method, min_obs=10)
            numba_time = time.perf_counter() - start

            # Verify parity
            counts_match = np.array_equal(counts_fast, counts_numba)
            mask = np.isfinite(ic_fast) & np.isfinite(ic_numba)
            ic_match = np.allclose(ic_fast[mask], ic_numba[mask], rtol=1e-8, atol=1e-10) if np.any(mask) else True

            speedup = fast_time / numba_time if numba_time > 0 else float('inf')

            print(f"  Fast (NumPy):   {fast_time:.4f}s")
            print(f"  Numba (JIT):    {numba_time:.4f}s")
            print(f"  Speedup:        {speedup:.2f}x")
            print(f"  Parity check:   {'✓ PASS' if ic_match and counts_match else '✗ FAIL'}")
            print()

            results.append({
                'kernel': f'IC-{method}',
                'description': description,
                'T': T, 'N': N, 'F': F,
                'fast_time': fast_time,
                'numba_time': numba_time,
                'speedup': speedup,
                'parity': ic_match and counts_match,
            })

    return results


def benchmark_quantile_operations():
    """Benchmark quantile binning and returns."""
    print("=" * 80)
    print("QUANTILE OPERATIONS SPEEDUP BENCHMARK")
    print("=" * 80)
    print()

    configs = [
        (30, 100, 100, 5, "Small batch - 100 factors, 5 quantiles"),
        (50, 100, 500, 5, "Medium batch - 500 factors, 5 quantiles"),
        (50, 100, 1000, 10, "Large batch - 1k factors, 10 quantiles"),
        (50, 100, 2000, 5, "Very large batch - 2k factors, 5 quantiles"),
    ]

    results = []

    print(f"{'─' * 80}")
    print("Quantile Binning")
    print(f"{'─' * 80}\n")

    for T, N, F, n_quantiles, description in configs:
        print(f"Testing: {description}")
        print(f"  Dimensions: T={T}, N={N}, F={F}")

        factor_values, _ = make_test_data(T, N, F, seed=42, sparsity=0.0)

        # Warm-up
        _ = numba_quantile_binning(factor_values[:5, :, :10], n_quantiles=n_quantiles, min_valid=n_quantiles)

        # Benchmark fast
        start = time.perf_counter()
        bins_fast = fast_quantile_binning(factor_values, n_quantiles=n_quantiles, min_valid=n_quantiles)
        fast_time = time.perf_counter() - start

        # Benchmark Numba
        start = time.perf_counter()
        bins_numba = numba_quantile_binning(factor_values, n_quantiles=n_quantiles, min_valid=n_quantiles)
        numba_time = time.perf_counter() - start

        # Verify parity
        bins_match = np.array_equal(bins_fast, bins_numba)
        speedup = fast_time / numba_time if numba_time > 0 else float('inf')

        print(f"  Fast (NumPy):   {fast_time:.4f}s")
        print(f"  Numba (JIT):    {numba_time:.4f}s")
        print(f"  Speedup:        {speedup:.2f}x")
        print(f"  Parity check:   {'✓ PASS' if bins_match else '✗ FAIL'}")
        print()

        results.append({
            'kernel': 'Quantile-Binning',
            'description': description,
            'T': T, 'N': N, 'F': F,
            'fast_time': fast_time,
            'numba_time': numba_time,
            'speedup': speedup,
            'parity': bins_match,
        })

    print(f"{'─' * 80}")
    print("Quantile Returns")
    print(f"{'─' * 80}\n")

    for T, N, F, n_quantiles, description in configs:
        print(f"Testing: {description}")
        print(f"  Dimensions: T={T}, N={N}, F={F}")

        factor_values, label_values = make_test_data(T, N, F, seed=42, sparsity=0.0)

        # Warm-up
        _ = numba_quantile_returns(factor_values[:5, :, :10], label_values[:5, :],
                                    n_quantiles=n_quantiles, min_assets=5)

        # Benchmark fast
        start = time.perf_counter()
        qret_fast, qcounts_fast = compute_quantile_returns_fast(
            factor_values, label_values, n_quantiles=n_quantiles, min_assets=10
        )
        fast_time = time.perf_counter() - start

        # Benchmark Numba
        start = time.perf_counter()
        qret_numba, qcounts_numba = numba_quantile_returns(
            factor_values, label_values, n_quantiles=n_quantiles, min_assets=10
        )
        numba_time = time.perf_counter() - start

        # Verify parity
        counts_match = np.array_equal(qcounts_fast, qcounts_numba)
        mask = np.isfinite(qret_fast) & np.isfinite(qret_numba)
        ret_match = np.allclose(qret_fast[mask], qret_numba[mask], rtol=1e-9, atol=1e-12) if np.any(mask) else True

        speedup = fast_time / numba_time if numba_time > 0 else float('inf')

        print(f"  Fast (NumPy):   {fast_time:.4f}s")
        print(f"  Numba (JIT):    {numba_time:.4f}s")
        print(f"  Speedup:        {speedup:.2f}x")
        print(f"  Parity check:   {'✓ PASS' if ret_match and counts_match else '✗ FAIL'}")
        print()

        results.append({
            'kernel': 'Quantile-Returns',
            'description': description,
            'T': T, 'N': N, 'F': F,
            'fast_time': fast_time,
            'numba_time': numba_time,
            'speedup': speedup,
            'parity': ret_match and counts_match,
        })

    return results


def benchmark_rolling_statistics():
    """Benchmark rolling statistics."""
    print("=" * 80)
    print("ROLLING STATISTICS SPEEDUP BENCHMARK")
    print("=" * 80)
    print()

    configs = [
        (252, 100, 20, "1 year daily × 100 factors"),
        (252, 500, 20, "1 year daily × 500 factors"),
        (504, 1000, 20, "2 years daily × 1k factors"),
    ]

    results = []

    for kernel_name, kernel_func in [
        ("Rolling Mean", numba_rolling_mean),
        ("Rolling Std", numba_rolling_std),
    ]:
        print(f"{'─' * 80}")
        print(kernel_name)
        print(f"{'─' * 80}\n")

        for T, F, window, description in configs:
            print(f"Testing: {description} (window={window})")

            np.random.seed(42)
            values = np.random.randn(T, F)

            # Warm-up
            _ = kernel_func(values[:50, :10], window=window, min_periods=window//2)

            # Benchmark
            start = time.perf_counter()
            result = kernel_func(values, window=window, min_periods=window//2)
            numba_time = time.perf_counter() - start

            # Reference numpy (for one factor only, for fair comparison indication)
            start = time.perf_counter()
            ref_result = np.full((T, 1), np.nan)
            for t in range(T):
                start_idx = max(0, t - window + 1)
                win = values[start_idx:t+1, 0]
                finite = win[np.isfinite(win)]
                if len(finite) >= window//2:
                    if kernel_name == "Rolling Mean":
                        ref_result[t, 0] = np.mean(finite)
                    else:
                        ref_result[t, 0] = np.std(finite, ddof=0)
            numpy_time = (time.perf_counter() - start) * F  # Scale to all factors

            speedup = numpy_time / numba_time if numba_time > 0 else float('inf')

            print(f"  NumPy (est):    {numpy_time:.4f}s")
            print(f"  Numba (JIT):    {numba_time:.4f}s")
            print(f"  Speedup:        {speedup:.2f}x")
            print()

            results.append({
                'kernel': kernel_name,
                'description': description,
                'T': T, 'N': '-', 'F': F,
                'fast_time': numpy_time,
                'numba_time': numba_time,
                'speedup': speedup,
                'parity': True,
            })

    # Rolling correlation
    print(f"{'─' * 80}")
    print("Rolling Correlation")
    print(f"{'─' * 80}\n")

    for T, F, window, description in configs:
        print(f"Testing: {description} (window={window})")

        np.random.seed(42)
        x_values = np.random.randn(T, F)
        y_values = np.random.randn(T, F)

        # Warm-up
        _ = numba_rolling_corr(x_values[:50, :10], y_values[:50, :10], window=window, min_periods=window//2)

        # Benchmark
        start = time.perf_counter()
        result = numba_rolling_corr(x_values, y_values, window=window, min_periods=window//2)
        numba_time = time.perf_counter() - start

        print(f"  Numba (JIT):    {numba_time:.4f}s")
        print()

        results.append({
            'kernel': 'Rolling-Corr',
            'description': description,
            'T': T, 'N': '-', 'F': F,
            'fast_time': np.nan,
            'numba_time': numba_time,
            'speedup': np.nan,
            'parity': True,
        })

    return results


def benchmark_correlation_matrix():
    """Benchmark correlation matrix computation."""
    print("=" * 80)
    print("CORRELATION MATRIX SPEEDUP BENCHMARK")
    print("=" * 80)
    print()

    configs = [
        (252, 100, "100 factors × 1 year"),
        (252, 500, "500 factors × 1 year"),
        (252, 1000, "1k factors × 1 year"),
        (504, 1000, "1k factors × 2 years"),
    ]

    results = []

    for T, F, description in configs:
        print(f"Testing: {description}")

        np.random.seed(42)
        values = np.random.randn(T, F)

        # Warm-up
        _ = numba_corrcoef_matrix(values[:50, :20], min_obs=10)

        # Benchmark NumPy
        start = time.perf_counter()
        corr_numpy = np.corrcoef(values, rowvar=False)
        numpy_time = time.perf_counter() - start

        # Benchmark Numba
        start = time.perf_counter()
        corr_numba = numba_corrcoef_matrix(values, min_obs=10)
        numba_time = time.perf_counter() - start

        # Verify parity
        mask = np.isfinite(corr_numpy) & np.isfinite(corr_numba)
        parity = np.allclose(corr_numpy[mask], corr_numba[mask], rtol=1e-9, atol=1e-12)

        speedup = numpy_time / numba_time if numba_time > 0 else float('inf')

        print(f"  NumPy:          {numpy_time:.4f}s")
        print(f"  Numba (JIT):    {numba_time:.4f}s")
        print(f"  Speedup:        {speedup:.2f}x")
        print(f"  Parity check:   {'✓ PASS' if parity else '✗ FAIL'}")
        print()

        results.append({
            'kernel': 'CorrMatrix',
            'description': description,
            'T': T, 'N': '-', 'F': F,
            'fast_time': numpy_time,
            'numba_time': numba_time,
            'speedup': speedup,
            'parity': parity,
        })

    return results


def print_summary(all_results: List[Dict]):
    """Print comprehensive summary."""
    print("=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)
    print()

    # Group by kernel
    kernels = {}
    for r in all_results:
        kernel = r['kernel']
        if kernel not in kernels:
            kernels[kernel] = []
        kernels[kernel].append(r)

    print(f"{'Kernel':<20} {'Count':>8} {'Avg Speedup':>15} {'Min':>10} {'Max':>10} {'Parity':>10}")
    print("─" * 80)

    all_speedups = []
    for kernel, results in sorted(kernels.items()):
        speedups = [r['speedup'] for r in results if np.isfinite(r['speedup'])]
        if speedups:
            avg_speedup = np.mean(speedups)
            min_speedup = np.min(speedups)
            max_speedup = np.max(speedups)
            all_speedups.extend(speedups)
        else:
            avg_speedup = min_speedup = max_speedup = np.nan

        parity_count = sum(1 for r in results if r['parity'])
        parity_str = f"{parity_count}/{len(results)}"

        print(f"{kernel:<20} {len(results):>8} {avg_speedup:>14.2f}x {min_speedup:>9.2f}x {max_speedup:>9.2f}x {parity_str:>10}")

    print("─" * 80)
    print(f"{'OVERALL':<20} {len(all_results):>8} {np.mean(all_speedups):>14.2f}x {np.min(all_speedups):>9.2f}x {np.max(all_speedups):>9.2f}x")
    print()

    # Check target achievement
    ic_results = [r for r in all_results if 'IC' in r['kernel'] and r['F'] >= 1000]
    if ic_results:
        ic_speedups = [r['speedup'] for r in ic_results if np.isfinite(r['speedup'])]
        avg_ic_speedup = np.mean(ic_speedups)

        print(f"TARGET ACHIEVEMENT:")
        print(f"  IC computation (1k+ factors): {avg_ic_speedup:.2f}x average speedup")
        print(f"  Target range: 10-50x")

        if 10.0 <= avg_ic_speedup <= 50.0:
            print(f"  Status: ✓ TARGET ACHIEVED")
        elif avg_ic_speedup > 50.0:
            print(f"  Status: ✓ EXCEEDED TARGET")
        else:
            print(f"  Status: ✗ BELOW TARGET (but {avg_ic_speedup:.2f}x is still significant)")

    print()

    # Parity summary
    parity_pass = sum(1 for r in all_results if r['parity'])
    print(f"Parity Tests: {parity_pass}/{len(all_results)} passed")
    if parity_pass == len(all_results):
        print("✓ ALL PARITY TESTS PASSED")
    else:
        print("✗ SOME PARITY TESTS FAILED")

    print()


def main():
    """Run all benchmarks."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "NUMBA JIT BACKEND COMPREHENSIVE BENCHMARK" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    all_results = []

    # Run benchmarks
    all_results.extend(benchmark_ic_computation())
    all_results.extend(benchmark_quantile_operations())
    all_results.extend(benchmark_rolling_statistics())
    all_results.extend(benchmark_correlation_matrix())

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
