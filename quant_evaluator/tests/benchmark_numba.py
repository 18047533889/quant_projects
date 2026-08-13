"""
Benchmark suite comparing Numba kernels vs reference implementations.
"""
import time
import numpy as np
from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    compute_quantile_returns_fast,
)
from quant_evaluator.kernels.numba_backend import (
    numba_pearson_ic_batch,
    numba_spearman_ic_batch,
    numba_quantile_binning,
    numba_quantile_returns,
    numba_rolling_mean,
    numba_rolling_std,
    numba_rolling_corr,
    numba_corrcoef_matrix,
)


def make_test_data(T, N, F, sparsity=0.0, seed=42):
    """Generate test data with optional sparsity."""
    np.random.seed(seed)
    factor_values = np.random.randn(T, N, F)
    label_values = np.random.randn(T, N)

    if sparsity > 0:
        mask_f = np.random.rand(T, N, F) < sparsity
        mask_l = np.random.rand(T, N) < sparsity
        factor_values[mask_f] = np.nan
        label_values[mask_l] = np.nan

    return factor_values, label_values


def benchmark_function(func, *args, warmup=2, runs=5, name="Function"):
    """Benchmark a function with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time, result


def benchmark_ic_computation():
    """Benchmark IC computation (Pearson and Spearman)."""
    print("\n" + "="*80)
    print("IC COMPUTATION BENCHMARKS")
    print("="*80)

    configs = [
        ("Small (100 × 500 × 50)", 100, 500, 50),
        ("Medium (252 × 1000 × 100)", 252, 1000, 100),
        ("Large (500 × 2000 × 200)", 500, 2000, 200),
    ]

    for name, T, N, F in configs:
        print(f"\n{name}: T={T}, N={N}, F={F}")
        factor_values, label_values = make_test_data(T, N, F, sparsity=0.15)

        # Pearson IC
        print("  Pearson IC:")
        time_fast, std_fast, _ = benchmark_function(
            fast_ic_batch, factor_values, label_values, "pearson", 30
        )
        time_numba, std_numba, _ = benchmark_function(
            numba_pearson_ic_batch, factor_values, label_values, 30
        )
        speedup = time_fast / time_numba
        print(f"    NumPy:  {time_fast*1000:7.2f} ± {std_fast*1000:5.2f} ms")
        print(f"    Numba:  {time_numba*1000:7.2f} ± {std_numba*1000:5.2f} ms")
        print(f"    Speedup: {speedup:.1f}×")

        # Spearman IC
        print("  Spearman IC:")
        time_fast, std_fast, _ = benchmark_function(
            fast_ic_batch, factor_values, label_values, "spearman", 30
        )
        time_numba, std_numba, _ = benchmark_function(
            numba_spearman_ic_batch, factor_values, label_values, 30
        )
        speedup = time_fast / time_numba
        print(f"    NumPy:  {time_fast*1000:7.2f} ± {std_fast*1000:5.2f} ms")
        print(f"    Numba:  {time_numba*1000:7.2f} ± {std_numba*1000:5.2f} ms")
        print(f"    Speedup: {speedup:.1f}×")


def benchmark_quantile_operations():
    """Benchmark quantile binning and returns."""
    print("\n" + "="*80)
    print("QUANTILE OPERATIONS BENCHMARKS")
    print("="*80)

    configs = [
        ("Small (100 × 500 × 50)", 100, 500, 50),
        ("Medium (252 × 1000 × 100)", 252, 1000, 100),
        ("Large (500 × 2000 × 200)", 500, 2000, 200),
    ]

    for name, T, N, F in configs:
        print(f"\n{name}: T={T}, N={N}, F={F}")
        factor_values, label_values = make_test_data(T, N, F, sparsity=0.15)

        # Quantile binning
        print("  Quantile Binning (10 quantiles):")
        time_fast, std_fast, _ = benchmark_function(
            fast_quantile_binning, factor_values, 10, 10
        )
        time_numba, std_numba, _ = benchmark_function(
            numba_quantile_binning, factor_values, 10, 10
        )
        speedup = time_fast / time_numba
        print(f"    NumPy:  {time_fast*1000:7.2f} ± {std_fast*1000:5.2f} ms")
        print(f"    Numba:  {time_numba*1000:7.2f} ± {std_numba*1000:5.2f} ms")
        print(f"    Speedup: {speedup:.1f}×")

        # Quantile returns
        print("  Quantile Returns (10 quantiles):")
        time_fast, std_fast, _ = benchmark_function(
            compute_quantile_returns_fast, factor_values, label_values, 10, 5
        )
        time_numba, std_numba, _ = benchmark_function(
            numba_quantile_returns, factor_values, label_values, 10, 5
        )
        speedup = time_fast / time_numba
        print(f"    NumPy:  {time_fast*1000:7.2f} ± {std_fast*1000:5.2f} ms")
        print(f"    Numba:  {time_numba*1000:7.2f} ± {std_numba*1000:5.2f} ms")
        print(f"    Speedup: {speedup:.1f}×")


def benchmark_rolling_stats():
    """Benchmark rolling statistics."""
    print("\n" + "="*80)
    print("ROLLING STATISTICS BENCHMARKS")
    print("="*80)

    configs = [
        ("Small (500 × 100)", 500, 100, 20),
        ("Medium (1000 × 200)", 1000, 200, 20),
        ("Large (2000 × 500)", 2000, 500, 20),
    ]

    for name, T, N, window in configs:
        print(f"\n{name}: T={T}, N={N}, window={window}")
        values = np.random.randn(T, N)
        values[np.random.rand(T, N) < 0.1] = np.nan

        # Rolling mean
        print("  Rolling Mean:")
        time_pandas, std_pandas, ref = benchmark_function(
            lambda v: np.array([
                np.array([
                    np.nanmean(v[max(0, t-window+1):t+1, n])
                    if t >= window-1 and np.sum(np.isfinite(v[max(0, t-window+1):t+1, n])) >= window//2
                    else np.nan
                    for n in range(N)
                ])
                for t in range(T)
            ]),
            values
        )
        time_numba, std_numba, _ = benchmark_function(
            numba_rolling_mean, values, window, window//2
        )
        speedup = time_pandas / time_numba
        print(f"    NumPy:  {time_pandas*1000:7.2f} ± {std_pandas*1000:5.2f} ms")
        print(f"    Numba:  {time_numba*1000:7.2f} ± {std_numba*1000:5.2f} ms")
        print(f"    Speedup: {speedup:.1f}×")


def benchmark_correlation_matrix():
    """Benchmark correlation matrix computation."""
    print("\n" + "="*80)
    print("CORRELATION MATRIX BENCHMARKS")
    print("="*80)

    configs = [
        ("Small (100 × 50)", 100, 50),
        ("Medium (500 × 100)", 500, 100),
        ("Large (1000 × 200)", 1000, 200),
    ]

    for name, T, F in configs:
        print(f"\n{name}: T={T}, F={F}")
        values = np.random.randn(T, F)
        values[np.random.rand(T, F) < 0.1] = np.nan

        print("  Correlation Matrix:")
        time_numpy, std_numpy, _ = benchmark_function(
            lambda v: np.corrcoef(v, rowvar=False) if not np.any(np.isnan(v))
            else np.array([[
                np.corrcoef(v[:, i], v[:, j])[0, 1] if i != j else 1.0
                for j in range(v.shape[1])
            ] for i in range(v.shape[1])]),
            values
        )
        time_numba, std_numba, _ = benchmark_function(
            numba_corrcoef_matrix, values, 20
        )
        speedup = time_numpy / time_numba
        print(f"    NumPy:  {time_numpy*1000:7.2f} ± {std_numpy*1000:5.2f} ms")
        print(f"    Numba:  {time_numba*1000:7.2f} ± {std_numba*1000:5.2f} ms")
        print(f"    Speedup: {speedup:.1f}×")


def main():
    """Run all benchmarks."""
    print("\n" + "="*80)
    print("NUMBA BACKEND PERFORMANCE BENCHMARKS")
    print("="*80)
    print("\nComparing Numba-JIT compiled kernels against reference implementations")
    print("Target: 10-50× speedup on large batches\n")

    benchmark_ic_computation()
    benchmark_quantile_operations()
    benchmark_rolling_stats()
    benchmark_correlation_matrix()

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
