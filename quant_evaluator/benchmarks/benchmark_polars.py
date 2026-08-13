"""
Benchmark polars backend vs numpy for IC and quantile operations.

Measures memory efficiency and speedup for various data sizes.
"""

import numpy as np
import time
import psutil
import os
from typing import Tuple

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("Warning: Polars not available")

from quant_evaluator.backends.polars_backend import (
    polars_ic_batch,
    polars_quantile_binning,
)
from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
)


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_ic(T: int, N: int, F: int, method: str = "pearson") -> dict:
    """
    Benchmark IC computation.

    Returns dict with timing and memory stats for both backends.
    """
    print(f"\n=== IC Benchmark (T={T}, N={N}, F={F}, method={method}) ===")

    np.random.seed(42)
    factors = np.random.randn(T, N, F).astype(np.float64)
    labels = np.random.randn(T, N).astype(np.float64)
    factor_ids = tuple(f"f{i:04d}" for i in range(F))

    results = {}

    # NumPy baseline
    print("Running NumPy...")
    mem_before = get_memory_mb()
    start = time.perf_counter()
    ic_numpy, counts_numpy = fast_ic_batch(factors, labels, method=method)
    numpy_time = time.perf_counter() - start
    mem_after = get_memory_mb()
    numpy_mem = mem_after - mem_before

    print(f"  Time: {numpy_time:.3f}s")
    print(f"  Memory delta: {numpy_mem:.1f} MB")

    results["numpy"] = {
        "time": numpy_time,
        "memory_mb": numpy_mem,
        "shape": ic_numpy.shape,
    }

    if not POLARS_AVAILABLE:
        return results

    # Polars backend
    print("Running Polars...")
    mem_before = get_memory_mb()
    start = time.perf_counter()
    ic_polars, counts_polars = polars_ic_batch(
        factors, labels, factor_ids, method=method
    )
    polars_time = time.perf_counter() - start
    mem_after = get_memory_mb()
    polars_mem = mem_after - mem_before

    print(f"  Time: {polars_time:.3f}s")
    print(f"  Memory delta: {polars_mem:.1f} MB")

    results["polars"] = {
        "time": polars_time,
        "memory_mb": polars_mem,
        "shape": ic_polars.shape,
    }

    # Compute speedup and memory efficiency
    speedup = numpy_time / polars_time
    memory_efficiency = numpy_mem / polars_mem if polars_mem > 0 else float('inf')

    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Memory efficiency: {memory_efficiency:.2f}x")

    results["speedup"] = speedup
    results["memory_efficiency"] = memory_efficiency

    # Verify correctness
    max_diff = np.nanmax(np.abs(ic_numpy - ic_polars))
    print(f"  Max IC difference: {max_diff:.2e}")

    return results


def benchmark_quantile(T: int, N: int, F: int, n_quantiles: int = 5) -> dict:
    """
    Benchmark quantile binning.

    Returns dict with timing and memory stats for both backends.
    """
    print(f"\n=== Quantile Benchmark (T={T}, N={N}, F={F}, Q={n_quantiles}) ===")

    np.random.seed(42)
    factors = np.random.randn(T, N, F).astype(np.float64)
    factor_ids = tuple(f"f{i:04d}" for i in range(F))

    results = {}

    # NumPy baseline
    print("Running NumPy...")
    mem_before = get_memory_mb()
    start = time.perf_counter()
    q_numpy = fast_quantile_binning(factors, n_quantiles=n_quantiles)
    numpy_time = time.perf_counter() - start
    mem_after = get_memory_mb()
    numpy_mem = mem_after - mem_before

    print(f"  Time: {numpy_time:.3f}s")
    print(f"  Memory delta: {numpy_mem:.1f} MB")

    results["numpy"] = {
        "time": numpy_time,
        "memory_mb": numpy_mem,
        "shape": q_numpy.shape,
    }

    if not POLARS_AVAILABLE:
        return results

    # Polars backend
    print("Running Polars...")
    mem_before = get_memory_mb()
    start = time.perf_counter()
    q_polars = polars_quantile_binning(
        factors, factor_ids, n_quantiles=n_quantiles
    )
    polars_time = time.perf_counter() - start
    mem_after = get_memory_mb()
    polars_mem = mem_after - mem_before

    print(f"  Time: {polars_time:.3f}s")
    print(f"  Memory delta: {polars_mem:.1f} MB")

    results["polars"] = {
        "time": polars_time,
        "memory_mb": polars_mem,
        "shape": q_polars.shape,
    }

    # Compute speedup and memory efficiency
    speedup = numpy_time / polars_time
    memory_efficiency = numpy_mem / polars_mem if polars_mem > 0 else float('inf')

    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Memory efficiency: {memory_efficiency:.2f}x")

    results["speedup"] = speedup
    results["memory_efficiency"] = memory_efficiency

    return results


def run_benchmark_suite():
    """Run a suite of benchmarks at different scales."""
    print("=" * 70)
    print("POLARS BACKEND BENCHMARK SUITE")
    print("=" * 70)

    if not POLARS_AVAILABLE:
        print("ERROR: Polars not installed. Install with: pip install polars")
        return

    all_results = {}

    # Small scale
    all_results["small_ic"] = benchmark_ic(T=50, N=100, F=50, method="pearson")

    # Medium scale
    all_results["medium_ic"] = benchmark_ic(T=252, N=500, F=200, method="pearson")

    # Large scale (factor-heavy)
    all_results["large_ic"] = benchmark_ic(T=252, N=1000, F=500, method="pearson")

    # Quantile benchmarks
    all_results["small_quantile"] = benchmark_quantile(T=50, N=100, F=50, n_quantiles=5)
    all_results["medium_quantile"] = benchmark_quantile(T=252, N=500, F=200, n_quantiles=5)
    all_results["large_quantile"] = benchmark_quantile(T=252, N=1000, F=500, n_quantiles=5)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in all_results.items():
        if "speedup" in result:
            print(f"\n{name}:")
            print(f"  Speedup: {result['speedup']:.2f}x")
            print(f"  Memory efficiency: {result['memory_efficiency']:.2f}x")

    # Calculate average speedup
    speedups = [r["speedup"] for r in all_results.values() if "speedup" in r]
    mem_effs = [r["memory_efficiency"] for r in all_results.values() if "memory_efficiency" in r]

    if speedups:
        print(f"\nAverage speedup: {np.mean(speedups):.2f}x (range: {np.min(speedups):.2f}x - {np.max(speedups):.2f}x)")
        print(f"Average memory efficiency: {np.mean(mem_effs):.2f}x (range: {np.min(mem_effs):.2f}x - {np.max(mem_effs):.2f}x)")

    return all_results


if __name__ == "__main__":
    results = run_benchmark_suite()
