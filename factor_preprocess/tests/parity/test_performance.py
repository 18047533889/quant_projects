"""
Performance benchmarks for fast kernels.

Measures speedup of fast implementations vs reference on large datasets.
"""
import numpy as np
import time
from typing import Callable, Dict, Tuple

from factor_preprocess.kernels.reference_bridge import (
    reference_cs_rank,
    reference_cs_zscore,
    reference_rolling_mean,
    reference_rolling_std,
)

from factor_preprocess.kernels.fast import (
    fast_cs_rank,
    fast_cs_zscore,
    fast_rolling_mean,
    fast_rolling_std,
    numba_rolling_mean,
    numba_rolling_std,
    get_capabilities,
    HAS_BOTTLENECK,
    HAS_NUMBA,
)


def time_function(func: Callable, *args, **kwargs) -> Tuple[float, any]:
    """
    Time a function call.

    Returns
    -------
    duration : float
        Execution time in seconds
    result : any
        Function return value
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    return duration, result


def benchmark_cross_sectional(
    dates: int = 1000,
    assets: int = 500,
    nan_rate: float = 0.05,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark cross-sectional operations.

    Parameters
    ----------
    dates : int
        Number of time periods
    assets : int
        Number of assets
    nan_rate : float
        Fraction of NaN values to inject
    seed : int
        Random seed

    Returns
    -------
    dict
        Benchmark results with timings and speedups
    """
    rng = np.random.RandomState(seed)
    data = rng.randn(dates, assets)
    nan_mask = rng.rand(dates, assets) < nan_rate
    data[nan_mask] = np.nan

    print(f"\n{'='*70}")
    print(f"Cross-Sectional Benchmarks: {dates} dates × {assets} assets")
    print(f"{'='*70}")

    results = {}

    # Rank benchmark
    print("\n[1] cs_rank (average ties)")
    ref_time, ref_result = time_function(reference_cs_rank, data, axis=-1)
    fast_time, fast_result = time_function(fast_cs_rank, data, axis=-1)

    # Verify parity
    assert np.allclose(ref_result, fast_result, rtol=1e-10, atol=1e-14, equal_nan=True)

    speedup = ref_time / fast_time
    results["rank"] = {
        "reference_time": ref_time,
        "fast_time": fast_time,
        "speedup": speedup,
    }
    print(f"  Reference: {ref_time:.4f}s")
    print(f"  Fast:      {fast_time:.4f}s")
    print(f"  Speedup:   {speedup:.1f}x")
    if HAS_BOTTLENECK:
        print(f"  (using bottleneck)")
    else:
        print(f"  (bottleneck not available, using scipy)")

    # Rank percentile benchmark
    print("\n[2] cs_rank (percentile)")
    ref_time, ref_result = time_function(reference_cs_rank, data, axis=-1, pct=True)
    fast_time, fast_result = time_function(fast_cs_rank, data, axis=-1, pct=True)

    assert np.allclose(ref_result, fast_result, rtol=1e-10, atol=1e-14, equal_nan=True)

    speedup = ref_time / fast_time
    results["rank_pct"] = {
        "reference_time": ref_time,
        "fast_time": fast_time,
        "speedup": speedup,
    }
    print(f"  Reference: {ref_time:.4f}s")
    print(f"  Fast:      {fast_time:.4f}s")
    print(f"  Speedup:   {speedup:.1f}x")

    # Z-score benchmark
    print("\n[3] cs_zscore")
    ref_time, ref_result = time_function(reference_cs_zscore, data, axis=-1)
    fast_time, fast_result = time_function(fast_cs_zscore, data, axis=-1)

    assert np.allclose(ref_result, fast_result, rtol=1e-10, atol=1e-14, equal_nan=True)

    speedup = ref_time / fast_time
    results["zscore"] = {
        "reference_time": ref_time,
        "fast_time": fast_time,
        "speedup": speedup,
    }
    print(f"  Reference: {ref_time:.4f}s")
    print(f"  Fast:      {fast_time:.4f}s")
    print(f"  Speedup:   {speedup:.1f}x")

    return results


def benchmark_rolling(
    dates: int = 2000,
    assets: int = 100,
    window: int = 60,
    nan_rate: float = 0.03,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark rolling operations.

    Parameters
    ----------
    dates : int
        Number of time periods
    assets : int
        Number of assets
    window : int
        Rolling window size
    nan_rate : float
        Fraction of NaN values to inject
    seed : int
        Random seed

    Returns
    -------
    dict
        Benchmark results with timings and speedups
    """
    rng = np.random.RandomState(seed)
    data = rng.randn(dates, assets) * 10 + 100
    nan_mask = rng.rand(dates, assets) < nan_rate
    data[nan_mask] = np.nan

    print(f"\n{'='*70}")
    print(f"Rolling Benchmarks: {dates} dates × {assets} assets, window={window}")
    print(f"{'='*70}")

    results = {}

    # Rolling mean benchmark
    print("\n[1] rolling_mean")
    ref_time, ref_result = time_function(reference_rolling_mean, data, window, axis=0)
    fast_time, fast_result = time_function(fast_rolling_mean, data, window, axis=0)

    assert np.allclose(ref_result, fast_result, rtol=1e-10, atol=1e-12, equal_nan=True)

    speedup = ref_time / fast_time
    results["rolling_mean_stride"] = {
        "reference_time": ref_time,
        "fast_time": fast_time,
        "speedup": speedup,
    }
    print(f"  Reference:    {ref_time:.4f}s")
    print(f"  Fast (stride): {fast_time:.4f}s")
    print(f"  Speedup:      {speedup:.1f}x")

    if HAS_NUMBA:
        numba_time, numba_result = time_function(numba_rolling_mean, data, window, axis=0)
        assert np.allclose(ref_result, numba_result, rtol=1e-10, atol=1e-12, equal_nan=True)
        speedup_numba = ref_time / numba_time
        results["rolling_mean_numba"] = {
            "reference_time": ref_time,
            "numba_time": numba_time,
            "speedup": speedup_numba,
        }
        print(f"  Fast (numba):  {numba_time:.4f}s")
        print(f"  Speedup:      {speedup_numba:.1f}x")
    else:
        print(f"  (numba not available)")

    # Rolling std benchmark
    print("\n[2] rolling_std")
    ref_time, ref_result = time_function(reference_rolling_std, data, window, axis=0)
    fast_time, fast_result = time_function(fast_rolling_std, data, window, axis=0)

    assert np.allclose(ref_result, fast_result, rtol=1e-10, atol=1e-12, equal_nan=True)

    speedup = ref_time / fast_time
    results["rolling_std_stride"] = {
        "reference_time": ref_time,
        "fast_time": fast_time,
        "speedup": speedup,
    }
    print(f"  Reference:    {ref_time:.4f}s")
    print(f"  Fast (stride): {fast_time:.4f}s")
    print(f"  Speedup:      {speedup:.1f}x")

    if HAS_NUMBA:
        numba_time, numba_result = time_function(numba_rolling_std, data, window, axis=0)
        assert np.allclose(ref_result, numba_result, rtol=1e-10, atol=1e-12, equal_nan=True)
        speedup_numba = ref_time / numba_time
        results["rolling_std_numba"] = {
            "reference_time": ref_time,
            "numba_time": numba_time,
            "speedup": speedup_numba,
        }
        print(f"  Fast (numba):  {numba_time:.4f}s")
        print(f"  Speedup:      {speedup_numba:.1f}x")
    else:
        print(f"  (numba not available)")

    return results


def main():
    """Run all benchmarks and report summary."""
    print("\n" + "="*70)
    print("FAST KERNEL BENCHMARKS")
    print("="*70)

    caps = get_capabilities()
    print("\nCapabilities:")
    for key, available in caps.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {key:20s}: {status}")

    # Cross-sectional benchmarks
    cs_results = benchmark_cross_sectional(
        dates=1000,
        assets=500,
        nan_rate=0.05
    )

    # Rolling benchmarks
    rolling_results = benchmark_rolling(
        dates=2000,
        assets=100,
        window=60,
        nan_rate=0.03
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print("\nCross-Sectional Speedups:")
    for op_name, metrics in cs_results.items():
        print(f"  {op_name:20s}: {metrics['speedup']:6.1f}x")

    print("\nRolling Speedups:")
    for op_name, metrics in rolling_results.items():
        print(f"  {op_name:20s}: {metrics['speedup']:6.1f}x")

    # Expected gains
    print(f"\n{'='*70}")
    print("Expected Performance Gains:")
    print(f"{'='*70}")
    print("\nWith bottleneck installed:")
    print("  cs_rank:           10-100x faster (depends on data size)")
    print("  cs_rank (pct):     10-100x faster")
    print("\nWith stride tricks (always available):")
    print("  rolling_mean:      10-50x faster")
    print("  rolling_std:       10-50x faster")
    print("\nWith numba installed:")
    print("  rolling_mean:      50-200x faster (parallel execution)")
    print("  rolling_std:       50-200x faster (parallel execution)")


if __name__ == "__main__":
    main()
