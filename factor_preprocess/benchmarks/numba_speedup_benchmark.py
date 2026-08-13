#!/usr/bin/env python3
"""
Benchmark script to measure speedup of Numba-accelerated transforms.

Compares Numba implementations against reference (bottleneck/numpy) implementations.
"""
import time
import numpy as np
from factor_preprocess.kernels.numba_transforms import (
    numba_rolling_mean,
    numba_rolling_std,
    numba_rolling_sum,
    numba_rolling_min,
    numba_rolling_max,
    numba_cs_rank,
    numba_cs_zscore,
    numba_cs_winsorize,
)
from factor_preprocess.kernels.reference_bridge import (
    reference_rolling_mean,
    reference_rolling_std,
    reference_cs_rank,
    reference_cs_zscore,
)

try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False


def time_function(func, *args, n_runs=5, warmup=1, **kwargs):
    """Time a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return np.median(times)


def benchmark_rolling_operations():
    """Benchmark rolling operations."""
    print("=" * 80)
    print("ROLLING OPERATIONS BENCHMARK")
    print("=" * 80)

    shapes = [
        (1000, 100, 20, "Small panel (1000×100, window=20)"),
        (2000, 500, 20, "Medium panel (2000×500, window=20)"),
        (5000, 1000, 50, "Large panel (5000×1000, window=50)"),
    ]

    for nrows, ncols, window, desc in shapes:
        print(f"\n{desc}")
        print("-" * 80)

        # Generate data
        rng = np.random.RandomState(42)
        data = rng.randn(nrows, ncols).astype(np.float64)
        nan_mask = rng.rand(nrows, ncols) < 0.02
        data[nan_mask] = np.nan

        # Rolling mean
        time_numba = time_function(numba_rolling_mean, data, window, 0)
        time_ref = time_function(reference_rolling_mean, data, window, 0)
        speedup = time_ref / time_numba
        print(f"  Rolling mean:  Numba={time_numba:.4f}s  Ref={time_ref:.4f}s  "
              f"Speedup={speedup:.1f}x")

        # Rolling std
        time_numba = time_function(numba_rolling_std, data, window, 0, 1)
        time_ref = time_function(reference_rolling_std, data, window, 0, 1)
        speedup = time_ref / time_numba
        print(f"  Rolling std:   Numba={time_numba:.4f}s  Ref={time_ref:.4f}s  "
              f"Speedup={speedup:.1f}x")

        # Rolling sum
        time_numba = time_function(numba_rolling_sum, data, window, 0)
        if HAS_BOTTLENECK:
            time_bn = time_function(bn.move_sum, data, window, axis=0, min_count=1)
            speedup = time_bn / time_numba
            print(f"  Rolling sum:   Numba={time_numba:.4f}s  Bottleneck={time_bn:.4f}s  "
                  f"Speedup={speedup:.1f}x")
        else:
            print(f"  Rolling sum:   Numba={time_numba:.4f}s  (bottleneck not available)")

        # Rolling min
        time_numba = time_function(numba_rolling_min, data, window, 0)
        if HAS_BOTTLENECK:
            time_bn = time_function(bn.move_min, data, window, axis=0, min_count=1)
            speedup = time_bn / time_numba
            print(f"  Rolling min:   Numba={time_numba:.4f}s  Bottleneck={time_bn:.4f}s  "
                  f"Speedup={speedup:.1f}x")

        # Rolling max
        time_numba = time_function(numba_rolling_max, data, window, 0)
        if HAS_BOTTLENECK:
            time_bn = time_function(bn.move_max, data, window, axis=0, min_count=1)
            speedup = time_bn / time_numba
            print(f"  Rolling max:   Numba={time_numba:.4f}s  Bottleneck={time_bn:.4f}s  "
                  f"Speedup={speedup:.1f}x")


def benchmark_cross_sectional_operations():
    """Benchmark cross-sectional operations."""
    print("\n" + "=" * 80)
    print("CROSS-SECTIONAL OPERATIONS BENCHMARK")
    print("=" * 80)

    shapes = [
        (500, 500, "Small cross-section (500×500)"),
        (1000, 1000, "Medium cross-section (1000×1000)"),
        (2000, 2000, "Large cross-section (2000×2000)"),
    ]

    for nrows, ncols, desc in shapes:
        print(f"\n{desc}")
        print("-" * 80)

        # Generate data
        rng = np.random.RandomState(42)
        data = rng.randn(nrows, ncols).astype(np.float64)
        nan_mask = rng.rand(nrows, ncols) < 0.02
        data[nan_mask] = np.nan

        # CS rank
        time_numba = time_function(numba_cs_rank, data, -1, pct=True)
        time_ref = time_function(reference_cs_rank, data, -1, method="average", pct=True)
        speedup = time_ref / time_numba
        print(f"  CS rank:       Numba={time_numba:.4f}s  Ref={time_ref:.4f}s  "
              f"Speedup={speedup:.1f}x")

        # CS zscore
        time_numba = time_function(numba_cs_zscore, data, -1, ddof=1)
        time_ref = time_function(reference_cs_zscore, data, -1, ddof=1)
        speedup = time_ref / time_numba
        print(f"  CS zscore:     Numba={time_numba:.4f}s  Ref={time_ref:.4f}s  "
              f"Speedup={speedup:.1f}x")

        # CS winsorize
        time_numba = time_function(numba_cs_winsorize, data, 0.05, 0.95, axis=-1)
        # Reference doesn't have winsorize, so just report absolute time
        print(f"  CS winsorize:  Numba={time_numba:.4f}s")


def main():
    """Run all benchmarks."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "NUMBA TRANSFORMS PERFORMANCE BENCHMARK" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    if HAS_BOTTLENECK:
        print("Bottleneck library: AVAILABLE")
    else:
        print("Bottleneck library: NOT AVAILABLE (install for full benchmarks)")

    benchmark_rolling_operations()
    benchmark_cross_sectional_operations()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
