"""
Benchmark numba_transforms against reference implementations.

Measures actual speedup for rolling operations, cross-sectional operations,
and winsorization. Target: 20-100x speedup for rolling operations.

Run with: python benchmarks/numba_speedup.py
"""
import time
import numpy as np
from typing import Tuple

# Reference implementations
from factor_preprocess.kernels.reference_bridge import (
    reference_rolling_mean,
    reference_rolling_std,
    reference_cs_rank,
    reference_cs_zscore,
)

# Numba implementations
try:
    from factor_preprocess.kernels.numba_transforms import (
        numba_rolling_mean,
        numba_rolling_std,
        numba_rolling_sum,
        numba_rolling_min,
        numba_rolling_max,
        numba_cs_rank,
        numba_cs_zscore,
        numba_cs_winsorize,
        has_numba,
    )
    HAS_NUMBA = has_numba()
except ImportError:
    HAS_NUMBA = False


def generate_panel(T: int, N: int, nan_pct: float = 0.03, seed: int = 42) -> np.ndarray:
    """Generate synthetic panel data for benchmarking."""
    rng = np.random.RandomState(seed)
    data = rng.randn(T, N) * 10 + 100
    nan_mask = rng.rand(T, N) < nan_pct
    data[nan_mask] = np.nan
    return data


def benchmark_function(func, *args, n_runs: int = 5, warmup: int = 1) -> Tuple[float, float]:
    """
    Benchmark a function with warmup and multiple runs.

    Returns
    -------
    tuple
        (mean_time, std_time) in seconds
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def format_speedup(ref_time: float, fast_time: float) -> str:
    """Format speedup ratio with color coding."""
    if fast_time == 0:
        return "N/A"
    speedup = ref_time / fast_time
    if speedup >= 20:
        return f"{speedup:.1f}x ✓"
    elif speedup >= 10:
        return f"{speedup:.1f}x ~"
    else:
        return f"{speedup:.1f}x ✗"


def benchmark_rolling_operations():
    """Benchmark rolling operations - primary speedup target."""
    print("=" * 80)
    print("ROLLING OPERATIONS BENCHMARK")
    print("=" * 80)
    print()

    configs = [
        # (T, N, window, description)
        (252, 100, 20, "Small panel (1 year daily, 100 assets)"),
        (500, 300, 20, "Medium panel (2 years daily, 300 assets)"),
        (1000, 500, 20, "Large panel (4 years daily, 500 assets)"),
        (1000, 1000, 20, "Very large panel (4 years daily, 1000 assets)"),
        (500, 300, 60, "Medium panel with large window (60 days)"),
    ]

    for T, N, window, desc in configs:
        print(f"{desc}:")
        print(f"  Shape: ({T}, {N}), Window: {window}")
        data = generate_panel(T, N)

        # Benchmark rolling mean
        ref_mean, ref_std = benchmark_function(reference_rolling_mean, data, window, 0, n_runs=3)
        numba_mean, numba_std = benchmark_function(numba_rolling_mean, data, window, 0, n_runs=3)
        speedup = format_speedup(ref_mean, numba_mean)

        print(f"  Rolling Mean:")
        print(f"    Reference: {ref_mean*1000:.1f}ms ± {ref_std*1000:.1f}ms")
        print(f"    Numba:     {numba_mean*1000:.1f}ms ± {numba_std*1000:.1f}ms")
        print(f"    Speedup:   {speedup}")

        # Benchmark rolling std
        ref_std_time, _ = benchmark_function(reference_rolling_std, data, window, 0, 1, n_runs=3)
        numba_std_time, _ = benchmark_function(numba_rolling_std, data, window, 0, 1, n_runs=3)
        speedup_std = format_speedup(ref_std_time, numba_std_time)

        print(f"  Rolling Std:")
        print(f"    Reference: {ref_std_time*1000:.1f}ms")
        print(f"    Numba:     {numba_std_time*1000:.1f}ms")
        print(f"    Speedup:   {speedup_std}")

        # Additional operations (sum, min, max) - Numba only
        sum_time, _ = benchmark_function(numba_rolling_sum, data, window, 0, n_runs=3)
        min_time, _ = benchmark_function(numba_rolling_min, data, window, 0, n_runs=3)
        max_time, _ = benchmark_function(numba_rolling_max, data, window, 0, n_runs=3)

        print(f"  Rolling Sum: {sum_time*1000:.1f}ms (Numba)")
        print(f"  Rolling Min: {min_time*1000:.1f}ms (Numba)")
        print(f"  Rolling Max: {max_time*1000:.1f}ms (Numba)")
        print()


def benchmark_cross_sectional_operations():
    """Benchmark cross-sectional operations."""
    print("=" * 80)
    print("CROSS-SECTIONAL OPERATIONS BENCHMARK")
    print("=" * 80)
    print()

    configs = [
        # (T, N, description)
        (100, 500, "Small universe (100 dates, 500 assets)"),
        (252, 1000, "Medium universe (1 year, 1000 assets)"),
        (500, 2000, "Large universe (2 years, 2000 assets)"),
        (1000, 3000, "Very large universe (4 years, 3000 assets)"),
    ]

    for T, N, desc in configs:
        print(f"{desc}:")
        print(f"  Shape: ({T}, {N})")
        data = generate_panel(T, N)

        # Benchmark rank
        ref_rank, _ = benchmark_function(reference_cs_rank, data, -1, "average", False, n_runs=3)
        numba_rank, _ = benchmark_function(numba_cs_rank, data, -1, False, n_runs=3)
        speedup_rank = format_speedup(ref_rank, numba_rank)

        print(f"  CS Rank:")
        print(f"    Reference: {ref_rank*1000:.1f}ms")
        print(f"    Numba:     {numba_rank*1000:.1f}ms")
        print(f"    Speedup:   {speedup_rank}")

        # Benchmark zscore
        ref_zscore, _ = benchmark_function(reference_cs_zscore, data, -1, 1, 0.0, n_runs=3)
        numba_zscore, _ = benchmark_function(numba_cs_zscore, data, -1, 1, 0.0, n_runs=3)
        speedup_zscore = format_speedup(ref_zscore, numba_zscore)

        print(f"  CS Z-score:")
        print(f"    Reference: {ref_zscore*1000:.1f}ms")
        print(f"    Numba:     {numba_zscore*1000:.1f}ms")
        print(f"    Speedup:   {speedup_zscore}")

        # Benchmark winsorization (Numba only)
        wins_time, _ = benchmark_function(numba_cs_winsorize, data, 0.05, 0.95, -1, n_runs=3)
        print(f"  CS Winsorize: {wins_time*1000:.1f}ms (Numba)")
        print()


def benchmark_realistic_workflow():
    """Benchmark a realistic factor preprocessing workflow."""
    print("=" * 80)
    print("REALISTIC WORKFLOW BENCHMARK")
    print("=" * 80)
    print()

    # Typical quant factor workflow:
    # 1. Load raw factor data (T dates × N assets)
    # 2. Rolling standardization (20-day window)
    # 3. Cross-sectional rank
    # 4. Cross-sectional winsorization

    T, N = 1000, 1000
    window = 20

    print(f"Workflow: {T} dates × {N} assets, window={window}")
    print(f"Steps: Rolling Mean → Rolling Std → CS Rank → CS Winsorize")
    print()

    data = generate_panel(T, N)

    # Reference workflow (partial - no reference winsorize)
    start_ref = time.perf_counter()
    rolling_mean_ref = reference_rolling_mean(data, window, 0)
    rolling_std_ref = reference_rolling_std(data, window, 0, 1)
    zscore_ref = (data - rolling_mean_ref) / rolling_std_ref
    rank_ref = reference_cs_rank(zscore_ref, -1, "average", True)
    time_ref = time.perf_counter() - start_ref

    # Numba workflow
    start_numba = time.perf_counter()
    rolling_mean_numba = numba_rolling_mean(data, window, 0)
    rolling_std_numba = numba_rolling_std(data, window, 0, 1)
    zscore_numba = (data - rolling_mean_numba) / rolling_std_numba
    rank_numba = numba_cs_rank(zscore_numba, -1, True)
    wins_numba = numba_cs_winsorize(rank_numba, 0.05, 0.95, -1)
    time_numba = time.perf_counter() - start_numba

    print(f"Reference workflow: {time_ref:.3f}s")
    print(f"Numba workflow:     {time_numba:.3f}s")
    print(f"Speedup:            {format_speedup(time_ref, time_numba)}")
    print()

    # Verify parity
    np.testing.assert_allclose(rolling_mean_ref, rolling_mean_numba, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(rolling_std_ref, rolling_std_numba, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(rank_ref, rank_numba, rtol=1e-10, atol=1e-12)
    print("✓ Parity verified: Numba results match reference exactly")
    print()


def print_summary():
    """Print summary and recommendations."""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Target Speedup Goals:")
    print("  Rolling operations: 20-100x")
    print("  Cross-sectional operations: 10-50x")
    print()
    print("Legend:")
    print("  ✓  Meets or exceeds target (≥20x for rolling, ≥10x for CS)")
    print("  ~  Approaching target (≥10x for rolling, ≥5x for CS)")
    print("  ✗  Below target")
    print()
    print("Usage:")
    print("  from factor_preprocess.kernels.numba_transforms import (")
    print("      numba_rolling_mean, numba_rolling_std,")
    print("      numba_cs_rank, numba_cs_zscore, numba_cs_winsorize")
    print("  )")
    print()
    print("Requirements:")
    print("  pip install 'factor-preprocess[fast]'  # Installs numba + bottleneck")
    print()


def main():
    """Run all benchmarks."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  NUMBA TRANSFORMS SPEEDUP BENCHMARK".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    if not HAS_NUMBA:
        print("ERROR: Numba not available!")
        print("Install with: pip install numba")
        print()
        return

    print("Numba detected. Running benchmarks...")
    print()

    # Warmup: compile numba functions
    print("Warming up Numba JIT compiler...")
    warmup_data = generate_panel(100, 50)
    numba_rolling_mean(warmup_data, 10, 0)
    numba_rolling_std(warmup_data, 10, 0, 1)
    numba_cs_rank(warmup_data, -1, False)
    numba_cs_zscore(warmup_data, -1, 1, 0.0)
    numba_cs_winsorize(warmup_data, 0.05, 0.95, -1)
    print("✓ Warmup complete")
    print()

    # Run benchmarks
    benchmark_rolling_operations()
    benchmark_cross_sectional_operations()
    benchmark_realistic_workflow()
    print_summary()


if __name__ == "__main__":
    main()
