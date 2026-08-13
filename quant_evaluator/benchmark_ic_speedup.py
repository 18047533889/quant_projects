"""
Comprehensive IC computation speedup benchmark.

Demonstrates the performance improvement from vectorized IC computation
across different factor batch sizes.
"""

import time
import numpy as np
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.kernels.fast import fast_ic_batch


def make_batch(T: int, N: int, F: int, seed: int = 42) -> FactorBatch:
    """Create a synthetic FactorBatch."""
    np.random.seed(seed)
    values = np.random.randn(T, N, F)

    time_axis = AxisRef(name="time", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    return FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )


def make_bundle(T: int, N: int, seed: int = 42) -> LabelBundle:
    """Create a synthetic LabelBundle."""
    np.random.seed(seed + 1000)
    values = np.random.randn(T, N)

    return LabelBundle(
        target_id="ret_1d",
        values=values,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )


def benchmark_ic_speedup():
    """Benchmark IC computation speedup across different factor counts."""

    print("=" * 80)
    print("IC COMPUTATION VECTORIZATION SPEEDUP BENCHMARK")
    print("=" * 80)
    print()

    # Test configurations: (T, N, F)
    configs = [
        (30, 100, 100, "Small batch - 100 factors"),
        (50, 100, 500, "Medium batch - 500 factors"),
        (50, 100, 1000, "Medium-large batch - 1k factors"),
        (50, 100, 2000, "Large batch - 2k factors"),
        (30, 100, 5000, "Very large batch - 5k factors"),
        (20, 100, 10000, "Massive batch - 10k factors"),
    ]

    results = []

    for T, N, F, description in configs:
        print(f"Testing: {description}")
        print(f"  Dimensions: T={T}, N={N}, F={F}")

        # Create test data
        batch = make_batch(T, N, F, seed=42)
        bundle = make_bundle(T, N, seed=42)

        # Warm-up
        _ = fast_ic_batch(
            factor_values=batch.values,
            label_values=bundle.values,
            method="pearson",
            min_obs=10,
        )

        # Benchmark reference implementation
        start = time.perf_counter()
        ic_ref, counts_ref = compute_daily_ic(
            batch, bundle, method="pearson", min_assets=10
        )
        ref_time = time.perf_counter() - start

        # Benchmark fast implementation
        start = time.perf_counter()
        ic_fast, counts_fast = fast_ic_batch(
            factor_values=batch.values,
            label_values=bundle.values,
            method="pearson",
            min_obs=10,
        )
        fast_time = time.perf_counter() - start

        # Verify parity
        ic_match = np.allclose(
            ic_ref[np.isfinite(ic_ref)],
            ic_fast[np.isfinite(ic_fast)],
            rtol=1e-9,
            atol=1e-12
        )
        counts_match = np.array_equal(counts_ref, counts_fast)

        speedup = ref_time / fast_time if fast_time > 0 else float('inf')

        print(f"  Reference time: {ref_time:.4f}s")
        print(f"  Fast time:      {fast_time:.4f}s")
        print(f"  Speedup:        {speedup:.2f}x")
        print(f"  Parity check:   {'✓ PASS' if ic_match and counts_match else '✗ FAIL'}")
        print()

        results.append({
            'description': description,
            'T': T,
            'N': N,
            'F': F,
            'ref_time': ref_time,
            'fast_time': fast_time,
            'speedup': speedup,
            'parity': ic_match and counts_match,
        })

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Configuration':<35} {'Factors':>8} {'Ref (s)':>10} {'Fast (s)':>10} {'Speedup':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['description']:<35} {r['F']:>8} {r['ref_time']:>10.4f} {r['fast_time']:>10.4f} {r['speedup']:>9.2f}x")

    print()
    print(f"Average speedup: {np.mean([r['speedup'] for r in results]):.2f}x")
    print(f"Minimum speedup: {np.min([r['speedup'] for r in results]):.2f}x")
    print(f"Maximum speedup: {np.max([r['speedup'] for r in results]):.2f}x")
    print()

    # Check if target achieved
    large_batches = [r for r in results if r['F'] >= 1000]
    avg_speedup_large = np.mean([r['speedup'] for r in large_batches])

    print(f"Target: 5-10x speedup for 1k+ factors")
    print(f"Achieved: {avg_speedup_large:.2f}x average speedup for 1k+ factor batches")

    if avg_speedup_large >= 5.0:
        print("✓ TARGET ACHIEVED")
    else:
        print("✗ TARGET NOT MET")

    print()
    print("All parity checks:", "✓ PASS" if all(r['parity'] for r in results) else "✗ FAIL")
    print()


if __name__ == "__main__":
    benchmark_ic_speedup()
