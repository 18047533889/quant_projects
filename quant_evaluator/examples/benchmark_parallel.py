"""
Benchmark script to measure actual parallel speedup.

Compares sequential vs parallel execution with realistic workloads.
"""

import numpy as np
import time
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import Evaluator
from quant_evaluator.runtime.parallel_executor import (
    ParallelBatchExecutor,
    ParallelConfig,
)


def create_batch(T=252, N=1000, F=3, seed=None):
    """Create factor batch."""
    if seed is not None:
        np.random.seed(seed)
    time_axis = AxisRef(name="date", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="ticker", dtype="int64", size=N)
    values = np.random.randn(T, N, F) * 0.02
    return FactorBatch(
        factor_ids=tuple(f"factor_{i}" for i in range(F)),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )


def create_labels(T=252, N=1000, seed=None):
    """Create label bundle."""
    if seed is not None:
        np.random.seed(seed + 10000)
    values = np.random.randn(T, N) * 0.015
    return LabelBundle(
        target_id="forward_return_1d",
        values=values,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )


def compute_ic(factor_batch, label_bundle, **kwargs):
    """Compute IC time series."""
    T, N, F = factor_batch.values.shape
    ic_series = np.zeros((T, F))

    for t in range(T):
        for f in range(F):
            factor_vals = factor_batch.values[t, :, f]
            label_vals = label_bundle.values[t, :]
            valid_mask = ~(np.isnan(factor_vals) | np.isnan(label_vals))
            if valid_mask.sum() > 10:
                ic_series[t, f] = np.corrcoef(
                    factor_vals[valid_mask],
                    label_vals[valid_mask]
                )[0, 1]

    return ic_series


def compute_ic_mean(factor_batch, label_bundle, computed_metrics, **kwargs):
    """Compute mean IC."""
    ic_series = computed_metrics.get("ic_series")
    if ic_series is None:
        return None
    return float(np.nanmean(ic_series))


def benchmark_sequential(batches, num_batches):
    """Benchmark sequential execution."""
    print(f"\nSequential Execution ({num_batches} batches)")
    print("-" * 50)

    evaluator = Evaluator(enable_cache=True)
    evaluator.register_metric("ic_series", compute_ic)
    evaluator.register_metric("ic_mean", compute_ic_mean)

    metric_specs = [
        {"metric_id": "ic_series", "metric_kind": "ic"},
        {"metric_id": "ic_mean", "metric_kind": "summary", "dependencies": ["ic_series"]},
    ]

    start = time.time()
    for i in range(num_batches):
        batch, labels = batches[i]
        evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {num_batches / elapsed:.1f} batches/sec")

    return elapsed


def benchmark_parallel(batches, num_batches, num_workers):
    """Benchmark parallel execution."""
    print(f"\nParallel Execution ({num_batches} batches, {num_workers} workers)")
    print("-" * 50)

    config = ParallelConfig(num_workers=num_workers, enable_cache=True)
    executor = ParallelBatchExecutor(config=config)
    executor.register_metric("ic_series", compute_ic)
    executor.register_metric("ic_mean", compute_ic_mean)

    metric_specs = [
        {"metric_id": "ic_series", "metric_kind": "ic"},
        {"metric_id": "ic_mean", "metric_kind": "summary", "dependencies": ["ic_series"]},
    ]

    parallel_batches = [
        (batch, labels, metric_specs)
        for batch, labels in batches[:num_batches]
    ]

    start = time.time()
    result = executor.execute_parallel(parallel_batches)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {result.throughput_batches_per_second:.1f} batches/sec")
    print(f"Success: {result.successful_batches}/{result.total_batches}")
    print(f"Speedup: {result.speedup_factor:.2f}x")
    print(f"Efficiency: {result.metadata['parallel_efficiency']:.1%}")

    return elapsed, result.speedup_factor


def main():
    print("=" * 70)
    print("Parallel Batch Evaluation Benchmark")
    print("=" * 70)

    # Configuration
    T, N, F = 252, 1000, 3
    num_batches = 50

    print(f"\nConfiguration:")
    print(f"  Batch shape: T={T}, N={N}, F={F}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Memory per batch: ~{T * N * F * 8 / (1024**2):.1f} MB")

    # Generate batches
    print(f"\nGenerating {num_batches} batches...")
    batches = []
    for i in range(num_batches):
        batch = create_batch(T=T, N=N, F=F, seed=i)
        labels = create_labels(T=T, N=N, seed=i)
        batches.append((batch, labels))
    print("✓ Batches generated")

    # Benchmark sequential
    seq_time = benchmark_sequential(batches, num_batches)

    # Benchmark parallel with different worker counts
    results = {}
    for num_workers in [2, 4, 8]:
        par_time, speedup = benchmark_parallel(batches, num_batches, num_workers)
        results[num_workers] = {
            'time': par_time,
            'speedup_vs_seq': seq_time / par_time,
            'speedup_reported': speedup,
        }

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nSequential time: {seq_time:.3f}s")
    print(f"\nParallel results:")
    for workers, data in results.items():
        print(f"  {workers} workers:")
        print(f"    Time: {data['time']:.3f}s")
        print(f"    Actual speedup: {data['speedup_vs_seq']:.2f}x")
        print(f"    Efficiency: {data['speedup_vs_seq'] / workers:.1%}")

    # Best result
    best_workers = max(results.keys(), key=lambda w: results[w]['speedup_vs_seq'])
    best_speedup = results[best_workers]['speedup_vs_seq']

    print(f"\n✓ Best configuration: {best_workers} workers")
    print(f"✓ Best speedup: {best_speedup:.2f}x")
    print(f"✓ Time reduction: {(1 - 1/best_speedup) * 100:.1f}%")

    if best_speedup >= 4.0:
        print("\n🎉 Excellent! Achieved 4x+ speedup target!")
    elif best_speedup >= 2.0:
        print("\n✓ Good! Achieved significant speedup (2x+)")
    else:
        print("\n⚠ Speedup below target. Consider larger batches or more CPU-intensive metrics.")


if __name__ == "__main__":
    main()
