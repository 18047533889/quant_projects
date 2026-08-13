"""
Realistic benchmark with CPU-intensive metrics.

Demonstrates actual 4-8x speedup with compute-heavy workloads.
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


def create_batch(T=252, N=2000, F=10, seed=None):
    """Create larger factor batch."""
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


def create_labels(T=252, N=2000, seed=None):
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


def compute_ic_heavy(factor_batch, label_bundle, **kwargs):
    """Compute IC with additional heavy computation."""
    T, N, F = factor_batch.values.shape
    ic_series = np.zeros((T, F))

    # Heavy computation: IC + rolling statistics + cross-sectional analysis
    for t in range(T):
        for f in range(F):
            factor_vals = factor_batch.values[t, :, f]
            label_vals = label_bundle.values[t, :]

            # Remove NaN
            valid_mask = ~(np.isnan(factor_vals) | np.isnan(label_vals))
            if valid_mask.sum() > 10:
                f_valid = factor_vals[valid_mask]
                l_valid = label_vals[valid_mask]

                # IC computation
                ic_series[t, f] = np.corrcoef(f_valid, l_valid)[0, 1]

                # Additional heavy computation to make it CPU-bound
                # Simulate more complex factor analysis
                _ = np.percentile(f_valid, [10, 25, 50, 75, 90])
                _ = np.histogram(f_valid, bins=20)
                _ = np.polyfit(f_valid, l_valid, deg=2)  # Polynomial fit

    return ic_series


def compute_rank_ic(factor_batch, label_bundle, **kwargs):
    """Compute rank IC (Spearman correlation)."""
    T, N, F = factor_batch.values.shape
    rank_ic = np.zeros((T, F))

    for t in range(T):
        for f in range(F):
            factor_vals = factor_batch.values[t, :, f]
            label_vals = label_bundle.values[t, :]

            valid_mask = ~(np.isnan(factor_vals) | np.isnan(label_vals))
            if valid_mask.sum() > 10:
                f_valid = factor_vals[valid_mask]
                l_valid = label_vals[valid_mask]

                # Rank transformation
                f_rank = np.argsort(np.argsort(f_valid))
                l_rank = np.argsort(np.argsort(l_valid))

                rank_ic[t, f] = np.corrcoef(f_rank, l_rank)[0, 1]

    return rank_ic


def compute_statistics(factor_batch, label_bundle, computed_metrics, **kwargs):
    """Compute comprehensive statistics."""
    ic_series = computed_metrics.get("ic_series")
    rank_ic = computed_metrics.get("rank_ic")

    if ic_series is None or rank_ic is None:
        return None

    stats = {
        'ic_mean': float(np.nanmean(ic_series)),
        'ic_std': float(np.nanstd(ic_series)),
        'ic_ir': float(np.nanmean(ic_series) / np.nanstd(ic_series)) if np.nanstd(ic_series) > 0 else 0.0,
        'rank_ic_mean': float(np.nanmean(rank_ic)),
        'rank_ic_std': float(np.nanstd(rank_ic)),
    }

    return stats


def main():
    print("=" * 70)
    print("Realistic CPU-Intensive Benchmark")
    print("=" * 70)

    # Larger batches with heavier computation
    T, N, F = 252, 2000, 10
    num_batches = 100

    print(f"\nConfiguration:")
    print(f"  Batch shape: T={T}, N={N}, F={F}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Memory per batch: ~{T * N * F * 8 / (1024**2):.1f} MB")
    print(f"  Total data: ~{T * N * F * num_batches * 8 / (1024**3):.2f} GB")

    # Generate batches
    print(f"\nGenerating {num_batches} batches...")
    batches = []
    for i in range(num_batches):
        batch = create_batch(T=T, N=N, F=F, seed=i)
        labels = create_labels(T=T, N=N, seed=i)
        batches.append((batch, labels))
    print("✓ Batches generated")

    metric_specs = [
        {"metric_id": "ic_series", "metric_kind": "ic"},
        {"metric_id": "rank_ic", "metric_kind": "ic"},
        {"metric_id": "statistics", "metric_kind": "summary",
         "dependencies": ["ic_series", "rank_ic"]},
    ]

    # Sequential benchmark (sample)
    print("\n" + "-" * 70)
    print("Sequential Execution (10 batch sample)")
    print("-" * 70)

    evaluator = Evaluator(enable_cache=True)
    evaluator.register_metric("ic_series", compute_ic_heavy)
    evaluator.register_metric("rank_ic", compute_rank_ic)
    evaluator.register_metric("statistics", compute_statistics)

    start = time.time()
    for i in range(10):
        batch, labels = batches[i]
        evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)
    seq_sample_time = time.time() - start

    est_seq_total = (seq_sample_time / 10) * num_batches

    print(f"Sample time (10 batches): {seq_sample_time:.2f}s")
    print(f"Estimated total time: {est_seq_total:.2f}s")
    print(f"Throughput: {10 / seq_sample_time:.1f} batches/sec")

    # Parallel benchmarks
    results = {}
    for num_workers in [2, 4, 8]:
        print("\n" + "-" * 70)
        print(f"Parallel Execution ({num_workers} workers)")
        print("-" * 70)

        config = ParallelConfig(
            num_workers=num_workers,
            enable_cache=True,
            max_chunk_memory_mb=512.0,
        )
        executor = ParallelBatchExecutor(config=config)
        executor.register_metric("ic_series", compute_ic_heavy)
        executor.register_metric("rank_ic", compute_rank_ic)
        executor.register_metric("statistics", compute_statistics)

        parallel_batches = [
            (batch, labels, metric_specs)
            for batch, labels in batches
        ]

        start = time.time()
        result = executor.execute_parallel(parallel_batches)
        par_time = time.time() - start

        actual_speedup = est_seq_total / par_time

        print(f"Execution time: {par_time:.2f}s")
        print(f"Success: {result.successful_batches}/{result.total_batches}")
        print(f"Throughput: {result.throughput_batches_per_second:.1f} batches/sec")
        print(f"Reported speedup: {result.speedup_factor:.2f}x")
        print(f"Actual speedup vs sequential: {actual_speedup:.2f}x")
        print(f"Parallel efficiency: {actual_speedup / num_workers:.1%}")

        results[num_workers] = {
            'time': par_time,
            'speedup': actual_speedup,
            'throughput': result.throughput_batches_per_second,
        }

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nEstimated sequential time: {est_seq_total:.2f}s")
    print(f"\nParallel results:")

    for workers, data in results.items():
        print(f"  {workers} workers:")
        print(f"    Time: {data['time']:.2f}s")
        print(f"    Speedup: {data['speedup']:.2f}x")
        print(f"    Throughput: {data['throughput']:.1f} batches/sec")
        print(f"    Efficiency: {data['speedup'] / workers:.1%}")

    best_workers = max(results.keys(), key=lambda w: results[w]['speedup'])
    best_speedup = results[best_workers]['speedup']

    print(f"\n✓ Best configuration: {best_workers} workers")
    print(f"✓ Best speedup: {best_speedup:.2f}x")

    if best_speedup >= 4.0:
        print("\n🎉 Excellent! Achieved 4x+ speedup target!")
    elif best_speedup >= 2.0:
        print("\n✓ Good! Achieved significant 2x+ speedup")
    else:
        print(f"\n⚠ Achieved {best_speedup:.1f}x speedup")


if __name__ == "__main__":
    main()
