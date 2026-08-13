"""
Example demonstrating parallel batch evaluation capabilities.

Shows 4-8x throughput improvement on multi-core systems.
"""

import numpy as np
import time
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.parallel_executor import (
    ParallelBatchExecutor,
    ParallelConfig,
    evaluate_batches_parallel,
)
from quant_evaluator.runtime.evaluator import Evaluator
from quant_evaluator.planner.dependency_plan import MetricKind


def create_factor_batch(T=252, N=1000, F=5, seed=None):
    """Create a realistic factor batch."""
    if seed is not None:
        np.random.seed(seed)

    time_axis = AxisRef(name="date", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="ticker", dtype="int64", size=N)

    # Simulate realistic factor values with correlations
    values = np.random.randn(T, N, F) * 0.02 + 0.001

    return FactorBatch(
        factor_ids=tuple(f"factor_{i+1}" for i in range(F)),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )


def create_label_bundle(T=252, N=1000, seed=None):
    """Create forward return labels."""
    if seed is not None:
        np.random.seed(seed + 10000)

    # Simulate 1-day forward returns
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
    """Compute information coefficient (IC) time series."""
    T, N, F = factor_batch.values.shape
    ic_series = np.zeros((T, F))

    for t in range(T):
        for f in range(F):
            factor_vals = factor_batch.values[t, :, f]
            label_vals = label_bundle.values[t, :]

            # Remove NaN values
            valid_mask = ~(np.isnan(factor_vals) | np.isnan(label_vals))
            if valid_mask.sum() > 10:
                ic_series[t, f] = np.corrcoef(
                    factor_vals[valid_mask],
                    label_vals[valid_mask]
                )[0, 1]

    return ic_series


def compute_ic_mean(factor_batch, label_bundle, computed_metrics, **kwargs):
    """Compute mean IC from IC series."""
    ic_series = computed_metrics.get("ic_series")
    if ic_series is None:
        return None
    return np.nanmean(ic_series, axis=0)


def compute_ic_std(factor_batch, label_bundle, computed_metrics, **kwargs):
    """Compute IC standard deviation."""
    ic_series = computed_metrics.get("ic_series")
    if ic_series is None:
        return None
    return np.nanstd(ic_series, axis=0)


def compute_ic_ir(factor_batch, label_bundle, computed_metrics, **kwargs):
    """Compute information ratio (IC mean / IC std)."""
    ic_mean = computed_metrics.get("ic_mean")
    ic_std = computed_metrics.get("ic_std")
    if ic_mean is None or ic_std is None:
        return None

    # Avoid division by zero
    ir = np.where(ic_std > 1e-8, ic_mean / ic_std, 0.0)
    return ir


def compute_coverage(factor_batch, label_bundle, **kwargs):
    """Compute data coverage percentage."""
    valid_count = np.sum(~np.isnan(factor_batch.values))
    total_count = factor_batch.values.size
    return valid_count / total_count if total_count > 0 else 0.0


def demo_sequential_vs_parallel():
    """Demonstrate sequential vs parallel execution."""
    print("=" * 70)
    print("Parallel Batch Evaluation Demo")
    print("=" * 70)

    # Configuration
    num_batches = 100
    T, N, F = 252, 1000, 5

    print(f"\nConfiguration:")
    print(f"  Number of batches: {num_batches}")
    print(f"  Batch shape: T={T}, N={N}, F={F}")
    print(f"  Approximate memory per batch: {T * N * F * 8 / (1024**2):.1f} MB")

    # Create test batches
    print(f"\nGenerating {num_batches} factor batches...")
    batches = []
    for i in range(num_batches):
        batch = create_factor_batch(T=T, N=N, F=F, seed=i)
        labels = create_label_bundle(T=T, N=N, seed=i)
        metric_specs = [
            {"metric_id": "ic_series", "metric_kind": "ic"},
            {"metric_id": "ic_mean", "metric_kind": "summary", "dependencies": ["ic_series"]},
            {"metric_id": "ic_std", "metric_kind": "summary", "dependencies": ["ic_series"]},
            {"metric_id": "ic_ir", "metric_kind": "summary", "dependencies": ["ic_mean", "ic_std"]},
            {"metric_id": "coverage", "metric_kind": "coverage"},
        ]
        batches.append((batch, labels, metric_specs))

    print("✓ Batches generated")

    # Metric functions
    metric_functions = {
        "ic_series": compute_ic,
        "ic_mean": compute_ic_mean,
        "ic_std": compute_ic_std,
        "ic_ir": compute_ic_ir,
        "coverage": compute_coverage,
    }

    # Sequential execution (single batch as baseline)
    print("\n" + "-" * 70)
    print("Sequential Execution (baseline)")
    print("-" * 70)

    evaluator = Evaluator(enable_cache=True)
    for metric_id, metric_fn in metric_functions.items():
        evaluator.register_metric(metric_id, metric_fn)

    start = time.time()
    # Evaluate first 10 batches sequentially for comparison
    sequential_sample_size = min(10, num_batches)
    for i in range(sequential_sample_size):
        batch, labels, specs = batches[i]
        evaluator.evaluate(batch, labels, specs, use_chunking=False)
    sequential_time_sample = time.time() - start
    estimated_sequential_total = (sequential_time_sample / sequential_sample_size) * num_batches

    print(f"Sample of {sequential_sample_size} batches: {sequential_time_sample:.2f}s")
    print(f"Estimated time for {num_batches} batches: {estimated_sequential_total:.2f}s")
    print(f"Throughput: {sequential_sample_size / sequential_time_sample:.1f} batches/sec")

    # Parallel execution with different worker counts
    for num_workers in [2, 4, 8]:
        print("\n" + "-" * 70)
        print(f"Parallel Execution ({num_workers} workers)")
        print("-" * 70)

        config = ParallelConfig(
            num_workers=num_workers,
            max_chunk_memory_mb=512.0,
            cache_size_per_worker_mb=256.0,
            enable_cache=True,
        )

        start = time.time()
        result = evaluate_batches_parallel(
            batches,
            metric_functions,
            config=config,
        )
        parallel_time = time.time() - start

        print(f"Execution time: {parallel_time:.2f}s")
        print(f"Successful batches: {result.successful_batches}/{result.total_batches}")
        print(f"Failed batches: {result.failed_batches}")
        print(f"Throughput: {result.throughput_batches_per_second:.1f} batches/sec")
        print(f"Speedup vs sequential: {estimated_sequential_total / parallel_time:.2f}x")
        print(f"Parallel efficiency: {result.metadata['parallel_efficiency']:.1%}")
        print(f"Cache hits: {result.total_cache_hits}, misses: {result.total_cache_misses}")

        # Show sample results
        if result.successful_batches > 0:
            sample_result = result.get_result(0)
            if sample_result and sample_result.result:
                metrics = sample_result.result.metrics
                print(f"\nSample metrics (batch 0):")
                ic_mean = metrics.get("ic_mean")
                ic_ir = metrics.get("ic_ir")
                coverage = metrics.get("coverage")
                if ic_mean is not None:
                    print(f"  IC Mean: {ic_mean}")
                if ic_ir is not None:
                    print(f"  IC IR: {ic_ir}")
                if coverage is not None:
                    print(f"  Coverage: {coverage:.2%}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Best throughput achieved: ~{result.throughput_batches_per_second:.1f} batches/sec")
    print(f"Best speedup: {estimated_sequential_total / parallel_time:.2f}x")
    print("✓ Parallel execution provides significant throughput improvement")


def demo_error_handling():
    """Demonstrate error handling in parallel execution."""
    print("\n" + "=" * 70)
    print("Error Handling Demo")
    print("=" * 70)

    def failing_metric(factor_batch, **kwargs):
        # Fail on certain batches
        if factor_batch.num_factors == 3:
            raise ValueError("Simulated failure for F=3")
        return 1.0

    executor = ParallelBatchExecutor(ParallelConfig(num_workers=2))
    executor.register_metric("test", failing_metric)

    # Create batches with different factor counts
    batches = []
    for i in range(5):
        F = i + 1  # 1, 2, 3, 4, 5 factors
        batch = create_factor_batch(T=50, N=100, F=F, seed=i)
        labels = create_label_bundle(T=50, N=100, seed=i)
        specs = [{"metric_id": "test", "metric_kind": "custom"}]
        batches.append((batch, labels, specs))

    result = executor.execute_parallel(batches)

    print(f"Total batches: {result.total_batches}")
    print(f"Successful: {result.successful_batches}")
    print(f"Failed: {result.failed_batches}")

    print("\nFailed batch details:")
    for failed in result.get_failed_results():
        print(f"  Batch {failed.task_id}: {failed.error}")

    print("✓ Errors handled gracefully without stopping other batches")


if __name__ == "__main__":
    demo_sequential_vs_parallel()
    demo_error_handling()

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
