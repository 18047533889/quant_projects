"""
Demo: Streaming evaluation for large-scale factor datasets.

Shows how to use StreamingEvaluator to process 100k+ factors with constant
memory usage through generator-based data loading.
"""

import numpy as np
import time
from typing import Iterator, Tuple

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.streaming_evaluator import (
    StreamingEvaluator,
    streaming_ic_updater,
    streaming_coverage_updater,
    streaming_summary_updater,
)
from quant_evaluator.planner.dependency_plan import MetricKind


def create_synthetic_data_generator(
    total_time: int = 1000,
    num_assets: int = 500,
    total_factors: int = 100000,
    chunk_time: int = 100,
    chunk_factors: int = 1000,
) -> Iterator[Tuple[FactorBatch, LabelBundle]]:
    """
    Generator that yields synthetic factor data in chunks.

    Simulates loading data from disk or database without loading entire dataset.
    """
    print(f"Creating generator for {total_factors} factors, {total_time} time periods, {num_assets} assets")
    print(f"Chunk size: {chunk_time} time periods × {chunk_factors} factors")
    print(f"Total memory if loaded at once: ~{total_time * num_assets * total_factors * 8 / (1024**3):.2f} GB")
    print(f"Chunk memory: ~{chunk_time * num_assets * chunk_factors * 8 / (1024**2):.2f} MB\n")

    # Generate labels once (shared across all factor chunks)
    label_values = np.random.randn(total_time, num_assets)

    # Yield chunks
    for t_start in range(0, total_time, chunk_time):
        t_end = min(t_start + chunk_time, total_time)

        for f_start in range(0, total_factors, chunk_factors):
            f_end = min(f_start + chunk_factors, total_factors)

            # Simulate loading chunk from storage
            T_chunk = t_end - t_start
            F_chunk = f_end - f_start

            # Create factor data with some correlation to labels
            factor_values = np.random.randn(T_chunk, num_assets, F_chunk)

            # Add some signal (correlation with labels)
            signal_strength = np.random.uniform(0.0, 0.3, size=F_chunk)
            for f in range(F_chunk):
                factor_values[:, :, f] = (
                    factor_values[:, :, f] * (1 - signal_strength[f]) +
                    label_values[t_start:t_end, :] * signal_strength[f]
                )

            # Create batch
            time_axis = AxisRef(name="time", dtype="datetime64", size=T_chunk)
            asset_axis = AxisRef(name="asset", dtype="int64", size=num_assets)

            batch = FactorBatch(
                factor_ids=tuple(f"factor_{i}" for i in range(f_start, f_end)),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values,
            )

            # Create labels
            labels = LabelBundle(
                target_id="forward_return_1d",
                values=label_values[t_start:t_end, :],
                horizon=1,
                decision_time=tuple(range(t_start, t_end)),
                label_start_time=tuple(range(t_start, t_end)),
                label_end_time=tuple(range(t_start + 1, t_end + 1)),
            )

            yield batch, labels


def demo_streaming_evaluation():
    """Demonstrate streaming evaluation on large dataset."""
    print("=" * 80)
    print("Streaming Evaluation Demo: 100k Factors")
    print("=" * 80)
    print()

    # Configuration
    total_time = 500
    num_assets = 500
    total_factors = 100000
    chunk_time = 100
    chunk_factors = 5000

    # Create streaming evaluator
    evaluator = StreamingEvaluator(
        chunk_size_time=chunk_time,
        chunk_size_factors=chunk_factors,
    )

    # Register streaming metrics
    evaluator.register_streaming_metric("ic", streaming_ic_updater, MetricKind.IC)
    evaluator.register_streaming_metric("coverage", streaming_coverage_updater, MetricKind.COVERAGE)
    evaluator.register_streaming_metric("summary", streaming_summary_updater, MetricKind.SUMMARY)

    metric_specs = [
        {"metric_id": "ic", "metric_kind": "ic"},
        {"metric_id": "coverage", "metric_kind": "coverage"},
        {"metric_id": "summary", "metric_kind": "summary"},
    ]

    # Create data generator
    data_gen = create_synthetic_data_generator(
        total_time=total_time,
        num_assets=num_assets,
        total_factors=total_factors,
        chunk_time=chunk_time,
        chunk_factors=chunk_factors,
    )

    # Evaluate
    print("Starting streaming evaluation...")
    start_time = time.time()

    result = evaluator.evaluate_stream(data_gen, metric_specs)

    elapsed = time.time() - start_time

    # Report results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Execution time: {elapsed:.2f} seconds")
    print(f"Chunks processed: {result.chunks_processed}")
    print(f"Total observations: {result.total_observations_processed:,}")
    print(f"Peak memory usage: {result.peak_memory_mb:.2f} MB")
    print(f"Throughput: {result.total_observations_processed / elapsed / 1e6:.2f} M obs/sec")
    print()

    # Metric results
    print("Metrics:")
    print(f"  Coverage: {result.get_metric('coverage'):.2%}")

    summary = result.get_metric('summary')
    print(f"  Summary - Mean: {summary['mean']:.6f}, Std: {summary['std']:.6f}, Count: {summary['count']:,}")

    ic_values = result.get_metric('ic')
    print(f"  IC shape: {ic_values.shape}")
    print(f"  IC mean: {np.nanmean(ic_values):.6f}")
    print(f"  IC std: {np.nanstd(ic_values):.6f}")
    print(f"  Valid IC values: {np.sum(np.isfinite(ic_values)):,} / {ic_values.size:,}")
    print()

    # Memory efficiency
    full_memory_gb = total_time * num_assets * total_factors * 8 / (1024**3)
    memory_reduction = full_memory_gb * 1024 / result.peak_memory_mb
    print(f"Memory efficiency: {memory_reduction:.1f}x reduction")
    print(f"  (Would need {full_memory_gb:.2f} GB for full batch, used {result.peak_memory_mb:.2f} MB)")
    print()


def demo_large_batch_conversion():
    """Demonstrate automatic chunking of large batch."""
    print("=" * 80)
    print("Large Batch Auto-Chunking Demo")
    print("=" * 80)
    print()

    # Create large batch
    T, N, F = 500, 500, 10000
    print(f"Creating large batch: T={T}, N={N}, F={F}")
    print(f"Total size: {T * N * F * 8 / (1024**2):.2f} MB\n")

    time_axis = AxisRef(name="time", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    factor_values = np.random.randn(T, N, F).astype(np.float32)
    label_values = np.random.randn(T, N).astype(np.float32)

    batch = FactorBatch(
        factor_ids=tuple(f"factor_{i}" for i in range(F)),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values,
        dtype="float32",
    )

    labels = LabelBundle(
        target_id="forward_return_1d",
        values=label_values,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )

    # Create evaluator with small chunk size
    evaluator = StreamingEvaluator(
        chunk_size_time=100,
        chunk_size_factors=1000,
    )

    evaluator.register_streaming_metric("coverage", streaming_coverage_updater, MetricKind.COVERAGE)

    metric_specs = [{"metric_id": "coverage", "metric_kind": "coverage"}]

    # Evaluate
    print("Processing with automatic chunking...")
    start_time = time.time()

    result = evaluator.evaluate_large_batch(batch, labels, metric_specs)

    elapsed = time.time() - start_time

    # Report
    print(f"\nProcessed {result.chunks_processed} chunks in {elapsed:.2f} seconds")
    print(f"Coverage: {result.get_metric('coverage'):.2%}")
    print(f"Peak memory: {result.peak_memory_mb:.2f} MB")
    print()


if __name__ == "__main__":
    # Run streaming demo with 100k factors
    demo_streaming_evaluation()

    print("\n" + "=" * 80)
    print()

    # Run large batch auto-chunking demo
    demo_large_batch_conversion()
