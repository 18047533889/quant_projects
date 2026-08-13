"""
Quick Reference: Parallel Batch Evaluation

Fast copy-paste examples for common use cases.
"""

# ============================================================================
# 1. BASIC PARALLEL EXECUTION
# ============================================================================

from quant_evaluator.runtime.parallel_executor import (
    ParallelBatchExecutor,
    ParallelConfig,
)

# Configure and create executor
config = ParallelConfig(num_workers=8)
executor = ParallelBatchExecutor(config=config)

# Register metrics (must be module-level functions, not lambdas)
def compute_ic(factor_batch, label_bundle, **kwargs):
    # Your IC computation
    return ic_series

executor.register_metric("ic_series", compute_ic)

# Prepare batches: list of (factor_batch, label_bundle, metric_specs)
batches = [
    (factor_batch_1, label_bundle_1, metric_specs),
    (factor_batch_2, label_bundle_2, metric_specs),
    # ... more batches
]

# Execute in parallel
result = executor.execute_parallel(batches)

# Check results
print(f"Success: {result.successful_batches}/{result.total_batches}")
print(f"Speedup: {result.speedup_factor:.2f}x")


# ============================================================================
# 2. CONVENIENCE FUNCTION
# ============================================================================

from quant_evaluator.runtime.parallel_executor import evaluate_batches_parallel

# Define all metrics
metric_functions = {
    "ic_series": compute_ic,
    "ic_mean": compute_ic_mean,
    "coverage": compute_coverage,
}

# Execute (one-liner style)
result = evaluate_batches_parallel(
    batches=batches,
    metric_functions=metric_functions,
    num_workers=8,
)


# ============================================================================
# 3. WITH DEPENDENCIES
# ============================================================================

executor = ParallelBatchExecutor(ParallelConfig(num_workers=4))

# Register metrics in dependency order (or any order)
executor.register_metric("ic_series", compute_ic_series)
executor.register_metric("ic_mean", compute_ic_mean)
executor.register_metric("ic_ir", compute_ic_ir)

# Specify dependencies in metric specs
metric_specs = [
    {"metric_id": "ic_series", "metric_kind": "ic"},
    {"metric_id": "ic_mean", "metric_kind": "summary",
     "dependencies": ["ic_series"]},
    {"metric_id": "ic_ir", "metric_kind": "summary",
     "dependencies": ["ic_mean", "ic_series"]},
]

batches = [(batch, labels, metric_specs) for batch, labels in data]
result = executor.execute_parallel(batches)


# ============================================================================
# 4. WITH BUDGET LIMITS
# ============================================================================

from quant_evaluator.runtime.budgets import ComputationBudget

budget = ComputationBudget(
    max_memory_mb=1024.0,
    max_operations=10000,
    allow_overflow=False,
)

result = executor.execute_parallel(batches, budget=budget)


# ============================================================================
# 5. ERROR HANDLING
# ============================================================================

result = executor.execute_parallel(batches)

# Check for errors
if result.failed_batches > 0:
    print(f"Warning: {result.failed_batches} batches failed")

    # Get details of failed batches
    for failed in result.get_failed_results():
        print(f"Batch {failed.task_id}: {failed.error}")

# Process successful results only
for success in result.get_successful_results():
    metrics = success.result.metrics
    # Process metrics...


# ============================================================================
# 6. CUSTOM CONFIGURATION
# ============================================================================

config = ParallelConfig(
    num_workers=8,                      # Worker processes
    max_chunk_memory_mb=512.0,          # Memory per chunk
    cache_size_per_worker_mb=256.0,     # Cache per worker
    enable_cache=True,                  # Enable caching
    use_chunking=True,                  # Enable chunking
    timeout_seconds=300.0,              # 5 min timeout
)

executor = ParallelBatchExecutor(config=config)


# ============================================================================
# 7. ACCESSING INDIVIDUAL RESULTS
# ============================================================================

result = executor.execute_parallel(batches)

# Get specific batch result
batch_5_result = result.get_result(task_id=5)
if batch_5_result and batch_5_result.result:
    ic_mean = batch_5_result.result.get_metric("ic_mean")
    print(f"Batch 5 IC Mean: {ic_mean}")

# Iterate all successful results
for task_result in result.get_successful_results():
    task_id = task_result.task_id
    metrics = task_result.result.metrics
    # Process each batch result...


# ============================================================================
# 8. PERFORMANCE MONITORING
# ============================================================================

result = executor.execute_parallel(batches)

print(f"Execution time: {result.total_execution_time_seconds:.2f}s")
print(f"Throughput: {result.throughput_batches_per_second:.1f} batches/sec")
print(f"Speedup: {result.speedup_factor:.2f}x")
print(f"Parallel efficiency: {result.metadata['parallel_efficiency']:.1%}")
print(f"Cache hits: {result.total_cache_hits}")
print(f"Cache misses: {result.total_cache_misses}")


# ============================================================================
# 9. SINGLE BATCH EXECUTION (convenience)
# ============================================================================

executor = ParallelBatchExecutor()
executor.register_metric("ic_mean", compute_ic_mean)

# Execute single batch (still uses parallel infrastructure)
result = executor.execute_single(
    factor_batch=batch,
    label_bundle=labels,
    metric_specs=[{"metric_id": "ic_mean", "metric_kind": "summary"}],
)

ic_mean = result.get_metric("ic_mean")


# ============================================================================
# 10. FULL EXAMPLE WITH FACTOR CREATION
# ============================================================================

import numpy as np
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

def create_batch(T, N, F, seed):
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

def create_labels(T, N, seed):
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

def compute_mean(factor_batch, **kwargs):
    return float(np.mean(factor_batch.values))

# Create 100 batches
batches = []
for i in range(100):
    batch = create_batch(T=252, N=1000, F=5, seed=i)
    labels = create_labels(T=252, N=1000, seed=i)
    specs = [{"metric_id": "mean", "metric_kind": "custom"}]
    batches.append((batch, labels, specs))

# Execute in parallel
executor = ParallelBatchExecutor(ParallelConfig(num_workers=8))
executor.register_metric("mean", compute_mean)
result = executor.execute_parallel(batches)

print(f"Processed {result.successful_batches} batches")
print(f"Speedup: {result.speedup_factor:.2f}x")


# ============================================================================
# COMMON PITFALLS
# ============================================================================

# ❌ DON'T: Use lambda functions (not picklable)
# executor.register_metric("mean", lambda **kwargs: np.mean(kwargs["factor_batch"].values))

# ✅ DO: Use def functions
def compute_mean(factor_batch, **kwargs):
    return np.mean(factor_batch.values)
executor.register_metric("mean", compute_mean)


# ❌ DON'T: Use closures with external state
# multiplier = 2.0
# def scaled_mean(factor_batch, **kwargs):
#     return np.mean(factor_batch.values) * multiplier  # Won't pickle

# ✅ DO: Pass state via metadata
def scaled_mean(factor_batch, metadata, **kwargs):
    multiplier = metadata.get("multiplier", 1.0)
    return np.mean(factor_batch.values) * multiplier


# ❌ DON'T: Expect shared cache between workers
# Workers have isolated caches, no sharing

# ✅ DO: Understand cache is per-worker for thread safety


# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

# 1. Use for 20+ batches (overhead amortization)
# 2. Each batch should take >100ms (minimize overhead ratio)
# 3. CPU-bound metrics benefit most (I/O-bound sees less speedup)
# 4. Monitor parallel efficiency (aim for >50%)
# 5. Adjust workers based on CPU count (start with CPU_COUNT - 1)
# 6. Enable caching for repeated computations
# 7. Use chunking for large batches to manage memory


# ============================================================================
# WHEN TO USE SEQUENTIAL VS PARALLEL
# ============================================================================

# Use PARALLEL when:
# - 20+ independent batches
# - CPU-bound metrics (IC, correlations, statistics)
# - Each batch > 100ms
# - 4+ CPU cores available

# Use SEQUENTIAL when:
# - < 10 batches
# - I/O-bound (reading from disk/network)
# - Each batch < 50ms
# - Memory constrained
