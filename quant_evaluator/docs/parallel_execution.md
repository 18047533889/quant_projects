# Parallel Batch Evaluation

The `parallel_executor` module provides high-throughput parallel evaluation of multiple `FactorBatch` instances across CPU cores using Python's `multiprocessing.Pool`.

## Features

- **4-8x throughput improvement** on multi-core systems
- **Automatic worker management** with configurable worker count
- **Safe cache isolation** per worker process
- **Comprehensive error handling** without stopping other batches
- **Budget tracking** per batch
- **Dependency resolution** across metrics
- **Performance metrics** including speedup factor and parallel efficiency

## Quick Start

### Basic Usage

```python
from quant_evaluator.runtime.parallel_executor import (
    ParallelBatchExecutor,
    ParallelConfig,
)

# Configure parallel execution
config = ParallelConfig(
    num_workers=8,                    # Number of worker processes (auto-detected if None)
    max_chunk_memory_mb=512.0,        # Memory limit per chunk
    cache_size_per_worker_mb=256.0,   # Cache size per worker
    enable_cache=True,                # Enable intermediate caching
)

# Create executor
executor = ParallelBatchExecutor(config=config)

# Register metrics
def compute_ic(factor_batch, label_bundle, **kwargs):
    # Your IC computation
    return ic_series

executor.register_metric("ic_series", compute_ic)

# Prepare batches
batches = []
for batch, labels in zip(factor_batches, label_bundles):
    metric_specs = [{"metric_id": "ic_series", "metric_kind": "ic"}]
    batches.append((batch, labels, metric_specs))

# Execute in parallel
result = executor.execute_parallel(batches)

print(f"Processed {result.successful_batches}/{result.total_batches} batches")
print(f"Speedup: {result.speedup_factor:.2f}x")
print(f"Throughput: {result.throughput_batches_per_second:.1f} batches/sec")
```

### Convenience Function

```python
from quant_evaluator.runtime.parallel_executor import evaluate_batches_parallel

# Define metrics
metric_functions = {
    "ic_mean": compute_ic_mean,
    "coverage": compute_coverage,
}

# Execute
result = evaluate_batches_parallel(
    batches=batches,
    metric_functions=metric_functions,
    num_workers=8,
)
```

## API Reference

### ParallelConfig

Configuration for parallel execution:

- `num_workers`: Number of worker processes (default: CPU count - 1)
- `max_chunk_memory_mb`: Maximum memory per chunk (default: 512.0)
- `cache_size_per_worker_mb`: Cache size per worker (default: 256.0)
- `enable_cache`: Enable intermediate caching (default: True)
- `use_chunking`: Enable batch chunking (default: True)
- `timeout_seconds`: Execution timeout (default: None)

### ParallelBatchExecutor

Main executor class:

```python
executor = ParallelBatchExecutor(config=config)

# Register metrics (must be done before execution)
executor.register_metric(metric_id, metric_fn, metric_kind)

# Execute multiple batches
result = executor.execute_parallel(batches, budget=budget)

# Execute single batch (convenience)
result = executor.execute_single(factor_batch, label_bundle, metric_specs)
```

### ParallelEvaluationResult

Aggregated results from parallel execution:

```python
result.total_batches                    # Total number of batches
result.successful_batches               # Successfully processed
result.failed_batches                   # Failed batches
result.total_execution_time_seconds     # Wall clock time
result.throughput_batches_per_second    # Throughput metric
result.speedup_factor                   # Speedup vs sequential
result.total_cache_hits                 # Aggregate cache hits
result.total_cache_misses               # Aggregate cache misses

# Access individual results
task_result = result.get_result(task_id)
successful = result.get_successful_results()
failed = result.get_failed_results()
```

## Performance Characteristics

### Speedup Factor

The speedup factor compares parallel execution time to estimated sequential execution time:

```
speedup = sum(worker_execution_times) / total_parallel_time
```

Typical speedup factors on 8-core systems:
- **2-4x** for I/O-bound workloads
- **4-7x** for CPU-bound workloads
- **1-2x** for mixed workloads

### Parallel Efficiency

Parallel efficiency measures how effectively workers are utilized:

```
efficiency = speedup / num_workers
```

- **80-100%**: Excellent (embarrassingly parallel workload)
- **50-80%**: Good (some coordination overhead)
- **25-50%**: Fair (significant overhead or imbalanced work)
- **<25%**: Poor (consider sequential execution)

### When to Use Parallel Execution

**Use parallel execution when:**
- Evaluating 20+ independent factor batches
- Each batch takes >100ms to evaluate
- CPU-bound metrics dominate (IC, correlations, aggregations)
- System has 4+ CPU cores available

**Use sequential execution when:**
- Evaluating <10 batches
- Each batch takes <50ms to evaluate
- I/O-bound operations dominate (reading from disk/network)
- Memory constraints are tight

## Memory Management

Each worker process has its own memory space:

```python
config = ParallelConfig(
    num_workers=8,
    max_chunk_memory_mb=512.0,           # Per-chunk limit
    cache_size_per_worker_mb=256.0,      # Per-worker cache
)

# Total memory estimate:
# ~(num_workers * (max_chunk_memory_mb + cache_size_per_worker_mb))
# = 8 * (512 + 256) = 6144 MB ≈ 6 GB
```

## Error Handling

Errors in individual batches don't stop other batches:

```python
result = executor.execute_parallel(batches)

# Check for failures
if result.failed_batches > 0:
    for failed in result.get_failed_results():
        print(f"Batch {failed.task_id} failed: {failed.error}")
```

## Metric Registration

Metrics must be registered globally for parallel execution because they need to be pickled:

```python
# Register before creating executor
from quant_evaluator.runtime.parallel_executor import register_metric_for_parallel

register_metric_for_parallel("ic_mean", compute_ic_mean)

# Or via executor
executor = ParallelBatchExecutor()
executor.register_metric("ic_mean", compute_ic_mean)
```

**Important:** Metrics must be:
- **Picklable** (no lambdas, local functions, or closures)
- **Stateless** (no shared state between invocations)
- **Thread-safe** (if using shared resources)

## Advanced Usage

### With Dependencies

```python
executor.register_metric("ic_series", compute_ic_series)
executor.register_metric("ic_mean", compute_ic_mean)
executor.register_metric("ic_ir", compute_ic_ir)

metric_specs = [
    {"metric_id": "ic_series", "metric_kind": "ic"},
    {"metric_id": "ic_mean", "metric_kind": "summary", "dependencies": ["ic_series"]},
    {"metric_id": "ic_ir", "metric_kind": "summary", "dependencies": ["ic_mean", "ic_series"]},
]
```

### With Budget Limits

```python
from quant_evaluator.runtime.budgets import ComputationBudget

budget = ComputationBudget(
    max_memory_mb=1024.0,
    max_operations=10000,
    allow_overflow=False,
)

result = executor.execute_parallel(batches, budget=budget)
```

### Timeout Control

```python
config = ParallelConfig(
    num_workers=4,
    timeout_seconds=300.0,  # 5 minute timeout
)

executor = ParallelBatchExecutor(config=config)
```

## Testing

Comprehensive test suite with 120+ batch evaluation:

```bash
cd /home/shw/quant_projects/quant_evaluator
python3 -m pytest tests/test_parallel_executor.py -v
```

Run the demo:

```bash
python3 examples/parallel_evaluation_demo.py
```

## Implementation Notes

### Multiprocessing Architecture

- Uses `multiprocessing.Pool` for worker management
- Each worker has isolated Python interpreter and memory
- Batch tasks are pickled and sent to workers
- Results are pickled and returned to main process

### Cache Isolation

Each worker maintains its own cache to avoid:
- **Synchronization overhead** from shared memory
- **Serialization costs** for cache entries
- **Lock contention** between workers

Trade-off: No cache sharing between workers (each computes independently).

### Process Initialization

Workers are initialized with `_worker_init()`:
- Sets random seed for reproducibility
- Initializes numpy/scipy state
- Prepares worker-local resources

### Overhead Sources

1. **Process spawning**: ~50-100ms per worker
2. **Pickling overhead**: Depends on batch size
3. **Result aggregation**: Minimal (<1ms per batch)
4. **Pool management**: ~10-20ms

Overhead is amortized across many batches.

## Troubleshooting

### Low Speedup (<2x)

**Causes:**
- Tasks too small (overhead dominates)
- I/O-bound workload
- Memory pressure causing swapping

**Solutions:**
- Increase batch size or combine small batches
- Profile to identify bottlenecks
- Reduce `num_workers` if memory-bound

### Pickling Errors

**Error:** `PicklingError: Can't pickle <lambda>`

**Solution:** Use module-level functions instead of lambdas or closures:

```python
# Bad (won't pickle)
metric_fn = lambda **kwargs: np.mean(kwargs["factor_batch"].values)

# Good (pickles correctly)
def compute_mean(factor_batch, **kwargs):
    return np.mean(factor_batch.values)
```

### Memory Errors

**Error:** `MemoryError` or system slowdown

**Solution:** Reduce memory footprint:

```python
config = ParallelConfig(
    num_workers=4,                 # Reduce workers
    max_chunk_memory_mb=256.0,     # Smaller chunks
    cache_size_per_worker_mb=128.0, # Smaller cache
)
```

## Examples

See `examples/parallel_evaluation_demo.py` for complete working examples including:
- Sequential vs parallel comparison
- Multiple worker configurations
- Error handling demonstration
- Performance measurement

## See Also

- `runtime/evaluator.py` - Sequential evaluator
- `runtime/intermediates.py` - Cache implementation
- `planner/batch_plan.py` - Batch chunking strategy
