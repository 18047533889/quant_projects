# Streaming Evaluator

Streaming evaluation mode for datasets too large to fit in memory. Processes data in chunks using generator-based loading with constant memory usage.

## Overview

The `StreamingEvaluator` enables evaluation of 100k+ factors on large datasets without loading everything into memory at once. It:

- Processes data through generators that yield chunks
- Accumulates statistics incrementally (sufficient statistics for correlation, coverage, summary stats)
- Maintains constant memory usage regardless of total dataset size
- Supports flexible chunking strategies (by time, assets, or factors)

## Key Features

### Constant Memory Usage
- Processes arbitrary dataset sizes with fixed memory footprint
- Memory usage determined by chunk size, not total data size
- Suitable for datasets that exceed available RAM

### Incremental Computation
- Accumulates sufficient statistics for metrics like IC (correlation)
- Supports multiple metric types: IC, coverage, summary statistics
- Correct mathematical results equivalent to batch computation

### Flexible Data Loading
- Generator-based interface for custom data sources
- Built-in batch-to-stream converter for large in-memory batches
- Configurable chunk dimensions (time, assets, factors)

## Usage

### Basic Streaming Evaluation

```python
from quant_evaluator.runtime.streaming_evaluator import (
    StreamingEvaluator,
    streaming_ic_updater,
    streaming_coverage_updater,
)
from quant_evaluator.planner.dependency_plan import MetricKind

# Create evaluator with chunk configuration
evaluator = StreamingEvaluator(
    chunk_size_time=100,      # Time periods per chunk
    chunk_size_factors=1000,  # Factors per chunk
)

# Register streaming metrics
evaluator.register_streaming_metric("ic", streaming_ic_updater, MetricKind.IC)
evaluator.register_streaming_metric("coverage", streaming_coverage_updater, MetricKind.COVERAGE)

# Define metrics to compute
metric_specs = [
    {"metric_id": "ic", "metric_kind": "ic"},
    {"metric_id": "coverage", "metric_kind": "coverage"},
]

# Evaluate on streaming data
result = evaluator.evaluate_stream(data_generator, metric_specs)

print(f"Chunks processed: {result.chunks_processed}")
print(f"Peak memory: {result.peak_memory_mb:.2f} MB")
print(f"Coverage: {result.get_metric('coverage'):.2%}")
```

### Custom Data Generator

```python
from typing import Iterator, Tuple
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

def load_data_chunks() -> Iterator[Tuple[FactorBatch, LabelBundle]]:
    """
    Generator that yields factor and label chunks.
    
    Load from disk, database, or any other source without 
    loading entire dataset into memory.
    """
    for time_chunk in time_ranges:
        for factor_chunk in factor_ranges:
            # Load chunk from storage
            factor_data = load_factors(time_chunk, factor_chunk)
            label_data = load_labels(time_chunk)
            
            # Create contracts
            batch = FactorBatch(
                factor_ids=factor_chunk.ids,
                time_axis=AxisRef(name="time", dtype="datetime64", size=len(time_chunk)),
                asset_axis=AxisRef(name="asset", dtype="int64", size=num_assets),
                values=factor_data,
            )
            
            labels = LabelBundle(
                target_id="forward_return_1d",
                values=label_data,
                horizon=1,
                decision_time=time_chunk.decisions,
                label_start_time=time_chunk.starts,
                label_end_time=time_chunk.ends,
            )
            
            yield batch, labels

# Use generator
result = evaluator.evaluate_stream(load_data_chunks(), metric_specs)
```

### Large Batch Auto-Chunking

For large batches already in memory, use `evaluate_large_batch` to automatically chunk:

```python
# Large batch: 10,000 factors × 500 time periods × 500 assets
batch = create_large_factor_batch(T=500, N=500, F=10000)
labels = create_label_bundle(T=500, N=500)

# Automatically chunks and processes with constant memory
result = evaluator.evaluate_large_batch(batch, labels, metric_specs)
```

## Streaming Metrics

### Built-in Updaters

Three streaming metric updaters are provided:

#### 1. `streaming_ic_updater`
Computes Information Coefficient (Pearson correlation between factors and labels).

- Accumulates sufficient statistics: sum(x), sum(y), sum(x²), sum(y²), sum(xy), count
- Handles chunking by time, assets, or factors
- Produces IC time series: shape (T, F)

```python
evaluator.register_streaming_metric("ic", streaming_ic_updater, MetricKind.IC)
```

#### 2. `streaming_coverage_updater`
Computes data coverage (fraction of valid/non-NaN observations).

- Tracks valid observation ratio per chunk
- Averages across all chunks
- Produces scalar coverage value

```python
evaluator.register_streaming_metric("coverage", streaming_coverage_updater, MetricKind.COVERAGE)
```

#### 3. `streaming_summary_updater`
Computes summary statistics (mean, std, count).

- Accumulates sum and sum-of-squares
- Computes variance from sufficient statistics
- Produces dict with mean, std, count

```python
evaluator.register_streaming_metric("summary", streaming_summary_updater, MetricKind.SUMMARY)
```

### Custom Streaming Metrics

Create custom streaming metrics by defining an updater function:

```python
from quant_evaluator.runtime.streaming_evaluator import StreamingMetricState

def custom_metric_updater(
    state: StreamingMetricState,
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> StreamingMetricState:
    """
    Update metric state incrementally from chunk.
    
    Args:
        state: Current accumulator state
        factor_batch: Current chunk of factor data
        label_bundle: Current chunk of label data
        
    Returns:
        Updated state with accumulated values
    """
    # Extract data from chunk
    values = factor_batch.values
    
    # Update accumulators
    state.custom_state['my_accumulator'] = state.custom_state.get('my_accumulator', 0)
    state.custom_state['my_accumulator'] += compute_chunk_value(values)
    
    return state

# Register custom metric
evaluator.register_streaming_metric(
    "custom_metric",
    custom_metric_updater,
    MetricKind.CUSTOM,
)
```

## Performance Characteristics

### Memory Usage
- **Constant**: O(chunk_size) regardless of total dataset size
- **Configurable**: Adjust chunk dimensions to fit available memory
- **Typical**: 10-500 MB per chunk for reasonable configurations

### Computation
- **Time Complexity**: O(T × N × F) same as batch processing
- **Overhead**: Minimal (< 5%) compared to full batch processing
- **Throughput**: Millions of observations per second

### Example Scaling

| Total Dataset | Full Batch Memory | Streaming Memory | Memory Reduction |
|---------------|-------------------|------------------|------------------|
| 100k factors, 1k days, 500 assets | ~400 GB | 200 MB | 2000x |
| 10k factors, 2k days, 1k assets | ~160 GB | 80 MB | 2000x |
| 1k factors, 5k days, 3k assets | ~120 GB | 60 MB | 2000x |

## Result Contract

The `StreamingEvaluationResult` contains:

```python
result = evaluator.evaluate_stream(data_gen, metric_specs)

# Finalized metrics
result.metrics: Dict[str, Any]
result.get_metric("ic")  # IC time series
result.get_metric("coverage")  # Coverage scalar

# Execution statistics
result.execution_time_seconds: float
result.chunks_processed: int
result.total_observations_processed: int
result.peak_memory_mb: float

# Metadata
result.metadata["chunk_size_time"]
result.metadata["chunk_size_factors"]
result.metadata["num_metrics"]
```

## Chunking Strategies

### By Time
```python
# Process 100 time periods at a time
evaluator = StreamingEvaluator(chunk_size_time=100, chunk_size_factors=10000)
```
- Good for: Sequential data loading, time-series databases
- Memory: O(chunk_time × N × F)

### By Factors
```python
# Process 1000 factors at a time
evaluator = StreamingEvaluator(chunk_size_time=1000, chunk_size_factors=1000)
```
- Good for: Factor databases, columnar storage
- Memory: O(T × N × chunk_factors)

### Balanced
```python
# Chunk both dimensions
evaluator = StreamingEvaluator(chunk_size_time=100, chunk_size_factors=1000)
```
- Good for: Very large datasets, limited memory
- Memory: O(chunk_time × N × chunk_factors)

## Examples

See `examples/streaming_large_scale_demo.py` for complete working examples:

1. **100k Factor Evaluation**: Processes 100,000 factors with generator-based loading
2. **Auto-Chunking**: Converts large in-memory batch to stream automatically

Run the demo:
```bash
python3 examples/streaming_large_scale_demo.py
```

## Limitations

1. **Metric Requirements**: Metrics must be decomposable into chunk-wise computations
2. **Sequential Processing**: Chunks are processed sequentially (no parallelization)
3. **State Size**: Some metrics may require large accumulator state (e.g., full IC matrix)

## When to Use

**Use streaming evaluation when:**
- Dataset size exceeds available RAM
- Evaluating 10k+ factors on long histories
- Data must be loaded incrementally (from disk, database, network)
- Memory efficiency is critical

**Use batch evaluation when:**
- Dataset fits comfortably in memory
- Need maximum throughput
- Require parallel processing
- Working with moderate factor counts (< 10k)

## Implementation Details

### IC Computation
The IC updater accumulates sufficient statistics and handles three chunking modes:

1. **Time chunks**: Concatenates IC values along time axis
2. **Factor chunks**: Concatenates IC values along factor axis
3. **Asset chunks**: Accumulates statistics (sum x, sum y, etc.) within time/factor cells

The finalization step computes Pearson correlation from accumulated statistics:

```
IC = (n·Σxy - Σx·Σy) / √((n·Σx² - (Σx)²) · (n·Σy² - (Σy)²))
```

### Memory Tracking
Peak memory is estimated by summing:
- Factor batch: `T × N × F × 8` bytes
- Label bundle: `T × N × 8` bytes  
- Validity masks (if present)

Accumulator state is typically much smaller than chunk data.
