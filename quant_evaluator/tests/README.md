# Integration and Extraction Tests for quant_evaluator

This directory contains comprehensive integration tests and import isolation verification for the quant_evaluator package.

## Test Structure

### Integration Tests (`tests/integration/`)

End-to-end workflow tests covering:

1. **Multi-Factor Evaluation** (`test_multi_factor_evaluation.py`)
   - Complete evaluation pipeline with multiple factors
   - IC computation (Pearson and Spearman)
   - Coverage tracking over time
   - Missing data handling
   - Metric preset combinations

2. **Slicing and Grouping** (`test_slicing_grouping.py`)
   - Time window slicing (rolling, expanding)
   - Asset group segmentation (market cap, sectors)
   - Grouped metric aggregation
   - Year-over-year comparisons

3. **Batch Processing** (`test_batch_processing.py`)
   - Large-scale factor batch processing (50+ factors)
   - Chunked time series processing
   - Parallel evaluation simulation
   - Incremental updates
   - High-dimensional scenarios (500 assets, 100 factors)
   - Sparse data handling
   - Cross-batch factor ranking

### Extraction Tests (`tests/extraction/`)

Import isolation verification ensuring core package independence:

1. **Core Import Isolation** (`test_import_isolation.py`)
   - Core package imports without optional dependencies
   - Contract/metric/diagnosis module independence
   - Adapter isolation verification
   - Minimal dependency validation (numpy/scipy only)
   - Error hierarchy verification
   - Public API export validation
   - Namespace isolation checks
   - Import order independence
   - Optional dependency handling

## Running Tests

Run all integration and extraction tests:
```bash
cd /home/shw/quant_projects/quant_evaluator
python3 -m pytest tests/integration/ tests/extraction/ -v
```

Run only integration tests:
```bash
python3 -m pytest tests/integration/ -v
```

Run only extraction tests:
```bash
python3 -m pytest tests/extraction/ -v
```

Run specific test file:
```bash
python3 -m pytest tests/integration/test_multi_factor_evaluation.py -v
```

## Test Coverage

### Integration Tests (23 tests)
- Multi-factor evaluation: 7 tests
- Slicing and grouping: 7 tests  
- Batch processing: 9 tests

### Extraction Tests (22 tests)
- Core import isolation: 5 tests
- Adapter isolation: 3 tests
- Minimal dependencies: 2 tests
- Error hierarchy: 2 tests
- Public API exports: 3 tests
- Namespace isolation: 2 tests
- Import order: 3 tests
- Optional dependency handling: 2 tests

**Total: 45 tests, all passing**

## Key Test Patterns

### Multi-Factor Workflow
```python
batch = FactorBatch(factor_ids=(...), values=...)
bundle = LabelBundle(target_id="ret_1d", values=...)

# Compute metrics
ic_series, _ = compute_daily_ic(batch, bundle)
coverage, _, _ = compute_coverage(batch, bundle)
q_returns, _ = compute_quantile_returns(batch, bundle)
```

### Time Slicing
```python
# Rolling window evaluation
for start in range(0, T - window_size, step):
    end = start + window_size
    sliced_batch = batch[start:end, :, :]
    # Evaluate on window
```

### Asset Grouping
```python
# Sector-based evaluation
for sector_name, asset_slice in sectors:
    sector_batch = batch[:, asset_slice, :]
    # Evaluate per sector
```

### Import Isolation
```python
# Core works without adapters
from quant_evaluator.metrics.ic import compute_daily_ic
# No adapter dependencies required
```

## Notes

- All tests work only with quant_evaluator package - no DataAccess or FactorEngine dependencies
- Tests use synthetic data with controlled characteristics
- Randomness is seeded for reproducibility
- Tests validate both correctness and robustness to edge cases
