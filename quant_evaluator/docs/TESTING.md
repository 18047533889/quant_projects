# QuantEvaluator Testing Guide

**Version:** 0.0.1a1  
**Last Updated:** 2026-08-14

## Overview

QuantEvaluator uses pytest for all testing. The test suite covers unit tests, integration tests, parity tests, and performance benchmarks.

## Quick Start

### Run All Tests

```bash
cd /home/shw/quant_projects/quant_evaluator
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=quant_evaluator --cov-report=term-missing
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/contracts/ tests/metrics/

# Integration tests
pytest tests/integration/

# Parity tests (fast vs reference)
pytest tests/kernels/test_parity.py

# Performance benchmarks (slow)
pytest tests/kernels/test_performance.py -m slow
```

## Test Organization

```
tests/
├── __init__.py
├── conftest.py                     # Shared fixtures
├── contracts/                      # Contract validation tests
│   ├── test_contracts.py          # FactorBatch, LabelBundle
│   └── __init__.py
├── metrics/                        # Metric correctness tests
│   ├── test_quality.py            # Coverage diagnostics
│   ├── test_ic.py                 # Information Coefficient
│   ├── test_quantile.py           # Quantile analysis
│   ├── test_turnover.py           # Turnover estimation
│   ├── test_ic_summary.py         # ICIR, t-stat, decay
│   ├── test_temporal.py           # Autocorrelation, stability
│   ├── test_exposure.py           # Risk factor exposure
│   ├── test_robustness.py         # HAC, bootstrap
│   ├── test_distribution.py       # Skew, kurtosis
│   ├── test_portfolio_stats.py    # Sharpe, drawdown
│   ├── test_multiple_testing.py   # FDR correction
│   └── __init__.py
├── integration/                    # End-to-end tests
│   ├── test_batch_processing.py   # Full evaluation pipeline
│   ├── test_multi_factor_evaluation.py
│   ├── test_slicing_grouping.py
│   └── __init__.py
├── kernels/                        # Fast implementation tests
│   ├── test_parity.py             # Reference vs fast
│   ├── test_performance.py        # Benchmarks
│   └── __init__.py
├── test_evaluator.py               # Evaluator class tests
├── test_registry.py                # Metric registry
├── test_batch_plan.py              # Batch planning
├── test_dependency_plan.py         # Dependency resolution
├── test_intermediates.py           # Caching tests
└── test_budgets.py                 # Resource tracking
```

## Shared Fixtures

Located in `tests/conftest.py`:

### Basic Fixtures

```python
@pytest.fixture
def small_batch():
    """Small batch for quick tests (T=10, N=20, F=1)"""
    return FactorBatch(
        factor_ids=("test_factor",),
        time_axis=AxisRef("time", "int64", size=10),
        asset_axis=AxisRef("asset", "int64", size=20),
        values=np.random.randn(10, 20, 1),
    )

@pytest.fixture
def medium_batch():
    """Medium batch (T=100, N=500, F=5)"""
    ...

@pytest.fixture
def large_batch():
    """Large batch for integration tests (T=252, N=3000, F=50)"""
    ...

@pytest.fixture
def labels():
    """Forward return labels with proper timing"""
    return LabelBundle(
        target_id="forward_return_1d",
        values=np.random.randn(10, 20),
        horizon=1,
        decision_time=tuple(range(10)),
        label_start_time=tuple(range(10)),
        label_end_time=tuple(range(1, 11)),
    )
```

### Synthetic Data Fixtures

```python
@pytest.fixture
def perfect_factor():
    """Factor with perfect IC=1.0"""
    ...

@pytest.fixture
def noise_factor():
    """Random noise factor with IC≈0"""
    ...

@pytest.fixture
def missing_data_batch():
    """Batch with 20% missing data"""
    ...

@pytest.fixture
def all_nan_batch():
    """Batch with all NaN values (edge case)"""
    ...
```

## Writing New Tests

### Test Structure Template

```python
import pytest
import numpy as np
from quant_evaluator import FactorBatch, LabelBundle
from quant_evaluator.metrics import your_metric_function


class TestYourMetric:
    """Test suite for your_metric_function."""
    
    def test_happy_path(self, small_batch, labels):
        """Test basic functionality with valid inputs."""
        result = your_metric_function(small_batch, labels)
        
        # Assertions
        assert result is not None
        assert result.shape == expected_shape
        assert not np.isnan(result).all()
    
    def test_edge_case_all_nan(self, all_nan_batch, labels):
        """Test behavior with all NaN inputs."""
        result = your_metric_function(all_nan_batch, labels)
        
        # Should return NaN, not crash
        assert np.isnan(result).all()
    
    def test_edge_case_single_observation(self):
        """Test with minimum valid observations."""
        batch = FactorBatch(
            factor_ids=("test",),
            time_axis=AxisRef("time", "int64", size=1),
            asset_axis=AxisRef("asset", "int64", size=2),
            values=np.array([[[1.0], [2.0]]]),
        )
        labels = LabelBundle(
            target_id="test",
            values=np.array([[1.0, 1.0]]),
            horizon=1,
            decision_time=(0,),
            label_start_time=(0,),
            label_end_time=(1,),
        )
        
        result = your_metric_function(batch, labels)
        # Define expected behavior
    
    def test_contract_violation_mismatched_shapes(self, small_batch):
        """Test error handling for invalid contracts."""
        bad_labels = LabelBundle(
            target_id="test",
            values=np.random.randn(5, 10),  # Wrong shape
            horizon=1,
            decision_time=tuple(range(5)),
            label_start_time=tuple(range(5)),
            label_end_time=tuple(range(1, 6)),
        )
        
        with pytest.raises(ContractError):
            your_metric_function(small_batch, bad_labels)
    
    def test_parameter_validation(self, small_batch, labels):
        """Test parameter validation."""
        # Invalid parameter
        with pytest.raises(ValueError):
            your_metric_function(small_batch, labels, invalid_param=-1)
    
    def test_known_result_golden_value(self):
        """Test against known correct result (golden value)."""
        # Fixed input
        batch = create_fixed_batch()
        labels = create_fixed_labels()
        
        result = your_metric_function(batch, labels)
        
        # Compare with known correct value
        expected = 0.123456  # From manual calculation or reference
        np.testing.assert_allclose(result, expected, rtol=1e-6)
```

### Parametrized Tests

For testing multiple scenarios:

```python
@pytest.mark.parametrize("method,expected_ic", [
    ("pearson", 0.95),
    ("spearman", 0.98),
])
def test_ic_methods(small_batch, labels, method, expected_ic):
    """Test different IC methods."""
    ic_series, _ = compute_daily_ic(small_batch, labels, method=method)
    mean_ic = np.nanmean(ic_series)
    assert abs(mean_ic - expected_ic) < 0.1


@pytest.mark.parametrize("n_quantiles", [3, 5, 10])
def test_quantile_counts(small_batch, labels, n_quantiles):
    """Test quantile binning with different n."""
    quantile_returns, quantile_counts = compute_quantile_returns(
        small_batch, labels, n_quantiles=n_quantiles
    )
    
    assert quantile_returns.shape[1] == n_quantiles
    assert quantile_counts.shape[1] == n_quantiles
```

## Contract Tests

Test that contracts enforce invariants:

```python
class TestFactorBatchContract:
    """Test FactorBatch contract validation."""
    
    def test_mismatched_time_dimension(self):
        """Time axis size must match values.shape[0]."""
        with pytest.raises(ValueError):
            FactorBatch(
                factor_ids=("test",),
                time_axis=AxisRef("time", "int64", size=10),
                asset_axis=AxisRef("asset", "int64", size=20),
                values=np.random.randn(5, 20, 1),  # Wrong T
            )
    
    def test_validity_mask_shape(self):
        """Validity mask must match values shape."""
        with pytest.raises(ValueError):
            FactorBatch(
                factor_ids=("test",),
                time_axis=AxisRef("time", "int64", size=10),
                asset_axis=AxisRef("asset", "int64", size=20),
                values=np.random.randn(10, 20, 1),
                validity=np.ones((10, 20), dtype=bool),  # Missing F dim
            )
```

## Integration Tests

Test complete workflows:

```python
def test_full_evaluation_pipeline():
    """Test complete evaluation from batch to bundle."""
    # 1. Create inputs
    batch = create_test_batch(T=100, N=500, F=10)
    labels = create_test_labels(T=100, N=500)
    
    # 2. Create evaluator
    evaluator = Evaluator(
        enable_cache=True,
        max_chunk_memory_mb=256.0,
    )
    
    # 3. Define metrics
    metric_specs = [
        {"metric_id": "mean_ic", "method": "spearman"},
        {"metric_id": "coverage"},
        {"metric_id": "ic_ir"},
        {"metric_id": "turnover"},
    ]
    
    # 4. Evaluate
    result = evaluator.evaluate(batch, labels, metric_specs)
    
    # 5. Verify results
    assert result.metrics is not None
    assert "mean_ic" in result.metrics
    assert "coverage" in result.metrics
    assert result.execution_time_seconds > 0
    
    # 6. Check diagnostics
    for factor_id in batch.factor_ids:
        diagnosis = result.get_diagnosis(factor_id)
        assert diagnosis is not None
        assert diagnosis.num_valid_observations > 0
```

## Parity Tests

Verify fast kernels match reference implementations:

```python
class TestICParity:
    """Test fast IC kernel matches reference."""
    
    def test_parity_small_batch(self, small_batch, labels):
        """Fast and reference should match on small batch."""
        # Reference implementation
        ic_ref, counts_ref = compute_daily_ic_reference(
            small_batch, labels, method="spearman"
        )
        
        # Fast implementation
        ic_fast, counts_fast = compute_daily_ic_fast(
            small_batch, labels, method="spearman"
        )
        
        # Should match within numerical tolerance
        np.testing.assert_allclose(ic_ref, ic_fast, rtol=1e-10)
        np.testing.assert_array_equal(counts_ref, counts_fast)
    
    @pytest.mark.parametrize("T,N,F", [
        (100, 500, 1),
        (252, 3000, 10),
        (500, 5000, 50),
    ])
    def test_parity_various_sizes(self, T, N, F):
        """Test parity across various batch sizes."""
        batch = create_test_batch(T, N, F)
        labels = create_test_labels(T, N)
        
        ic_ref, _ = compute_daily_ic_reference(batch, labels)
        ic_fast, _ = compute_daily_ic_fast(batch, labels)
        
        np.testing.assert_allclose(ic_ref, ic_fast, rtol=1e-10)
```

## Performance Tests

Benchmark critical paths (marked as slow):

```python
@pytest.mark.slow
class TestICPerformance:
    """Benchmark IC computation performance."""
    
    def test_large_batch_speed(self, benchmark):
        """Benchmark large batch IC computation."""
        batch = create_test_batch(T=252, N=5000, F=100)
        labels = create_test_labels(T=252, N=5000)
        
        # Benchmark
        result = benchmark(
            compute_daily_ic,
            batch,
            labels,
            method="spearman",
        )
        
        # Verify not too slow (< 10s for this size)
        assert benchmark.stats['mean'] < 10.0
    
    def test_fast_vs_reference_speedup(self):
        """Measure speedup of fast kernel."""
        batch = create_test_batch(T=252, N=3000, F=50)
        labels = create_test_labels(T=252, N=3000)
        
        # Time reference
        start = time.time()
        compute_daily_ic_reference(batch, labels)
        ref_time = time.time() - start
        
        # Time fast
        start = time.time()
        compute_daily_ic_fast(batch, labels)
        fast_time = time.time() - start
        
        speedup = ref_time / fast_time
        print(f"Speedup: {speedup:.1f}x")
        
        # Should be at least 2x faster
        assert speedup >= 2.0
```

## Golden Value Tests

Tests with fixed inputs and known outputs:

```python
def test_ic_golden_value():
    """Test IC with known correct result."""
    # Fixed factor values (ascending ranks)
    factor_values = np.array([
        [[1.0], [2.0], [3.0], [4.0], [5.0]]
    ])  # Shape: (1, 5, 1)
    
    # Fixed labels (ascending returns)
    label_values = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0]
    ])  # Shape: (1, 5)
    
    batch = FactorBatch(
        factor_ids=("test",),
        time_axis=AxisRef("time", "int64", size=1),
        asset_axis=AxisRef("asset", "int64", size=5),
        values=factor_values,
    )
    
    labels = LabelBundle(
        target_id="test",
        values=label_values,
        horizon=1,
        decision_time=(0,),
        label_start_time=(0,),
        label_end_time=(1,),
    )
    
    # Compute IC
    ic_series, _ = compute_daily_ic(batch, labels, method="spearman")
    
    # Perfect rank correlation = 1.0
    np.testing.assert_allclose(ic_series[0, 0], 1.0, rtol=1e-10)
```

## Test Markers

Use pytest markers to categorize tests:

```python
# Mark slow tests
@pytest.mark.slow
def test_large_scale_evaluation():
    ...

# Mark integration tests
@pytest.mark.integration
def test_full_pipeline():
    ...

# Mark tests requiring optional dependencies
@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_spearman_ic():
    ...
```

Run specific markers:
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Only integration tests
pytest tests/ -m integration
```

## Coverage Reporting

### Generate Coverage Report

```bash
pytest tests/ --cov=quant_evaluator --cov-report=html
```

View report:
```bash
open htmlcov/index.html
```

### Coverage Goals

- **Core metrics:** 100% line coverage
- **Contracts:** 100% branch coverage
- **Integration:** Key workflows covered
- **Overall:** >90% coverage

### Check Coverage

```bash
pytest tests/ --cov=quant_evaluator --cov-report=term --cov-fail-under=90
```

## Continuous Integration

### CI Configuration (example)

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest tests/ --cov=quant_evaluator --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## Debugging Failed Tests

### Run Single Test

```bash
pytest tests/metrics/test_ic.py::TestIC::test_daily_ic_spearman -v
```

### Run with Print Statements

```bash
pytest tests/ -s
```

### Drop into Debugger on Failure

```bash
pytest tests/ --pdb
```

### Show Local Variables on Failure

```bash
pytest tests/ -l
```

## Adding New Metrics

When adding a new metric, create corresponding test file:

1. **Create test file:** `tests/metrics/test_your_metric.py`
2. **Write unit tests:** Cover happy path, edge cases, errors
3. **Add golden value test:** Fixed input/output pair
4. **Add integration test:** Use in full evaluation pipeline
5. **Add parity test** (if fast kernel exists)
6. **Update coverage:** Ensure >90% coverage

## Best Practices

1. **Test names:** Use descriptive names (`test_ic_with_missing_data`, not `test1`)
2. **Fixtures:** Use shared fixtures from `conftest.py`
3. **Assertions:** Use `np.testing.assert_allclose` for floats
4. **Edge cases:** Always test NaN, empty, single observation
5. **Contracts:** Test invalid inputs raise appropriate errors
6. **Determinism:** Use fixed random seeds for reproducibility
7. **Speed:** Keep unit tests fast (<1s each), mark slow tests
8. **Independence:** Tests should not depend on each other

## Troubleshooting

**Q: Tests fail with "Insufficient observations"**  
A: Check `min_assets` parameter or use larger test fixtures

**Q: Numerical precision failures**  
A: Use `rtol`/`atol` in `assert_allclose`, don't use `==` for floats

**Q: Tests pass locally but fail in CI**  
A: Check for non-deterministic behavior (random seeds, file I/O)

**Q: Slow test suite**  
A: Mark slow tests with `@pytest.mark.slow` and skip them during development

---

**Last Updated:** 2026-08-14  
**Contributors:** Quant Platform Team
