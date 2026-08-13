# Integration Test Fixtures

Comprehensive test fixtures for integration testing across the quant platform.

## Overview

This package provides three categories of utilities:

1. **Synthetic Data Generation** (`synthetic_data.py`)
2. **Mock Adapters** (`mock_adapters.py`)
3. **Test Assertion Helpers** (`test_helpers.py`)

## Synthetic Data Generation

Generate realistic factor, label, and exposure data with controlled statistical properties.

### Basic Usage

```python
from integration_tests.fixtures import (
    generate_factor_panel,
    generate_label_bundle,
    PanelConfig,
    DataCharacteristics,
)

# Configure panel dimensions
config = PanelConfig(
    num_times=252,      # One trading year
    num_assets=100,     # Universe size
    num_factors=5,      # Number of factors
    start_date="2024-01-01",
    freq="D",           # Daily frequency
    seed=42,            # Reproducibility
)

# Configure statistical properties
characteristics = DataCharacteristics(
    factor_mean=0.0,
    factor_std=1.0,
    factor_autocorr=0.1,            # Temporal persistence
    cross_sectional_correlation=0.3, # Cross-asset correlation
    missing_rate=0.05,               # 5% missing data
    signal_strength=0.05,            # IC level for labels
)

# Generate factor panel
factor_values, time_index, asset_ids = generate_factor_panel(
    config, characteristics
)
# Returns: (T, N, F) array, DatetimeIndex, asset ID array

# Generate corresponding labels
label_bundle = generate_label_bundle(
    factor_values, 
    time_index,
    characteristics,
)
# Returns: dict with values, timing info, and metadata
```

### Advanced Data Generation

#### Correlated Factors

```python
from integration_tests.fixtures import generate_correlated_factors

# Define correlation structure
correlation_matrix = np.array([
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.4],
    [0.3, 0.4, 1.0],
])

factors = generate_correlated_factors(
    num_times=252,
    num_assets=100,
    num_factors=3,
    correlation_matrix=correlation_matrix,
    seed=42,
)
# Returns: (T, N, F) with specified correlation
```

#### Realistic Market Data

```python
from integration_tests.fixtures import generate_realistic_market_data

market_data = generate_realistic_market_data(
    config,
    include_fundamental=True,
    include_technical=True,
    include_alternative=False,
)
# Returns: dict with factor_panel, factor_names, metadata
# Includes realistic patterns: persistent fundamentals, trending momentum, etc.
```

#### Exposure Matrix

```python
from integration_tests.fixtures import generate_exposure_matrix

exposures, factor_names = generate_exposure_matrix(
    num_times=252,
    num_assets=100,
    characteristics=characteristics,
    seed=42,
)
# Returns: (T, N, K) exposures and factor name list
```

## Mock Adapters

Controllable mock implementations for DataAccess and FactorEngine integration testing.

### DataAccess Adapter

```python
from integration_tests.fixtures import MockDataAccessAdapter, AdapterStatus

# Create stable adapter (no failures)
da_adapter = MockDataAccessAdapter(
    fail_rate=0.0,
    latency_ms=10.0,
    seed=42,
)

# Read factor
response = da_adapter.read_factor(
    factor_id="momentum_20d",
    start_date="2024-01-01",
    end_date="2024-12-31",
    universe="top_500",
)

assert response.status == AdapterStatus.SUCCESS
factor_data = response.data
metadata = response.metadata

# Write factor
response = da_adapter.write_factor(
    factor_id="new_factor",
    data=np.random.randn(100, 50),
    metadata={"timing": "daily", "pit_safe": True},
)

# Read universe
response = da_adapter.read_universe(
    universe_id="top_500",
    as_of_date="2024-01-01",
)
asset_ids = response.data

# Track requests
print(f"Total requests: {da_adapter.request_count}")
print(f"History: {da_adapter.request_history}")
```

### FactorEngine Adapter

```python
from integration_tests.fixtures import MockFactorEngineAdapter

fe_adapter = MockFactorEngineAdapter(
    fail_rate=0.0,
    latency_ms=50.0,
    seed=42,
)

# Register custom operator
def mock_rolling_mean(inputs, params):
    window = params["window"]
    data = inputs["data"]
    # Simple implementation
    return data  # Placeholder

fe_adapter.register_operator(
    "rolling_mean",
    mock_rolling_mean,
    metadata={"category": "time_series", "parameters": ["window"]},
)

# Execute operator
response = fe_adapter.execute_operator(
    operator_name="rolling_mean",
    inputs={"data": np.random.randn(100, 50)},
    params={"window": 20},
)

result = response.data

# Materialize factor expression
response = fe_adapter.materialize_factor(
    factor_expr="momentum(close, 20)",
    start_date="2024-01-01",
    end_date="2024-12-31",
    universe="top_500",
)
```

### Evaluation Backend

```python
from integration_tests.fixtures import MockEvaluationBackend

eval_backend = MockEvaluationBackend(
    fail_rate=0.0,
    latency_ms=30.0,
    seed=42,
)

# Register custom metric
def mock_ic(factor_values, label_values):
    # Compute correlation
    corr = np.corrcoef(factor_values.flatten(), label_values.flatten())[0, 1]
    return {"value": corr, "std_error": 0.01}

eval_backend.register_metric("pearson_ic", mock_ic)

# Evaluate
response = eval_backend.evaluate(
    factor_values=np.random.randn(100, 50, 3),
    label_values=np.random.randn(100, 50),
    metric_ids=("pearson_ic", "rank_ic"),
)

results = response.data
```

### Pre-configured Adapter Factories

```python
from integration_tests.fixtures import (
    create_stable_adapters,
    create_flaky_adapters,
    create_slow_adapters,
)

# Happy path testing
da, fe = create_stable_adapters(seed=42)

# Robustness testing (10-15% failure rate)
da, fe = create_flaky_adapters(seed=42)

# Performance testing (high latency)
da, fe = create_slow_adapters(seed=42)
```

## Test Assertion Helpers

Specialized assertions for validating factor data and evaluation results.

### Panel Shape Validation

```python
from integration_tests.fixtures import assert_panel_shape

assert_panel_shape(
    data=factor_values,
    expected_shape=(252, 100, 5),
    name="factor_panel",
)
```

### Factor Property Validation

```python
from integration_tests.fixtures import assert_factor_properties

assert_factor_properties(
    factor_values=factor_values,
    check_finite=True,               # No infinities
    check_range=(-5.0, 5.0),        # Valid range
    max_missing_rate=0.1,            # Max 10% missing
    name="momentum_factor",
)
```

### Timing Consistency

```python
from integration_tests.fixtures import assert_timing_consistency

assert_timing_consistency(
    decision_time=("2024-01-01", "2024-01-02"),
    label_start_time=("2024-01-02", "2024-01-03"),
    label_end_time=("2024-01-03", "2024-01-04"),
    min_horizon=1,
)
```

### No Lookahead Bias

```python
from integration_tests.fixtures import assert_no_lookahead

assert_no_lookahead(
    factor_values=factor_values,
    label_values=label_values,
    decision_time=decision_times,
    label_start_time=label_start_times,
    min_lag=1,
)
```

### Numeric Comparison

```python
from integration_tests.fixtures import assert_numeric_close

assert_numeric_close(
    actual=computed_ic,
    expected=0.05,
    rtol=1e-5,
    atol=1e-8,
    name="IC",
)
```

### Evaluation Result Comparison

```python
from integration_tests.fixtures import compare_evaluation_results

result1 = {"pearson_ic": {"value": 0.05, "std_error": 0.01}}
result2 = {"pearson_ic": {"value": 0.051, "std_error": 0.01}}

comparison = compare_evaluation_results(
    result1, result2, 
    rtol=1e-2,
)
# Returns: {"pearson_ic": True}
```

### Advanced Assertions

#### Correlation Structure

```python
from integration_tests.fixtures import assert_correlation_structure

expected_corr = np.array([
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.4],
    [0.3, 0.4, 1.0],
])

assert_correlation_structure(
    factor_panel=factor_values,
    expected_correlation=expected_corr,
    rtol=0.1,
)
```

#### Cross-Sectional Neutrality

```python
from integration_tests.fixtures import assert_cross_sectional_neutrality

assert_cross_sectional_neutrality(
    factor_values=neutralized_factor,
    exposure_matrix=exposures,
    rtol=0.05,  # Max 5% correlation with exposures
)
```

#### Stationarity Check

```python
from integration_tests.fixtures import assert_stationary

assert_stationary(
    values=factor_returns,
    max_autocorr=0.95,
    name="factor_returns",
)
```

#### Panel Summary

```python
from integration_tests.fixtures import summarize_panel

summary = summarize_panel(factor_values, name="momentum")
# Returns: {
#     "name": "momentum",
#     "shape": (252, 100, 5),
#     "missing_rate": 0.05,
#     "mean": 0.01,
#     "std": 1.02,
#     "min": -4.5,
#     "max": 4.8,
#     ...
# }
```

## Complete Integration Test Example

```python
import pytest
from integration_tests.fixtures import (
    generate_factor_panel,
    generate_label_bundle,
    PanelConfig,
    DataCharacteristics,
    create_stable_adapters,
    assert_panel_shape,
    assert_factor_properties,
    assert_timing_consistency,
    assert_no_lookahead,
)


def test_end_to_end_evaluation():
    """Complete integration test using fixtures."""
    
    # Setup
    config = PanelConfig(num_times=100, num_assets=50, num_factors=3, seed=42)
    characteristics = DataCharacteristics(signal_strength=0.05, missing_rate=0.05)
    
    # Generate data
    factor_values, time_index, asset_ids = generate_factor_panel(config, characteristics)
    label_bundle = generate_label_bundle(factor_values, time_index, characteristics)
    
    # Validate data quality
    assert_panel_shape(factor_values, (100, 50, 3), "factors")
    assert_factor_properties(factor_values, check_finite=True, max_missing_rate=0.1)
    assert_timing_consistency(
        label_bundle["decision_time"],
        label_bundle["label_start_time"],
        label_bundle["label_end_time"],
    )
    assert_no_lookahead(
        factor_values,
        label_bundle["values"],
        label_bundle["decision_time"],
        label_bundle["label_start_time"],
    )
    
    # Setup adapters
    da_adapter, fe_adapter = create_stable_adapters(seed=42)
    
    # Write factors to mock DA
    for f in range(3):
        response = da_adapter.write_factor(
            factor_id=f"factor_{f}",
            data=factor_values[:, :, f],
            metadata={"timing": "daily"},
        )
        assert response.status.value == "success"
    
    # Compute derived factor via mock FE
    response = fe_adapter.execute_operator(
        "rolling_mean",
        inputs={"data": factor_values[:, :, 0]},
        params={"window": 20},
    )
    assert response.status.value == "success"
    
    print("Integration test passed!")
```

## Design Principles

1. **Realistic but Controllable**: Data has realistic statistical properties but with deterministic generation via seeds
2. **Flexible Configuration**: Easy to adjust dimensions, characteristics, and failure modes
3. **Composable**: Fixtures can be combined and layered for complex scenarios
4. **Observable**: All adapters track requests for debugging and validation
5. **Type-Safe**: Proper typing throughout for IDE support

## Testing Strategy

Use these fixtures to test:

- **Happy path**: Stable adapters, clean data
- **Edge cases**: High missing rates, extreme values, timing edge cases
- **Failure modes**: Flaky adapters, partial failures
- **Performance**: Slow adapters, large panels
- **Integration**: Multi-package workflows (QE + FP, FA + FO, etc.)

## Extension

To add new fixtures:

1. Add generation functions to `synthetic_data.py` following existing patterns
2. Add mock implementations to `mock_adapters.py` with request tracking
3. Add specialized assertions to `test_helpers.py` with clear error messages
4. Update this README with usage examples
5. Add corresponding unit tests in `tests/fixtures/`
