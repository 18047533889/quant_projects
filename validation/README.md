# Runtime Data Validation

Opt-in validation framework for factor computation pipelines providing schema contracts, quality checks, and input sanitization.

## Overview

The `validation/` package provides three complementary validation layers:

1. **Schema Contracts** (`contracts.py`) — Pydantic-based schema validation for data structures
2. **Quality Checks** (`checks.py`) — Statistical validation and quality gates
3. **Input Sanitization** (`sanitize.py`) — Defensive data cleaning and normalization

All validation is **opt-in** with configurable strict mode for production environments.

## Installation

Core validation requires only `numpy`. Schema validation requires `pydantic>=2.0`:

```bash
# Core validation (checks + sanitize)
pip install numpy

# Full validation (including schema contracts)
pip install pydantic>=2.0
```

## Quick Start

### Schema Validation

Validate factor batch data structure and metadata:

```python
from validation import validate_factor_batch, ValidationConfig

# Validate a factor batch
batch = validate_factor_batch(
    factor_name="momentum_20d",
    timestamps=["2024-01-01", "2024-01-02", "2024-01-03"],
    instruments=["000001.SZ", "000002.SZ", "600000.SH"],
    values=np.random.randn(3, 3),
    metadata={"source": "production", "version": "1.0"},
)

# Strict mode for production
config = ValidationConfig(strict=True, allow_nan=False, allow_inf=False)
batch = validate_factor_batch(
    factor_name="momentum_20d",
    timestamps=timestamps,
    instruments=instruments,
    values=values,
    config=config,
)
```

### Quality Checks

Monitor data quality with statistical validation:

```python
from validation import check_data_quality

# Comprehensive quality check
report = check_data_quality(
    values,
    max_nan_fraction=0.5,
    max_inf_fraction=0.0,
    outlier_method="iqr",
    max_outlier_fraction=0.05,
)

print(report.summary())
# Data Quality Report: 1000 rows × 50 cols
#   NaN: 2.5%
#   Inf: 0.0%
#   Zero: 10.2%
#   iqr: 25 outliers (2.5%), bounds=[-3.210, 3.180]
#   Status: PASS

if not report.passed:
    for warning in report.warnings:
        print(f"  Warning: {warning}")
```

### Input Sanitization

Clean and normalize factor inputs with configurable policies:

```python
from validation import sanitize_factor_inputs

# Sanitize with quantitative finance policies
result = sanitize_factor_inputs(
    values,
    replace_inf="clip",
    max_abs_value=1e10,
    winsorize=(0.01, 0.99),
    normalize=True,
    normalize_method="zscore",
)

print(result.summary())
# Sanitization: 42 changes
#   inf_replaced: 2
#   extreme_clipped: 5
#   winsorized: 35

clean_values = result.sanitized
```

## Components

### 1. Schema Contracts (`validation.contracts`)

Pydantic-based validation for data structures:

#### FactorBatchSchema

Validates factor computation output:

```python
from validation import validate_factor_batch

batch = validate_factor_batch(
    factor_name="factor_name",  # required, non-empty, no leading underscore
    timestamps=sorted_dates,     # sorted, no duplicates
    instruments=sorted_codes,    # sorted, no duplicates
    values=np.ndarray,           # 2D numeric array
    metadata={"key": "value"},   # optional
)

# Automatic shape validation
batch.validate_shape()  # values.shape == (len(timestamps), len(instruments))

# Strict mode checks
config = ValidationConfig(strict=True, allow_nan=False)
batch.validate_finiteness(config)
```

#### FeatureBundleSchema

Validates feature vectors for model training:

```python
from validation import validate_feature_bundle

bundle = validate_feature_bundle(
    canonical="momentum",
    operator_semantic_version="1.0.0",
    params={"window": 20},
    normalized_ast_hash="abc123...",  # min 8 chars
    source_snapshot="snapshot_v1",
    values=np.array([...]),           # optional, numeric
    row_ids=["row1", "row2", ...],    # must align with values
)
```

#### LabelBundleSchema

Validates labels for supervised learning:

```python
from validation import validate_label_bundle

bundle = validate_label_bundle(
    label_name="return_1d",
    horizon_bars=1,                   # >= 1
    return_basis="vwap_to_vwap",
    values=np.array([...]),
    row_ids=[...],
)
```

#### PredictionBatchSchema

Validates model predictions:

```python
from validation import validate_prediction_batch

batch = validate_prediction_batch(
    values=np.array([0.1, 0.2, 0.3]),  # 1D numeric
    row_ids=["r1", "r2", "r3"],
    timestamps=["2024-01-01", ...],    # optional
    status="ok",                        # validated enum
    model_version="v1.0",
    artifact_id="artifact_123",
    allow_nan=False,                    # enforce finite predictions
)
```

### 2. Quality Checks (`validation.checks`)

Statistical validation and quality gates:

#### Comprehensive Check

```python
from validation import check_data_quality

report = check_data_quality(
    values,
    max_nan_fraction=0.5,
    max_inf_fraction=0.0,
    max_zero_fraction=0.99,
    outlier_method="iqr",          # "iqr", "zscore", "mad", or None
    outlier_threshold=3.0,
    max_outlier_fraction=0.05,
    check_distribution_stats=True,
)

# Report attributes
report.n_rows, report.n_cols
report.nan_fraction, report.inf_fraction, report.zero_fraction
report.outlier_stats          # OutlierStats or None
report.distribution           # dict of statistics
report.warnings               # list[str]
report.passed                 # bool
```

#### NaN Rate Monitoring

```python
from validation import check_nan_rates

passed, stats, warnings = check_nan_rates(
    values,
    max_total_nan=0.5,
    max_row_nan=0.8,    # time-series dimension
    max_col_nan=0.8,    # cross-sectional dimension
)

# stats: {"total_nan": 0.05, "max_row_nan": 0.12, "n_bad_rows": 2, ...}
```

#### Outlier Detection

```python
from validation import check_outliers

stats = check_outliers(
    values,
    method="iqr",      # "iqr", "zscore", "mad"
    threshold=3.0,
)

print(stats.summary())
# iqr: 50 outliers (5.0%), bounds=[-3.210, 3.180]
```

#### Distribution Analysis

```python
from validation import check_distribution

stats, warnings = check_distribution(
    values,
    check_skewness=True,
    check_kurtosis=True,
    max_skewness=10.0,
    max_kurtosis=100.0,
)

# stats: {"mean": 0.01, "std": 1.5, "median": 0.0, "skewness": 0.3,
#         "excess_kurtosis": 2.1, "min": -5.2, "max": 6.8, ...}
```

### 3. Input Sanitization (`validation.sanitize`)

Defensive data cleaning with configurable policies:

#### Numeric Array Sanitization

```python
from validation import sanitize_numeric_array

result = sanitize_numeric_array(
    values,
    replace_inf="nan",                  # or numeric value
    replace_nan=0.0,                    # or None to keep NaN
    clip_min=-10.0,
    clip_max=10.0,
    winsorize_quantiles=(0.01, 0.99),
    force_finite=False,
)

clean_values = result.sanitized
print(f"Made {result.n_changes} changes")
# Changes: {"inf_replaced": 5, "nan_replaced": 3, "clipped": 12, "winsorized": 20}
```

#### DataFrame Sanitization

```python
from validation import sanitize_dataframe
import pandas as pd

result = sanitize_dataframe(
    df,
    numeric_cols=["col1", "col2"],  # or None for all numeric
    replace_inf="nan",
    replace_nan=None,
    check_sorted_index=True,
    check_duplicated_index=True,
    coerce_dtypes=False,            # attempt to coerce object cols
)

clean_df = result.sanitized
```

#### Factor-Specific Sanitization

```python
from validation import sanitize_factor_inputs

result = sanitize_factor_inputs(
    values,
    strict=False,
    max_abs_value=1e15,
    replace_inf="clip",             # "nan", "clip", "zero"
    winsorize=(0.01, 0.99),
    normalize=True,
    normalize_method="zscore",      # "zscore", "minmax", "robust"
)

clean_values = result.sanitized
```

## Configuration

### Strict Mode

Enable strict validation for production:

```python
from validation import ValidationConfig

# Production configuration
config = ValidationConfig(
    strict=True,
    allow_nan=False,
    allow_inf=False,
    max_nan_fraction=0.0,
    check_sorted=True,
    check_duplicates=True,
    check_dtypes=True,
)

# Apply to validation
batch = validate_factor_batch(..., config=config)
```

### Integration with Factor Engine

Integrate validation into factor computation pipeline:

```python
# After factor computation
from validation import check_data_quality, sanitize_factor_inputs

# 1. Check raw output quality
report = check_data_quality(factor_values)
if not report.passed:
    logger.warning(f"Quality issues: {report.warnings}")

# 2. Sanitize if needed
if report.nan_fraction > 0.1 or report.outlier_stats.outlier_fraction > 0.05:
    result = sanitize_factor_inputs(
        factor_values,
        replace_inf="clip",
        winsorize=(0.01, 0.99),
    )
    factor_values = result.sanitized
    logger.info(f"Sanitized: {result.n_changes} changes")

# 3. Validate final schema
batch = validate_factor_batch(
    factor_name=name,
    timestamps=dates,
    instruments=stocks,
    values=factor_values,
    config=ValidationConfig(strict=True),
)
```

## Testing

Run the validation test suite:

```bash
# All validation tests
pytest tests/validation/ -v

# Specific test modules
pytest tests/validation/test_contracts.py -v
pytest tests/validation/test_checks.py -v
pytest tests/validation/test_sanitize.py -v

# Skip tests requiring pydantic
pytest tests/validation/ -v -k "not pydantic"
```

## Design Principles

1. **Opt-in**: All validation is explicit; no automatic validation unless requested
2. **Composable**: Each validation layer can be used independently
3. **Configurable**: Strict mode for production, permissive mode for research
4. **Non-invasive**: Original data is never modified (copies are made)
5. **Informative**: Detailed reports and warnings explain validation failures

## Performance Considerations

- Schema validation (pydantic): ~1-10ms per validation
- Quality checks: O(n) for most checks, O(n log n) for sorting-based checks
- Sanitization: O(n) array operations with copy-on-write

For large-scale batch processing, consider:
- Skip distribution statistics for huge arrays (`check_distribution_stats=False`)
- Use sampling for quality checks on very large datasets
- Apply sanitization only when quality checks fail

## Future Enhancements

Planned additions:
- Cross-sectional consistency checks (correlation stability, rank preservation)
- Time-series continuity validation (regime shifts, structural breaks)
- Multi-factor consistency checks (correlation matrix validation)
- Automated sanitization policy selection based on quality report
- Integration with factor registry for per-factor validation rules
