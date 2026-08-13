# Diagnosis Module

The diagnosis module provides comprehensive factor quality assessment capabilities for `quant_evaluator`.

## Overview

Two main components:

1. **`factor.py`** - `FactorDiagnosis` with coverage/validity/distribution summaries
2. **`warnings.py`** - `WarningSystem` for detecting and reporting issues with structured severity levels

## Key Features

### FactorDiagnosis

Computes:
- **Coverage**: valid observation ratio, missing counts
- **Validity**: detection of NaN/Inf values
- **Distribution**: min/max/mean, constant detection
- **Warnings**: simple text warnings tuple

### WarningSystem

Structured warning detection with configurable thresholds:

- **Coverage issues** (3 severity levels)
  - Critical: coverage < 50%
  - Medium: coverage < 80%  
  - Low: coverage < target threshold
  
- **Constant values**: zero variance detection

- **Validity issues**: NaN and Inf detection

- **Range issues**: suspiciously narrow value ranges

- **Outlier detection**: extreme values via z-score (|z| > 10 default)

Each warning includes:
- `factor_id`
- `severity`: CRITICAL, HIGH, MEDIUM, LOW, INFO
- `category`: coverage, constant, validity, range, outliers
- `message`: human-readable description
- `details`: optional structured metadata dict

## Usage

```python
from quant_evaluator.diagnosis import (
    diagnose_factor,
    diagnose_all_factors,
    WarningSystem,
    WarningSeverity,
)

# Basic diagnosis
diagnosis = diagnose_factor(factor_batch, factor_idx=0)
print(f"Coverage: {diagnosis.coverage:.2%}")
print(f"Valid: {diagnosis.num_valid_observations}")

# Structured warnings
ws = WarningSystem(
    missing_rate_high=0.5,
    outlier_z_threshold=10.0,
)
warnings = ws.diagnose_factor(diagnosis, factor_batch=batch, factor_idx=0)

# Format output
print(ws.format_warnings(warnings, include_details=True))

# Filter by severity
critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]

# Batch diagnosis
diagnostics = diagnose_all_factors(factor_batch)
warnings_by_factor = ws.diagnose_all_factors(diagnostics, factor_batch=batch)
```

## Configuration

`WarningSystem` thresholds are configurable:

```python
ws = WarningSystem(
    missing_rate_high=0.5,       # Critical threshold
    missing_rate_medium=0.2,     # Medium threshold
    coverage_low=0.8,            # Low warning threshold
    outlier_z_threshold=10.0,    # Z-score for outliers
    constant_tolerance=1e-12,    # Epsilon for constant detection
)
```

## Testing

All 32 tests pass:
- 11 tests for `diagnose_factor` and `diagnose_all_factors`
- 21 tests for `WarningSystem` methods and formatting

Run: `pytest tests/test_diagnosis.py -v`

## Example

See `examples/example_diagnosis.py` for complete usage patterns including:
- Basic factor diagnosis
- Structured warning detection
- Batch processing with multiple factors
- Severity filtering and reporting
