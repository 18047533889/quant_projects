# quant_evaluator

Pure Evidence/Metric/Diagnosis Engine for quantitative factor evaluation.

## Overview

`quant_evaluator` is a batch-first evaluator returning typed evidence bundles. It consumes explicit `FactorBatch` and `LabelBundle` from the caller and never infers labels, shifts, or fills missing data.

## Status

**Version:** 0.0.1a1 (Alpha)

**Package Boundaries:**
- ✅ Owns: Evidence metrics, diagnostics, typed contracts
- ❌ Does NOT own: Data fetch, factor computation, admission decisions, representation

## Installation

```bash
cd /home/shw/quant_projects/quant_evaluator
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Core Contracts

### FactorBatch
Batch of factor values with explicit axes and validity. Batch-first is mandatory.

### LabelBundle
Explicit forward labels with strict timing. QE validates timing, never infers or shifts labels.

### EvaluationRequest / EvaluationBundle
Request contains factor batch, labels, metrics. Bundle contains versioned metrics, diagnostics, and optional series refs. No admission decisions.

## Core Metrics

### Coverage Diagnostics
- `compute_coverage`: Fraction of valid (factor, label) pairs
- `compute_per_time_coverage`: Coverage per time period

### Information Coefficient (IC)
- `compute_daily_ic`: Daily Pearson or Spearman IC with pairwise-finite filtering
- `compute_mean_ic`: Mean IC with minimum period validation

### Quantile Metrics
- `assign_quantiles`: Quantile binning with average-tie handling
- `compute_quantile_returns`: Average returns per quantile
- `compute_top_bottom_spread`: Top-bottom quantile spread

### Turnover
- `compute_turnover`: Canonical 0.5 * sum(abs(delta_weights))
- `compute_turnover_series`: Time series of turnover
- `estimate_turnover_from_ranks`: Proxy from rank correlation

## Error Taxonomy

```
QuantEvaluatorError
├── ContractError (schema/input violations)
│   ├── SchemaVersionError
│   ├── MissingInputError
│   ├── InvalidContractError
│   ├── TimingContractError
│   └── SnapshotMismatchError
├── CapabilityError (unsupported features)
│   ├── UnsupportedMetricError
│   └── OptionalDependencyMissing
├── DataError (evidence quality)
│   ├── InsufficientObservations
│   ├── InvalidValidityMask
│   ├── MissingLabelError
│   └── EvidenceUnavailableError
└── NumericalFailure (computation errors)
    └── OverflowOrNonFiniteError
```

## Usage Example

```python
import numpy as np
from quant_evaluator import FactorBatch, LabelBundle, AxisRef
from quant_evaluator.metrics import compute_daily_ic, compute_coverage

# Create factor batch
time_axis = AxisRef(name="time", dtype="datetime64", size=100)
asset_axis = AxisRef(name="asset", dtype="int64", size=500)
values = np.random.randn(100, 500, 1)

batch = FactorBatch(
    factor_ids=("momentum_5d",),
    time_axis=time_axis,
    asset_axis=asset_axis,
    values=values,
)

# Create label bundle with explicit timing
labels = np.random.randn(100, 500)
bundle = LabelBundle(
    target_id="forward_return_1d",
    values=labels,
    horizon=1,
    decision_time=tuple(range(100)),
    label_start_time=tuple(range(100)),
    label_end_time=tuple(range(1, 101)),
)

# Compute metrics
coverage, num_valid, num_total = compute_coverage(batch, bundle)
ic_series, valid_counts = compute_daily_ic(batch, bundle, method="spearman")

print(f"Coverage: {coverage:.2%}")
print(f"Mean RankIC: {np.nanmean(ic_series):.4f}")
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=quant_evaluator --cov-report=term-missing
```

## Package Structure

```
quant_evaluator/
├── __init__.py              # Public API exports
├── pyproject.toml           # Package metadata
├── api/                     # Request/result contracts
├── contracts/               # Core data contracts and errors
├── metrics/                 # Reference metric implementations
│   ├── quality.py          # Coverage diagnostics
│   ├── ic.py               # Information Coefficient
│   ├── quantile.py         # Quantile-based metrics
│   └── turnover.py         # Turnover estimation
├── diagnosis/               # Factor diagnostics
├── adapters/                # Optional DA/FE adapters (stubs)
└── tests/                   # Test suite with golden values
    ├── contracts/
    └── metrics/
```

## Design Principles

1. **Batch-first**: Single-factor API is a thin wrapper over batch
2. **Reference-first**: Reference implementations with clear semantics
3. **Fail-closed**: No implicit fills, shifts, or inference
4. **Typed contracts**: All inputs/outputs use explicit dataclasses
5. **Independent**: Core imports without DA/FE installed

## Not in Scope

- Data fetching or PIT joins (DataAccess owns)
- Factor computation or DSL (FactorEngine owns)
- Admission decisions or registry (FactorAssets owns)
- Preprocessing or neutralization (FactorPreprocess owns)
- Model training or portfolio optimization (out of scope)

## Next Steps (Wave 1+)

- [ ] Freeze public API signatures
- [ ] Add advanced metrics (HAC, multiple testing, portfolio stats)
- [ ] Implement fast/production kernels with reference parity
- [ ] Complete DA/FE adapters after contract freeze
- [ ] Real corpus regression tests
- [ ] Benchmark baselines

## License

[To be determined]

## Contact

Quant Platform Team
