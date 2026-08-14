# Modeling

Model input preparation layer for quantitative factor platform.

## Purpose

Provides **contracts and adapters** for preparing factor data for predictive models, with strict temporal contracts to prevent future leakage.

**Scope**: `FactorAssets -> FactorPreprocess -> ModelInput` (NOT model training)

## Key Features

- **Temporal contracts**: FitWindow, SplitSpec, OutOfFoldSpec with built-in leakage detection
- **Stateless transforms**: rank, zscore, winsorization (reference implementations)
- **Fitted transforms**: Cross-sectional scalers with fit window validation
- **Exposure decomposition**: Soft neutralization and residual computation
- **Adapter pattern**: Reuses factor_preprocess implementations instead of copying

## Installation

```bash
pip install -e .                    # Basic (contracts only)
pip install -e ".[preprocess]"      # With factor_preprocess adapter
```

## Test Results

**90 tests collected: 89 passed, 1 skipped** (0.65s serial)

**Skip reason**: "factor_preprocess is installed" (test validates error when unavailable)

## Quick Example

```python
from datetime import datetime
from modeling import FitWindow, SplitSpec
from modeling.preprocess import create_fitted_scaler

# Define split with leakage prevention
split = SplitSpec(
    split_id="fold_1",
    train_start=datetime(2020, 1, 1),
    train_end=datetime(2020, 12, 31),
    val_start=datetime(2021, 1, 1),
    val_end=datetime(2021, 6, 30),
)

# Validation uses TRAIN period for fitting (no leakage)
fit_window = split.get_fit_window("val")
assert fit_window.fit_end <= split.val_start  # True
```

## Production Readiness

- ✅ Temporal contracts production-ready
- ✅ Leakage prevention enforced at type level
- ✅ Adapter with precise field mapping (requires_industry, requires_size)
- ✅ Fail-closed on unknown inputs
- ⚠️ Reference transforms for testing only (use adapter for production)

## License

Proprietary
