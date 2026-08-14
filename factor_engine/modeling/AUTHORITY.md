# Modeling Package Authority Declaration

**Issue**: MODEL2-P0-006  
**Date**: 2026-08-14  
**Status**: RESOLVED

---

## Single Source of Truth

**`factor_engine/modeling/`** is the **AUTHORITATIVE** modeling package for:

- ✓ Model training (`trainer.py`)
- ✓ Walk-forward splitting (`walk_forward.py`)
- ✓ Label-interval purge (`purge_overlap`, `purge_before_boundary`)
- ✓ Embargo enforcement (`apply_embargo`)
- ✓ Artifact lifecycle (`artifact.py`, `predictor.py`)
- ✓ Temporal contracts (`EmbargoSpec`, `ApplicationWindow`, `LabelContract`)
- ✓ Gap days enforcement (`WalkForwardSpec.gap_days`)
- ✓ OOS prediction safety (`ModelArtifact.predict_oos`)

**All temporal leakage prevention enforcement happens here.**

---

## Standalone `modeling/` Package

The standalone package at `/home/shw/quant_projects/modeling/` is:

- **Scope**: Minimal contracts for `factor_preprocess` adapters
- **Not Responsible For**: Model training, enforcement, walk-forward
- **Use Case**: Bridging factor selection (FactorAssets) to factor_preprocess
- **Migration Path**: Use `factor_engine.modeling` for all production code

**This package defines contracts but delegates enforcement to factor_engine.**

---

## Why Two Packages?

Historical reason: The standalone package was created as a minimal interface layer before the full `factor_engine.modeling` redesign. It was intended for adapter contracts only, but its scope was never clearly documented until MODEL2-P0-006.

**Resolution**: Explicit authority declaration in both packages (2026-08-14).

---

## Migration Guide

### If you're using standalone `modeling/`:

#### Walk-Forward Splitting

**Before** (standalone):
```python
from modeling.contracts import SplitSpec

split = SplitSpec(
    split_id="fold1",
    train_start=datetime(2020, 1, 1),
    train_end=datetime(2020, 12, 31),
    val_start=datetime(2021, 1, 1),
    val_end=datetime(2021, 6, 30),
    gap_days=5,  # NOW ENFORCED (as of FM2-P0-006)
)
```

**After** (factor_engine, recommended):
```python
from modeling.dataset import PanelDataset
from modeling.walk_forward import WalkForwardSpec, make_walk_forward_splits

spec = WalkForwardSpec(
    train_lookback_bars=252,
    validation_bars=63,
    test_bars=126,
    gap_days=5,  # Enforced at split creation
)

folds = make_walk_forward_splits(panel_dataset, spec)
# Returns WalkForwardFold objects with purge/embargo already applied
```

#### Fitted Transforms

**Before** (standalone, bypass possible):
```python
from modeling.preprocess.fitted import CrossSectionalScaler

scaler = CrossSectionalScaler()
scaler.fit(X_train, fit_window)

# DANGEROUS: Could bypass with apply_start_time=None
scaler.transform(X_val, apply_start_time=None)  # OLD CODE: Silent bypass
```

**After** (fail-closed, as of FM2-P0-012~015):
```python
from modeling.preprocess.fitted import CrossSectionalScaler

scaler = CrossSectionalScaler()
scaler.fit(X_train, fit_window)

# OOS use: MUST provide application period
scaler.transform(X_val, apply_start_time=val_start)  # NOW REQUIRED

# In-sample use: explicit API
scaler.fit_transform(X_train, fit_window)  # No bypass possible
```

#### Purge and Embargo

**Not available in standalone** — use factor_engine:

```python
from modeling.walk_forward import purge_overlap, purge_and_embargo
from modeling.contracts import LabelContract

label_contract = LabelContract(
    label_name="vwap_to_vwap",
    horizon_bars=1,
)

# Purge training rows whose labels overlap validation
train_ds = purge_overlap(train_ds, validation_ds, label_contract)

# Or combined purge + embargo
train_ds = purge_and_embargo(
    train_ds,
    validation_ds,
    label_contract,
    spec=walk_forward_spec,
)
```

---

## Enforcement Status (Post-Remediation)

| Contract | Factor Engine | Standalone | Status |
|----------|---------------|------------|--------|
| `gap_days` | ✓ Enforced | ✓ Enforced (NEW) | PASS |
| `purge_overlap` | ✓ Enforced | N/A (not its role) | PASS |
| `embargo` | ✓ Enforced | N/A (not its role) | PASS |
| `ApplicationWindow` | ✓ Enforced | N/A (adapters only) | PASS |
| `FittedTransform.transform()` | N/A | ✓ Enforced (NEW) | PASS |
| `ModelArtifact.predict_oos` | ✓ Enforced | N/A (not its role) | PASS |

**All contracts are now runtime-enforced (not just documented).**

---

## Hard Gates

```python
MODELING_SINGLE_AUTHORITY = PASS  # Explicit declaration in both packages
GAP_DAYS_ACTUALLY_ENFORCED = PASS  # SplitSpec.__post_init__ checks calendar gap
PURGE_OVERLAP_ACTUALLY_ENFORCED = PASS  # Wired to all training paths
OOS_PREDICTION_NO_BYPASS = PASS  # predict_oos enforces ApplicationWindow
FITTED_TRANSFORM_REQUIRES_APPLICATION_PERIOD = PASS  # transform() requires apply_start_time
```

---

## Test Coverage

### Factor Engine (Existing, All Passing)
- `tests/modeling/test_gap_days_enforcement.py` — 7 tests (gap_days enforcement)
- `tests/modeling/test_split_purge_overlap_audit.py` — 11 tests (purge correctness)
- `tests/modeling/test_application_window_oos_safety.py` — 20+ tests (OOS safety)

### Standalone (NEW, as of FM2-P0-006)
- `tests/test_gap_days_enforcement.py` — 10 tests (SplitSpec gap enforcement)
- `tests/test_oos_transform_enforcement.py` — 12 tests (transform() enforcement)
- `tests/test_leakage_prevention.py` — 15 existing tests (leakage detection)

**Total**: 75+ tests proving enforcement (not just parameter existence).

---

## References

- **Taskbook**: MODEL2-P0-006 + FM2-P0 series (005, 006, 008-015)
- **Analysis**: `/home/shw/quant_projects/MODEL2_P0_006_SPLIT_BRAIN_ANALYSIS.md`
- **Memory**: `factor-engine-model-split-brain-remediation.md` (to be created)
- **Date**: 2026-08-14

---

## For New Code

**Rule**: Always use `factor_engine.modeling` for:
- Model training and prediction
- Walk-forward splitting
- Purge and embargo
- Temporal safety enforcement

**Exception**: Only use standalone `modeling` for:
- Writing adapters to `factor_preprocess`
- Defining minimal contracts for non-factor_engine consumers

When in doubt, use `factor_engine.modeling`.
