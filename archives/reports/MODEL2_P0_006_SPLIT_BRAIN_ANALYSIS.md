# MODEL2-P0-006: Modeling Split-Brain Analysis & Remediation Plan

**Date**: 2026-08-14  
**Status**: CRITICAL - Dual authority sources with conflicting enforcement  
**Scope**: FM2-P0 series (005, 006, 008-015)

---

## Executive Summary

Two separate Modeling packages exist with **conflicting implementations**:

1. **`factor_engine/modeling/`** — Authoritative, production-ready package with full enforcement
2. **`modeling/modeling/`** — Standalone "model input preparation" package with **unenforced contracts**

**Critical Finding**: The standalone package defines `gap_days`, `FitWindow`, and `ApplicationWindow` contracts but **does NOT enforce them**. This creates a false sense of safety where contracts exist but temporal leakage is not prevented.

---

## Authority Decision

**SINGLE AUTHORITY**: `factor_engine/modeling/` is the canonical modeling package.

**Evidence**:
- Full implementation: EmbargoSpec, ApplicationWindow, WalkForwardSpec, purge_overlap, predict_oos
- 30+ tests with actual enforcement validation (test_gap_days_enforcement.py, test_application_window_oos_safety.py)
- Integrated with factor_engine operators and execution
- Production-ready with full trainer/artifact/predictor lifecycle

**Standalone package role**: Minimal contracts for **adapters only** (factor_preprocess integration). It should NOT reimplement enforcement logic.

---

## Gap Analysis

### FM2-P0-005/006: gap_days NOT Enforced in Standalone

**File**: `modeling/modeling/contracts.py:66-89`

```python
class SplitSpec:
    gap_days: int = 0
    
    def __post_init__(self):
        # Checks val_start >= train_end (line 78)
        if self.val_start < self.train_end:
            raise SplitError("val_start must be >= train_end (no overlap)")
        # BUT: Does NOT check val_start >= train_end + gap_days
```

**Status**: **FAIL** — Parameter exists but enforcement missing  
**Impact**: Silent temporal leakage (features from gap period leak into validation)

**Factor_engine enforcement** (working correctly):
- `walk_forward.py:246-249` — Adds gap_days to validation start position
- `walk_forward.py:300-325` — check_fold_order validates actual calendar gap
- `test_gap_days_enforcement.py` — 7 tests proving enforcement

---

### FM2-P0-008~011: purge_overlap Wiring

**File**: `factor_engine/modeling/trainer.py:336-338`

```python
train_ds = purge_and_embargo(train_ds, validation_ds, label_contract)
if validation_ds is None and evaluation_boundary is not None:
    train_ds = purge_before_boundary(train_ds, evaluation_boundary, label_contract)
```

**Status**: **PASS** — Wired correctly to all training paths  
**Evidence**:
- `test_split_purge_overlap_audit.py` — 11 tests proving label-interval purge correctness
- `walk_forward.py:343-414` — purge_overlap with validation logging and fail-closed semantics

**Standalone package**: Does NOT have purge_overlap (correctly — not its responsibility)

---

### FM2-P0-012~015: OOS Prediction Bypass

**File**: `factor_engine/modeling/artifact.py:427-482`

**Public OOS API** (correctly enforced):
```python
def predict_oos(self, X, *, application_window: Any, dates=None):
    if application_window is None:
        raise ValueError("predict_oos requires application_window")
    # Validates dates against window
    Xt = self.preprocessing.transform(X, application_window=application_window)
```

**Internal API** (intentionally unchecked):
```python
def predict(self, X):
    # No application_window required (in-sample/internal use only)
    Xt = self.preprocessing.transform(X)
```

**Status**: **PARTIAL** — Public API enforced, but transform() is placeholder

**Gap**: `FrozenPreprocessing.transform()` line 241-263:
```python
def transform(self, X, *, application_window: Any = None):
    if application_window is not None:
        # MODEL-P0-005: Optional application window validation
        # (not enforced in base implementation for backward compatibility,
        # but ModelArtifact.predict_oos enforces it at the public API level)
        pass  # <-- PLACEHOLDER, NO ACTUAL ENFORCEMENT
```

**Impact**: The window is checked at `predict_oos()` level (dates validation), but the preprocessing transform itself does NOT validate temporal boundaries.

---

### Standalone Package: CrossSectionalScaler.transform()

**File**: `modeling/modeling/preprocess/fitted.py:135-151`

```python
def transform(self, X, apply_start_time: Optional[Any] = None):
    if not self._fitted:
        raise FitWindowError("Transform must be fitted before applying")
    
    # For OOS data, require explicit application period
    # Only allow None for in-sample (fit_transform on training data)
    if apply_start_time is not None:
        self._validate_temporal_ordering(apply_start_time)
    # <-- If apply_start_time is None, NO validation happens
```

**Status**: **FAIL** — Silent bypass when `apply_start_time=None`  
**Expected**: Should raise error for public transform() calls without apply_start_time  
**Actual**: Silently accepts None (caller can bypass all leakage checks)

---

## Hard Gates (Current Status)

```python
MODELING_SINGLE_AUTHORITY = FAIL  # No explicit declaration
GAP_DAYS_ACTUALLY_ENFORCED = FAIL  # Standalone SplitSpec not enforcing
PURGE_OVERLAP_ACTUALLY_ENFORCED = PASS  # Factor_engine correctly wired
OOS_PREDICTION_NO_BYPASS = PARTIAL  # predict_oos enforced, transform() placeholder
FITTED_TRANSFORM_REQUIRES_APPLICATION_PERIOD = FAIL  # Standalone allows None silently
```

---

## Remediation Tasks

### 1. Establish Single Authority (MODEL2-P0-006)

**Action**: Add authority declaration to both packages

**File**: `factor_engine/modeling/__init__.py`
```python
"""FactorEngine model-layer redesign package (AUTHORITATIVE).

This is the SINGLE SOURCE OF TRUTH for model training, artifacts, walk-forward,
embargo, purge, and temporal contracts. The standalone `modeling/` package is
for adapter contracts only and MUST NOT reimplement enforcement logic.
"""
```

**File**: `modeling/modeling/__init__.py`
```python
"""Modeling: Model input preparation layer (ADAPTERS ONLY).

This package provides minimal contracts for adapting factor_preprocess to
factor_engine. It does NOT implement model training, walk-forward, or enforcement.

For actual model training, use factor_engine.modeling (the authoritative package).
"""
```

### 2. Fix gap_days Enforcement in Standalone (FM2-P0-005/006)

**File**: `modeling/modeling/contracts.py:66-90`

**Current**:
```python
def __post_init__(self):
    if self.val_start < self.train_end:
        raise SplitError("val_start must be >= train_end (no overlap)")
```

**Fixed**:
```python
def __post_init__(self):
    if self.val_start < self.train_end:
        raise SplitError("val_start must be >= train_end (no overlap)")
    
    # FM2-P0-006: Enforce gap_days
    if self.gap_days > 0 and self.val_start is not None:
        required_gap_days = self.gap_days + 1  # gap_days=5 means 6 calendar days apart
        actual_gap_days = (self.val_start - self.train_end).days
        if actual_gap_days < required_gap_days:
            raise SplitError(
                f"gap_days={self.gap_days} requires {required_gap_days} days between "
                f"train_end and val_start, but actual gap is {actual_gap_days} days"
            )
```

### 3. Fix CrossSectionalScaler.transform() (FM2-P0-012~015)

**File**: `modeling/modeling/preprocess/fitted.py:44-59`

**Current**:
```python
def transform(self, X, apply_start_time: Optional[Any] = None):
    """Apply fitted transform to new data.
    
    Args:
        apply_start_time: Start time of data (required for OOS validation)
    """
    pass
```

**Fixed**:
```python
def transform(self, X, apply_start_time: Any):
    """Apply fitted transform to new data (OOS ONLY).
    
    Args:
        apply_start_time: Start time of data (REQUIRED for OOS validation).
            For in-sample use during training, use fit_transform() instead.
    
    Raises:
        TypeError: If apply_start_time is None (use fit_transform for in-sample)
        FutureLeakageError: If apply_start_time is before fit_window.fit_end
    """
    if apply_start_time is None:
        raise TypeError(
            "transform() requires apply_start_time for OOS safety. "
            "For in-sample (training), use fit_transform() instead."
        )
    self._validate_temporal_ordering(apply_start_time)
    # ... rest of transform
```

### 4. Add Tests for Standalone Package (FM2-P0-006)

**File**: `modeling/tests/test_gap_days_enforcement.py` (NEW)

```python
def test_gap_days_enforced_in_split_spec():
    """SplitSpec.__post_init__ must enforce gap_days."""
    with pytest.raises(SplitError, match="gap_days"):
        SplitSpec(
            split_id="insufficient_gap",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 3),  # Only 3 days gap
            val_end=datetime(2021, 6, 30),
            gap_days=5,  # Requires 6 days
        )

def test_gap_days_zero_allows_adjacent():
    """gap_days=0 allows adjacent dates."""
    split = SplitSpec(
        split_id="adjacent",
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 12, 31),
        val_start=datetime(2021, 1, 1),  # Adjacent
        val_end=datetime(2021, 6, 30),
        gap_days=0,
    )
    assert split.val_start >= split.train_end
```

**File**: `modeling/tests/test_fitted_transform_oos_enforcement.py` (NEW)

```python
def test_transform_requires_apply_start_time():
    """transform() must require apply_start_time (no silent bypass)."""
    scaler = CrossSectionalScaler()
    X_train = np.random.randn(100, 5)
    fit_window = FitWindow(datetime(2020, 1, 1), datetime(2020, 12, 31))
    scaler.fit(X_train, fit_window)
    
    X_val = np.random.randn(50, 5)
    with pytest.raises(TypeError, match="requires apply_start_time"):
        scaler.transform(X_val, apply_start_time=None)

def test_fit_transform_for_in_sample():
    """fit_transform() is the correct API for in-sample use."""
    scaler = CrossSectionalScaler()
    X_train = np.random.randn(100, 5)
    fit_window = FitWindow(datetime(2020, 1, 1), datetime(2020, 12, 31))
    
    # Correct: fit_transform for in-sample
    X_transformed = scaler.fit_transform(X_train, fit_window)
    assert X_transformed.shape == X_train.shape
```

### 5. Update Documentation

**File**: `factor_engine/modeling/AUTHORITY.md` (NEW)

```markdown
# Modeling Package Authority

## Single Source of Truth

`factor_engine/modeling/` is the AUTHORITATIVE modeling package for:
- Model training (trainer.py)
- Walk-forward splitting (walk_forward.py)
- Label-interval purge (purge_overlap, purge_before_boundary)
- Embargo enforcement (apply_embargo)
- Artifact lifecycle (artifact.py, predictor.py)
- Temporal contracts (EmbargoSpec, ApplicationWindow, LabelContract)

## Standalone `modeling/` Package

The `modeling/` package at `/home/shw/quant_projects/modeling/` is:
- **Scope**: Minimal contracts for factor_preprocess adapters
- **Not Responsible For**: Model training, enforcement, walk-forward
- **Migration Path**: Use factor_engine.modeling for all production code

## Migration Guide

### If you're using standalone `modeling/`:

**Before** (standalone):
```python
from modeling.contracts import SplitSpec
split = SplitSpec(..., gap_days=5)  # Defined but NOT enforced
```

**After** (factor_engine):
```python
from modeling.walk_forward import WalkForwardSpec, make_walk_forward_splits
spec = WalkForwardSpec(..., gap_days=5)  # Enforced at split creation
folds = make_walk_forward_splits(ds, spec)
```

### If you're using fitted transforms:

**Before** (standalone, bypass possible):
```python
scaler.transform(X_val, apply_start_time=None)  # Silent bypass
```

**After** (fail-closed):
```python
# OOS use: MUST provide application period
scaler.transform(X_val, apply_start_time=val_start)  # Enforced

# In-sample use: explicit API
scaler.fit_transform(X_train, fit_window)  # No bypass possible
```
```

---

## Test Coverage Requirements

### Gap Days Enforcement
- [ ] `test_gap_days_enforced_in_split_spec()` — Rejects insufficient gap
- [ ] `test_gap_days_with_calendar_dates()` — Calendar-day arithmetic correct
- [ ] `test_gap_days_zero_backward_compat()` — gap_days=0 works

### Purge Overlap (Already Complete ✓)
- [x] `test_purge_overlap_missing_session_gap()` — Handles calendar gaps correctly
- [x] `test_purge_overlap_label_matures_inside_validation()` — Drops overlapping rows
- [x] `test_purge_overlap_horizon_one_with_no_actual_overlap()` — Conservatively purges

### OOS Prediction Enforcement
- [ ] `test_transform_requires_apply_start_time()` — No silent None bypass
- [ ] `test_predict_oos_requires_window()` — predict_oos fails without ApplicationWindow
- [ ] `test_cannot_apply_oos_transform_to_training_period()` — Temporal boundary enforced

### Integration Tests
- [ ] `test_factor_engine_is_authority()` — Import factor_engine.modeling works
- [ ] `test_standalone_minimal_contracts_only()` — Standalone has no trainer/walk_forward
- [ ] `test_gap_days_purge_embargo_integration()` — All three mechanisms work together

---

## Implementation Order

1. **Authority declaration** (non-breaking) — Update __init__.py docstrings
2. **gap_days enforcement** (breaking for invalid code) — Fix SplitSpec.__post_init__
3. **CrossSectionalScaler.transform()** (breaking) — Require apply_start_time
4. **Tests** — Add 6+ enforcement tests
5. **Documentation** — Create AUTHORITY.md migration guide
6. **Commit** — Single commit explaining consolidation

---

## Risk Assessment

**Breaking Changes**:
- SplitSpec with `gap_days > 0` but insufficient actual gap will now raise SplitError
- CrossSectionalScaler.transform(X, apply_start_time=None) will raise TypeError

**Mitigation**:
- Both are **fail-closed improvements** (silent bugs → loud errors)
- Standalone package is NOT used in production (factor_engine is)
- Tests will catch any internal misuse

**Timeline**: 2-3 hours (changes are small, tests are critical)

---

## Deliverables

1. ✓ Analysis complete (this document)
2. □ Authority declarations in both packages
3. □ gap_days enforcement in standalone SplitSpec
4. □ CrossSectionalScaler.transform() enforcement
5. □ 6+ tests proving enforcement (not just parameter existence)
6. □ AUTHORITY.md migration guide
7. □ Commit with consolidation rationale

---

## Hard Gates (Target)

```python
MODELING_SINGLE_AUTHORITY = PASS  # Explicit declaration in both packages
GAP_DAYS_ACTUALLY_ENFORCED = PASS  # SplitSpec.__post_init__ checks calendar gap
PURGE_OVERLAP_ACTUALLY_ENFORCED = PASS  # Already wired correctly ✓
OOS_PREDICTION_NO_BYPASS = PASS  # transform() requires apply_start_time
FITTED_TRANSFORM_REQUIRES_APPLICATION_PERIOD = PASS  # No silent None allowed
```

All gates must be **runtime-enforced** (not just documented).
