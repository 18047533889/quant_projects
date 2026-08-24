# Regression Models Window Maturity Audit

## Executive Summary

**Status**: MULTIPLE CRITICAL DEFECTS FOUND  
**File**: `cleaned_operators/regression_models.py`  
**Impact**: Predictive regression operators and AR coefficient use fewer observations than declared window W

## Concrete Defects

### DEFECT 1: Predictive Regression Underfitting (CRITICAL)
**Operators**: `ts_huber_regression_predictive_resid`, `ts_ridge_regression_predictive_resid`  
**Lines**: 290-302, 375-390

**Current behavior**:
- Declared window W should fit on [t-W, t-1] = W prior observations
- Current: `segment = [start:row+1]` extracts W rows including current row
- With `predictive=True`, `_regression_resid` uses `y[:-1], x[:-1]` → **W-1 observations**
- **Actual fit window: [t-W+1, t-1] = W-1 observations, NOT W**

**Example**: window=20, row=99
- start = 80, segment = [80:100] = 20 rows (indices 80-99)
- predictive fit uses [:-1] → 19 rows (indices 80-98)
- Expected: 20 prior observations [79-98], Actual: 19 observations [80-98]

**Root cause**: Window extraction doesn't account for the need to drop current row in predictive mode.

**Fix**: Extract W+1 rows for predictive mode: `start = max(0, row - w)` instead of `max(0, row - w + 1)`

---

### DEFECT 2: AR Coefficient Window Shrinkage (CRITICAL)
**Operator**: `ts_ar_coefficient`  
**Lines**: 486-507

**Current behavior**:
- Declared window W should compute AR(lag) over W pairs
- Current: `segment = [start:row+1]` extracts W rows
- Then `current = segment[lg:]`, `lagged = segment[:-lg]` → **W-lag pairs**
- **Actual: W-lag pairs, NOT W pairs**

**Example**: window=20, lag=1, row=99
- start = 80, segment = [80:100] = 20 rows
- current = [81-99] = 19, lagged = [80-98] = 19 → 19 pairs
- Expected: 20 pairs, Actual: 19 pairs

**Root cause**: Segment doesn't include enough history to form W pairs after lagging.

**Fix**: Extract W+lag rows: `start = max(0, row - w - lg + 1)`

---

## Poison Test Specifications

### Test 1: Predictive regression uses exactly W prior observations
- Setup: W=10, perturb row at t-11 (outside declared window)
- Expected: No change in output at t
- Current: **WILL FAIL** (t-11 is outside actual W-1 window, so no change even with bug)
- Better test: Verify coefficient stability vs window size (W vs W+1 should differ)

### Test 2: Predictive fit excludes current observation
- Setup: Perturb current row x[t], y[t] to extreme values
- Expected: No change in fitted β (only residual changes)
- Current: Should PASS (this part is correct)

### Test 3: AR coefficient uses exactly W pairs
- Setup: W=10, lag=1, perturb row at t-11
- Expected: Change in AR coefficient at t (t-11 should be in window)
- Current: **WILL FAIL** (t-11 not in actual W-1 window)

### Test 4: Full window vs actual window comparison
- Setup: Manually compute with W observations, compare to operator with window=W
- Expected: Match
- Current: **WILL FAIL** (operator uses W-1)

---

## Remediation Plan

1. Fix predictive regression window extraction (2 operators)
2. Fix AR coefficient window extraction (1 operator)
3. Add 4 poison tests covering window boundaries
4. Verify compile + serial test pass
5. Commit with evidence
