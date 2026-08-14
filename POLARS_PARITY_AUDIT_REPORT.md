# Polars Backend Parity Audit Report

**Generated:** 2026-08-14
**Mission:** Wave1-Agent1-PolarsParity

## Executive Summary

**Total Violations Found:** 3
**High Severity:** 2
**Medium Severity:** 0
**Low Severity:** 1
**Fixes Required:** 2

## Violation Details

### Violation 1: ts_mean,ts_std,ts_sum,ts_min,ts_max,ts_median,ts_var - inf_handling

**Severity:** LOW
**Fix Required:** NO

**Location:** `factor_engine/backend/polars_expr_emitter.py:124`

**Pandas Behavior:**
```
rolling().mean() treats ±Inf as missing (silently excludes)
```

**Polars Behavior:**
```
_sanitize_nan_for_compute(drop_inf=True) explicitly converts Inf to NULL
```

### Violation 2: ts_zscore - zero_std_handling

**Severity:** HIGH
**Fix Required:** YES

**Location:** `factor_engine/cleaned_operators/common/time_series.py:1636`

**Pandas Behavior:**
```
.replace(0, 1) changes std=0 to std=1, giving (x-mean)/1
```

**Polars Behavior:**
```
Correctly checks std==0 and returns 0.0
```

### Violation 3: ts_zscore,cs_zscore - semantic_policy

**Severity:** HIGH
**Fix Required:** YES

**Location:** `factor_engine/backend/numeric_semantics.py:80`

**Pandas Behavior:**
```
Not enforced - uses .replace(0, 1) hack
```

**Polars Behavior:**
```
Correctly implements zero_fill=0.0 policy
```

## Priority Fixes

1. **ts_zscore** (factor_engine/cleaned_operators/common/time_series.py:1636)
   - Issue: zero_std_handling

2. **ts_zscore,cs_zscore** (factor_engine/backend/numeric_semantics.py:80)
   - Issue: semantic_policy
