# Operator ABI Consistency Violations - Final Report
## Wave1-Agent3-OperatorABI
**Date:** 2026-08-14  
**Agent:** Wave1-Agent3-OperatorABI  
**Mission:** Validate operator ABI consistency - param_names/param_types must match function signatures

---

## Executive Summary

**Total Violations Found:** 16 (after fixing validation script bug)  
- **High Severity:** 16 (all parameter mismatches)
- **Medium Severity:** 0
- **Low Severity:** 0

**Files Scanned:** 179 operator files in `cleaned_operators/`  
**Files with Violations:** 6 files  
**Registration Patterns Found:**
- 4 files use `_PARAMS` + `_KERNELS` pattern (all clean, 0 violations)
- 86 files use class-based operators (16 violations found)

---

## Validation Script Bug Fixed

**Initial Report:** 238 violations (INCORRECT)  
**Root Cause:** Script used `ast.walk()` which visits all nodes in tree, causing metadata from one operator class to be matched with methods from a different class in the same file.

**Fix:** Changed from `ast.walk()` to direct iteration of `tree.body`, tracking class context properly. Now correctly matches metadata to methods within the same class.

**Corrected Report:** 16 violations (CORRECT)

---

## Violations by File

| File | Violations | Issue Type |
|------|------------|------------|
| time_semantic_ops.py | 6 | Empty signatures with declared params |
| return_decomp.py | 4 | Missing `price_basis`, wrong param names |
| panel_batch1.py | 3 | Empty signatures with declared params |
| fiscal_event_ops.py | 1 | Phantom `'...'` and `min_count` |
| production_repairs.py | 1 | Extra phantom parameters |
| research_transform.py | 1 | Missing `depth` parameter |

---

## Detailed Violations

### 1. time_semantic_ops.py (6 violations)
All operators have empty function signatures (`def _calculate_series(self, **_)`) but declare parameters in metadata.

- `report_asof`: Declared `['value', 'report_date', 'asof_date', 'max_staleness_days']`, actual `[]`
- `event_window_return_asof`: Declared `['ret', 'event_date', 'window_before', 'window_after']`, actual `[]`
- `financial_snapshot_lag`: Declared `['x', 'period_id', 'fiscal_date', 'lag_periods']`, actual `[]`
- `same_calendar_day_mean`: Declared `['x', 'window']`, actual `[]`
- `same_calendar_month_return`: Declared `['x', 'window']`, actual `[]`
- `same_clock_lag`: Declared `['x', 'clock_time', 'lag_minutes']`, actual `[]`

**Pattern:** These operators use `**kwargs` to accept all params without declaring them in signature.  
**Impact:** param_names validation and documentation are disconnected from actual implementation.

### 2. return_decomp.py (4 violations)
Return decomposition operators missing `price_basis` parameter and use `open` instead of `open_px`.

- `overnight_return`: Missing `['open_px', 'price_basis']`, phantom `['open']`
- `open_close_return`: Missing `['open_px', 'price_basis']`, phantom `['open']`
- `open_to_vwap_return`: Missing `['open_px', 'price_basis']`, phantom `['open']`
- `vwap_to_close_return`: Missing `['price_basis']`

**Pattern:** Parameter naming inconsistency (`open` vs `open_px`) and missing `price_basis` contract parameter.  
**Impact:** DSL will accept `open` but kernel expects `open_px`. Price adjustment policy not validated.

### 3. panel_batch1.py (3 violations)
Panel operators have empty signatures but declare panel parameters.

- `panel_day_night_beta_gap`: Declared 7 params, actual `[]`
- `pastor_stambaugh_beta`: Declared 4 params, actual `[]`
- `price_delay_score`: Declared 4 params, actual `[]`

**Pattern:** Same as time_semantic_ops.py - `**kwargs` implementation.

### 4. fiscal_event_ops.py (1 violation)
- `row_sum_skipna`: Declared `['...', 'min_count']`, actual `[]`

**Issue:** Literal string `'...'` in param_names (invalid identifier).

### 5. production_repairs.py (1 violation)
- `ts_regression_slope`: Declared extra params `['lag', 'retval', 'min_periods', 'add_intercept']` not in signature

**Pattern:** Metadata not updated after function signature simplified.

### 6. research_transform.py (1 violation)
- `ts_signature_mahalanobis_anomaly`: Missing `depth` parameter in declaration

**Pattern:** Function signature has `depth` param but metadata doesn't declare it.

---

## Root Cause Analysis

### Type 1: Empty Signature with **kwargs (12 violations)
Operators that accept all parameters through `**kwargs` without declaring them:
```python
def _calculate_series(self, **kwargs):
    # Extract params from kwargs inside
```

**Why it happens:** Flexible implementation pattern, but breaks param_names contract.  
**Fix:** Declare parameters explicitly in signature even if using `**kwargs` for extras.

### Type 2: Parameter Name Mismatch (4 violations)
Declared param names don't match actual parameter names (`open` vs `open_px`).

**Why it happens:** Refactoring changed param names in function but not in metadata.  
**Fix:** Update metadata param_names to match function signature exactly.

### Type 3: Missing Parameters (1 violation)
Function signature has more params than declared in metadata.

**Why it happens:** Function evolved but metadata wasn't updated.  
**Fix:** Add missing params to metadata declaration.

### Type 4: Phantom Parameters (1 violation)
Metadata declares params that don't exist in function signature.

**Why it happens:** Metadata copied from template or not updated after simplification.  
**Fix:** Remove phantom params from metadata.

---

## Impact Assessment

### Low Impact (Safe to defer)
- Empty signature operators with `**kwargs` (12 violations) - still work at runtime
- Research-only operators (research_transform.py)

### Medium Impact (Should fix)
- `ts_regression_slope` (production_repairs.py) - may be in use
- `ts_signature_mahalanobis_anomaly` - missing param not documented

### High Impact (Fix priority)
- `return_decomp.py` (4 violations) - fundamental return calculations
  - Wrong param name (`open` vs `open_px`) will cause runtime errors
  - Missing `price_basis` contract breaks price adjustment validation

---

## Remediation Plan

### Immediate (High Priority - 4 operators)
Fix return_decomp.py violations - these are fundamental return calculations:
1. Update metadata: `'open'` → `'open_px'`
2. Add missing `'price_basis'` parameter to all declarations

### Short-term (Medium Priority - 2 operators)
1. Fix `ts_regression_slope` - remove phantom params
2. Fix `ts_signature_mahalanobis_anomaly` - add missing `depth`

### Long-term (Low Priority - 10 operators)
Fix empty signature operators by declaring params explicitly:
- time_semantic_ops.py (6 operators)
- panel_batch1.py (3 operators)  
- fiscal_event_ops.py (1 operator)

---

## Validation Script

**Location:** `scripts/validate_operator_abi.py`

**Key Features:**
- Class-aware AST parsing (fixes initial bug)
- Handles both `_PARAMS`/`_KERNELS` and class-based patterns
- Reports missing, phantom, and order-mismatch violations
- Memory-efficient (< 15 GiB)

**Usage:**
```bash
ulimit -v 15728640 && python3 scripts/validate_operator_abi.py
```

**Outputs:**
- Console: Summary with severity breakdown
- File: `audit_reports/abi_violations_corrected_2026-08-14.txt`

---

## CI Integration Recommendation

Add to pre-commit hooks:
```yaml
- repo: local
  hooks:
    - id: operator-abi-check
      name: Validate Operator ABI Consistency
      entry: python3 scripts/validate_operator_abi.py
      language: system
      files: ^factor_engine/cleaned_operators/.*\.py$
      pass_filenames: false
```

---

## Deliverables

- [x] Validation script: `scripts/validate_operator_abi.py` (bug fixed)
- [x] Full violations report: `audit_reports/abi_violations_corrected_2026-08-14.txt`
- [x] Final manifest: `audit_reports/ABI_VIOLATIONS_FINAL_MANIFEST_2026-08-14.md`
- [x] Identified 16 real violations across 6 files
- [x] Prioritized violations by impact
- [ ] Fixes (deferred - need user decision on approach)

---

## Constraints Observed

✓ NO git checkout/restore/stash/clean  
✓ NO batch AST/regex rewrites  
✓ NO editing operator_catalog.py or operator_policy.py  
✓ Memory limit: 15 GiB (ulimit -v 15728640)  
✓ Read-only audit completed, surgical fixes identified

---

**Status:** Validation complete, ready for remediation phase.
