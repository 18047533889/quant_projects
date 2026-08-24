# Q-P0-001/002 Audit Summary

**Worker:** Worker-Q  
**Date:** 2026-08-14  
**Status:** ✗ NOT PRODUCTION READY

## Tasks Completed

### Q-P0-001: _PHASE1_NATIVE_OPS Manual Authority Investigation
✅ **COMPLETED**
- Located manual list: `backend/q_backend/q_capability.py:37-103`
- Contains 110 operators declared as "native"
- Serves as capability authority for `QBackendCapability.supports_native()`
- **Problem identified:** Can diverge from actual implementation state

### Q-P0-002: Generate Q_NATIVE_WITHOUT_LOWERING
✅ **COMPLETED**
```
Q_NATIVE_WITHOUT_LOWERING = DeclaredNative - LoweringExists
                          = 110 - 80 = 30 operators
```

**Critical Finding:** 30 operators declared native WITHOUT lowering implementation

## Key Findings

### 1. Manual Authority Violation
- Manual `_PHASE1_NATIVE_OPS` set serves as capability authority
- No automatic verification against actual implementation
- Violates principle: production capability must be evidence-based

### 2. Critical Gap: 30 Missing Lowerings
The following 30 operators are declared native but have NO lowering in `QCompiler._operator_map`:

```
bfill, cs_clip, cs_percentile_rank, cs_quantile, cs_winsorize,
group_count, group_max, group_mean, group_median, group_min,
group_std, group_sum, replace, resample, time_bucket, true_range,
ts_argmax_age, ts_argmin_age, ts_decay_exp, ts_decay_linear,
ts_distance_to_high, ts_distance_to_low, ts_max_drawdown, ts_moment,
ts_new_high, ts_new_low, ts_percentile, ts_quantile, ts_sum_decay, vwap
```

**Impact:**
- These operators claim to be native but silently fall back to Pandas
- Performance expectations violated
- No compile-time detection

### 3. No Parity Testing
- Polars/DuckDB have 6-stage evidence certification pipeline
- Q backend has NO parity tests
- No `evidence/q_verified.json`
- Cannot verify correctness

## Deliverables

### 1. Evidence Framework
**File:** `backend/q_backend/q_capability_evidence.py` (321 lines)

Features:
- 5-pass evidence model (DeclaredNative, LoweringExists, CompilePass, RuntimePass, ParityPass)
- Auto-derived capability from actual implementation
- `get_q_native_without_lowering()` function
- Hard gates for production admission

```python
def get_q_native_without_lowering() -> frozenset[str]:
    """Q-P0-002: Must be empty for production."""
    return get_declared_native_ops() - get_lowering_exists_ops()
```

### 2. Test Suite
**File:** `tests/q_backend/test_q_capability_evidence.py` (286 lines)

Coverage:
- Evidence framework structure tests
- Gap detection tests
- Hard gate verification
- Integration tests

### 3. Verification Script
**File:** `scripts/verify_q_capability_evidence.py` (executable)

Usage:
```bash
# Quick gate check
python scripts/verify_q_capability_evidence.py --gates-only

# Full report
python scripts/verify_q_capability_evidence.py

# Detailed breakdown
python scripts/verify_q_capability_evidence.py --detailed

# JSON output
python scripts/verify_q_capability_evidence.py --json
```

### 4. Findings Document
**File:** `Q_P0_001_002_FINDINGS.md` (comprehensive audit report)

## Hard Gates

### Gate 1: Q_NATIVE_WITHOUT_LOWERING
**Status:** ✗ FAIL  
**Current:** 30 operators  
**Required:** 0 operators  
**Blocker:** YES

### Gate 2: Q_MANUAL_AUTHORITY_REMOVED
**Status:** ✗ FAIL  
**Current:** `_PHASE1_NATIVE_OPS` still exists  
**Required:** Evidence-based authority only  
**Blocker:** YES

## Comparison: Q vs Polars/DuckDB

| Aspect | Polars/DuckDB | Q Backend |
|--------|---------------|-----------|
| Evidence stages | 6 passes | 0 (none) |
| Evidence storage | `evidence/primitive_verified.json` | ❌ None |
| Parity tests | ✓ Comprehensive | ❌ None |
| Authority | Evidence-based | ❌ Manual list |
| Auto-verification | ✓ Yes | ❌ No |
| Gap detection | ✓ Automatic | ❌ Manual audit needed |

## Recommendations

### Immediate Actions (Phase 2)
1. **Remove 30 unimplemented operators from declared native set**
   - Reduces declared scope from 110 → 80
   - Makes `Q_NATIVE_WITHOUT_LOWERING == 0`
   - Honest capability reporting

2. **Replace manual authority with evidence-based**
   - Update `q_capability.py` to use `get_q_production_safe_ops()`
   - Remove `_PHASE1_NATIVE_OPS` manual list
   - Add startup assertion

3. **Add hard gate to CI/CD**
   ```bash
   python scripts/verify_q_capability_evidence.py --gates-only
   ```

### Future Work (Phase 3-5)
1. Implement missing lowerings (if needed)
2. Create Q parity test suite
3. Add evidence JSON storage
4. Integrate with capability registry

## Production Readiness

```
Q Backend Production Status: ✗ NOT READY

Blocking Issues:
✗ 30 operators declared without implementation
✗ Manual authority can diverge from reality
✗ No parity testing infrastructure
✗ No evidence verification

Required:
✓ Evidence framework implemented (this audit)
✗ Q_NATIVE_WITHOUT_LOWERING == 0
✗ Manual authority removed
✗ Parity tests created
```

## Files Created

1. `backend/q_backend/q_capability_evidence.py` - Evidence framework
2. `tests/q_backend/test_q_capability_evidence.py` - Test suite
3. `scripts/verify_q_capability_evidence.py` - Verification tool
4. `Q_P0_001_002_FINDINGS.md` - Detailed audit report
5. `Q_P0_001_002_SUMMARY.md` - This summary

## Verification

Run verification:
```bash
cd /home/shw/quant_projects/factor_engine
python3 scripts/verify_q_capability_evidence.py
```

Expected output:
```
Q_NATIVE_WITHOUT_LOWERING: 30 operators
Production Ready: NO
```

## Next Steps

**Assigned to:** Q backend maintainer  
**Priority:** P0 (blocks production)  
**Estimated effort:** 16-26 hours

1. Review findings document
2. Decide: implement missing lowerings OR reduce scope
3. Apply Phase 2 recommendations
4. Re-run verification until gates pass
5. Add to CI/CD pipeline

---

**Audit Complete**  
Evidence-based capability framework delivered and verified.
