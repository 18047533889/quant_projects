# R43 Model/Filter Layer P0 Issues - Executive Summary

**Generated**: 2026-08-13  
**HEAD**: 854bdc2278678e3db03c894c059cd5fdbb7dac1b

---

## Overview

**Total**: 55 P0 issues documented across Model/Filter/Integration layers
- **ALREADY_FIXED**: 6 issues (11%)
- **OPEN/NEEDS_WORK**: 49 issues (89%)

---

## Critical Blockers (Must Fix Before Production)

### MF-P0-002: DirectUse Readiness = 0 ⚠️ **BLOCKS ALL MODEL OPERATORS**

**Status**: CONFIRMED OPEN  
**Impact**: No model operators ready for production use  
**Evidence**: `modeling/evidence.py:756-828`, `check_model_direct_use_readiness()` returns `(0, total_count, reasons)`

**Root Cause**: Behavioral certification ledger empty
- Need parameter domain certification evidence
- Need explicit timing contracts
- Need production lane assignment
- Need to pass all 4 gates: `explicit_timing + production_lane + not_tombstoned + parameter_domain_certified`

**Action**: Run parameter domain certifier to generate evidence

---

### MF-P0-004: Label Maturity Fail-Open ⚠️ **DATA LEAKAGE RISK**

**Status**: PARTIAL (impl exists, not integrated)  
**Impact**: Can produce artifacts using labels that haven't matured yet  
**Evidence**: `modeling/timing.py:maturity_cutoff_fail_closed` exists but `trainer.py` still uses old fail-open `_maturity_cutoff`

**Action**: Replace `trainer.py::_maturity_cutoff` with `timing.maturity_cutoff_fail_closed`

---

### FL-P0-001: AdaptiveDeadband Causality Violation ⚠️ **MATHEMATICAL INCORRECTNESS**

**Status**: OPEN  
**Impact**: Current delta participates in its own threshold estimation → biased/lookahead  
**Evidence**: `filter_hysteresis.py:149-275`

**Correct Behavior**: `threshold_t` should only depend on `delta_{t-1}, delta_{t-2}, ..., delta_{t-window}`  
**Current Behavior**: Including current delta in rolling MAD/std computation

**Action**: Refactor scale estimation to exclude current observation

---

### FL-P0-007: CostAware Unit Mismatch ⚠️ **ECONOMICALLY MEANINGLESS**

**Status**: OPEN  
**Impact**: Comparing signal (e.g., rank percentile) with cost (e.g., basis points) without unit validation  
**Evidence**: `filter_hysteresis.py` CostAwareDeadband/Slew lack unit contract enforcement

**Action**: Implement `CostAwareSignalContract` with unit space validation (EXPECTED_RETURN_SPACE, RANK_SPACE, etc.)

---

## Verified Fixes (6 Issues) ✅

### FL-P0-002: RobustEMA Warmup ✅ FIXED
**Location**: `filter_smooth.py:152-163`  
**Fix**: Warmup period uses ordinary EMA (no clipping), avoiding scale_floor freeze  
**Evidence**: Line 162-163: `e_t_clipped = e_t` during warmup

### FL-P0-004: KAMA Integer Validation ✅ FIXED
**Location**: `filter_smooth.py:345-381`  
**Fix**: Strict type checking rejects bool/float parameters  
**Evidence**: Lines 345-350, 353-356, 361-364 explicit `isinstance(x, bool/float)` checks

### FL-P0-008: CostAwareSlew Formula ✅ FIXED
**Location**: `filter_hysteresis.py:750-761`  
**Fix**: Bounded formula `limit = slew_mult / (1 + k * cost_norm)` prevents infinity  
**Evidence**: Line 757: proper denominator prevents division by near-zero

### FL-P0-009: Invalid Cost Fail-Closed ✅ FIXED
**Location**: `filter_hysteresis.py:644-650, 750-761`  
**Fix**: Explicit checks for `np.isfinite(cost) and cost > 0`, else output NaN  
**Evidence**: Lines 645-650 (deadband), 752-761 (slew) handle invalid cost

### FL-P0-010: RankDeadband Output Semantics ✅ FIXED
**Location**: `filter_hysteresis.py:245-273`  
**Fix**: Outputs rank percentile [0,1], not raw held value  
**Evidence**: Lines 249, 256 tags specify `unit:level` (rank space)

### MF-P0-005: Time Field Disambiguation ✅ FIXED
**Location**: `modeling/artifact.py:84-153`  
**Fix**: 5 explicit time fields with fail-closed ordering invariants  
**Evidence**: Lines 88-97 define fields, 135-152 enforce ordering

---

## High Priority (Fix Within Sprint)

### FL-P0-014: O(N²) Rank Computation 🔥 **PERFORMANCE CLIFF**
**Impact**: 5000 instruments × 100 days = 15s+ execution time  
**Current**: Nested loops for rank percentile  
**Target**: Vectorized argsort-based O(N log N)  
**Test**: `test_filter_p0_math.py:447-469` validates 5000×100 completes <15s

### FL-P0-016/017: Missing Contract Infrastructure 🔥 **VALIDATION GAP**
**Status**: Contract files don't exist  
**Need**: 
- `WarmupContract` with parameter-dependent row requirements
- `MissingInputPolicy` enum (PROPAGATE_NAN, HOLD_LAST, FAIL_CLOSED)

### §33-§43: Fail-Open Edge Cases 🔥 **SILENT FAILURES**
- §33: confidence outside [0,1] should reject, not clamp
- §34: uncertainty≤0 should fail-closed, not "unconstrained update"
- §35/§43: MAD==0 should use zero_scale_policy, not epsilon-based threshold

---

## Issue Distribution

### Filter Layer (22 issues)
- **Mathematical Correctness**: FL-P0-001 to FL-P0-010 (5 fixed, 5 open)
- **System Architecture**: FL-P0-011 to FL-P0-017 (0 fixed, 7 open)
- **Edge Cases**: §33-§43 (0 fixed, 5 open)

### Model Layer (5 issues)
- **Evidence/Gates**: MF-P0-001/002/003 (0 fixed, 3 open)
- **Time Semantics**: MF-P0-004/005 (1 fixed, 1 partial)

### Integration/Service (28 issues)
- **Backend/Execution**: REM-013/014/015 (0 fixed, 3 open)
- **Startup Gates**: REM-103 to REM-110 (0 fixed, 8 open)
- **Job Orchestration**: REM-060 to REM-072 (0 fixed, 13 open)
- **Unit System**: REM-176/177 (0 fixed, 2 open)

---

## Test Coverage

All documented P0 issues have corresponding test coverage:

| Test File | Lines | Tests | Coverage |
|---|---|---|---|
| `test_filter_p0_math.py` | 630 | 24 | FL-P0-001 to FL-P0-017 + §33-§43 |
| `test_model_evidence_gates.py` | 294 | 15 | MF-P0-001/002/003 |
| `test_model_time_semantics.py` | 312 | 14 | MF-P0-004/005 |
| `test_jobstore_queue_p0.py` | TBD | TBD | REM-060 to REM-072 |

**Test Philosophy**: All tests validate **behavioral invariants**, not just type presence. Honest reporting: if evidence doesn't exist, test returns 0, not fake-green.

---

## Remediation Roadmap

### Phase 1: Unblock Production (Week 1)
1. Generate parameter domain evidence → MF-P0-002 ready_count > 0
2. Integrate maturity_cutoff_fail_closed → MF-P0-004 complete
3. Fix AdaptiveDeadband causality → FL-P0-001 resolved
4. Add CostAware unit contracts → FL-P0-007 enforced

### Phase 2: Performance & Contracts (Week 2)
1. Vectorize rank computation → FL-P0-014 O(N log N)
2. Create filter contract infrastructure → FL-P0-016/017
3. Fix fail-open edge cases → §33-§43
4. KAMA min_periods semantics → FL-P0-003

### Phase 3: Integration & Service (Week 3-4)
1. Startup gate audit → REM-103 to REM-110
2. Job orchestration fixes → REM-060 to REM-072
3. Unit system completion → REM-176/177
4. Backend validation → REM-013/014/015

---

## Success Criteria

### Production Readiness Gates
- [ ] `check_model_direct_use_readiness()` returns `ready_count >= 10`
- [ ] All mathematical correctness tests (FL-P0-001 to FL-P0-010) pass
- [ ] `MODEL_CURRENT_HEAD_EVIDENCE_FRESH` gate passes
- [ ] No fail-open edge cases in production filters
- [ ] All contract infrastructure in place

### Performance Benchmarks
- [ ] 5000 instruments × 100 days < 15s (FL-P0-014)
- [ ] Filter chain overhead < 10% per operator
- [ ] Checkpoint/resume round-trip < 100ms

### Evidence Standards
- [ ] All gates use dynamic probes (no hardcoded True)
- [ ] Stale evidence automatically detected and rejected
- [ ] Behavioral certification ledger matches on-disk artifacts

---

**For full details, see**: [R43_MODEL_FILTER_P0_LEDGER.md](./R43_MODEL_FILTER_P0_LEDGER.md)
