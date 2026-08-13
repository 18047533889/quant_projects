# R47 Operators Numerical Correctness Spot Check Report

**Date:** 2026-08-13  
**Test File:** `tests/spot_check_numerical_correctness.py`  
**Total Test Cases:** 32  
**Status:** ✅ ALL PASSING

---

## Summary

Comprehensive numerical correctness verification for R47 operators with **hand-calculable test cases** designed to catch real bugs, not just crashes.

### Operators Tested: 12

#### Fiscal Operators (7)
1. `fiscal_acceleration` - Second difference with fiscal gap semantics
2. `fiscal_pct_change` - Percentage change with absolute value denominator
3. `fiscal_rolling_std` - Rolling standard deviation over fiscal events
4. `fiscal_accrual_quality` - Accrual quality ratio
5. `fiscal_direction_consistency` - Sign consistency metric
6. `fiscal_standardized_surprise` - Standardized earnings surprise
7. `years_since_date` - **🐛 BUG FIXED**: Period ordinal base offset error

#### Kalman Filter Variants (1)
8. `ts_alpha_beta_filter` - Position + velocity tracking filter

#### Cross-Sectional Operators (2)
9. `cs_factor_bucket_return` - Factor bucketing with return aggregation
10. `cs_empirical_bayes_shrinkage` - Precision-weighted shrinkage

#### Time-Semantic Operators (2)
11. `financial_snapshot_lag` - Event-semantic lag (skips NaN)
12. `same_calendar_day_mean` - Day-of-week grouped mean

---

## Bug Found and Fixed

### 🐛 Critical Bug: `years_since_date` Period Ordinal Base Offset

**File:** `cleaned_operators/fundamental/fiscal_batch3.py:289`

**Root Cause:**  
The function assumed `period_ordinal` returns 0 for "2000Q1", but it actually returns 8000. This caused the year calculation to be off by 2000 years.

**Impact:**  
- For "2024Q1" (ordinal 8096), calculated year was 4024 instead of 2024
- All year calculations were wrong by exactly 2000 years

**Fix Applied:**
```python
# Before (WRONG):
year_offset = ordinal // ppy
current_year = base_year + year_offset  # 2000 + 2024 = 4024 ❌

# After (CORRECT):
base_ordinal = period_ordinal(f"{base_year}Q1")  # 8000
adjusted_ordinal = ordinal - base_ordinal  # 8096 - 8000 = 96
year_offset = adjusted_ordinal // ppy  # 96 // 4 = 24
current_year = base_year + year_offset  # 2000 + 24 = 2024 ✅
```

**Verification:**
- Test case: event_date="2020-01-01", period_id="2024Q1"
- Expected: ~4.0 years
- Before fix: 2004 years ❌
- After fix: 4.0 years ✅

---

## Test Coverage Details

### Hand-Calculable Test Cases

Each operator has 2-3 tests with **manually verified expected outputs**:

#### Example: `fiscal_acceleration`
```python
# Input: quarterly earnings [100, 110, 125, 145, 150]
# acceleration_t = value_t - 2*value_{t-1} + value_{t-2}
# t=2: 125 - 2*110 + 100 = 5 ✓
# t=3: 145 - 2*125 + 110 = 5 ✓
# t=4: 150 - 2*145 + 125 = -15 ✓
```

#### Example: `cs_factor_bucket_return`
```python
# 12 stocks, 2 buckets
# Bucket 0 (ranks 1-6): returns [0.01..0.06], mean = 0.035 ✓
# Bucket 1 (ranks 7-12): returns [0.07..0.12], mean = 0.095 ✓
```

### Edge Cases Tested

1. **All NaN input** → All NaN output (fail-closed)
2. **Single valid value** → NaN (insufficient data)
3. **Extreme values** (±1e10) → No overflow/underflow
4. **Sign changes** → Absolute value guards working
5. **Near-zero denominators** → Epsilon guards prevent division by zero
6. **Missing data patterns** → Proper NaN propagation
7. **Insufficient history** → Fail-closed behavior

### Causality Verification

**Test:** `test_fiscal_operators_no_future_leakage`

Verifies that fiscal operators respect TRUE_GAP semantics:
- Uses 6 fiscal periods
- Changes last period value
- **Asserts:** First 5 periods unchanged (no future leakage) ✓

---

## Test Execution Results

```bash
$ python3 -m pytest tests/spot_check_numerical_correctness.py -v

===================== 32 passed, 38 warnings in 1.64s ======================

Test Breakdown:
- Fiscal operators: 14 tests
- Edge cases & robustness: 6 tests
- Causality: 1 test
- Kalman filters: 3 tests
- Cross-sectional: 3 tests
- Time-semantic: 3 tests
- Metadata: 2 tests
```

---

## Key Findings

### ✅ Strengths

1. **Epsilon guards work correctly** - Division by zero properly handled
2. **Fail-closed semantics** - Insufficient data returns NaN (not wrong values)
3. **Numerical stability** - No overflow/underflow on extreme values
4. **Causality respected** - No future leakage in time-series operators
5. **Cross-sectional breadth checks** - Minimum 10 stocks enforced

### 🐛 Bugs Found: 1

1. **`years_since_date`** - Period ordinal base offset error (FIXED)

### 📊 Coverage Gaps

These R47 operator families need numerical correctness tests:

1. **Filter operators** (13 remaining):
   - `ts_bessel_lowpass_causal`, `ts_fir_lowpass_causal`
   - `ts_h_infinity_level_filter`, `ts_adaptive_noise_kalman`, `ts_student_t_kalman_filter`
   - `ts_mcginley_dynamic`, `ts_vidya`, `ts_one_euro_filter`
   - `ts_nlms_filter`, `ts_rls_filter`
   - `ts_ssa_denoise_trailing`, `ts_wavelet_shrinkage_trailing`
   - `ts_total_variation_filter_trailing`, `ts_l1_trend_filter_trailing`

2. **Technical indicators** (9 remaining):
   - `HMA`, `QQE`, `RSX`, `ALMA`, `CoppockCurve`
   - `ElderRay`, `FisherTransform`
   - `turnover_chip_age_cost_surface`, `turnover_chip_overhang_surface`

3. **Panel operators** (remaining):
   - `price_delay_score`, `panel_async_beta_ex_self`
   - `panel_factor_pocket_strength`, `pastor_stambaugh_beta`

4. **Fiscal batch 3** (4 remaining):
   - `fiscal_capital_stock`, `fiscal_perpetual_inventory`
   - `cash_flow_lifecycle_stage`, `laborforce_efficiency`

**Estimated total R47 operators:** ~47  
**Currently tested:** 12 (26%)

---

## Recommendations

### Immediate Actions

1. ✅ **DONE** - Fix `years_since_date` bug
2. ✅ **DONE** - Verify fix with comprehensive tests
3. ⏭️ **NEXT** - Expand coverage to filter operators (high complexity)
4. ⏭️ **NEXT** - Add technical indicator numerical tests

### Test Strategy

For remaining operators:

1. **Hand-calculable inputs** - Use small datasets with known outputs
2. **Edge case focus** - Zero, NaN, single value, extreme values
3. **Mathematical properties** - Test invariants (e.g., Kalman covariance > 0)
4. **Regression protection** - Lock in correct behavior before refactoring

### Quality Gates

- [ ] All R47 operators have ≥2 numerical correctness tests
- [x] All existing tests passing (32/32)
- [x] Critical bugs found and fixed (1/1)
- [ ] Coverage ≥80% of R47 operators

---

## Usage

Run all numerical correctness tests:
```bash
pytest tests/spot_check_numerical_correctness.py -v
```

Run specific category:
```bash
pytest tests/spot_check_numerical_correctness.py::TestFiscalBatch1 -v
pytest tests/spot_check_numerical_correctness.py::TestKalmanFilters -v
```

Run with detailed output:
```bash
pytest tests/spot_check_numerical_correctness.py -vv --tb=short
```

---

## Appendix: Test Categories

### TestFiscalBatch1 (9 tests)
- Hand-computed values with fiscal event semantics
- Sign change handling
- Zero denominator guards
- Insufficient periods handling

### TestFiscalBatch2 (2 tests)
- Standardized surprise with zero std handling
- Hand-computed surprise values

### TestFiscalBatch3 (3 tests)
- Years since date calculation
- Leap year handling
- Future date handling

### TestEdgeCasesAndRobustness (6 tests)
- All NaN inputs
- Single valid values
- Extreme values (±1e10)
- Mixed sign robustness

### TestCausality (1 test)
- Future leakage prevention
- TRUE_GAP semantics verification

### TestKalmanFilters (3 tests)
- Position tracking accuracy
- Velocity estimation
- Missing data handling

### TestCrossSectionalOperators (3 tests)
- Bucketing correctness
- Shrinkage precision weighting
- Ascending/descending sort

### TestTimeSemanticOperators (3 tests)
- Event-semantic lag (NaN skipping)
- Insufficient history handling
- Day-of-week grouping

### Metadata Tests (2 tests)
- Operator counting
- Coverage tracking
