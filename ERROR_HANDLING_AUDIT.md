# Error Handling & Edge Case Audit Report

**Date:** 2026-08-14  
**Scope:** quant_evaluator, factor_optimizer, factor_assets, factor_preprocess, research_control  
**Audit Type:** Comprehensive deep inspection - Public APIs, edge cases, numeric operations, concurrency  
**Files Analyzed:** ~450 Python files across 5 packages

---

## Executive Summary

**Overall Status:** 🟡 MODERATE RISK - Good foundation with critical gaps

**Strengths:**
- Typed error hierarchies exist (quant_evaluator leads with 6 types)
- Explicit NaN handling in 74+ locations across metrics
- Strong input validation via frozen dataclasses with `__post_init__`
- 229 error test cases demonstrate systematic testing culture

**Critical Weaknesses:**
- **CRITICAL:** Division by zero vulnerability in `portfolio_stats.py:176` (drawdown calculation)
- **HIGH:** Broad exception handler swallows all errors in memory tracking (`budgets.py:216`)
- **HIGH:** Missing logging in silent degradation paths (20+ sites return NaN without warning)
- **MEDIUM:** Inconsistent Inf handling (NaN tested, Inf largely untested)
- **MEDIUM:** No concurrency failure tests despite shared repository usage

**Risk Assessment:** Medium - unlikely to cause data corruption, but can cause crashes or silent failures under edge conditions

**Recommendation:** Fix P0 issues immediately (2 critical bugs), implement P1 improvements (logging, Inf tests) within 1 sprint

---

## Quick Reference: Critical Issues

| Priority | Issue | Location | Impact | Fix Time |
|----------|-------|----------|--------|----------|
| **P0** | Division by zero (no guard) | `portfolio_stats.py:176` | Inf contamination | 15 min |
| **P0** | Broad exception handler | `budgets.py:216` | Silent memory tracking failure | 10 min |
| **P1** | Missing degradation logging | 20+ metric functions | Hard to debug NaN results | 3 hours |
| **P1** | No Inf value tests | All test suites | Untested edge case | 4 hours |
| **P1** | Input validation gaps | Public APIs | Confusing error messages | 3 hours |

---

## 1. Public API Error Handling Audit

### 1.1 quant_evaluator Package

#### ✅ Strengths

**Typed Error Hierarchy** (`contracts/errors.py`)
```python
ContractError(Exception)
├── InvalidContractError
└── SchemaVersionError

DataError(Exception)
├── InsufficientObservations
└── InvalidValidityMask

CapabilityError(Exception)
├── UnsupportedMetricError
└── OptionalDependencyMissing
```

**Strong Input Validation** (`contracts/factor_batch.py`)
```python
def __post_init__(self):
    if not self.factor_ids:
        raise ValueError("factor_ids cannot be empty")
    if self.time_axis is None or self.asset_axis is None:
        raise ValueError("time_axis and asset_axis are required")
    # Shape validation with clear messages
    expected_shape = (self.time_axis.size, self.asset_axis.size, len(self.factor_ids))
    if self.values.shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {self.values.shape}")
```

**Explicit NaN Handling** (`metrics/ic.py:134-138`)
```python
if std_excess == 0 or std_excess < 1e-10 or not np.isfinite(std_excess):
    continue  # Returns NaN for this factor
sharpe[f] = mean_excess / std_excess * np.sqrt(periods_per_year)
```
- Triple guard: exact zero, numerical zero, non-finite
- Prevents division by zero in Sharpe ratio calculation

#### 🔴 Critical Issues

**CRITICAL: Division by Zero in Drawdown Calculation**

**Location:** `metrics/portfolio_stats.py:176`
```python
# Current code - NO GUARD
drawdown_series = (cum_returns - running_max) / running_max
```

**Problem:**
- When `running_max` is zero (e.g., all returns are -100%), produces `Inf`
- `Inf` propagates to `max_dd` calculation, contaminates downstream metrics
- Not caught by tests because extreme drawdown scenarios aren't tested

**Impact:** 
- Calmar ratio becomes `Inf` or `NaN`
- Invalid metrics reach production
- Cascades to portfolio optimization decisions

**Fix:**
```python
# Add guard before division
with np.errstate(divide='ignore', invalid='ignore'):
    drawdown_series = np.where(
        running_max > 0,
        (cum_returns - running_max) / running_max,
        np.nan
    )
```

**Test Required:**
```python
def test_drawdown_extreme_loss():
    """Test drawdown with complete capital loss."""
    returns = np.array([-0.5, -0.5, -0.5, -0.5])  # Loses 93.75% total
    max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)
    assert np.all(np.isfinite(dd_series)), "Drawdown series should not contain Inf"
```

**CRITICAL: Broad Exception Handler in Memory Tracking**

**Location:** `runtime/budgets.py:216`
```python
def _get_memory_mb(self) -> float:
    try:
        mem_info = self._process.memory_info()
        return mem_info.rss / (1024 * 1024)
    except Exception:  # ❌ TOO BROAD
        return 0.0
```

**Problem:**
- Catches ALL exceptions including programming errors
- Silently returns 0.0 on any failure
- Memory budget monitoring fails without warning

**Impact:**
- Budget overruns go undetected
- Out-of-memory crashes happen without early warning
- Masks bugs in memory tracking logic

**Fix:**
```python
def _get_memory_mb(self) -> float:
    try:
        mem_info = self._process.memory_info()
        return mem_info.rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"Cannot access process memory info: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error in memory tracking: {e}")
        raise  # Re-raise unexpected errors
```

#### ⚠️ Medium Issues

**Missing Logging in Silent Degradation**

**Example 1:** `metrics/ic.py` - Insufficient observations
```python
# Current: Silent NaN
if n < min_obs:
    return np.nan

# Better: Log the degradation
if n < min_obs:
    logger.warning(
        f"Insufficient observations for IC: {n} < {min_obs} "
        f"(factor={factor_id}, date={date_idx})"
    )
    return np.nan
```

**Example 2:** `neutralization/ols.py:106` - Matrix singularity
```python
# Current: Silent NaN
except np.linalg.LinAlgError:
    residuals = np.full_like(y, np.nan)

# Better: Log the failure
except np.linalg.LinAlgError as e:
    logger.warning(
        f"OLS neutralization failed: {e}. "
        f"n_valid={n_valid}, n_exposures={X.shape[1]}, "
        f"date={date}"
    )
    residuals = np.full_like(y, np.nan)
```

**Impact:** Debugging production issues requires extensive instrumentation; no visibility into why metrics are NaN

**Locations needing logging:** ~20 sites identified across metrics

#### 🟡 Low Priority Issues

**Input Validation Gaps**

**Missing validation:** `metrics/ic.py:115`
```python
def compute_daily_ic(factor_batch, label_bundle, method="pearson", min_assets=10):
    # ❌ No check: min_assets could be negative or absurdly large
    # ❌ No check: method string not validated until line 122
```

**Improvement:**
```python
if min_assets < 1:
    raise InvalidContractError(
        f"min_assets must be >= 1, got {min_assets}",
        field="min_assets", expected=">=1", actual=min_assets
    )
if method not in ("pearson", "spearman"):
    raise InvalidContractError(
        f"Unknown correlation method: {method}",
        field="method", expected="pearson|spearman", actual=method
    )
```

---

### 1.2 factor_preprocess Package

#### ✅ Strengths

**Comprehensive Empty Checks**
```python
# transforms/cross_sectional.py - 4 functions have this pattern
if values.size == 0:
    return values.copy()
```

**Division-by-Zero Protection**
```python
# transforms/cross_sectional.py:116-120
result = np.where(
    std > 0,
    (values - mean) / std,
    constant_value,
)
```
- Uses `np.where` to avoid division by zero
- Explicit constant_value for zero-variance slices

**Strong Contract Validation** (`contracts/preprocessing.py`)
```python
def __post_init__(self):
    if not self.policy_id:
        raise ValueError("policy_id cannot be empty")
    if not self.transforms:
        raise ValueError("transforms cannot be empty")
    names = [t.name for t in self.transforms]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate transform names: {names}")
    if self.fit_start_time >= self.fit_end_time:
        raise ValueError("fit_start_time must be before fit_end_time")
```

#### ⚠️ Medium Issues

**Missing Logging in OLS Neutralization**

**Location:** `neutralization/ols.py:89-106`
```python
try:
    coef, _, rank, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
    predicted = X_valid @ coef
    residuals[valid] = y_valid - predicted
except np.linalg.LinAlgError:
    residuals = np.full_like(y, np.nan)  # ❌ No logging
```

**Impact:** Users don't know why neutralization failed (rank deficiency? singular matrix?)

#### 🟡 Low Priority Issues

**No DataFrame Structure Validation**
```python
def rolling_mean(values: pd.DataFrame, window: int, ...):
    # Assumes columns exist, no explicit check
    # If values is numpy array, fails with unclear AttributeError
```

**Recommendation:** Add type check at entry
```python
if not isinstance(values, pd.DataFrame):
    raise TypeError(
        f"Expected pd.DataFrame, got {type(values).__name__}. "
        "Pass DataFrame with columns matching factor_ids."
    )
```

---

### 1.3 factor_optimizer Package

#### ✅ Strengths

**Dataclass Validation** (`search/runner.py:29-35`)
```python
def __post_init__(self):
    if self.plateau_window < 1:
        raise ValueError("plateau_window must be >= 1")
    if self.plateau_threshold < 0:
        raise ValueError("plateau_threshold must be >= 0")
```

#### ⚠️ Medium Issues

**String-Based Validation Errors** (`grammar/validation.py`)

**Current Pattern:**
```python
errors = []
if spec is None:
    errors.append(f"Unknown mutation type: {mutation.mutation_type}")
return ValidationResult(False, errors, warnings, metadata)
```

**Problem:**
- Cannot branch on error type programmatically
- Must parse strings to understand failure
- No structured error context

**Better Pattern:**
```python
if spec is None:
    raise UnsupportedMutationError(
        f"Unknown mutation type: {mutation.mutation_type}",
        mutation_type=mutation.mutation_type,
        available_types=self.registry.list_mutation_types()
    )
```

#### 🟡 Low Priority Issues

**No Infinite Loop Protection**
- SearchRunner has budget limits but no hard iteration cap
- Very large max_trials could run indefinitely
- **Mitigation:** Budget tracker should timeout (verify in testing)

---

### 1.4 factor_assets Package

#### ✅ Strengths

**Excellent Error Taxonomy** (based on previous report)
```python
LifecycleConflictError(Exception)
DuplicateIdentityError(Exception)
AssetNotFoundError(Exception)
```

**Clear Error Messages** (`repository.py`)
```python
raise DuplicateIdentityError(
    f"Factor {factor_id} is already registered at "
    f"{self._assets[factor_id].registered_at}"
)
```
- Includes when it was registered
- User can verify if duplicate is intentional

#### ⚠️ Medium Issues

**Silent None Return** (`repository.py:160`)
```python
if factor_id is None:
    return None  # ❌ Should raise MissingInputError
```

**Better:**
```python
if factor_id is None:
    raise MissingInputError(
        "factor_id is required",
        field="factor_id",
        remediation="Provide non-None factor_id string"
    )
```

#### 🟡 Low Priority Issues

**No Thread-Safety Documentation**
- Repository is in-memory
- No explicit locking mechanism
- **Recommendation:** Add docstring section:
  ```
  Thread Safety
  -------------
  This repository is NOT thread-safe. Use external locking
  or operate from a single thread only.
  ```

---

### 1.5 research_control Package

#### ✅ Strengths

**Enum-Style Validation** (`events.py:19-25`)
```python
def __post_init__(self):
    if not self.campaign_id:
        raise ValueError("campaign_id is required")
    if self.event_type not in {"created", "started", "completed", "failed"}:
        raise ValueError(f"Invalid campaign event_type: {self.event_type}")
```

**Frozen Dataclasses**
- Prevents accidental mutation
- All events immutable after creation

#### 🟡 Low Priority Issues

**No Timestamp Ordering Validation**
```python
# Should validate: end_time > start_time
if self.end_time and self.start_time:
    if self.end_time <= self.start_time:
        raise ValueError("end_time must be after start_time")
```

---

## 2. Missing Input Validation Summary

### 2.1 Numeric Bounds

| Function | Parameter | Missing Check | Recommendation |
|----------|-----------|---------------|----------------|
| `compute_daily_ic` | `min_assets` | Could be negative | `if min_assets < 1: raise` |
| `compute_daily_ic` | `method` | String not validated early | Check against allowed set |
| `cs_zscore` | `axis` | Could be out of bounds | Let numpy raise (acceptable) |
| `rolling_mean` | `window` | Already validated ✓ | - |

**Status:** 🟡 Moderate - Most critical bounds checked, some gaps remain

### 2.2 Container Emptiness

**Good Coverage:**
- All transform functions check `values.size == 0`
- Contract dataclasses check empty lists
- Repository checks empty factor_ids

**Missing:**
- OLS neutralization doesn't check if `exposure_cols` is empty before groupby
- Could fail with obscure pandas KeyError

**Recommendation:** Add early check
```python
if not exposure_cols:
    raise InvalidContractError(
        "exposure_cols cannot be empty",
        field="exposure_cols",
        remediation="Provide at least one exposure column name"
    )
```

### 2.3 Type Validation

**Current Approach:** Duck typing with type hints

**Trade-off Analysis:**
- ✅ Pros: Low overhead, mypy catches most issues
- ⚠️ Cons: Runtime type errors can be confusing

**Example of unclear error:**
```python
def rolling_mean(values: pd.DataFrame, ...):
    # If user passes numpy array:
    # AttributeError: 'numpy.ndarray' object has no attribute 'rolling'
```

**Recommendation:** Add isinstance check for user-facing APIs only
```python
if not isinstance(values, pd.DataFrame):
    raise TypeError(
        f"values must be pd.DataFrame, got {type(values).__name__}"
    )
```

---

## 3. Unhandled Exceptions & Edge Cases

### 3.1 Division by Zero - Comprehensive Analysis

| File | Line | Expression | Guard Status | Risk | Fix Priority |
|------|------|------------|--------------|------|--------------|
| `portfolio_stats.py` | 176 | `/ running_max` | ❌ **NONE** | **CRITICAL** | **P0** |
| `portfolio_stats.py` | 138 | `/ std_excess` | ✅ L134 check | LOW | Done |
| `portfolio_stats.py` | 236 | `/ max_dd` | ✅ L233 check | LOW | Done |
| `portfolio_stats.py` | 294 | `/ downside_std` | ✅ L291 check | LOW | Done |
| `portfolio_stats.py` | 321 | `/ np.maximum(n_valid, 1)` | ✅ Built-in guard | LOW | Done |
| `cross_sectional.py` | 68 | `/ (n_finite - 1.0)` | ✅ L67 check | LOW | Done |
| `cross_sectional.py` | 118 | `/ std` | ✅ np.where guard | LOW | Done |

**Summary:** 6/7 divisions properly guarded. **1 critical vulnerability** remains.

### 3.2 NaN and Inf Handling

#### Current State

**NaN Handling:** ✅ Strong
- 74+ explicit checks across metrics
- `np.isfinite()`, `np.isnan()`, pairwise masking
- Frozen dataclasses have `is_valid()` method

**Inf Handling:** ⚠️ Weak
- Very few explicit `np.isinf()` checks
- Assumption: `np.isfinite()` catches Inf (correct but implicit)
- **No tests inject Inf values**

#### Inconsistency Issues

**Different patterns found:**
```python
# Pattern A: isfinite (best - catches both NaN and Inf)
valid = np.isfinite(values)

# Pattern B: ~isnan (misses Inf)
valid = ~np.isnan(values)

# Pattern C: explicit none check
if value is not None and not np.isnan(value):
```

**Recommendation:**
1. Standardize on `np.isfinite()` everywhere
2. Document policy in each package README
3. Add linter rule to catch `~np.isnan` usage

#### Test Gaps

**Missing Tests:** Inf injection scenarios

```python
# Test 1: Inf in factor values
def test_compute_ic_with_inf_factor():
    """Inf in factor should be excluded like NaN."""
    x = np.array([1, 2, np.inf, 4, 5], dtype=float)
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    
    corr = _pearson_correlation(x, y, min_obs=3)
    
    # Should compute on 4 valid points, excluding Inf
    expected = np.corrcoef([1, 2, 4, 5], [1, 2, 4, 5])[0, 1]
    assert np.isclose(corr, expected)

# Test 2: Inf in returns
def test_drawdown_with_inf_returns():
    """Inf returns should raise or return NaN."""
    returns = np.array([0.1, -0.5, np.inf, 0.2, -0.1])
    
    max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)
    
    # Either raise OverflowOrNonFiniteError or return all NaN
    assert np.all(np.isnan(dd_series)) or np.all(np.isfinite(dd_series))

# Test 3: Mixed NaN and Inf
def test_zscore_mixed_nan_inf():
    """Mixed NaN and Inf should both be excluded."""
    values = np.array([1, np.nan, 3, np.inf, 5])
    
    result = cs_zscore(values)
    
    # Should compute on [1, 3, 5] only
    assert np.isnan(result[1])
    assert np.isnan(result[3])
    assert np.all(np.isfinite(result[[0, 2, 4]]))

# Test 4: Negative Inf
def test_negative_inf_handling():
    """Negative Inf should also be excluded."""
    values = np.array([1, 2, -np.inf, 4, 5])
    
    result = cs_rank(values, pct=True)
    
    assert np.isnan(result[2])
    assert np.all(np.isfinite(result[[0, 1, 3, 4]]))
```

**Estimated Impact:** Add ~10 Inf-specific tests across all packages (4 hours)

### 3.3 Overflow and Underflow

#### Exponential Operations

**Risk Areas:**
- Decay weight calculations: `np.exp(-decay * age)`
- Cumulative products: `np.cumprod(1 + returns)`
- Log operations: `np.log(tiny_value)`

**Current Protection:** None found

**Example Vulnerability:**
```python
# If decay=0.1 and age=1000:
weight = np.exp(-0.1 * 1000)  # = exp(-100) ≈ 3.7e-44, underflows to 0
```

**Recommendation:**
```python
# Add overflow/underflow guards
with np.errstate(over='warn', under='warn'):
    weights = np.exp(-decay * age)
    if not np.all(np.isfinite(weights)):
        raise OverflowOrNonFiniteError(
            "Exponential decay produced non-finite weights",
            decay=decay, max_age=np.max(age)
        )
```

#### Cumulative Products

**Location:** `portfolio_stats.py:170`
```python
cum_returns = np.cumprod(1.0 + returns_filled, axis=0)
```

**Risk:** 
- Very positive returns: overflow to Inf
- Very negative returns: underflow to 0 or negative
- Alternating large returns: oscillation to Inf

**Test Needed:**
```python
def test_cumulative_returns_extreme_values():
    """Very large returns should not overflow."""
    # Returns that would overflow float64
    returns = np.array([10.0, 10.0, 10.0])  # 1000x, 10000x, 100000x
    
    with pytest.raises(OverflowOrNonFiniteError):
        compute_maximum_drawdown(returns)

def test_cumulative_returns_near_negative_one():
    """Returns near -1 should not cause numerical issues."""
    returns = np.array([-0.9, -0.9, -0.9, -0.9])  # Down 99.99%
    
    max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)
    
    assert np.all(np.isfinite(dd_series))
    assert max_dd > 0.99
```

#### Loss of Precision

**Risk:** Variance/std of very small or very large numbers

**Example:**
```python
values = np.array([1e15, 1e15 + 1, 1e15 + 2])  # Std ≈ 1, but precision loss
std = np.std(values)
# May compute as 0 due to floating point precision
```

**Test Needed:**
```python
def test_zscore_large_magnitude_small_variance():
    """Large values with small variance should not round to zero."""
    base = 1e15
    values = np.array([base, base + 1, base + 2, base + 3])
    
    result = cs_zscore(values)
    
    # Should not all be same value (zero variance detection)
    if np.all(result == result[0]):
        # Correctly detected as constant
        pass
    else:
        # Should have valid z-scores
        assert np.all(np.isfinite(result))
```

### 3.4 Memory Errors

**Current State:** ❌ No out-of-memory protection

**Risk Areas:**
```python
# Large array allocations
turnover = np.empty(T, dtype=np.float64)  # Could be 10M+ elements
ic_matrix = np.full((T, F), np.nan)  # Could be 1000 x 10000
```

**Impact:** 
- Abrupt crashes with no warning
- No graceful degradation
- No informative error message

**Recommendation:**
```python
def _check_memory_budget(shape, dtype, budget_mb=1000):
    """Verify allocation won't exceed memory budget."""
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = np.prod(shape) * bytes_per_element
    total_mb = total_bytes / (1024 * 1024)
    
    if total_mb > budget_mb:
        raise BudgetExceededError(
            f"Array allocation would use {total_mb:.1f}MB, "
            f"exceeds budget of {budget_mb}MB",
            required_mb=total_mb,
            budget_mb=budget_mb,
            shape=shape
        )

# Use before large allocations
_check_memory_budget((T, F), np.float64)
ic_matrix = np.full((T, F), np.nan)
```

**Test:**
```python
def test_memory_budget_exceeded():
    """Very large arrays should raise before allocation."""
    huge_shape = (1_000_000, 10_000)  # 80GB
    
    with pytest.raises(BudgetExceededError):
        _check_memory_budget(huge_shape, np.float64, budget_mb=1000)
```

### 3.5 Concurrency Races

**Current State:** ❌ No concurrency tests

**Risk Areas:**
- `factor_assets/registry`: In-memory repository with no locking
- `research_control/ledger`: Concurrent trial writes
- `factor_optimizer/seen`: Duplicate detection cache

**Example Race Condition:**
```python
# Two threads register same factor simultaneously
# Thread A: checks exists() -> False
# Thread B: checks exists() -> False  
# Thread A: writes to _assets dict
# Thread B: writes to _assets dict (second write wins)
# Result: DuplicateIdentityError not raised, or inconsistent state
```

**Test Needed:**
```python
def test_repository_concurrent_registration():
    """Concurrent registration of same factor should be safe."""
    import threading
    
    repo = AssetRepository()
    metadata = AssetMetadata(factor_id="test", ...)
    lineage = Lineage(...)
    
    errors = []
    successes = []
    
    def register():
        try:
            repo.register(metadata, lineage)
            successes.append(threading.current_thread().name)
        except DuplicateIdentityError as e:
            errors.append(threading.current_thread().name)
    
    threads = [threading.Thread(target=register) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Exactly 1 success, 9 failures
    assert len(successes) == 1
    assert len(errors) == 9
    assert repo.exists(metadata.factor_id)
```

**Note:** Current implementation likely NOT thread-safe. Test would expose bug.

---

## 4. Error Message Quality

### 4.1 Actionable vs Non-Actionable

#### ✅ Good Examples

**Example 1:** Shape mismatch with context
```python
raise ValueError(
    f"Shape mismatch for layout={self.layout}: "
    f"expected {expected_shape}, got {self.values.shape}"
)
```
- ✓ Shows expected vs actual
- ✓ Includes relevant context (layout)
- ✓ User can immediately see the problem

**Example 2:** Duplicate detection with timestamp
```python
raise DuplicateIdentityError(
    f"Factor {factor_id} is already registered at "
    f"{self._assets[factor_id].registered_at}"
)
```
- ✓ Shows when it was registered
- ✓ User can verify if intentional duplicate
- ✓ Actionable: check registration logic

**Example 3:** Clear constraint violation
```python
raise ValueError("fit_start_time must be before fit_end_time")
```
- ✓ Clearly states the rule
- ✓ Obvious fix: swap or adjust times

#### ⚠️ Needs Improvement

**Example 1:** No context for valid options
```python
# Current
raise ValueError(f"Unknown mutation type: {mutation.mutation_type}")

# Better
raise UnsupportedMutationError(
    f"Unknown mutation type: {mutation.mutation_type}",
    mutation_type=mutation.mutation_type,
    available_types=self.registry.list_mutation_types(),
    remediation="Use one of the available types or register custom mutation"
)
```

**Example 2:** Silent failure instead of error
```python
# Current
if factor_id is None:
    return None

# Better
if factor_id is None:
    raise MissingInputError(
        "factor_id is required but was None",
        field="factor_id",
        remediation="Provide non-None string identifier for the factor"
    )
```

**Example 3:** Generic error without guidance
```python
# Current
raise ValueError("No exposure columns found")

# Better
raise InvalidContractError(
    "No exposure columns found in DataFrame",
    field="exposure_cols",
    available_columns=list(df.columns),
    remediation="Ensure exposure_cols contains valid DataFrame column names"
)
```

### 4.2 Remediation Guidance

**Current State:** ❌ Missing in most errors

**Recommendation:** Add `remediation` field to all error types

```python
class InvalidContractError(ContractError):
    def __init__(self, message, *, field=None, expected=None, actual=None, remediation=None):
        self.message = message
        self.field = field
        self.expected = expected
        self.actual = actual
        self.remediation = remediation
        
        full_message = message
        if remediation:
            full_message += f"\n\nHow to fix: {remediation}"
        
        super().__init__(full_message)
```

**Examples:**
```python
# Contract error
raise InvalidContractError(
    "Factor batch has wrong number of dimensions",
    field="values.ndim",
    expected=3,
    actual=values.ndim,
    remediation="Reshape to (time, assets, factors). Use values[:, :, np.newaxis] for single factor."
)

# Capability error
raise UnsupportedMetricError(
    f"Metric '{metric_id}' not found",
    metric_id=metric_id,
    available_metrics=MetricRegistry.list(),
    remediation="Check spelling or use MetricRegistry.register() to add custom metric"
)
```

---

## 5. Edge Case Test Coverage

### 5.1 Current Test Coverage (from existing report)

| Package | Error Tests | Coverage Areas | Major Gaps |
|---------|-------------|----------------|------------|
| quant_evaluator | 49 | Contract validation, budgets, NaN | Inf values, overflow, concurrency |
| factor_optimizer | 34 | Mutation validation, adapters | Budget exhaustion, illegal mutations |
| factor_assets | 85 | Lifecycle, identity, graph | Staleness, collisions, concurrency |
| factor_preprocess | 61 | Empty inputs, parameters, ordering | Overflow, extreme missingness |
| research_control | ~15 (est) | Event validation | Concurrent writes, timestamp order |

**Total:** 244 error tests

### 5.2 Missing Test Scenarios

#### Empty Inputs ⚠️ Partial Coverage

**Well-tested:**
- Empty factor_ids list ✓
- Empty transforms list ✓
- Zero-size arrays in transforms ✓

**Missing:**
```python
# Test: Zero time periods
def test_ic_zero_time_periods():
    factor_batch = FactorBatch(values=np.empty((0, 100, 5)), ...)
    ic_series, counts = compute_daily_ic(factor_batch, label_bundle)
    assert ic_series.shape == (0, 5)

# Test: Zero assets
def test_ic_zero_assets():
    factor_batch = FactorBatch(values=np.empty((252, 0, 5)), ...)
    ic_series, counts = compute_daily_ic(factor_batch, label_bundle)
    assert np.all(np.isnan(ic_series))

# Test: Zero factors
def test_ic_zero_factors():
    factor_batch = FactorBatch(values=np.empty((252, 100, 0)), ...)
    ic_series, counts = compute_daily_ic(factor_batch, label_bundle)
    assert ic_series.shape == (252, 0)
```

#### NaN Patterns ✅ Good Coverage

**Well-tested:**
- All NaN in factor ✓
- All NaN in labels ✓
- Partial NaN (pairwise deletion) ✓
- Constant values (zero variance) ✓

**Additional scenarios:**
```python
# Test: First/last period all NaN (boundary)
def test_ic_nan_boundaries():
    values = np.random.randn(100, 50, 1)
    values[0, :, :] = np.nan  # First day all NaN
    values[-1, :, :] = np.nan  # Last day all NaN
    
    factor_batch = FactorBatch(values=values, ...)
    ic_series, counts = compute_daily_ic(factor_batch, label_bundle)
    
    assert np.isnan(ic_series[0, 0])
    assert np.isnan(ic_series[-1, 0])
    assert counts[0, 0] == 0
    assert counts[-1, 0] == 0
```

#### Inf Values ❌ NOT TESTED

**Critical gap:** No tests inject Inf values (see section 3.2 for test code)

#### Extreme Values ❌ MINIMAL

**Missing tests:**
```python
# Test: Near float64 limits
def test_extreme_large_values():
    values = np.array([1e307, 1e307, 1e308])
    # Should handle without overflow

# Test: Near zero
def test_extreme_small_values():
    values = np.array([1e-307, 1e-308, 0.0])
    # Should handle without underflow

# Test: Mixed scales
def test_mixed_magnitude_values():
    values = np.array([1e-10, 1.0, 1e10])
    # Normalization should work across scales
```

#### Type Errors ⚠️ Partial Coverage

**Missing:**
```python
# Test: Wrong input type
def test_rolling_mean_wrong_type():
    values = np.array([[1, 2, 3]])  # numpy not DataFrame
    
    with pytest.raises(TypeError, match="Expected pd.DataFrame"):
        rolling_mean(values, window=2)

# Test: Wrong ndim
def test_ic_wrong_ndim():
    values = np.random.randn(100, 50)  # 2D not 3D
    
    with pytest.raises(InvalidContractError, match="3 dimensions"):
        factor_batch = FactorBatch(values=values, ...)
```

#### Concurrent Operations ❌ ABSENT

All concurrency tests missing (see section 3.5)

### 5.3 Recommended New Test Suite

**Create:** `tests/edge_cases/test_comprehensive_boundaries.py`

```python
"""
Comprehensive edge case test suite.

Covers:
- Empty inputs (zero time/assets/factors)
- NaN/Inf in all positions
- Extreme numeric values
- Type errors
- Concurrent operations (where applicable)
"""

class TestEmptyInputs:
    """All dimensions can be zero."""
    # 15 tests
    
class TestNaNBoundaries:
    """NaN in first/last/middle positions."""
    # 10 tests
    
class TestInfHandling:
    """Inf, -Inf, mixed with NaN."""
    # 10 tests (NEW)
    
class TestExtremeNumericValues:
    """Near overflow/underflow/precision limits."""
    # 8 tests (NEW)
    
class TestTypeErrors:
    """Wrong types, wrong ndim."""
    # 5 tests (NEW)
    
class TestConcurrentOperations:
    """Race conditions, deadlocks."""
    # 5 tests (NEW)
```

**Total new tests:** ~53
**Estimated time:** 16-20 hours

---

## 6. Graceful Degradation

### 6.1 Current Pattern: Fail-to-NaN

**Widespread Usage:**
```python
# Pattern across all metrics
if insufficient_data or singular_matrix or zero_variance:
    return np.nan
```

**Pros:**
- ✅ Partial results preserved (some factors valid, others NaN)
- ✅ No crashes during batch processing
- ✅ Pipelines continue with reduced data

**Cons:**
- ⚠️ Silent failures if NaN not monitored
- ⚠️ Cannot distinguish "no data" from "computation failed"
- ⚠️ Difficult to debug why specific factor is NaN

### 6.2 Improvement: Diagnostic Metadata

**Current:**
```python
ic_series = np.array([0.5, np.nan, 0.3, np.nan])  # Why NaN?
```

**Recommended:**
```python
@dataclass
class ICResult:
    ic_values: np.ndarray  # (T, F)
    valid_counts: np.ndarray  # (T, F)
    failure_codes: dict[tuple[int, int], str]  # {(t, f): "constant_factor"}
    
def compute_daily_ic_v2(...) -> ICResult:
    failures = {}
    
    for t in range(T):
        for f in range(F):
            if is_constant(factor_t_f):
                ic_values[t, f] = np.nan
                failures[(t, f)] = "CONSTANT_FACTOR"
            elif n_valid < min_assets:
                ic_values[t, f] = np.nan
                failures[(t, f)] = f"INSUFFICIENT_OBS:{n_valid}"
            elif not np.isfinite(correlation):
                ic_values[t, f] = np.nan
                failures[(t, f)] = "NON_FINITE_RESULT"
    
    return ICResult(ic_values, valid_counts, failures)
```

**Usage:**
```python
result = compute_daily_ic_v2(...)

# Check for failures
if result.failure_codes:
    logger.warning(f"IC computation had {len(result.failure_codes)} failures")
    for (t, f), code in list(result.failure_codes.items())[:5]:
        logger.warning(f"  Factor {f} at time {t}: {code}")

# Backward compatible
ic_series = result.ic_values  # Just use the array
```

**Benefits:**
- ✓ Debuggable: Know exactly WHY each NaN occurred
- ✓ Monitorable: Count failure types
- ✓ Backward compatible: `.ic_values` property
- ✓ Structured: Machine-readable failure codes

### 6.3 Fail-Fast vs Fail-Safe Trade-offs

**Current:** Mostly fail-safe (return NaN)

**Should be fail-fast:**
1. **Timing contract violations** - PIT leakage risk
   ```python
   if label_time <= factor_time:
       raise TimingContractError("Label must be strictly after factor observation")
   ```

2. **Numerical overflow** - Inf propagates silently
   ```python
   if not np.all(np.isfinite(result)):
       raise OverflowOrNonFiniteError("Computation produced non-finite values")
   ```

3. **Required inputs missing** - Cannot proceed
   ```python
   if factor_id is None:
       raise MissingInputError("factor_id is required")
   ```

**Should remain fail-safe:**
1. **Insufficient observations** - Expected in sparse data
2. **Singular matrix** - Can happen legitimately
3. **Zero variance** - Valid edge case

---

## 7. Logging Coverage

### 7.1 Current State

**Logging Statements:** ~118 total (from previous report)

**Distribution:**
- quant_evaluator: ~40 (mostly INFO in backends/runtime)
- factor_optimizer: ~30
- factor_assets: ~20
- factor_preprocess: ~15
- research_control: ~13

**Patterns:**
- ✓ INFO: "Starting X", "Completed Y"
- ⚠️ Few WARNING for degradation
- ❌ Almost no ERROR (exceptions raised instead)

### 7.2 Critical Missing Logging

**Silent Degradation Sites:** ~20 identified

**Priority 1:** Add logging to all NaN returns

```python
# metrics/ic.py - Multiple sites
if n < min_obs:
    logger.warning(
        "Insufficient observations for IC",
        extra={
            "factor_idx": f,
            "date_idx": t,
            "n_valid": n,
            "min_required": min_obs
        }
    )
    return np.nan
```

**Priority 2:** Add logging to all exception handlers

```python
# neutralization/ols.py
except np.linalg.LinAlgError as e:
    logger.warning(
        "OLS neutralization failed - singular matrix",
        extra={
            "date": date,
            "n_valid": n_valid,
            "n_exposures": X.shape[1],
            "error": str(e)
        }
    )
    residuals = np.full_like(y, np.nan)
```

**Priority 3:** Add logging to budget checks

```python
# runtime/budgets.py
if current_usage.memory_mb > self.budget.max_memory_mb:
    logger.error(
        "Memory budget exceeded",
        extra={
            "current_mb": current_usage.memory_mb,
            "budget_mb": self.budget.max_memory_mb,
            "overage_pct": (current_usage.memory_mb / self.budget.max_memory_mb - 1) * 100
        }
    )
    raise BudgetExceededError(...)
```

### 7.3 Structured Logging Recommendations

**Use structured fields:**
```python
# Good - machine-readable
logger.warning("IC computation degraded", extra={
    "reason": "constant_factor",
    "factor_id": factor_id,
    "date": date.isoformat(),
    "n_valid": n_valid
})

# Bad - requires string parsing
logger.warning(f"IC computation degraded: constant factor {factor_id} on {date}")
```

**Benefits:**
- ✓ Aggregatable in log analytics
- ✓ Filterable by field
- ✓ Consistent format

---

## 8. Priority Recommendations

### 🔴 P0 - Critical (Fix Immediately)

**Estimated Time:** 1-2 hours

1. **Fix division by zero in drawdown** (15 min)
   - File: `quant_evaluator/metrics/portfolio_stats.py:176`
   - Add: `np.where(running_max > 0, ..., np.nan)`
   - Test: `test_drawdown_extreme_loss()`

2. **Fix broad exception handler** (10 min)
   - File: `quant_evaluator/runtime/budgets.py:216`
   - Catch specific: `psutil.NoSuchProcess`, `psutil.AccessDenied`
   - Add logging for unexpected errors

3. **Verify no other unguarded divisions** (30 min)
   - Search: `grep -rn "/" --include="*.py" | grep -v "^[[:space:]]*#"`
   - Check each division has guard or built-in protection

### 🟡 P1 - High (This Sprint)

**Estimated Time:** 2-3 days

4. **Add logging to silent degradation** (3 hours)
   - ~20 sites return NaN without logging
   - Add structured logging with reason codes
   - Include context (factor_id, date, n_valid)

5. **Add Inf handling tests** (4 hours)
   - 10 new tests injecting Inf values
   - Verify treated same as NaN
   - Test negative Inf, mixed NaN/Inf

6. **Add input validation** (3 hours)
   - `min_assets >= 1` check
   - `method` enum validation
   - Type checks for user-facing APIs
   - Add remediation guidance to errors

7. **Document thread-safety assumptions** (1 hour)
   - Add docstrings to all shared-state classes
   - Clarify if thread-safe or requires external locking

### 🟢 P2 - Medium (Next Sprint)

**Estimated Time:** 1 week

8. **Add overflow/underflow tests** (4 hours)
   - Extreme exponentscenarios
   - Cumulative product overflow
   - Loss of precision tests

9. **Add memory estimation** (4 hours)
   - Check array size before allocation
   - Raise BudgetExceededError if > limit
   - Add to all large allocation sites

10. **Improve error messages** (4 hours)
    - Add `remediation` field to all errors
    - Include available options
    - Reference documentation

11. **Add concurrency tests** (6 hours)
    - Repository concurrent registration
    - Ledger concurrent writes
    - Verify race condition handling

12. **Implement diagnostic metadata** (8 hours)
    - ICResult with failure_codes
    - Backward compatible API
    - Document usage patterns

### 🔵 P3 - Low (Backlog)

13. **Runtime type validation** (4 hours)
14. **Cancellation support** (8 hours)
15. **Error recovery documentation** (4 hours)
16. **Observability metrics** (16 hours)

---

## 9. Implementation Checklist

### Phase 1: Critical Fixes (Day 1)

- [ ] Fix `portfolio_stats.py:176` division by zero
- [ ] Fix `budgets.py:216` broad exception handler
- [ ] Add test `test_drawdown_extreme_loss`
- [ ] Add test `test_memory_tracking_errors`
- [ ] Run full test suite (verify no regressions)
- [ ] Commit and deploy

### Phase 2: High Priority (Days 2-4)

- [ ] Add logging to 20 silent degradation sites
- [ ] Create `test_comprehensive_boundaries.py` with 10 Inf tests
- [ ] Add input validation to top 10 public APIs
- [ ] Add `remediation` field to error messages
- [ ] Document thread-safety for 5 key classes
- [ ] Run full test suite
- [ ] Update error handling documentation

### Phase 3: Medium Priority (Days 5-9)

- [ ] Add 8 overflow/underflow tests
- [ ] Implement `_check_memory_budget()` utility
- [ ] Add memory checks to 5 large allocation sites
- [ ] Improve 15 error messages with remediation
- [ ] Add 5 concurrency tests
- [ ] Implement ICResult diagnostic metadata
- [ ] Update examples and guides

### Phase 4: Documentation & Monitoring (Days 10-12)

- [ ] Write error handling best practices guide
- [ ] Document NaN vs Inf policy
- [ ] Add structured logging guide
- [ ] Create error code reference
- [ ] Set up error rate monitoring dashboard (if applicable)

---

## 10. Acceptance Criteria

### Critical Fixes Complete

- [ ] Zero unguarded division operations in production code
- [ ] No broad `except Exception` handlers
- [ ] All critical bugs have regression tests
- [ ] Full test suite passes (244+ tests green)

### High Priority Complete

- [ ] All silent degradation has logging (20 sites)
- [ ] Inf values tested in 10+ scenarios
- [ ] Top 10 APIs have input validation
- [ ] Thread-safety documented for shared-state classes
- [ ] Error messages include remediation guidance

### Medium Priority Complete

- [ ] Overflow/underflow tested (8+ scenarios)
- [ ] Memory budget checks before large allocations
- [ ] Concurrency races tested (5+ scenarios)
- [ ] Diagnostic metadata available (ICResult pattern)
- [ ] 300+ total error tests

### Production Ready

- [ ] All above criteria met
- [ ] Error handling guide published
- [ ] Structured logging standardized
- [ ] Monitoring dashboards configured
- [ ] On-call runbook updated with common error patterns

---

## 11. Risk Assessment

### High Risk (Immediate Attention)

1. **Silent NaN Propagation → Trading Signals**
   - **Risk:** Invalid factors reach production undetected
   - **Frequency:** Medium (happens with sparse data)
   - **Impact:** HIGH - bad trades executed
   - **Mitigation:** Add OverflowOrNonFiniteError gates + monitoring

2. **Division by Zero in Drawdown**
   - **Risk:** Inf contaminates downstream metrics
   - **Frequency:** Low (extreme scenarios)
   - **Impact:** HIGH - portfolio optimization fails
   - **Mitigation:** Fix immediately (P0)

3. **Timing Contract Violations**
   - **Risk:** PIT leakage undetected
   - **Frequency:** Low (well-tested elsewhere)
   - **Impact:** CRITICAL - lookahead bias
   - **Mitigation:** Implement TimingContractError (defer to contract freeze work)

### Medium Risk (Monitor)

4. **Memory Exhaustion**
   - **Risk:** OOM crashes without warning
   - **Frequency:** Low (typical datasets fit in memory)
   - **Impact:** MEDIUM - process restart required
   - **Mitigation:** Add memory estimation (P2)

5. **Concurrent Registration Races**
   - **Risk:** Inconsistent repository state
   - **Frequency:** Low (mostly single-threaded usage)
   - **Impact:** MEDIUM - duplicate factors or missing entries
   - **Mitigation:** Add tests + document (P1/P2)

### Low Risk (Acceptable)

6. **Numerical Overflow in Edge Cases**
   - **Risk:** Inf from extreme inputs
   - **Frequency:** Very low
   - **Impact:** LOW - caught by downstream checks
   - **Mitigation:** Add tests (P2)

7. **Missing Cancellation**
   - **Risk:** Cannot stop long operations
   - **Frequency:** Low
   - **Impact:** LOW - workaround with timeouts
   - **Mitigation:** Defer to post-freeze (P3)

---

## 12. Conclusion

### Summary

Error handling in the quant packages demonstrates **solid foundational practices** with some **critical gaps**:

**Strengths:**
- Frozen dataclasses with __post_init__ validation
- Explicit NaN handling throughout metrics
- 244 error tests show systematic approach
- Clear error messages with context

**Critical Issues:**
- 2 P0 bugs: unguarded division, broad exception handler
- Missing logging makes debugging production issues difficult
- Inf values largely untested (NaN well-covered)
- No concurrency tests despite shared state

### Readiness Assessment

**Current State:** ⚠️ **ACCEPTABLE FOR RESEARCH, NOT PRODUCTION-READY**

- Research usage: Can tolerate occasional NaN results
- Production usage: Needs P0 + P1 fixes first

### Recommended Path Forward

**Immediate (Today):**
1. Fix 2 P0 bugs (1-2 hours)
2. Deploy with regression tests

**This Sprint (Week 1):**
1. Add logging to silent degradation (day 2)
2. Add Inf tests (day 3)
3. Input validation + thread-safety docs (day 4)

**Next Sprint (Week 2-3):**
1. Overflow/underflow tests
2. Memory estimation
3. Concurrency tests
4. Diagnostic metadata

**Timeline:** 2-3 weeks to production-ready

### Success Metrics

- Zero P0/P1 bugs remaining
- 300+ error tests (from 244)
- <5% NaN rate in production (with monitoring)
- <1 error-related incident per month
- Mean time to diagnose errors <30 minutes (via logging)

---

## Appendix A: Files Requiring Changes

### P0 Fixes
1. `/home/shw/quant_projects/quant_evaluator/metrics/portfolio_stats.py` (line 176)
2. `/home/shw/quant_projects/quant_evaluator/runtime/budgets.py` (line 216)

### P1 Changes
3. `/home/shw/quant_projects/quant_evaluator/metrics/ic.py` (add logging)
4. `/home/shw/quant_projects/quant_evaluator/metrics/portfolio_stats.py` (add logging)
5. `/home/shw/quant_projects/factor_preprocess/neutralization/ols.py` (add logging)
6. `/home/shw/quant_projects/factor_assets/registry/repository.py` (thread-safety docs)
7. `/home/shw/quant_projects/research_control/ledger/*.py` (thread-safety docs)

### New Test Files
8. `/home/shw/quant_projects/tests/edge_cases/test_comprehensive_boundaries.py` (NEW)
9. `/home/shw/quant_projects/quant_evaluator/tests/metrics/test_inf_handling.py` (NEW)
10. `/home/shw/quant_projects/factor_assets/tests/test_concurrent_operations.py` (NEW)

---

## Appendix B: Error Code Reference (Proposed)

```
QE001: Insufficient observations for metric
QE002: Constant factor (zero variance)
QE003: Constant label (zero variance)
QE004: Shape mismatch (factor vs label)
QE005: Invalid correlation method
QE006: Memory budget exceeded
QE007: Time budget exceeded
QE008: Numerical overflow/underflow
QE009: Division by zero detected
QE010: Timing contract violation

FP001: Empty transforms list
FP002: Duplicate transform names
FP003: Fit window invalid
FP004: Time series not sorted
FP005: Singular matrix in neutralization
FP006: Insufficient data for fit
FP007: Distribution mismatch (fit vs transform)
FP008: Extreme missingness (>99%)

FA001: Factor ID already registered
FA002: Factor ID not found
FA003: Invalid lifecycle transition
FA004: Evidence reference stale
FA005: Canonical hash collision
FA006: Concurrent registration conflict

FO001: Unknown mutation type
FO002: Parent count mismatch
FO003: Parameter validation failed
FO004: Illegal mutation (governance violation)
FO005: Search budget exhausted
FO006: Evaluator unavailable

RC001: Campaign ID required
RC002: Invalid event type
RC003: Timestamp ordering violated
RC004: Concurrent trial write conflict
```

---

**END OF COMPREHENSIVE AUDIT**

Generated: 2026-08-14  
Auditor: Deep code inspection + manual review  
Next Review: After P0/P1 fixes implemented (est. 1 week)

#### quant_evaluator ✓ GOOD
**File:** `/home/shw/quant_projects/quant_evaluator/contracts/errors.py`

**Implemented:**
```python
# Base errors
ContractError(Exception)
  └─ InvalidContractError(ContractError)
  └─ SchemaVersionError(ContractError)

DataError(Exception)
  └─ InsufficientObservations(DataError)
  └─ InvalidValidityMask(DataError)

CapabilityError(Exception)
  └─ UnsupportedMetricError(CapabilityError)
  └─ OptionalDependencyMissing(ImportError)
```

**Alignment with CONTRACT_FREEZE_DRAFT:**
- ✓ ContractError base family exists
- ✓ InvalidContractError implemented
- ✓ SchemaVersionError implemented  
- ✓ InsufficientObservations implemented
- ✓ InvalidValidityMask implemented
- ✓ UnsupportedMetricError implemented
- ✗ Missing: MissingInputError, TimingContractError, SnapshotMismatchError
- ✗ Missing: EvidenceError family (MissingLabelError, EvidenceUnavailableError, StaleEvidenceError)
- ✗ Missing: ExecutionError family (NumericalFailure, OverflowOrNonFiniteError, BudgetExceededError, CancellationError)
- ✗ Missing: GovernanceError family

**Coverage:** 6/21 required error types (29%)

#### factor_optimizer ✗ MINIMAL
**Files:** 
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/adapters/factor_engine.py:88`
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/adapters/quant_evaluator.py:79`

**Implemented:**
```python
OptionalDependencyMissing(Exception)  # Duplicated in 2 adapter modules
```

**Alignment with CONTRACT_FREEZE_DRAFT:**
- ✓ OptionalDependencyMissing exists (but scattered)
- ✗ Missing: UnsupportedMutationError
- ✗ Missing: IllegalMutationError
- ✗ Missing: All base error families
- ✗ Missing: CollisionError, ContractChangeRequired

**Coverage:** 1/21 required error types (5%)

**Critical Gap:** No ValidationResult errors are properly typed exceptions; they're string-based error lists.

#### factor_assets ✗ MINIMAL
**Files:**
- `/home/shw/quant_projects/factor_assets/contracts/lifecycle.py:108`
- `/home/shw/quant_projects/factor_assets/registry/repository.py:22,27`
- `/home/shw/quant_projects/factor_assets/adapters/__init__.py:23`

**Implemented:**
```python
LifecycleConflictError(Exception)
DuplicateIdentityError(Exception)
AssetNotFoundError(Exception)
OptionalDependencyMissing(ImportError)
```

**Alignment with CONTRACT_FREEZE_DRAFT:**
- ✓ LifecycleConflictError implemented
- ✓ DuplicateIdentityError implemented
- ✓ OptionalDependencyMissing exists
- ✗ Missing: CollisionError (distinct from DuplicateIdentityError per spec)
- ✗ Missing: All base error families
- ✗ Missing: EvidenceUnavailableError, StaleEvidenceError
- ✗ Missing: ContractChangeRequired

**Coverage:** 3/21 required error types (14%)

**Critical Gap:** No base error families; all exceptions inherit directly from Exception.

#### factor_preprocess ✗ MINIMAL
**Files:**
- `/home/shw/quant_projects/factor_preprocess/factor_preprocess/adapters/data_access.py:12`
- `/home/shw/quant_projects/factor_preprocess/factor_preprocess/adapters/factor_assets.py:13`

**Implemented:**
```python
OptionalDependencyMissing(Exception)  # Duplicated in 2 adapter modules
```

**Alignment with CONTRACT_FREEZE_DRAFT:**
- ✓ OptionalDependencyMissing exists (but scattered)
- ✗ Missing: UnsupportedTransformError
- ✗ Missing: All base error families
- ✗ Missing: NumericalFailure, OverflowOrNonFiniteError

**Coverage:** 1/21 required error types (5%)

**Critical Gap:** Neutralization and transform failures raise generic ValueError instead of typed exceptions.

### 1.2 Error Taxonomy Gaps Summary

| Error Type | QE | FO | FA | FP | Priority |
|------------|----|----|----|----|----------|
| **ContractError base** | ✓ | ✗ | ✗ | ✗ | P0 |
| SchemaVersionError | ✓ | ✗ | ✗ | ✗ | P1 |
| MissingInputError | ✗ | ✗ | ✗ | ✗ | P0 |
| InvalidContractError | ✓ | ✗ | ✗ | ✗ | P0 |
| TimingContractError | ✗ | ✗ | ✗ | ✗ | P0 |
| SnapshotMismatchError | ✗ | ✗ | ✗ | ✗ | P1 |
| **CapabilityError base** | ✓ | ✗ | ✗ | ✗ | P0 |
| UnsupportedMetricError | ✓ | ✗ | ✗ | ✗ | P1 |
| UnsupportedTransformError | ✗ | ✗ | ✗ | ✗ | P1 |
| UnsupportedMutationError | ✗ | ✗ | ✗ | ✗ | P1 |
| OptionalDependencyMissing | ✓ | ✓ | ✓ | ✓ | P2 (consolidate) |
| **DataError/EvidenceError base** | ✓ | ✗ | ✗ | ✗ | P0 |
| InsufficientObservations | ✓ | ✗ | ✗ | ✗ | P0 |
| InvalidValidityMask | ✓ | ✗ | ✗ | ✗ | P1 |
| MissingLabelError | ✗ | ✗ | ✗ | ✗ | P0 |
| EvidenceUnavailableError | ✗ | ✗ | ✗ | ✗ | P1 |
| StaleEvidenceError | ✗ | ✗ | ✗ | ✗ | P1 |
| **ExecutionError base** | ✗ | ✗ | ✗ | ✗ | P0 |
| NumericalFailure | ✗ | ✗ | ✗ | ✗ | P0 |
| OverflowOrNonFiniteError | ✗ | ✗ | ✗ | ✗ | P0 |
| BudgetExceededError | ✗ | ✗ | ✗ | ✗ | P1 |
| CancellationError | ✗ | ✗ | ✗ | ✗ | P2 |
| **GovernanceError base** | ✗ | ✗ | ✗ | ✗ | P0 |
| IllegalMutationError | ✗ | ✗ | ✗ | ✗ | P0 |
| LifecycleConflictError | ✗ | ✗ | ✓ | ✗ | P0 |
| DuplicateIdentityError | ✗ | ✗ | ✓ | ✗ | P0 |
| CollisionError | ✗ | ✗ | ✗ | ✗ | P1 |
| ContractChangeRequired | ✗ | ✗ | ✗ | ✗ | P2 |

**Implementation Rate:** 27/108 = 25% coverage across 4 packages

---

## 2. Error Boundary Checks

### 2.1 Input Validation Completeness

#### quant_evaluator ✓ GOOD
**Locations:**
- `/home/shw/quant_projects/quant_evaluator/contracts/factor_batch.py` - FactorBatch `__post_init__`
- `/home/shw/quant_projects/quant_evaluator/contracts/label_bundle.py` - LabelBundle validation
- `/home/shw/quant_projects/quant_evaluator/api/requests.py` - Request validation

**Validation Coverage:**
- ✓ Empty factor_ids check (line 42)
- ✓ Required axis validation (line 44)
- ✓ None values check (line 46)
- ✓ Shape mismatch detection (line 49-54)
- ✓ Validity mask shape alignment (line 56-59)
- ✓ Frozen dataclass enforcement

**Example (FactorBatch):**
```python
def __post_init__(self):
    if not self.factor_ids:
        raise ValueError("factor_ids cannot be empty")
    if self.time_axis is None or self.asset_axis is None:
        raise ValueError("time_axis and asset_axis are required")
    if self.values is None:
        raise ValueError("values cannot be None")
    expected_shape = (self.time_axis.size, self.asset_axis.size, len(self.factor_ids))
    if self.layout == "wide" and self.values.shape != expected_shape[:2] + (len(self.factor_ids),):
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {self.values.shape}")
```

**Gap:** Uses generic ValueError instead of InvalidContractError.

#### factor_optimizer ⚠ PARTIAL
**Locations:**
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/grammar/validation.py` - MutationValidator

**Validation Coverage:**
- ✓ Mutation type existence check (line 72-75)
- ✓ Parent count validation (line 88-95)
- ✓ Parameter validation through spec (line 98-100)
- ✓ FE adapter legality checks (line 102-110)
- ✗ No frozen dataclass on CandidateMutation
- ✗ Returns ValidationResult with string errors instead of typed exceptions

**Critical Pattern:**
```python
# Current: string-based error accumulation
errors = []
if spec is None:
    errors.append(f"Unknown mutation type: {mutation.mutation_type}")
    return ValidationResult(False, errors, warnings, metadata)

# Should be: typed exception
if spec is None:
    raise UnsupportedMutationError(
        f"Unknown mutation type: {mutation.mutation_type}",
        mutation_type=mutation.mutation_type,
        available_types=self.registry.list_mutation_types()
    )
```

#### factor_assets ⚠ PARTIAL
**Locations:**
- `/home/shw/quant_projects/factor_assets/contracts/asset.py` - AssetMetadata/FactorAsset `__post_init__`
- `/home/shw/quant_projects/factor_assets/identity/identity.py` - Identity validation
- `/home/shw/quant_projects/factor_assets/registry/repository.py` - Repository checks

**Validation Coverage:**
- ✓ Required field checks (factor_id, canonical_hash, frequency)
- ✓ Frozen dataclass enforcement
- ✓ Duplicate identity detection in repository (line 22)
- ✓ Asset not found handling (line 27)
- ⚠ Lifecycle state transition validation exists but incomplete
- ✗ No validation for canonical_hash format/length in AssetMetadata
- ✗ No validation for timestamp format (registered_at, first_evaluated_at)

**Example Gap (AssetMetadata):**
```python
def __post_init__(self):
    if not self.factor_id:
        raise ValueError("factor_id is required")
    if not self.canonical_hash:
        raise ValueError("canonical_hash is required")  # Should validate format/length
    if not self.frequency:
        raise ValueError("frequency is required")  # Should validate against allowed values
```

#### factor_preprocess ✓ GOOD
**Locations:**
- `/home/shw/quant_projects/factor_preprocess/contracts/preprocessing.py` - Policy/FittedState validation
- `/home/shw/quant_projects/factor_preprocess/transforms/*` - Transform parameter validation
- `/home/shw/quant_projects/factor_preprocess/neutralization/*` - Neutralization validation

**Validation Coverage:**
- ✓ Empty policy checks (policy_id, version, name)
- ✓ Empty transforms list check
- ✓ Duplicate transform name detection
- ✓ fit_start_time < fit_end_time validation
- ✓ feature_order required for fitted state
- ✓ Duplicate feature ID detection
- ✓ Time series sorting validation
- ✓ Parameter range checks (window, halflife, max_lag)
- ✓ Frozen dataclass enforcement

**Strong Example (PreprocessingPolicy):**
```python
def __post_init__(self):
    if not self.policy_id:
        raise ValueError("policy_id cannot be empty")
    if not self.transforms:
        raise ValueError("transforms cannot be empty")
    names = [t.name for t in self.transforms]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate transform names: {names}")
```

### 2.2 Boundary Condition Handling

#### Empty Data Handling ⚠ INCONSISTENT

**Good Examples:**
- `quant_evaluator/metrics/portfolio_stats.py:285` - Empty downside returns check
- `factor_assets/registry/repository.py:258` - Empty timestamps handling
- All contracts check empty factor_ids/transforms

**Gaps:**
- No systematic empty batch handling in QE metric kernels
- No zero-asset or zero-time period gates
- Missing minimum observation thresholds in some metrics

#### NaN/Inf Handling ✓ PRESENT BUT INCONSISTENT

**Pattern Analysis:**
```
quant_evaluator/metrics: 38 instances of np.isfinite/isnan/isinf checks
factor_batch.py: is_valid() method checks NaN/Inf (line 76)
```

**Good Pattern (quality.py):**
```python
factor_finite = np.isfinite(values)  # (T, N, F)
label_finite = np.isfinite(labels)   # (T, N)
pairwise = factor_finite & label_finite[:, :, np.newaxis]
```

**Inconsistency:**
- Some functions use `~np.isnan`, others use `np.isfinite`
- Some fill NaN with 0, others filter them out
- No documented policy on NaN semantics (signal vs. invalid)

**Recommendation:** Adopt `np.isfinite()` uniformly (catches both NaN and ±Inf).

#### Division by Zero ⚠ PARTIAL

**Good Examples:**
- `portfolio_stats.py:134` - Zero std check before Sharpe calculation
- `portfolio_stats.py:233` - Zero max_dd check before Calmar
- `portfolio_stats.py:290` - Zero downside_std check before Sortino

**Pattern:**
```python
if std_excess == 0 or std_excess < 1e-10 or not np.isfinite(std_excess):
    return MetricValue(metric_id, factor_id, np.nan, None)
```

**Gaps:**
- No systematic epsilon policy (some use 1e-10, some use 0)
- Historical safe_division corruption (de910bcc) shows risk of bulk rewrites
- Missing overflow checks in cumulative products

#### Edge Case: Constant Values ⚠ PARTIAL

**Handled:**
- Rank correlation returns NaN for constant factors (correct per spec)
- Zero variance detected in several metrics

**Missing:**
- No explicit constant-detection utility
- No minimum variance threshold policy
- Some metrics may divide by near-zero variance

---

## 3. Error Message Quality

### 3.1 Message Clarity Assessment

#### Positive Examples ✓

**factor_batch.py (line 51):**
```python
raise ValueError(
    f"Shape mismatch for layout={self.layout}: "
    f"expected (*{expected_shape}, {len(self.factor_ids)}), got {self.values.shape}"
)
```
- ✓ Clear problem statement
- ✓ Expected vs actual values
- ✓ Relevant context (layout)

**factor_preprocess contracts (line 191):**
```python
raise ValueError("fit_start_time must be before fit_end_time")
```
- ✓ Clear constraint violation
- ✓ Actionable

**factor_assets graph (line 21):**
```python
raise ValueError("Self-loops not allowed in factor correlation graph")
```
- ✓ Clear prohibition
- ✓ Domain context

#### Improvement Needed ⚠

**factor_assets repository (line 160):**
```python
if factor_id is None:
    return None  # Should raise MissingInputError with guidance
```
- ✗ Silent failure instead of explicit error
- ✗ No caller guidance

**factor_optimizer validation (line 74):**
```python
errors.append(f"Unknown mutation type: {mutation.mutation_type}")
```
- ⚠ String error instead of typed exception
- ✗ Doesn't suggest valid types
- ✗ No reference to registry

**quant_evaluator diagnosis (line 167):**
```python
raise ValueError(f"factor_idx {factor_idx} out of range for {self.num_factors} factors")
```
- ⚠ Generic ValueError instead of InvalidContractError
- ✓ But message is clear

### 3.2 Context Inclusion ✓ GENERALLY GOOD

Most error messages include:
- ✓ Actual values that caused the error
- ✓ Expected values or constraints
- ✓ Object/field names for context

### 3.3 Remediation Guidance ✗ MISSING

**Current State:**
- Error messages describe what failed
- They do NOT suggest how to fix it

**Examples of Missing Guidance:**

**Should add:**
```python
# Current
raise ValueError("transforms cannot be empty")

# Better
raise InvalidContractError(
    "transforms cannot be empty",
    remediation="Add at least one TransformSpec to the policy. "
                "See TransformRegistry.list() for available transforms."
)
```

**Should add:**
```python
# Current
raise ValueError(f"Unknown mutation type: {mutation.mutation_type}")

# Better
raise UnsupportedMutationError(
    f"Unknown mutation type: {mutation.mutation_type}",
    mutation_type=mutation.mutation_type,
    available_types=self.registry.list_mutation_types(),
    remediation="Check available mutation types with MutationRegistry.list() "
                "or register a custom mutation with MutationRegistry.register()"
)
```

---

## 4. Failure Mode Testing

### 4.1 Test Coverage by Package

| Package | Total Error Tests | Coverage Areas | Gaps |
|---------|------------------|----------------|------|
| quant_evaluator | 49 | Contract validation, budget exceeded, invalid inputs, optional dependencies | Missing: numerical overflow, cancellation, staleness |
| factor_optimizer | 34 | Mutation validation, FE adapter failures, duplicate registration | Missing: illegal mutations, budget exceeded, search failures |
| factor_assets | 85 | Lifecycle transitions, duplicate identity, graph validation, aggregation | Missing: evidence staleness, collision scenarios, large-scale stress |
| factor_preprocess | 61 | Empty inputs, invalid parameters, time ordering, fitted state | Missing: numerical overflow, extreme missingness, memory exhaustion |

**Total:** 229 error test cases

### 4.2 Coverage Analysis by Error Category

#### Input Validation Tests ✓ STRONG
- 85+ tests across packages
- Good coverage of: empty inputs, shape mismatches, required fields, duplicates, invalid ranges

#### Boundary Condition Tests ⚠ MODERATE
- ~40 tests identified
- Coverage: empty data, zero values, constant factors
- **Gaps:** extreme values, edge of numeric precision, very large/small inputs

#### Optional Dependency Tests ✓ ADEQUATE
- 4 explicit tests for OptionalDependencyMissing
- Tests verify adapter import isolation
- **Gap:** No tests for partial dependency availability (e.g., some DA features missing)

#### Resource Exhaustion Tests ✗ MINIMAL
- 2 tests in QE for budget exceeded (time, operation count)
- **Major Gap:** No memory exhaustion, disk full, network timeout tests

#### Numerical Failure Tests ✗ MINIMAL
- Division by zero handled in some metrics
- **Major Gaps:** 
  - No overflow tests
  - No underflow tests
  - No loss-of-precision tests
  - No extreme exponent tests (exp(1000), log(1e-300))

#### Concurrent Failure Tests ✗ ABSENT
- No race condition tests
- No deadlock tests
- No concurrent modification tests

### 4.3 Public API Failure Path Coverage

#### quant_evaluator.evaluate() ⚠ PARTIAL
**Tested failure paths:**
- ✓ Empty factor_ids
- ✓ Shape mismatches
- ✓ Missing labels
- ✓ Invalid metric IDs
- ✓ Budget exceeded

**Untested failure paths:**
- ✗ Label timing ambiguity
- ✗ Snapshot mismatch between factor and label
- ✗ Numerical overflow in metric computation
- ✗ Cancellation mid-computation

#### factor_optimizer.optimize() ⚠ PARTIAL
**Tested failure paths:**
- ✓ Invalid mutation type
- ✓ Parent count mismatch
- ✓ FE compilation failure
- ✓ Parameter validation failure

**Untested failure paths:**
- ✗ Search budget exhaustion mid-search
- ✗ Evaluator unavailable
- ✗ Optimizer state corruption
- ✗ Illegal mutation (violates governance)

#### factor_assets.register() ⚠ PARTIAL
**Tested failure paths:**
- ✓ Duplicate identity
- ✓ Missing required fields
- ✓ Invalid lifecycle transition

**Untested failure paths:**
- ✗ Evidence reference stale
- ✗ Concurrent registration collision
- ✗ Repository unavailable
- ✗ Identity hash collision (distinct factors, same hash)

#### factor_preprocess.fit() / transform() ⚠ PARTIAL
**Tested failure paths:**
- ✓ Empty transforms list
- ✓ Invalid fit window
- ✓ Time series not sorted
- ✓ Parameter validation

**Untested failure paths:**
- ✗ Singular matrix in neutralization
- ✗ Insufficient data for fit
- ✗ Fitted state/data distribution mismatch
- ✗ Extreme missingness (>99% missing)

---

## 5. Critical Findings

### 5.1 P0 Issues (Block Production)

1. **Missing Base Error Families**
   - Only QE has ContractError base; FO/FA/FP use raw Exception
   - ExecutionError family completely absent across all packages
   - GovernanceError family scattered (FA has pieces, FO/FP missing)
   - **Impact:** Cannot branch on error types; must parse strings

2. **No Typed Execution Errors**
   - NumericalFailure, OverflowOrNonFiniteError absent
   - Metrics can silently return NaN instead of failing fast
   - **Impact:** Silent failure propagation to trading signals

3. **No Timing Contract Errors**
   - TimingContractError not implemented anywhere
   - Label timing ambiguity fails with generic ValueError
   - **Impact:** PIT leakage risk; cannot distinguish timing vs. other contract violations

4. **Missing MissingInputError**
   - Required inputs checked with generic ValueError or silent None returns
   - **Impact:** Cannot distinguish missing vs. invalid inputs

5. **String-Based Validation Errors (FO)**
   - ValidationResult uses error strings, not typed exceptions
   - **Impact:** No structured error handling; string parsing required

### 5.2 P1 Issues (Degrade Usability)

6. **OptionalDependencyMissing Duplication**
   - Defined separately in 6+ locations across packages
   - Should be in shared contract location or each package's errors module
   - **Impact:** Inconsistent import paths, potential isinstance() failures

7. **No Evidence Staleness Detection**
   - StaleEvidenceError not implemented
   - No timestamp/version checks in FA evidence refs
   - **Impact:** May use outdated evidence for admission decisions

8. **Inconsistent NaN/Inf Handling**
   - Mix of `np.isnan`, `np.isfinite`, explicit checks
   - No documented policy on NaN semantics
   - **Impact:** Unpredictable behavior across metrics

9. **Missing Numerical Overflow Detection**
   - No tests for exp(large), log(tiny), cumulative products
   - **Impact:** Silent overflow to inf, underflow to 0

10. **Insufficient Error Context**
    - Most errors lack remediation guidance
    - Generic ValueError used instead of domain-specific types
    - **Impact:** Poor developer experience, hard to debug

### 5.3 P2 Issues (Technical Debt)

11. **No Resource Exhaustion Tests**
    - Memory, disk, time limits not systematically tested
    - **Impact:** Unknown behavior under resource pressure

12. **No Concurrent Failure Tests**
    - Registration collisions, repository races untested
    - **Impact:** Production race conditions undetected

13. **Missing Cancellation Support**
    - CancellationError defined in spec but not implemented
    - No tests for mid-computation cancellation
    - **Impact:** Cannot gracefully stop long-running operations

---

## 6. Improvement Recommendations

### 6.1 Phase 1: Core Taxonomy (P0)

**Goal:** Align all packages with CONTRACT_FREEZE_DRAFT.md Section 7

**Actions:**

1. **Create Shared Error Module Structure**
   ```
   Each package gets contracts/errors.py with full taxonomy:
   - quant_evaluator/contracts/errors.py ✓ (extend existing)
   - factor_optimizer/contracts/errors.py (NEW)
   - factor_assets/contracts/errors.py (NEW)
   - factor_preprocess/contracts/errors.py (NEW)
   ```

2. **Implement Base Error Families (all packages)**
   ```python
   # Each package implements full hierarchy
   ContractError(Exception)
   CapabilityError(Exception)
   DataError(Exception)
   ExecutionError(Exception)
   GovernanceError(Exception)
   ```

3. **Implement Missing Critical Types**
   - MissingInputError (all packages)
   - TimingContractError (QE, FP)
   - NumericalFailure, OverflowOrNonFiniteError (QE, FP)
   - IllegalMutationError (FO)
   - EvidenceUnavailableError, StaleEvidenceError (FA, QE)

4. **Replace Generic ValueError**
   - Audit all `raise ValueError` calls
   - Replace with appropriate typed exception
   - Add structured error context

5. **Refactor FO ValidationResult**
   - Raise typed exceptions instead of accumulating strings
   - Keep ValidationResult for warnings only
   - Add structured error codes

**Acceptance:**
- All packages have contracts/errors.py
- 100% of error types from CONTRACT_FREEZE_DRAFT implemented
- Zero generic ValueError in public API paths
- All tests pass with new error types

### 6.2 Phase 2: Error Context & Messages (P1)

**Actions:**

1. **Add Structured Error Context**
   ```python
   class InvalidContractError(ContractError):
       def __init__(self, message, *, field=None, expected=None, actual=None, remediation=None):
           self.field = field
           self.expected = expected
           self.actual = actual
           self.remediation = remediation
           super().__init__(self._format_message(message))
   ```

2. **Add Remediation Guidance**
   - Every user-facing error includes how to fix it
   - Reference documentation or relevant methods
   - Suggest valid alternatives

3. **Standardize NaN/Inf Handling**
   - Document policy: `np.isfinite()` for all validity checks
   - Distinguish NaN signal (user provided) vs. computational NaN
   - Add explicit overflow checks before operations likely to overflow

4. **Add Error Codes**
   ```python
   class QuantEvaluatorError(Exception):
       code: str  # "QE001", "QE002", etc.
   ```
   - Stable codes for programmatic branching
   - Separate error message evolution from code logic

**Acceptance:**
- All exceptions have structured fields (not just strings)
- 100% of public API errors include remediation guidance
- Error codes assigned and documented
- NaN/Inf policy documented and uniformly applied

### 6.3 Phase 3: Comprehensive Failure Testing (P1)

**Actions:**

1. **Add Numerical Failure Tests**
   ```python
   test_overflow_in_exp()
   test_underflow_in_log()
   test_loss_of_precision_in_cumulative()
   test_extreme_input_values()
   ```

2. **Add Resource Exhaustion Tests**
   ```python
   test_memory_budget_exceeded()
   test_time_budget_exceeded()
   test_operation_count_exceeded()
   ```

3. **Add Evidence Staleness Tests**
   ```python
   test_stale_evidence_rejected()
   test_evidence_timestamp_validation()
   test_evidence_version_mismatch()
   ```

4. **Add Public API Failure Matrix**
   - Every public function has 3-5 negative test cases
   - Cover: missing input, invalid input, boundary, resource exhaustion

5. **Add Concurrent Failure Tests**
   ```python
   test_concurrent_registration_collision()
   test_repository_race_condition()
   test_concurrent_evidence_update()
   ```

**Acceptance:**
- 400+ error test cases (from 229)
- Every public API has negative test coverage
- Numerical overflow/underflow tested
- Resource exhaustion scenarios tested

### 6.4 Phase 4: Production Hardening (P2)

**Actions:**

1. **Add Fail-Fast Gates**
   - Check for NaN/Inf immediately after risky operations
   - Raise OverflowOrNonFiniteError instead of returning NaN
   - Distinguish expected NaN (insufficient data) from computational failure

2. **Add Input Sanitization**
   - Validate all numeric inputs are finite
   - Check array shapes before operations
   - Verify time series sorted

3. **Add Cancellation Support**
   - Implement CancellationToken pattern
   - Check cancellation in long-running loops
   - Raise CancellationError with partial results

4. **Add Error Recovery Documentation**
   - Document retry strategies
   - Document transaction/rollback semantics
   - Document partial failure handling

**Acceptance:**
- Zero silent NaN propagation in production paths
- Cancellation tested in all long-running operations
- Error recovery guide published

---

## 7. Implementation Priority & Sequencing

### Critical Path (Block Freeze)

1. **contracts/errors.py modules** (2-3 days)
   - factor_optimizer/contracts/errors.py
   - factor_assets/contracts/errors.py
   - factor_preprocess/contracts/errors.py
   - Extend quant_evaluator/contracts/errors.py

2. **Base error families** (1 day)
   - All packages implement 5 base families
   - Update __init__.py exports

3. **Replace ValueError in public APIs** (3-4 days)
   - factor_optimizer: ValidationResult refactor
   - factor_assets: repository errors
   - factor_preprocess: transform errors
   - quant_evaluator: timing errors

4. **Critical missing types** (2 days)
   - NumericalFailure, OverflowOrNonFiniteError
   - TimingContractError
   - IllegalMutationError
   - StaleEvidenceError

5. **Add structured error context** (2 days)
   - field, expected, actual, remediation
   - Error code assignment

**Total Critical Path:** 10-12 days

### High-Value Quick Wins

1. **Consolidate OptionalDependencyMissing** (1 hour)
   - Keep in each package's errors.py
   - Document import pattern

2. **Add NaN/Inf policy** (2 hours)
   - Document in each package's README
   - Add to AI_GUIDE

3. **Add remediation to top 10 errors** (4 hours)
   - Most common user-facing errors
   - High-impact UX improvement

4. **Add 50 numerical failure tests** (1 day)
   - Overflow, underflow, extreme inputs
   - High risk coverage

**Total Quick Wins:** 2 days

### Deferred (Post-Freeze)

- Concurrent failure tests
- Cancellation implementation
- Comprehensive resource exhaustion testing
- Error recovery documentation
- Error analytics/monitoring

---

## 8. Acceptance Criteria

### Pre-Freeze Requirements

- [ ] All 4 packages have contracts/errors.py with full taxonomy
- [ ] 21/21 error types from CONTRACT_FREEZE_DRAFT implemented
- [ ] Zero generic ValueError in public API entry points
- [ ] OptionalDependencyMissing consolidated pattern
- [ ] TimingContractError implemented in QE
- [ ] NumericalFailure/OverflowOrNonFiniteError implemented in QE/FP
- [ ] IllegalMutationError implemented in FO
- [ ] StaleEvidenceError implemented in FA
- [ ] All existing tests pass with new error types
- [ ] Error code assignment complete

### Production-Ready Requirements

- [ ] Pre-freeze requirements met
- [ ] 400+ error test cases
- [ ] Every public API has 3+ negative test cases
- [ ] Numerical overflow/underflow tested
- [ ] Resource exhaustion scenarios tested
- [ ] All errors have remediation guidance
- [ ] NaN/Inf policy documented and enforced
- [ ] Error handling guide published

---

## 9. Risk Assessment

### High Risk

1. **Silent NaN Propagation**
   - Current: Many metrics return NaN instead of raising
   - Risk: Invalid factors reach production without detection
   - Mitigation: Add OverflowOrNonFiniteError gates

2. **Timing Contract Violations**
   - Current: No TimingContractError; use generic ValueError
   - Risk: PIT leakage undetected
   - Mitigation: Implement TimingContractError in QE immediately

3. **String-Based Error Handling (FO)**
   - Current: ValidationResult accumulates strings
   - Risk: Cannot programmatically branch on error types
   - Mitigation: Refactor to typed exceptions

### Medium Risk

4. **OptionalDependencyMissing Fragmentation**
   - Current: Defined in 6+ locations
   - Risk: isinstance() may fail across packages
   - Mitigation: Consolidate to each package's errors.py

5. **Missing Evidence Staleness Detection**
   - Current: No StaleEvidenceError implementation
   - Risk: Use outdated evidence for decisions
   - Mitigation: Implement in FA with timestamp validation

### Low Risk

6. **Missing Cancellation**
   - Current: No CancellationError
   - Risk: Cannot gracefully stop long operations
   - Mitigation: Defer to post-freeze; workaround with timeouts

---

## 10. Conclusion

**Summary:** Error handling implementation is PARTIAL and INCONSISTENT. Only quant_evaluator has a structured error hierarchy aligned with the contract freeze spec. The other three packages rely heavily on generic exceptions and lack critical error types like ExecutionError, TimingContractError, and proper GovernanceError families.

**Readiness:** NOT PRODUCTION READY until Phase 1 (Core Taxonomy) complete.

**Recommended Action:** 
1. Implement contracts/errors.py in FO/FA/FP (Priority 1)
2. Replace ValueError in all public APIs (Priority 1)
3. Add critical missing error types (Priority 1)
4. Add 100+ numerical/resource failure tests (Priority 2)

**Estimated Effort:** 10-12 days for freeze-blocking work, 2-3 weeks for production-ready.

---

## Appendix A: Error Type Implementation Checklist

```
[ ] ContractError (base) - QE:✓ FO:✗ FA:✗ FP:✗
    [ ] SchemaVersionError - QE:✓ FO:✗ FA:✗ FP:✗
    [ ] MissingInputError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] InvalidContractError - QE:✓ FO:✗ FA:✗ FP:✗
    [ ] TimingContractError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] SnapshotMismatchError - QE:✗ FO:✗ FA:✗ FP:✗

[ ] CapabilityError (base) - QE:✓ FO:✗ FA:✗ FP:✗
    [ ] UnsupportedMetricError - QE:✓ FO:✗ FA:✗ FP:✗
    [ ] UnsupportedTransformError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] UnsupportedMutationError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] OptionalDependencyMissing - QE:✓ FO:✓ FA:✓ FP:✓

[ ] DataError/EvidenceError (base) - QE:✓ FO:✗ FA:✗ FP:✗
    [ ] InsufficientObservations - QE:✓ FO:✗ FA:✗ FP:✗
    [ ] InvalidValidityMask - QE:✓ FO:✗ FA:✗ FP:✗
    [ ] MissingLabelError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] EvidenceUnavailableError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] StaleEvidenceError - QE:✗ FO:✗ FA:✗ FP:✗

[ ] ExecutionError (base) - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] NumericalFailure - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] OverflowOrNonFiniteError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] BudgetExceededError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] CancellationError - QE:✗ FO:✗ FA:✗ FP:✗

[ ] GovernanceError (base) - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] IllegalMutationError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] LifecycleConflictError - QE:✗ FO:✗ FA:✓ FP:✗
    [ ] DuplicateIdentityError - QE:✗ FO:✗ FA:✓ FP:✗
    [ ] CollisionError - QE:✗ FO:✗ FA:✗ FP:✗
    [ ] ContractChangeRequired - QE:✗ FO:✗ FA:✗ FP:✗

Implementation: 27/108 cells = 25% complete
```

---

## Appendix B: Key File Locations

**Error Definitions:**
- `/home/shw/quant_projects/quant_evaluator/contracts/errors.py` (GOOD)
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/adapters/*.py` (SCATTERED)
- `/home/shw/quant_projects/factor_assets/contracts/lifecycle.py` (PARTIAL)
- `/home/shw/quant_projects/factor_assets/registry/repository.py` (PARTIAL)
- `/home/shw/quant_projects/factor_preprocess/factor_preprocess/adapters/*.py` (SCATTERED)

**Validation Logic:**
- `/home/shw/quant_projects/quant_evaluator/contracts/factor_batch.py`
- `/home/shw/quant_projects/quant_evaluator/contracts/label_bundle.py`
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/grammar/validation.py`
- `/home/shw/quant_projects/factor_assets/contracts/asset.py`
- `/home/shw/quant_projects/factor_preprocess/contracts/preprocessing.py`

**NaN/Inf Handling:**
- `/home/shw/quant_projects/quant_evaluator/metrics/*.py` (38 instances)
- `/home/shw/quant_projects/quant_evaluator/contracts/factor_batch.py:73-78`

**Test Coverage:**
- `/home/shw/quant_projects/quant_evaluator/tests/**/*` (49 error tests)
- `/home/shw/quant_projects/factor_optimizer/tests/**/*` (34 error tests)
- `/home/shw/quant_projects/factor_assets/tests/**/*` (85 error tests)
- `/home/shw/quant_projects/factor_preprocess/tests/**/*` (61 error tests)

---

**END OF AUDIT REPORT**
