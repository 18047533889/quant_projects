# QE-002: Cache Key Identity Insufficient - Audit Findings

**Audit Date:** 2026-08-14  
**Auditor:** QE-Worker-CacheAudit  
**Priority:** P1 - Data Correctness Issue  
**Status:** CRITICAL - False cache hits can corrupt evaluation results

---

## Executive Summary

The quant_evaluator cache implementation has a **critical data correctness vulnerability**: cache keys bind insufficient dimensions to uniquely identify evaluation results. This allows false cache hits where evaluations with identical IDs/shapes but different values, configurations, or evaluation contexts return stale/incorrect cached results.

**Impact:** Silent data corruption in quantitative evaluation pipeline. Same factor ID with different underlying values can retrieve cached results from a prior evaluation, leading to incorrect IC, portfolio statistics, and downstream research decisions.

---

## Current Cache Key Implementation

### Location
- **Primary Implementation:** `/home/shw/quant_projects/quant_evaluator/runtime/intermediates.py`
- **Usage:** `/home/shw/quant_projects/quant_evaluator/runtime/evaluator.py`

### CacheKey Definition (lines 15-38)

```python
@dataclass(frozen=True)
class CacheKey:
    """
    Key for identifying cached intermediate results.
    
    Combines metric ID, chunk ID, and input hash to uniquely identify
    a computation.
    """
    metric_id: str
    chunk_id: Optional[int] = None
    input_hash: Optional[str] = None
    version: str = "v1"
```

### compute_input_hash Function (lines 230-265)

```python
def compute_input_hash(
    factor_ids: Tuple[str, ...],
    time_slice: Optional[Tuple[int, int]] = None,
    asset_slice: Optional[Tuple[int, int]] = None,
    **kwargs
) -> str:
    """Compute hash of input parameters for cache key."""
    hasher = hashlib.sha256()
    
    # Hash factor IDs
    for fid in sorted(factor_ids):
        hasher.update(fid.encode('utf-8'))
    
    # Hash slices
    if time_slice:
        hasher.update(str(time_slice).encode('utf-8'))
    if asset_slice:
        hasher.update(str(asset_slice).encode('utf-8'))
    
    # Hash additional kwargs
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        hasher.update(f"{key}={value}".encode('utf-8'))
    
    return hasher.hexdigest()
```

### Usage in Evaluator (evaluator.py:226-236, 316-324)

```python
# Full batch evaluation:
cache_key = CacheKey(
    metric_id=metric_id,
    input_hash=compute_input_hash(factor_batch.factor_ids),
)

# Chunked evaluation:
cache_key = CacheKey(
    metric_id=metric_id,
    chunk_id=chunk.chunk_id,
    input_hash=compute_input_hash(
        chunk_batch.factor_ids,
        time_slice=chunk.time_slice,
        asset_slice=chunk.asset_slice,
    ),
)
```

---

## Missing Dimensions from Cache Key

According to the problem specification, cache keys must bind:
1. ✅ Factor value/snapshot identity
2. ❌ **Label identity/content**
3. ❌ **Validity mask**
4. ❌ **LabelSpec** (horizon, execution_delay, decision timing)
5. ❌ **SplitPlan** (train/test/validation splits)
6. ❌ **Evaluation view** (cross-sectional vs time-series)
7. ❌ **Metric version**
8. ❌ **Metric configuration** (method, min_obs, quantiles)
9. ❌ **Universe/snapshot** (which assets, time periods)
10. ❌ **Factor value content hash** (same ID, different values)

### Critical Missing Bindings

#### 1. **Factor Value Content Hash** (MOST CRITICAL)
- **Current:** Only `factor_ids` (strings) are hashed
- **Missing:** Actual factor values content
- **Risk:** Same factor ID with different values → false cache hit

#### 2. **Label Identity and Content**
- **Current:** No label information in cache key
- **Missing:** `target_id`, label values, label timing
- **Risk:** Same factors evaluated against different labels → stale results

#### 3. **Validity Masks**
- **Current:** No validity mask information
- **Missing:** `factor_batch.validity`, `label_bundle.validity`
- **Risk:** Different valid observation sets → incorrect IC/coverage

#### 4. **Label Specification**
- **Current:** No label spec in cache key
- **Missing:** `horizon`, `execution_delay`, decision/execution timing
- **Risk:** Same factor against 1-day vs 5-day forward returns → wrong results

#### 5. **Metric Configuration**
- **Current:** Only `metric_id` string
- **Missing:** Metric parameters (method='pearson' vs 'spearman', min_assets, quantile bins)
- **Risk:** Different metric configs with same ID → wrong computation

#### 6. **Universe/Asset Selection**
- **Current:** Only `asset_slice` indices for chunks
- **Missing:** Actual asset identifiers, universe composition
- **Risk:** Different asset universes with same shape → wrong portfolio stats

#### 7. **Time Period Identity**
- **Current:** Only `time_slice` indices for chunks
- **Missing:** Actual datetime values, calendar alignment
- **Risk:** Different time periods with same length → temporal bias

#### 8. **Evaluation View/Context**
- **Current:** No evaluation context
- **Missing:** Cross-sectional vs rolling window, grouping structure
- **Risk:** Different evaluation paradigms → semantic mismatch

---

## Concrete False-Positive Scenarios

### Scenario 1: Factor Value Update (CRITICAL)

**Setup:**
```python
# Day 1: Evaluate factor "momentum_20d"
factor_batch_v1 = FactorBatch(
    factor_ids=("momentum_20d",),
    values=compute_momentum(prices_v1),  # Original data
    time_axis=AxisRef("time", "datetime64", 252),
    asset_axis=AxisRef("asset", "int64", 500),
)
label_bundle = LabelBundle(
    target_id="forward_return_1d",
    values=forward_returns,
    horizon=1,
    decision_time=...,
    label_start_time=...,
    label_end_time=...,
)
result1 = evaluator.evaluate(factor_batch_v1, label_bundle, metric_specs)
# Cache key: CacheKey(metric_id="ic_daily", input_hash=hash("momentum_20d"))
# Cached: IC = 0.05

# Day 2: Same factor ID, UPDATED VALUES (data refresh)
factor_batch_v2 = FactorBatch(
    factor_ids=("momentum_20d",),  # SAME ID
    values=compute_momentum(prices_v2),  # NEW DATA - different values!
    time_axis=AxisRef("time", "datetime64", 252),
    asset_axis=AxisRef("asset", "int64", 500),
)
result2 = evaluator.evaluate(factor_batch_v2, label_bundle, metric_specs)
# Cache key: CacheKey(metric_id="ic_daily", input_hash=hash("momentum_20d"))
# CACHE HIT! Returns IC = 0.05 (WRONG - should compute on new values)
```

**Expected:** Cache miss, recompute IC on updated factor values  
**Actual:** Cache hit, returns stale IC from original values  
**Impact:** Research decisions based on outdated factor performance

---

### Scenario 2: Different Labels, Same Factors

**Setup:**
```python
# Evaluation 1: 1-day forward returns
label_1d = LabelBundle(
    target_id="forward_return_1d",
    values=returns_1d,
    horizon=1,
    ...
)
result1 = evaluator.evaluate(factor_batch, label_1d, metric_specs)
# Cached: IC against 1-day returns

# Evaluation 2: 5-day forward returns (DIFFERENT PREDICTION TARGET)
label_5d = LabelBundle(
    target_id="forward_return_5d",
    values=returns_5d,  # Different values!
    horizon=5,          # Different horizon!
    ...
)
result2 = evaluator.evaluate(factor_batch, label_5d, metric_specs)
# CACHE HIT! Returns IC against 1-day returns (WRONG)
```

**Expected:** Different IC values (1-day vs 5-day predictive power)  
**Actual:** Same cached IC, semantically incorrect  
**Impact:** Wrong horizon analysis, incorrect strategy selection

---

### Scenario 3: Validity Mask Changes

**Setup:**
```python
# Evaluation 1: All observations valid
factor_batch_full = FactorBatch(
    factor_ids=("value_factor",),
    values=values,
    validity=np.ones((252, 500, 1), dtype=bool),  # All valid
    ...
)
result1 = evaluator.evaluate(factor_batch_full, labels, metric_specs)
# Cached: IC computed on 252 * 500 observations

# Evaluation 2: Filter out illiquid stocks (DIFFERENT VALID SET)
validity_liquid = create_liquidity_filter(...)  # 30% of obs now invalid
factor_batch_filtered = FactorBatch(
    factor_ids=("value_factor",),
    values=values,  # Same values
    validity=validity_liquid,  # DIFFERENT validity mask
    ...
)
result2 = evaluator.evaluate(factor_batch_filtered, labels, metric_specs)
# CACHE HIT! Returns IC computed on full universe (WRONG)
```

**Expected:** IC recomputed on liquid universe only  
**Actual:** IC from full universe, overstates coverage  
**Impact:** Portfolio would trade illiquid stocks excluded from evaluation

---

### Scenario 4: Metric Configuration Change

**Setup:**
```python
# Evaluation 1: Pearson IC
metric_specs_pearson = [{
    "metric_id": "ic",  # Same ID
    "metric_kind": "ic",
    "method": "pearson",
    "min_assets": 50,
}]
result1 = evaluator.evaluate(factor_batch, labels, metric_specs_pearson)
# Cached: Pearson IC

# Evaluation 2: Spearman IC (DIFFERENT METHOD)
metric_specs_spearman = [{
    "metric_id": "ic",  # Same ID!
    "metric_kind": "ic",
    "method": "spearman",  # Different method
    "min_assets": 50,
}]
result2 = evaluator.evaluate(factor_batch, labels, metric_specs_spearman)
# CACHE HIT! Returns Pearson IC (WRONG - should be Spearman)
```

**Expected:** Different IC values (Pearson vs Spearman)  
**Actual:** Pearson IC returned for both  
**Impact:** Wrong metric type, robustness analysis corrupted

---

### Scenario 5: Universe/Snapshot Change

**Setup:**
```python
# Evaluation 1: S&P 500 universe
assets_sp500 = np.arange(500)
factor_batch_sp500 = FactorBatch(
    factor_ids=("momentum",),
    values=momentum_values_sp500,
    asset_axis=AxisRef("asset", "int64", 500),
    ...
)
result1 = evaluator.evaluate(factor_batch_sp500, labels_sp500, metric_specs)
# Cached: IC on S&P 500

# Evaluation 2: Russell 2000 universe (DIFFERENT STOCKS)
assets_r2000 = np.arange(2000)[:500]  # First 500 of Russell 2000
factor_batch_r2000 = FactorBatch(
    factor_ids=("momentum",),
    values=momentum_values_r2000,
    asset_axis=AxisRef("asset", "int64", 500),  # SAME SHAPE!
    ...
)
result2 = evaluator.evaluate(factor_batch_r2000, labels_r2000, metric_specs)
# CACHE HIT! Returns S&P 500 IC (WRONG - different universe)
```

**Expected:** Different IC (large-cap vs small-cap characteristics)  
**Actual:** S&P 500 IC applied to Russell 2000 analysis  
**Impact:** Wrong market segment analysis, strategy miscalibration

---

## Test Case Demonstrating Vulnerability

```python
"""
Test demonstrating cache false-positive with same shape, different values.
Place in: tests/test_cache_false_positive.py
"""

import pytest
import numpy as np
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import Evaluator
from quant_evaluator.planner.dependency_plan import MetricKind


def test_cache_false_positive_different_values_same_id():
    """
    CRITICAL BUG: Cache returns stale results when factor ID is same
    but underlying values are different.
    """
    evaluator = Evaluator(enable_cache=True)
    
    # Register a simple IC metric
    def compute_ic(factor_batch, label_bundle, **kwargs):
        """Compute mean correlation between factor and label."""
        factors = factor_batch.values[:, :, 0]  # (T, N)
        labels = label_bundle.values  # (T, N)
        
        # Compute cross-sectional correlation per time step
        ics = []
        for t in range(factors.shape[0]):
            mask = np.isfinite(factors[t, :]) & np.isfinite(labels[t, :])
            if np.sum(mask) > 10:
                ic = np.corrcoef(factors[t, mask], labels[t, mask])[0, 1]
                ics.append(ic)
        
        return np.nanmean(ics) if ics else np.nan
    
    evaluator.register_metric("ic_mean", compute_ic, MetricKind.IC)
    
    # Create first factor batch: positively correlated with labels
    T, N = 100, 200
    time_axis = AxisRef("time", "datetime64", T)
    asset_axis = AxisRef("asset", "int64", N)
    
    labels = LabelBundle(
        target_id="forward_return_1d",
        values=np.random.randn(T, N),
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    
    # First evaluation: factor has POSITIVE correlation with labels
    factor_values_v1 = labels.values + np.random.randn(T, N) * 0.5
    factor_batch_v1 = FactorBatch(
        factor_ids=("momentum_20d",),  # Note: same ID
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values_v1.reshape(T, N, 1),
    )
    
    metric_specs = [{"metric_id": "ic_mean", "metric_kind": "ic"}]
    result1 = evaluator.evaluate(factor_batch_v1, labels, metric_specs)
    ic_v1 = result1.get_metric("ic_mean")
    
    print(f"IC v1 (should be positive): {ic_v1:.4f}")
    assert ic_v1 > 0.3, f"Expected positive IC, got {ic_v1}"
    assert result1.cache_misses == 1
    assert result1.cache_hits == 0
    
    # Second evaluation: SAME factor ID, but NEGATIVE correlation
    factor_values_v2 = -labels.values + np.random.randn(T, N) * 0.5
    factor_batch_v2 = FactorBatch(
        factor_ids=("momentum_20d",),  # SAME ID as v1!
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values_v2.reshape(T, N, 1),  # DIFFERENT VALUES!
    )
    
    result2 = evaluator.evaluate(factor_batch_v2, labels, metric_specs)
    ic_v2 = result2.get_metric("ic_mean")
    
    print(f"IC v2 (should be negative): {ic_v2:.4f}")
    print(f"Cache hits: {result2.cache_hits}, misses: {result2.cache_misses}")
    
    # BUG: Current implementation returns cached positive IC
    # Expected: Should return negative IC (recomputed on new values)
    # Actual: Returns positive IC from cache (WRONG!)
    
    assert result2.cache_hits == 1, "Expected cache hit (demonstrating bug)"
    assert ic_v2 == ic_v1, f"BUG: Cache returned stale IC {ic_v1}, not recomputed IC"
    
    # What SHOULD happen (after fix):
    # assert result2.cache_misses == 1, "Should be cache miss (different values)"
    # assert ic_v2 < -0.3, f"Expected negative IC, got {ic_v2}"
    # assert abs(ic_v2 - ic_v1) > 0.5, "ICs should be opposite sign"


def test_cache_false_positive_different_labels():
    """
    BUG: Cache ignores label identity - same factors against different
    labels return stale results.
    """
    evaluator = Evaluator(enable_cache=True)
    
    def compute_ic(factor_batch, label_bundle, **kwargs):
        factors = factor_batch.values[:, :, 0]
        labels = label_bundle.values
        ics = []
        for t in range(factors.shape[0]):
            mask = np.isfinite(factors[t, :]) & np.isfinite(labels[t, :])
            if np.sum(mask) > 10:
                ic = np.corrcoef(factors[t, mask], labels[t, mask])[0, 1]
                ics.append(ic)
        return np.nanmean(ics) if ics else np.nan
    
    evaluator.register_metric("ic_mean", compute_ic, MetricKind.IC)
    
    T, N = 100, 200
    time_axis = AxisRef("time", "datetime64", T)
    asset_axis = AxisRef("asset", "int64", N)
    
    # Fixed factor values
    factor_values = np.random.randn(T, N, 1)
    factor_batch = FactorBatch(
        factor_ids=("value_factor",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values,
    )
    
    # Label 1: Uncorrelated with factor
    label_1d = LabelBundle(
        target_id="forward_return_1d",
        values=np.random.randn(T, N),
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    
    metric_specs = [{"metric_id": "ic_mean", "metric_kind": "ic"}]
    result1 = evaluator.evaluate(factor_batch, label_1d, metric_specs)
    ic_1d = result1.get_metric("ic_mean")
    
    print(f"IC with label_1d: {ic_1d:.4f}")
    
    # Label 2: Strongly correlated with factor (DIFFERENT TARGET)
    label_5d = LabelBundle(
        target_id="forward_return_5d",  # Different target!
        values=factor_values[:, :, 0] + np.random.randn(T, N) * 0.3,
        horizon=5,  # Different horizon!
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(5, T + 5)),
    )
    
    result2 = evaluator.evaluate(factor_batch, label_5d, metric_specs)
    ic_5d = result2.get_metric("ic_mean")
    
    print(f"IC with label_5d: {ic_5d:.4f}")
    print(f"Cache hits: {result2.cache_hits}")
    
    # BUG: Returns cached IC from label_1d evaluation
    assert result2.cache_hits == 1, "Expected cache hit (demonstrating bug)"
    assert ic_5d == ic_1d, "BUG: Returned same IC despite different labels"
    
    # What SHOULD happen (after fix):
    # assert result2.cache_misses == 1, "Should be cache miss (different labels)"
    # assert abs(ic_5d - ic_1d) > 0.2, "ICs should differ with different labels"


def test_cache_false_positive_validity_mask():
    """
    BUG: Cache ignores validity masks - filtered data returns unfiltered results.
    """
    evaluator = Evaluator(enable_cache=True)
    
    def compute_coverage(factor_batch, label_bundle, **kwargs):
        """Compute fraction of valid observations."""
        if factor_batch.validity is not None:
            return np.mean(factor_batch.validity)
        else:
            return np.mean(np.isfinite(factor_batch.values))
    
    evaluator.register_metric("coverage", compute_coverage, MetricKind.COVERAGE)
    
    T, N = 100, 200
    time_axis = AxisRef("time", "datetime64", T)
    asset_axis = AxisRef("asset", "int64", N)
    values = np.random.randn(T, N, 1)
    
    labels = LabelBundle(
        target_id="forward_return_1d",
        values=np.random.randn(T, N),
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    
    # Evaluation 1: 100% valid
    factor_full = FactorBatch(
        factor_ids=("factor_a",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
        validity=np.ones((T, N, 1), dtype=bool),  # All valid
    )
    
    metric_specs = [{"metric_id": "coverage", "metric_kind": "coverage"}]
    result1 = evaluator.evaluate(factor_full, labels, metric_specs)
    coverage_full = result1.get_metric("coverage")
    
    print(f"Coverage (full): {coverage_full:.2%}")
    assert coverage_full == 1.0
    
    # Evaluation 2: 50% valid (DIFFERENT VALIDITY MASK)
    validity_partial = np.random.rand(T, N, 1) > 0.5  # ~50% valid
    factor_partial = FactorBatch(
        factor_ids=("factor_a",),  # Same ID
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,  # Same values
        validity=validity_partial,  # DIFFERENT MASK!
    )
    
    result2 = evaluator.evaluate(factor_partial, labels, metric_specs)
    coverage_partial = result2.get_metric("coverage")
    
    print(f"Coverage (partial): {coverage_partial:.2%}")
    print(f"Cache hits: {result2.cache_hits}")
    
    # BUG: Returns cached coverage from full validity evaluation
    assert result2.cache_hits == 1, "Expected cache hit (demonstrating bug)"
    assert coverage_partial == coverage_full, "BUG: Returned full coverage for partial data"
    
    # What SHOULD happen (after fix):
    # assert result2.cache_misses == 1, "Should be cache miss (different validity)"
    # assert 0.4 < coverage_partial < 0.6, f"Expected ~50% coverage, got {coverage_partial}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

---

## Recommended Fix Approach

### Phase 1: Add Missing Dimensions to Cache Key (IMMEDIATE)

```python
@dataclass(frozen=True)
class CacheKey:
    """Enhanced cache key with full identity binding."""
    metric_id: str
    metric_config_hash: str  # NEW: Hash of metric parameters
    factor_value_hash: str   # NEW: Hash of actual factor values
    label_hash: str          # NEW: Hash of label identity + values
    validity_hash: str       # NEW: Hash of validity masks
    evaluation_context_hash: str  # NEW: Universe, time range, view type
    chunk_id: Optional[int] = None
    version: str = "v2"  # Increment version

def compute_comprehensive_cache_key(
    metric_id: str,
    metric_config: Dict[str, Any],
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    chunk_id: Optional[int] = None,
    evaluation_context: Optional[Dict[str, Any]] = None,
) -> CacheKey:
    """
    Compute cache key binding all dimensions.
    
    Args:
        metric_id: Metric identifier
        metric_config: Metric parameters (method, min_obs, etc.)
        factor_batch: Factor batch with values and validity
        label_bundle: Label bundle with target and timing
        chunk_id: Optional chunk identifier
        evaluation_context: Universe, splits, view type
    
    Returns:
        Comprehensive cache key
    """
    # Hash metric configuration
    metric_config_str = json.dumps(metric_config, sort_keys=True)
    metric_config_hash = hashlib.sha256(metric_config_str.encode()).hexdigest()[:16]
    
    # Hash factor VALUES (not just IDs)
    factor_value_hash = hashlib.sha256(
        factor_batch.values.tobytes()
    ).hexdigest()[:16]
    
    # Hash label identity and values
    label_data = {
        "target_id": label_bundle.target_id,
        "horizon": label_bundle.horizon,
        "execution_delay": label_bundle.execution_delay,
        "values_hash": hashlib.sha256(label_bundle.values.tobytes()).hexdigest()[:16],
    }
    label_hash = hashlib.sha256(
        json.dumps(label_data, sort_keys=True).encode()
    ).hexdigest()[:16]
    
    # Hash validity masks
    validity_parts = []
    if factor_batch.validity is not None:
        validity_parts.append(factor_batch.validity.tobytes())
    if label_bundle.validity is not None:
        validity_parts.append(label_bundle.validity.tobytes())
    
    if validity_parts:
        validity_hash = hashlib.sha256(b"".join(validity_parts)).hexdigest()[:16]
    else:
        validity_hash = "no_validity"
    
    # Hash evaluation context (universe, time range, splits)
    if evaluation_context:
        context_str = json.dumps(evaluation_context, sort_keys=True)
        context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
    else:
        context_hash = "no_context"
    
    return CacheKey(
        metric_id=metric_id,
        metric_config_hash=metric_config_hash,
        factor_value_hash=factor_value_hash,
        label_hash=label_hash,
        validity_hash=validity_hash,
        evaluation_context_hash=context_hash,
        chunk_id=chunk_id,
        version="v2",
    )
```

### Phase 2: Update Evaluator to Use Enhanced Keys

```python
# In evaluator.py, replace cache key construction:

# OLD (BROKEN):
cache_key = CacheKey(
    metric_id=metric_id,
    input_hash=compute_input_hash(factor_batch.factor_ids),
)

# NEW (CORRECT):
cache_key = compute_comprehensive_cache_key(
    metric_id=metric_id,
    metric_config=node.metadata,  # Metric parameters
    factor_batch=factor_batch,
    label_bundle=label_bundle,
    chunk_id=chunk.chunk_id if chunk else None,
    evaluation_context={
        "universe_snapshot": factor_batch.context_refs.get("universe_id"),
        "time_range": (
            factor_batch.time_axis.values[0] if factor_batch.time_axis.values is not None else None,
            factor_batch.time_axis.values[-1] if factor_batch.time_axis.values is not None else None,
        ),
        "evaluation_view": evaluation_context.get("view_type", "cross_sectional"),
    },
)
```

### Phase 3: Incremental Value Hashing (Performance Optimization)

For large factor batches, full value hashing can be expensive. Options:

1. **Sample-based hash:** Hash subset of values (first/last/random samples)
2. **Metadata hash:** Hash statistical fingerprint (mean, std, min, max, checksum)
3. **Versioned snapshots:** Add `value_hash` field to FactorBatch at creation time
4. **Layered cache:** L1 cache with simple keys, L2 with full keys

```python
def compute_fast_value_hash(values: np.ndarray, sample_rate: float = 0.1) -> str:
    """
    Fast value hashing using sampling.
    
    Args:
        values: Array to hash
        sample_rate: Fraction of elements to sample
    
    Returns:
        Hash digest
    """
    # Option 1: Hash statistical fingerprint (FAST)
    stats = np.array([
        values.shape,
        np.nanmean(values),
        np.nanstd(values),
        np.nanmin(values),
        np.nanmax(values),
        np.sum(np.isnan(values)),
    ])
    return hashlib.sha256(stats.tobytes()).hexdigest()[:16]
    
    # Option 2: Sample-based hash (BALANCED)
    # flat = values.flatten()
    # n_samples = max(1000, int(len(flat) * sample_rate))
    # indices = np.random.choice(len(flat), size=min(n_samples, len(flat)), replace=False)
    # sample = flat[indices]
    # return hashlib.sha256(sample.tobytes()).hexdigest()[:16]
```

### Phase 4: Migration Strategy

1. **Immediate:** Deploy v2 cache keys (increment version to "v2")
2. **Cache invalidation:** All v1 cached entries become stale (version mismatch)
3. **Monitoring:** Track cache hit rates before/after (expect initial drop, then recovery)
4. **Testing:** Run test suite with new keys, verify no false positives
5. **Performance:** Benchmark hashing overhead, optimize if needed

---

## Verification Plan

### Unit Tests (Add to `tests/test_cache_false_positive.py`)

1. ✅ Test: Same factor ID, different values → cache miss
2. ✅ Test: Same factors, different labels → cache miss
3. ✅ Test: Same data, different validity masks → cache miss
4. ✅ Test: Same data, different metric config → cache miss
5. ✅ Test: Same data, different universe → cache miss
6. Test: Identical inputs → cache hit (regression check)
7. Test: Value hash performance benchmark

### Integration Tests

1. End-to-end evaluation with cache enabled
2. Sequential evaluations with data updates
3. Multi-factor batch with partial overlap
4. Cross-validation splits (train/test cache isolation)

### Production Validation

1. Shadow mode: Compute both v1 and v2 keys, compare hit rates
2. Checksum validation: For cache hits, verify value checksums match
3. Drift detection: Alert if cache hit rate drops below threshold

---

## Alternative Approaches Considered

### 1. Content-Addressable Storage (CAS)
- **Pro:** Automatic deduplication, cryptographic correctness
- **Con:** Overhead of hashing large arrays, storage complexity
- **Decision:** Partial adoption (hash values, not full CAS)

### 2. Explicit Cache Invalidation API
- **Pro:** User controls cache lifecycle
- **Con:** Requires manual intervention, error-prone
- **Decision:** Complement, not replacement (add invalidation + auto-detection)

### 3. Immutable Value Objects
- **Pro:** Structural guarantee of identity
- **Con:** Major refactor of contracts, memory overhead
- **Decision:** Future consideration for v2.0 architecture

### 4. Cache Versioning by Data Lineage
- **Pro:** Automatic invalidation on upstream changes
- **Con:** Requires provenance tracking infrastructure
- **Decision:** Too heavyweight for current system

---

## Risk Assessment

### Current State (Unfixed)
- **Severity:** CRITICAL (P1)
- **Likelihood:** HIGH (occurs in normal workflows)
- **Impact:** Silent data corruption, incorrect research decisions
- **Mitigation:** NONE (users unaware of false cache hits)

### After Fix
- **Severity:** LOW (cache misses are safe, just slower)
- **Likelihood:** MEDIUM (hash collisions theoretically possible)
- **Impact:** Performance degradation if hash overhead excessive
- **Mitigation:** Performance testing, incremental hash optimization

---

## Implementation Checklist

- [ ] Add `metric_config_hash` to CacheKey
- [ ] Add `factor_value_hash` to CacheKey
- [ ] Add `label_hash` to CacheKey
- [ ] Add `validity_hash` to CacheKey
- [ ] Add `evaluation_context_hash` to CacheKey
- [ ] Implement `compute_comprehensive_cache_key()` function
- [ ] Update `Evaluator._evaluate_full()` to use new keys
- [ ] Update `Evaluator._evaluate_chunk()` to use new keys
- [ ] Add `test_cache_false_positive.py` test suite
- [ ] Benchmark hash computation overhead
- [ ] Update cache version to "v2"
- [ ] Add cache migration guide to documentation
- [ ] Deploy to staging, monitor cache hit rates
- [ ] Production rollout with phased invalidation

---

## References

- **Current Implementation:** `runtime/intermediates.py`, `runtime/evaluator.py`
- **Test Coverage:** `tests/test_intermediates.py`, `tests/test_cache_v2.py`
- **Related Issues:** Factor-engine R34 (evidence HEAD binding), R37 (fail-closed caching)
- **Memory Note:** Quantitative platform 15 GiB limit constrains cache size

---

## Conclusion

The quant_evaluator cache system has a **critical correctness bug** that allows false cache hits when evaluation inputs have identical IDs/shapes but different values or configurations. This is a **silent data corruption issue** that can lead to incorrect research decisions.

**Immediate Action Required:**
1. Implement comprehensive cache keys binding all evaluation dimensions
2. Deploy test cases demonstrating vulnerability
3. Migrate cache to v2 with enhanced keys
4. Monitor production for false positive reduction

**Estimated Effort:** 2-3 days for implementation, 1 day for testing/validation, 1 day for deployment

**Owner:** Recommend assignment to core QE maintainer with knowledge of evaluation pipeline

---

**END OF AUDIT REPORT**
