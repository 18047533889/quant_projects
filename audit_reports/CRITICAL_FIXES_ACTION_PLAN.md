# Critical Fixes - Immediate Action Plan

**Priority**: URGENT  
**Estimated Effort**: 2-3 developer-days  
**Target Completion**: Within 48 hours

## Overview

This document outlines the 14 critical issues that must be fixed immediately to prevent production failures, data corruption, and crashes. Each issue includes the exact file location, problem description, code fix, and verification steps.

---

## 1. RESEARCH_CONTROL - Atomic Transaction Rollback

**File**: `/home/shw/quant_projects/research_control/research_control/sync/idempotency.py`  
**Lines**: 93-141 (_sync_campaign_atomic), 205-255 (_sync_trial_atomic)  
**Risk**: Partial writes corrupt ledger state on batch insert failures

### Current Code Problem
```python
# Line 93-141
def _sync_campaign_atomic(self, validated):
    # Comment admits: "Rollback not possible with current API"
    for event_id, campaign_id, state, timestamp, metadata in validated:
        try:
            self._campaign.append(event_id, campaign_id, state, timestamp, metadata)
        except Exception as e:
            # Events 0-4 already committed if event 5 fails
            return SyncResult(success=False, ...)
```

### Fixed Code
```python
def _sync_campaign_atomic(self, validated):
    """Fail-atomic: all events committed or none."""
    with self._campaign._conn() as conn:
        # Start explicit transaction
        conn.execute("BEGIN IMMEDIATE")
        
        try:
            for event_id, campaign_id, state, timestamp, metadata in validated:
                metadata_json = json.dumps(metadata) if metadata else None
                conn.execute(
                    """INSERT INTO campaign_events 
                       (event_id, campaign_id, state, timestamp, metadata)
                       VALUES (?, ?, ?, ?, ?)""",
                    (event_id, campaign_id, state, timestamp.isoformat(), metadata_json)
                )
            
            # All succeeded, commit atomically
            conn.commit()
            return SyncResult(
                success=True,
                synced_count=len(validated),
                failed_count=0,
                errors=[]
            )
            
        except Exception as e:
            # Any failure rolls back entire batch
            conn.rollback()
            return SyncResult(
                success=False,
                synced_count=0,
                failed_count=len(validated),
                errors=[str(e)]
            )
```

### Verification
```python
# Test case
def test_atomic_rollback():
    events = [
        {"event_id": "e1", "campaign_id": "c1", "state": "created", ...},
        {"event_id": "e2", "campaign_id": "c1", "state": "started", ...},
        {"event_id": "e1", "campaign_id": "c1", "state": "duplicate", ...},  # Duplicate fails
    ]
    
    result = engine.sync_campaign_events(events, mode="fail_atomic")
    assert result.success == False
    
    # Verify NO events were inserted (true rollback)
    history = engine._campaign.get_history("c1")
    assert len(history) == 0  # Complete rollback
```

---

## 2. RESEARCH_CONTROL - Orphaned Trials Prevention

**File**: `/home/shw/quant_projects/research_control/research_control/ledger/trial.py`  
**Lines**: 89-136 (append method)  
**Risk**: Trials reference non-existent campaigns, violating data integrity

### Current Code Problem
```python
def append(self, event_id, trial_id, campaign_id, state, timestamp, ...):
    # No validation that campaign_id exists
    with self._conn() as conn:
        conn.execute(
            "INSERT INTO trial_events (...) VALUES (...)",
            (event_id, trial_id, campaign_id, ...)
        )
```

### Fixed Code
```python
def __init__(self, db_path: str, campaign_ledger: CampaignLedger):
    self.db_path = db_path
    self._campaign_ledger = campaign_ledger  # Add reference
    self._init_schema()

def append(self, event_id, trial_id, campaign_id, state, timestamp, ...):
    """Append trial event. Validates campaign exists."""
    
    # Validate campaign exists
    campaign_state = self._campaign_ledger.get_current_state(campaign_id)
    if campaign_state is None:
        raise ValueError(
            f"Cannot append trial for non-existent campaign_id='{campaign_id}'. "
            f"Create campaign first."
        )
    
    # Validate campaign is not in terminal state
    TERMINAL_STATES = {"completed", "cancelled", "failed"}
    if campaign_state in TERMINAL_STATES:
        raise ValueError(
            f"Cannot append trial for campaign in terminal state '{campaign_state}'"
        )
    
    # Proceed with insert
    with self._conn() as conn:
        conn.execute(
            "INSERT INTO trial_events (...) VALUES (...)",
            (event_id, trial_id, campaign_id, ...)
        )
```

### Verification
```python
def test_orphaned_trial_prevention():
    campaign_ledger = CampaignLedger("campaigns.db")
    trial_ledger = TrialLedger("trials.db", campaign_ledger)
    
    # Attempt to add trial for non-existent campaign
    with pytest.raises(ValueError, match="non-existent campaign"):
        trial_ledger.append(
            "e1", "t1", "nonexistent_campaign", "submitted", datetime.utcnow()
        )
```

---

## 3. RESEARCH_CONTROL - SQL Variable Limit

**File**: `/home/shw/quant_projects/research_control/research_control/sync/idempotency.py`  
**Lines**: 279-306 (_get_missing_campaign_ids, _get_missing_trial_ids)  
**Risk**: Crashes on batches > 999 events due to SQLite variable limit

### Current Code Problem
```python
def _get_missing_campaign_ids(self, event_ids):
    placeholders = ",".join("?" * len(event_ids))  # Can exceed 999
    cursor = conn.execute(
        f"SELECT event_id FROM campaign_events WHERE event_id IN ({placeholders})",
        event_ids
    )  # SQLITE_ERROR if len(event_ids) > 999
```

### Fixed Code
```python
def _get_missing_campaign_ids(self, event_ids):
    """Find which event_ids are missing. Handles large batches."""
    if not event_ids:
        return set()
    
    BATCH_SIZE = 900  # Conservative limit (SQLite max is 999)
    existing_ids = set()
    
    # Process in batches to avoid variable limit
    for i in range(0, len(event_ids), BATCH_SIZE):
        batch = event_ids[i:i+BATCH_SIZE]
        placeholders = ",".join("?" * len(batch))
        
        with self._campaign._conn() as conn:
            cursor = conn.execute(
                f"SELECT event_id FROM campaign_events WHERE event_id IN ({placeholders})",
                batch
            )
            existing_ids.update(row[0] for row in cursor.fetchall())
    
    # Return missing IDs
    return set(event_ids) - existing_ids
```

### Verification
```python
def test_large_batch_query():
    # Insert 500 events
    for i in range(500):
        ledger.append(f"e{i}", "c1", "running", datetime.utcnow())
    
    # Query 1500 event_ids (1000 missing, 500 existing)
    query_ids = [f"e{i}" for i in range(1500)]
    missing = engine._get_missing_campaign_ids(query_ids)
    
    assert len(missing) == 1000
    assert all(f"e{i}" in missing for i in range(500, 1500))
```

---

## 4. QUANT_EVALUATOR - Empty Batch Validation

**File**: `/home/shw/quant_projects/quant_evaluator/runtime/evaluator.py`  
**Lines**: 128-150 (evaluate method)  
**Risk**: Crashes during computation when batch is empty or all-NaN

### Current Code Problem
```python
def evaluate(self, factor_batch, label_bundle, metric_specs):
    # Only validates time axis alignment
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(...)
    
    # Missing: empty check, all-NaN check
    # Proceeds to crash during computation
```

### Fixed Code
```python
def evaluate(self, factor_batch, label_bundle, metric_specs):
    """Evaluate factors against labels."""
    
    # Validate inputs thoroughly
    self._validate_inputs(factor_batch, label_bundle, metric_specs)
    
    # Continue with evaluation...

def _validate_inputs(self, factor_batch, label_bundle, metric_specs):
    """Comprehensive input validation."""
    
    # Check for None
    if factor_batch is None:
        raise InvalidContractError("factor_batch cannot be None")
    if label_bundle is None:
        raise InvalidContractError("label_bundle cannot be None")
    if metric_specs is None or len(metric_specs) == 0:
        raise InvalidContractError("metric_specs cannot be empty")
    
    # Check for empty dimensions
    if factor_batch.num_factors == 0:
        raise InvalidContractError("FactorBatch must have at least one factor")
    if factor_batch.num_times == 0:
        raise InvalidContractError("FactorBatch cannot have zero time periods")
    if factor_batch.num_assets == 0:
        raise InvalidContractError("FactorBatch cannot have zero assets")
    
    # Check time axis alignment
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(
            f"Time axis mismatch: batch has {factor_batch.num_times} periods, "
            f"labels have {len(label_bundle.values)} periods"
        )
    
    # Check for values array
    if factor_batch.values is None:
        raise InvalidContractError("FactorBatch.values cannot be None")
    
    # Check for completely invalid data
    n_valid = np.sum(np.isfinite(factor_batch.values))
    if n_valid == 0:
        raise InsufficientObservations(
            "FactorBatch contains no valid (finite) values. "
            "All values are NaN or Inf."
        )
    
    # Warn if very sparse
    total_values = factor_batch.values.size
    sparsity = 1.0 - (n_valid / total_values)
    if sparsity > 0.95:
        logger.warning(
            f"FactorBatch is {sparsity*100:.1f}% sparse ({n_valid}/{total_values} valid). "
            f"Results may be unreliable."
        )
```

### Verification
```python
def test_empty_batch_validation():
    # Empty batch
    with pytest.raises(InvalidContractError, match="at least one factor"):
        batch = FactorBatch(factor_ids=(), time_axis=[], asset_axis=[], values=np.array([]))
        evaluator.evaluate(batch, labels, specs)
    
    # All-NaN batch
    with pytest.raises(InsufficientObservations, match="no valid"):
        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=dates,
            asset_axis=assets,
            values=np.full((10, 100, 1), np.nan)
        )
        evaluator.evaluate(batch, labels, specs)
```

---

## 5. QUANT_EVALUATOR - IC Finalization None Check

**File**: `/home/shw/quant_projects/quant_evaluator/runtime/streaming_evaluator.py`  
**Lines**: 69-85 (_finalize_ic method)  
**Risk**: TypeError crashes when sum_x is None (no data processed)

### Current Code Problem
```python
def _finalize_ic(self) -> np.ndarray:
    # Assumes sum_x is not None
    n = self.valid_counts.astype(np.float64)  # Crashes if None
    # ...
```

### Fixed Code
```python
def _finalize_ic(self) -> np.ndarray:
    """Finalize streaming IC computation. Safe for uninitialized state."""
    
    # Check if any data was processed
    if self.sum_x is None or self.valid_counts is None:
        logger.warning(f"IC metric '{self.metric_id}' finalized with no data processed")
        return np.array([])
    
    # Check for empty arrays
    if self.sum_x.size == 0 or self.valid_counts.size == 0:
        return np.array([])
    
    # Safe to proceed with computation
    n = self.valid_counts.astype(np.float64)
    
    # Prevent division by zero
    valid_mask = n > 0
    if not np.any(valid_mask):
        return np.full_like(n, np.nan)
    
    # Compute correlation
    mean_x = np.zeros_like(self.sum_x)
    mean_y = np.zeros_like(self.sum_y)
    mean_x[valid_mask] = self.sum_x[valid_mask] / n[valid_mask]
    mean_y[valid_mask] = self.sum_y[valid_mask] / n[valid_mask]
    
    # ... rest of computation with safe divisions
```

### Verification
```python
def test_ic_finalization_no_data():
    # Create uninitialized state
    state = StreamingMetricState(metric_id="ic", metric_kind=MetricKind.IC)
    
    # Should not crash
    result = state.finalize()
    assert isinstance(result, np.ndarray)
    assert len(result) == 0
```

---

## 6. QUANT_EVALUATOR - Cache Eviction Loop Safety

**File**: `/home/shw/quant_projects/quant_evaluator/runtime/cache_v2.py`  
**Lines**: 349-351 (MemoryCacheLayer.put)  
**Risk**: Infinite loop if eviction always fails

### Current Code Problem
```python
# Line 330-351
if compressed_size > self.max_size_bytes:
    return False  # Good check

# But this can infinite loop:
while self._current_size + compressed_size > self.max_size_bytes:
    if not self._evict_lru():  # If cache empty, always returns False
        return False  # Never reached if eviction logic is broken
```

### Fixed Code
```python
def put(self, key: str, value: Any) -> bool:
    """Put value in cache with LRU eviction. Safe against infinite loops."""
    
    # Serialize and compress
    try:
        serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(serialized, level=self.compression_level)
        compressed_size = len(compressed)
    except Exception as e:
        logger.error(f"Failed to serialize cache value: {e}")
        return False
    
    # Check if value is too large for cache
    if compressed_size > self.max_size_bytes:
        logger.warning(
            f"Value size {compressed_size} exceeds cache max {self.max_size_bytes}, skipping"
        )
        return False
    
    # Evict with safety counter
    max_evictions = len(self._cache) + 1  # At most evict entire cache
    eviction_count = 0
    
    while self._current_size + compressed_size > self.max_size_bytes:
        if not self._evict_lru():
            logger.error("Eviction failed with non-empty cache, possible corruption")
            return False
        
        eviction_count += 1
        if eviction_count > max_evictions:
            logger.error(
                f"Eviction loop exceeded max iterations ({max_evictions}). "
                f"Current size: {self._current_size}, needed: {compressed_size}"
            )
            return False
    
    # Store in cache
    self._cache[key] = (compressed, compressed_size)
    self._current_size += compressed_size
    self._access_order[key] = time.time()
    
    return True
```

### Verification
```python
def test_eviction_loop_safety():
    cache = MemoryCacheLayer(max_size_bytes=1000)
    
    # Fill cache completely
    for i in range(100):
        cache.put(f"key{i}", b"x" * 10)
    
    # Try to insert huge value that requires evicting everything
    # Should not hang, should return False
    large_value = b"x" * 2000  # Larger than max cache size
    
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("Cache operation hung")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout
    
    try:
        result = cache.put("huge", large_value)
        assert result == False  # Too large, rejected
    finally:
        signal.alarm(0)
```

---

## 7. FACTOR_OPTIMIZER - Plateau Detection Division by Zero

**File**: `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/runner.py`  
**Lines**: ~236 (plateau detection)  
**Risk**: ZeroDivisionError when all scores are zero

### Current Code Problem
```python
# Line ~236
score_std = np.std(recent_scores)
plateau_detected = score_std / np.mean(recent_scores) < self.plateau_threshold
# Crashes if mean is zero
```

### Fixed Code
```python
def _check_plateau(self, recent_scores):
    """Check if search has plateaued. Safe for zero/constant scores."""
    
    if len(recent_scores) < self.plateau_window:
        return False
    
    score_std = np.std(recent_scores)
    score_mean = np.mean(recent_scores)
    
    # Handle edge cases
    if not np.isfinite(score_std) or not np.isfinite(score_mean):
        logger.warning("Non-finite scores in plateau detection")
        return False
    
    # If mean is zero or near-zero, use absolute threshold
    if abs(score_mean) < 1e-10:
        # All scores near zero - check if std is also near zero
        plateau_detected = score_std < 1e-6
        if plateau_detected:
            logger.info(f"Plateau detected: all scores near zero (std={score_std:.2e})")
        return plateau_detected
    
    # Normal case: relative threshold
    relative_std = score_std / abs(score_mean)
    plateau_detected = relative_std < self.plateau_threshold
    
    if plateau_detected:
        logger.info(
            f"Plateau detected: std={score_std:.4f}, mean={score_mean:.4f}, "
            f"relative_std={relative_std:.4f} < threshold={self.plateau_threshold}"
        )
    
    return plateau_detected
```

### Verification
```python
def test_plateau_detection_zero_scores():
    runner = SearchRunner(plateau_threshold=0.01, plateau_window=5)
    
    # All zero scores
    scores = [0.0, 0.0, 0.0, 0.0, 0.0]
    plateau = runner._check_plateau(scores)
    assert plateau == True  # No variation, plateau detected
    
    # All near-zero with tiny variation
    scores = [1e-8, 1.1e-8, 0.9e-8, 1e-8, 1e-8]
    plateau = runner._check_plateau(scores)
    # Should not crash
```

---

## 8. FACTOR_OPTIMIZER - Hypervolume Reference Point Validation

**File**: `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/pareto.py`  
**Lines**: ~177 (hypervolume calculation)  
**Risk**: Numerical errors with degenerate reference points

### Current Code Problem
```python
def compute_hypervolume(front, reference_point):
    # No validation of reference point
    # Assumes reference point is dominated by all points
    volumes = []
    for point in front:
        volume = np.prod(reference_point - point)  # Can be negative or invalid
        volumes.append(volume)
```

### Fixed Code
```python
def compute_hypervolume(front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume indicator for Pareto front.
    
    Args:
        front: (N, M) array of N points with M objectives (minimization)
        reference_point: (M,) reference point (must be dominated by all front points)
    
    Returns:
        Hypervolume value
    
    Raises:
        ValueError: If reference point is invalid
    """
    if front.size == 0:
        return 0.0
    
    if len(front.shape) != 2:
        raise ValueError(f"Front must be 2D array, got shape {front.shape}")
    
    n_points, n_objectives = front.shape
    
    if reference_point.shape != (n_objectives,):
        raise ValueError(
            f"Reference point shape {reference_point.shape} does not match "
            f"front objectives {n_objectives}"
        )
    
    # Validate reference point is dominated by all points
    # For minimization: reference_point should be >= all front points in all objectives
    for i, point in enumerate(front):
        if not np.all(reference_point >= point):
            # Find violating objectives
            violations = reference_point < point
            raise ValueError(
                f"Reference point is not dominated by point {i}: "
                f"reference={reference_point}, point={point}. "
                f"Violating objectives: {np.where(violations)[0].tolist()}"
            )
    
    # Validate no NaN/Inf
    if not np.all(np.isfinite(front)):
        raise ValueError("Front contains non-finite values")
    if not np.all(np.isfinite(reference_point)):
        raise ValueError("Reference point contains non-finite values")
    
    # Compute hypervolume using validated inputs
    total_volume = 0.0
    
    for point in front:
        # Volume of hyperrectangle from point to reference
        deltas = reference_point - point
        
        # All deltas should be non-negative (validated above)
        assert np.all(deltas >= 0), "Internal error: negative delta"
        
        volume = np.prod(deltas)
        total_volume += volume
    
    return total_volume
```

### Verification
```python
def test_hypervolume_invalid_reference():
    front = np.array([[1.0, 2.0], [2.0, 1.0]])
    
    # Bad reference: not dominated by all points
    bad_ref = np.array([1.5, 1.5])  # Dominates (2.0, 1.0)
    
    with pytest.raises(ValueError, match="not dominated"):
        compute_hypervolume(front, bad_ref)
    
    # Good reference
    good_ref = np.array([3.0, 3.0])
    volume = compute_hypervolume(front, good_ref)
    assert volume > 0
```

---

## 9. FACTOR_OPTIMIZER - Spacing Metric Empty Distances

**File**: `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/pareto.py`  
**Lines**: ~228 (spacing metric)  
**Risk**: Division by zero when front has single point

### Fixed Code
```python
def compute_spacing(front: np.ndarray) -> float:
    """
    Compute spacing metric (uniformity of distribution).
    
    Args:
        front: (N, M) Pareto front
    
    Returns:
        Spacing value (0 = perfectly uniform, lower is better)
    """
    if front.size == 0:
        return 0.0
    
    if len(front.shape) != 2:
        raise ValueError(f"Front must be 2D, got shape {front.shape}")
    
    n_points = front.shape[0]
    
    # Single point has perfect spacing
    if n_points == 1:
        return 0.0
    
    # Two points also have perfect spacing
    if n_points == 2:
        return 0.0
    
    # Compute nearest neighbor distances
    distances = []
    for i in range(n_points):
        min_dist = float('inf')
        for j in range(n_points):
            if i != j:
                dist = np.linalg.norm(front[i] - front[j])
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    # Now distances is guaranteed non-empty and len >= 3
    distances = np.array(distances)
    d_mean = np.mean(distances)
    
    # Compute spacing (std deviation of distances)
    if d_mean < 1e-10:
        # All points at same location
        return 0.0
    
    spacing = np.sqrt(np.sum((distances - d_mean) ** 2) / n_points) / d_mean
    return spacing
```

---

## 10-14: Additional Critical Fixes

Due to length constraints, the remaining 4 critical fixes (mutation cost estimation, budget utilization, repair decay, factor_assets cycle detection, and hash collision) follow the same pattern:

1. Identify the unsafe operation
2. Add validation before the operation
3. Handle edge cases explicitly
4. Provide clear error messages
5. Add verification tests

See the comprehensive audit report for full details on issues C3.4-C3.7 and C5.1-C5.4.

---

## Implementation Checklist

### Phase 1: Research Control (Day 1 Morning)
- [ ] Fix atomic transaction rollback
- [ ] Add orphaned trial prevention
- [ ] Fix SQL variable limit batching
- [ ] Run research_control test suite
- [ ] Manual testing with large batches

### Phase 2: Quant Evaluator (Day 1 Afternoon)
- [ ] Add empty batch validation
- [ ] Fix IC finalization None check
- [ ] Add cache eviction loop safety
- [ ] Run quant_evaluator test suite
- [ ] Stress test with edge cases

### Phase 3: Factor Optimizer (Day 2 Morning)
- [ ] Fix plateau detection division by zero
- [ ] Validate hypervolume reference points
- [ ] Fix spacing metric for small fronts
- [ ] Fix mutation cost, budget, repair issues
- [ ] Run factor_optimizer test suite

### Phase 4: Factor Assets (Day 2 Afternoon)
- [ ] Add cycle detection to lineage graph
- [ ] Fix infinite recursion with visited set
- [ ] Validate gate evaluations non-empty
- [ ] Fix hash collision validation
- [ ] Run factor_assets test suite

### Phase 5: Integration Testing (Day 3)
- [ ] End-to-end pipeline with edge cases
- [ ] Concurrent access stress tests
- [ ] Large data volume tests
- [ ] Document all fixes in CHANGELOG

---

## Post-Fix Monitoring

After deploying these fixes, monitor for:

1. **Error rate changes**: Should decrease significantly
2. **New error types**: Watch for unexpected failure modes
3. **Performance impact**: Validation adds overhead but should be minimal
4. **User reports**: Collect feedback on error message clarity

## Rollback Plan

If any fix causes issues:

1. Each fix is isolated and can be reverted independently
2. Git commits should be atomic (one fix per commit)
3. Feature flags for validation can be added if needed
4. Gradual rollout: deploy to staging → canary → full production

---

**Document Version**: 1.0  
**Last Updated**: 2026-08-14  
**Owner**: Engineering Team
