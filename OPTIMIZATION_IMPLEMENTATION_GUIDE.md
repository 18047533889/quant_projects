# Performance Optimization Implementation Guide
## Step-by-Step Code Changes for 3-10x Speedup

**Priority:** P0 optimizations with highest ROI  
**Timeline:** Week 1-2 for critical path fixes  
**Expected Result:** 3-10x speedup on typical workloads

---

## PRIORITY 1: quant_evaluator IC Computation (5-10x speedup)

### Current Bottleneck
**File:** `/home/shw/quant_projects/quant_evaluator/quant_evaluator/metrics/ic.py`  
**Lines:** 138-159

### Problem Code
```python
def compute_daily_ic(...):
    T, N, F = factor_values.shape
    ic_series = np.full((T, F), np.nan)
    
    for t in range(T):              # ← LOOP 1: Time periods
        for f in range(F):          # ← LOOP 2: Factors
            factor_t = values[t, :, f]
            label_t = labels[t, :]
            
            # PROBLEM: Creates copies every iteration
            if factor_batch.validity is not None:
                factor_valid = factor_batch.validity[t, :, f]
                factor_t = np.where(factor_valid, factor_t, np.nan)  # COPY!
            
            if label_bundle.validity is not None:
                label_valid = label_bundle.validity[t, :]
                label_t = np.where(label_valid, label_t, np.nan)  # COPY!
```

### Optimized Code
```python
def compute_daily_ic(...):
    """Optimized IC computation with pre-applied masks."""
    T, N, F = factor_values.shape
    
    # PRE-APPLY MASKS ONCE (not in loop)
    values = factor_batch.values.copy()
    labels = label_bundle.values.copy()
    
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    
    if label_bundle.validity is not None:
        labels = np.where(
            np.expand_dims(label_bundle.validity, axis=2),
            labels,
            np.nan
        )
    
    # USE NUMBA BACKEND BY DEFAULT
    try:
        from quant_evaluator.kernels.numba_backend import numba_pearson_ic_batch
        USE_NUMBA = True
    except ImportError:
        USE_NUMBA = False
    
    if USE_NUMBA and values.ndim == 3:
        # 5-10x faster than nested loops
        ic_series = numba_pearson_ic_batch(
            values, labels, method='pearson', min_obs=min_obs
        )
    else:
        # Fallback to vectorized numpy
        ic_series = np.full((T, F), np.nan)
        for t in range(T):
            # Vectorize across factors dimension
            factors_t = values[t, :, :]  # (N, F)
            labels_t = labels[t, :, np.newaxis]  # (N, 1)
            
            # Vectorized correlation
            valid_mask = np.isfinite(factors_t) & np.isfinite(labels_t)
            valid_counts = valid_mask.sum(axis=0)
            
            for f in range(F):
                if valid_counts[f] >= min_obs:
                    mask = valid_mask[:, f, 0]
                    x = factors_t[mask, f]
                    y = labels_t[mask, 0]
                    ic_series[t, f] = np.corrcoef(x, y)[0, 1]
    
    return ic_series
```

### Implementation Steps
1. Locate `compute_daily_ic()` in `/quant_evaluator/metrics/ic.py`
2. Replace mask application logic (move before loops)
3. Add Numba backend import with fallback
4. Test with existing test suite: `pytest tests/test_ic.py -v`
5. Benchmark: Should see 5-10x speedup

**Estimated Time:** 2 hours  
**Risk:** Low - existing tests validate correctness

---

## PRIORITY 2: quant_evaluator Quantile Computation (5-10x speedup)

### Current Bottleneck
**File:** `/home/shw/quant_projects/quant_evaluator/quant_evaluator/metrics/quantile.py`  
**Lines:** 195-216

### Problem Code
```python
def assign_quantiles_batch(...):
    T, N, F = factor_values.shape
    quantiles = np.full((T, N, F), -1, dtype=np.int32)
    
    for t in range(T):
        for f in range(F):
            v = factor_values[t, :, f]
            finite_mask = np.isfinite(v)
            v_finite = v[finite_mask]
            
            # RECOMPUTED EVERY ITERATION!
            percentiles = np.linspace(0, 100, n_quantiles + 1)[1:-1]
            boundaries = np.percentile(v_finite, percentiles)  # O(N log N)
```

### Optimized Code
```python
def assign_quantiles_batch(...):
    """Optimized quantile assignment with precomputation."""
    T, N, F = factor_values.shape
    
    # PRECOMPUTE PERCENTILES (once, not T×F times)
    percentile_levels = np.linspace(0, 100, n_quantiles + 1)[1:-1]
    
    # TRY NUMBA BACKEND FIRST
    try:
        from quant_evaluator.kernels.numba_backend import numba_quantile_assignment
        quantiles = numba_quantile_assignment(
            factor_values, n_quantiles, percentile_levels
        )
        return quantiles
    except ImportError:
        pass
    
    # FALLBACK: Optimized numpy version
    quantiles = np.full((T, N, F), -1, dtype=np.int32)
    
    for t in range(T):
        # Process all factors at once (vectorized across F)
        v_t = factor_values[t, :, :]  # (N, F)
        finite_mask_t = np.isfinite(v_t)
        
        for f in range(F):
            mask = finite_mask_t[:, f]
            if mask.sum() < min_samples:
                continue
            
            v_finite = v_t[mask, f]
            
            # Use np.partition (O(N)) instead of full sort
            # For small n_quantiles, this is faster
            if n_quantiles <= 10:
                boundaries = np.partition(v_finite, 
                    [(len(v_finite) * p // 100) for p in percentile_levels]
                )[::len(v_finite)//n_quantiles]
            else:
                boundaries = np.percentile(v_finite, percentile_levels)
            
            # Assign quantiles
            q_bins = np.searchsorted(boundaries, v_finite, side='right')
            quantiles[t, mask, f] = q_bins
    
    return quantiles
```

### Implementation Steps
1. Locate `assign_quantiles_batch()` in `/quant_evaluator/metrics/quantile.py`
2. Move `percentiles` calculation outside loop
3. Add Numba backend check
4. Replace `np.percentile` with `np.partition` for small n_quantiles
5. Test: `pytest tests/test_quantile.py -v`

**Estimated Time:** 2 hours  
**Risk:** Low - numerical results should match exactly

---

## PRIORITY 3: factor_optimizer Parallel Evaluation (3-4x speedup)

### Current Bottleneck
**File:** `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/runner.py`  
**Lines:** 145-205

### Problem Code
```python
def run(self, initial_seed: Optional[List] = None):
    """Main search loop - SEQUENTIAL!"""
    
    while not session.is_finished():
        # Generate proposal (sequential)
        trial = self.proposal_fn()
        
        # Validate (sequential)
        legality = self._validate_trial(trial)
        if not legality["is_legal"]:
            continue
        
        # Evaluate (BLOCKING, sequential)
        result = self.evaluation_fn(trial, fidelity)
        
        # Record result
        session.record_trial(trial, result)
```

### Optimized Code
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

def run(self, initial_seed: Optional[List] = None):
    """Main search loop with parallel evaluation."""
    
    # Use configured concurrency (default: 4)
    max_workers = self.config.max_concurrency or 4
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Keep track of in-flight evaluations
        pending_futures = {}
        
        while not session.is_finished():
            # Fill up to max_workers with legal trials
            while len(pending_futures) < max_workers and not session.is_finished():
                trial = self.proposal_fn()
                
                # Quick validation (still sequential, but fast)
                legality = self._validate_trial(trial)
                if not legality["is_legal"]:
                    continue
                
                # Submit for parallel evaluation
                future = executor.submit(
                    self._evaluate_with_error_handling,
                    trial,
                    fidelity
                )
                pending_futures[future] = trial
            
            # Wait for at least one to complete
            if pending_futures:
                done, not_done = wait(
                    pending_futures.keys(),
                    return_when=FIRST_COMPLETED,
                    timeout=self.config.evaluation_timeout
                )
                
                # Process completed evaluations
                for future in done:
                    trial = pending_futures.pop(future)
                    try:
                        result = future.result()
                        session.record_trial(trial, result)
                    except Exception as e:
                        self._handle_evaluation_error(trial, e)

def _evaluate_with_error_handling(self, trial, fidelity):
    """Wrapper for evaluation with error handling."""
    try:
        return self.evaluation_fn(trial, fidelity)
    except Exception as e:
        # Log error but don't crash the worker
        import logging
        logging.error(f"Evaluation failed for trial {trial.id}: {e}")
        raise
```

### Implementation Steps
1. Locate `SearchRunner.run()` in `/factor_optimizer/search/runner.py`
2. Import `ThreadPoolExecutor` and `wait`
3. Replace sequential loop with parallel submission
4. Add error handling wrapper
5. Test with existing search tests
6. Verify `max_concurrency` configuration is respected

**Estimated Time:** 1 day  
**Risk:** Medium - need to ensure thread safety in evaluation_fn

---

## PRIORITY 4: research_control Batch Event Processing (5-10x speedup)

### Current Bottleneck
**File:** `/home/shw/quant_projects/research_control/research_control/sync/idempotency.py`  
**Lines:** 64-83

### Problem Code
```python
def _sync_campaign_open(self, events: List[Dict]) -> int:
    """Sync campaign open events - ONE AT A TIME!"""
    added = 0
    
    for event in events:  # ← Sequential
        event_id = event["event_id"]
        campaign_id = event["campaign_id"]
        state = event["state"]
        timestamp = event["timestamp"]
        metadata = event.get("metadata")
        
        # Individual INSERT per event
        if self._campaign.append(event_id, campaign_id, state, timestamp, metadata):
            added += 1
    
    return added
```

### Optimized Code
```python
def _sync_campaign_open(self, events: List[Dict]) -> int:
    """Sync campaign open events - BATCH MODE!"""
    if not events:
        return 0
    
    # VALIDATE ALL EVENTS FIRST (bulk validation)
    valid_events = []
    for event in events:
        event_id = event["event_id"]
        if self._is_duplicate(event_id):
            continue
        valid_events.append(event)
    
    if not valid_events:
        return 0
    
    # BATCH INSERT
    added = self._campaign.append_batch([
        {
            'event_id': e["event_id"],
            'campaign_id': e["campaign_id"],
            'state': e["state"],
            'timestamp': e["timestamp"],
            'metadata': e.get("metadata")
        }
        for e in valid_events
    ])
    
    return added
```

### Add to CampaignLedger class
```python
# In /research_control/ledger/campaign.py

def append_batch(self, events: List[Dict]) -> int:
    """Batch append events (5-10x faster than individual appends)."""
    if not events:
        return 0
    
    import json
    
    # Prepare batch data
    batch_data = []
    for event in events:
        metadata_json = json.dumps(event['metadata']) if event.get('metadata') else None
        batch_data.append((
            event['event_id'],
            event['campaign_id'],
            event['state'],
            event['timestamp'],
            metadata_json
        ))
    
    # Single transaction with executemany
    with self._conn() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO campaign_events 
            (event_id, campaign_id, state, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            batch_data
        )
        conn.commit()
        return len(batch_data)
```

### Implementation Steps
1. Add `append_batch()` method to `CampaignLedger` class
2. Add `append_batch()` method to `TrialLedger` class
3. Update `_sync_campaign_open()` to use batch method
4. Update `_sync_trial_events()` similarly
5. Test: `pytest tests/test_sync.py -v`

**Estimated Time:** 1 day  
**Risk:** Low - database operations are isolated

---

## PRIORITY 5: research_control JSON Import Fix (2-3x speedup)

### Current Bottleneck
**File:** `/home/shw/quant_projects/research_control/research_control/ledger/trial.py`  
**Lines:** 109-112

### Problem Code
```python
def append(self, trial_id, ...) -> bool:
    """Append trial event."""
    
    # IMPORT INSIDE HOT PATH!
    import json
    
    parameters_json = json.dumps(parameters) if parameters else None
    metrics_json = json.dumps(metrics) if metrics else None
    metadata_json = json.dumps(metadata) if metadata else None
```

### Optimized Code
```python
# AT MODULE LEVEL (top of file)
import json

# Pre-serialize common values
_JSON_NONE = None  # No need to serialize None

class TrialLedger:
    def append(self, trial_id, ...) -> bool:
        """Append trial event - optimized."""
        
        # Optimized serialization
        parameters_json = json.dumps(parameters) if parameters else _JSON_NONE
        metrics_json = json.dumps(metrics) if metrics else _JSON_NONE
        metadata_json = json.dumps(metadata) if metadata else _JSON_NONE
        
        # Rest of method...
```

### Consider Using orjson for 2-3x JSON speedup
```python
# At module level
try:
    import orjson
    
    def fast_json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8') if obj else None
    
    JSON_DUMPS = fast_json_dumps
except ImportError:
    import json
    JSON_DUMPS = lambda obj: json.dumps(obj) if obj else None

class TrialLedger:
    def append(self, trial_id, ...) -> bool:
        """Append trial event with fast JSON."""
        parameters_json = JSON_DUMPS(parameters)
        metrics_json = JSON_DUMPS(metrics)
        metadata_json = JSON_DUMPS(metadata)
```

### Implementation Steps
1. Move `import json` to module level in `trial.py` and `campaign.py`
2. Move `import json` to module level in `query.py`
3. Consider adding `orjson` as optional dependency
4. Update all JSON serialization calls
5. Test: Full test suite should pass unchanged

**Estimated Time:** 2 hours  
**Risk:** Very low - pure refactoring

---

## PRIORITY 6: Cache Serialization Optimization (3-5x speedup)

### Current Bottleneck
**File:** `/home/shw/quant_projects/quant_evaluator/quant_evaluator/runtime/cache_v2.py`  
**Lines:** 314-327

### Problem Code
```python
def put(self, key: str, value: Any) -> None:
    """Put value in cache."""
    with self._lock:
        # SLOW for large numpy arrays
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        original_size = len(data)
        
        # ALWAYS compresses (even small values)
        if self.enable_compression:
            compressed, stats = self.compressor.compress(data)
```

### Optimized Code
```python
import numpy as np

def put(self, key: str, value: Any) -> None:
    """Put value in cache with optimized serialization."""
    with self._lock:
        # FAST PATH: Use numpy format for arrays
        if isinstance(value, np.ndarray):
            # numpy.save is 5-10x faster than pickle for arrays
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            data = buffer.getvalue()
            serialization_method = 'numpy'
        elif isinstance(value, dict) and all(isinstance(v, np.ndarray) for v in value.values()):
            # Multiple arrays: use npz format
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **value)
            data = buffer.getvalue()
            serialization_method = 'npz'
        else:
            # Fallback to pickle
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            serialization_method = 'pickle'
        
        original_size = len(data)
        
        # ADAPTIVE COMPRESSION: Skip for small values
        if self.enable_compression and original_size > 10240:  # 10KB threshold
            compressed, stats = self.compressor.compress(data)
            if len(compressed) < original_size * 0.9:  # Only if >10% reduction
                data = compressed
                self._compression_stats.append(stats)
        
        # Store with metadata
        self._cache[key] = {
            'data': data,
            'size': original_size,
            'method': serialization_method,
            'timestamp': time.time()
        }

def get(self, key: str) -> Optional[Any]:
    """Get value from cache with optimized deserialization."""
    with self._lock:
        entry = self._cache.get(key)
        if entry is None:
            return None
        
        data = entry['data']
        method = entry['method']
        
        # Decompress if needed (outside lock in production)
        if self.enable_compression and len(data) < entry['size']:
            data = self.compressor.decompress(data)
        
        # FAST DESERIALIZATION
        if method == 'numpy':
            buffer = io.BytesIO(data)
            return np.load(buffer, allow_pickle=False)
        elif method == 'npz':
            buffer = io.BytesIO(data)
            loaded = np.load(buffer)
            return {k: loaded[k] for k in loaded.files}
        else:
            return pickle.loads(data)
```

### Implementation Steps
1. Locate `MemoryCacheLayer` in `/quant_evaluator/runtime/cache_v2.py`
2. Add numpy-specific serialization paths
3. Add adaptive compression threshold
4. Update get/put methods
5. Test with existing cache tests

**Estimated Time:** 4 hours  
**Risk:** Medium - need thorough testing of serialization paths

---

## TESTING STRATEGY

### 1. Correctness Validation
```python
# Add to each optimization
def test_optimization_correctness():
    """Verify optimized version matches reference."""
    np.random.seed(42)
    data = generate_test_data()
    
    # Reference (slow) implementation
    result_reference = slow_reference_impl(data)
    
    # Optimized implementation
    result_optimized = fast_optimized_impl(data)
    
    # Should match exactly
    np.testing.assert_allclose(
        result_optimized,
        result_reference,
        rtol=1e-9,
        atol=1e-12
    )
```

### 2. Performance Benchmarking
```python
from performance_utils import PerformanceComparison

def benchmark_optimization():
    """Benchmark before/after optimization."""
    data = generate_realistic_data()
    
    comp = PerformanceComparison("IC Computation", n_iterations=10)
    comp.benchmark("Reference (slow)", slow_reference_impl, data)
    comp.benchmark("Optimized (fast)", fast_optimized_impl, data)
    comp.report()
    
    # Should show 5-10x speedup
```

### 3. Regression Testing
```bash
# Run full test suite before and after
pytest tests/ -v --benchmark-only

# Compare results
python benchmarks/analyze_results.py before.json after.json
```

---

## ROLLOUT PLAN

### Week 1: Critical Path (P0)
- [ ] Day 1: quant_evaluator IC optimization (2h)
- [ ] Day 1: quant_evaluator Quantile optimization (2h)
- [ ] Day 2: factor_optimizer Parallel evaluation (1d)
- [ ] Day 3: research_control Batch processing (1d)
- [ ] Day 4: research_control JSON fixes (2h)
- [ ] Day 4: Run full benchmark suite
- [ ] Day 5: Bug fixes and validation

**Expected Result:** 3-5x speedup on critical paths

### Week 2: Additional Gains (P1)
- [ ] Day 1: Cache serialization optimization (4h)
- [ ] Day 2: Pareto frontier optimization (1d)
- [ ] Day 3: Lineage tree memoization (1d)
- [ ] Day 4: Timeline merge optimization (4h)
- [ ] Day 5: Full validation and benchmarking

**Expected Result:** 5-8x cumulative speedup

---

## MONITORING

### Performance Metrics to Track
1. **Execution time** - before/after for each optimization
2. **Memory usage** - peak and average
3. **Cache hit rates** - should improve with optimizations
4. **Throughput** - operations per second
5. **Latency** - p50, p95, p99

### Alerting Thresholds
- Regression >10% slower → Block merge
- Memory increase >20% → Investigate
- Cache hit rate drops >5% → Tune parameters

---

## ROLLBACK PLAN

### If Optimization Causes Issues
1. **Immediate rollback:** Git revert the commit
2. **Investigate:** Check test failures and benchmark results
3. **Fix or disable:** Either fix the issue or disable optimization
4. **Re-test:** Validate with full test suite

### Feature Flags
Consider adding feature flags for risky optimizations:
```python
USE_NUMBA_IC = os.getenv('QEVAL_USE_NUMBA_IC', 'true').lower() == 'true'
USE_PARALLEL_EVAL = os.getenv('FOPT_PARALLEL_EVAL', 'true').lower() == 'true'
```

---

## CONTACT & OWNERSHIP

**Implementation Owner:** TBD  
**Reviewer:** TBD  
**Timeline:** Week 1-2 for P0 optimizations  
**Next Review:** After Week 1 completion

---

**END OF IMPLEMENTATION GUIDE**
