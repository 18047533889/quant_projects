# FO-004: Test Split API Audit

**Date:** 2026-08-14  
**Status:** CRITICAL CONTAMINATION RISK IDENTIFIED  
**Auditor:** FO-Worker-TestSplitAudit

---

## Executive Summary

**FINDING: HIGH CONTAMINATION RISK**

The current FactorOptimizer search API exposes full evaluation metrics during search without train/test separation. Any metric returned from QE evaluation is immediately accessible to:
- Search decision logic (best score tracking, plateau detection)
- Pareto frontier ranking
- Lineage analysis and pruning
- LLM prompt generation
- Admission policy decisions

**Critical Issue:** No architectural separation exists between train-time metrics (safe for search) and test-time metrics (must be sealed until final evaluation).

---

## 1. Current API Surface Analysis

### 1.1 Core Search Objects

**SearchSession** (`factor_optimizer/search/runner.py:39-93`)
- **Exposed:** `best_score: Optional[float]` - directly accessible
- **Exposed:** `best_trial_id: Optional[str]` - links to trial with full metrics
- **Exposed:** `trials: List[Trial]` - full list of all trials
- **Method:** `update_best(trial_id, score)` - uses raw score
- **Method:** `successful_trials()` - returns trials with evaluation refs

**Trial** (`factor_optimizer/contracts/trial.py:23-109`)
- **Exposed:** `metadata: Dict[str, Any]` - stores `{"score": ..., "fidelity": ...}`
- **Exposed:** `evaluation_ref: Optional[str]` - reference to QE evidence bundle
- **Method:** `to_dict()` - serializes all fields including metadata

**SearchRunner** (`factor_optimizer/search/runner.py:95-238`)
- **Line 190:** `metadata={"score": result.get("score"), "fidelity": fidelity}`
- **Line 194:** `score = result.get("score")` - directly extracts from evaluation
- **Line 199:** `session.update_best(trial.trial_id, score)` - uses for tracking
- **Line 201:** `budget_tracker.record_evaluation(cost=result.get("cost", 1.0))`

### 1.2 Evaluation Integration

**QuantEvaluatorAdapter** (`factor_optimizer/adapters/quant_evaluator.py:6-77`)
```python
def evaluate(...) -> Dict[str, Any]:
    """Returns:
        - evaluation_id (str)
        - metrics (dict): Metric name -> value  # ← ALL METRICS EXPOSED
        - diagnostics (dict)
        - evidence_ref (str)
    """
```

**MockQEAdapter** (`factor_optimizer/adapters/quant_evaluator.py:178-213`)
```python
mock_metrics = {
    "rank_ic": random.uniform(-0.1, 0.15),      # ← Test contamination
    "ic_mean": random.uniform(-0.08, 0.12),     # ← Test contamination
    "ic_std": random.uniform(0.05, 0.15),
    "turnover": random.uniform(0.1, 0.3),
    "sharpe": random.uniform(-0.5, 2.0),        # ← Test contamination
}
```

All metrics are returned in the same dictionary with no train/test separation.

---

## 2. Contamination Vectors

### 2.1 Direct Access During Search

**Vector 1: Best Score Tracking**
```python
# runner.py:199
session.update_best(trial.trial_id, score)
```
If `score` is test IC, search is directly optimizing on test data.

**Vector 2: Plateau Detection**
```python
# runner.py:195-197
recent_scores.append(score)
if self._check_plateau(recent_scores):
    session.finish(reason="plateau_detected")
```
Stopping criterion uses the same score → test leakage.

**Vector 3: Trial Metadata**
```python
# runner.py:187-191
trial.update_status(
    TrialStatus.EVALUATED,
    evaluation_ref=result.get("evaluation_id"),
    metadata={"score": result.get("score"), "fidelity": fidelity},  # ← Stored
)
```
Score persisted in Trial object, accessible via `session.trials[i].metadata["score"]`.

### 2.2 Indirect Access Through Evidence

**Vector 4: Evidence Bundle Retrieval**
```python
# quant_evaluator.py:45-58
def get_evidence(self, evaluation_id: str) -> Dict[str, Any]:
    """Returns:
        - metrics (dict): Full metric results  # ← Includes test metrics
        - timeseries (dict): Optional time-series metrics
    """
```

From `trial.evaluation_ref`, can retrieve full evidence including test split metrics.

### 2.3 Multi-Objective Optimization

**Vector 5: Pareto Frontier**
```python
# pareto.py:16-19
@dataclass
class ParetoPoint:
    trial_id: str
    objectives: Tuple[float, ...]  # ← Could include test metrics
    metadata: Dict[str, Any]
```

If objectives include test IC, Pareto ranking leaks test data.

**Vector 6: Lineage Scoring**
```python
# lineage.py:218-241
def best_in_lineage(self, trial_id: str) -> Optional[LineageNode]:
    scored_nodes = [
        self.nodes[tid] for tid in lineage_ids
        if tid in self.nodes and self.nodes[tid].score is not None
    ]
    return max(scored_nodes, key=lambda n: n.score)  # ← Uses raw score
```

Lineage pruning decisions based on scores → test contamination if score is test metric.

### 2.4 LLM Prompt Generation

**Vector 7: Performance Metrics in Prompts**
```python
# prompts.py:159-160
Current Performance Metrics:
{performance_metrics}  # ← Template includes metrics
```

If `performance_metrics` includes test scores, LLM sees test data and biases future proposals.

### 2.5 Admission Policy

**Vector 8: Expected Value Estimation**
```python
# decisions.py:212-214
if expected_value < self.criteria.min_expected_value:
    rejection_reasons.append(RejectionReason.LOW_EXPECTED_VALUE)
```

If `expected_value` is computed from test metrics, admission decisions leak test data.

---

## 3. Example Contamination Scenario

```python
# CURRENT DANGEROUS PATTERN
def run_search():
    runner = SearchRunner(config, proposal_fn, evaluation_fn)
    session = runner.run("search-001")
    
    # Evaluation returns EVERYTHING
    def evaluation_fn(trial, fidelity):
        result = qe_adapter.evaluate(factor, labels)
        return {
            "evaluation_id": "eval_123",
            "score": result["metrics"]["test_ic"],  # ← TEST METRIC!
            "cost": 10.0,
        }
    
    # Search optimizes on test metric
    session.update_best(trial.trial_id, score)  # ← Optimizing on test!
    
    # Plateau detection on test metric
    if improvement_on_test < threshold:
        stop_search()  # ← Stopping on test signal
    
    # LLM sees test performance
    prompt = f"Parent IC: {trial.metadata['score']}"  # ← Test leakage
    
    # RESULT: Overfit to test set, invalid research
```

---

## 4. Architecture Sketch: SealedTestResult

### 4.1 Core Principle

**Train/Test Separation:**
- **Train metrics:** Available during search (validation IC, train IC)
- **Test metrics:** Sealed until final evaluation, never accessible during search

### 4.2 Proposed Object Model

```python
@dataclass
class TrainMetrics:
    """Metrics available during search (train/validation only)."""
    train_ic: float
    validation_ic: float
    train_sharpe: float
    turnover: float
    complexity_score: float
    coverage: float
    
    def primary_score(self) -> float:
        """Single score for search optimization (validation IC)."""
        return self.validation_ic


@dataclass
class SealedTestResult:
    """Test metrics sealed until final evaluation."""
    _test_ic: float
    _test_sharpe: float
    _test_ic_std: float
    _is_unsealed: bool = False
    _unsealed_at: Optional[datetime] = None
    _unsealed_by: Optional[str] = None
    
    def unseal(self, authorized_context: str) -> "UnsealedTestResult":
        """Unseal test results after search completes."""
        if self._is_unsealed:
            raise ValueError("Already unsealed")
        
        self._is_unsealed = True
        self._unsealed_at = datetime.now()
        self._unsealed_by = authorized_context
        
        return UnsealedTestResult(
            test_ic=self._test_ic,
            test_sharpe=self._test_sharpe,
            test_ic_std=self._test_ic_std,
            unsealed_at=self._unsealed_at,
        )
    
    def __repr__(self):
        return f"<SealedTestResult sealed={not self._is_unsealed}>"
    
    # Block attribute access
    def __getattribute__(self, name):
        if name.startswith("_test_") and not object.__getattribute__(self, "_is_unsealed"):
            raise AttributeError(f"Test metric '{name}' is sealed during search")
        return object.__getattribute__(self, name)


@dataclass
class EvaluationResult:
    """Split evaluation result returned during search."""
    evaluation_id: str
    train_metrics: TrainMetrics
    test_metrics: SealedTestResult  # ← Sealed, not accessible
    diagnostics: Dict[str, Any]
    fidelity: int
    
    def search_score(self) -> float:
        """Score for search optimization (validation only)."""
        return self.train_metrics.primary_score()


@dataclass
class UnsealedTestResult:
    """Unsealed test results after search completion."""
    test_ic: float
    test_sharpe: float
    test_ic_std: float
    unsealed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_ic": self.test_ic,
            "test_sharpe": self.test_sharpe,
            "test_ic_std": self.test_ic_std,
            "unsealed_at": self.unsealed_at.isoformat(),
        }
```

### 4.3 Modified Search Objects

```python
@dataclass
class Trial:
    """Modified to only expose train metrics."""
    trial_id: str
    mutation_id: str
    status: TrialStatus
    
    # CHANGED: No raw score, only train evaluation
    train_evaluation: Optional[EvaluationResult] = None
    
    def search_score(self) -> Optional[float]:
        """Score for search decisions (validation only)."""
        if self.train_evaluation is None:
            return None
        return self.train_evaluation.search_score()


@dataclass
class SearchSession:
    """Modified to track validation scores only."""
    session_id: str
    config: SearchConfig
    trials: List[Trial]
    
    # CHANGED: Track validation score, not test
    best_validation_score: Optional[float] = None
    best_trial_id: Optional[str] = None
    
    def update_best(self, trial_id: str, validation_score: float) -> bool:
        """Update best validation score (not test)."""
        if self.best_validation_score is None or validation_score > self.best_validation_score:
            self.best_validation_score = validation_score
            self.best_trial_id = trial_id
            return True
        return False


class SearchRunner:
    """Modified to use validation scores only."""
    
    def run(self, session_id: str) -> SearchSession:
        # ...
        result = self.evaluation_fn(trial, fidelity)
        
        # CHANGED: Extract validation score only
        validation_score = result.search_score()  # ← Not test score
        
        trial.update_status(
            TrialStatus.EVALUATED,
            train_evaluation=result,  # ← Contains sealed test
        )
        
        session.update_best(trial.trial_id, validation_score)  # ← Validation only
```

### 4.4 Final Evaluation After Search

```python
def finalize_search_results(session: SearchSession) -> FinalSearchReport:
    """Unseal test results after search completes."""
    
    # Search is complete, now safe to unseal
    final_results = []
    
    for trial in session.successful_trials():
        if trial.train_evaluation is None:
            continue
        
        # Unseal test metrics
        test_result = trial.train_evaluation.test_metrics.unseal(
            authorized_context=f"finalize_search_{session.session_id}"
        )
        
        final_results.append({
            "trial_id": trial.trial_id,
            "validation_score": trial.search_score(),  # What search saw
            "test_result": test_result.to_dict(),      # What we report
        })
    
    return FinalSearchReport(
        session_id=session.session_id,
        best_validation_trial=session.best_trial_id,
        results=final_results,
    )
```

---

## 5. QE Adapter Contract Changes

### 5.1 Current Contract (UNSAFE)

```python
def evaluate(...) -> Dict[str, Any]:
    return {
        "evaluation_id": "eval_123",
        "metrics": {
            "rank_ic": 0.12,      # ← Train? Test? Unclear
            "sharpe": 1.5,        # ← Which split?
        },
        "diagnostics": {...},
    }
```

### 5.2 Proposed Contract (SAFE)

```python
def evaluate(...) -> EvaluationResult:
    """
    Evaluate factor with train/validation/test splits.
    
    Returns:
        EvaluationResult with:
        - train_metrics: TrainMetrics (accessible)
        - test_metrics: SealedTestResult (sealed)
        - diagnostics: Coverage, warnings, etc.
    """
    # QE computes on all splits
    train_ic = compute_ic(factor_values, labels, split="train")
    val_ic = compute_ic(factor_values, labels, split="validation")
    test_ic = compute_ic(factor_values, labels, split="test")
    
    return EvaluationResult(
        evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
        train_metrics=TrainMetrics(
            train_ic=train_ic,
            validation_ic=val_ic,  # ← Used for search
            train_sharpe=compute_sharpe(train_ic),
            turnover=compute_turnover(factor_values),
            complexity_score=estimate_complexity(factor_def),
            coverage=compute_coverage(factor_values),
        ),
        test_metrics=SealedTestResult(
            _test_ic=test_ic,      # ← Sealed
            _test_sharpe=compute_sharpe(test_ic),
            _test_ic_std=compute_ic_std(test_ic),
        ),
        diagnostics={...},
        fidelity=fidelity,
    )
```

---

## 6. Implementation Approach

### Phase 1: Contract Definition (Week 1)
1. Define `TrainMetrics`, `SealedTestResult`, `EvaluationResult` contracts
2. Add `SealedTestResult` with `__getattribute__` protection
3. Write unit tests for sealing/unsealing behavior
4. Document train/validation/test split semantics

### Phase 2: QE Adapter Update (Week 2)
1. Update `QuantEvaluatorAdapter` protocol to return `EvaluationResult`
2. Update `MockQEAdapter` to generate split metrics
3. Add integration tests verifying sealing

### Phase 3: Search API Migration (Week 3)
1. Update `Trial` to store `EvaluationResult` instead of raw metadata
2. Update `SearchSession` to track validation scores
3. Update `SearchRunner` to use `result.search_score()`
4. Migrate Pareto, Lineage, Plateau to use validation scores only

### Phase 4: Validation & Rollout (Week 4)
1. Add audit tests that attempt to access sealed test metrics (should fail)
2. Add contamination detector that scans for test metric access
3. Update documentation and examples
4. Gradual rollout with monitoring

### Phase 5: Final Evaluation Tooling
1. Implement `finalize_search_results()` with unsealing
2. Add reporting tools that show validation vs test performance
3. Add overfit detection (validation score >> test score)

---

## 7. Backward Compatibility Strategy

### Option A: Hard Break (Recommended)
- Increment to v0.2.0
- All evaluation functions must return `EvaluationResult`
- Clear error messages if old dict format used
- Migration guide for existing code

### Option B: Gradual Transition
- Support both dict and `EvaluationResult` for v0.1.x
- Emit deprecation warnings when dict used
- Auto-wrap dict in `EvaluationResult` with train=test (log warning)
- Hard break in v0.2.0

**Recommendation:** Option A (hard break) - contamination risk is too high to allow gradual transition.

---

## 8. Audit Test Suite

```python
def test_sealed_test_metrics_not_accessible_during_search():
    """Verify test metrics cannot be accessed during search."""
    session = run_search_with_sealed_results()
    
    for trial in session.trials:
        with pytest.raises(AttributeError, match="sealed during search"):
            _ = trial.train_evaluation.test_metrics._test_ic
        
        # Validation score should be accessible
        assert trial.search_score() is not None


def test_unseal_requires_authorization():
    """Verify unsealing requires explicit authorization."""
    result = create_sealed_result()
    
    # Cannot unseal during search
    with pytest.raises(AttributeError):
        _ = result.test_metrics._test_ic
    
    # Can unseal with authorization
    unsealed = result.test_metrics.unseal("final_evaluation")
    assert unsealed.test_ic is not None
    
    # Cannot unseal twice
    with pytest.raises(ValueError, match="Already unsealed"):
        result.test_metrics.unseal("duplicate")


def test_search_optimization_uses_validation_only():
    """Verify search optimizes on validation, not test."""
    session = run_search()
    
    # Best score should be validation score
    best_trial = next(t for t in session.trials if t.trial_id == session.best_trial_id)
    assert session.best_validation_score == best_trial.search_score()
    
    # Test metrics should still be sealed
    with pytest.raises(AttributeError):
        _ = best_trial.train_evaluation.test_metrics._test_ic


def test_contamination_detector():
    """Detect any code paths that access test metrics during search."""
    with ContaminationMonitor() as monitor:
        run_search()
    
    violations = monitor.get_violations()
    assert len(violations) == 0, f"Test contamination detected: {violations}"
```

---

## 9. Risk Assessment

### Current State: CRITICAL
- **Contamination Probability:** 100% (no separation exists)
- **Impact:** Invalid research results, production losses
- **Detectability:** Low (silent overfitting)

### After Implementation: LOW
- **Contamination Probability:** <1% (architectural prevention)
- **Impact:** Blocked by type system and runtime checks
- **Detectability:** High (audit tests, logging)

---

## 10. Recommendations

### Immediate Actions (P0)
1. **FREEZE:** Document current API as unsafe for production
2. **IMPLEMENT:** Phase 1-2 (contracts + QE adapter) within 2 weeks
3. **AUDIT:** Review any existing searches for contamination

### Short-term (P1)
1. Complete Phase 3-4 (search API migration + validation)
2. Add contamination detection to CI/CD pipeline
3. Update all examples and documentation

### Long-term (P2)
1. Implement overfit detection and reporting
2. Add multi-fidelity evaluation with progressive unsealing
3. Consider differential privacy for intermediate scores

---

## 11. References

**Audited Files:**
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/runner.py`
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/contracts/trial.py`
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/adapters/quant_evaluator.py`
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/pareto.py`
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/lineage.py`
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/llm/prompts.py`
- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/policy/decisions.py`

**Test Coverage:**
- `/home/shw/quant_projects/factor_optimizer/tests/search/test_runner.py`
- `/home/shw/quant_projects/factor_optimizer/tests/adapters/test_quant_evaluator.py`

---

**END OF AUDIT**
