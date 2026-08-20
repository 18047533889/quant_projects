# FO-001/003/004: Split Handling Design Analysis

**Tasks**:
- FO-001: External SplitPlan first (don't recreate splits inside optimizer)
- FO-003: Split permissions (train/validation/test separation)
- FO-004: Sealed test API (prevent test contamination)

**Status**: ⚠️ PREMATURE - factor_optimizer is currently stub/mock implementation

---

## Current Implementation State

### SearchRunner (factor_optimizer/search/runner.py)

**Reality**: 
- No split handling whatsoever
- Evaluation via abstract `evaluation_fn: Callable[[Trial, int], Dict[str, Any]]`
- No concept of train/validation/test
- No concept of SplitPlan

### QuantEvaluatorAdapter (factor_optimizer/adapters/quant_evaluator.py)

**Reality**:
- Protocol/interface only
- Mock implementation for testing (random metrics)
- Real QE integration doesn't exist yet
- Comment: "Note: QE doesn't exist yet, so this is a forward-looking stub"

### Current Evaluation Flow

```python
# SearchRunner.run():
result = self.evaluation_fn(trial, fidelity)
# Returns: {"evaluation_id": str, "score": float, "cost": float}
```

No split context, no label handling, no contamination prevention possible.

---

## Problem Assessment

### FO-001: External SplitPlan First

**Issue**: Not applicable yet. There's no split creation logic to refactor.

**When needed**: When SearchRunner actually calls real QE evaluation:
```python
# Future design:
def evaluate_trial(
    trial: Trial,
    split_plan: SplitPlan,  # ← Passed from caller
    labels: LabelBundle,
    fidelity: int
) -> EvaluationResult:
    # Use split_plan, don't create splits here
    pass
```

### FO-003: Split Permissions

**Issue**: Not applicable yet. No splits to enforce permissions on.

**When needed**: When SearchRunner accesses evaluation results:
```python
# Future design:
@dataclass
class SearchEvaluationResult:
    train_metrics: Dict[str, float]  # For fitting
    validation_metrics: Dict[str, float]  # For selection
    # test_metrics: NOT AVAILABLE during search
```

### FO-004: Sealed Test API

**Issue**: Not applicable yet. No test data to seal.

**When needed**: After search completes:
```python
# Future design:
class SealedTestResult:
    """Only accessible after search frozen."""
    
    @classmethod
    def from_best_trial(
        cls,
        trial_id: str,
        split_plan: SplitPlan
    ) -> 'SealedTestResult':
        # Re-evaluate best trial on test split
        # Return sealed, immutable result
        pass
```

---

## Recommended Architecture (Future Implementation)

### Phase 1: Accept External SplitPlan (FO-001)

```python
class SearchRunner:
    def run(
        self,
        session_id: str,
        split_plan: SplitPlan,  # ← Required from caller
        labels: LabelBundle,
    ) -> SearchSession:
        # Validate split_plan has train + validation
        # Pass to evaluation_fn
        # Never create or modify splits
        pass
```

### Phase 2: Enforce Split Permissions (FO-003)

```python
@dataclass
class TrialEvaluation:
    """Evaluation result during search - test hidden."""
    trial_id: str
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    diagnostics: Dict[str, Any]
    
    # No test_metrics property - not accessible
    
    def selection_score(self) -> float:
        """Use validation metrics for selection."""
        return self.validation_metrics["rank_ic"]
```

### Phase 3: Sealed Test Evaluation (FO-004)

```python
class SearchSession:
    def seal_and_test(self) -> SealedTestResult:
        """
        Freeze search and evaluate best on test.
        Can only be called once. Irreversible.
        """
        if self._test_result is not None:
            raise RuntimeError("Test already evaluated")
        
        if not self.is_finished():
            raise RuntimeError("Must finish search before testing")
        
        # Re-evaluate best trial on test split
        best_trial = self._get_best_trial()
        test_eval = self._evaluate_on_test_split(best_trial)
        
        self._test_result = SealedTestResult(
            trial_id=best_trial.trial_id,
            test_metrics=test_eval.metrics,
            frozen_at=datetime.now()
        )
        
        return self._test_result
```

---

## Current Task Status

### FO-001: External SplitPlan First
- **Status**: NOT_APPLICABLE (no split logic exists yet)
- **Action**: Defer until real QE integration
- **Priority**: P1 → P3 (design guidance only)

### FO-003: Split Permissions
- **Status**: NOT_APPLICABLE (no splits to protect)
- **Action**: Defer until evaluation results include splits
- **Priority**: P1 → P3 (design guidance only)

### FO-004: Sealed Test API
- **Status**: NOT_APPLICABLE (no test data)
- **Action**: Defer until splits implemented
- **Priority**: P1 → P3 (design guidance only)

---

## Recommendation

**DO NOT IMPLEMENT NOW**. These are design principles for future development:

1. **Document as architecture guidelines** in factor_optimizer/docs/split_handling.md
2. **Add to SearchRunner docstring** as future contract
3. **Create stub types** (SplitPlan, SealedTestResult) with NotImplementedError
4. **Revisit when QE integration is real**

**Why defer**:
- Current FO is mock/stub
- No real evaluation pipeline
- No contamination risk (no test data)
- Premature implementation = wasted effort

**Protection until then**:
- Mark SearchRunner as "experimental, not production"
- Document split handling requirements
- Add type stubs for future API

---

## Action Items

1. ✅ Document split handling architecture in this file
2. [ ] Add stub types to factor_optimizer/contracts/splits.py
3. [ ] Update SearchRunner docstring with split contract
4. [ ] Mark tasks as DEFERRED in ledger
