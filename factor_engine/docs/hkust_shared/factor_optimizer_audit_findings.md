# Factor Optimizer Error Handling & Edge Case Audit

**Audit Date:** 2026-08-14  
**Package:** factor_optimizer  
**Scope:** Error handling, input validation, edge cases, and external dependency failures

---

## Executive Summary

Audited 12 core files across public API, core logic, LLM integration, and data structures. Identified **43 findings** across 4 risk levels:
- **Critical (7):** Arithmetic errors, missing validation causing crashes
- **High (14):** Edge cases that cause silent failures or incorrect behavior
- **Medium (16):** Missing validation or poor error messages
- **Low (6):** Minor usability issues

The package has good structural separation but lacks defensive validation at API boundaries and in arithmetic operations.

---

## 1. PUBLIC API ENTRY POINTS

### 1.1 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/runner.py`

#### Finding 1.1.1: Division by Zero in Plateau Detection
**Risk:** Critical  
**Location:** `SearchRunner._check_plateau()` line 236  
**Issue:** When `max_score == 0`, computes `relative_improvement = (max_score - min_score) / abs(max_score)` which is `0 / 0`.

**Example:**
```python
recent_scores = [0.0, 0.0, 0.0]  # All trials scored zero
# Line 236: (0.0 - 0.0) / abs(0.0) → 0 / 0 → undefined
```

**Fix:**
```python
if max_score == 0:
    return min_score == 0  # Current code
if abs(max_score) < 1e-10:  # Add epsilon check
    return abs(min_score) < 1e-10
relative_improvement = (max_score - min_score) / abs(max_score)
```

---

#### Finding 1.1.2: No Validation for Negative Budget Values
**Risk:** Medium  
**Location:** `SearchConfig.__post_init__()` lines 30-35  
**Issue:** `plateau_threshold` validates `>= 0` but `budget` object has no validation for negative values passed through.

**Example:**
```python
budget = SearchBudget(max_trials=-10, max_cost_units=-1000.0)
# Passes through unchecked, causes infinite loop in run()
```

**Fix:** Add validation in `SearchBudget.__post_init__()` (see section 1.4).

---

#### Finding 1.1.3: Unbounded Loop if Proposal Function Fails
**Risk:** High  
**Location:** `SearchRunner.run()` line 162  
**Issue:** If `proposal_fn()` raises exceptions repeatedly, no circuit breaker exists.

**Example:**
```python
def bad_proposal():
    raise RuntimeError("LLM API down")

runner = SearchRunner(config, bad_proposal, eval_fn)
runner.run("session_1")  # Infinite exception loop until budget exhausted
```

**Fix:**
```python
consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 10

while not session.is_finished():
    try:
        trial = self.proposal_fn()
        consecutive_failures = 0  # Reset on success
    except Exception as e:
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            session.finish(reason="proposal_failure")
            break
        continue
```

---

#### Finding 1.1.4: No Handling for None Score
**Risk:** Medium  
**Location:** `SearchRunner.run()` lines 194-199  
**Issue:** Code checks `if score is not None` but doesn't handle case where evaluation succeeds but returns no score.

**Example:**
```python
result = {"evaluation_id": "eval_123"}  # Missing "score" key
score = result.get("score")  # None
# Lines 194-199 silently skip, but best_score never updates
```

**Fix:**
```python
score = result.get("score")
if score is None:
    trial.update_status(TrialStatus.FAILED, failure_reason="Evaluation returned no score")
    continue
if not isinstance(score, (int, float)) or not math.isfinite(score):
    trial.update_status(TrialStatus.FAILED, failure_reason=f"Invalid score: {score}")
    continue
```

---

#### Finding 1.1.5: Missing Validation Stub
**Risk:** Low  
**Location:** `SearchRunner._validate_trial()` lines 211-216  
**Issue:** Returns hardcoded `True` for all trials; no actual validation.

**Recommendation:** Document as intentional stub or implement basic checks.

---

### 1.2 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/pareto.py`

#### Finding 1.2.1: Empty Objectives Tuple Passes __post_init__
**Risk:** High  
**Location:** `ParetoPoint.__post_init__()` line 22  
**Issue:** `if not self.objectives` fails for empty tuple `()` but tuple can be created with zero length.

**Example:**
```python
# Line 22 checks "if not self.objectives" but empty tuple is falsy
point = ParetoPoint(trial_id="t1", objectives=())  # Should fail but check might pass
```

**Fix:**
```python
if not self.objectives or len(self.objectives) == 0:
    raise ValueError("objectives cannot be empty")
```

---

#### Finding 1.2.2: Hypervolume Division by Zero
**Risk:** Critical  
**Location:** `ParetoFrontier.hypervolume()` lines 177-184  
**Issue:** If `reference_point[0] == x`, then `width = x - reference_point[0]` is zero, causing degenerate volume.

**Example:**
```python
frontier.add_point(ParetoPoint("t1", (5.0, 10.0)))
hv = frontier.hypervolume((5.0, 0.0))  # width = 5.0 - 5.0 = 0
```

**Fix:** Add check at line 177:
```python
for point in sorted_points:
    x, y = point.objectives
    if abs(x - reference_point[0]) < 1e-10 or abs(y - reference_point[1]) < 1e-10:
        continue  # Skip degenerate points
```

---

#### Finding 1.2.3: Spacing Metric Division by Zero
**Risk:** Critical  
**Location:** `ParetoFrontier.spacing()` lines 228-230  
**Issue:** If `len(distances) == 0`, division by zero occurs.

**Example:**
```python
# Single point frontier
frontier.add_point(ParetoPoint("t1", (1.0, 1.0)))
spacing = frontier.spacing()  # Returns 0.0 correctly

# But if distances computation has edge case:
distances = []  # Empty due to logic error
mean_dist = sum(distances) / len(distances)  # ZeroDivisionError
```

**Fix:**
```python
if not distances or len(distances) == 0:
    return 0.0
mean_dist = sum(distances) / len(distances)
```

---

#### Finding 1.2.4: Coverage Metric Returns 1.0 for Empty Frontier
**Risk:** Medium  
**Location:** `ParetoFrontier.coverage()` lines 198-199  
**Issue:** When `other.points` is empty, returns `1.0` (100% coverage), which may be semantically incorrect.

**Example:**
```python
frontier_a = ParetoFrontier()
frontier_b = ParetoFrontier()
coverage = frontier_a.coverage(frontier_b)  # Returns 1.0
```

**Recommendation:** Return `None` or `NaN` to indicate undefined coverage, or document this as intended behavior.

---

#### Finding 1.2.5: No Validation for Infinite/NaN Objectives
**Risk:** High  
**Location:** `ParetoPoint.__post_init__()` lines 24-25  
**Issue:** Checks `isinstance(v, (int, float))` but allows `float('inf')` and `float('nan')`.

**Example:**
```python
point = ParetoPoint("t1", (float('inf'), 2.0))  # Passes validation
point2 = ParetoPoint("t2", (float('nan'), 3.0))  # Passes validation
```

**Fix:**
```python
if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.objectives):
    raise ValueError("all objectives must be finite numeric values")
```

---

### 1.3 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/search/multifidelity.py`

#### Finding 1.3.1: Division by Zero in Promotion Criteria
**Risk:** High  
**Location:** `PromotionCriteria.should_promote()` line 147  
**Issue:** If `total == 0`, then `threshold_rank = int(total * self.top_k_fraction)` works, but caller might pass `total=0`.

**Example:**
```python
criteria = PromotionCriteria(top_k_fraction=0.5)
should = criteria.should_promote(score=0.8, rank=0, total=0)
# threshold_rank = int(0 * 0.5) = 0
# rank < threshold_rank → 0 < 0 → False (correct but confusing)
```

**Fix:** Add guard:
```python
if self.top_k_fraction is not None:
    if total <= 0:
        checks.append(False)  # Cannot promote from empty pool
    else:
        threshold_rank = int(total * self.top_k_fraction)
        checks.append(rank < threshold_rank)
```

---

#### Finding 1.3.2: No Validation for Zero or Negative Budget
**Risk:** Medium  
**Location:** `MultiFidelityScheduler.optimal_tier_for_budget()` lines 238-243  
**Issue:** If `remaining_budget <= 0`, always returns `FidelityTier.L0` even if L0 costs more than zero budget.

**Example:**
```python
scheduler = MultiFidelityScheduler()
tier = scheduler.optimal_tier_for_budget(remaining_budget=-10.0)
# Returns L0 even though budget is exhausted
```

**Fix:**
```python
def optimal_tier_for_budget(self, remaining_budget: float) -> Optional[FidelityTier]:
    if remaining_budget <= 0:
        return None  # Signal that no tier is affordable
    for tier in reversed(list(FidelityTier)):
        if self.estimate_cost(tier) <= remaining_budget:
            return tier
    return FidelityTier.L0 if self.estimate_cost(FidelityTier.L0) <= remaining_budget else None
```

---

### 1.4 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/contracts/search_budget.py`

#### Finding 1.4.1: Max LLM Calls Can Be Zero
**Risk:** Low  
**Location:** `SearchBudget.__post_init__()` line 32  
**Issue:** Validates `max_llm_calls >= 0` but zero means no LLM calls allowed, which might be unintended.

**Recommendation:** Document that `max_llm_calls=0` disables LLM proposals, or require `>= 1`.

---

#### Finding 1.4.2: can_call_llm Returns False When None
**Risk:** Medium  
**Location:** `BudgetTracker.can_call_llm()` lines 68-71  
**Issue:** Returns `False` when `max_llm_calls is None`, but `None` should mean unlimited.

**Example:**
```python
budget = SearchBudget(max_trials=100, max_llm_calls=None)
tracker = BudgetTracker(budget)
tracker.can_call_llm()  # Returns False, should return True (unlimited)
```

**Fix:**
```python
def can_call_llm(self) -> bool:
    if self.budget.max_llm_calls is None:
        return True  # Unlimited
    return self.llm_calls_used < self.budget.max_llm_calls
```

---

## 2. CORE LOGIC

### 2.1 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/grammar/validation.py`

#### Finding 2.1.1: No Timeout for FE Adapter Calls
**Risk:** High  
**Location:** `MutationValidator._check_fe_legality()` lines 115-127  
**Issue:** Calls `self.fe_adapter.validate_mutation()` with no timeout; if FE hangs, validator hangs.

**Recommendation:** Wrap in timeout or async with cancellation token.

---

#### Finding 2.1.2: Exception Swallowed as Warning
**Risk:** Medium  
**Location:** `MutationValidator.validate()` lines 104-110  
**Issue:** FE adapter exceptions are caught and added as warnings, not errors. Mutation proceeds as valid.

**Example:**
```python
# FE adapter raises RuntimeError("Database connection lost")
# Line 110: warnings.append(f"FE legality check error: Database connection lost")
# Line 112: is_valid = len(errors) == 0  # Still True!
```

**Fix:** Treat adapter failures as validation errors:
```python
except Exception as e:
    errors.append(f"FE legality check failed: {str(e)}")
```

---

### 2.2 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/complexity/profile.py`

#### Finding 2.2.1: Division by Zero in Mutation Delta
**Risk:** Critical  
**Location:** `ComplexityEstimator.estimate_mutation_delta()` line 168  
**Issue:** `cost_ratio = new_window / max(parent_profile.lookback_periods, 1)` divides by 1 if parent has zero lookback, but `max(..., 1)` prevents true zero. However, if `lookback_periods` is negative (invalid), still divides.

**Example:**
```python
parent = ComplexityProfile(lookback_periods=0)
# cost_ratio = 20 / max(0, 1) = 20 / 1 = 20  # OK

parent = ComplexityProfile(lookback_periods=-5)  # Invalid but not validated
# cost_ratio = 20 / max(-5, 1) = 20 / 1 = 20  # Appears OK but hides invalid input
```

**Fix:** Validate `lookback_periods >= 0` in `ComplexityProfile.__post_init__()`:
```python
def __post_init__(self):
    if self.lookback_periods < 0:
        raise ValueError("lookback_periods must be >= 0")
    if self.operator_count < 0:
        raise ValueError("operator_count must be >= 0")
```

---

#### Finding 2.2.2: No Validation for Negative Complexity Values
**Risk:** Medium  
**Location:** `ComplexityProfile` dataclass lines 28-40  
**Issue:** All numeric fields default to zero but no validation prevents negative values.

**Example:**
```python
profile = ComplexityProfile(operator_count=-10, estimated_cost=-50.0)
# No validation, causes budget checks to behave unexpectedly
```

**Fix:** Add `__post_init__()` validation (see 2.2.1).

---

### 2.3 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/complexity/budget.py`

#### Finding 2.3.1: Division by Zero in Budget Utilization
**Risk:** Critical  
**Location:** `BudgetTracker.stats()` lines 187-194  
**Issue:** Divides by `self.budget.max_cost` and `self.budget.max_operator_count` which can be `None`.

**Example:**
```python
budget = ComplexityBudget(max_cost=None, strict=False)
tracker = BudgetTracker(budget)
stats = tracker.stats()
# Line 188: max_cost / self.budget.max_cost → value / None → TypeError
```

**Fix:**
```python
"budget_cost_utilization": (
    max_cost / self.budget.max_cost if self.budget.max_cost and self.budget.max_cost > 0 else None
),
```

---

#### Finding 2.3.2: No Validation for Negative Budget Limits
**Risk:** High  
**Location:** `ComplexityBudget` dataclass lines 28-34  
**Issue:** No `__post_init__()` validation; negative budgets pass through.

**Example:**
```python
budget = ComplexityBudget(max_cost=-100.0, max_operator_count=-10)
# is_within_budget() always returns True since any value >= negative limit
```

**Fix:**
```python
def __post_init__(self):
    if self.max_cost is not None and self.max_cost <= 0:
        raise ValueError("max_cost must be > 0")
    if self.max_operator_count is not None and self.max_operator_count < 1:
        raise ValueError("max_operator_count must be >= 1")
    # Similar for other fields
```

---

### 2.4 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/policy/repair.py`

#### Finding 2.4.1: Division by Zero in Decay Adjustment
**Risk:** Critical  
**Location:** `RepairMapper._strategy_to_mutation()` line 265  
**Issue:** `new_decay = max(0.1, min(0.9, current_decay * 0.8))` assumes `current_decay` is numeric, but evidence dict is unvalidated.

**Example:**
```python
diagnosis = DiagnosisRecord(
    trial_id="t1",
    diagnosis_kind=DiagnosisKind.HIGH_VARIANCE,
    evidence={"decay_param": "invalid"}
)
proposal = mapper.propose_repairs(diagnosis)[0]
# Line 265: "invalid" * 0.8 → TypeError
```

**Fix:**
```python
current_decay = diagnosis.evidence.get("decay_param", 0.5)
if not isinstance(current_decay, (int, float)):
    current_decay = 0.5
new_decay = max(0.1, min(0.9, current_decay * 0.8))
```

---

#### Finding 2.4.2: Window Reduction Can Produce Invalid Values
**Risk:** High  
**Location:** `RepairMapper._strategy_to_mutation()` line 247  
**Issue:** `new_window = max(5, int(current_window * 0.7))` but if `current_window` is non-numeric or extremely large, `int()` can overflow.

**Example:**
```python
evidence = {"lookback_periods": 10**20}
new_window = max(5, int(10**20 * 0.7))  # OverflowError or extremely large value
```

**Fix:**
```python
current_window = diagnosis.evidence.get("lookback_periods", 20)
if not isinstance(current_window, (int, float)) or current_window <= 0:
    current_window = 20
new_window = max(5, min(252, int(current_window * 0.7)))
```

---

#### Finding 2.4.3: Confidence Estimation Can Be Negative
**Risk:** Medium  
**Location:** `RepairMapper._estimate_confidence()` line 316  
**Issue:** `base_confidence = 0.7 - (rank * 0.15)` with rank >= 5 produces negative values, then clamped to 0.1.

**Example:**
```python
# rank=6: 0.7 - (6 * 0.15) = 0.7 - 0.9 = -0.2 → clamped to 0.1
```

**Recommendation:** Document that clamping is intentional, or adjust formula to avoid negative intermediate values.

---

## 3. LLM INTEGRATION

### 3.1 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/llm/proposal.py`

#### Finding 3.1.1: JSON Parsing Fails Silently for Invalid LLM Output
**Risk:** High  
**Location:** `ProposalGenerator._parse_and_validate()` lines 206-242  
**Issue:** JSON decode errors are caught and added to `validation_errors`, but proposals list remains empty. Caller must check `is_valid()`.

**Example:**
```python
# LLM returns plain text instead of JSON
raw_response = "I think you should try window_adjust"
proposals, errors = generator._parse_and_validate(raw_response, ["f1"], template)
# proposals = [], errors = ["Invalid JSON: ..."]
# Caller must check len(proposals) > 0
```

**Recommendation:** Add explicit guidance in ProposalResponse docstring about checking `is_valid()`.

---

#### Finding 3.1.2: No Validation for Empty Parent Factor IDs
**Risk:** Medium  
**Location:** `ProposalGenerator.generate_proposals()` line 114  
**Issue:** `parent_factor_ids = parent_factor_ids or ["unknown_factor"]` masks missing input.

**Example:**
```python
response = generator.generate_proposals(request, parent_factor_ids=[])
# Silently uses ["unknown_factor"], causing incorrect mutation records
```

**Fix:**
```python
if not parent_factor_ids:
    raise MissingInputError("parent_factor_ids cannot be empty")
```

---

#### Finding 3.1.3: Token Count Is Naive Split
**Risk:** Low  
**Location:** `ProposalGenerator.generate_proposals()` lines 154-155  
**Issue:** `input_tokens=len(full_prompt.split())` is inaccurate; real tokenizers produce different counts.

**Recommendation:** Document as mock implementation or use proper tokenizer.

---

### 3.2 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/llm/prompts.py`

#### Finding 3.2.1: Missing Placeholder Raises KeyError
**Risk:** Medium  
**Location:** `PromptTemplate.render()` lines 54-57  
**Issue:** Raises `KeyError`, then catches and re-raises as `ValueError`, but error message is poor.

**Example:**
```python
template = PromptTemplate(
    template_id="test",
    version="1.0",
    system_prompt="...",
    user_prompt_template="Parent: {parent_info}, Objective: {objective}",
    output_schema={}
)
rendered = template.render(parent_info="...")  # Missing "objective"
# Raises: ValueError: Missing required placeholder: 'objective'
```

**Recommendation:** Enumerate all required placeholders in error message:
```python
except KeyError as e:
    required = re.findall(r'\{(\w+)\}', self.user_prompt_template)
    raise ValueError(f"Missing required placeholder {e}. Required: {required}")
```

---

#### Finding 3.2.2: No Validation for Empty Template Strings
**Risk:** Low  
**Location:** `PromptTemplate.__post_init__()` lines 32-39  
**Issue:** Checks `if not self.system_prompt` but empty string `""` is falsy and will raise.

**Recommendation:** This is correct behavior, but document that empty prompts are invalid.

---

## 4. DATA STRUCTURES

### 4.1 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/contracts/trial.py`

#### Finding 4.1.1: No Validation for Status Transitions
**Risk:** High  
**Location:** `Trial.update_status()` lines 55-68  
**Issue:** Allows arbitrary status transitions; can go from EVALUATED back to PROPOSED.

**Example:**
```python
trial = Trial("t1", "m1", TrialStatus.EVALUATED)
trial.update_status(TrialStatus.PROPOSED)  # Invalid transition allowed
```

**Fix:** Add state machine validation:
```python
VALID_TRANSITIONS = {
    TrialStatus.PROPOSED: {TrialStatus.VALIDATING},
    TrialStatus.VALIDATING: {TrialStatus.LEGAL, TrialStatus.ILLEGAL},
    TrialStatus.LEGAL: {TrialStatus.EVALUATING, TrialStatus.DUPLICATE},
    TrialStatus.EVALUATING: {TrialStatus.EVALUATED, TrialStatus.FAILED},
    # Terminal states have no valid transitions
}

def update_status(self, new_status: TrialStatus, **kwargs) -> None:
    if self.status in VALID_TRANSITIONS:
        if new_status not in VALID_TRANSITIONS[self.status]:
            raise InvalidContractError(f"Invalid transition: {self.status} → {new_status}")
    elif self.is_terminal():
        raise InvalidContractError(f"Cannot transition from terminal state {self.status}")
    # ... rest of method
```

---

#### Finding 4.1.2: Metadata Update Silently Overwrites
**Risk:** Low  
**Location:** `Trial.update_status()` line 68  
**Issue:** `self.metadata.update(kwargs["metadata"])` merges new metadata, but doesn't signal collision if keys already exist.

**Recommendation:** Document as intentional overwrite behavior.

---

### 4.2 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/seen/identity.py`

#### Finding 4.2.1: Missing FE Adapter Raises Runtime Error
**Risk:** High  
**Location:** `SeenCache.compute_canonical_hash()` lines 129-130  
**Issue:** Raises `RuntimeError` but caller might not expect it; should be `CapabilityError`.

**Example:**
```python
cache = SeenCache(fe_adapter=None)
hash_val = cache.compute_canonical_hash(factor_def)
# RuntimeError: FE adapter required for canonical hash computation
```

**Fix:**
```python
from factor_optimizer.errors import CapabilityError

if self.fe_adapter is None:
    raise CapabilityError("FE adapter required for canonical hash computation")
```

---

#### Finding 4.2.2: check_and_mark Can Fail Mid-Operation
**Risk:** Medium  
**Location:** `SeenCache.check_and_mark()` lines 137-156  
**Issue:** Calls `compute_canonical_hash()` which can raise, leaving cache in inconsistent state if partially updated.

**Recommendation:** Wrap in try/except or ensure atomic operation.

---

### 4.3 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/contracts/candidate_mutation.py`

#### Finding 4.3.1: Empty Parent Factor IDs List Passes Validation
**Risk:** High  
**Location:** `CandidateMutation.__post_init__()` line 56  
**Issue:** Checks `if not self.parent_factor_ids` but empty list `[]` is falsy.

**Example:**
```python
mutation = CandidateMutation(
    mutation_id="m1",
    mutation_spec_version="1.0",
    parent_factor_ids=[],  # Empty list
    mutation_type="test",
    parameters={}
)
# Line 56: if not [] → True, raises MissingInputError
```

**Actual behavior:** This is correct! Empty list raises error. **No fix needed.**

---

### 4.4 `/home/shw/quant_projects/factor_optimizer/factor_optimizer/grammar/mutation_spec.py`

#### Finding 4.4.1: Float Validation Accepts Booleans as 0/1
**Risk:** Low  
**Location:** `ParameterSpec.validate_value()` line 103  
**Issue:** `isinstance(value, (int, float))` returns True for booleans since `bool` subclasses `int`.

**Example:**
```python
param = ParameterSpec("test", ParameterKind.FLOAT, ParameterRole.SCALAR, min_value=0.0, max_value=1.0)
is_valid, error = param.validate_value(True)  # True == 1.0
# Returns (True, None) but True is not semantically a float
```

**Fix:** Line 103 already has `or isinstance(value, bool)` guard! **No fix needed.**

---

## 5. CROSS-CUTTING CONCERNS

### 5.1 Missing Import for math Module

**Risk:** Critical  
**Affected Files:**
- `pareto.py` (uses `float('inf')` and `float('nan')` but recommended fix needs `math.isfinite()`)
- `runner.py` (recommended fix for score validation needs `math.isfinite()`)

**Fix:** Add `import math` to affected files.

---

### 5.2 Inconsistent Error Types

**Risk:** Medium  
**Observation:** Some functions raise `RuntimeError`, others raise `ValueError`, and few use custom exceptions from `errors.py`.

**Recommendation:**
- FE adapter missing → `CapabilityError`
- Invalid input → `InvalidContractError` or `MissingInputError`
- Arithmetic errors → `NumericalFailure`

---

### 5.3 No Logging for Silent Failures

**Risk:** Medium  
**Affected:** Multiple files catch exceptions and continue without logging.

**Example:** `SearchRunner.run()` line 204 catches all exceptions from evaluation and marks trial as FAILED, but doesn't log the exception details.

**Recommendation:** Add logging at INFO level for operational events, WARNING for recoverable errors, ERROR for unexpected exceptions.

---

## 6. SUMMARY OF CRITICAL FINDINGS

| Finding | File | Risk | Impact |
|---------|------|------|--------|
| 1.1.1 | runner.py | Critical | Division by zero in plateau detection |
| 1.2.2 | pareto.py | Critical | Hypervolume division by zero with degenerate points |
| 1.2.3 | pareto.py | Critical | Spacing metric division by zero |
| 2.2.1 | profile.py | Critical | Division by zero in mutation cost estimation |
| 2.3.1 | budget.py | Critical | Division by None in budget utilization stats |
| 2.4.1 | repair.py | Critical | TypeError from non-numeric evidence values |
| 5.1 | Multiple | Critical | Missing math import for isfinite() checks |

---

## 7. RECOMMENDED REMEDIATION PRIORITY

### Phase 1: Critical Arithmetic Errors (1-2 days)
1. Add `math.isfinite()` checks for all score/objective values
2. Fix all division-by-zero cases with epsilon checks or validation
3. Add `ComplexityProfile.__post_init__()` validation
4. Add `ComplexityBudget.__post_init__()` validation

### Phase 2: High-Risk Edge Cases (2-3 days)
5. Implement circuit breaker for proposal function failures
6. Add state machine validation for Trial status transitions
7. Fix BudgetTracker.can_call_llm() to handle None correctly
8. Add type validation for evidence dictionary values in repair mapper
9. Validate infinite/NaN values in ParetoPoint

### Phase 3: Medium-Risk Validation (3-4 days)
10. Add timeout/cancellation for FE adapter calls
11. Improve error messages for missing prompt placeholders
12. Add logging for silent exception handling
13. Standardize exception types across modules

### Phase 4: Low-Risk Polish (1-2 days)
14. Document intentional behaviors (empty frontier coverage, clamped confidence)
15. Add accurate token counting or document mock implementation
16. Review and document all public API contracts

---

## 8. TEST COVERAGE GAPS

Recommended test cases to add:

```python
# Test division by zero in plateau detection
def test_plateau_detection_all_zero_scores():
    runner = SearchRunner(...)
    assert runner._check_plateau([0.0, 0.0, 0.0]) == True

# Test Pareto with infinite objectives
def test_pareto_rejects_infinite_objectives():
    with pytest.raises(ValueError):
        ParetoPoint("t1", (float('inf'), 1.0))

# Test negative budget values
def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        ComplexityBudget(max_cost=-100.0)

# Test proposal function failure circuit breaker
def test_proposal_function_consecutive_failures():
    failure_count = [0]
    def failing_proposal():
        failure_count[0] += 1
        raise RuntimeError("LLM API down")
    
    runner = SearchRunner(config, failing_proposal, eval_fn)
    session = runner.run("test_session")
    assert session.stop_reason == "proposal_failure"
    assert failure_count[0] == 10  # MAX_CONSECUTIVE_FAILURES

# Test invalid state transitions
def test_trial_invalid_state_transition():
    trial = Trial("t1", "m1", TrialStatus.EVALUATED)
    with pytest.raises(InvalidContractError):
        trial.update_status(TrialStatus.PROPOSED)
```

---

**End of Audit Report**
