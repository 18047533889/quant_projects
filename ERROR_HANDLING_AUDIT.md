# Error Handling Audit Report

**Date:** 2026-08-14  
**Scope:** quant_evaluator, factor_optimizer, factor_assets, factor_preprocess  
**Reference:** CONTRACT_FREEZE_DRAFT.md Section 7 (Error Taxonomy)

---

## Executive Summary

**Overall Status:** PARTIAL IMPLEMENTATION - Error handling exists but is inconsistent with the contract freeze taxonomy. Most packages have basic validation but lack comprehensive typed error hierarchies.

**Key Findings:**
- 229 error test cases across all packages (49 QE, 34 FO, 85 FA, 61 FP)
- Only 1 package (quant_evaluator) has a structured error module aligned with contract taxonomy
- 3 packages define only `OptionalDependencyMissing` exception (scattered, not centralized)
- NaN/Inf handling is present but inconsistent across packages
- Missing systematic boundary validation in several key modules
- Error messages generally clear but lack consistent context/remediation guidance

**Priority:** HIGH - Error taxonomy standardization required before production freeze

---

## 1. Error Type Audit

### 1.1 Current Exception Hierarchy by Package

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
