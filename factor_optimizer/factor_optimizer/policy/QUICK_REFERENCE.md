# Repair Strategies Quick Reference

## Common Use Cases

### 1. Low IC Factor (Weak Predictive Power)
```python
from factor_optimizer.policy import DiagnosisRecord, DiagnosisKind, RepairMapper

diagnosis = DiagnosisRecord(
    trial_id="trial_001",
    diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
    evidence={"ic": 0.03, "ic_std": 0.05}
)

mapper = RepairMapper()
proposals = mapper.propose_repairs(diagnosis)
# Primary: ADD_INTERACTION → add cross-sectional interaction term
```

### 2. High Turnover Factor (Excessive Churn)
```python
diagnosis = DiagnosisRecord(
    trial_id="trial_002",
    diagnosis_kind=DiagnosisKind.HIGH_TURNOVER,
    evidence={"turnover_rate": 0.85, "prediction_horizon": 1}
)

proposals = mapper.propose_repairs(diagnosis)
# Primary: INCREASE_HORIZON → lengthen prediction horizon
```

### 3. High Complexity Factor (Exceeds Budget)
```python
diagnosis = DiagnosisRecord(
    trial_id="trial_003",
    diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
    evidence={"lookback_periods": 120, "operator_count": 50}
)

proposals = mapper.propose_repairs(diagnosis)
# Primary: REDUCE_WINDOW → shorten lookback period
```

### 4. Numerical Instability (NaN/Inf Production)
```python
diagnosis = DiagnosisRecord(
    trial_id="trial_004",
    diagnosis_kind=DiagnosisKind.NUMERICAL_INSTABILITY,
    evidence={"nan_count": 150, "total_observations": 1000}
)

proposals = mapper.propose_repairs(diagnosis)
# Primary: WINSORIZE → cap outliers at 1st/99th percentiles
```

## Quick API Reference

### Generate Repairs
```python
mapper = RepairMapper()
proposals = mapper.propose_repairs(diagnosis, max_proposals=3)
ranked = mapper.rank_repairs(proposals)
mutation_spec = mapper.generate_mutation_spec(ranked[0])
```

### Make Admission Decision
```python
from factor_optimizer.policy import AdmissionCriteria, AdmissionPolicy

criteria = AdmissionCriteria(
    max_complexity_cost=100.0,
    min_expected_value=0.2
)
policy = AdmissionPolicy(criteria)

decision = policy.decide(
    mutation_id="mut_001",
    trial_id="trial_001",
    complexity_cost=75.0,
    expected_value=0.5
)

if decision.is_approved():
    # Execute mutation
    pass
```

## Diagnosis Types

| Code | Description | Primary Strategy |
|------|-------------|------------------|
| `LOW_SIGNAL` | Weak predictive power (low IC) | `ADD_INTERACTION` |
| `HIGH_TURNOVER` | Excessive portfolio churn | `INCREASE_HORIZON` |
| `HIGH_COMPLEXITY` | Exceeds compute/memory budget | `REDUCE_WINDOW` |
| `POOR_COVERAGE` | Insufficient valid data points | `REDUCE_WINDOW` |
| `HIGH_VARIANCE` | Unstable across periods | `ADD_REGULARIZATION` |
| `NUMERICAL_INSTABILITY` | NaN/Inf production | `WINSORIZE` |
| `OVERFITTING` | Train/test performance gap | `ADD_REGULARIZATION` |
| `TIMING_VIOLATION` | Look-ahead bias or PIT violation | `LAG_CORRECTION` |
| `SEMANTIC_DUPLICATE` | Equivalent to existing factor | `ABANDON` |
| `DOMAIN_MISMATCH` | Wrong data domain | `ABANDON` |

## Repair Strategies

| Strategy | Mutation Type | Effect |
|----------|---------------|--------|
| `ADD_INTERACTION` | `add_interaction` | Add cross-sectional interaction term |
| `INCREASE_HORIZON` | `horizon_adjust` | Add 5 days to prediction horizon |
| `REDUCE_WINDOW` | `window_adjust` | Reduce lookback by 30% |
| `INCREASE_WINDOW` | `window_adjust` | Increase lookback by 50% |
| `WINSORIZE` | `add_winsorization` | Cap at 1st/99th percentiles |
| `ADD_REGULARIZATION` | `add_regularization` | Add L2 regularization (α=0.1) |
| `LAG_CORRECTION` | `lag_adjustment` | Add 1-day lag |
| `ADJUST_DECAY` | `decay_adjust` | Reduce decay parameter by 20% |
| `OPERATOR_SWAP` | `operator_swap` | Replace with alternative operator |
| `FILTER_UNIVERSE` | `filter_universe` | Restrict to coverage ≥70% |
| `THRESHOLD_TUNE` | `threshold_adjust` | Reduce threshold by 10% |
| `CHANGE_NORMALIZATION` | `normalization_change` | Switch to robust normalization |
| `ABANDON` | N/A | No repair available |

## Admission Verdicts

- `ADMITTED`: Approved for evaluation
- `CONDITIONAL`: Admitted with constraints
- `REJECTED`: Will not evaluate
- `DEFERRED`: Pending resource/budget

## Rejection Reasons

- `BUDGET_EXCEEDED`: Exceeds resource budget
- `DUPLICATE`: Semantic duplicate of known factor
- `COMPLEXITY_LIMIT`: Too complex
- `LOW_EXPECTED_VALUE`: Expected value below threshold
- `PARENT_QUALITY`: Parent factor quality insufficient
- `DOMAIN_VIOLATION`: Wrong domain/market
- `TIMING_VIOLATION`: Look-ahead or PIT violation
- `ILLEGAL`: Violates grammar/legality rules
- `POLICY_VIOLATION`: Violates organizational policy

## Example Workflow

```python
# 1. Diagnose failure
diagnosis = DiagnosisRecord(
    trial_id="trial_001",
    diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
    evidence={"ic": 0.03}
)

# 2. Generate repairs
mapper = RepairMapper()
proposals = mapper.propose_repairs(diagnosis)
ranked = mapper.rank_repairs(proposals)

# 3. Create mutation spec
mutation_spec = mapper.generate_mutation_spec(ranked[0])

# 4. Make admission decision
policy = AdmissionPolicy()
decision = policy.decide(
    mutation_id="mut_001",
    trial_id=diagnosis.trial_id,
    complexity_cost=75.0,
    expected_value=0.5
)

# 5. Execute if approved
if decision.is_approved():
    print(f"✓ Mutation {decision.mutation_id} admitted")
else:
    print(f"✗ Rejected: {decision.primary_rejection_reason().value}")
```

## Test Commands

```bash
# All policy tests
pytest tests/policy/ -v

# Repair tests only
pytest tests/policy/test_repair.py -v

# Integration tests
pytest tests/policy/test_integration.py -v

# Run examples
python3 examples/repair_workflow_examples.py
```

## Key Metrics

- **10** diagnosis types
- **13** repair strategies
- **4** admission verdicts
- **9** rejection reasons
- **60** test cases (all passing)
- **704** lines of implementation code
- **1,054** lines of test code
