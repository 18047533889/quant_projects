# Policy Module: Repair Strategies & Admission Decisions

This module implements the diagnosis-to-repair mapping and admission decision logic for the factor optimizer.

## Overview

The policy module consists of two main components:

1. **Repair Strategies** (`repair.py`): Maps failure diagnoses to concrete repair mutations
2. **Admission Decisions** (`decisions.py`): Evaluates mutation proposals against admission criteria

## Repair Strategies

### Diagnosis-to-Repair Mapping

The `RepairMapper` encodes domain knowledge about which mutations address specific failure modes:

| Diagnosis | Primary Strategy | Use Case |
|-----------|-----------------|----------|
| `LOW_SIGNAL` | `ADD_INTERACTION` | Low IC → add interaction terms to enhance predictive power |
| `HIGH_TURNOVER` | `INCREASE_HORIZON` | Excessive churn → lengthen prediction horizon |
| `HIGH_COMPLEXITY` | `REDUCE_WINDOW` | Exceeds compute budget → shorten lookback period |
| `POOR_COVERAGE` | `REDUCE_WINDOW` | Insufficient data points → require less history |
| `HIGH_VARIANCE` | `ADD_REGULARIZATION` | Unstable across periods → add smoothing |
| `NUMERICAL_INSTABILITY` | `WINSORIZE` | NaN/Inf production → cap extreme outliers |
| `OVERFITTING` | `ADD_REGULARIZATION` | Train/test gap → reduce model complexity |
| `TIMING_VIOLATION` | `LAG_CORRECTION` | Look-ahead bias → adjust timing/alignment |
| `SEMANTIC_DUPLICATE` | `ABANDON` | Equivalent to existing factor → no repair |
| `DOMAIN_MISMATCH` | `ABANDON` | Wrong data domain → no repair |

### Key Features

#### 1. Diagnosis Records

```python
from factor_optimizer.policy.repair import DiagnosisRecord, DiagnosisKind

diagnosis = DiagnosisRecord(
    trial_id="trial_001",
    diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
    severity=0.7,
    evidence={"ic": 0.02, "lookback_periods": 20},
    secondary_diagnoses=[DiagnosisKind.HIGH_VARIANCE]
)
```

#### 2. Repair Proposals

```python
from factor_optimizer.policy.repair import RepairMapper

mapper = RepairMapper()
proposals = mapper.propose_repairs(diagnosis, max_proposals=3)

# Primary proposal
print(proposals[0].strategy)  # RepairStrategy.ADD_INTERACTION
print(proposals[0].mutation_type)  # "add_interaction"
print(proposals[0].parameters)  # {"interaction_type": "cross_sectional"}
print(proposals[0].confidence)  # 0.7
print(proposals[0].expected_improvement)  # "Enhance predictive power with interaction terms"
```

#### 3. Priority Ranking

```python
# Rank multiple proposals by priority
ranked = mapper.rank_repairs(proposals)

# Priority score = base_weight × confidence × severity_factor
# Higher priority repairs are ranked first
```

#### 4. MutationSpec Generation

```python
# Generate MutationSpec compatible with CandidateMutation
mutation_spec = mapper.generate_mutation_spec(proposals[0])

# mutation_spec contains:
# - mutation_type: str
# - parameters: Dict[str, Any]
# - provenance: Dict with source, diagnosis_kind, repair_strategy, confidence
# - expected_improvement: str
# - evidence_context: Dict (from diagnosis evidence)
```

### Repair Strategies

| Strategy | Mutation Type | Parameters | Effect |
|----------|---------------|------------|--------|
| `REDUCE_WINDOW` | `window_adjust` | `new_window` | Reduce lookback by 30% (min 5 periods) |
| `INCREASE_WINDOW` | `window_adjust` | `new_window` | Increase lookback by 50% (max 252 periods) |
| `INCREASE_HORIZON` | `horizon_adjust` | `new_horizon` | Add 5 days to prediction horizon (max 20) |
| `ADD_INTERACTION` | `add_interaction` | `interaction_type` | Add cross-sectional interaction term |
| `ADJUST_DECAY` | `decay_adjust` | `new_decay` | Reduce decay parameter by 20% |
| `WINSORIZE` | `add_winsorization` | `lower_quantile`, `upper_quantile` | Cap at 1st/99th percentiles |
| `ADD_REGULARIZATION` | `add_regularization` | `regularization_strength` | Add L2 regularization (α=0.1) |
| `LAG_CORRECTION` | `lag_adjustment` | `additional_lag_days` | Add 1-day lag |
| `OPERATOR_SWAP` | `operator_swap` | `target_operator`, `current_operator` | Replace with alternative operator |
| `FILTER_UNIVERSE` | `filter_universe` | `min_coverage` | Restrict to higher coverage (≥70%) |
| `THRESHOLD_TUNE` | `threshold_adjust` | `new_threshold` | Reduce threshold by 10% |
| `CHANGE_NORMALIZATION` | `normalization_change` | `normalization_method` | Switch to robust normalization |

### Confidence Scoring

Repair confidence is calculated as:

```
base_confidence = 0.7 - (rank × 0.15)
if severity > 0.8: base_confidence *= 0.8
if len(secondary_diagnoses) > 2: base_confidence *= 0.85
final_confidence = clamp(base_confidence, 0.1, 0.95)
```

## Admission Decisions

### Admission Criteria

```python
from factor_optimizer.policy.decisions import AdmissionCriteria, AdmissionPolicy

criteria = AdmissionCriteria(
    max_complexity_cost=100.0,
    min_expected_value=0.2,
    max_lookback_periods=252,
    allowed_domains=["equity", "futures"],
    allowed_sources=["market_data"],
    require_parent_evidence=True,
    min_parent_quality=0.6
)

policy = AdmissionPolicy(criteria=criteria)
```

### Making Decisions

```python
decision = policy.decide(
    mutation_id="mut_001",
    trial_id="trial_001",
    complexity_cost=80.0,
    expected_value=0.5,
    parent_quality=0.7,
    domains=["equity"],
    lookback_periods=60
)

print(decision.verdict)  # AdmissionVerdict.ADMITTED
print(decision.is_approved())  # True
```

### Admission Verdicts

| Verdict | Meaning | Status |
|---------|---------|--------|
| `ADMITTED` | Approved for evaluation | Approved |
| `CONDITIONAL` | Admitted with constraints | Approved |
| `REJECTED` | Will not evaluate | Not approved |
| `DEFERRED` | Pending resource/budget | Not approved |

### Rejection Reasons

- `BUDGET_EXCEEDED`: Exceeds resource budget
- `DUPLICATE`: Semantic duplicate of known factor
- `ILLEGAL`: Violates grammar/legality rules
- `DOMAIN_VIOLATION`: Wrong domain/market
- `TIMING_VIOLATION`: Look-ahead or PIT violation
- `LOW_EXPECTED_VALUE`: Expected value below threshold
- `COMPLEXITY_LIMIT`: Too complex
- `PARENT_QUALITY`: Parent factor quality insufficient
- `POLICY_VIOLATION`: Violates organizational policy

### Decision Tracking

```python
# Retrieve decisions
decision = policy.get_decision("decision_trial_001")
trial_decisions = policy.get_decisions_for_trial("trial_001")

# Get statistics
stats = policy.get_admission_stats()
print(stats)
# {
#     "total": 10,
#     "admitted": 7,
#     "rejected": 3,
#     "deferred": 0,
#     "conditional": 0,
#     "admission_rate": 0.7,
#     "rejection_reasons": {"complexity_limit": 2, "low_expected_value": 1}
# }
```

## Integration Workflow

### Complete Diagnosis → Repair → Admission Flow

```python
from factor_optimizer.policy.repair import DiagnosisRecord, DiagnosisKind, RepairMapper
from factor_optimizer.policy.decisions import AdmissionCriteria, AdmissionPolicy
from factor_optimizer.contracts.candidate_mutation import CandidateMutation
from datetime import datetime

# 1. Create diagnosis from evaluation failure
diagnosis = DiagnosisRecord(
    trial_id="trial_001",
    diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
    severity=0.6,
    evidence={"ic": 0.03, "lookback_periods": 20}
)

# 2. Generate repair proposals
mapper = RepairMapper()
proposals = mapper.propose_repairs(diagnosis, max_proposals=3)
ranked_proposals = mapper.rank_repairs(proposals)

# 3. Convert to mutation spec
best_proposal = ranked_proposals[0]
mutation_spec = mapper.generate_mutation_spec(best_proposal)

# 4. Create candidate mutation
candidate = CandidateMutation(
    mutation_id="mut_001",
    mutation_spec_version="1.0",
    parent_factor_ids=["parent_001"],
    mutation_type=mutation_spec["mutation_type"],
    parameters=mutation_spec["parameters"],
    mechanism_hypothesis=mutation_spec["expected_improvement"],
    complexity_estimate={"cost": 75.0},
    created_at=datetime.now(),
    producer="repair_mapper"
)

# 5. Make admission decision
criteria = AdmissionCriteria(
    max_complexity_cost=100.0,
    min_expected_value=0.2
)
policy = AdmissionPolicy(criteria=criteria)

decision = policy.decide(
    mutation_id=candidate.mutation_id,
    trial_id=diagnosis.trial_id,
    complexity_cost=75.0,
    expected_value=0.5
)

# 6. Execute if admitted
if decision.is_approved():
    print(f"Mutation {candidate.mutation_id} admitted for evaluation")
    # Proceed with evaluation
else:
    print(f"Mutation rejected: {decision.primary_rejection_reason()}")
```

## Design Principles

1. **Fail-closed**: Unrepairable failures (e.g., semantic duplicates) generate no proposals
2. **Evidence-based**: Repairs use diagnosis evidence to parameterize mutations
3. **Confidence decay**: Lower-ranked strategies have reduced confidence
4. **Severity penalty**: High-severity diagnoses reduce repair confidence
5. **Fallback chains**: Each proposal includes alternative strategies if primary fails
6. **Provenance tracking**: Full audit trail from diagnosis → proposal → mutation → decision

## Testing

Run the test suite:

```bash
# All policy tests
pytest tests/policy/ -v

# Repair tests only
pytest tests/policy/test_repair.py -v

# Decision tests only
pytest tests/policy/test_decisions.py -v

# Integration tests
pytest tests/policy/test_integration.py -v
```

## Extension Points

### Adding New Diagnoses

```python
class DiagnosisKind(Enum):
    # ... existing diagnoses
    NEW_DIAGNOSIS = "new_diagnosis"

# Update RepairMapper._rules
self._rules[DiagnosisKind.NEW_DIAGNOSIS] = [
    RepairStrategy.SOME_STRATEGY,
    RepairStrategy.FALLBACK_STRATEGY,
    RepairStrategy.ABANDON,
]
```

### Adding New Repair Strategies

```python
class RepairStrategy(Enum):
    # ... existing strategies
    NEW_STRATEGY = "new_strategy"

# Update RepairMapper._strategy_to_mutation
elif strategy == RepairStrategy.NEW_STRATEGY:
    # Extract evidence, compute parameters
    return "mutation_type", {"param1": value1}

# Update priority weights
self._priority_weights[RepairStrategy.NEW_STRATEGY] = 0.75

# Update descriptions
descriptions[RepairStrategy.NEW_STRATEGY] = "Expected improvement"
```

### Custom Admission Criteria

```python
class CustomAdmissionPolicy(AdmissionPolicy):
    def decide(self, mutation_id, trial_id, **kwargs):
        # Add custom logic
        custom_check = self._custom_validation(kwargs)
        if not custom_check:
            rejection_reasons.append(RejectionReason.POLICY_VIOLATION)
        
        return super().decide(mutation_id, trial_id, **kwargs)
```

## References

- Diagnosis module: (to be implemented)
- Evaluation module: `factor_optimizer.search.runner`
- Mutation contracts: `factor_optimizer.contracts.candidate_mutation`
