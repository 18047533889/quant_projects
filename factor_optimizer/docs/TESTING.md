# FactorOptimizer Testing Guide

**Version:** 0.1.0  
**Last Updated:** 2026-08-14

## Quick Start

```bash
cd /home/shw/quant_projects/factor_optimizer
pytest tests/
```

## Test Organization

```
tests/
├── contracts/              # Contract validation
│   ├── test_candidate_mutation.py
│   ├── test_search_budget.py
│   └── test_trial.py
├── grammar/               # Mutation grammar
│   ├── test_mutation_spec.py
│   ├── test_registry.py
│   └── test_validation.py
├── seen/                  # Deduplication
│   └── test_identity.py
├── complexity/            # Complexity estimation
│   └── test_profile.py
├── policy/               # Admission & repair
│   ├── test_decisions.py
│   └── test_repair.py
├── search/               # Search orchestration
│   ├── test_runner.py
│   ├── test_multifidelity.py
│   ├── test_pareto.py
│   ├── test_plateau.py
│   └── test_lineage.py
└── llm/                  # LLM stubs
    ├── test_prompts.py
    ├── test_proposal.py
    └── test_records.py
```

## Key Test Patterns

### Budget Tracking
```python
def test_budget_exhaustion():
    budget = SearchBudget(max_trials=10)
    tracker = BudgetTracker(budget)
    
    for i in range(10):
        assert tracker.can_propose_trial()
        tracker.record_trial()
    
    assert tracker.is_exhausted()
```

### Mutation Validation
```python
def test_valid_mutation():
    registry = get_mutation_registry()
    validator = MutationValidator(registry)
    
    mutation = CandidateMutation(...)
    result = validator.validate(mutation)
    
    assert result.is_valid
```

### Pareto Frontier
```python
def test_dominance():
    point1 = ParetoPoint("t1", (0.05, 0.2))  # High IC, high turnover
    point2 = ParetoPoint("t2", (0.06, 0.15)) # Higher IC, lower turnover
    
    assert point2.dominates(point1)
```

## Coverage Goals
- **Contracts:** 100%
- **Grammar:** 100%
- **Search logic:** >95%
- **Overall:** >90%

---

**Last Updated:** 2026-08-14
