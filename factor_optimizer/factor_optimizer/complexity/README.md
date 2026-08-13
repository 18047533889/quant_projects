# Complexity Profile System

The complexity profile system provides estimation and budget tracking for factor complexity during optimization.

## Components

### ComplexityProfile
A dataclass that captures complexity metrics for a factor:
- `operator_count`: Number of operators in the factor
- `max_depth`: Maximum AST depth
- `lookback_periods`: Maximum lookback window
- `stateful_operators`: Number of stateful operators
- `estimated_cost`: Relative compute cost estimate
- `memory_estimate`: Estimated memory usage in MB

### ComplexityEstimator
Estimates complexity through a FE adapter:
```python
from factor_optimizer.complexity import ComplexityEstimator, ComplexityProfile

# With FE adapter
estimator = ComplexityEstimator(fe_adapter=my_fe_adapter)
profile = estimator.estimate(factor_definition)

# Estimate mutation delta
new_profile = estimator.estimate_mutation_delta(
    parent_profile=profile,
    mutation_type="window_adjust",
    parameters={"new_window": 40}
)
```

### ComplexityBudget
Define and enforce complexity constraints:
```python
from factor_optimizer.complexity import ComplexityBudget, create_default_budget

# Create custom budget
budget = ComplexityBudget(
    max_cost=100.0,
    max_operator_count=20,
    max_lookback_periods=252,
    strict=True  # Raise exception if exceeded
)

# Check if profile is within budget
is_ok = budget.is_within_budget(profile)

# Enforce budget (raises BudgetExceededError if violated)
budget.enforce(profile)
```

### BudgetTracker
Track budget usage during optimization:
```python
from factor_optimizer.complexity import BudgetTracker

tracker = BudgetTracker(budget=budget)

# Record each candidate
for candidate in candidates:
    profile = estimator.estimate(candidate.definition)
    is_within = tracker.record(profile, candidate.id)
    
    if not is_within:
        print(f"Candidate {candidate.id} exceeds budget")

# Get statistics
stats = tracker.stats()
print(f"Evaluated: {stats['total_evaluated']}")
print(f"Violations: {stats['violations_count']}")
print(f"Cost utilization: {stats['budget_cost_utilization']:.1%}")
```

## Integration with FE

The complexity system delegates actual complexity analysis to FactorEngine through the adapter protocol:

```python
from factor_optimizer.adapters import create_fe_adapter

# Create FE adapter (requires factor_engine installed)
fe_adapter = create_fe_adapter()

# Use with estimator
estimator = ComplexityEstimator(fe_adapter=fe_adapter)
profile = estimator.estimate(factor)
```

## End-to-End Example

```python
from factor_optimizer.complexity import (
    ComplexityEstimator,
    ComplexityBudget,
    BudgetTracker,
    create_default_budget
)

# Setup
estimator = ComplexityEstimator(fe_adapter=fe_adapter)
budget = create_default_budget(cost_multiplier=2.0)
tracker = BudgetTracker(budget=budget)

# During optimization
for mutation in mutations:
    # Estimate complexity
    profile = estimator.estimate(mutation.factor_definition)
    
    # Check budget
    if tracker.record(profile, mutation.id):
        # Within budget - proceed with evaluation
        result = evaluate(mutation)
    else:
        # Exceeds budget - skip
        print(f"Skipping {mutation.id} - exceeds complexity budget")

# Review results
stats = tracker.stats()
print(f"Budget utilization: {stats['budget_cost_utilization']:.1%}")
print(f"Rejected {stats['violations_count']} candidates for complexity")
```

## Design Principles

1. **Adapter-based**: FO does not reimplement FE's complexity analysis. It wraps FE through an adapter protocol.

2. **Multi-dimensional budgets**: Supports constraints on cost, operator count, lookback, depth, memory, etc.

3. **Strict vs. non-strict mode**: Budgets can raise exceptions (strict) or just track violations (non-strict).

4. **Mutation deltas**: Can estimate complexity changes from mutations using heuristics, avoiding full recompilation.

5. **History tracking**: BudgetTracker maintains full history for post-analysis and debugging.
