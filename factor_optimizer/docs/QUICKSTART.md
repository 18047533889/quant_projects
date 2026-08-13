# FactorOptimizer Quick Start

**5-Minute Guide to Factor Search and Mutation**

## Installation

```bash
cd /home/shw/quant_projects/factor_optimizer
pip install -e .

# With optional adapters
pip install -e ".[factor_engine,quant_evaluator]"
```

## Basic Example

Define and validate a mutation:

```python
from factor_optimizer.grammar import get_mutation_registry, MutationValidator
from factor_optimizer.contracts import CandidateMutation

# Get available mutations
registry = get_mutation_registry()
print(f"Available mutations: {registry.list_mutation_types()}")

# Create mutation proposal
mutation = CandidateMutation(
    mutation_id="mut_001",
    mutation_spec_version="1.0",
    parent_factor_ids=["F_parent"],
    mutation_type="window_adjust",
    parameters={"window": 20, "adjustment": +5},
    mechanism_hypothesis="Longer window may smooth noise",
    expected_signatures=["higher_stability", "lower_turnover"],
)

# Validate
validator = MutationValidator(registry=registry)
result = validator.validate(mutation)

if result.is_valid:
    print("Mutation is legal!")
else:
    print(f"Errors: {result.errors}")
```

## Search with Budget

```python
from factor_optimizer.contracts import SearchBudget, BudgetTracker

# Define budget
budget = SearchBudget(
    max_trials=100,
    max_evaluations=50,
    max_cost_units=1000.0,
    max_llm_calls=20,
)

# Track usage
tracker = BudgetTracker(budget=budget)

# Simulate trials
for i in range(10):
    if tracker.can_propose_trial():
        tracker.record_trial()
        print(f"Trial {tracker.trials_used}/100")
    
    if tracker.can_evaluate():
        tracker.record_evaluation(cost=10.0)

# Check remaining
print(f"Remaining trials: {tracker.remaining_trials()}")
print(f"Budget exhausted: {tracker.is_exhausted()}")
```

## Trial Management

```python
from factor_optimizer.contracts import Trial, TrialStatus

# Create trial
trial = Trial(
    trial_id="trial_001",
    mutation_id="mut_001",
    status=TrialStatus.PROPOSED,
    parent_factor_ids=["F_parent"],
)

print(f"Initial status: {trial.status}")

# Update status
trial.update_status(TrialStatus.LEGAL, legality_check={"passed": True})
print(f"After validation: {trial.status}")

trial.update_status(
    TrialStatus.EVALUATED,
    evaluation_ref="eval_001",
)
print(f"After evaluation: {trial.status}")

# Check if successful
if trial.is_successful():
    print("Trial succeeded!")
```

## Seen Cache (Deduplication)

```python
from factor_optimizer.seen import SeenCache

# Create cache
seen = SeenCache()

# Mark factor as seen
record = seen.mark_seen(
    canonical_hash="abc123",
    trial_id="trial_001",
    factor_id="F_001",
    metadata={"campaign": "momentum_search"},
)

# Check before expensive evaluation
if seen.is_seen("abc123"):
    existing = seen.get_record("abc123")
    print(f"Already seen in trial {existing.trial_id}")
else:
    # Proceed with evaluation
    pass
```

## Admission Policy

```python
from factor_optimizer.policy import AdmissionPolicy, AdmissionCriteria

# Define criteria
criteria = AdmissionCriteria(
    max_complexity_cost=100.0,
    min_expected_value=0.03,
    max_lookback_periods=252,
    allowed_domains=["equity"],
    require_parent_evidence=True,
)

# Create policy
policy = AdmissionPolicy(criteria=criteria)

# Make decision
decision = policy.decide(
    mutation_id="mut_001",
    trial_id="trial_001",
    complexity_cost=75.0,
    expected_value=0.045,
    lookback_periods=60,
    domains=["equity"],
    has_parent_evidence=True,
)

if decision.is_approved():
    print("Mutation admitted for evaluation")
else:
    print(f"Rejected: {decision.primary_rejection_reason()}")
```

## Multi-Fidelity Evaluation

```python
from factor_optimizer.search import MultiFidelityScheduler, FidelityTier

# Create scheduler
scheduler = MultiFidelityScheduler()

# Start at low fidelity
current_tier = FidelityTier.L0

spec = scheduler.get_spec(current_tier)
print(f"Tier {current_tier.name}: {spec.sample_fraction*100}% sample")
print(f"Estimated cost: {scheduler.estimate_cost(current_tier)}")

# Promote if promising
score = 0.042
rank = 5
total = 100

if scheduler.should_promote(current_tier, score, rank, total):
    next_tier = scheduler.next_tier(current_tier)
    print(f"Promoting to {next_tier.name}")
```

## Lineage Tracking

```python
from factor_optimizer.search import LineageTree, LineageNode

# Create tree
tree = LineageTree()

# Add seed factor
seed = LineageNode(
    trial_id="trial_000",
    parent_ids=[],
    mutation_type="seed",
    generation=0,
    score=0.035,
)
tree.add_node(seed)

# Add child mutation
child = LineageNode(
    trial_id="trial_001",
    parent_ids=["trial_000"],
    mutation_type="window_adjust",
    generation=1,
    score=0.042,
)
tree.add_node(child)

# Query lineage
path = tree.path_to_seed("trial_001")
print(f"Path to seed: {path}")

best = tree.best_in_lineage("trial_001")
print(f"Best in lineage: {best.score}")
```

## Pareto Frontier

```python
from factor_optimizer.search import ParetoFrontier, ParetoPoint

# Create frontier (IC vs turnover)
frontier = ParetoFrontier(objective_names=["ic", "turnover"])

# Add points
frontier.add_point(ParetoPoint(
    trial_id="trial_001",
    objectives=(0.042, 0.25),  # (IC, turnover)
))

frontier.add_point(ParetoPoint(
    trial_id="trial_002",
    objectives=(0.038, 0.15),  # Lower IC, lower turnover
))

# Non-dominated point
frontier.add_point(ParetoPoint(
    trial_id="trial_003",
    objectives=(0.045, 0.20),  # Best IC, medium turnover
))

print(f"Frontier size: {frontier.size}")

# Get extremes
extremes = frontier.extremes()
best_ic_trial = extremes[0]  # Best on first objective
print(f"Best IC: {best_ic_trial.trial_id}")
```

## Plateau Detection

```python
from factor_optimizer.search import PlateauDetector, PlateauConfig

# Configure detector
config = PlateauConfig(
    window_size=20,
    min_relative_improvement=0.001,
)

detector = PlateauDetector(config)

# Track scores
scores = [0.03, 0.032, 0.035, 0.037, 0.038, 0.038, 0.038, 0.038]

for score in scores:
    detector.add_score(score)
    if detector.is_plateau():
        print(f"Plateau detected after {len(scores)} trials")
        break
```

## Complete Search Session

```python
from factor_optimizer.contracts import SearchBudget, BudgetTracker, Trial
from factor_optimizer.seen import SeenCache
from factor_optimizer.policy import AdmissionPolicy
from factor_optimizer.search import LineageTree, LineageNode

# 1. Setup
budget = SearchBudget(max_trials=100, max_evaluations=50)
tracker = BudgetTracker(budget)
seen = SeenCache()
policy = AdmissionPolicy()
tree = LineageTree()

# 2. Search loop
trial_count = 0

while not tracker.is_exhausted():
    # Generate mutation (simplified)
    canonical_hash = f"hash_{trial_count}"
    
    # Check if seen
    if seen.is_seen(canonical_hash):
        continue
    
    # Create trial
    trial = Trial(
        trial_id=f"trial_{trial_count}",
        mutation_id=f"mut_{trial_count}",
        status=TrialStatus.PROPOSED,
        parent_factor_ids=["F_seed"],
    )
    
    tracker.record_trial()
    
    # Validate
    trial.update_status(TrialStatus.LEGAL)
    
    # Check admission
    decision = policy.decide(
        mutation_id=trial.mutation_id,
        trial_id=trial.trial_id,
        complexity_cost=50.0,
        expected_value=0.04,
    )
    
    if not decision.is_approved():
        continue
    
    # Evaluate (simulated)
    if tracker.can_evaluate():
        tracker.record_evaluation(cost=10.0)
        trial.update_status(TrialStatus.EVALUATED, evaluation_ref="eval_001")
        
        # Mark as seen
        seen.mark_seen(canonical_hash, trial.trial_id)
        
        # Add to lineage
        node = LineageNode(
            trial_id=trial.trial_id,
            parent_ids=trial.parent_factor_ids,
            mutation_type="window_adjust",
            generation=1,
            score=0.04,
        )
        tree.add_node(node)
    
    trial_count += 1

print(f"Search complete: {trial_count} trials, {tracker.evaluations_used} evaluations")
print(f"Lineage depth: {tree.depth('trial_0')}")
```

## Repair Proposals

```python
from factor_optimizer.policy import RepairMapper, DiagnosisRecord, DiagnosisKind

# Create diagnosis
diagnosis = DiagnosisRecord(
    trial_id="trial_001",
    diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
    severity=0.8,
    evidence={"operator_count": 50, "max_depth": 10},
)

# Get repair proposals
mapper = RepairMapper()
proposals = mapper.propose_repairs(diagnosis, max_proposals=3)

for proposal in proposals:
    print(f"Strategy: {proposal.strategy}")
    print(f"Mutation: {proposal.mutation_type}")
    print(f"Confidence: {proposal.confidence}")
```

## Tips

1. **Budget management:** Always check `can_propose_trial()` and `can_evaluate()` before operations
2. **Deduplication:** Use SeenCache to avoid re-evaluating identical factors
3. **Multi-fidelity:** Start with low fidelity (L0) to screen candidates cheaply
4. **Lineage tracking:** Track ancestry to understand which mutations are productive
5. **Plateau detection:** Stop search early if no improvement

## Next Steps

- **Full API:** See [API_REFERENCE.md](API_REFERENCE.md)
- **Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Testing:** See [TESTING.md](TESTING.md)
- **Platform Guide:** See [PLATFORM_GUIDE.md](../../PLATFORM_GUIDE.md)
