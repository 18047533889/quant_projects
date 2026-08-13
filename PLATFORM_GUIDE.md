# Quant Platform Guide

**Version:** 1.0  
**Last Updated:** 2026-08-14

## Overview

The Quant Platform consists of four coordinated packages for quantitative factor research:

1. **QuantEvaluator (QE)**: Evidence generation - compute metrics over factors
2. **FactorAssets (FA)**: Governance - factor registry and lifecycle
3. **FactorOptimizer (FO)**: Search - mutation proposals and orchestration
4. **FactorPreprocess (FP)**: Preparation - transform factors for models

Each package has clear boundaries and communicates via protocols, not hard dependencies.

## Package Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Research Workflow                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │    FactorOptimizer (FO)           │
         │    - Mutation proposals            │
         │    - Search orchestration          │
         │    - Budget tracking               │
         └────────┬────────────────────┬──────┘
                  │                    │
                  ▼                    ▼
    ┌─────────────────────┐   ┌─────────────────────┐
    │  FactorEngine (FE)  │   │ QuantEvaluator (QE) │
    │  - Legality check   │   │ - IC, coverage      │
    │  - Canonical hash   │   │ - Quantiles         │
    │  - Execution        │   │ - Robustness        │
    └──────────┬──────────┘   └──────────┬──────────┘
               │                          │
               │        ┌─────────────────▼──────────┐
               │        │  FactorAssets (FA)         │
               │        │  - Registry                 │
               └────────▶  - Lifecycle states         │
                        │  - Lineage tracking         │
                        │  - Selection gates          │
                        └─────────────┬───────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │  FactorPreprocess (FP)      │
                        │  - Cross-sectional          │
                        │  - Rolling (causal)         │
                        │  - Neutralization           │
                        │  - Model-ready features     │
                        └─────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │    Model Training           │
                        │    (External)               │
                        └─────────────────────────────┘
```

## Data Flow

### 1. Factor Discovery & Search

**FO generates mutations:**
```python
from factor_optimizer import CandidateMutation, SearchBudget

# FO: Propose mutation
mutation = CandidateMutation(
    mutation_id="mut_001",
    parent_factor_ids=["F_parent"],
    mutation_type="window_adjust",
    parameters={"window": 20, "adjustment": +5},
)
```

**FE validates legality:**
```python
# FE: Check if legal
from factor_engine import validate_expression

is_legal = validate_expression(mutated_expression)
canonical_hash = compute_canonical_hash(mutated_expression)
```

**QE evaluates performance:**
```python
from quant_evaluator import FactorBatch, LabelBundle

# QE: Compute metrics
batch = FactorBatch(factor_ids=("F_new",), ...)
labels = LabelBundle(target_id="forward_return_1d", ...)

result = evaluator.evaluate(batch, labels, metrics)
```

### 2. Factor Registration & Lifecycle

**FA registers with evidence:**
```python
from factor_assets import AssetRepository, EvidenceRef

# FA: Register new factor
repo = AssetRepository()

evidence = EvidenceRef(
    evaluation_id=result.request_id,
    factor_id="F_new",
    metric_snapshot={"mean_ic": 0.045, "coverage": 0.95},
    timestamp="2026-08-14T10:00:00Z",
)

asset = repo.register(metadata=metadata, lineage=lineage, tags={})

# FA: Transition through lifecycle
asset = repo.transition(
    factor_id="F_new",
    to_state=LifecycleState.EVALUATED,
    evidence_refs=[evidence],
)
```

**FA applies selection gates:**
```python
from factor_assets.selection import ThresholdGate

# FA: Check admission gates
gate = ThresholdGate("min_ic_gate", "1.0", threshold=0.03)
evaluation = gate.evaluate("F_new", "eval_001", "mean_ic", 0.045)

if evaluation.passed:
    asset = repo.transition("F_new", LifecycleState.APPROVED)
```

### 3. Feature Preparation for Models

**FP transforms selected factors:**
```python
from factor_preprocess.transforms import cs_rank, rolling_zscore
from factor_preprocess.representation import build_linear_ready

# FP: Get approved factors
approved = repo.list_by_state(LifecycleState.APPROVED)

# FP: Apply transformations
ranked = cs_rank(factor_values, axis=-1, pct=True)
rolling = rolling_zscore(factor_df, window=60, ...)

# FP: Build model-ready features
config = LinearReadyConfig(add_intercept=True, standardize=True)
result = build_linear_ready(processed_values, config)

# Result: X matrix ready for sklearn, statsmodels, etc.
```

## Integration Patterns

### Pattern 1: End-to-End Factor Development

```python
# 1. Search for mutations (FO)
budget = SearchBudget(max_trials=100, max_evaluations=50)
tracker = BudgetTracker(budget)

while not tracker.is_exhausted():
    # FO: Generate mutation
    mutation = generate_mutation()
    
    # FE: Validate (via adapter)
    if not fe_adapter.validate_mutation(mutation):
        continue
    
    canonical_hash = fe_adapter.get_canonical_hash(mutation)
    
    # Check if already seen
    if seen.is_seen(canonical_hash):
        continue
    
    # FE: Execute factor
    factor_values = fe_adapter.execute(mutation)
    
    # QE: Evaluate
    batch = FactorBatch(...)
    result = qe_evaluator.evaluate(batch, labels, metrics)
    
    tracker.record_evaluation()
    
    # FA: Register if promising
    if result.get_metric("mean_ic") > 0.03:
        factor_id = create_factor_id(canonical_hash)
        evidence = EvidenceRef(...)
        asset = repo.register(metadata, lineage, tags)
        repo.transition(factor_id, LifecycleState.EVALUATED, [evidence])

# 2. Select best factors (FA)
approved = repo.list_by_state(LifecycleState.APPROVED)

# 3. Prepare for model (FP)
feature_bundle = preprocess_factors(approved)

# 4. Train model (External)
model.fit(feature_bundle.X, y)
```

### Pattern 2: Batch Re-Evaluation

```python
# FA: Get all evaluated factors
evaluated = repo.list_by_state(LifecycleState.EVALUATED)

# QE: Re-evaluate on new data
for asset in evaluated:
    factor_values = fetch_factor_values(asset.factor_id)
    batch = FactorBatch(...)
    result = evaluator.evaluate(batch, new_labels, metrics)
    
    # FA: Update evidence
    new_evidence = EvidenceRef(...)
    repo.transition(
        factor_id=asset.factor_id,
        to_state=LifecycleState.EVALUATED,  # Self-transition
        evidence_refs=[new_evidence],
    )
```

### Pattern 3: Production Deployment

```python
# 1. FA: Select production-ready factors
production = repo.list_by_state(LifecycleState.PRODUCTION_READY)

# 2. FP: Fit transforms on training data
policy = get_policy("production_full")
fitted_state = fit_policy(policy, train_data)

# 3. FP: Transform live data
live_features = apply_policy(policy, fitted_state, live_data)

# 4. Model: Generate signals
signals = model.predict(live_features.X)
```

## Best Practices

### 1. Clear Boundaries

**DO:**
- QE computes metrics, FA stores evidence refs
- FO proposes mutations, FE validates them
- FP transforms features, models train on them

**DON'T:**
- QE storing factor values (that's DataAccess)
- FA computing metrics (that's QE)
- FO executing factors (that's FE)

### 2. Protocol-Based Integration

```python
# Define protocol
class FactorIdentityProvider(Protocol):
    def get_canonical_hash(self, expression) -> str: ...

# Implement in FE
class FEIdentityAdapter:
    def get_canonical_hash(self, expression) -> str:
        return fe.compute_hash(expression)

# Use in FA
identity_provider = FEIdentityAdapter()
canonical_hash = identity_provider.get_canonical_hash(expr)
```

### 3. Deduplication Early

```python
# Check before expensive evaluation
canonical_hash = fe_adapter.get_canonical_hash(expression)

if seen.is_seen(canonical_hash):
    print("Already evaluated, skipping")
    return

# Proceed with evaluation
...
```

### 4. Evidence References, Not Raw Values

```python
# GOOD: Store reference
evidence = EvidenceRef(
    evaluation_id="eval_001",
    metric_snapshot={"mean_ic": 0.045},  # Bounded summary
)

# BAD: Store raw values
# evidence = {"ic_series": np.array([...])}  # 252x1 array
```

### 5. Explicit Causality

```python
# GOOD: Explicit lag
rolling_mean(df, window=20, ...)  # Uses shift(1) internally

# BAD: Implicit current
df.rolling(20).mean()  # Includes current observation!
```

## Common Workflows

### Workflow 1: New Factor Search

1. **FO**: Generate mutation candidates
2. **FE**: Validate legality, compute canonical hash
3. **FO**: Check SeenCache for duplicates
4. **FE**: Execute factor if novel
5. **QE**: Evaluate performance
6. **FA**: Register with evidence if promising
7. **FA**: Apply gates, transition to APPROVED

### Workflow 2: Re-Evaluation

1. **FA**: List factors by state
2. **QE**: Re-evaluate on new data window
3. **FA**: Update evidence refs
4. **FA**: Re-apply gates, update lifecycle state

### Workflow 3: Production Pipeline

1. **FA**: Select PRODUCTION_READY factors
2. **FP**: Fit transforms on training data
3. **FP**: Apply transforms to live data
4. **Model**: Generate predictions

### Workflow 4: Factor Family Management

1. **FA**: Cluster correlated factors
2. **FA**: Detect lineage relationships
3. **FA**: Select family representatives
4. **FP**: Aggregate or combine factors

## Package Dependencies

**External Dependencies:**
- QE: numpy, scipy (optional)
- FA: None (stdlib only)
- FO: pyyaml, numpy
- FP: numpy, pandas, scipy

**Inter-Package (Optional):**
- FO → FE (legality validation)
- FO → QE (evaluation)
- FA → FE (identity)
- FA → QE (evidence retrieval)

**Key:** All inter-package dependencies are via protocols, not imports.

## Versioning & Compatibility

Each package maintains semantic versioning independently:
- **Major:** Breaking API changes
- **Minor:** New features, backwards compatible
- **Patch:** Bug fixes

**Current Versions:**
- QuantEvaluator: 0.0.1a1 (alpha)
- FactorAssets: 0.1.0 (initial)
- FactorOptimizer: 0.1.0 (initial)
- FactorPreprocess: 0.1.0 (initial)

## Getting Started

### Installation

```bash
cd /home/shw/quant_projects

# Install all packages
pip install -e quant_evaluator/
pip install -e factor_assets/
pip install -e factor_optimizer/
pip install -e factor_preprocess/

# With optional dependencies
pip install -e "factor_optimizer[factor_engine,quant_evaluator]"
```

### Quick Example

```python
# 1. Evaluate a factor (QE)
from quant_evaluator import FactorBatch, LabelBundle, Evaluator

evaluator = Evaluator()
result = evaluator.evaluate(batch, labels, metrics)

# 2. Register with evidence (FA)
from factor_assets import AssetRepository, EvidenceRef

repo = AssetRepository()
evidence = EvidenceRef(evaluation_id=result.request_id, ...)
asset = repo.register(metadata, lineage, tags)
repo.transition(asset.factor_id, LifecycleState.EVALUATED, [evidence])

# 3. Prepare for model (FP)
from factor_preprocess.transforms import cs_rank
from factor_preprocess.representation import build_linear_ready

ranked = cs_rank(factor_values, axis=-1)
features = build_linear_ready(ranked, config)

# 4. Train model
model.fit(features.X, y)
```

## Further Reading

**Package Documentation:**
- [QuantEvaluator Docs](quant_evaluator/docs/)
- [FactorAssets Docs](factor_assets/docs/)
- [FactorOptimizer Docs](factor_optimizer/docs/)
- [FactorPreprocess Docs](factor_preprocess/docs/)

**Key Concepts:**
- Batch-first processing (QE)
- Append-only repository (FA)
- Conservative validation (FO)
- Causal transforms (FP)

---

**Last Updated:** 2026-08-14  
**Authors:** Quant Platform Team
