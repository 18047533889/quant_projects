# Adapters Layer

The adapters layer provides **optional** integration with FactorEngine (FE) and QuantEvaluator (QE) through clean Protocol-based interfaces. FO core does NOT directly depend on FE or QE.

## Architecture

```
┌─────────────────────────────────────────┐
│         FactorOptimizer Core            │
│  (search, mutation, seen, complexity)   │
└──────────────┬──────────────────────────┘
               │ depends on
               ▼
┌─────────────────────────────────────────┐
│       Adapter Protocols (this pkg)      │
│   FactorEngineAdapter | QEAdapter       │
└──────────────┬──────────────────────────┘
               │ optional implementations
               ▼
┌──────────────────────┬──────────────────┐
│   FactorEngine (FE)  │  QuantEval (QE)  │
│  (if installed)      │  (if installed)  │
└──────────────────────┴──────────────────┘
```

## Protocols

### FactorEngineAdapter

Protocol for FE integration:

```python
@runtime_checkable
class FactorEngineAdapter(Protocol):
    def compute_canonical_hash(self, factor_definition: Any) -> str:
        """Compute FE canonical identity hash for deduplication."""
        ...

    def validate_mutation(self, mutation: Any, spec: Any) -> Dict[str, Any]:
        """Validate mutation legality (operator existence, parameter bounds)."""
        ...

    def estimate_complexity(self, factor_definition: Any) -> Dict[str, Any]:
        """Estimate complexity (operator count, depth, lookback)."""
        ...

    def get_operator_metadata(self, operator_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Get operator metadata for mutation grammar."""
        ...
```

### QuantEvaluatorAdapter

Protocol for QE integration:

```python
@runtime_checkable
class QuantEvaluatorAdapter(Protocol):
    def evaluate(
        self,
        factor_batch: Any,
        labels: Any,
        metrics: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit evaluation request to QE."""
        ...

    def get_evidence(self, evaluation_id: str) -> Dict[str, Any]:
        """Retrieve full evidence bundle by evaluation ID."""
        ...

    def list_metrics(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available metrics from QE catalog."""
        ...
```

## Usage

### Creating Adapters

**FactorEngine adapter (real):**

```python
from factor_optimizer.adapters import create_fe_adapter, FEOptionalDependencyMissing

try:
    fe_adapter = create_fe_adapter()
    # Use adapter
except FEOptionalDependencyMissing:
    print("FactorEngine not installed")
    fe_adapter = None
```

**QuantEvaluator adapter (mock for now, real when QE exists):**

```python
from factor_optimizer.adapters import (
    create_qe_adapter,
    create_mock_qe_adapter,
    QEOptionalDependencyMissing,
)

try:
    qe_adapter = create_qe_adapter()
except QEOptionalDependencyMissing:
    # QE doesn't exist yet, use mock
    qe_adapter = create_mock_qe_adapter()
```

### Using with FO Core

**SeenCache with FE adapter:**

```python
from factor_optimizer.seen import SeenCache
from factor_optimizer.adapters import create_fe_adapter

fe_adapter = create_fe_adapter()
cache = SeenCache(fe_adapter=fe_adapter)

# Check and mark factors
was_seen, record = cache.check_and_mark(factor, trial_id="trial_001")
if was_seen:
    print(f"Factor already seen in {record.trial_id}")
```

**ComplexityEstimator with FE adapter:**

```python
from factor_optimizer.complexity import ComplexityEstimator
from factor_optimizer.adapters import create_fe_adapter

fe_adapter = create_fe_adapter()
estimator = ComplexityEstimator(fe_adapter=fe_adapter)

# Estimate complexity
profile = estimator.estimate(factor)
print(f"Operators: {profile.operator_count}")
print(f"Cost: {profile.estimated_cost}")
```

**Search loop with QE adapter:**

```python
from factor_optimizer.adapters import create_mock_qe_adapter

qe_adapter = create_mock_qe_adapter()

# Evaluate candidates
for candidate in candidates:
    result = qe_adapter.evaluate(
        factor_batch=candidate,
        labels=labels,
        metrics=["rank_ic", "turnover"],
    )
    
    if result["metrics"]["rank_ic"] > threshold:
        print(f"Good candidate: {result['evaluation_id']}")
```

## Real FE Integration

The FE adapter integrates with FactorEngine's:

1. **`mining.campaign.candidate_semantic_hash`** - Canonical identity computation
2. **`cleaned_operators.registry.OperatorRegistry`** - Operator catalog and validation
3. **Expression AST analysis** - Complexity estimation from parsed expressions

**Example with real FE:**

```python
import sys
sys.path.insert(0, "factor_engine")

import api
from expr.field import field
from factor_optimizer.adapters import create_fe_adapter

# Create adapter
fe_adapter = create_fe_adapter()

# Create factors
close_field = field("close")
factor1 = api.ts_mean(close_field, 20)
factor2 = api.ts_mean(close_field, 20)  # Semantically same

# Compute hashes
hash1 = fe_adapter.compute_canonical_hash(factor1)
hash2 = fe_adapter.compute_canonical_hash(factor2)

assert hash1 == hash2  # Same semantic identity

# Estimate complexity
profile = fe_adapter.estimate_complexity(factor1)
print(profile)
# {
#   "operator_count": 1,
#   "max_depth": 1,
#   "lookback_periods": 20,
#   "estimated_cost": 2.0,
#   "domains": [],
#   "sources": []
# }

# Get operator metadata
catalog = fe_adapter.get_operator_metadata(["ts_mean"])
print(catalog["ts_mean"])
# {
#   "canonical": "ts_mean",
#   "backends": ["pandas_numpy", "polars"],
#   "param_names": ["x", "window"],
#   "status": "implemented",
#   ...
# }
```

## Mock Adapters for Testing

**Creating mock FE adapter:**

```python
from factor_optimizer.adapters.factor_engine import FactorEngineAdapter

class SimpleMockAdapter:
    def compute_canonical_hash(self, factor_definition):
        import hashlib
        return hashlib.sha256(str(factor_definition).encode()).hexdigest()

    def validate_mutation(self, mutation, spec):
        return {"is_legal": True, "reason": "mock", "metadata": {}}

    def estimate_complexity(self, factor_definition):
        return {
            "operator_count": 1,
            "max_depth": 1,
            "lookback_periods": 0,
            "estimated_cost": 1.0,
            "domains": [],
            "sources": [],
        }

    def get_operator_metadata(self, operator_names=None):
        return {}

mock_adapter = SimpleMockAdapter()
assert isinstance(mock_adapter, FactorEngineAdapter)  # Protocol check
```

**Using mock QE adapter:**

```python
from factor_optimizer.adapters import create_mock_qe_adapter

qe_adapter = create_mock_qe_adapter()

# Mock evaluation
result = qe_adapter.evaluate("factor", "labels", metrics=["rank_ic"])
print(result)
# {
#   "evaluation_id": "mock_eval_abc123",
#   "metrics": {"rank_ic": 0.05},
#   "diagnostics": {"coverage": 0.95, "warnings": []},
#   "evidence_ref": "mock_evidence_abc123"
# }

# Mock metrics catalog
metrics = qe_adapter.list_metrics(tier="core")
for metric in metrics:
    print(f"{metric['metric_id']}: {metric['name']}")
# rank_ic: Rank IC
# ic_mean: IC Mean
# ic_std: IC Std
# turnover: Turnover
```

## Optional Dependencies

Both FE and QE are **optional**. If not installed:

```python
from factor_optimizer.adapters import create_fe_adapter, FEOptionalDependencyMissing

try:
    adapter = create_fe_adapter()
except FEOptionalDependencyMissing as e:
    print(f"FE not available: {e}")
    # Fall back to mock or disable feature
```

**Error messages:**

- **FE missing:** `"factor-engine not installed or not in Python path. Ensure factor_engine is available in the parent directory."`
- **QE missing:** `"quant-evaluator not installed. This is a future package; for now, use mock adapters in tests."`

## Testing

The adapters layer includes comprehensive tests:

1. **Protocol tests** - Verify protocol interfaces
2. **Mock adapter tests** - Test mock implementations
3. **Integration tests** - Test with real FE (when available)
4. **Error handling tests** - Test missing dependency behavior

**Run tests:**

```bash
pytest tests/adapters/ -v

# With integration tests (requires FE)
pytest tests/adapters/ -v -m integration

# Skip integration tests
pytest tests/adapters/ -v -m "not integration"
```

## Design Principles

1. **Protocol-based** - Use `typing.Protocol` with `@runtime_checkable`
2. **Optional dependencies** - FO core never directly imports FE/QE
3. **Fail-closed** - Missing dependencies raise explicit exceptions
4. **Testable** - Mock adapters for testing without dependencies
5. **Clean boundaries** - Adapters translate between FO and FE/QE domains

## Files

```
factor_optimizer/adapters/
├── __init__.py              # Public exports
├── factor_engine.py         # FE adapter protocol + implementation
└── quant_evaluator.py       # QE adapter protocol + implementation

tests/adapters/
├── test_factor_engine.py    # FE adapter tests
├── test_quant_evaluator.py  # QE adapter tests
└── test_integration.py      # Integration tests with FO core
```

## Future Work

- [ ] QE real adapter when QuantEvaluator package exists
- [ ] Performance optimization for batch hash computation
- [ ] Caching layer for operator metadata
- [ ] Async evaluation support in QE adapter
- [ ] More sophisticated complexity estimation (data source analysis)
