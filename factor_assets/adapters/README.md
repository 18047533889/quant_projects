# Factor Assets Adapters

Optional integration adapters for connecting Factor Assets with FactorEngine, QuantEvaluator, and DataAccess.

## Overview

The `adapters/` layer provides **optional** integration with external packages. FA core does not depend on these packages — they enable enhanced functionality when available.

**Key principle:** Protocol-based design with lazy imports. Missing dependencies do not break FA core.

## Available Adapters

### 1. `quant_evaluator.py` - Evidence Integration

**Purpose:** Convert QE evaluation results to FA evidence references.

**Components:**
- `EvidenceProvider` (Protocol): Interface for evidence integration
- `QEEvidenceProvider`: Implementation for QuantEvaluator

**Usage:**
```python
try:
    from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
    
    provider = QEEvidenceProvider()
    
    # Convert QE bundle to FA evidence ref
    bundle_ref = provider.create_evidence_bundle_ref(
        bundle=qe_evaluation_bundle,
        run_id="eval-run-001",
        qe_version="0.1.0",
    )
    
    # Extract individual metric refs
    evidence_refs = provider.create_evidence_refs(
        bundle=qe_evaluation_bundle,
        run_id="eval-run-001",
    )
    
except OptionalDependencyMissing:
    # QE not available — handle gracefully
    logger.warning("QE integration not available")
```

**Key features:**
- Converts `EvaluationBundle` → `EvidenceBundleRef`
- Extracts individual `EvidenceRef` for each metric
- Does NOT duplicate metric values (only references)
- Bounded summary statistics only (no distributions)
- Graceful failure when QE unavailable

### 2. `factor_engine.py` - Identity Integration

**Purpose:** Obtain factor canonical identity from FE expression system.

**Components:**
- `FEIdentityProvider`: Standard provider (requires FE parser)
- `FEIdentityProviderFromExpr`: Provider accepting parsed Expr nodes

**Usage:**
```python
try:
    from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr
    
    provider = FEIdentityProviderFromExpr(compiler_generation="fe-0.9.7")
    
    # Get full identity from FE Expr node
    identity = provider.get_full_identity(
        expression=expr_node,
        complexity_score=2.5,
    )
    
    # Or get components individually
    canonical_hash = provider.get_canonical_hash(expr_node)
    canonical_repr = provider.get_canonical_repr(expr_node)
    fe_identity_ref = provider.get_identity_ref(expr_node)
    
except OptionalDependencyMissing:
    # FE not available
    logger.warning("FE identity integration not available")
```

**Key features:**
- Delegates to FE's canonical expression system
- Does NOT duplicate FE parser logic
- Deterministic hash computation
- Supports Expr nodes directly (no string parsing yet)
- Optional compiler generation tracking

### 3. `data_access.py` - Factor Value Integration (Future)

**Purpose:** Optional integration for reading factor values and catalog metadata.

**Components:**
- `FactorValueReader` (Protocol): Interface for factor value reads
- `CatalogReader` (Protocol): Interface for catalog reads
- `DAFactorValueReader`: Implementation (pending DA availability)
- `DACatalogReader`: Implementation (pending DA availability)

**Status:** Protocols defined, implementations pending DA package availability.

**Usage (future):**
```python
try:
    from factor_assets.adapters.data_access import DAFactorValueReader
    
    reader = DAFactorValueReader()
    
    # Check availability
    available = reader.check_factor_availability("F1234567890abcdef")
    
    # Read values
    values = reader.read_factor_values(
        factor_id="F1234567890abcdef",
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        universe="US_STOCKS",
    )
    
except OptionalDependencyMissing:
    # DA not available
    pass
```

## Installation

### Core Only (No Adapters)
```bash
pip install factor_assets
```

### With Adapter Dependencies
```bash
pip install factor_assets[adapters]
```

## Design Principles

### 1. Protocol-Based Design

All adapters define **protocols** that FA core can depend on without importing the actual implementations:

```python
class EvidenceProvider(Protocol):
    """Protocol for evidence integration."""
    def create_evidence_bundle_ref(self, bundle: object, run_id: str) -> EvidenceBundleRef:
        ...
```

Protocols enable:
- Type safety without hard dependencies
- Mock implementations for testing
- Multiple adapter implementations

### 2. Lazy Imports

Adapters use try/except imports with graceful failure:

```python
try:
    from quant_evaluator import EvaluationBundle
    QE_AVAILABLE = True
except ImportError:
    QE_AVAILABLE = False
    EvaluationBundle = None
```

### 3. Explicit Error Handling

When optional dependencies are missing, raise informative errors:

```python
class OptionalDependencyMissing(ImportError):
    def __init__(self, package_name: str, adapter_name: str):
        super().__init__(
            f"Adapter '{adapter_name}' requires optional package '{package_name}'. "
            f"Install with: pip install factor_assets[adapters]"
        )
```

### 4. No Data Duplication

Adapters create **references**, not copies:
- Evidence refs point to QE results (no metric duplication)
- Identity refs point to FE canonical hashes (no parser duplication)
- Value refs point to DA storage (no materialization)

## Testing

All adapters have comprehensive tests using **mock implementations**:

```bash
# Run adapter tests
pytest tests/test_adapter_quant_evaluator.py
pytest tests/test_adapter_factor_engine.py
pytest tests/test_adapter_data_access.py
```

Tests verify:
- Protocol contracts
- Error handling for missing dependencies
- Reference conversion correctness
- Deterministic behavior
- Graceful degradation

## Integration Patterns

### Pattern 1: Optional Enhancement

```python
# Try to use adapter, fall back gracefully
try:
    from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
    evidence_provider = QEEvidenceProvider()
    use_qe_integration = True
except OptionalDependencyMissing:
    evidence_provider = None
    use_qe_integration = False

# Later in code
if use_qe_integration:
    evidence_ref = evidence_provider.create_evidence_bundle_ref(bundle, run_id)
else:
    # Manual evidence ref creation
    evidence_ref = create_manual_evidence_ref(...)
```

### Pattern 2: Dependency Injection

```python
from factor_assets.identity.canonical import FactorIdentityProvider

class AssetFactory:
    def __init__(self, identity_provider: FactorIdentityProvider):
        self.identity_provider = identity_provider
    
    def create_asset(self, expression):
        identity = self.identity_provider.get_full_identity(expression)
        # ... create asset

# With FE adapter
try:
    from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr
    provider = FEIdentityProviderFromExpr()
except OptionalDependencyMissing:
    provider = MockIdentityProvider()  # Fallback

factory = AssetFactory(provider)
```

### Pattern 3: Protocol Testing

```python
# Test against protocol, not concrete implementation
from factor_assets.adapters.data_access import FactorValueReader

class TestMyFeature:
    def test_with_mock_reader(self):
        # Create mock that satisfies protocol
        class MockReader:
            def read_factor_values(self, factor_id, start_date, end_date, universe=None):
                return {"mock": "data"}
            
            def check_factor_availability(self, factor_id, as_of_date=None):
                return True
        
        reader = MockReader()
        # Test feature using mock reader
        result = my_feature(reader, ...)
```

## Future Extensions

### Potential Adapters

1. **Backtesting Integration**
   - Adapter for backtesting frameworks
   - Signal generation from asset registry
   - Performance attribution

2. **Monitoring Integration**
   - Adapter for monitoring/alerting systems
   - Asset health metrics
   - Evidence quality tracking

3. **Workflow Integration**
   - Adapter for workflow orchestration
   - Asset lifecycle automation
   - Approval workflows

### Implementation Guidelines

When adding new adapters:

1. Define protocol in appropriate module
2. Implement adapter with lazy imports
3. Raise `OptionalDependencyMissing` when unavailable
4. Write tests with mock implementations
5. Document usage patterns
6. Update `pyproject.toml` optional dependencies

## Architecture Context

The adapters layer sits at the boundary between FA and external systems:

```
┌─────────────────────────────────────────┐
│         Factor Assets Core              │
│  (contracts, registry, identity, etc.)  │
└───────────────┬─────────────────────────┘
                │ Protocols
┌───────────────┴─────────────────────────┐
│           Adapters Layer                │
│  (quant_evaluator, factor_engine, etc.) │
└───────────────┬─────────────────────────┘
                │ Optional Integration
┌───────────────┴─────────────────────────┐
│      External Systems                   │
│  (QE, FE, DA, Backtesting, etc.)       │
└─────────────────────────────────────────┘
```

**Key invariants:**
- FA core never imports adapters directly
- Adapters never import each other
- External systems know nothing about FA
- Protocols define clean boundaries

## See Also

- `contracts/` - Core FA contracts
- `identity/` - Identity management
- `registry/` - Asset repository
- `evidence_ref.py` - Evidence reference contracts
