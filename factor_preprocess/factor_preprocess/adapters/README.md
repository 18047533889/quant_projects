# Adapters

Optional integration layer for factor_preprocess with other platform packages.

## Philosophy

The adapters layer provides **optional, protocol-based integration** with other packages:

- **factor_preprocess core** has zero dependencies on `factor_assets` or `dataaccess`
- Adapters fail gracefully if dependencies are missing
- Protocol-based design allows mock implementations for testing
- Callers can provide custom providers without installing the full packages

## Available Adapters

### 1. Factor Assets Adapter

Converts `FactorSet` objects from `factor_assets` package into preprocessing input.

**File:** `factor_assets.py`

**Protocol:** `FactorSetProvider`

**Key Methods:**
- `get_factor_values()` - Retrieve single factor
- `get_factor_batch()` - Retrieve multiple factors efficiently
- `validate_factor_set()` - Validate FactorSet before loading

**Usage:**

```python
from factor_preprocess.adapters import factor_assets

# Check if factor_assets is available
if factor_assets.check_factor_assets_available():
    # Create adapter (auto-detects provider)
    adapter = factor_assets.create_adapter()
    
    # Load a FactorSet
    data = adapter.load_factor_set(
        factor_set=my_factor_set,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Returns:
    # {
    #     'values': np.ndarray (n_times, n_assets, n_factors),
    #     'dates': np.ndarray,
    #     'assets': np.ndarray,
    #     'factor_ids': list,
    #     'metadata': dict,
    #     'set_metadata': dict
    # }
```

**With Custom Provider:**

```python
from factor_preprocess.adapters.factor_assets import (
    FactorAssetsAdapter,
    FactorSetProvider
)

class MyCustomProvider:
    """Custom provider implementation."""
    
    def get_factor_values(self, factor_id, start_date=None, end_date=None, universe=None):
        # Your implementation
        return {...}
    
    def get_factor_batch(self, factor_ids, start_date=None, end_date=None, universe=None):
        # Your implementation
        return {...}
    
    def validate_factor_set(self, factor_set):
        # Your validation
        return True

# Use custom provider
provider = MyCustomProvider()
adapter = FactorAssetsAdapter(provider)
```

### 2. Data Access Adapter

Fetches exposure context (industry, sector, size, beta) from `dataaccess` package for use in neutralization.

**File:** `data_access.py`

**Protocol:** `ExposureProvider`

**Key Methods:**
- `get_industry_exposure()` - Industry classification
- `get_size_exposure()` - Market cap, log market cap
- `get_sector_exposure()` - Sector classification
- `get_beta_exposure()` - Market beta
- `get_custom_exposure()` - Custom exposures

**Usage:**

```python
from factor_preprocess.adapters import data_access

# Check if dataaccess is available
if data_access.check_data_access_available():
    adapter = data_access.create_adapter()
    
    # Fetch single exposure
    industry_data = adapter.fetch_industry_exposure(
        market="ashare",
        start_date="2024-01-01",
        end_date="2024-12-31",
        industry_classification="SW_L1"
    )
    
    # Fetch multiple exposures efficiently
    exposures = adapter.fetch_multi_exposure(
        market="ashare",
        exposure_types=["industry", "size", "beta"]
    )
    
    # Returns dict mapping exposure_type -> data:
    # {
    #     'industry': {
    #         'values': np.ndarray,
    #         'dates': np.ndarray,
    #         'assets': np.ndarray,
    #         'classification': str,
    #         'categories': list,
    #         'metadata': dict
    #     },
    #     'size': {...},
    #     'beta': {...}
    # }
```

## Error Handling

Adapters raise `OptionalDependencyMissing` when a required package is not available:

```python
from factor_preprocess.adapters.factor_assets import (
    OptionalDependencyMissing,
    create_adapter
)

try:
    adapter = create_adapter()
    data = adapter.load_factor_set(my_set)
except OptionalDependencyMissing as e:
    print(f"Feature not available: {e.feature_name}")
    print(f"Install with: pip install {e.package_name}")
    # Fallback to alternative approach
```

## Testing

Both adapters include comprehensive test suites with mock providers:

```bash
# Run adapter tests
pytest tests/adapters/ -v

# Test specific adapter
pytest tests/adapters/test_factor_assets.py -v
pytest tests/adapters/test_data_access.py -v
```

**Test Coverage:**
- Protocol compliance validation
- Mock provider implementations
- Adapter functionality
- Edge cases (empty inputs, missing data)
- Error handling
- Multi-market support

## Design Principles

### 1. Protocol-Based Boundaries

Use Python `Protocol` (PEP 544) for structural typing:

```python
from typing import Protocol

class FactorSetProvider(Protocol):
    """Protocol defines required interface."""
    
    def get_factor_values(self, factor_id: str, ...) -> Dict[str, Any]:
        ...
```

Benefits:
- No inheritance required
- Duck typing with type checking
- Easy to mock for testing
- Clear interface contracts

### 2. Fail-Closed by Default

Missing dependencies raise explicit exceptions rather than silently degrading:

```python
if not check_factor_assets_available():
    raise OptionalDependencyMissing(
        package_name="factor_assets",
        feature_name="FactorSet integration"
    )
```

### 3. Single Responsibility

Each adapter does **one thing**:
- **factor_assets**: Load factor data
- **data_access**: Load exposure context

They do not:
- Perform transformations (that's core's job)
- Cache results (caller's responsibility)
- Implement business logic

### 4. Batch-First API

All data retrieval methods support batch operations:

```python
# Batch is primary interface
data = adapter.load_factor_set(factor_set)  # Multiple factors at once

# Single factor is convenience wrapper
data = adapter.load_single_factor(factor_id)
```

## Implementation Notes

### Factor Assets Adapter

**Input:** `FactorSet` (from factor_assets)
- Contains factor IDs, metadata, universe, frequency
- Does NOT contain factor values

**Output:** Dict with preprocessor-ready arrays
- `values`: 3D array (time, assets, factors)
- `dates`, `assets`, `factor_ids`: aligned axes
- `metadata`: per-factor metadata
- `set_metadata`: FactorSet-level metadata

**Validation:**
- All factors in set must exist in provider
- Set must be well-formed (non-empty, valid IDs)
- Provider decides date range/universe filtering behavior

### Data Access Adapter

**Input:** Market identifier, date range, exposure types
- Supports multiple markets (ashare, us, etc.)
- Multiple classification schemes (SW_L1, GICS, etc.)
- Flexible metric selection (market_cap, log_market_cap)

**Output:** Dict with exposure arrays
- `values`: 2D array (time, assets) for each exposure
- `dates`, `assets`: aligned with factor data
- `categories`: unique exposure values (for categorical)
- `metadata`: source and provenance

**Multi-Exposure Fetch:**
- Single call fetches multiple exposures
- Ensures temporal alignment across exposures
- Reduces round-trips to underlying storage

## Extension Points

### Adding a New Adapter

1. Define a Protocol in a new file (e.g., `model_registry.py`)
2. Create Adapter class that wraps the Protocol
3. Implement `check_*_available()` and `create_adapter()`
4. Add comprehensive tests with mock provider
5. Update `adapters/__init__.py`

Example structure:

```python
from typing import Protocol, Dict, Any

class ModelRegistryProvider(Protocol):
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        ...

class ModelRegistryAdapter:
    def __init__(self, provider: ModelRegistryProvider):
        self._provider = provider
    
    def fetch_config(self, model_id: str) -> Dict[str, Any]:
        return self._provider.get_model_config(model_id)

def check_model_registry_available() -> bool:
    try:
        import model_registry
        return True
    except ImportError:
        return False

def create_adapter(provider=None) -> ModelRegistryAdapter:
    if provider:
        return ModelRegistryAdapter(provider)
    if not check_model_registry_available():
        raise OptionalDependencyMissing("model_registry", "Model config")
    # Create default provider...
```

### Custom Providers

Implement the Protocol interface to provide custom data sources:

```python
class FileBasedFactorProvider:
    """Load factors from parquet files."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def get_factor_values(self, factor_id, ...):
        path = f"{self.data_dir}/{factor_id}.parquet"
        df = pd.read_parquet(path)
        # Convert to expected format
        return {...}
    
    def get_factor_batch(self, factor_ids, ...):
        # Batch load from files
        ...
    
    def validate_factor_set(self, factor_set):
        # Check all files exist
        ...

# Use custom provider
provider = FileBasedFactorProvider("/data/factors")
adapter = FactorAssetsAdapter(provider)
```

## See Also

- Core preprocessing: `factor_preprocess.transforms`
- Neutralization: `factor_preprocess.neutralization`
- Feature contracts: `factor_preprocess.contracts.feature_bundle`
