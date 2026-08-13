"""
Adapters package - optional integration with other packages.

All adapters are optional and fail gracefully if dependencies are missing.

## Available Adapters

- **factor_assets**: Convert FactorSet to preprocessing input
- **data_access**: Fetch exposure context (industry, sector, size, beta)

## Usage

```python
from factor_preprocess.adapters import factor_assets, data_access

# Check availability
if factor_assets.check_factor_assets_available():
    adapter = factor_assets.create_adapter()
    data = adapter.load_factor_set(my_factor_set)

# Or handle gracefully
try:
    adapter = data_access.create_adapter()
    industry_exp = adapter.fetch_industry_exposure(market="ashare")
except data_access.OptionalDependencyMissing as e:
    print(f"Feature not available: {e}")
```

## Protocol-Based Design

Core preprocessing does not depend on these packages. Adapters use Protocol
definitions so you can provide mock implementations for testing.
"""

# Re-export for convenience
from factor_preprocess.adapters.factor_assets import (
    OptionalDependencyMissing as FactorAssetsError,
)
from factor_preprocess.adapters.data_access import (
    OptionalDependencyMissing as DataAccessError,
)

__all__ = [
    "FactorAssetsError",
    "DataAccessError",
]
