# Unified Backend Capability Contracts

**Implementation:** FE-BE-P0-001 through FE-BE-P0-004  
**Date:** 2026-08-14  
**Status:** Complete

## Overview

The backend capability contract system provides platform-wide single authority for backend classification, capability metadata, and execution routing decisions. This document describes the unified contract system that eliminates duplicate authorities and ensures fail-closed production behavior.

## Single Authority (FE-BE-P0-001)

All backend capability contracts are defined in `backend.contracts`:

```python
from backend.contracts import (
    BackendKind,           # pandas_numpy | polars | duckdb_sql | clickhouse_sql
    ExecutionKind,         # native_expr | polars_pandas_delegate | ...
    CapabilityLevel,       # production_safe | parity_verified | ...
    PhysicalImplementationSpec,  # Explicit operator implementation declaration
)
```

### No Duplicate Authorities

The following modules re-export these types but **do not define their own versions**:

- `backend.polars_backend_kind` re-exports `ExecutionKind` and `PhysicalImplementationSpec`
- All other backend modules import from `backend.contracts`

**Test:** Verify with `is` identity check:

```python
from backend.contracts import ExecutionKind as ContractsExecutionKind
from backend.polars_backend_kind import ExecutionKind as PolarsExecutionKind

assert ContractsExecutionKind is PolarsExecutionKind  # Same object
```

## Unified Capability Record (FE-BE-P0-002)

### Merged Types

Previously, the codebase had two capability record types with different field types:

- `BackendCapability`: Used string types (`execution_kind: str`, `status: str`)
- `BackendCapabilityRecord`: Used enum types (`execution_kind: ExecutionKind`, `level: CapabilityLevel`)

These are now **unified into a single type**:

```python
@dataclass(frozen=True)
class BackendCapability:
    canonical: str
    backend: BackendKind              # Enum, not string
    level: CapabilityLevel            # Enum, not string
    execution_kind: ExecutionKind     # Enum, not string
    
    # Feature support flags
    supports_nulls: bool = False
    supports_nan: bool = False
    supports_inf: bool = False
    # ... (other flags)
    
    # Legacy compatibility
    @property
    def status(self) -> str:
        """Legacy property mapping level to status string."""
        return self.level.value
```

**Alias for backward compatibility:**

```python
BackendCapabilityRecord = BackendCapability  # They are the same type
```

### Usage

All capability queries return the unified `BackendCapability`:

```python
from backend.operator_capability import capability_for

cap = capability_for("ts_mean", "pandas_numpy")

# All fields are properly typed enums
assert isinstance(cap.backend, BackendKind)
assert isinstance(cap.level, CapabilityLevel)
assert isinstance(cap.execution_kind, ExecutionKind)

# Legacy string access still works
assert cap.status == "production_safe"  # Returns string
assert cap.status == cap.level.value    # Same value
```

### CSV Export

The `to_csv_row()` method automatically converts enums to strings for serialization:

```python
row = cap.to_csv_row()
assert isinstance(row["backend"], str)           # "pandas_numpy"
assert isinstance(row["level"], str)             # "production_safe"
assert isinstance(row["execution_kind"], str)    # "native_expr"
```

## Production Fail-Closed (FE-BE-P0-003)

### Explicit Specs Required

Production systems must use **explicit `PhysicalImplementationSpec`** declarations. Heuristic source inspection is for research/diagnostics only.

#### Operator Declaration

```python
from backend.contracts import ExecutionKind, PhysicalImplementationSpec

class MyOperator:
    canonical = "my_operator"
    
    # REQUIRED for production use
    _physical_spec = PhysicalImplementationSpec(
        canonical="my_operator",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
    )
    
    def _calculate_series(self, df):
        return df  # Implementation
```

#### Production Mode

Functions that classify backend capabilities now support `production_mode`:

```python
from backend.polars_backend_kind import polars_backend_kind

# Production mode (default): requires explicit spec, warns on missing
kind = polars_backend_kind(operator, production_mode=True)

# Research mode: falls back to heuristics with warning
kind = polars_backend_kind(operator, production_mode=False)
```

**Behavior:**

| Scenario | `production_mode=True` | `production_mode=False` |
|----------|------------------------|-------------------------|
| Has `_physical_spec` | Uses explicit spec (no warning) | Uses explicit spec (no warning) |
| Missing `_physical_spec` | Returns `UNSUPPORTED` + warns | Uses heuristics + warns |

### Research-Only Functions

Functions marked **RESEARCH/DIAGNOSTIC USE ONLY**:

```python
from backend.polars_backend_kind import capability_quality

# Docstring explicitly states research-only use
help(capability_quality)  # Shows "RESEARCH/DIAGNOSTIC USE ONLY (FE-BE-P0-003)"

# Has production_mode parameter
quality = capability_quality("ts_mean", "polars", production_mode=False)
```

**Do not use these functions for:**
- Production routing decisions
- Admission gates
- Evidence certification
- Cost estimation in production plans

**Use them for:**
- Debugging/diagnostics
- Audit reports
- Research exploration

## Spec Error Handling (FE-BE-P0-004)

### Production Eligibility

`PhysicalImplementationSpec` provides `is_production_eligible()` to check if an implementation can be used in production:

```python
from backend.contracts import ExecutionKind, PhysicalImplementationSpec

spec_native = PhysicalImplementationSpec(
    canonical="test",
    backend="polars",
    execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
)
assert spec_native.is_production_eligible()  # True

spec_delegate = PhysicalImplementationSpec(
    canonical="test",
    backend="polars",
    execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
)
assert not spec_delegate.is_production_eligible()  # False - delegates not prod-safe
```

### Error Categories

When retrieving a spec fails, distinguish:

1. **MISSING**: Operator doesn't declare `_physical_spec` (expected during migration)
2. **INVALID**: Spec exists but malformed/inconsistent (hard error)
3. **INFRASTRUCTURE_ERROR**: System error retrieving spec (hard error)

Production routers must **hard-fail** on INVALID and INFRASTRUCTURE_ERROR, not silently fall back.

## Migration Guide

### For Operator Authors

1. Add explicit `_physical_spec` to all production operators:

```python
from backend.contracts import ExecutionKind, PhysicalImplementationSpec

class MyOperator:
    canonical = "my_operator"
    
    # Add this declaration
    _physical_spec = PhysicalImplementationSpec(
        canonical="my_operator",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
    )
```

2. Choose the correct `ExecutionKind`:
   - `POLARS_NATIVE_EXPR`: Uses Polars expression API (`pl.col()`, etc.)
   - `POLARS_NUMPY_KERNEL`: Per-column numpy kernel via `map_elements()`
   - `POLARS_PANDAS_DELEGATE`: Delegates to pandas (gap coverage only)
   - `DUCKDB_NATIVE_SQL`: Native DuckDB SQL pushdown

### For Backend Consumers

1. Replace `BackendCapabilityRecord` with `BackendCapability`:

```python
# Before
from backend.operator_capability import BackendCapabilityRecord
record: BackendCapabilityRecord = ...

# After (both work, but prefer BackendCapability)
from backend.operator_capability import BackendCapability
record: BackendCapability = ...
```

2. Handle enum types instead of strings:

```python
# Before
if cap.status == "production_safe":
    ...

# After (preferred)
from backend.contracts import CapabilityLevel
if cap.level == CapabilityLevel.PRODUCTION_SAFE:
    ...

# Or use legacy property (still works)
if cap.status == "production_safe":
    ...
```

3. Stop using heuristic functions in production routing:

```python
# DON'T do this in production routers:
from backend.polars_backend_kind import capability_quality
quality = capability_quality(canonical, backend)  # Research only!

# DO use explicit capability queries:
from backend.operator_capability import capability_for
cap = capability_for(canonical, backend)
if cap.level == CapabilityLevel.PRODUCTION_SAFE:
    ...
```

## Testing

Tests proving the unified contract system:

- `tests/backend/test_unified_capability_contracts.py`: Full contract verification
- `tests/backend/test_r13_backend_honesty.py`: Updated to use `PolarsImplementationKind`

Run tests:

```bash
python3 -m pytest tests/backend/test_unified_capability_contracts.py -xvs
```

## References

- **FE-BE-P0-001**: Single authority for BackendKind, ExecutionKind, CapabilityLevel
- **FE-BE-P0-002**: Merged BackendCapability/BackendCapabilityRecord
- **FE-BE-P0-003**: Production fail-closed on missing specs
- **FE-BE-P0-004**: Distinguish MISSING vs INVALID vs INFRASTRUCTURE_ERROR

---

**Last Updated:** 2026-08-14  
**Reviewers:** Factor Engine Core Team
