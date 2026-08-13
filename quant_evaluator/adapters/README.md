# Adapters Layer

## Overview

The `adapters/` layer provides optional integration boundaries for external platform components (DataAccess, FactorEngine) and explicit reference/debug tools (pandas).

**Core principle**: Core `quant_evaluator` modules MUST NOT import adapters. Adapters are lazy-loaded and fail gracefully when dependencies are unavailable.

## Modules

### 1. `data_access.py` - DataAccess Integration (Optional)

Provides Protocol-based boundaries for DataAccess integration:

- **`ContextProvider`**: Protocol for calendar/timing queries
- **`UniverseProvider`**: Protocol for universe/asset queries  
- **`DataAccessAdapter`**: Converts DA reads to `FactorBatch`/`LabelBundle`

**Status**: Stub implementation. Full implementation blocked on DA contract freeze.

**Usage**:
```python
from quant_evaluator.adapters.data_access import DataAccessAdapter

adapter = DataAccessAdapter()  # Raises OptionalDependencyMissing if DA not installed
batch = adapter.da_frame_to_factor_batch(df, factor_ids=["momentum", "value"])
```

### 2. `factor_engine.py` - FactorEngine Integration (Optional)

Provides Protocol-based boundaries for FactorEngine integration:

- **`FactorBatchProvider`**: Protocol for factor execution
- **`FactorIdentityProvider`**: Protocol for canonical factor IDs
- **`FactorEngineAdapter`**: Converts FE results to `FactorBatch`, extracts canonical IDs

**Status**: Stub implementation. Full implementation blocked on FE contract freeze.

**Design**: Delegates to FE public API only. Never duplicates FE kernels or identity logic.

**Usage**:
```python
from quant_evaluator.adapters.factor_engine import FactorEngineAdapter

adapter = FactorEngineAdapter()  # Raises OptionalDependencyMissing if FE not installed
batch = adapter.fe_result_to_factor_batch(fe_result, factor_ids=["momentum_20d"])
factor_id = adapter.extract_factor_id(factor_expr)
```

### 3. `pandas.py` - Pandas Reference/Debug Adapter (Explicit)

Provides explicit conversions between pandas DataFrames and QE contracts.

**⚠️ REFERENCE/DEBUG ONLY**: This is NOT a production fallback. Use only for:
- Reference metric implementations
- Debug/validation workflows
- Small-scale testing with manual data
- Prototype evaluation scripts

**Key conversions**:
- `dataframe_to_factor_batch()`: DataFrame → FactorBatch
- `factor_batch_to_dataframe()`: FactorBatch → DataFrame
- `dataframe_to_label_bundle()`: DataFrame → LabelBundle
- `label_bundle_to_dataframe()`: LabelBundle → DataFrame

**Usage**:
```python
from quant_evaluator.adapters.pandas import PandasAdapter

adapter = PandasAdapter()  # Raises OptionalDependencyMissing if pandas not installed
batch = adapter.dataframe_to_factor_batch(
    df,
    factor_cols=["momentum", "value"],
    time_col="date",
    asset_col="code"
)
```

## Design Principles

1. **Lazy Import**: Adapters use lazy import patterns. Missing dependencies raise `OptionalDependencyMissing`.

2. **Protocol-Based Boundaries**: Public APIs use Protocol types, not concrete DA/FE types.

3. **No Core Dependencies**: Core QE modules never import adapters. Adapters are opt-in.

4. **Fail-Closed**: Adapters raise on ambiguous conversions rather than guessing.

5. **No Kernel Duplication**: FE adapter delegates to FE public API; never duplicates operator kernels or identity logic.

6. **Explicit Reference**: Pandas adapter is explicitly documented as reference/debug only, never an implicit production fallback.

## Testing

All adapters have comprehensive tests:

- Lazy import verification
- Protocol compliance
- Core isolation (core modules don't import adapters)
- Conversion logic (pandas adapter fully tested)
- Integration tests (DA/FE adapters, blocked on contract freeze)

Run tests:
```bash
pytest tests/adapters/ -v
```

## Implementation Status

| Module | Status | Blocker |
|--------|--------|---------|
| `data_access.py` | Stub | DA contract freeze |
| `factor_engine.py` | Stub | FE contract freeze |
| `pandas.py` | **Complete** | None |

## Future Work

1. **DA Adapter**: Implement full conversions after DA DataFrame/timing contract freeze
2. **FE Adapter**: Implement full conversions after FE MaterializedResult/identity API freeze
3. **Integration Tests**: Add real corpus tests after DA/FE contracts stabilize
4. **Advanced Protocols**: Consider streaming/chunked protocols for large batches

## Notes

- Adapters are NOT runtime dependencies for core QE
- Each adapter tests verify core isolation (no imports)
- Stub methods document their blockers explicitly
- All timing is explicit; adapters never infer dates from data
