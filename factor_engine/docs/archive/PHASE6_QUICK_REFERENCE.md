# Phase 6 Quick Reference Guide

## File Locations

All Phase 6 modules are in: `/home/shw/quant_projects/factor_engine/cleaned_operators/common/`

```
polars_relation.py      - Module 24 (25 operators)
polars_fiscal.py        - Module 25 (18 operators)
polars_report.py        - Module 26 (7 operators)
polars_period.py        - Module 27 (5 operators)
polars_update.py        - Module 28 (4 operators)
polars_event.py         - Module 29 (30 operators)
polars_ts_complex.py    - Module 30 (44+ operators)
```

## Implementation Status

**Status:** ✓ Structural Complete, Algorithms Pending  
**Operators:** 133 skeleton implementations  
**Tests:** All modules import successfully  

## Next Steps for Implementation

### 1. Pick an Operator to Implement

Example: `relation_hhi` in `polars_relation.py`

### 2. Locate the TODO

```python
def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
    def _xform(long: pl.DataFrame) -> pl.DataFrame:
        total = pl.col("_v").sum().over("_r")
        share = pl.col("_v") / total
        hhi = (share * share).sum().over("_r")
        return long.with_columns(hhi.alias("_v"))

    return _cs_long_transform(x, _xform)
```

Note: This one is actually implemented! Most others have:
```python
# TODO: Implement complex algorithm
cols = _numeric_cols(x)
return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])
```

### 3. Replace Placeholder with Algorithm

**Pattern for time-series operators:**
```python
def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
    from cleaned_operators.parameter_validation import strict_integer
    w = strict_integer(window, "window", minimum=1)
    
    cols = _numeric_cols(x)
    return x.with_columns([
        pl.col(c).rolling_mean(window_size=w).alias(c) 
        for c in cols
    ])
```

**Pattern for cross-sectional operators:**
```python
def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
    def _xform(long: pl.DataFrame) -> pl.DataFrame:
        # Implement cross-sectional logic here
        result = pl.col("_v").mean().over("_r")
        return long.with_columns(result.alias("_v"))
    
    return _cs_long_transform(x, _xform)
```

### 4. Write Tests

Create test file: `tests/operators/test_polars_relation.py`

```python
import polars as pl
import pytest
from cleaned_operators.common.polars_relation import RelationHHINative

def test_relation_hhi_basic():
    op = RelationHHINative()
    df = pl.DataFrame({
        "date": ["2020-01-01"] * 3,
        "stock1": [0.5, 0.3, 0.2],
        "stock2": [0.3, 0.4, 0.3],
        "stock3": [0.2, 0.3, 0.5],
    })
    result = op._calculate_series(df)
    assert result is not None
    # Add assertions for expected values
```

## Common Patterns

### Parameter Validation
```python
from cleaned_operators.parameter_validation import (
    strict_integer,
    strict_finite_scalar
)

# For integer parameters
w = strict_integer(window, "window", minimum=1)

# For float parameters
alpha = strict_finite_scalar(alpha, "alpha", minimum=0.0, maximum=1.0)
```

### Helper Functions
```python
# Get numeric columns (excluding date, stock_code)
cols = _numeric_cols(df)

# Add date back to result
result = _with_meta(result_df, source_df)

# Cross-sectional transformation
return _cs_long_transform(x, transform_function)

# Group transformation
return _group_long_transform(x, group, transform_function)
```

### Polars Expressions
```python
# Rolling operations
pl.col(c).rolling_mean(window_size=w)
pl.col(c).rolling_std(window_size=w)
pl.col(c).rolling_sum(window_size=w)

# Window operations (cross-sectional)
pl.col("_v").mean().over("_r")
pl.col("_v").std().over("_r")
pl.col("_v").quantile(0.5).over("_r")

# Conditional logic
pl.when(condition).then(value).otherwise(alternative)

# Handle NaN/null
pl.col(c).fill_nan(None)
pl.col(c).is_null()
pl.col(c).is_not_null()
```

## Priority Implementation List

### High Priority (Common Usage)
1. **relation_hhi** - Already implemented!
2. **relation_entropy** - Information theory fundamental
3. **fiscal_pct_change** - Basic financial analysis
4. **fiscal_standardized_surprise** - SUE widely used
5. **event_frequency** - Basic event counting
6. **event_cumulative_return_past** - Event study fundamental

### Medium Priority (Advanced Analytics)
7. **ts_kalman_filter** - State estimation
8. **ts_garch_volatility** - Volatility modeling
9. **ts_hurst_exponent** - Long-range dependence
10. **ts_spectral_density** - Frequency analysis

### Lower Priority (Specialized)
- Chaos theory operators (ts_lyapunov_*, ts_correlation_dimension)
- Advanced RQA operators (ts_determinism, ts_laminarity)
- Pattern discovery (ts_matrix_profile_*, ts_motif_*)

## Testing Strategy

### Unit Tests
```bash
# Test single module
pytest tests/operators/test_polars_relation.py -v

# Test all Phase 6 modules
pytest tests/operators/test_polars_relation.py \
       tests/operators/test_polars_fiscal.py \
       tests/operators/test_polars_event.py -v
```

### Integration Tests
```python
# Test operator registration
from cleaned_operators.base_polars import get_registered_operators
ops = get_registered_operators(backend="polars")
assert "relation_hhi" in ops
```

### Validation Tests
```python
# Test parameter validation
op = RelationHHINative()
with pytest.raises(OperatorParameterError):
    op._calculate_series(df, invalid_param=-1)
```

## Documentation Standards

### Operator Docstring
```python
class OperatorNameNative(SeriesOperator):
    """Brief description.
    
    Mathematical definition if applicable:
    HHI = Σ(share_i)²
    
    References:
    - Paper/Book citation if applicable
    """
```

### Parameter Documentation
Add to metadata description field (Chinese) and docstring (English).

## Common Pitfalls to Avoid

1. **Don't use pandas** - Only Polars expressions
2. **Don't use .to_numpy()** - Breaks lazy evaluation
3. **Validate parameters** - Always use strict_integer/strict_finite_scalar
4. **Handle NaN properly** - Use fill_nan(None) for cross-sectional ops
5. **Test edge cases** - Empty data, single row, all nulls
6. **Match param_specs** - Ensure ParamRole matches parameter usage

## Resources

- **Reference implementation:** `polars_daily_native.py`
- **Base classes:** `cleaned_operators/base_polars.py`
- **Parameter validation:** `cleaned_operators/parameter_validation.py`
- **Polars docs:** https://pola-rs.github.io/polars/

## Questions?

Check the reference pattern in `polars_daily_native.py` - it has complete, working examples of:
- Time-series operators (ts_log_return, ts_rank, ts_sharpe)
- Cross-sectional operators (cs_std, cs_mad, cs_mad_zscore)
- Group operators (group_neutralize, group_normalize, group_winsorize)

---

**Implementation Guide Version:** 1.0  
**Last Updated:** 2026-08-13
