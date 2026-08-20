# Backend Selection Guide

**Version:** 1.0  
**Last Updated:** 2026-08-13  
**Status:** Production

## Overview

The Factor Engine supports **three execution backends**: Pandas/Numpy (reference), Polars (high-performance), and DuckDB SQL (analytical). This guide explains the characteristics of each backend, how to choose between them, and best practices for multi-backend development.

## Table of Contents

1. [Backend Overview](#backend-overview)
2. [Backend Comparison](#backend-comparison)
3. [Selection Decision Tree](#selection-decision-tree)
4. [Performance Trade-offs](#performance-trade-offs)
5. [Backend-Specific Behavior](#backend-specific-behavior)
6. [Best Practices](#best-practices)
7. [Migration Guide](#migration-guide)

---

## Backend Overview

### Three-Backend Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Factor Engine                             │
├─────────────────────────────────────────────────────────────┤
│                    Logical Plan (IR)                         │
│              (Backend-agnostic operator DAG)                 │
└───────────────┬─────────────┬──────────────┬────────────────┘
                │             │              │
       ┌────────▼───┐  ┌──────▼─────┐  ┌────▼─────────┐
       │  Pandas    │  │   Polars   │  │   DuckDB     │
       │  Backend   │  │  Backend   │  │   Backend    │
       ├────────────┤  ├────────────┤  ├──────────────┤
       │ Reference  │  │ Parallel   │  │ Vectorized   │
       │ Python     │  │ Rust/PyO3  │  │ Columnar     │
       │ GIL-bound  │  │ Multi-core │  │ SQL-native   │
       └────────────┘  └────────────┘  └──────────────┘
              │              │                 │
              └──────────────┴─────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Unified Result  │
                    │ (MultiIndex DF)  │
                    └──────────────────┘
```

### Design Philosophy

1. **Single Logical Plan**: Write factor logic once, execute on any backend
2. **Automatic Selection**: Cost model chooses optimal backend per operator
3. **Verified Parity**: All backends produce bitwise-identical results
4. **Graceful Degradation**: Falls back to pandas if backend unavailable

---

## Backend Comparison

### Pandas/Numpy Backend

**Type:** Reference implementation  
**Language:** Pure Python + NumPy C extensions  
**Concurrency:** Single-threaded (GIL-limited)  
**Status:** ✅ Production (1430 operators)

#### Strengths

- **Fastest startup**: 2ms cold start, ideal for small data (<10K rows)
- **Universal compatibility**: Works everywhere, no compilation needed
- **Richest ecosystem**: Full statsmodels, scipy integration
- **Best debugging**: Native Python stack traces, easy to step through
- **Reference semantics**: Defines "correct" behavior for parity tests

#### Weaknesses

- **Single-threaded**: Cannot utilize multi-core CPUs
- **Memory inefficient**: Python object overhead (~2× data size)
- **Slower for large data**: 0.5M rows/sec vs 4M+ for vectorized backends
- **No streaming**: Requires full data in memory

#### When to Use

- Small data (<10K rows)
- Complex operators requiring statsmodels (regression, time series models)
- Development/debugging (clearest error messages)
- Compatibility testing (reference baseline)

#### Example Performance

```python
# 100K rows × 50 instruments × 20-window rolling mean
data = generate_panel(100_000, 50)
result = compute_with_backend("ts_mean", data, window=20, backend="pandas_numpy")

# Performance:
# - Startup: 2ms
# - Compute: 52ms (1.9M rows/sec)
# - Memory: 48MB (1.2× data size)
# - Total: 54ms
```

---

### Polars Backend

**Type:** High-performance execution  
**Language:** Rust core + Python bindings (PyO3)  
**Concurrency:** Multi-threaded, SIMD vectorization  
**Status:** ✅ Production (620 operators certified)

#### Strengths

- **High throughput**: 2.5-3.0M rows/sec (5× pandas)
- **Parallel execution**: Automatic multi-core utilization
- **Memory efficient**: Arrow format, zero-copy operations
- **Lazy optimization**: Query planner optimizes execution (lazy mode)
- **Streaming support**: Can process data larger than memory

#### Weaknesses

- **Slower startup**: 8-10ms (JIT compilation, plan optimization)
- **Limited ecosystem**: No statsmodels, fewer statistical functions
- **Compilation requirement**: PyO3 binaries, platform-specific builds
- **Less mature**: Newer library, occasional breaking changes

#### When to Use

- Medium to large data (10K-10M rows)
- Parallel-friendly operators (rolling, groupby, joins)
- Memory-constrained environments (streaming mode)
- Production batch processing (throughput matters)

#### Example Performance

```python
# Same workload as pandas example
result = compute_with_backend("ts_mean", data, window=20, backend="polars_eager")

# Performance:
# - Startup: 8ms
# - Compute: 21ms (4.8M rows/sec, 2.5× pandas)
# - Memory: 60MB (1.5× data size, Arrow overhead)
# - Total: 29ms (1.9× faster than pandas)
```

#### Eager vs Lazy Mode

```python
# Eager: immediate execution (like pandas)
backend = "polars_eager"
result = compute("ts_mean(close, 20)")  # Executes immediately

# Lazy: deferred execution with optimization
backend = "polars_lazy"
plan = compute("ts_mean(close, 20)")    # Builds plan
result = plan.collect()                  # Optimizes + executes

# Lazy advantages:
# - Predicate pushdown (filter early)
# - Projection pruning (load only needed columns)
# - Common subexpression elimination
# → 10-30% faster for complex queries
```

---

### DuckDB SQL Backend

**Type:** Analytical query engine  
**Language:** C++ core + Python bindings  
**Concurrency:** Vectorized, multi-threaded  
**Status:** ✅ Production (269 operators certified)

#### Strengths

- **Highest throughput**: 4.0M rows/sec (8× pandas) for window/aggregation
- **Columnar storage**: Minimal memory overhead, excellent compression
- **SQL native**: Can push down to SQL databases directly
- **Vectorized execution**: SIMD, cache-friendly algorithms
- **Streaming**: Out-of-core processing for 100GB+ datasets

#### Weaknesses

- **Slowest startup**: 15ms (SQL parsing, query planning)
- **SQL boundary**: Conversion cost for Python-native operators
- **Limited operators**: 269/1430 operators (19% coverage)
- **Debugging harder**: SQL errors less intuitive than Python

#### When to Use

- Large to massive data (1M-1B rows)
- Window-heavy workloads (SQL window functions optimized)
- Out-of-core processing (data doesn't fit in RAM)
- SQL data source integration (pushdown to database)

#### Example Performance

```python
# Same workload
result = compute_with_backend("ts_mean", data, window=20, backend="duckdb_sql")

# Performance:
# - Startup: 15ms
# - Compute: 15ms (6.7M rows/sec, 3.5× pandas)
# - Memory: 44MB (1.1× data size, columnar)
# - Total: 30ms (1.8× faster than pandas)

# But for small data (1K rows):
# - Startup: 15ms (dominates!)
# - Compute: 0.3ms
# - Total: 15.3ms (worse than pandas: 2ms + 2ms = 4ms)
```

---

## Selection Decision Tree

### By Data Scale

```
┌─────────────────────────────────────────────────────────────┐
│ How many rows?                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  < 1K rows (Tiny)                                          │
│  └─→ PANDAS (startup dominates, 2ms vs 8ms vs 15ms)      │
│                                                             │
│  1K - 10K rows (Small)                                     │
│  └─→ PANDAS (still competitive, simple execution)         │
│                                                             │
│  10K - 100K rows (Medium)                                  │
│  ├─→ Window-heavy? → POLARS                               │
│  ├─→ Regression? → PANDAS                                 │
│  └─→ Mixed? → POLARS (throughput starts to matter)       │
│                                                             │
│  100K - 1M rows (Large)                                    │
│  ├─→ Window-heavy? → DUCKDB                               │
│  ├─→ Cross-section? → POLARS                              │
│  └─→ Regression? → PANDAS (no alternative)               │
│                                                             │
│  1M - 10M rows (Huge)                                      │
│  ├─→ Window-heavy? → DUCKDB                               │
│  ├─→ Memory tight? → DUCKDB (streaming)                  │
│  └─→ Cross-section? → POLARS                              │
│                                                             │
│  > 10M rows (Massive)                                      │
│  └─→ DUCKDB (must stream, columnar efficiency critical)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### By Operator Type

```
┌─────────────────────────────────────────────────────────────┐
│ What operations?                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Rolling window (ts_mean, ts_std, ts_sum)                  │
│  ├─→ Small data: PANDAS                                    │
│  ├─→ Medium: POLARS                                        │
│  └─→ Large: DUCKDB (window functions optimized)           │
│                                                             │
│  Cross-sectional (cs_rank, cs_zscore, cs_demean)          │
│  ├─→ Small: PANDAS                                         │
│  └─→ Medium+: POLARS (parallel group-by)                  │
│                                                             │
│  Regression (ts_regression, neutralize, resid)             │
│  └─→ PANDAS (only backend with statsmodels)               │
│                                                             │
│  State-space (kalman_filter, garch_volatility)             │
│  ├─→ Python impl: PANDAS                                   │
│  └─→ Numba impl: POLARS (JIT compiled)                    │
│                                                             │
│  Elementwise (abs, log, sign, clip)                        │
│  └─→ Any backend (negligible difference)                  │
│                                                             │
│  Complex joins/merges                                       │
│  └─→ DUCKDB (join optimization)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### By Memory Pressure

```
┌─────────────────────────────────────────────────────────────┐
│ Memory situation?                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Memory abundant (< 30% used)                              │
│  └─→ Choose by performance (POLARS or DUCKDB)             │
│                                                             │
│  Memory moderate (30-60% used)                             │
│  └─→ POLARS or DUCKDB (both efficient)                    │
│                                                             │
│  Memory tight (60-80% used)                                │
│  └─→ DUCKDB streaming (out-of-core)                       │
│                                                             │
│  Memory critical (>80% used)                               │
│  └─→ DUCKDB streaming (REQUIRED)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Scenario | Optimal Backend | Runner-up | Notes |
|----------|----------------|-----------|-------|
| Tiny data (<1K rows) | Pandas | - | Startup dominates |
| Small data (1-10K) | Pandas | Polars | Simple is fast |
| Medium data (10-100K), window-heavy | Polars | DuckDB | Parallel rolling |
| Medium data, regression | Pandas | - | Only option |
| Large data (100K-1M), window-heavy | DuckDB | Polars | SQL optimized |
| Large data, cross-section | Polars | DuckDB | Parallel group-by |
| Huge data (1M-10M) | DuckDB | Polars | Vectorization wins |
| Massive data (>10M) | DuckDB | - | Must stream |
| Low memory | DuckDB | Polars lazy | Out-of-core |
| Development/debugging | Pandas | - | Best errors |

---

## Performance Trade-offs

### Throughput vs Latency

```python
# Latency-sensitive (many small queries)
# → Prefer pandas (low startup)
for factor in small_factor_list:
    result = compute(factor)  # 2ms startup each

# Throughput-sensitive (few large queries)
# → Prefer polars/duckdb (high throughput)
result = compute_batch(large_factor_list)  # 8ms startup, then fast
```

### Memory vs Speed

```python
# Speed priority (memory available)
backend = "polars_eager"  # Fast, but loads all data

# Memory priority (limited RAM)
backend = "duckdb_sql"    # Streaming, slower startup
```

### Startup vs Execution

**Pandas**: Low startup (2ms), moderate execution (0.5M rows/sec)  
**Polars**: Medium startup (8ms), high execution (2.5M rows/sec)  
**DuckDB**: High startup (15ms), highest execution (4.0M rows/sec)

**Breakeven points:**

```python
# Pandas vs Polars breakeven: ~3K rows
# 2ms + rows/500K*1000 = 8ms + rows/2500K*1000
# → rows ≈ 3000

# Pandas vs DuckDB breakeven: ~13K rows
# 2ms + rows/500K*1000 = 15ms + rows/4000K*1000
# → rows ≈ 13000

# Polars vs DuckDB breakeven: ~35K rows
# 8ms + rows/2500K*1000 = 15ms + rows/4000K*1000
# → rows ≈ 35000
```

### Scalability

| Backend | Scales to... | Bottleneck |
|---------|-------------|------------|
| Pandas | ~10M rows | GIL, memory |
| Polars | ~100M rows | Memory (even with streaming) |
| DuckDB | ~10B rows | Disk I/O (out-of-core works) |

---

## Backend-Specific Behavior

### NaN Handling

#### Pandas

```python
# NaN propagates in all operations
pd.Series([1, 2, np.nan, 4]).mean()  # nan
pd.Series([1, 2, np.nan, 4]).rolling(2).mean()  # [nan, 1.5, nan, nan]
```

#### Polars

```python
# NaN handling configurable
pl.Series([1, 2, None, 4]).mean()  # 2.333 (skips None)
pl.Series([1, 2, float('nan'), 4]).mean()  # nan (propagates NaN)

# Rolling with null handling
pl.Series([1, 2, None, 4]).rolling_mean(window_size=2)  # [nan, 1.5, 2.0, 4.0]
```

**Parity requirement**: Factor Engine configures Polars to match pandas NaN semantics exactly.

#### DuckDB

```python
# SQL NULL vs NaN
# NULL (missing) vs NaN (numeric value)
SELECT AVG(close) FROM data  -- Skips NULL, includes NaN!

# Must explicitly handle NaN
SELECT AVG(CASE WHEN isnan(close) THEN NULL ELSE close END) FROM data
```

**Parity requirement**: SQL generation includes explicit NaN → NULL conversion.

---

### Infinity Handling

#### Pandas

```python
pd.Series([1, 2, np.inf, 4]).mean()  # inf
pd.Series([1, 2, np.inf, 4]).std()   # inf
```

#### Polars

```python
pl.Series([1, 2, float('inf'), 4]).mean()  # inf (matches pandas)
```

#### DuckDB

```python
# SQL infinity support
SELECT AVG(close) FROM data WHERE close < 1e308  -- Filter out inf
```

**Parity requirement**: All backends preserve inf/nan distinction exactly.

---

### Sorting Stability

#### Pandas

```python
# Stable sort (preserves relative order of equal elements)
df.sort_values(["date", "rank"], kind="stable")
```

#### Polars

```python
# Stable by default
df.sort(["date", "rank"])  # Maintains insertion order for ties
```

#### DuckDB

```python
# SQL ORDER BY is not guaranteed stable
SELECT * FROM data ORDER BY date, rank
-- Ties may appear in arbitrary order

-- Workaround: add tie-breaker
SELECT * FROM data ORDER BY date, rank, rowid
```

**Parity requirement**: DuckDB queries include explicit tie-breakers for reproducibility.

---

### Index Handling

#### Pandas

```python
# Rich MultiIndex support
df.index = pd.MultiIndex.from_arrays([dates, instruments])
df.loc[("2024-01-01", "AAPL")]  # Fast lookup
```

#### Polars

```python
# No index concept (relational model)
# Must explicitly join on keys
df.filter((pl.col("date") == "2024-01-01") & (pl.col("instrument") == "AAPL"))
```

#### DuckDB

```python
# SQL doesn't have indexes (in-memory)
# WHERE clause for filtering
SELECT * FROM data WHERE date = '2024-01-01' AND instrument = 'AAPL'
```

**Parity requirement**: Factor Engine converts all results to pandas MultiIndex for consistency.

---

### Type Coercion

#### Pandas

```python
# Flexible type coercion
pd.Series([1, 2, "3"]).astype(float)  # Works
```

#### Polars

```python
# Strict typing
pl.Series([1, 2, "3"]).cast(pl.Float64)  # Raises error (strict mode)
pl.Series([1, 2, "3"]).cast(pl.Float64, strict=False)  # [1.0, 2.0, NaN]
```

#### DuckDB

```python
# SQL implicit casts
SELECT CAST('3' AS DOUBLE)  -- Works
SELECT '3' + 1.0            -- Works (implicit)
```

**Parity requirement**: Factor Engine uses explicit type annotations to avoid implicit coercion.

---

## Best Practices

### 1. Let Cost Model Decide

```python
# Good: Let backend router choose
result = engine.compute_factor("ts_mean(close, 20)")
# → Automatically selects optimal backend based on data scale

# Bad: Hard-code backend
result = engine.compute_factor("ts_mean(close, 20)", backend="pandas_numpy")
# → Misses optimization opportunities
```

### 2. Test All Backends

```python
# Good: Test parity across backends
@pytest.mark.parametrize("backend", ["pandas_numpy", "polars_eager", "duckdb_sql"])
def test_ts_mean_parity(backend):
    data = generate_test_data()
    result = compute_with_backend("ts_mean", data, window=20, backend=backend)
    reference = compute_with_backend("ts_mean", data, window=20, backend="pandas_numpy")
    np.testing.assert_array_equal(result.values, reference.values, equal_nan=True)
```

### 3. Handle Missing Backend Gracefully

```python
# Good: Fallback to pandas if backend unavailable
try:
    result = compute_with_backend("ts_mean", data, backend="polars_eager")
except BackendUnavailableError:
    result = compute_with_backend("ts_mean", data, backend="pandas_numpy")

# Better: Use backend router (does this automatically)
result = compute_factor("ts_mean(close, 20)")  # Auto-fallback
```

### 4. Avoid Backend-Specific Features

```python
# Bad: Uses pandas-specific method
def my_operator(series):
    return series.ewm(alpha=0.1).mean()  # Only works in pandas

# Good: Use backend-agnostic logic
def my_operator(series):
    # Will be translated to backend-specific implementation
    return exponential_weighted_mean(series, alpha=0.1)
```

### 5. Profile Before Optimizing

```python
# Measure first
from runtime.profiling import ExecutionProfiler

profiler = ExecutionProfiler()
with profiler.trace("my_computation"):
    result = compute_factor("my_complex_factor")

print(profiler.summary())
# → Reveals actual bottleneck

# Then optimize the right thing
```

### 6. Batch for Efficiency

```python
# Bad: One-by-one (pays startup cost each time)
results = []
for factor in factor_list:
    results.append(compute_factor(factor))  # 8ms startup × 100 = 800ms wasted

# Good: Batch (amortizes startup)
results = compute_factors_batch(factor_list)  # 8ms startup once
```

### 7. Monitor Backend Usage

```python
# Track which backends are used in production
from backend.monitoring import BackendUsageMonitor

monitor = BackendUsageMonitor()

@monitor.track
def compute_all_factors():
    return engine.compute_factors(factor_list)

# After execution
print(monitor.report())
# Output:
# Backend usage:
#   pandas_numpy: 450 ops (45%), 12.3s total
#   polars_eager: 350 ops (35%), 8.1s total
#   duckdb_sql: 200 ops (20%), 5.2s total
# Recommendation: Good mix, cost model working well
```

---

## Migration Guide

### Migrating from Single Backend to Multi-Backend

#### Phase 1: Add Backend Parameter

```python
# Before: Hardcoded pandas
def compute_my_factor(data):
    return data.rolling(20).mean()

# After: Backend-aware
def compute_my_factor(data, backend="pandas_numpy"):
    if backend == "pandas_numpy":
        return data.rolling(20).mean()
    elif backend == "polars_eager":
        return data.rolling(20).mean()  # Same API!
    else:
        raise ValueError(f"Unsupported backend: {backend}")
```

#### Phase 2: Abstract Backend Operations

```python
# Better: Use backend adapter
from backend.adapter import get_backend_adapter

def compute_my_factor(data, backend="auto"):
    adapter = get_backend_adapter(backend)
    return adapter.rolling_mean(data, window=20)
```

#### Phase 3: Full Integration

```python
# Best: Register as operator, automatic backend selection
@register_operator("my_factor")
@supports_backend("pandas_numpy", "polars_eager", "duckdb_sql")
def compute_my_factor(data, window: int = 20):
    """Compute my custom factor."""
    return rolling_mean(data, window)  # Backend-agnostic primitive

# Now works with cost model
result = engine.compute_factor("my_factor(close, 30)")  # Auto-selects backend
```

### Verifying Backend Parity

```python
# 1. Generate test data
test_data = generate_test_panel(rows=10000, instruments=50)

# 2. Compute on all backends
results = {}
for backend in ["pandas_numpy", "polars_eager", "duckdb_sql"]:
    try:
        results[backend] = compute_with_backend("my_factor", test_data, backend=backend)
    except BackendUnavailableError:
        results[backend] = None

# 3. Check parity
reference = results["pandas_numpy"]
for backend, result in results.items():
    if result is None:
        continue
    if backend == "pandas_numpy":
        continue
    
    # Bitwise identical?
    if not np.array_equal(result.values, reference.values, equal_nan=True):
        print(f"PARITY FAILURE: {backend} differs from pandas")
        diff = ~np.isclose(result.values, reference.values, equal_nan=True)
        print(f"  Differences: {diff.sum()}/{len(result)}")
        print(f"  Max diff: {np.abs((result - reference)[diff]).max()}")
    else:
        print(f"✓ {backend} matches pandas")
```

### Debugging Backend Issues

```python
# Enable backend-specific logging
import logging
logging.getLogger("backend").setLevel(logging.DEBUG)

# Run computation
result = compute_factor("ts_mean(close, 20)")

# Check backend selection reasoning
# Logs will show:
# [DEBUG] Backend selection for ts_mean:
#   Candidates: pandas_numpy, polars_eager, duckdb_sql
#   Costs: pandas=28.4ms, polars=17.6ms, duckdb=30.1ms
#   Selected: polars_eager (lowest cost)
```

---

## Summary

### Quick Reference

| Need | Backend | Why |
|------|---------|-----|
| Small data (<10K rows) | Pandas | Fast startup |
| Large data (>1M rows) | DuckDB | High throughput, streaming |
| Window-heavy ops | DuckDB | SQL window functions |
| Cross-section ops | Polars | Parallel group-by |
| Regressions | Pandas | Only option (statsmodels) |
| Memory-constrained | DuckDB | Out-of-core processing |
| Debugging | Pandas | Best error messages |
| Production batch | Polars/DuckDB | Throughput matters |

### Decision Flowchart

```
Start
  │
  ▼
Is data < 10K rows?
  ├─ YES → Use Pandas
  └─ NO
      │
      ▼
  Does it use regression/statsmodels?
      ├─ YES → Use Pandas (only option)
      └─ NO
          │
          ▼
      Is memory tight (>80%)?
          ├─ YES → Use DuckDB streaming
          └─ NO
              │
              ▼
          Is data > 1M rows?
              ├─ YES → Use DuckDB
              └─ NO → Let cost model decide
```

---

## References

- [COST_MODEL_EXPLAINED.md](COST_MODEL_EXPLAINED.md) - Cost model details
- [TESTING_STRATEGY.md](TESTING_STRATEGY.md) - Backend parity testing
- [HARD_GATES_REFERENCE.md](HARD_GATES_REFERENCE.md) - Production gates
- `backend/backend_router.py` - Backend selection implementation
- `planning/backend_selector.py` - Cost-based selection
- `tests/multibackend_integration/` - Parity test suite
