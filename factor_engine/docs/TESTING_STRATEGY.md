# Testing Strategy

**Version:** 1.0  
**Last Updated:** 2026-08-13  
**Status:** Production

## Overview

The Factor Engine employs a multi-layered testing strategy designed to ensure correctness, performance, and production readiness across 85+ hard gates. This document describes the complete testing approach used to maintain quality across 1400+ operators and three execution backends.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Organization](#test-organization)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [Regression Testing Matrix](#regression-testing-matrix)
6. [Performance Testing](#performance-testing)
7. [Hard Gates & Certification](#hard-gates--certification)
8. [Backend Parity Testing](#backend-parity-testing)
9. [Running Tests](#running-tests)

---

## Testing Philosophy

### Core Principles

1. **Fail-Closed by Default**: Tests must explicitly verify correctness, not merely check for absence of errors
2. **Evidence-Based Certification**: Production readiness requires concrete test evidence, not assumptions
3. **Additive Testing**: New tests never break existing tests; regression failures signal real issues
4. **Serial Execution**: Tests run single-process to ensure determinism and avoid memory constraints
5. **Comprehensive Coverage**: Every operator must pass semantic, mathematical, and execution audits

### Testing Pyramid

```
                    ┌─────────────────┐
                    │  Hard Gates (85)│  ← Production readiness
                    └─────────────────┘
                  ┌─────────────────────┐
                  │ Integration Tests   │  ← Cross-component
                  └─────────────────────┘
              ┌───────────────────────────┐
              │   Backend Parity Tests    │  ← 3-backend consistency
              └───────────────────────────┘
          ┌─────────────────────────────────┐
          │     Operator Unit Tests         │  ← Individual operators
          └─────────────────────────────────┘
      ┌─────────────────────────────────────────┐
      │    Mathematical Correctness Tests       │  ← Ground truth oracles
      └─────────────────────────────────────────┘
```

---

## Test Organization

### Directory Structure

```
tests/
├── conftest.py              # Global pytest configuration
├── r25/ ... r47/            # Round-based audit test suites
├── operators/               # Operator-specific unit tests
├── backend/                 # Backend execution tests
├── multibackend_integration/# Cross-backend parity tests
├── fundamental/             # Fundamental data & PIT tests
├── runtime/                 # Runtime governance tests
├── perf/                    # Performance regression gates
└── operator_golden/         # Golden reference tests
```

### Round-Based Testing (R-Series)

The engine uses **audit rounds** (R25, R30, R32, etc.) for systematic quality gates:

- **R25**: DataAccess platform closure (875 tests)
- **R27**: Cache/Write/Atomic closure (922 tests)
- **R30**: All-operator final review (185 new tests)
- **R32**: System architecture & hard gates (50 gates)
- **R35**: Model operators & Numba kernels (87 tests)
- **R38**: Resource governance (35/1/0 gates)
- **R39**: Performance closure (295 tests, -26% TTDC)
- **R40**: 260-item full remediation (315+2 tests)

Each round has:
- Dedicated test directory (`tests/rXX/`)
- Audit script (`scripts/audit_rXX_hard_gates.py`)
- Closure report (`docs/RXX_*_REPORT.md`)

---

## Unit Testing

### Operator Unit Tests

**Location:** `tests/operators/`

Each operator must have tests covering:

1. **Basic functionality** with canonical inputs
2. **Edge cases**: NaN, Inf, empty panels, single instrument
3. **Parameter validation**: invalid params must fail
4. **Window behavior**: sufficient history requirements
5. **Cross-sectional behavior**: rank stability, neutralization correctness

#### Example: Testing `ts_mean` operator

```python
# tests/operators/test_ts_mean.py
import pandas as pd
import numpy as np
from cleaned_operators.registry import OperatorRegistry

def test_ts_mean_basic():
    """Basic rolling mean correctness."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    data = pd.Series(range(100), index=idx)
    
    result = OperatorRegistry.call("ts_mean", data, window=10)
    
    # Verify against pandas reference
    expected = data.rolling(window=10, min_periods=10).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_ts_mean_nan_handling():
    """NaN values should propagate correctly."""
    data = pd.Series([1, 2, np.nan, 4, 5])
    result = OperatorRegistry.call("ts_mean", data, window=3)
    
    # Window containing NaN should produce NaN
    assert pd.isna(result.iloc[2])
    
def test_ts_mean_insufficient_history():
    """Insufficient history produces NaN at start."""
    data = pd.Series(range(5))
    result = OperatorRegistry.call("ts_mean", data, window=10)
    
    # All values should be NaN (need 10, have 5)
    assert result.isna().all()

def test_ts_mean_panel():
    """Multi-instrument panel behavior."""
    idx = pd.MultiIndex.from_product([
        pd.date_range("2024-01-01", periods=50),
        ["A", "B", "C"]
    ], names=["date", "instrument"])
    
    data = pd.Series(np.random.randn(150), index=idx)
    result = OperatorRegistry.call("ts_mean", data, window=10)
    
    # Should maintain panel structure
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["date", "instrument"]
    
    # Each instrument computed independently
    for inst in ["A", "B", "C"]:
        inst_data = data.xs(inst, level="instrument")
        inst_result = result.xs(inst, level="instrument")
        expected = inst_data.rolling(10, min_periods=10).mean()
        pd.testing.assert_series_equal(inst_result, expected, check_names=False)
```

### Mathematical Correctness Tests

**Location:** `tests/operators/`, `tests/operator_golden/`

Operators with complex mathematics require **oracle validation**:

```python
# tests/operators/test_kalman_filter_correctness.py
import numpy as np
from pykalman import KalmanFilter as ReferenceKF
from cleaned_operators.registry import OperatorRegistry

def test_kalman_level_against_pykalman():
    """Kalman level estimation must match pykalman reference."""
    np.random.seed(42)
    observations = np.random.randn(100)
    
    # Our implementation
    result = OperatorRegistry.call("kalman_level", 
                                    pd.Series(observations),
                                    observation_covariance=1.0,
                                    transition_covariance=0.01)
    
    # Reference implementation
    kf = ReferenceKF(
        transition_matrices=[1],
        observation_matrices=[1],
        observation_covariance=1.0,
        transition_covariance=0.01,
        initial_state_mean=observations[0],
        initial_state_covariance=1.0
    )
    filtered_state_means, _ = kf.filter(observations)
    expected = pd.Series(filtered_state_means.flatten())
    
    # Must match within numerical tolerance
    np.testing.assert_allclose(result.values, expected.values, 
                               rtol=1e-6, atol=1e-9)
```

---

## Integration Testing

### Cross-Component Integration

**Location:** `tests/`, `tests/multibackend_integration/`

Integration tests verify that multiple components work together correctly:

#### DataAccess + Execution Integration

```python
# tests/test_data_access_execution_integration.py
def test_factor_computation_with_real_data_source(tmp_path):
    """End-to-end: data source → factor compute → materialize."""
    from data_access.lakehouse import setup_test_lakehouse
    from engine.factor_engine import FactorEngine
    from storage.materialize.materializer import ParquetMaterializer
    
    # Setup test data
    lakehouse = setup_test_lakehouse(tmp_path)
    lakehouse.write_daily_data("close", test_close_data())
    
    # Compute factor
    engine = FactorEngine(data_source=lakehouse)
    result = engine.compute_factor("ts_mean(close, 20)")
    
    # Materialize
    materializer = ParquetMaterializer(tmp_path / "factors")
    materializer.write_factor("ma20", result)
    
    # Round-trip verify
    reloaded = materializer.read_factor("ma20")
    pd.testing.assert_series_equal(result, reloaded)
```

#### Multi-Backend Cost Router Integration

```python
# tests/multibackend_integration/test_cost_model.py
def test_backend_selection_respects_cost_model():
    """Backend router should choose optimal backend based on data scale."""
    from planning.backend_selector import select_optimal_backend_for_node
    from planning.backend_region import PhysicalBackend
    
    # Small data → Pandas
    decision_small = select_optimal_backend_for_node(
        estimated_rows=1000,
        estimated_columns=10,
        estimated_bytes=80_000,
        available_memory_bytes=8 * 1024**3,
        operator_names=["ts_mean", "ts_std"]
    )
    assert decision_small.chosen_backend == PhysicalBackend.PANDAS_NUMPY
    
    # Large data → DuckDB
    decision_large = select_optimal_backend_for_node(
        estimated_rows=10_000_000,
        estimated_columns=100,
        estimated_bytes=8 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        operator_names=["ts_mean", "ts_std"]
    )
    assert decision_large.chosen_backend == PhysicalBackend.DUCKDB_SQL
```

---

## Regression Testing Matrix

### Full Regression Suite

The complete regression suite includes **1195+ tests** across all modules:

```bash
# Run full regression
pytest tests/ -v --tb=short --maxfail=5

# Current baseline (2026-08-13):
# - 1195 passed
# - 4 failed (pre-existing, documented in CLAUDE.md memory)
```

### Regression Test Matrix

| Test Category | Test Count | Run Time | Frequency |
|---------------|------------|----------|-----------|
| Operator Units | 450+ | 120s | Every commit |
| Backend Parity | 320+ | 180s | Every commit |
| DataAccess | 180+ | 90s | Every commit |
| Integration | 150+ | 240s | Pre-merge |
| Hard Gates | 85 | 60s | Pre-merge |
| Performance | 40+ | 300s | Nightly |
| **Total** | **1195+** | **~15min** | - |

### Critical Regression Paths

#### 1. Operator Semantic Stability

```python
# tests/test_operator_semantic_regression.py
def test_operator_hash_stability():
    """Operator semantic hash must remain stable across refactors."""
    from runtime.factor_identity import scoped_operator_contract_hash
    from planner.logical_plan import PlanNode
    
    # Known stable hashes (from R32 baseline)
    STABLE_HASHES = {
        "ts_mean": "7a8f3c2e1b9d",
        "cs_rank": "4b2e7f8a1c3d",
        "ts_regression": "9d1c4f7e2a8b"
    }
    
    for op_name, expected_hash in STABLE_HASHES.items():
        plan = make_plan(op_name)
        actual_hash = scoped_operator_contract_hash(plan)[:12]
        assert actual_hash == expected_hash, \
            f"{op_name} hash changed! This invalidates all cached factors."
```

#### 2. Backend Parity Regression

```python
# tests/backend/test_backend_parity_regression.py
@pytest.mark.parametrize("operator", CORE_OPERATORS)
def test_three_backend_parity(operator):
    """Core operators must produce identical results across all backends."""
    data = generate_test_panel()
    
    result_pandas = execute_with_backend(operator, data, "pandas_numpy")
    result_polars = execute_with_backend(operator, data, "polars")
    result_duckdb = execute_with_backend(operator, data, "duckdb_sql")
    
    # Bitwise identical (except NaN placement)
    np.testing.assert_array_equal(result_pandas.values, result_polars.values)
    np.testing.assert_array_equal(result_pandas.values, result_duckdb.values)
```

---

## Performance Testing

### Performance Regression Gates

**Location:** `tests/perf/`, `benchmarks/`

Performance tests ensure computational efficiency doesn't regress:

#### Throughput Benchmarks

```python
# tests/perf/test_regression_gate.py
def test_batch_throughput_regression():
    """Batch execution must maintain throughput SLA."""
    from benchmarks.backend_operator_bench import run_throughput_benchmark
    
    # Baseline: 129.3s for 300 factors × 252 days × 100 instruments (R39)
    # Target: <130s (no regression) or <100s (R39 target achieved: 96.3s)
    
    result = run_throughput_benchmark(
        num_factors=300,
        num_days=252,
        num_instruments=100,
        backend="polars"
    )
    
    assert result.total_time_seconds < 130.0, \
        f"Throughput regression: {result.total_time_seconds:.1f}s > 130s baseline"
```

#### Memory Regression Gates

```python
# tests/perf/test_memory_regression.py
def test_memory_footprint_regression():
    """Memory usage must not exceed calibrated bounds."""
    import psutil
    
    process = psutil.Process()
    baseline_mb = process.memory_info().rss / 1024**2
    
    # Large factor DAG execution
    result = compute_large_dag(num_factors=500)
    
    peak_mb = process.memory_info().rss / 1024**2
    delta_mb = peak_mb - baseline_mb
    
    # R36 calibration: 500 factors should use <4GB peak
    assert delta_mb < 4096, \
        f"Memory regression: {delta_mb:.0f}MB > 4GB budget"
```

### Benchmark Suite

```bash
# Run performance benchmarks
python benchmarks/backend_operator_bench.py --backend=all

# Sample output:
# Backend: pandas_numpy
#   ts_mean(close, 20): 0.52ms/Krows (1.92M rows/sec)
#   ts_std(close, 20): 0.68ms/Krows (1.47M rows/sec)
# Backend: polars_eager
#   ts_mean(close, 20): 0.21ms/Krows (4.76M rows/sec)
#   ts_std(close, 20): 0.29ms/Krows (3.45M rows/sec)
# Backend: duckdb_sql
#   ts_mean(close, 20): 0.15ms/Krows (6.67M rows/sec)
#   ts_std(close, 20): 0.19ms/Krows (5.26M rows/sec)
```

---

## Hard Gates & Certification

### Hard Gate Philosophy

**Hard gates** are binary pass/fail checks that must all pass before production deployment. They verify:

- **Correctness**: Mathematical accuracy, semantic preservation
- **Safety**: Data integrity, concurrency safety, memory bounds
- **Performance**: Throughput SLAs, latency budgets
- **Completeness**: Coverage, documentation, evidence

### Running Hard Gates

```bash
# Run all hard gates for a specific round
python scripts/audit_r32_hard_gates.py

# Output format:
# {
#   "R32_CALENDAR_OUT_OF_COVERAGE_FAILS": true,
#   "R32_CSE_ORPHAN_SHARED_ZERO": true,
#   "R32_CATALOG_SCHEMA_VERSIONED": true,
#   ...
#   "R32_HARD_BLOCKERS_ZERO": true  ← Overall pass/fail
# }
```

### Hard Gate Categories

See **[HARD_GATES_REFERENCE.md](HARD_GATES_REFERENCE.md)** for complete gate documentation.

---

## Backend Parity Testing

### Three-Backend Consistency

The Factor Engine supports three backends:
- **Pandas/Numpy**: Reference implementation
- **Polars**: High-performance parallel execution
- **DuckDB SQL**: Vectorized analytical queries

All three must produce **bitwise-identical results** (modulo NaN ordering).

### Parity Test Structure

```python
# tests/multibackend_integration/test_backend_parity.py
PARITY_OPERATORS = [
    "ts_mean", "ts_std", "ts_rank", "ts_delta",
    "cs_rank", "cs_zscore", "cs_demean",
    # ... 620+ operators certified for parity
]

@pytest.mark.parametrize("operator", PARITY_OPERATORS)
@pytest.mark.parametrize("backend", ["pandas_numpy", "polars", "duckdb_sql"])
def test_operator_backend_parity(operator, backend):
    """All certified operators produce identical results across backends."""
    data = load_test_panel("standard_100x50")  # 100 days × 50 instruments
    
    # Execute on target backend
    result = execute_operator(operator, data, backend=backend)
    
    # Compare to pandas reference (golden)
    reference = execute_operator(operator, data, backend="pandas_numpy")
    
    assert_parity(result, reference, operator=operator, backend=backend)

def assert_parity(result, reference, operator, backend):
    """Assert bitwise parity with detailed failure reporting."""
    if not np.array_equal(result.values, reference.values, equal_nan=True):
        # Detailed diagnosis
        diff = ~np.isclose(result.values, reference.values, equal_nan=True)
        diff_count = diff.sum()
        max_diff = np.abs((result - reference)[diff]).max()
        
        raise AssertionError(
            f"Backend parity failure:\n"
            f"  Operator: {operator}\n"
            f"  Backend: {backend}\n"
            f"  Differences: {diff_count}/{len(result)} values\n"
            f"  Max diff: {max_diff}\n"
            f"  Sample diffs:\n{(result - reference)[diff].head()}"
        )
```

### Parity Coverage

```bash
# Check backend parity coverage
python scripts/audit_backend_parity.py

# Current status (2026-08-13):
# - Pandas: 1430 operators (reference)
# - Polars: 620 operators certified (43%)
# - DuckDB: 269 operators certified (19%)
# - Parity divergences: 0 (all certified ops match)
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/operators/test_ts_operators.py

# Run specific test
pytest tests/operators/test_ts_operators.py::test_ts_mean_basic

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Selection

```bash
# Run only unit tests
pytest tests/operators/

# Run only integration tests
pytest tests/multibackend_integration/

# Run specific round
pytest tests/r39/

# Run by marker
pytest tests/ -m "not slow"
```

### Debugging Failed Tests

```bash
# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Drop into debugger on failure
pytest tests/ --pdb

# Run last failed tests
pytest tests/ --lf
```

### Performance Profiling

```bash
# Profile test execution time
pytest tests/ --durations=10

# Memory profiling
pytest tests/ --memprof
```

### Parallel Execution (Not Recommended)

```bash
# Serial execution (recommended for determinism)
pytest tests/ -n 1

# Parallel execution (use only for quick checks)
pytest tests/ -n auto  # WARNING: May cause false failures
```

**Note**: The Factor Engine test suite is designed for **serial execution** to ensure:
- Deterministic results (no race conditions)
- Controlled memory usage (small test machines)
- Accurate performance measurements

---

## Best Practices

### Writing New Tests

1. **Start with the simplest case**: Verify basic functionality first
2. **Test edge cases systematically**: NaN, Inf, empty, single-value
3. **Use golden references**: Compare against known-good implementations
4. **Make assertions specific**: Don't just check "no error", verify correctness
5. **Document expected behavior**: Explain what the test validates and why

### Test Data Generation

```python
# Good: Deterministic, reproducible test data
def generate_test_panel(seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=100)
    instruments = [f"STOCK_{i:03d}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, instruments])
    return pd.Series(np.random.randn(5000), index=idx)

# Bad: Non-deterministic test data
def generate_test_panel_bad():
    # No seed! Results vary across runs
    return pd.Series(np.random.randn(5000))
```

### Assertion Patterns

```python
# Good: Specific, informative assertions
assert result.shape == (5000,), f"Expected 5000 rows, got {result.shape[0]}"
assert not result.isna().all(), "Result should not be all NaN"
np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-9)

# Bad: Generic assertions
assert result is not None  # Too weak
assert len(result) > 0  # Doesn't check correctness
```

---

## Continuous Integration

### Pre-Commit Checks

```bash
# Run before every commit
./scripts/pre_commit_check.sh

# Includes:
# 1. Linting (ruff, black)
# 2. Type checking (mypy)
# 3. Fast unit tests (<60s)
# 4. Critical hard gates
```

### Pre-Merge Requirements

- ✅ All unit tests pass (1195+)
- ✅ All hard gates pass (85)
- ✅ Backend parity maintained (0 divergences)
- ✅ No performance regressions (>5%)
- ✅ Code coverage ≥85%

### Nightly Builds

- Full regression suite
- Extended performance benchmarks
- Memory leak detection
- Cross-platform validation (Linux, macOS)

---

## Troubleshooting

### Common Test Failures

#### "Registry frozen" errors

```python
# Symptom: RuntimeError: Cannot register operator, registry is frozen

# Solution: Use the _ensure_registry_writable_for_tests fixture (auto-enabled)
# Or manually thaw in test:
from cleaned_operators.registry import OperatorRegistry, _BOOTSTRAP_TOKEN
OperatorRegistry.thaw_for_bootstrap(_BOOTSTRAP_TOKEN)
```

#### Backend parity failures

```python
# Symptom: Results differ across backends

# Diagnosis:
# 1. Check for floating-point precision issues (use np.isclose)
# 2. Verify NaN handling (pandas vs polars differ in NaN propagation)
# 3. Check for missing backend certification

# Solution: See BACKEND_SELECTION_GUIDE.md for backend-specific behavior
```

#### Performance regression false positives

```bash
# Symptom: Performance tests fail intermittently

# Causes:
# - System load (other processes competing for CPU)
# - Thermal throttling
# - First-run compilation (Numba, PyO3)

# Solution: Run benchmarks multiple times, use median
pytest tests/perf/ --benchmark-warmup=on --benchmark-min-rounds=5
```

---

## References

- [HARD_GATES_REFERENCE.md](HARD_GATES_REFERENCE.md) - Complete hard gate documentation
- [BACKEND_SELECTION_GUIDE.md](BACKEND_SELECTION_GUIDE.md) - Backend selection and parity
- [COST_MODEL_EXPLAINED.md](COST_MODEL_EXPLAINED.md) - Performance cost model details
- `tests/conftest.py` - Global pytest configuration
- `scripts/audit_*_hard_gates.py` - Hard gate audit scripts
