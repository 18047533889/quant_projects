# Cost Model Explained

**Version:** 1.0  
**Last Updated:** 2026-08-13  
**Status:** Production

## Overview

The Factor Engine uses a **two-tier cost model** to optimize execution planning: **static priors** based on operator complexity and backend characteristics, with **bounded runtime calibration** from actual measurements. This document explains how the cost model works, how to calibrate it, and how to debug cost-related issues.

## Table of Contents

1. [Cost Model Architecture](#cost-model-architecture)
2. [How Cost Estimation Works](#how-cost-estimation-works)
3. [Static Cost Priors](#static-cost-priors)
4. [Runtime Calibration](#runtime-calibration)
5. [Cost-Based Backend Selection](#cost-based-backend-selection)
6. [Calibrating Costs](#calibrating-costs)
7. [Debugging Cost Decisions](#debugging-cost-decisions)
8. [Performance Tuning](#performance-tuning)

---

## Cost Model Architecture

### Two-Tier Design

```
┌─────────────────────────────────────────────────────────┐
│                    Cost Estimation                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────┐    ┌─────────────────────┐   │
│  │  Static Cost Prior   │    │ Runtime Calibration │   │
│  │  (Operator + Backend)│───▶│  (Bounded ±30%)     │   │
│  │                      │    │                     │   │
│  │ • Complexity class   │    │ • Actual samples    │   │
│  │ • Algorithm family   │    │ • EMA smoothing     │   │
│  │ • Base + coefficients│    │ • Calibration factor│   │
│  └──────────────────────┘    └─────────────────────┘   │
│            │                            │                │
│            └────────────┬───────────────┘                │
│                         ▼                                │
│                ┌──────────────────┐                      │
│                │  Final Estimate  │                      │
│                │  (ms, per task)  │                      │
│                └──────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Static Prior is Primary**: Cost model starts from operator/backend complexity analysis
2. **Bounded Calibration**: Runtime measurements adjust by at most ±30% (prevents drift)
3. **Fail-Safe**: If no prior exists, use conservative generic fallback
4. **Transparent**: Every cost decision includes reasoning (explainability)
5. **Adaptive**: Continuously learns from actual execution (EMA updates)

---

## How Cost Estimation Works

### Cost Components

For a given operator on a backend, total cost has multiple components:

```
Total Cost = Startup + Compute + Transfer + Penalties × Calibration
```

#### 1. Startup Cost

**One-time overhead** per backend invocation:
- Import time (pandas: ~2ms, polars: ~8ms, duckdb: ~15ms)
- Connection pool initialization (SQL backends)
- JIT compilation warmup (Numba kernels)

```python
startup_cost_ms = {
    "pandas_numpy": 2.0,
    "polars_eager": 8.0,
    "polars_lazy": 10.0,
    "duckdb_sql": 15.0
}
```

#### 2. Compute Cost

**Per-row processing time** based on throughput:

```python
compute_ms = (rows / 1_000_000) / throughput_mrows_per_sec * 1000

# Example: 100K rows on polars (2.5M rows/sec)
# compute_ms = (100_000 / 1_000_000) / 2.5 * 1000 = 40ms
```

Throughput varies by backend:
- **Pandas**: 0.5M rows/sec (single-threaded, GIL-limited)
- **Polars eager**: 2.5M rows/sec (parallel, SIMD)
- **Polars lazy**: 3.0M rows/sec (optimized query plan)
- **DuckDB SQL**: 4.0M rows/sec (vectorized, columnar)

#### 3. Transfer Cost

**Data movement** between backends:

```python
transfer_ms = conversion_penalty + (data_size_gb * 50.0)

# Example: 100MB polars → duckdb
# transfer_ms = 10ms + (0.1 * 50) = 15ms
```

Conversion penalties:
- Same backend: 0ms
- Pandas ↔ Polars: 8ms + data size
- Pandas/Polars → DuckDB: 20ms + data size (SQL boundary)

#### 4. Penalties

**Adjustments** for suboptimal matches:

```python
# Profile mismatch: operator type doesn't match backend strength
if operator_profile not in backend.preferred_profiles:
    profile_penalty = compute_ms * 0.3  # 30% penalty

# Window size penalty: large windows on non-optimized backends
if avg_window > 100 and backend == "pandas_numpy":
    window_penalty = compute_ms * 0.5  # 50% penalty

# Memory pressure penalty: risk of OOM
if estimated_memory > available * 0.8:
    if backend.supports_streaming:
        memory_penalty = compute_ms * 0.2  # Streaming overhead
    else:
        memory_penalty = 1000.0  # Large penalty, avoid OOM
```

#### 5. Calibration Factor

**Bounded adjustment** from actual measurements:

```python
calibrated_cost = static_estimate * calibration_factor

# calibration_factor ∈ [0.7, 1.3]  # ±30% bound
```

---

## Static Cost Priors

### Prior Structure

Each operator/backend combination has a static prior:

```python
@dataclass(frozen=True)
class StaticCostPrior:
    canonical: str            # Operator name
    backend: str              # Backend identifier
    base_cost_ms: float       # Fixed overhead (setup, kernel dispatch)
    row_coefficient: float    # ms per 1M rows
    window_coefficient: float # Additional cost per window size
    feature_dim_coefficient: float  # For model operators
    
    # Classification metadata
    complexity_class: str     # "low", "medium", "high", "very_high"
    algorithm_family: str     # "rolling", "regression", "filter", etc.
    requires_sort: bool       # Needs sorted input
    requires_full_pass: bool  # Cannot stream (needs all data)
```

### Example Priors

#### Low Complexity: `ts_delta`

```python
StaticCostPrior(
    canonical="ts_delta",
    backend="pandas_numpy",
    base_cost_ms=0.5,         # Simple operation
    row_coefficient=0.2,      # Fast: 5M rows/sec
    window_coefficient=0.0,   # No window dependency
    complexity_class="low",
    algorithm_family="elementwise",
    requires_sort=False,
    requires_full_pass=False
)
```

**Estimate for 100K rows:** 0.5 + (0.1 × 0.2) = **0.52ms**

#### Medium Complexity: `ts_mean`

```python
StaticCostPrior(
    canonical="ts_mean",
    backend="pandas_numpy",
    base_cost_ms=1.0,
    row_coefficient=0.5,      # Moderate: 2M rows/sec
    window_coefficient=0.02,  # Slight window dependency
    complexity_class="medium",
    algorithm_family="rolling",
    requires_sort=True,       # Needs time ordering
    requires_full_pass=False
)
```

**Estimate for 100K rows, window=20:**
- Base: 1.0ms
- Row cost: 0.1 × 0.5 = 0.05ms
- Window cost: 0.02 × (20/20) = 0.02ms
- **Total: 1.07ms**

#### High Complexity: `ts_regression`

```python
StaticCostPrior(
    canonical="ts_regression",
    backend="pandas_numpy",
    base_cost_ms=5.0,         # Expensive setup
    row_coefficient=2.0,      # Slow: 0.5M rows/sec
    window_coefficient=0.1,   # Strong window dependency
    complexity_class="high",
    algorithm_family="regression",
    requires_sort=True,
    requires_full_pass=False
)
```

**Estimate for 100K rows, window=60:**
- Base: 5.0ms
- Row cost: 0.1 × 2.0 = 0.2ms
- Window cost: 0.1 × (60/20) = 0.3ms
- **Total: 5.5ms**

#### Very High Complexity: `kalman_filter`

```python
StaticCostPrior(
    canonical="kalman_filter",
    backend="pandas_numpy",
    base_cost_ms=10.0,        # Complex initialization
    row_coefficient=5.0,      # Very slow: 0.2M rows/sec
    window_coefficient=0.0,   # Stateful, not window-based
    complexity_class="very_high",
    algorithm_family="state_space_filter",
    requires_sort=True,
    requires_full_pass=True   # Needs complete history
)
```

**Estimate for 100K rows:** 10.0 + (0.1 × 5.0) = **10.5ms**

### Backend Multipliers

Same operator has different costs on different backends:

```python
# ts_mean on different backends (100K rows, window=20)

# Pandas: 1.0 + 0.1×0.5 + 0.02×1 = 1.07ms
# Polars: 0.8 + 0.1×0.2 + 0.01×1 = 0.83ms  (2.4× faster)
# DuckDB: 0.6 + 0.1×0.15 + 0.005×1 = 0.62ms (1.7× faster)
```

### Storing Static Priors

Priors are stored in `benchmarks/static_cost_priors.json`:

```json
{
  "priors": [
    {
      "canonical": "ts_mean",
      "backend": "pandas_numpy",
      "base_cost_ms": 1.0,
      "row_coefficient": 0.5,
      "window_coefficient": 0.02,
      "complexity_class": "medium",
      "algorithm_family": "rolling",
      "requires_sort": true,
      "requires_full_pass": false
    },
    ...
  ]
}
```

---

## Runtime Calibration

### Calibration Philosophy

**Problem**: Static priors are estimates based on typical workloads, but actual performance varies by:
- Hardware (CPU model, cache size, NUMA topology)
- Data characteristics (sparsity, entropy, cache locality)
- System load (competing processes, thermal throttling)

**Solution**: Learn from actual measurements, but **bound the adjustment** to prevent drift.

### Calibration Structure

```python
@dataclass
class RuntimeCalibration:
    canonical: str
    backend: str
    static_estimate_ms: float
    actual_ms: float
    sample_count: int
    calibration_factor: float  # ∈ [0.7, 1.3]
    
    CALIBRATION_LOWER_BOUND = 0.7  # Most 30% faster than prior
    CALIBRATION_UPPER_BOUND = 1.3  # Most 30% slower than prior
```

### Update Algorithm

**Exponential Moving Average (EMA)** with bounding:

```python
def update_calibration(existing, new_actual, new_static_estimate):
    # Compute factor from this sample
    new_factor = new_actual / new_static_estimate if new_static_estimate > 0 else 1.0
    
    # Bound the sample factor
    new_factor = max(0.7, min(1.3, new_factor))
    
    # EMA update (alpha=0.1, weights recent samples more)
    updated_factor = 0.1 * new_factor + 0.9 * existing.calibration_factor
    
    # Bound the smoothed factor
    updated_factor = max(0.7, min(1.3, updated_factor))
    
    return RuntimeCalibration(
        canonical=existing.canonical,
        backend=existing.backend,
        static_estimate_ms=new_static_estimate,
        actual_ms=new_actual,
        sample_count=existing.sample_count + 1,
        calibration_factor=updated_factor
    )
```

### Example Calibration Evolution

```python
# Initial: no calibration, factor = 1.0
static_estimate = 10.0ms
calibrated = 10.0ms × 1.0 = 10.0ms

# Sample 1: actual = 12ms (20% slower)
sample_factor = 12 / 10 = 1.2  (within [0.7, 1.3])
new_factor = 0.1 × 1.2 + 0.9 × 1.0 = 1.02
calibrated = 10.0ms × 1.02 = 10.2ms

# Sample 2: actual = 13ms (30% slower)
sample_factor = 13 / 10 = 1.3  (at upper bound)
new_factor = 0.1 × 1.3 + 0.9 × 1.02 = 1.048
calibrated = 10.0ms × 1.048 = 10.48ms

# Sample 3: actual = 8ms (20% faster)
sample_factor = 8 / 10 = 0.8  (within [0.7, 1.3])
new_factor = 0.1 × 0.8 + 0.9 × 1.048 = 1.023
calibrated = 10.0ms × 1.023 = 10.23ms

# Converges to average behavior, bounded to ±30%
```

### Why Bounded Calibration?

**Without bounds**, calibration can drift arbitrarily:

```python
# Pathological case: outlier samples
# Sample 1: actual = 100ms (10× slower, system was thrashing)
# → factor becomes 10.0, all future estimates 10× too high
# → Backend never selected again, even after system recovers

# With bounds:
# Sample 1: actual = 100ms → clamped factor = 1.3
# → At most 30% increase, outlier can't dominate
```

**Prevents**:
- Thermal throttling outliers from permanent impact
- First-run compilation overhead from skewing estimates
- Noisy measurements from corrupting the model

---

## Cost-Based Backend Selection

### Selection Algorithm

For each factor computation node, the planner:

1. **Identifies candidate backends** (based on data scale, operator profile)
2. **Estimates cost** for each candidate (static + calibration + penalties)
3. **Selects minimum-cost backend**
4. **Records decision reasoning** (for debuggability)

### Example Selection Decision

```python
from planning.backend_selector import select_optimal_backend_for_node

decision = select_optimal_backend_for_node(
    estimated_rows=100_000,
    estimated_columns=50,
    estimated_bytes=40_000_000,  # ~40MB
    available_memory_bytes=8 * 1024**3,
    operator_names=["ts_mean", "ts_std", "ts_rank"],
    window_params=[20, 20, None],
    parent_backend=None,
    performance_priority="balanced"
)

# Output:
# RoutingDecision(
#     chosen_backend=PhysicalBackend.POLARS_EAGER,
#     estimated_cost_ms=15.2,
#     memory_footprint_bytes=60_000_000,
#     confidence_score=0.87,
#     reasoning=[
#         "Data scale: medium (100,000 rows)",
#         "Memory pressure: low (40.0MB / 8192.0MB)",
#         "Operator profile: window_heavy",
#         "Candidates: ['POLARS_EAGER', 'PANDAS_NUMPY', 'DUCKDB_SQL']",
#         "Polars selected: 15.2ms < pandas 28.5ms < duckdb 25.1ms"
#     ],
#     alternatives=[
#         (PhysicalBackend.DUCKDB_SQL, 25.1, "cost=25.1ms, memory=44.0MB"),
#         (PhysicalBackend.PANDAS_NUMPY, 28.5, "cost=28.5ms, memory=48.0MB")
#     ]
# )
```

### Selection Strategy by Data Scale

| Rows | Scale | Preferred Backend | Reasoning |
|------|-------|-------------------|-----------|
| <1K | Tiny | Pandas | Startup dominates, pandas fastest start |
| 1K-10K | Small | Pandas | Startup still significant |
| 10K-100K | Medium | Polars | Throughput starts to matter |
| 100K-1M | Large | Polars/DuckDB | Throughput dominates |
| 1M-10M | Huge | DuckDB | Vectorization wins |
| >10M | Massive | DuckDB | Must stream |

### Selection Strategy by Operator Profile

| Profile | Preferred Backend | Reasoning |
|---------|-------------------|-----------|
| Window-heavy | DuckDB > Polars > Pandas | SQL optimized for window functions |
| Cross-section | Polars > DuckDB > Pandas | Parallel group-by operations |
| Elementwise | Pandas ≈ Polars | Simple operations, overhead matters |
| Regression | Pandas | Statsmodels integration |
| Mixed | Cost-based | Evaluate each candidate |

### Transfer Affinity

**Avoid unnecessary backend switches**:

```python
# Parent node used polars
parent_backend = PhysicalBackend.POLARS_EAGER

# Child node candidates: polars (cost=10ms), duckdb (cost=9ms)
# But duckdb requires transfer: 9 + 5 (conversion) = 14ms
# → Choose polars (10ms) despite higher compute cost
```

### Memory Pressure Override

**Critical memory pressure forces streaming backends**:

```python
if memory_pressure == MemoryPressure.CRITICAL:
    # Filter out non-streaming backends
    candidates = [b for b in candidates if b.supports_streaming]
    # Only polars_lazy and duckdb_sql remain
```

---

## Calibrating Costs

### When to Calibrate

Calibrate when:
- **New hardware deployment**: Different CPU, memory bandwidth
- **Workload changes**: Different data characteristics (sparsity, distribution)
- **Backend upgrades**: Polars 0.20 → 1.0, DuckDB 0.9 → 1.0
- **Periodic refresh**: Monthly recalibration recommended

### Calibration Methods

#### Method 1: Benchmark Suite

```bash
# Run full operator benchmark suite
python benchmarks/backend_operator_bench.py --backend=all --output=costs.json

# Generates calibrated priors from actual measurements
# Output: benchmarks/calibrated_cost_priors.json
```

**Benchmark output:**
```
Backend: pandas_numpy
  ts_mean(close, 20): 0.52ms/Krows (1.92M rows/sec) [500 samples]
  ts_std(close, 20): 0.68ms/Krows (1.47M rows/sec) [500 samples]
  ts_regression(close, high, 60): 2.15ms/Krows (0.47M rows/sec) [200 samples]

Backend: polars_eager
  ts_mean(close, 20): 0.21ms/Krows (4.76M rows/sec) [500 samples]
  ts_std(close, 20): 0.29ms/Krows (3.45M rows/sec) [500 samples]

Backend: duckdb_sql
  ts_mean(close, 20): 0.15ms/Krows (6.67M rows/sec) [500 samples]
  ts_std(close, 20): 0.19ms/Krows (5.26M rows/sec) [500 samples]
```

#### Method 2: Production Sampling

```python
# Enable production cost recording
from backend.cost_calibration import get_cost_model_store

store = get_cost_model_store()

# After each factor computation
store.record_actual(
    canonical="ts_mean",
    backend="polars_eager",
    actual_ms=12.5,
    rows=100_000,
    window=20
)

# Calibration updates automatically (EMA)
```

#### Method 3: Targeted Calibration Script

```bash
# Calibrate specific operators
python scripts/calibrate_backend_costs.py \
    --operators ts_mean,ts_std,cs_rank \
    --backends pandas_numpy,polars_eager \
    --data-scales small,medium,large \
    --samples 100

# Output: Updated static_cost_priors.json
```

### Calibration Best Practices

1. **Warmup runs**: Discard first 10 samples (JIT compilation, cache warmup)
2. **Multiple scales**: Calibrate on small, medium, large datasets
3. **Realistic data**: Use production-like data (sparsity, distribution)
4. **Stable environment**: No competing workloads during calibration
5. **Statistical rigor**: Use median (not mean) to avoid outliers

### Example Calibration Session

```python
import numpy as np
from benchmarks.backend_operator_bench import calibrate_operator_cost

# Generate test data
data = generate_test_panel(rows=100_000, instruments=50)

# Warm up (discard)
for _ in range(10):
    execute_operator("ts_mean", data, window=20, backend="polars_eager")

# Measure
samples = []
for _ in range(100):
    start = time.perf_counter()
    execute_operator("ts_mean", data, window=20, backend="polars_eager")
    elapsed_ms = (time.perf_counter() - start) * 1000
    samples.append(elapsed_ms)

# Compute robust statistics
median_ms = np.median(samples)
p25_ms = np.percentile(samples, 25)
p75_ms = np.percentile(samples, 75)
iqr_ms = p75_ms - p25_ms

print(f"ts_mean (polars, 100K rows):")
print(f"  Median: {median_ms:.2f}ms")
print(f"  IQR: [{p25_ms:.2f}, {p75_ms:.2f}]ms")
print(f"  Variability: {iqr_ms/median_ms:.1%}")

# Update prior
throughput = 100_000 / median_ms  # rows/ms
mrows_per_sec = throughput / 1000
print(f"  Throughput: {mrows_per_sec:.2f}M rows/sec")

# Recommended prior: row_coefficient = 1.0 / (mrows_per_sec * 1000)
row_coef = 1.0 / mrows_per_sec
print(f"  Recommended row_coefficient: {row_coef:.3f}")
```

---

## Debugging Cost Decisions

### Enable Cost Decision Logging

```python
import logging
logging.getLogger("planning.backend_selector").setLevel(logging.DEBUG)

# Now all cost decisions are logged
```

**Log output:**
```
[DEBUG] Backend selection for node_123:
  Data: 50000 rows, 20 columns, 4.0MB
  Operators: ['ts_mean', 'ts_std']
  Candidates: pandas_numpy, polars_eager
  
  pandas_numpy:
    startup: 2.0ms
    compute: 25.0ms (50K rows @ 2.0M rows/sec)
    transfer: 0.0ms (no parent)
    penalties: 0.0ms
    calibration: 1.05×
    TOTAL: 28.4ms
  
  polars_eager:
    startup: 8.0ms
    compute: 10.0ms (50K rows @ 5.0M rows/sec)
    transfer: 0.0ms (no parent)
    penalties: 0.0ms
    calibration: 0.98×
    TOTAL: 17.6ms
  
  SELECTED: polars_eager (17.6ms < 28.4ms, confidence=0.89)
```

### Visualize Cost Breakdown

```python
from planning.backend_selector import select_optimal_backend_for_node
import matplotlib.pyplot as plt

decision = select_optimal_backend_for_node(...)

# Extract cost components (would need to expose these)
backends = ["pandas", "polars", "duckdb"]
startup = [2.0, 8.0, 15.0]
compute = [25.0, 10.0, 8.0]
transfer = [0.0, 0.0, 5.0]
penalties = [5.0, 0.0, 0.0]

# Stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bottoms = np.zeros(3)
for component, values in [("Startup", startup), ("Compute", compute), 
                           ("Transfer", transfer), ("Penalties", penalties)]:
    ax.bar(backends, values, bottom=bottoms, label=component)
    bottoms += values

ax.set_ylabel("Cost (ms)")
ax.set_title("Cost Breakdown by Backend")
ax.legend()
plt.show()
```

### Trace Execution Costs

```python
from runtime.profiling import ExecutionProfiler

profiler = ExecutionProfiler()

with profiler.trace("factor_computation"):
    result = engine.compute_factor("ts_mean(close, 20)")

# Print actual costs
print(profiler.summary())

# Output:
# ┌─────────────────┬──────────┬──────────┬───────────┐
# │ Operator        │ Backend  │ Estimate │ Actual    │
# ├─────────────────┼──────────┼──────────┼───────────┤
# │ column(close)   │ pandas   │ 1.0ms    │ 1.2ms     │
# │ ts_mean(...,20) │ polars   │ 17.6ms   │ 19.1ms    │
# │ finalize        │ pandas   │ 2.0ms    │ 1.8ms     │
# ├─────────────────┼──────────┼──────────┼───────────┤
# │ TOTAL           │          │ 20.6ms   │ 22.1ms    │
# └─────────────────┴──────────┴──────────┴───────────┘
#
# Estimate accuracy: 93% (well calibrated)
```

### Compare Estimate vs Actual

```python
from backend.cost_calibration import get_cost_model_store

store = get_cost_model_store()

# Get calibration for operator
calib = store.get_calibration("ts_mean", "polars_eager")

print(f"ts_mean on polars_eager:")
print(f"  Static estimate: {calib.static_estimate_ms:.2f}ms")
print(f"  Actual (last): {calib.actual_ms:.2f}ms")
print(f"  Calibration factor: {calib.calibration_factor:.3f}")
print(f"  Sample count: {calib.sample_count}")
print(f"  Error: {abs(calib.actual_ms / calib.static_estimate_ms - 1):.1%}")
```

### Common Cost Misestimations

#### Problem: Startup cost dominates (small data)

**Symptom:** Heavy backend (DuckDB) selected for tiny dataset

**Diagnosis:**
```python
# 1K rows, DuckDB selected
# startup=15ms, compute=0.3ms, total=15.3ms
# vs pandas: startup=2ms, compute=2ms, total=4ms
```

**Fix:** Increase small-data penalty for heavy backends:
```python
if scale == DataScale.TINY and backend in (DUCKDB, POLARS_LAZY):
    penalty += 10.0  # Discourage heavy backends for tiny data
```

#### Problem: Transfer cost ignored

**Symptom:** Frequent backend switches despite small cost savings

**Diagnosis:**
```python
# Parent: polars (cost 10ms)
# Child: duckdb (cost 9ms) → selected
# But transfer: 5ms → actual total 14ms > 10ms
```

**Fix:** Already implemented (transfer_ms accounts for parent mismatch)

#### Problem: Calibration stuck at wrong value

**Symptom:** Calibration factor = 1.3, but recent samples are faster

**Diagnosis:**
```python
# Old samples (system under load): 20ms, 22ms, 25ms → factor climbed to 1.3
# New samples (system idle): 15ms, 16ms, 14ms → EMA slow to adapt
```

**Fix:** Increase EMA alpha (faster adaptation):
```python
# Current: alpha=0.1 (slow adaptation)
# Change to: alpha=0.3 (faster adaptation)
calibration_factor = 0.3 * new_factor + 0.7 * old_factor
```

Or reset calibration:
```python
store._runtime_calibrations.clear()  # Start fresh
```

---

## Performance Tuning

### Optimization Strategy

```
1. Profile current performance (where is time spent?)
2. Identify bottleneck operators (which operators are slow?)
3. Check cost model accuracy (are estimates close to actual?)
4. Optimize hot paths (improve slow operators or switch backends)
5. Re-calibrate (update cost model with new baselines)
```

### Profiling Current Performance

```python
from runtime.profiling import ExecutionProfiler

profiler = ExecutionProfiler(enable_detailed=True)

with profiler.trace("batch_factors"):
    results = engine.compute_factors(factor_list)

# Identify bottlenecks
report = profiler.hottest_operators(top_k=10)

# Output:
# ┌─────────────────────┬──────────┬───────────┬──────────┐
# │ Operator            │ Count    │ Total (s) │ Avg (ms) │
# ├─────────────────────┼──────────┼───────────┼──────────┤
# │ ts_regression       │ 150      │ 45.2      │ 301.3    │  ← 60% of time
# │ kalman_filter       │ 50       │ 18.7      │ 374.0    │  ← 25% of time
# │ ts_mean             │ 500      │ 6.5       │ 13.0     │  ← 9% of time
# │ cs_rank             │ 300      │ 3.2       │ 10.7     │
# └─────────────────────┴──────────┴───────────┴──────────┘
```

### Optimization Techniques

#### 1. Backend Switching

```python
# Before: ts_regression on pandas (301ms average)
# Try: switch to polars or use Numba-optimized kernel

# Check if operator has faster backend
from backend.operator_capability import get_best_backend

operator, backend = get_best_backend("ts_regression", mode="production")
print(f"Best backend for ts_regression: {backend}")

# If polars available, force it
result = compute_with_backend("ts_regression", data, backend="polars")
```

#### 2. Operator Fusion

```python
# Before: ts_mean → ts_std (two passes)
# After: ts_mean_std (single pass, compute both)

# Define fused operator
@register_operator("ts_mean_std")
def ts_mean_std(series, window):
    rolling = series.rolling(window)
    mean = rolling.mean()
    std = rolling.std()
    return mean, std

# Cost: 1.2× single operator (not 2×)
```

#### 3. Materialization Strategy

```python
# Expensive intermediate result used multiple times
intermediate = compute_factor("kalman_filter(close)")  # 300ms

# Materialize to avoid recomputation
materializer.write_factor("_temp_kalman", intermediate)

# Use materialized version
result1 = compute_factor("ts_mean(_temp_kalman, 20)")  # 10ms
result2 = compute_factor("ts_std(_temp_kalman, 20)")   # 10ms
# vs recompute: 300ms + 10ms + 300ms + 10ms = 620ms
```

#### 4. Batch Processing

```python
# Before: compute factors one-by-one (serial)
for factor in factor_list:
    result = engine.compute_factor(factor)  # Each has overhead

# After: batch compute (amortize overhead)
results = engine.compute_factors_batch(factor_list)

# Savings: shared data loading, CSE across factors, backend reuse
# Speedup: 2-5× depending on factor overlap
```

#### 5. Streaming for Large Data

```python
# Before: load all data into memory (OOM for >10M rows)
data = load_data(start_date, end_date)  # 50GB
result = compute_factor(data)

# After: streaming computation
result = compute_factor_streaming(
    start_date=start_date,
    end_date=end_date,
    chunk_size=1_000_000  # Process 1M rows at a time
)

# Memory: 50GB → 5GB (10× reduction)
# Time: Similar (slight overhead for chunking)
```

### Performance Tuning Checklist

- [ ] Profile to identify bottlenecks (use ExecutionProfiler)
- [ ] Check cost model accuracy (estimate vs actual within 20%)
- [ ] Try faster backends for hot operators
- [ ] Materialize expensive common subexpressions
- [ ] Use batch API for multiple factors
- [ ] Enable streaming for large datasets
- [ ] Calibrate cost model on production hardware
- [ ] Monitor for performance regressions (set SLAs)

### Setting Performance SLAs

```python
# Define SLAs for critical paths
PERFORMANCE_SLAS = {
    "single_factor_compute": 100.0,  # ms, p50
    "batch_100_factors": 5000.0,     # ms, p50
    "large_dag_500_factors": 30000.0, # ms, p95
}

# Test against SLA
actual_p50 = benchmark_single_factor_compute()
assert actual_p50 < PERFORMANCE_SLAS["single_factor_compute"], \
    f"Performance regression: {actual_p50}ms > {PERFORMANCE_SLAS['single_factor_compute']}ms"
```

---

## References

- [BACKEND_SELECTION_GUIDE.md](BACKEND_SELECTION_GUIDE.md) - Backend comparison and selection
- [TESTING_STRATEGY.md](TESTING_STRATEGY.md) - Performance regression testing
- [HARD_GATES_REFERENCE.md](HARD_GATES_REFERENCE.md) - Performance hard gates
- `backend/cost_calibration.py` - Cost model implementation
- `planning/backend_selector.py` - Backend selection logic
- `benchmarks/backend_operator_bench.py` - Calibration benchmarks
