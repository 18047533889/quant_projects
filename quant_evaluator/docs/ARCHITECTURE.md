# QuantEvaluator Architecture

**Version:** 0.0.1a1  
**Last Updated:** 2026-08-14

## Overview

QuantEvaluator is a batch-first evidence generation engine that computes metrics over factor values and forward labels. It maintains strict boundaries: no data fetching, no factor computation, no admission decisions.

## Design Principles

### 1. Batch-First API

All operations designed for `(T, N, F)` tensors:
- **T**: Time periods
- **N**: Assets (cross-section)
- **F**: Factors

Single-factor operations are thin wrappers that add a dimension.

**Rationale:** Performance-critical path is evaluating hundreds of factors simultaneously. Batch operations leverage vectorization and avoid Python loops.

### 2. Fail-Closed

No implicit behavior:
- No automatic label shifting or inference
- No forward-filling of missing data
- No silent type coercion
- Raise errors on contract violations

**Rationale:** In quantitative research, silent data leakage (future information) can invalidate years of work. Explicit contracts prevent subtle bugs.

### 3. Reference-First

Clear semantics before optimization:
- Reference implementations in pure NumPy
- Explicit algorithms with citations
- Golden value test fixtures
- Fast kernels added only with parity tests

**Rationale:** Correctness over speed. Fast implementations must match reference behavior exactly.

### 4. Typed Contracts

All inputs/outputs use frozen dataclasses:
- `FactorBatch`: Factor values with explicit axes
- `LabelBundle`: Forward labels with strict timing
- `EvaluationBundle`: Versioned metrics and diagnostics

**Rationale:** Type safety catches errors at the boundary. Immutability prevents mutation bugs.

### 5. Independent Core

Core functionality requires only:
- `numpy`
- `scipy` (optional, for rank correlation)
- Standard library

No dependencies on:
- DataAccess (data fetching)
- FactorEngine (computation)
- FactorAssets (governance)

**Rationale:** Enables independent testing, faster CI, and clear ownership boundaries.

## Module Organization

```
quant_evaluator/
├── __init__.py              # Public API surface
├── contracts/               # Core data contracts
│   ├── factor_batch.py     # FactorBatch, AxisRef
│   ├── label_bundle.py     # LabelBundle with timing
│   └── errors.py           # Exception taxonomy
├── api/                     # Request/response contracts
│   └── requests.py         # EvaluationRequest, EvaluationBundle
├── metrics/                 # Reference implementations
│   ├── quality.py          # Coverage diagnostics
│   ├── ic.py               # Information Coefficient
│   ├── quantile.py         # Quantile-based metrics
│   ├── turnover.py         # Turnover estimation
│   ├── ic_summary.py       # ICIR, t-stat, decay
│   ├── temporal.py         # Autocorrelation, stability
│   ├── exposure.py         # Risk factor exposure
│   ├── robustness.py       # HAC, bootstrap
│   ├── distribution.py     # Skew, kurtosis, outliers
│   ├── portfolio_stats.py  # Sharpe, drawdown, Sortino
│   └── multiple_testing.py # FDR correction
├── runtime/                 # Execution engine
│   ├── evaluator.py        # Main evaluator class
│   ├── intermediates.py    # Caching layer
│   └── budgets.py          # Resource tracking
├── planner/                 # Batch planning
│   ├── batch_plan.py       # Memory-aware chunking
│   └── dependency_plan.py  # Metric dependency DAG
├── registry/                # Metric registry
│   ├── metrics.py          # Metric catalog
│   └── presets.py          # Preset configurations
├── kernels/                 # Fast implementations
│   ├── fast.py             # Vectorized kernels
│   └── reference_bridge.py # Parity checkers
├── diagnosis/               # Factor diagnostics
│   └── factor.py           # Per-factor health checks
└── adapters/                # Optional integrations
    ├── data_access.py      # DA adapter (stub)
    └── factor_engine.py    # FE adapter (stub)
```

## Core Contracts

### FactorBatch

```python
@dataclass(frozen=True)
class FactorBatch:
    factor_ids: Tuple[str, ...]      # Factor identifiers
    time_axis: AxisRef               # Time dimension
    asset_axis: AxisRef              # Asset dimension
    values: np.ndarray               # Shape: (T, N, F)
    validity: Optional[np.ndarray]   # Shape: (T, N, F), bool mask
    layout: str = "wide"             # Data layout
    dtype: str = "float64"           # Value dtype
    context_refs: Dict[str, Any]     # Context metadata
    value_hash: Optional[str]        # Content hash
```

**Invariants:**
- `values.shape[0] == time_axis.size`
- `values.shape[1] == asset_axis.size`
- `values.shape[2] == len(factor_ids)`
- `validity` if present must match `values.shape`

### LabelBundle

```python
@dataclass(frozen=True)
class LabelBundle:
    target_id: str                   # Label identifier
    values: np.ndarray               # Shape: (T, N)
    horizon: int                     # Forward horizon (> 0)
    execution_delay: int = 0         # Execution delay
    decision_time: Tuple[Any, ...]   # Decision timestamps (required)
    execution_time: Tuple[Any, ...]  # Execution timestamps
    label_start_time: Tuple[Any, ...] # Label start (required)
    label_end_time: Tuple[Any, ...]   # Label end (required)
    validity: Optional[np.ndarray]   # Shape: (T, N)
```

**Timing Invariants:**
- `decision_time[t] < label_start_time[t]` (no look-ahead)
- `label_start_time[t] < label_end_time[t]` (positive horizon)
- `horizon > 0` (must be forward-looking)

### EvaluationBundle

```python
@dataclass(frozen=True)
class EvaluationBundle:
    request_id: str
    factor_ids: Tuple[str, ...]
    label_id: str
    timestamp: str
    schema_version: str = "0.1"
    metric_values: Dict[str, MetricValue]
    diagnostics: Dict[str, FactorDiagnosis]
    grouped_metrics: Optional[Dict[str, Dict[str, MetricValue]]]
    series_refs: Optional[Dict[str, str]]
    metric_versions: Dict[str, str]
    config_hash: Optional[str]
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]
```

**Access Pattern:**
```python
bundle.get_metric("mean_ic", factor_id="momentum_5d")
bundle.get_diagnosis("momentum_5d")
```

## Metric Categories

### 1. Coverage Diagnostics (`metrics/quality.py`)

- **compute_coverage**: Overall fraction of valid observations
- **compute_per_time_coverage**: Coverage per time period

**Purpose:** Identify data quality issues before computing IC.

### 2. Information Coefficient (`metrics/ic.py`)

- **compute_daily_ic**: Daily Pearson or Spearman IC
- **compute_mean_ic**: Mean IC with minimum period check

**Algorithm:**
```
For each time t, asset i:
  ic[t, f] = correlation(factor[t, :, f], label[t, :])
  Only include pairwise-finite observations
```

### 3. IC Summary (`metrics/ic_summary.py`)

- **compute_icir**: Information Coefficient / Information Ratio
- **compute_ic_tstat**: T-statistic and p-value
- **compute_ic_decay**: IC across multiple horizons
- **compute_ic_stability**: Rolling window stability

**ICIR Formula:**
```
ICIR = mean(IC) / std(IC)
```

### 4. Quantile Metrics (`metrics/quantile.py`)

- **assign_quantiles**: Quantile binning with tie handling
- **compute_quantile_returns**: Average returns per quantile
- **compute_top_bottom_spread**: Top minus bottom quantile

**Algorithm:**
```
For each time t:
  1. Rank factor values cross-sectionally
  2. Assign to n quantiles (0 = lowest, n-1 = highest)
  3. Compute mean return per quantile
```

### 5. Turnover (`metrics/turnover.py`)

- **compute_turnover**: Canonical `0.5 * sum(|delta_weights|)`
- **compute_turnover_series**: Time series of turnover
- **estimate_turnover_from_ranks**: Proxy from rank correlation

**Canonical Definition:**
```
turnover[t] = 0.5 * sum_i |w[t, i] - w[t-1, i]|
```

### 6. Temporal Analysis (`metrics/temporal.py`)

- **compute_autocorrelation**: ACF of factor or IC series
- **compute_ic_autocorrelation**: IC decay over lags
- **compute_rank_stability**: Rank correlation over time
- **compute_half_life**: AR(1) half-life estimation

**Purpose:** Understand factor persistence and IC decay.

### 7. Exposure Analysis (`metrics/exposure.py`)

- **compute_factor_loadings**: Risk factor exposures via regression
- **compute_sector_exposure**: Sector concentration
- **compute_style_exposure**: Style factor loadings
- **compute_concentration_hhi**: Herfindahl-Hirschman Index

**Purpose:** Measure systematic risk exposures.

### 8. Robustness (`metrics/robustness.py`)

- **compute_subsample_ic**: Bootstrap subsampling stability
- **compute_hac_variance**: HAC standard errors (Newey-West)
- **compute_hac_tstat**: HAC-corrected t-statistics
- **compute_block_bootstrap_ci**: Block bootstrap confidence intervals

**Purpose:** Correct for autocorrelation and cross-sectional dependence.

### 9. Distribution Analysis (`metrics/distribution.py`)

- **compute_skewness**: Skewness of factor distribution
- **compute_kurtosis**: Excess kurtosis
- **detect_outliers_iqr**: IQR-based outlier detection
- **detect_outliers_zscore**: Z-score outlier detection

**Purpose:** Identify distribution anomalies.

### 10. Portfolio Statistics (`metrics/portfolio_stats.py`)

- **compute_long_short_returns**: Long-short portfolio returns
- **compute_sharpe_ratio**: Risk-adjusted return
- **compute_maximum_drawdown**: Largest peak-to-trough decline
- **compute_sortino_ratio**: Downside-risk adjusted return

**Purpose:** Translate IC into portfolio performance metrics.

### 11. Multiple Testing (`metrics/multiple_testing.py`)

- **bonferroni_correction**: Bonferroni-adjusted p-values
- **benjamini_hochberg_correction**: FDR control
- **holm_bonferroni_correction**: Holm's sequential method
- **sidak_correction**: Šidák correction

**Purpose:** Control false discovery rate when testing many factors.

## Execution Engine

### Evaluator Class

```python
class Evaluator:
    def __init__(
        self,
        enable_cache=True,
        cache_size_mb=1024.0,
        budget=None,
        max_chunk_memory_mb=512.0,
    ):
        ...
    
    def evaluate(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        metric_specs: List[Dict],
        use_chunking: bool = True,
    ) -> EvaluationResult:
        ...
```

**Features:**
- Automatic metric dependency resolution
- Intermediate result caching
- Memory-aware chunking for large batches
- Resource budget tracking

### Dependency Planning

Metrics have dependencies:
```
ic_ir -> mean_ic -> daily_ic
      -> ic_std  -> daily_ic
```

The planner:
1. Builds dependency DAG
2. Topological sort for execution order
3. Caches intermediates to avoid recomputation

### Batch Planning

For large batches:
1. Estimate memory per chunk
2. Split into chunks fitting `max_chunk_memory_mb`
3. Process chunks sequentially
4. Aggregate results

**Chunk Strategy:**
- Prefer splitting on factor dimension (F)
- Keep time dimension intact for temporal metrics
- Merge results at the end

## Error Taxonomy

```
QuantEvaluatorError (base)
├── ContractError (input validation)
│   ├── SchemaVersionError
│   ├── MissingInputError
│   ├── InvalidContractError
│   ├── TimingContractError
│   └── SnapshotMismatchError
├── CapabilityError (unsupported features)
│   ├── UnsupportedMetricError
│   └── OptionalDependencyMissing
├── DataError (evidence quality)
│   ├── InsufficientObservations
│   ├── InvalidValidityMask
│   ├── MissingLabelError
│   └── EvidenceUnavailableError
└── NumericalFailure (computation errors)
    └── OverflowOrNonFiniteError
```

**Error Philosophy:**
- Fail fast on contract violations
- Provide actionable error messages
- Never silently return bad results

## Integration Boundaries

### DataAccess (Optional Adapter)

QE does NOT fetch data. If DA integration is needed:

```python
from quant_evaluator.adapters.data_access import DAAdapter

adapter = DAAdapter(da_client)
batch = adapter.fetch_factor_batch(
    factor_ids=["momentum_5d"],
    universe="HS300",
    start_date="2023-01-01",
    end_date="2023-12-31",
)
```

### FactorEngine (Optional Adapter)

QE does NOT compute factors. If FE integration is needed:

```python
from quant_evaluator.adapters.factor_engine import FEAdapter

adapter = FEAdapter(fe_client)
batch = adapter.compute_factor_batch(
    expressions=["ts_rank(close, 20)"],
    universe="HS300",
    start_date="2023-01-01",
    end_date="2023-12-31",
)
```

### FactorAssets (Separate Package)

QE produces evidence, FA consumes it:

```python
from quant_evaluator import FactorBatch, LabelBundle, Evaluator
from factor_assets import AssetRepository, EvidenceRef

# 1. QE: Generate evidence
evaluator = Evaluator()
result = evaluator.evaluate(batch, labels, metric_specs)

# 2. FA: Store evidence reference
evidence_ref = EvidenceRef(
    evaluation_id=result.request_id,
    factor_id="momentum_5d",
    metric_values=result.metric_values,
    timestamp=result.timestamp,
)

# 3. FA: Register asset
repo = AssetRepository()
asset = repo.register(
    factor_id="momentum_5d",
    expression="ts_rank(close, 20)",
    evidence_refs=[evidence_ref],
)
```

## Performance Considerations

### Memory Usage

**Batch Memory:**
```
memory_mb ≈ T * N * F * 8 bytes / (1024^2)
          ≈ 252 * 3000 * 100 * 8 / (1024^2)
          ≈ 576 MB
```

**Chunking Trigger:**
- If `memory_mb > max_chunk_memory_mb`, split on F dimension
- Process in chunks of size `F_chunk = F * max_chunk_memory_mb / memory_mb`

### Computation Complexity

| Metric | Complexity | Notes |
|--------|-----------|-------|
| Coverage | O(TNF) | Simple mask sum |
| Daily IC | O(TN log N * F) | Spearman requires ranking |
| Quantile Returns | O(TN log N * F) | Cross-sectional sort |
| Turnover | O(TN * F) | Pairwise differences |
| HAC | O(T² * F) | Autocorrelation correction |

### Optimization Strategy

1. **Reference first**: Implement in pure NumPy
2. **Profile**: Identify bottlenecks with real data
3. **Optimize**: Write fast kernels for hot paths
4. **Parity test**: Ensure results match reference
5. **Fallback**: Reference as backup if fast fails

**Current Status:**
- Reference implementations complete
- Fast kernels stub (Wave 2)
- Parity tests in `tests/kernels/test_parity.py`

## Testing Strategy

### Unit Tests

Each metric has dedicated tests:
- `tests/metrics/test_ic.py`
- `tests/metrics/test_quantile.py`
- `tests/metrics/test_turnover.py`
- etc.

**Coverage:**
- Happy path with synthetic data
- Edge cases (all NaN, single valid observation)
- Contract violations (mismatched shapes)

### Integration Tests

- `tests/integration/test_batch_processing.py`: Full pipeline
- `tests/integration/test_multi_factor_evaluation.py`: Batch operations
- `tests/integration/test_slicing_grouping.py`: Subset evaluations

### Parity Tests

- `tests/kernels/test_parity.py`: Fast vs reference
- `tests/kernels/test_performance.py`: Benchmark speed

### Golden Value Tests

Fixed datasets with known correct outputs to catch regressions.

## Future Extensions (Wave 2+)

### Fast Kernels

- Numba JIT compilation for IC computation
- Cython for quantile binning
- Vectorized turnover estimation

### Advanced Metrics

- Multi-horizon IC decay analysis
- Cross-sectional dispersion metrics
- Factor timing analysis
- Tail risk metrics

### Distributed Execution

- Ray/Dask integration for large-scale evaluation
- Parallel evaluation across factors
- Distributed caching

### Streaming Evaluation

- Incremental IC updates
- Online quantile tracking
- Real-time monitoring

## References

- **Information Coefficient**: Grinold & Kahn, "Active Portfolio Management"
- **HAC Standard Errors**: Newey & West (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"
- **Multiple Testing**: Benjamini & Hochberg (1995), "Controlling the False Discovery Rate"

---

**Last Updated:** 2026-08-14  
**Authors:** Quant Platform Team
