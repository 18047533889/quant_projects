# QuantEvaluator API Reference

**Version:** 0.0.1a1  
**Last Updated:** 2026-08-14

## Table of Contents

1. [Contracts](#contracts)
2. [Metrics](#metrics)
3. [Runtime](#runtime)
4. [Registry](#registry)
5. [Planner](#planner)
6. [Diagnosis](#diagnosis)
7. [Errors](#errors)

---

## Contracts

### FactorBatch

Container for factor values with explicit axis metadata.

```python
@dataclass(frozen=True)
class FactorBatch:
    factor_ids: Tuple[str, ...]
    time_axis: AxisRef
    asset_axis: AxisRef
    values: np.ndarray
    validity: Optional[np.ndarray] = None
    layout: str = "wide"
    dtype: str = "float64"
    context_refs: Dict[str, Any] = field(default_factory=dict)
    value_hash: Optional[str] = None
```

**Parameters:**
- `factor_ids`: Tuple of factor identifiers (length F)
- `time_axis`: Time dimension metadata (size T)
- `asset_axis`: Asset dimension metadata (size N)
- `values`: Factor values, shape `(T, N, F)`
- `validity`: Optional boolean mask, shape `(T, N, F)`. True = valid, False = missing
- `layout`: Data layout, default "wide"
- `dtype`: Value data type, default "float64"
- `context_refs`: Additional context metadata
- `value_hash`: Optional content hash for caching

**Properties:**
- `num_factors: int` - Number of factors (F)
- `num_times: int` - Number of time periods (T)
- `num_assets: int` - Number of assets (N)

**Methods:**
- `is_valid(time_idx: int, asset_idx: int, factor_idx: int) -> bool`
  - Check if specific observation is valid

**Example:**
```python
from quant_evaluator import FactorBatch, AxisRef
import numpy as np

batch = FactorBatch(
    factor_ids=("momentum_5d", "reversal_20d"),
    time_axis=AxisRef(name="time", dtype="datetime64", size=100),
    asset_axis=AxisRef(name="asset", dtype="int64", size=500),
    values=np.random.randn(100, 500, 2),
)

print(f"Batch shape: {batch.num_times} x {batch.num_assets} x {batch.num_factors}")
```

---

### AxisRef

Metadata for a single axis/dimension.

```python
@dataclass(frozen=True)
class AxisRef:
    name: str
    dtype: str
    size: int
    values: Optional[np.ndarray] = None
```

**Parameters:**
- `name`: Axis name (e.g., "time", "asset")
- `dtype`: Data type (e.g., "datetime64", "int64")
- `size`: Axis length
- `values`: Optional actual axis values (must match size)

**Example:**
```python
time_axis = AxisRef(
    name="time",
    dtype="datetime64",
    size=252,
    values=pd.date_range("2023-01-01", periods=252, freq="B").values
)
```

---

### LabelBundle

Forward labels with explicit timing contracts.

```python
@dataclass(frozen=True)
class LabelBundle:
    target_id: str
    values: np.ndarray
    horizon: int
    execution_delay: int = 0
    decision_time: Tuple[Any, ...]
    execution_time: Tuple[Any, ...] = field(default_factory=tuple)
    label_start_time: Tuple[Any, ...]
    label_end_time: Tuple[Any, ...]
    validity: Optional[np.ndarray] = None
    source_ref: Optional[str] = None
    calendar_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Parameters:**
- `target_id`: Label identifier (e.g., "forward_return_1d")
- `values`: Label values, shape `(T, N)`
- `horizon`: Forward horizon in periods (must be > 0)
- `execution_delay`: Execution delay in periods (default 0)
- `decision_time`: Decision timestamps (length T, required)
- `execution_time`: Execution timestamps (length T)
- `label_start_time`: Label start timestamps (length T, required)
- `label_end_time`: Label end timestamps (length T, required)
- `validity`: Optional boolean mask, shape `(T, N)`
- `source_ref`: Data source reference
- `calendar_ref`: Trading calendar reference
- `metadata`: Additional metadata

**Timing Invariants:**
- `horizon > 0` (forward-looking)
- `decision_time[t] < label_start_time[t]` (no look-ahead bias)
- `label_start_time[t] < label_end_time[t]` (positive period)

**Methods:**
- `num_observations() -> int` - Total number of observations (T * N)

**Example:**
```python
from quant_evaluator import LabelBundle
import numpy as np

labels = LabelBundle(
    target_id="forward_return_1d",
    values=np.random.randn(100, 500),
    horizon=1,
    decision_time=tuple(range(100)),
    label_start_time=tuple(range(100)),
    label_end_time=tuple(range(1, 101)),
)
```

---

### EvaluationRequest

Request specification for factor evaluation.

```python
@dataclass(frozen=True)
class EvaluationRequest:
    batch_or_factor_ids: Any
    label_bundle: Any
    metric_ids: Tuple[str, ...] = ("pearson_ic", "rank_ic", "coverage")
    slices: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    tier: str = "core"
    cost_budget: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Parameters:**
- `batch_or_factor_ids`: FactorBatch or list of factor IDs
- `label_bundle`: LabelBundle with forward labels
- `metric_ids`: Tuple of metric identifiers to compute
- `slices`: Optional subset specifications
- `context`: Execution context
- `tier`: Metric tier ("core", "extended", "research")
- `cost_budget`: Optional resource budget
- `metadata`: Additional request metadata

---

### EvaluationBundle

Evaluation result bundle with versioned metrics.

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
    grouped_metrics: Optional[Dict[str, Dict[str, MetricValue]]] = None
    series_refs: Optional[Dict[str, str]] = None
    metric_versions: Dict[str, str] = field(default_factory=dict)
    config_hash: Optional[str] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Methods:**
- `get_metric(metric_id: str, factor_id: Optional[str] = None) -> Optional[MetricValue]`
  - Retrieve specific metric value
- `get_diagnosis(factor_id: str) -> Optional[FactorDiagnosis]`
  - Retrieve factor diagnostics

**Example:**
```python
# After evaluation
bundle = evaluator.evaluate(batch, labels, metric_specs)

# Access metrics
mean_ic = bundle.get_metric("mean_ic", factor_id="momentum_5d")
print(f"Mean IC: {mean_ic.value:.4f}")

# Access diagnostics
diagnosis = bundle.get_diagnosis("momentum_5d")
print(f"Coverage: {diagnosis.coverage:.2%}")
```

---

### MetricValue

Individual metric result with metadata.

```python
@dataclass(frozen=True)
class MetricValue:
    metric_id: str
    value: Optional[float]
    valid: bool
    observation_count: int
    metric_version: str = "0.1"
    warnings: Tuple[str, ...] = field(default_factory=tuple)
```

**Parameters:**
- `metric_id`: Metric identifier
- `value`: Computed metric value (None if invalid)
- `valid`: Whether metric is valid
- `observation_count`: Number of valid observations used
- `metric_version`: Metric implementation version
- `warnings`: Computation warnings

---

### FactorDiagnosis

Per-factor health diagnostics.

```python
@dataclass(frozen=True)
class FactorDiagnosis:
    factor_id: str
    num_valid_observations: int
    num_missing: int
    coverage: float
    is_constant: bool
    has_nans: bool
    has_infs: bool
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    warnings: Tuple[str, ...] = field(default_factory=tuple)
```

**Parameters:**
- `factor_id`: Factor identifier
- `num_valid_observations`: Count of valid observations
- `num_missing`: Count of missing observations
- `coverage`: Fraction of valid observations
- `is_constant`: Whether factor is constant
- `has_nans`: Whether factor contains NaN
- `has_infs`: Whether factor contains Inf
- `min_value`: Minimum value (if any valid)
- `max_value`: Maximum value (if any valid)
- `mean_value`: Mean value (if any valid)
- `warnings`: Diagnostic warnings

---

## Metrics

### Coverage Diagnostics (`metrics.quality`)

#### compute_coverage

```python
def compute_coverage(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> Tuple[float, int, int]:
```

Compute overall fraction of valid (factor, label) pairs.

**Parameters:**
- `factor_batch`: Factor values
- `label_bundle`: Forward labels
- `min_assets`: Minimum assets per time period (default 10)

**Returns:**
- `coverage`: Fraction of valid pairs [0, 1]
- `num_valid`: Count of valid observations
- `num_total`: Total possible observations

**Example:**
```python
from quant_evaluator.metrics import compute_coverage

coverage, n_valid, n_total = compute_coverage(batch, labels, min_assets=10)
print(f"Coverage: {coverage:.2%} ({n_valid}/{n_total})")
```

---

#### compute_per_time_coverage

```python
def compute_per_time_coverage(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
```

Compute coverage per time period.

**Parameters:**
- `factor_batch`: Factor values
- `label_bundle`: Forward labels

**Returns:**
- `coverage`: Coverage array, shape `(T, F)`, values in [0, 1]

---

### Information Coefficient (`metrics.ic`)

#### compute_daily_ic

```python
def compute_daily_ic(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    method: str = "pearson",
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
```

Compute daily Information Coefficient with pairwise-finite filtering.

**Parameters:**
- `factor_batch`: Factor values
- `label_bundle`: Forward labels
- `method`: Correlation method, "pearson" or "spearman" (default "pearson")
- `min_assets`: Minimum pairwise-valid assets (default 10)

**Returns:**
- `ic_series`: Daily IC, shape `(T, F)`. NaN if insufficient observations
- `valid_counts`: Valid observation counts, shape `(T, F)`

**Algorithm:**
```
For each time t, factor f:
  1. Extract factor[t, :, f] and label[t, :]
  2. Keep only pairwise-finite observations
  3. If count >= min_assets:
       ic[t, f] = correlation(factor, label)
     Else:
       ic[t, f] = NaN
```

**Example:**
```python
from quant_evaluator.metrics import compute_daily_ic
import numpy as np

# Spearman (rank) IC
ic_series, counts = compute_daily_ic(batch, labels, method="spearman")

# Mean IC across time
mean_ic = np.nanmean(ic_series, axis=0)
for i, factor_id in enumerate(batch.factor_ids):
    print(f"{factor_id}: {mean_ic[i]:.4f}")
```

---

#### compute_mean_ic

```python
def compute_mean_ic(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
```

Compute mean IC with minimum period validation.

**Parameters:**
- `ic_series`: Daily IC time series, shape `(T, F)`
- `valid_counts`: Optional valid counts (unused currently)
- `min_periods`: Minimum number of periods (default 20)

**Returns:**
- `mean_ic`: Mean IC per factor, shape `(F,)`. NaN if < min_periods
- `ic_std`: IC standard deviation, shape `(F,)`

**Example:**
```python
from quant_evaluator.metrics import compute_daily_ic, compute_mean_ic

ic_series, _ = compute_daily_ic(batch, labels)
mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)

print(f"Mean IC: {mean_ic[0]:.4f} ± {ic_std[0]:.4f}")
```

---

### IC Summary (`metrics.ic_summary`)

#### compute_icir

```python
def compute_icir(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
```

Compute Information Coefficient / Information Ratio.

**Formula:** `ICIR = mean(IC) / std(IC)`

**Parameters:**
- `ic_series`: Daily IC time series, shape `(T, F)`
- `min_periods`: Minimum periods (default 20)

**Returns:**
- `icir`: IC/IR per factor, shape `(F,)`. NaN if < min_periods

---

#### compute_ic_tstat

```python
def compute_ic_tstat(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
```

Compute t-statistic and p-value for IC.

**Parameters:**
- `ic_series`: Daily IC time series, shape `(T, F)`
- `min_periods`: Minimum periods (default 20)

**Returns:**
- `t_stat`: T-statistic, shape `(F,)`
- `p_value`: Two-tailed p-value, shape `(F,)`

**Example:**
```python
from quant_evaluator.metrics import compute_ic_tstat

t_stat, p_value = compute_ic_tstat(ic_series, min_periods=20)

for i, factor_id in enumerate(batch.factor_ids):
    if p_value[i] < 0.05:
        print(f"{factor_id}: t={t_stat[i]:.2f}, p={p_value[i]:.4f} (significant)")
```

---

#### compute_ic_decay

```python
def compute_ic_decay(
    factor_batch: FactorBatch,
    label_bundles: List[LabelBundle],
    method: str = "pearson",
    min_assets: int = 10,
) -> np.ndarray:
```

Compute IC decay across multiple horizons.

**Parameters:**
- `factor_batch`: Factor values
- `label_bundles`: List of LabelBundle at different horizons
- `method`: Correlation method (default "pearson")
- `min_assets`: Minimum assets (default 10)

**Returns:**
- `decay_matrix`: IC matrix, shape `(num_horizons, F)`

---

#### compute_ic_stability

```python
def compute_ic_stability(
    ic_series: np.ndarray,
    window_size: int = 60,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
```

Compute IC stability using rolling windows.

**Parameters:**
- `ic_series`: Daily IC, shape `(T, F)`
- `window_size`: Rolling window size (default 60)
- `min_periods`: Minimum periods per window (default 20)

**Returns:**
- `stability_series`: Rolling mean IC, shape `(num_windows, F)`
- `valid_windows`: Number of valid windows per factor, shape `(F,)`

---

### Quantile Metrics (`metrics.quantile`)

#### assign_quantiles

```python
def assign_quantiles(
    values: np.ndarray,
    n_quantiles: int = 5,
    method: str = "average",
) -> np.ndarray:
```

Assign cross-sectional quantile bins.

**Parameters:**
- `values`: Input values
- `n_quantiles`: Number of quantiles (default 5)
- `method`: Tie-breaking method (default "average")

**Returns:**
- `quantiles`: Quantile assignments, -1 for NaN

---

#### compute_quantile_returns

```python
def compute_quantile_returns(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
```

Compute average returns per quantile.

**Parameters:**
- `factor_batch`: Factor values
- `label_bundle`: Forward returns
- `n_quantiles`: Number of quantiles (default 5)
- `min_assets`: Minimum assets per quantile (default 10)

**Returns:**
- `quantile_returns`: Mean returns, shape `(T, n_quantiles, F)`
- `quantile_counts`: Asset counts, shape `(T, n_quantiles, F)`

**Example:**
```python
from quant_evaluator.metrics import compute_quantile_returns

returns, counts = compute_quantile_returns(batch, labels, n_quantiles=5)

# Mean return per quantile (averaged across time)
mean_returns = np.nanmean(returns, axis=0)  # Shape: (5, F)

# Top quantile (0) vs bottom quantile (4)
spread = mean_returns[0] - mean_returns[4]
print(f"Top-bottom spread: {spread[0]:.4f}")
```

---

#### compute_top_bottom_spread

```python
def compute_top_bottom_spread(
    quantile_returns: np.ndarray,
    top_q: int = 0,
    bottom_q: int = -1,
) -> np.ndarray:
```

Compute top minus bottom quantile spread.

**Parameters:**
- `quantile_returns`: Quantile returns, shape `(T, n_quantiles, F)`
- `top_q`: Top quantile index (default 0 = highest)
- `bottom_q`: Bottom quantile index (default -1 = lowest)

**Returns:**
- `spread`: Top - bottom spread, shape `(T, F)`

---

### Turnover (`metrics.turnover`)

#### compute_turnover

```python
def compute_turnover(
    weights_t0: np.ndarray,
    weights_t1: np.ndarray,
    method: str = "half_sum_abs",
) -> float:
```

Compute portfolio turnover between two periods.

**Formula:** `turnover = 0.5 * sum(|w_t1 - w_t0|)`

**Parameters:**
- `weights_t0`: Weights at time t
- `weights_t1`: Weights at time t+1
- `method`: Turnover method (default "half_sum_abs")

**Returns:**
- `turnover`: Turnover scalar in [0, 1]

---

#### compute_turnover_series

```python
def compute_turnover_series(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
```

Compute turnover time series.

**Parameters:**
- `weights`: Weight matrix, shape `(T, N)`
- `method`: Turnover method (default "half_sum_abs")

**Returns:**
- `turnover`: Turnover series, shape `(T,)`. First value is NaN

---

#### estimate_turnover_from_ranks

```python
def estimate_turnover_from_ranks(
    factor_batch: FactorBatch,
    window: int = 1,
) -> np.ndarray:
```

Estimate turnover from rank correlation proxy.

**Parameters:**
- `factor_batch`: Factor values
- `window`: Lag window (default 1)

**Returns:**
- `turnover`: Estimated turnover, shape `(T, F)`

---

### Temporal Analysis (`metrics.temporal`)

#### compute_autocorrelation

```python
def compute_autocorrelation(
    series: np.ndarray,
    max_lag: int = 20,
    min_obs: int = 30,
) -> np.ndarray:
```

Compute autocorrelation function.

**Parameters:**
- `series`: Time series, shape `(T,)` or `(T, F)`
- `max_lag`: Maximum lag (default 20)
- `min_obs`: Minimum observations (default 30)

**Returns:**
- `acf`: Autocorrelation, shape `(max_lag+1,)` or `(max_lag+1, F)`

---

#### compute_ic_autocorrelation

```python
def compute_ic_autocorrelation(
    ic_series: np.ndarray,
    max_lag: int = 20,
    min_obs: int = 30,
) -> np.ndarray:
```

Compute IC autocorrelation (IC decay).

**Parameters:**
- `ic_series`: IC time series, shape `(T, F)`
- `max_lag`: Maximum lag (default 20)
- `min_obs`: Minimum observations (default 30)

**Returns:**
- `acf`: IC autocorrelation, shape `(max_lag+1, F)`

---

#### compute_rank_stability

```python
def compute_rank_stability(
    factor_values: np.ndarray,
    lag: int = 1,
    method: str = "spearman",
) -> np.ndarray:
```

Compute cross-sectional rank stability over time.

**Parameters:**
- `factor_values`: Factor values, shape `(T, N, F)`
- `lag`: Lag periods (default 1)
- `method`: Correlation method (default "spearman")

**Returns:**
- `stability`: Rank correlation, shape `(T-lag, F)`

---

#### compute_half_life

```python
def compute_half_life(
    ic_series: np.ndarray,
    min_periods: int = 60,
) -> np.ndarray:
```

Estimate IC half-life from AR(1) model.

**Parameters:**
- `ic_series`: IC time series, shape `(T, F)`
- `min_periods`: Minimum periods (default 60)

**Returns:**
- `half_life`: Half-life in periods, shape `(F,)`. NaN if model fails

---

### Exposure Analysis (`metrics.exposure`)

#### compute_factor_loadings

```python
def compute_factor_loadings(
    factor_values: np.ndarray,
    risk_factors: np.ndarray,
    intercept: bool = True,
    min_obs: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

Compute risk factor exposures via OLS regression.

**Parameters:**
- `factor_values`: Factor values, shape `(T, N)`
- `risk_factors`: Risk factors, shape `(T, K)`
- `intercept`: Add intercept (default True)
- `min_obs`: Minimum observations (default 10)

**Returns:**
- `loadings`: Factor loadings, shape `(T, K)` or `(T, K+1)` with intercept
- `r_squared`: R², shape `(T,)`
- `residuals`: Residuals, shape `(T, N)`

---

#### compute_sector_exposure

```python
def compute_sector_exposure(
    factor_values: np.ndarray,
    sector_labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
```

Compute sector exposure concentration.

**Parameters:**
- `factor_values`: Factor values, shape `(T, N)`
- `sector_labels`: Sector IDs, shape `(N,)`
- `weights`: Optional weights, shape `(T, N)`

**Returns:**
- `sector_exposure`: Mean exposure per sector, shape `(T, num_sectors)`
- `sector_counts`: Asset counts, shape `(T, num_sectors)`

---

#### compute_concentration_hhi

```python
def compute_concentration_hhi(
    factor_values: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
```

Compute Herfindahl-Hirschman Index (concentration).

**Parameters:**
- `factor_values`: Factor values, shape `(T, N)`
- `weights`: Optional weights, shape `(T, N)`

**Returns:**
- `hhi`: HHI per time, shape `(T,)`. Range [1/N, 1]

---

### Robustness (`metrics.robustness`)

#### compute_subsample_ic_std

```python
def compute_subsample_ic_std(
    ic_series: np.ndarray,
    num_subsamples: int = 100,
    subsample_fraction: float = 0.8,
    random_seed: Optional[int] = None,
) -> np.ndarray:
```

Compute IC robustness via bootstrap subsampling.

**Parameters:**
- `ic_series`: IC time series, shape `(T, F)`
- `num_subsamples`: Number of bootstrap samples (default 100)
- `subsample_fraction`: Fraction to sample (default 0.8)
- `random_seed`: Random seed for reproducibility

**Returns:**
- `robustness_std`: Standard deviation across subsamples, shape `(F,)`

---

#### compute_hac_tstat

```python
def compute_hac_tstat(
    ic_series: np.ndarray,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> Tuple[np.ndarray, np.ndarray]:
```

Compute HAC-corrected t-statistics (Newey-West).

**Parameters:**
- `ic_series`: IC time series, shape `(T, F)`
- `max_lag`: Maximum lag for HAC (default 5)
- `kernel`: Kernel type, "bartlett" or "uniform" (default "bartlett")

**Returns:**
- `t_stat_hac`: HAC-corrected t-stat, shape `(F,)`
- `se_hac`: HAC standard errors, shape `(F,)`

**Example:**
```python
from quant_evaluator.metrics import compute_hac_tstat

t_stat, se = compute_hac_tstat(ic_series, max_lag=5)
print(f"HAC t-stat: {t_stat[0]:.2f} (SE: {se[0]:.4f})")
```

---

#### compute_block_bootstrap_ci

```python
def compute_block_bootstrap_ci(
    ic_series: np.ndarray,
    block_length: int = 10,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
```

Compute block bootstrap confidence intervals.

**Parameters:**
- `ic_series`: IC time series, shape `(T, F)`
- `block_length`: Block length (default 10)
- `num_bootstrap`: Number of bootstrap samples (default 1000)
- `confidence_level`: CI level (default 0.95)
- `random_seed`: Random seed

**Returns:**
- `ci_lower`: Lower bound, shape `(F,)`
- `ci_upper`: Upper bound, shape `(F,)`

---

### Distribution Analysis (`metrics.distribution`)

#### compute_skewness

```python
def compute_skewness(
    values: np.ndarray,
    axis: int = 0,
    min_obs: int = 10,
) -> np.ndarray:
```

Compute distribution skewness.

---

#### compute_kurtosis

```python
def compute_kurtosis(
    values: np.ndarray,
    axis: int = 0,
    min_obs: int = 10,
    excess: bool = True,
) -> np.ndarray:
```

Compute excess kurtosis.

---

#### detect_outliers_iqr

```python
def detect_outliers_iqr(
    values: np.ndarray,
    axis: int = 0,
    multiplier: float = 1.5,
    min_obs: int = 10,
) -> np.ndarray:
```

Detect outliers using IQR method.

**Returns:** Boolean mask, True = outlier

---

### Portfolio Statistics (`metrics.portfolio_stats`)

#### compute_sharpe_ratio

```python
def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> np.ndarray:
```

Compute Sharpe ratio.

**Formula:** `Sharpe = (mean_return - rf) / std_return * sqrt(periods_per_year)`

---

#### compute_maximum_drawdown

```python
def compute_maximum_drawdown(
    returns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

Compute maximum drawdown.

**Returns:**
- `max_drawdown`: Maximum drawdown
- `drawdown_series`: Full drawdown series
- `peak_indices`: Peak indices

---

### Multiple Testing (`metrics.multiple_testing`)

#### bonferroni_correction

```python
def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
```

Bonferroni multiple testing correction.

**Returns:**
- `adjusted_p_values`: Bonferroni-adjusted p-values
- `reject_mask`: Boolean mask, True = reject null

---

#### benjamini_hochberg_correction

```python
def benjamini_hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int]:
```

Benjamini-Hochberg FDR control.

**Returns:**
- `adjusted_p_values`: BH-adjusted p-values
- `reject_mask`: Boolean mask
- `n_discoveries`: Number of discoveries

---

## Runtime

### Evaluator

Main evaluation orchestrator with caching and chunking.

```python
class Evaluator:
    def __init__(
        self,
        enable_cache: bool = True,
        cache_size_mb: float = 1024.0,
        budget: Optional[ComputationBudget] = None,
        max_chunk_memory_mb: float = 512.0,
    ):
```

**Parameters:**
- `enable_cache`: Enable intermediate caching (default True)
- `cache_size_mb`: Cache size limit in MB (default 1024)
- `budget`: Optional resource budget
- `max_chunk_memory_mb`: Chunk size limit (default 512)

**Methods:**

#### register_metric

```python
def register_metric(
    self,
    metric_id: str,
    metric_fn: Callable,
    metric_kind: MetricKind = MetricKind.CUSTOM,
) -> None:
```

Register custom metric function.

---

#### evaluate

```python
def evaluate(
    self,
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    metric_specs: List[Dict],
    use_chunking: bool = True,
) -> EvaluationResult:
```

Evaluate factor batch.

**Parameters:**
- `factor_batch`: Factor values
- `label_bundle`: Forward labels
- `metric_specs`: List of metric specifications
- `use_chunking`: Enable automatic chunking (default True)

**Returns:**
- `EvaluationResult` with metrics and diagnostics

**Example:**
```python
evaluator = Evaluator(enable_cache=True)

metric_specs = [
    {"metric_id": "mean_ic", "method": "spearman"},
    {"metric_id": "coverage"},
]

result = evaluator.evaluate(batch, labels, metric_specs)
print(f"Execution time: {result.execution_time_seconds:.2f}s")
```

---

#### clear_cache

```python
def clear_cache(self) -> None:
```

Clear intermediate cache.

---

#### get_cache_stats

```python
def get_cache_stats(self) -> Dict:
```

Get cache statistics.

**Returns:** Dict with cache hit/miss counts, size, etc.

---

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    metrics: Dict[str, Any]
    diagnostics: Dict[str, Any]
    execution_time_seconds: float
    chunks_processed: int
    cache_hits: int
    cache_misses: int
    resource_usage: Optional[ResourceUsage]
    metadata: Dict[str, Any]
```

**Methods:**
- `get_metric(metric_id: str, default=None) -> Any`
- `has_metric(metric_id: str) -> bool`
- `to_dict() -> Dict`

---

## Registry

### MetricRegistry

Centralized metric catalog.

```python
# Get default registry
from quant_evaluator.registry import get_metric, list_metrics

# List all metrics
all_metrics = list_metrics()

# Get specific metric
spec = get_metric("mean_ic")
print(f"{spec.display_name}: {spec.description}")
```

**Functions:**
- `register_metric(spec: MetricSpec) -> None`
- `get_metric(name: str) -> MetricSpec`
- `list_metrics() -> List[str]`
- `list_metrics_by_status(status: MetricStatus) -> List[str]`
- `list_metrics_by_tier(tier: MetricTier) -> List[str]`

---

## Planner

### create_batch_plan

```python
def create_batch_plan(
    factor_batch: FactorBatch,
    max_chunk_memory_mb: float = 512.0,
) -> BatchPlan:
```

Create memory-aware batch chunking plan.

---

### resolve_metric_dependencies

```python
def resolve_metric_dependencies(
    metric_specs: List[Dict],
) -> MetricDependencyGraph:
```

Resolve metric dependency DAG for execution ordering.

---

## Diagnosis

### diagnose_factor

```python
def diagnose_factor(
    factor_batch: FactorBatch,
    factor_idx: int = 0,
) -> FactorDiagnosis:
```

Diagnose single factor health.

---

### diagnose_all_factors

```python
def diagnose_all_factors(
    factor_batch: FactorBatch,
) -> Dict[str, FactorDiagnosis]:
```

Diagnose all factors in batch.

---

## Errors

### Exception Hierarchy

```
QuantEvaluatorError (base)
├── ContractError
│   ├── SchemaVersionError
│   ├── MissingInputError
│   ├── InvalidContractError
│   ├── TimingContractError
│   └── SnapshotMismatchError
├── CapabilityError
│   ├── UnsupportedMetricError
│   └── OptionalDependencyMissing
├── DataError
│   ├── InsufficientObservations
│   ├── InvalidValidityMask
│   ├── MissingLabelError
│   └── EvidenceUnavailableError
└── NumericalFailure
    └── OverflowOrNonFiniteError
```

**Usage:**
```python
from quant_evaluator import InsufficientObservations, ContractError

try:
    ic_series, _ = compute_daily_ic(batch, labels, min_assets=1000)
except InsufficientObservations as e:
    print(f"Not enough data: {e}")
except ContractError as e:
    print(f"Contract violation: {e}")
```

---

**Last Updated:** 2026-08-14  
**Version:** 0.0.1a1
