# QuantEvaluator Quick Start

**5-Minute Guide to Evaluating Factors**

## Installation

```bash
cd /home/shw/quant_projects/quant_evaluator
pip install -e .
```

## Basic Example

Evaluate a single factor in under 10 lines:

```python
import numpy as np
from quant_evaluator import FactorBatch, LabelBundle, AxisRef
from quant_evaluator.metrics import compute_daily_ic, compute_coverage

# Create factor batch (T=100 days, N=500 stocks, F=1 factor)
time_axis = AxisRef(name="time", dtype="datetime64", size=100)
asset_axis = AxisRef(name="asset", dtype="int64", size=500)
factor_values = np.random.randn(100, 500, 1)  # Shape: (T, N, F)

batch = FactorBatch(
    factor_ids=("momentum_5d",),
    time_axis=time_axis,
    asset_axis=asset_axis,
    values=factor_values,
)

# Create forward return labels (horizon=1 day)
labels = np.random.randn(100, 500)  # Shape: (T, N)
bundle = LabelBundle(
    target_id="forward_return_1d",
    values=labels,
    horizon=1,
    decision_time=tuple(range(100)),
    label_start_time=tuple(range(100)),
    label_end_time=tuple(range(1, 101)),
)

# Compute metrics
coverage, num_valid, num_total = compute_coverage(batch, bundle)
ic_series, valid_counts = compute_daily_ic(batch, bundle, method="spearman")

print(f"Coverage: {coverage:.2%}")
print(f"Mean RankIC: {np.nanmean(ic_series):.4f}")
print(f"IC Std: {np.nanstd(ic_series):.4f}")
```

**Output:**
```
Coverage: 98.50%
Mean RankIC: 0.0234
IC Std: 0.0891
```

## Multi-Factor Evaluation

Evaluate multiple factors in a single batch:

```python
# Create batch with 3 factors
factor_values = np.random.randn(100, 500, 3)  # Shape: (T, N, 3)

batch = FactorBatch(
    factor_ids=("momentum_5d", "reversal_20d", "volume_trend"),
    time_axis=time_axis,
    asset_axis=asset_axis,
    values=factor_values,
)

# Evaluate all at once
ic_series, valid_counts = compute_daily_ic(batch, bundle, method="spearman")

# ic_series.shape = (100, 3) - daily IC for each factor
for i, factor_id in enumerate(batch.factor_ids):
    mean_ic = np.nanmean(ic_series[:, i])
    print(f"{factor_id}: IC={mean_ic:.4f}")
```

## Common Metrics

### Information Coefficient (IC)

```python
from quant_evaluator.metrics import compute_daily_ic, compute_mean_ic

# Daily IC time series
ic_series, valid_counts = compute_daily_ic(
    batch, bundle, 
    method="spearman",  # or "pearson"
    min_assets=10
)

# Mean IC with minimum period check
mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)
```

### Coverage

```python
from quant_evaluator.metrics import compute_coverage, compute_per_time_coverage

# Overall coverage
coverage, num_valid, num_total = compute_coverage(batch, bundle, min_assets=10)

# Per-time coverage (T, F) array
per_time_cov = compute_per_time_coverage(batch, bundle)
```

### Quantile Analysis

```python
from quant_evaluator.metrics import (
    compute_quantile_returns,
    compute_top_bottom_spread
)

# Quantile returns (T, n_quantiles, F)
quantile_returns, quantile_counts = compute_quantile_returns(
    batch, bundle, 
    n_quantiles=5,
    min_assets=10
)

# Top-bottom spread (T, F)
spread = compute_top_bottom_spread(
    quantile_returns,
    top_q=0,      # Top quantile (0 = highest)
    bottom_q=-1   # Bottom quantile (-1 = lowest)
)

print(f"Mean spread: {np.nanmean(spread):.4f}")
```

### Turnover

```python
from quant_evaluator.metrics import estimate_turnover_from_ranks

# Estimate turnover from rank correlation (T, F)
turnover = estimate_turnover_from_ranks(batch, window=1)

print(f"Mean turnover: {np.nanmean(turnover):.2%}")
```

## Using the Evaluator Class

For production workflows, use the `Evaluator` class:

```python
from quant_evaluator.runtime import Evaluator, MetricKind

# Create evaluator with caching
evaluator = Evaluator(
    enable_cache=True,
    cache_size_mb=1024.0,
    max_chunk_memory_mb=512.0,
)

# Register custom metrics if needed
evaluator.register_metric(
    metric_id="custom_ic",
    metric_fn=my_custom_ic_function,
    metric_kind=MetricKind.IC,
)

# Define metrics to compute
metric_specs = [
    {"metric_id": "mean_ic", "method": "spearman"},
    {"metric_id": "coverage", "min_assets": 10},
    {"metric_id": "ic_ir"},
    {"metric_id": "turnover"},
]

# Evaluate
result = evaluator.evaluate(
    factor_batch=batch,
    label_bundle=bundle,
    metric_specs=metric_specs,
    use_chunking=True,  # Auto-chunk if batch is large
)

# Access results
print(f"Mean IC: {result.get_metric('mean_ic'):.4f}")
print(f"Coverage: {result.get_metric('coverage'):.2%}")
print(f"IC/IR: {result.get_metric('ic_ir'):.4f}")
print(f"Execution time: {result.execution_time_seconds:.2f}s")
print(f"Cache hits: {result.cache_hits}")
```

## Working with Validity Masks

Handle missing data explicitly:

```python
# Create validity mask (True = valid, False = missing)
validity = np.ones((100, 500, 1), dtype=bool)
validity[:10, :, :] = False  # First 10 days are invalid

batch = FactorBatch(
    factor_ids=("momentum_5d",),
    time_axis=time_axis,
    asset_axis=asset_axis,
    values=factor_values,
    validity=validity,  # Explicit validity
)

# Metrics automatically respect validity
ic_series, valid_counts = compute_daily_ic(batch, bundle)
# valid_counts shows actual observations per day
```

## Error Handling

```python
from quant_evaluator import (
    InsufficientObservations,
    TimingContractError,
    ContractError,
)

try:
    ic_series, counts = compute_daily_ic(batch, bundle, min_assets=1000)
except InsufficientObservations as e:
    print(f"Not enough observations: {e}")
    
try:
    # Mismatched time dimensions
    bad_batch = FactorBatch(
        factor_ids=("test",),
        time_axis=AxisRef("time", "int", size=50),  # Wrong size
        asset_axis=asset_axis,
        values=np.random.randn(50, 500, 1),
    )
    compute_daily_ic(bad_batch, bundle)
except ContractError as e:
    print(f"Contract violation: {e}")
```

## Next Steps

- **Full API:** See [API_REFERENCE.md](API_REFERENCE.md)
- **Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Testing:** See [TESTING.md](TESTING.md)
- **Advanced metrics:** Explore `metrics/ic_summary.py`, `metrics/temporal.py`, `metrics/portfolio_stats.py`

## Common Patterns

### Pattern 1: Batch Processing Multiple Factors

```python
# Evaluate 50 factors at once
factor_ids = [f"factor_{i}" for i in range(50)]
factor_values = np.random.randn(252, 3000, 50)  # 1 year, 3k stocks, 50 factors

batch = FactorBatch(
    factor_ids=tuple(factor_ids),
    time_axis=AxisRef("time", "datetime64", size=252),
    asset_axis=AxisRef("asset", "int64", size=3000),
    values=factor_values,
)

# Single call evaluates all 50
ic_series, counts = compute_daily_ic(batch, bundle, method="spearman")
# ic_series.shape = (252, 50)

# Find best factors
mean_ics = np.nanmean(ic_series, axis=0)
best_idx = np.argsort(mean_ics)[-5:]  # Top 5
print("Best factors:", [factor_ids[i] for i in best_idx])
```

### Pattern 2: Time-Series Analysis

```python
from quant_evaluator.metrics import (
    compute_ic_autocorrelation,
    compute_half_life,
)

# IC autocorrelation (decay)
acf = compute_ic_autocorrelation(ic_series, max_lag=20)

# Half-life from AR(1) model
half_life = compute_half_life(ic_series, min_periods=60)
print(f"IC half-life: {half_life[0]:.1f} days")
```

### Pattern 3: Robustness Checks

```python
from quant_evaluator.metrics import (
    compute_subsample_ic_std,
    compute_hac_tstat,
)

# Subsample stability
subsample_std = compute_subsample_ic_std(
    ic_series,
    num_subsamples=100,
    subsample_fraction=0.8,
)

# HAC-corrected t-stat
t_stat_hac, se_hac = compute_hac_tstat(
    ic_series,
    max_lag=5,
    kernel="bartlett",
)

print(f"Subsample std: {subsample_std[0]:.4f}")
print(f"HAC t-stat: {t_stat_hac[0]:.2f}")
```

## Tips

1. **Batch over single**: Always prefer batch operations over loops
2. **Explicit timing**: Never let QE infer decision/label timing
3. **Validity masks**: Use validity instead of NaN for missing data
4. **Min observations**: Set appropriate `min_assets`/`min_periods` thresholds
5. **Chunking**: Enable for large batches (>10GB memory)

## Troubleshooting

**Q: My IC is always NaN**  
A: Check `valid_counts` array - likely insufficient pairwise-valid observations. Lower `min_assets` or check data quality.

**Q: Slow evaluation**  
A: Enable chunking with `use_chunking=True` or use the `Evaluator` class with caching.

**Q: Memory errors**  
A: Reduce `max_chunk_memory_mb` or process factors in smaller batches.

**Q: TimingContractError**  
A: Verify `decision_time`, `label_start_time`, `label_end_time` are correctly aligned and `horizon > 0`.
