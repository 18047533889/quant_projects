# FactorPreprocess Quick Start

**5-Minute Guide to Model Input Preparation**

## Installation

```bash
cd /home/shw/quant_projects/factor_preprocess
pip install -e .
```

## Basic Example

Apply cross-sectional transforms:

```python
import numpy as np
import pandas as pd
from factor_preprocess.transforms import cs_rank, cs_zscore, cs_winsor

# Factor values (100 days, 500 stocks)
values = np.random.randn(100, 500)

# Cross-sectional rank (per day)
ranked = cs_rank(values, axis=-1, pct=True)  # [0, 1] percentile ranks

# Z-score standardization (per day)
zscore = cs_zscore(values, axis=-1)

# Winsorization (per day)
winsorized = cs_winsor(values, lower=0.01, upper=0.99, axis=-1)

print(f"Ranked shape: {ranked.shape}")  # (100, 500)
print(f"Mean per day: {np.mean(ranked, axis=1)[:5]}")  # All ≈0.5
```

## Rolling Transforms

Time-series transforms with explicit lag:

```python
from factor_preprocess.transforms import rolling_mean, rolling_zscore

# Convert to DataFrame (required for rolling ops)
df = pd.DataFrame({
    'date': np.repeat(range(100), 500),
    'asset': np.tile(range(500), 100),
    'value': np.random.randn(100 * 500),
})

# Rolling mean (20-day, lagged)
df['rolling_mean'] = rolling_mean(
    df,
    window=20,
    min_periods=15,
    asset_col='asset',
    time_col='date',
    value_col='value',
)

# Rolling z-score
df['rolling_zscore'] = rolling_zscore(
    df,
    window=60,
    min_periods=30,
    asset_col='asset',
    time_col='date',
    value_col='value',
)

print(df.head())
```

## Neutralization

Remove systematic exposures:

```python
from factor_preprocess.neutralization import ols_neutralize

# Factor values
factor_df = pd.DataFrame({
    'date': np.repeat(range(100), 500),
    'asset': np.tile(range(500), 100),
    'value': np.random.randn(100 * 500),
})

# Risk factor exposures (e.g., market, size, industry)
exposures_df = pd.DataFrame({
    'date': np.repeat(range(100), 500),
    'asset': np.tile(range(500), 100),
    'market_beta': np.random.randn(100 * 500),
    'log_size': np.random.randn(100 * 500),
})

# Neutralize
residuals = ols_neutralize(
    values=factor_df,
    exposures=exposures_df,
    date_col='date',
    asset_col='asset',
    value_col='value',
)

factor_df['neutral_value'] = residuals
print(f"Neutralized correlation with market: {factor_df[['value', 'neutral_value']].corr()}")
```

## Multichannel Representation

Build multiple feature channels:

```python
from factor_preprocess.representation import build_multichannel, MultichannelConfig

# Raw factor values (T=100, N=500, F=10)
values = np.random.randn(100, 500, 10)

# Configure channels
config = MultichannelConfig(
    channels=['raw', 'rank', 'zscore'],
    rank_pct=True,
    zscore_ddof=1,
)

# Build
result = build_multichannel(values, config, axis=-2)

print(f"Channels: {result.channels}")
print(f"Shape per channel: {result.shape}")
print(f"Total shape: {result.as_stacked().shape}")  # (T, N, F*3)
```

## Linear Model Ready

Prepare features for linear models:

```python
from factor_preprocess.representation import build_linear_ready, LinearReadyConfig

# Factor values with NaNs
values = np.random.randn(100, 50)
values[10:15, :5] = np.nan

# Configure
config = LinearReadyConfig(
    fill_method='zero',
    add_intercept=True,
    standardize=True,
)

# Build
result = build_linear_ready(values, config, feature_names=[f"f_{i}" for i in range(50)])

print(f"X shape: {result.X.shape}")  # (100, 51) with intercept
print(f"Features: {result.feature_names[:5]}")
print(f"Has intercept: {result.has_intercept}")
```

## Tree Model Ready

Prepare for tree-based models:

```python
from factor_preprocess.representation import build_tree_ready, TreeReadyConfig

# Configure
config = TreeReadyConfig(
    handle_missing='keep',
    add_missing_indicator=True,
    clip_outliers=True,
    outlier_std_threshold=3.0,
)

# Build
result = build_tree_ready(values, config)

print(f"X shape: {result.X.shape}")
print(f"Missing indicators: {result.missing_indicators}")
print(f"Outliers clipped: {result.preprocessing_stats['outliers_clipped']}")
```

## Policy-Based Pipeline

Use preset transformation policies:

```python
from factor_preprocess.registry import get_default_policy_registry

# Get registry
registry = get_default_policy_registry()

# List available policies
policies = registry.all_policies()
for policy in policies:
    print(f"{policy.name}: {policy.description}")

# Get specific policy
causal_policy = registry.get("causal_basic")

print(f"Steps: {[step.name for step in causal_policy.steps]}")
print(f"Causal safe: {causal_policy.causal_safe}")
```

## Complete Workflow

```python
import numpy as np
import pandas as pd
from factor_preprocess.transforms import cs_rank, rolling_mean
from factor_preprocess.neutralization import ols_neutralize
from factor_preprocess.representation import build_linear_ready, LinearReadyConfig

# 1. Raw factors (252 days, 3000 stocks, 20 factors)
T, N, F = 252, 3000, 20
raw_values = np.random.randn(T, N, F)

# 2. Cross-sectional rank
ranked = cs_rank(raw_values, axis=1, pct=True)

# 3. Convert to long format for rolling ops
long_df = pd.DataFrame([
    {
        'date': t,
        'asset': n,
        'factor_id': f,
        'value': ranked[t, n, f],
    }
    for t in range(T)
    for n in range(N)
    for f in range(F)
])

# 4. Rolling standardization per asset
long_df['rolling_value'] = long_df.groupby(['asset', 'factor_id']).apply(
    lambda g: rolling_zscore(
        g[['date', 'value']].rename(columns={'value': 'val'}),
        window=60,
        min_periods=30,
        asset_col='asset',
        time_col='date',
        value_col='val',
    )
).reset_index(drop=True)

# 5. Neutralize exposures (simplified)
# ... neutralization step ...

# 6. Pivot back to wide
processed = long_df.pivot_table(
    index=['date', 'asset'],
    columns='factor_id',
    values='rolling_value',
).values.reshape(T, N, F)

# 7. Final representation for model
config = LinearReadyConfig(
    fill_method='zero',
    add_intercept=True,
    standardize=True,
)

# Convert to 2D (T*N, F)
flat_values = processed.reshape(-1, F)
result = build_linear_ready(flat_values, config)

print(f"Final features for model: {result.X.shape}")
```

## Tips

1. **Causality:** All rolling transforms exclude current observation (shift=1)
2. **Asset isolation:** Time-series ops are grouped by asset
3. **NaN handling:** Explicit validity, never silent fills
4. **Batch-first:** All transforms optimized for multi-factor processing
5. **Fold-local:** Fit transforms only on training data

## Next Steps

- **Full API:** See [API_REFERENCE.md](API_REFERENCE.md)
- **Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Testing:** See [TESTING.md](TESTING.md)
- **Platform Guide:** See [PLATFORM_GUIDE.md](../../PLATFORM_GUIDE.md)
