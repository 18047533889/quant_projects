# FactorPreprocess Architecture

**Version:** 0.1.0  
**Last Updated:** 2026-08-14

## Overview

FactorPreprocess is the model-input preparation layer, operating between factor selection (FactorAssets) and model training. It transforms selected factors into model-ready features with explicit causality guarantees.

## Design Principles

1. **Batch-First API**: Optimized for multi-factor processing
2. **Explicit Causality**: All rolling transforms exclude current observation (shift=1)
3. **Fold-Local Fitting**: No full-sample statistics before train/test split
4. **Asset Isolation**: Time-series operations grouped per asset
5. **NaN Preservation**: No silent fills, explicit validity tracking

## Module Organization

```
factor_preprocess/
├── contracts/              # Core data contracts
│   ├── policy.py          # PreprocessingPolicy, TransformSpec
│   ├── state.py           # FittedState
│   └── feature_bundle.py  # FeatureBundle
├── transforms/            # Stateless transforms
│   ├── cross_sectional.py # cs_rank, cs_zscore, cs_winsor, etc.
│   ├── rolling.py         # rolling_mean, rolling_std, ewma
│   ├── volatility.py      # volatility_scale, realized_volatility
│   ├── missingness.py     # forward_fill, missing_indicator
│   └── freshness.py       # days_since_update, freshness_score
├── neutralization/        # Factor neutralization
│   ├── ols.py            # OLS residuals
│   ├── regularized.py    # Ridge, Lasso, Elastic Net
│   └── diagnostics.py    # Condition number, exposure checks
├── representation/        # Model-specific builders
│   ├── multichannel.py   # Multi-channel features
│   ├── linear_ready.py   # Linear model preparation
│   └── tree_ready.py     # Tree model preparation
└── registry/             # Transform & policy catalogs
    ├── transforms.py
    └── policies.py
```

## Transform Categories

### Cross-Sectional (5 functions)
Per-time transformations: rank, zscore, demean, winsor, scale.

### Rolling/Temporal (4 functions)
Causal time-series: rolling_mean, rolling_std, rolling_zscore, ewma.
**Key:** All use shift(1) to exclude current observation.

### Volatility (3 functions)
Realized volatility scaling for risk-adjusted signals.

### Missingness (4 functions)
Explicit missing data handling: forward_fill, indicators, run lengths.

### Freshness (4 functions)
Data age tracking: days_since_update, observation_age, freshness_score.

### Neutralization (5 functions)
Remove systematic exposures: OLS, Ridge, Lasso, Elastic Net.

### Representation (3 builders)
Model-specific preparation: multichannel, linear_ready, tree_ready.

## Causality Guarantees

**Rolling Transforms:**
```python
# CORRECT (causal)
rolling_mean(df, window=20, ...)  # Uses shift(1) internally

# WRONG (would leak future)
df.rolling(20).mean()  # Includes current observation!
```

**Fold-Local Fitting:**
```python
# CORRECT
fit_state = fit_transform(X_train)  # Fit only on training
X_test_transformed = apply_transform(X_test, fit_state)

# WRONG
X_all_transformed = fit_transform(X_all)  # Leak from test set!
```

## Integration Boundaries

**FactorAssets (FA):**
- FA selects which factors to use
- FP transforms selected factors
- FA does not preprocess

**Model Training (External):**
- FP produces FeatureBundle
- Model trains on features
- FP does not train models

---

**Last Updated:** 2026-08-14
