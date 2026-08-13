# Regime-Adaptive Factor Preprocessing

Online regime detection and adaptive factor transformations for quantitative research.

## Modules

### detector.py
Online regime detection using rolling statistics:
- `detect_variance_regime()` - volatility-based regimes (low/high vol)
- `detect_correlation_regime()` - correlation-based regimes (low/high correlation)

Both functions:
- Use causal rolling windows (shift(1) excludes current observation)
- Return `RegimeState` with regime labels, strength, and transition flags
- Support multi-regime classification (2+)
- Compute cross-sectional percentiles for regime assignment

### adaptive_weights.py
Regime-dependent factor weighting:
- `fit_regime_weights()` - learns weights per regime using historical data
- `regime_adaptive_weights()` - applies learned weights based on current regime

Weighting methods:
- `equal` - uniform weights (1/n_factors)
- `volatility_inverse` - inverse volatility weighting (w ∝ 1/σ)
- `sharpe` - Sharpe ratio weighting (w ∝ μ/σ, requires target)

### switching.py
Regime-based transform switching:
- `fit_regime_switching()` - learns transform parameters per regime
- `regime_switching_transform()` - applies regime-specific transforms

Transform types:
- `zscore` - standardization using regime-specific mean/std
- `winsor` - regime-specific winsorization bounds (5th/95th percentiles)
- `scale` - regime-specific scaling by std (zero-centered)
- `rank` - cross-sectional percentile rank (no regime-specific params)
- `none` - pass-through (identity)

## Usage

```python
import pandas as pd
from factor_preprocess.regime import (
    detect_variance_regime,
    fit_regime_weights,
    regime_adaptive_weights,
    fit_regime_switching,
    regime_switching_transform,
)

# Detect volatility regimes
regime_state = detect_variance_regime(
    df,  # Must have: asset_id, date, value columns
    window=20,
    n_regimes=2
)

# Split train/test for fold-safety
train_df = df[df['date'] < split_date]
test_df = df[df['date'] >= split_date]
train_regimes = regime_state.regime[df['date'] < split_date]
test_regimes = regime_state.regime[df['date'] >= split_date]

# Learn regime-adaptive weights on training data
weight_state = fit_regime_weights(
    train_df,
    regime_labels=train_regimes,
    factor_cols=['factor1', 'factor2', 'factor3'],
    method='sharpe',
    target=train_df['returns'],  # Required for sharpe method
)

# Apply weights to test data
weighted_df = regime_adaptive_weights(
    test_df,
    regime_labels=test_regimes,
    fitted_state=weight_state,
    factor_cols=['factor1', 'factor2', 'factor3']
)

# Learn regime-specific transforms on training data
transform_state = fit_regime_switching(
    train_df[['date', 'asset_id', 'factor1']].rename(columns={'factor1': 'value'}),
    regime_labels=train_regimes,
    transform_type='zscore',
)

# Apply transforms to test data
transformed = regime_switching_transform(
    test_df[['date', 'asset_id', 'factor1']].rename(columns={'factor1': 'value'}),
    regime_labels=test_regimes,
    fitted_state=transform_state
)
```

## Key Properties

- **Causal**: All operations use shift(1) to prevent lookahead bias
- **Fold-safe**: fit/apply split with explicit fit_window prevents data leakage
- **Fail-closed**: NaN inputs or unknown regimes produce NaN outputs
- **Staleness detection**: Optional checks to ensure regime labels match fitted state
- **Cross-sectional**: Regime detection uses cross-sectional percentiles at each time point

## Test Coverage

48 tests covering:
- Causality (no future leakage)
- Fold safety (fit/apply separation)
- Edge cases (insufficient data, NaN handling, unknown regimes)
- Multi-asset/multi-factor scenarios
- Staleness detection
