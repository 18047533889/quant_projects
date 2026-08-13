# Advanced Neutralization Methods

This directory contains advanced neutralization techniques for factor preprocessing.

## Modules

### 1. `pca_neutralization.py`
**PCA-based neutralization** using principal component analysis.

- Extracts orthogonal exposure factors via PCA
- Neutralizes against top-k principal components
- Handles highly collinear exposures
- Options for variance threshold or fixed component count
- Fold-safe: per-date operation prevents time-series leakage

**Key function**: `pca_neutralize()`

### 2. `robust_regression.py`
**Robust regression methods** for outlier resistance.

#### Huber Regression
- Uses Huber loss (quadratic for small residuals, linear for large)
- 95% efficiency vs OLS with delta=1.35
- Moderate outlier robustness
- Implemented via iteratively reweighted least squares (IRLS)

#### LAD Regression
- Least Absolute Deviations (L1 regression)
- Equivalent to median regression (quantile=0.5)
- Highly robust to outliers (50% breakdown point)
- Minimizes sum of absolute residuals

**Key functions**: `huber_neutralize()`, `lad_neutralize()`

### 3. `quantile_regression.py`
**Quantile regression neutralization** at specified quantiles.

- Neutralize at any quantile (median, upper/lower tail)
- Useful for asymmetric exposure effects
- IRLS implementation with asymmetric weights
- Robust to outliers at all quantiles

**Key function**: `quantile_neutralize()`

### 4. `kernel_regression.py`
**Non-parametric kernel regression** for non-linear relationships.

- Handles non-linear exposure-factor relationships
- Three kernel types: Gaussian, Epanechnikov, Tricube
- Local constant (Nadaraya-Watson) or local linear regression
- Automatic bandwidth selection via Scott's rule
- Fold-safe: per-date operation

**Key function**: `kernel_neutralize()`

## Usage Example

```python
from factor_preprocess.neutralization.advanced import (
    pca_neutralize,
    huber_neutralize,
    lad_neutralize,
    quantile_neutralize,
    kernel_neutralize,
)

# PCA neutralization (handles collinearity)
residuals = pca_neutralize(
    values_df, exposures_df,
    n_components=5,  # or use variance_threshold=0.95
    min_observations=10
)

# Huber regression (moderate outlier robustness)
residuals = huber_neutralize(
    values_df, exposures_df,
    delta=1.35,  # Huber threshold
    min_observations=10
)

# LAD regression (high outlier robustness)
residuals = lad_neutralize(
    values_df, exposures_df,
    min_observations=10
)

# Quantile regression (neutralize at 75th percentile)
residuals = quantile_neutralize(
    values_df, exposures_df,
    quantile=0.75,
    min_observations=10
)

# Kernel regression (non-linear relationships)
residuals = kernel_neutralize(
    values_df, exposures_df,
    kernel="gaussian",
    bandwidth=1.0,  # or None for auto
    local_constant=True,
    min_observations=10
)
```

## When to Use Each Method

- **PCA**: When exposures are highly correlated or you want dimensionality reduction
- **Huber**: When you need outlier robustness but want to maintain efficiency close to OLS
- **LAD**: When you need maximum outlier robustness (extreme outliers present)
- **Quantile**: When exposure effects vary across the factor distribution
- **Kernel**: When exposure-factor relationships are non-linear

## Fold Safety

All methods operate per-date (cross-sectionally) to prevent time-series leakage:
- PCA decomposition fitted separately for each date
- Robust regression fitted independently per date
- Kernel weights computed within each date's cross-section
- No information from future dates used in neutralization

## Testing

Run tests:
```bash
pytest tests/neutralization/advanced/ -v
```

All 43 tests pass with full coverage of edge cases, outlier scenarios, and fold safety.
