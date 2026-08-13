# Quantitative Factor Visualization Utilities

Comprehensive visualization toolkit for quantitative factor analysis using matplotlib and seaborn.

## Overview

This package provides production-ready plotting functions for:

- **IC Analysis**: Time series, rolling statistics, distributions, heatmaps
- **Factor Performance**: Distributions, quantile returns, turnover, cumulative returns
- **Multi-Factor Comparison**: IC comparison, correlation matrices, IC decay, performance benchmarking
- **Statistical Diagnostics**: Residual analysis, Q-Q plots, autocorrelation, heteroskedasticity

## Installation

Dependencies (add to requirements.txt):
```txt
matplotlib>=3.5.0
seaborn>=0.12.0
scipy>=1.9.0
```

## Quick Start

```python
import pandas as pd
from utils.visualization import (
    plot_ic_timeseries,
    plot_quantile_returns,
    plot_multi_factor_ic,
    plot_residual_analysis,
)

# IC time series
ic_series = pd.Series(...)  # Your IC data with datetime index
fig = plot_ic_timeseries(ic_series, show_mean=True, show_std_bands=True)
fig.savefig('ic_analysis.png', dpi=150, bbox_inches='tight')

# Quantile returns
quantile_returns = pd.DataFrame(...)  # Returns by quantile
fig = plot_quantile_returns(quantile_returns, show_spread=True)
fig.savefig('quantile_returns.png', dpi=150, bbox_inches='tight')

# Multi-factor comparison
ic_df = pd.DataFrame(...)  # Multiple factors as columns
fig = plot_multi_factor_ic(ic_df, rolling_window=20)
fig.savefig('multi_factor_ic.png', dpi=150, bbox_inches='tight')
```

## Module Structure

```
utils/visualization/
├── __init__.py              # Package exports
├── ic_plots.py             # IC analysis plots
├── factor_plots.py         # Factor performance plots
├── comparison_plots.py     # Multi-factor comparison plots
├── diagnostic_plots.py     # Statistical diagnostic plots
├── examples.py             # Usage examples with synthetic data
└── README.md               # This file
```

## API Reference

### IC Plots (`ic_plots.py`)

#### `plot_ic_timeseries(ic_series, ...)`
Plot IC time series with optional mean line and standard deviation bands.

**Parameters:**
- `ic_series`: pd.Series - Time series of IC values with datetime index
- `figsize`: tuple - Figure size (default: (14, 6))
- `title`: str - Plot title
- `show_mean`: bool - Show mean IC line (default: True)
- `show_std_bands`: bool - Show ±1 std bands (default: True)
- `ax`: plt.Axes - Existing axes (optional)

**Returns:** plt.Axes

#### `plot_rolling_ic(ic_series, windows=[20, 60, 120], ...)`
Plot rolling IC mean and standard deviation for multiple windows.

**Parameters:**
- `ic_series`: pd.Series - IC values
- `windows`: list - Rolling window sizes in trading days
- `figsize`: tuple - Figure size (default: (14, 8))
- `title`: str - Overall figure title

**Returns:** plt.Figure

#### `plot_ic_distribution(ic_series, bins=50, ...)`
Plot IC distribution with histogram, KDE, and statistical summary.

**Parameters:**
- `ic_series`: pd.Series - IC values
- `figsize`: tuple - Figure size (default: (12, 6))
- `bins`: int - Number of histogram bins
- `title`: str - Plot title
- `show_stats`: bool - Show statistical annotations (default: True)

**Returns:** plt.Figure

#### `plot_ic_heatmap(ic_df, ...)`
Plot heatmap of IC values across multiple factors and time periods.

**Parameters:**
- `ic_df`: pd.DataFrame - Factors as columns, dates as index
- `figsize`: tuple - Figure size (default: (12, 8))
- `title`: str - Plot title
- `annot`: bool - Annotate cells with values (default: False)
- `cmap`: str - Colormap name (default: 'RdBu_r')
- `vmin`, `vmax`: float - Color scale limits

**Returns:** plt.Figure

---

### Factor Plots (`factor_plots.py`)

#### `plot_factor_distribution(factor_values, bins=50, ...)`
Plot factor value distribution with optional quantile markers.

**Parameters:**
- `factor_values`: pd.Series - Factor values (cross-sectional or flattened)
- `figsize`: tuple - Figure size (default: (12, 6))
- `title`: str - Plot title
- `bins`: int - Number of histogram bins
- `show_quantiles`: bool - Show quantile lines (default: True)
- `quantiles`: list - Quantile levels to display (default: [0.1, 0.5, 0.9])

**Returns:** plt.Figure

#### `plot_quantile_returns(quantile_returns, show_spread=True, ...)`
Plot returns by factor quantile with mean and spread analysis.

**Parameters:**
- `quantile_returns`: pd.DataFrame - Returns by quantile (index=date, columns=quantile)
- `figsize`: tuple - Figure size (default: (14, 8))
- `title`: str - Overall plot title
- `show_spread`: bool - Show top-bottom spread subplot (default: True)

**Returns:** plt.Figure

#### `plot_turnover_analysis(turnover, rolling_window=20, ...)`
Plot factor turnover over time with rolling average.

**Parameters:**
- `turnover`: pd.Series - Daily turnover values (fraction of portfolio changed)
- `figsize`: tuple - Figure size (default: (14, 6))
- `title`: str - Plot title
- `rolling_window`: int - Window for rolling average

**Returns:** plt.Figure

#### `plot_cumulative_returns(returns, benchmark=None, show_drawdown=True, ...)`
Plot cumulative returns with optional benchmark and drawdown.

**Parameters:**
- `returns`: pd.Series - Period returns
- `benchmark`: pd.Series - Benchmark returns with same index (optional)
- `figsize`: tuple - Figure size (default: (14, 7))
- `title`: str - Plot title
- `show_drawdown`: bool - Show drawdown subplot (default: True)

**Returns:** plt.Figure

---

### Comparison Plots (`comparison_plots.py`)

#### `plot_multi_factor_ic(ic_df, rolling_window=20, ...)`
Plot IC time series for multiple factors with rolling averages.

**Parameters:**
- `ic_df`: pd.DataFrame - IC values with factors as columns, dates as index
- `figsize`: tuple - Figure size (default: (14, 8))
- `title`: str - Plot title
- `show_mean`: bool - Show mean IC for each factor (default: True)
- `rolling_window`: int - Window for rolling average (optional)

**Returns:** plt.Figure

#### `plot_factor_correlation_matrix(factor_df, method='pearson', ...)`
Plot correlation matrix heatmap for multiple factors.

**Parameters:**
- `factor_df`: pd.DataFrame - Factor values with factors as columns
- `figsize`: tuple - Figure size (default: (10, 8))
- `title`: str - Plot title
- `method`: str - Correlation method ('pearson', 'spearman', 'kendall')
- `annot`: bool - Annotate cells (default: True)

**Returns:** plt.Figure

#### `plot_ic_decay(ic_decay_df, lags=None, ...)`
Plot IC decay over multiple forward periods for multiple factors.

**Parameters:**
- `ic_decay_df`: pd.DataFrame - IC values by lag period (factors as columns, lags as index)
- `figsize`: tuple - Figure size (default: (12, 6))
- `title`: str - Plot title
- `lags`: list - Specific lag periods to highlight (optional)

**Returns:** plt.Figure

#### `plot_performance_comparison(performance_df, metrics=[...], ...)`
Plot comprehensive performance comparison across multiple factors.

**Parameters:**
- `performance_df`: pd.DataFrame - Performance metrics (factors as index, metrics as columns)
- `metrics`: list - Specific metrics to display (default: ['IC_mean', 'IR', 'Sharpe', 'Max_DD'])
- `figsize`: tuple - Figure size (default: (14, 10))
- `title`: str - Overall title

**Returns:** plt.Figure

---

### Diagnostic Plots (`diagnostic_plots.py`)

#### `plot_residual_analysis(residuals, fitted_values=None, ...)`
Comprehensive residual diagnostic plots.

**Parameters:**
- `residuals`: pd.Series - Model residuals
- `fitted_values`: pd.Series - Fitted values for residual vs fitted plot (optional)
- `figsize`: tuple - Figure size (default: (14, 10))
- `title`: str - Overall title

**Returns:** plt.Figure

**Subplots:**
1. Residuals vs Fitted (with LOWESS smoothing)
2. Normal Q-Q Plot
3. Scale-Location (heteroskedasticity)
4. Residual distribution with normality tests

#### `plot_qq_plot(data, distribution='norm', ...)`
Quantile-Quantile plot against theoretical distribution.

**Parameters:**
- `data`: pd.Series - Data to test
- `distribution`: str - Distribution name ('norm', 't', 'uniform', etc.)
- `figsize`: tuple - Figure size (default: (8, 8))
- `title`: str - Plot title

**Returns:** plt.Figure

#### `plot_autocorrelation(data, lags=40, alpha=0.05, ...)`
Plot ACF and PACF for time series data.

**Parameters:**
- `data`: pd.Series - Time series data
- `lags`: int - Number of lags to plot (default: 40)
- `figsize`: tuple - Figure size (default: (14, 6))
- `title`: str - Overall title
- `alpha`: float - Significance level for confidence intervals (default: 0.05)

**Returns:** plt.Figure

#### `plot_heteroskedasticity(residuals, fitted_values, ...)`
Analyze heteroskedasticity in residuals.

**Parameters:**
- `residuals`: pd.Series - Model residuals
- `fitted_values`: pd.Series - Fitted values
- `figsize`: tuple - Figure size (default: (14, 6))
- `title`: str - Plot title

**Returns:** plt.Figure

**Subplots:**
1. Squared residuals vs fitted (with LOWESS trend)
2. Absolute residuals vs fitted

---

## Design System

All plots use a consistent dark theme with carefully chosen colors:

### Color Palette
- **Primary**: `#38BDF8` (cyan) - Main data series
- **Secondary**: `#6EE7B7` (emerald) - Supporting elements
- **Accent**: `#E9A568` (amber) - Highlights and warnings
- **Additional**: `#3B6DFF`, `#F472B6`, `#A78BFA`, `#FCD34D`, `#34D399`

### Surface Levels
- Background: `#0A0D12` (deepest)
- Canvas: `#0F131C`
- Panel: `#161D2B`
- Border: `#1E2636`

### Typography
- Title: `#F9FAFB` (near white)
- Labels: `#E5E7EB` (light gray)
- Annotations: `#9CA3AF` (medium gray)

### Best Practices
- All plots support dark backgrounds
- Consistent grid styling (alpha=0.2)
- Rounded legend boxes with transparency
- No top/right spines
- Monospace font for statistics
- High DPI recommended (150+)

---

## Usage Examples

### Example 1: Basic IC Analysis
```python
import pandas as pd
from utils.visualization import plot_ic_timeseries, plot_ic_distribution

# Load your IC data
ic_series = pd.Series(...)

# Time series plot
fig = plot_ic_timeseries(ic_series)
fig.savefig('ic_ts.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')

# Distribution analysis
fig = plot_ic_distribution(ic_series, bins=60, show_stats=True)
fig.savefig('ic_dist.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')
```

### Example 2: Factor Performance Analysis
```python
from utils.visualization import plot_quantile_returns, plot_turnover_analysis

# Quantile returns (DataFrame with Q1, Q2, ..., Q5 columns)
quantile_returns = pd.DataFrame(...)
fig = plot_quantile_returns(quantile_returns, show_spread=True)
fig.savefig('quantiles.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')

# Turnover analysis
turnover = pd.Series(...)
fig = plot_turnover_analysis(turnover, rolling_window=20)
fig.savefig('turnover.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')
```

### Example 3: Multi-Factor Comparison
```python
from utils.visualization import plot_multi_factor_ic, plot_factor_correlation_matrix

# Multiple factors (DataFrame with factor columns)
ic_df = pd.DataFrame({
    'Momentum': [...],
    'Value': [...],
    'Quality': [...],
})

# IC comparison
fig = plot_multi_factor_ic(ic_df, rolling_window=20)
fig.savefig('multi_ic.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')

# Correlation matrix
factor_values = pd.DataFrame(...)  # Cross-sectional factor values
fig = plot_factor_correlation_matrix(factor_values, method='spearman')
fig.savefig('corr.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')
```

### Example 4: Model Diagnostics
```python
from utils.visualization import plot_residual_analysis, plot_autocorrelation

# After model fitting
residuals = pd.Series(...)
fitted_values = pd.Series(...)

# Comprehensive residual analysis
fig = plot_residual_analysis(residuals, fitted_values)
fig.savefig('diagnostics.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')

# Check for autocorrelation
fig = plot_autocorrelation(residuals, lags=40)
fig.savefig('acf.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')
```

---

## Running Examples

Generate all example plots with synthetic data:

```bash
cd /home/shw/quant_projects
python -m utils.visualization.examples
```

This will create 16 example plots in `examples/` directory demonstrating all functions.

---

## Integration with Factor Engine

```python
# Example: Visualize factor engine results
from factor_engine.evaluation import evaluate_factor
from utils.visualization import plot_ic_timeseries, plot_quantile_returns

# Evaluate factor
results = evaluate_factor(factor_data, returns_data)

# Plot IC
fig = plot_ic_timeseries(results['ic_series'])
fig.savefig('factor_ic.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')

# Plot quantile performance
fig = plot_quantile_returns(results['quantile_returns'])
fig.savefig('factor_quantiles.png', dpi=150, bbox_inches='tight', facecolor='#0A0D12')
```

---

## Notes

- All functions return matplotlib Figure or Axes objects for further customization
- Plots are designed for dark backgrounds; use `facecolor='#0A0D12'` when saving
- Recommended DPI: 150+ for high-quality output
- All plots handle missing values (NaN) gracefully via `.dropna()`
- For large datasets, consider downsampling time series before plotting
- LOWESS smoothing requires `statsmodels` (optional dependency)

---

## License

Part of the quant_projects platform.
