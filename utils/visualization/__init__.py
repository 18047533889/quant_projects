"""
Visualization utilities for quantitative factor analysis.

Provides comprehensive plotting tools for:
- Information Coefficient (IC) analysis
- Factor performance evaluation
- Multi-factor comparison
- Statistical diagnostics
"""

from .ic_plots import (
    plot_ic_timeseries,
    plot_rolling_ic,
    plot_ic_distribution,
    plot_ic_heatmap,
)

from .factor_plots import (
    plot_factor_distribution,
    plot_quantile_returns,
    plot_turnover_analysis,
    plot_cumulative_returns,
)

from .comparison_plots import (
    plot_multi_factor_ic,
    plot_factor_correlation_matrix,
    plot_ic_decay,
    plot_performance_comparison,
)

from .diagnostic_plots import (
    plot_residual_analysis,
    plot_qq_plot,
    plot_autocorrelation,
    plot_heteroskedasticity,
)

__all__ = [
    # IC plots
    'plot_ic_timeseries',
    'plot_rolling_ic',
    'plot_ic_distribution',
    'plot_ic_heatmap',
    # Factor plots
    'plot_factor_distribution',
    'plot_quantile_returns',
    'plot_turnover_analysis',
    'plot_cumulative_returns',
    # Comparison plots
    'plot_multi_factor_ic',
    'plot_factor_correlation_matrix',
    'plot_ic_decay',
    'plot_performance_comparison',
    # Diagnostic plots
    'plot_residual_analysis',
    'plot_qq_plot',
    'plot_autocorrelation',
    'plot_heteroskedasticity',
]
