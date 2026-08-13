"""
Factor performance visualization utilities.

Provides plotting functions for factor distributions, quantile analysis,
turnover, and cumulative returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict


def plot_factor_distribution(
    factor_values: pd.Series,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Factor Distribution",
    bins: int = 50,
    show_quantiles: bool = True,
    quantiles: List[float] = [0.1, 0.5, 0.9],
) -> plt.Figure:
    """
    Plot factor value distribution with optional quantile markers.

    Parameters
    ----------
    factor_values : pd.Series
        Factor values (cross-sectional or time-series flattened)
    figsize : tuple
        Figure size
    title : str
        Plot title
    bins : int
        Number of histogram bins
    show_quantiles : bool
        Whether to show quantile lines
    quantiles : list
        Quantile levels to display

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='#0A0D12')

    factor_clean = factor_values.dropna()

    # Histogram with KDE
    ax1.set_facecolor('#0F131C')
    ax1.hist(factor_clean, bins=bins, color='#38BDF8', alpha=0.6,
             edgecolor='#161D2B', density=True, label='Histogram')

    # KDE overlay
    from scipy import stats
    kde = stats.gaussian_kde(factor_clean)
    x_range = np.linspace(factor_clean.min(), factor_clean.max(), 300)
    ax1.plot(x_range, kde(x_range), color='#6EE7B7', linewidth=2, label='KDE')

    if show_quantiles:
        for q in quantiles:
            q_val = factor_clean.quantile(q)
            ax1.axvline(x=q_val, color='#E9A568', linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'Q{q:.0%}: {q_val:.3f}')

    ax1.set_xlabel('Factor Value', color='#E5E7EB', fontsize=11)
    ax1.set_ylabel('Density', color='#E5E7EB', fontsize=11)
    ax1.set_title(title, color='#F9FAFB', fontsize=12, pad=10)
    ax1.legend(framealpha=0.9, facecolor='#161D2B', edgecolor='#1E2636',
               labelcolor='#E5E7EB', fontsize=8)
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Q-Q plot
    ax2.set_facecolor('#0F131C')
    stats.probplot(factor_clean, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_color('#38BDF8')
    ax2.get_lines()[0].set_markersize(3)
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[1].set_color('#E9A568')
    ax2.get_lines()[1].set_linewidth(2)

    ax2.set_xlabel('Theoretical Quantiles', color='#E5E7EB', fontsize=11)
    ax2.set_ylabel('Sample Quantiles', color='#E5E7EB', fontsize=11)
    ax2.set_title('Q-Q Plot (Normal)', color='#F9FAFB', fontsize=12, pad=10)
    ax2.grid(True, alpha=0.2, color='#1E2636')
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.spines['bottom'].set_color('#1E2636')
    ax2.spines['left'].set_color('#1E2636')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Stats text
    stats_text = (
        f'Mean: {factor_clean.mean():.4f}\n'
        f'Std: {factor_clean.std():.4f}\n'
        f'Skew: {factor_clean.skew():.4f}\n'
        f'Kurt: {factor_clean.kurtosis():.4f}\n'
        f'N: {len(factor_clean):,}'
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='#161D2B', alpha=0.9,
                     edgecolor='#1E2636'),
            color='#E5E7EB', fontsize=9, family='monospace')

    plt.tight_layout()
    return fig


def plot_quantile_returns(
    quantile_returns: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Quantile Returns Analysis",
    show_spread: bool = True,
) -> plt.Figure:
    """
    Plot returns by factor quantile with mean and spread analysis.

    Parameters
    ----------
    quantile_returns : pd.DataFrame
        Returns by quantile, index=date, columns=quantile labels
    figsize : tuple
        Figure size
    title : str
        Overall plot title
    show_spread : bool
        Whether to show top-bottom spread subplot

    Returns
    -------
    plt.Figure
        The figure object
    """
    n_rows = 2 if show_spread else 1
    fig = plt.figure(figsize=figsize, facecolor='#0A0D12')
    gs = fig.add_gridspec(n_rows, 1, hspace=0.3, height_ratios=[2, 1] if show_spread else [1])

    # Cumulative returns by quantile
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0F131C')

    colors = ['#E74C3C', '#E9A568', '#9CA3AF', '#6EE7B7', '#38BDF8']
    cum_returns = (1 + quantile_returns).cumprod()

    for i, col in enumerate(quantile_returns.columns):
        color = colors[i % len(colors)]
        ax1.plot(cum_returns.index, cum_returns[col], color=color,
                linewidth=2, alpha=0.8, label=f'Q{col}')

    ax1.set_ylabel('Cumulative Return', color='#E5E7EB', fontsize=11)
    ax1.set_title(title, color='#F9FAFB', fontsize=13, pad=15)
    ax1.legend(loc='upper left', framealpha=0.9, facecolor='#161D2B',
               edgecolor='#1E2636', labelcolor='#E5E7EB', ncol=len(quantile_returns.columns))
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    if show_spread:
        # Top-bottom spread
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor('#0F131C')

        top_col = quantile_returns.columns[-1]
        bottom_col = quantile_returns.columns[0]
        spread = quantile_returns[top_col] - quantile_returns[bottom_col]
        cum_spread = (1 + spread).cumprod()

        ax2.plot(cum_spread.index, cum_spread.values, color='#38BDF8',
                linewidth=2, alpha=0.8, label='Long-Short')
        ax2.axhline(y=1, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.5)

        ax2.set_xlabel('Date', color='#E5E7EB', fontsize=11)
        ax2.set_ylabel('Cumulative Spread', color='#E5E7EB', fontsize=11)
        ax2.set_title(f'Top-Bottom Spread (Q{top_col} - Q{bottom_col})',
                     color='#F9FAFB', fontsize=11, pad=10)
        ax2.legend(loc='upper left', framealpha=0.9, facecolor='#161D2B',
                   edgecolor='#1E2636', labelcolor='#E5E7EB')
        ax2.grid(True, alpha=0.2, color='#1E2636')
        ax2.tick_params(colors='#9CA3AF', labelsize=9)
        ax2.spines['bottom'].set_color('#1E2636')
        ax2.spines['left'].set_color('#1E2636')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Add performance stats
        sharpe = spread.mean() / spread.std() * np.sqrt(252)
        stats_text = (
            f'Mean: {spread.mean()*252:.2%}\n'
            f'Std: {spread.std()*np.sqrt(252):.2%}\n'
            f'Sharpe: {sharpe:.3f}'
        )
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#161D2B', alpha=0.9,
                         edgecolor='#1E2636'),
                color='#E5E7EB', fontsize=9, family='monospace')

    plt.tight_layout()
    return fig


def plot_turnover_analysis(
    turnover: pd.Series,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Factor Turnover Analysis",
    rolling_window: int = 20,
) -> plt.Figure:
    """
    Plot factor turnover over time with rolling average.

    Parameters
    ----------
    turnover : pd.Series
        Daily turnover values (fraction of portfolio changed)
    figsize : tuple
        Figure size
    title : str
        Plot title
    rolling_window : int
        Window for rolling average

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, facecolor='#0A0D12')

    # Turnover time series
    ax1.set_facecolor('#0F131C')
    ax1.plot(turnover.index, turnover.values, color='#38BDF8',
            linewidth=1, alpha=0.5, label='Daily Turnover')

    rolling_avg = turnover.rolling(window=rolling_window, min_periods=1).mean()
    ax1.plot(rolling_avg.index, rolling_avg.values, color='#6EE7B7',
            linewidth=2, alpha=0.9, label=f'{rolling_window}D MA')

    mean_turnover = turnover.mean()
    ax1.axhline(y=mean_turnover, color='#E9A568', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'Mean: {mean_turnover:.2%}')

    ax1.set_ylabel('Turnover', color='#E5E7EB', fontsize=11)
    ax1.set_title(title, color='#F9FAFB', fontsize=12, pad=10)
    ax1.legend(loc='upper right', framealpha=0.9, facecolor='#161D2B',
               edgecolor='#1E2636', labelcolor='#E5E7EB')
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Turnover distribution
    ax2.set_facecolor('#0F131C')
    ax2.hist(turnover.dropna(), bins=50, color='#38BDF8', alpha=0.6,
            edgecolor='#161D2B', density=True)

    from scipy import stats
    kde = stats.gaussian_kde(turnover.dropna())
    x_range = np.linspace(turnover.min(), turnover.max(), 200)
    ax2.plot(x_range, kde(x_range), color='#6EE7B7', linewidth=2, label='KDE')

    ax2.axvline(x=mean_turnover, color='#E9A568', linestyle='--',
               linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('Turnover', color='#E5E7EB', fontsize=11)
    ax2.set_ylabel('Density', color='#E5E7EB', fontsize=11)
    ax2.set_title('Turnover Distribution', color='#F9FAFB', fontsize=11, pad=10)
    ax2.legend(framealpha=0.9, facecolor='#161D2B', edgecolor='#1E2636',
               labelcolor='#E5E7EB')
    ax2.grid(True, alpha=0.2, color='#1E2636')
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.spines['bottom'].set_color('#1E2636')
    ax2.spines['left'].set_color('#1E2636')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Stats
    stats_text = (
        f'Mean: {turnover.mean():.2%}\n'
        f'Median: {turnover.median():.2%}\n'
        f'Std: {turnover.std():.2%}\n'
        f'Min: {turnover.min():.2%}\n'
        f'Max: {turnover.max():.2%}'
    )
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#161D2B', alpha=0.9,
                     edgecolor='#1E2636'),
            color='#E5E7EB', fontsize=9, family='monospace')

    plt.tight_layout()
    return fig


def plot_cumulative_returns(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (14, 7),
    title: str = "Cumulative Returns",
    show_drawdown: bool = True,
) -> plt.Figure:
    """
    Plot cumulative returns with optional benchmark and drawdown.

    Parameters
    ----------
    returns : pd.Series
        Period returns
    benchmark : pd.Series, optional
        Benchmark returns with same index
    figsize : tuple
        Figure size
    title : str
        Plot title
    show_drawdown : bool
        Whether to show drawdown subplot

    Returns
    -------
    plt.Figure
        The figure object
    """
    n_rows = 2 if show_drawdown else 1
    fig = plt.figure(figsize=figsize, facecolor='#0A0D12')
    gs = fig.add_gridspec(n_rows, 1, hspace=0.3, height_ratios=[2, 1] if show_drawdown else [1])

    # Cumulative returns
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0F131C')

    cum_returns = (1 + returns).cumprod()
    ax1.plot(cum_returns.index, cum_returns.values, color='#38BDF8',
            linewidth=2, alpha=0.9, label='Factor')

    if benchmark is not None:
        cum_benchmark = (1 + benchmark).cumprod()
        ax1.plot(cum_benchmark.index, cum_benchmark.values, color='#E9A568',
                linewidth=2, alpha=0.8, label='Benchmark')

    ax1.set_ylabel('Cumulative Return', color='#E5E7EB', fontsize=11)
    ax1.set_title(title, color='#F9FAFB', fontsize=13, pad=15)
    ax1.legend(loc='upper left', framealpha=0.9, facecolor='#161D2B',
               edgecolor='#1E2636', labelcolor='#E5E7EB')
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Performance stats
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    total_ret = cum_returns.iloc[-1] - 1

    stats_text = (
        f'Total: {total_ret:.2%}\n'
        f'Annual: {annual_ret:.2%}\n'
        f'Vol: {annual_vol:.2%}\n'
        f'Sharpe: {sharpe:.3f}'
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='#161D2B', alpha=0.9,
                     edgecolor='#1E2636'),
            color='#E5E7EB', fontsize=9, family='monospace')

    if show_drawdown:
        # Drawdown
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor('#0F131C')

        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max

        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        color='#E74C3C', alpha=0.5)
        ax2.plot(drawdown.index, drawdown.values, color='#E74C3C',
                linewidth=1, alpha=0.8)

        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax2.axhline(y=max_dd, color='#E9A568', linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'Max DD: {max_dd:.2%}')

        ax2.set_xlabel('Date', color='#E5E7EB', fontsize=11)
        ax2.set_ylabel('Drawdown', color='#E5E7EB', fontsize=11)
        ax2.set_title('Drawdown', color='#F9FAFB', fontsize=11, pad=10)
        ax2.legend(loc='lower left', framealpha=0.9, facecolor='#161D2B',
                   edgecolor='#1E2636', labelcolor='#E5E7EB')
        ax2.grid(True, alpha=0.2, color='#1E2636')
        ax2.tick_params(colors='#9CA3AF', labelsize=9)
        ax2.spines['bottom'].set_color('#1E2636')
        ax2.spines['left'].set_color('#1E2636')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig
