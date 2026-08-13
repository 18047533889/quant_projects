"""
Information Coefficient (IC) visualization utilities.

Provides plotting functions for IC analysis including time series,
rolling statistics, distributions, and heatmaps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any


def plot_ic_timeseries(
    ic_series: pd.Series,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Information Coefficient Time Series",
    show_mean: bool = True,
    show_std_bands: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot IC time series with optional mean line and standard deviation bands.

    Parameters
    ----------
    ic_series : pd.Series
        Time series of IC values with datetime index
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    show_mean : bool
        Whether to show mean IC line
    show_std_bands : bool
        Whether to show ±1 std bands
    ax : plt.Axes, optional
        Existing axes to plot on

    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0A0D12')
        ax.set_facecolor('#0F131C')

    # Plot IC series
    ax.plot(ic_series.index, ic_series.values,
            color='#38BDF8', linewidth=1.2, alpha=0.8, label='IC')

    # Add zero line
    ax.axhline(y=0, color='#E9A568', linestyle='--',
               linewidth=1, alpha=0.5, label='Zero')

    if show_mean:
        mean_ic = ic_series.mean()
        ax.axhline(y=mean_ic, color='#6EE7B7', linestyle='-',
                   linewidth=1.5, alpha=0.7, label=f'Mean IC: {mean_ic:.4f}')

    if show_std_bands:
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ax.fill_between(ic_series.index,
                        mean_ic - std_ic, mean_ic + std_ic,
                        color='#6EE7B7', alpha=0.15, label='±1 Std')

    ax.set_xlabel('Date', color='#E5E7EB', fontsize=11)
    ax.set_ylabel('IC', color='#E5E7EB', fontsize=11)
    ax.set_title(title, color='#F9FAFB', fontsize=13, pad=15)
    ax.legend(loc='upper left', framealpha=0.9, facecolor='#161D2B',
              edgecolor='#1E2636', labelcolor='#E5E7EB')
    ax.grid(True, alpha=0.2, color='#1E2636', linestyle='-', linewidth=0.5)
    ax.tick_params(colors='#9CA3AF', labelsize=9)
    ax.spines['bottom'].set_color('#1E2636')
    ax.spines['left'].set_color('#1E2636')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return ax


def plot_rolling_ic(
    ic_series: pd.Series,
    windows: list = [20, 60, 120],
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Rolling IC Statistics",
) -> plt.Figure:
    """
    Plot rolling IC mean and standard deviation for multiple windows.

    Parameters
    ----------
    ic_series : pd.Series
        Time series of IC values
    windows : list
        List of rolling window sizes (in trading days)
    figsize : tuple
        Figure size
    title : str
        Overall figure title

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig = plt.figure(figsize=figsize, facecolor='#0A0D12')
    gs = fig.add_gridspec(2, 1, hspace=0.3)

    colors = ['#38BDF8', '#6EE7B7', '#E9A568']

    # Rolling mean
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0F131C')
    for i, window in enumerate(windows):
        rolling_mean = ic_series.rolling(window=window, min_periods=1).mean()
        ax1.plot(rolling_mean.index, rolling_mean.values,
                color=colors[i % len(colors)], linewidth=1.5,
                alpha=0.8, label=f'{window}D MA')

    ax1.axhline(y=0, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Rolling Mean IC', color='#E5E7EB', fontsize=11)
    ax1.set_title(f'{title} - Mean', color='#F9FAFB', fontsize=12, pad=10)
    ax1.legend(loc='upper left', framealpha=0.9, facecolor='#161D2B',
               edgecolor='#1E2636', labelcolor='#E5E7EB')
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Rolling std
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0F131C')
    for i, window in enumerate(windows):
        rolling_std = ic_series.rolling(window=window, min_periods=1).std()
        ax2.plot(rolling_std.index, rolling_std.values,
                color=colors[i % len(colors)], linewidth=1.5,
                alpha=0.8, label=f'{window}D Std')

    ax2.set_xlabel('Date', color='#E5E7EB', fontsize=11)
    ax2.set_ylabel('Rolling Std IC', color='#E5E7EB', fontsize=11)
    ax2.set_title(f'{title} - Standard Deviation', color='#F9FAFB',
                  fontsize=12, pad=10)
    ax2.legend(loc='upper left', framealpha=0.9, facecolor='#161D2B',
               edgecolor='#1E2636', labelcolor='#E5E7EB')
    ax2.grid(True, alpha=0.2, color='#1E2636')
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.spines['bottom'].set_color('#1E2636')
    ax2.spines['left'].set_color('#1E2636')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_ic_distribution(
    ic_series: pd.Series,
    figsize: Tuple[int, int] = (12, 6),
    bins: int = 50,
    title: str = "IC Distribution",
    show_stats: bool = True,
) -> plt.Figure:
    """
    Plot IC distribution with histogram and KDE, plus statistical summary.

    Parameters
    ----------
    ic_series : pd.Series
        Time series of IC values
    figsize : tuple
        Figure size
    bins : int
        Number of histogram bins
    title : str
        Plot title
    show_stats : bool
        Whether to show statistical annotations

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='#0A0D12')

    # Histogram with KDE
    ax1.set_facecolor('#0F131C')
    ax1.hist(ic_series.dropna(), bins=bins, color='#38BDF8',
             alpha=0.6, edgecolor='#161D2B', density=True, label='Histogram')

    # KDE
    from scipy import stats
    kde = stats.gaussian_kde(ic_series.dropna())
    x_range = np.linspace(ic_series.min(), ic_series.max(), 200)
    ax1.plot(x_range, kde(x_range), color='#6EE7B7',
             linewidth=2, label='KDE')

    # Normal distribution overlay
    mu, sigma = ic_series.mean(), ic_series.std()
    normal_dist = stats.norm.pdf(x_range, mu, sigma)
    ax1.plot(x_range, normal_dist, color='#E9A568',
             linewidth=2, linestyle='--', alpha=0.7, label='Normal')

    ax1.axvline(x=0, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('IC Value', color='#E5E7EB', fontsize=11)
    ax1.set_ylabel('Density', color='#E5E7EB', fontsize=11)
    ax1.set_title(title, color='#F9FAFB', fontsize=12, pad=10)
    ax1.legend(framealpha=0.9, facecolor='#161D2B',
               edgecolor='#1E2636', labelcolor='#E5E7EB')
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Box plot
    ax2.set_facecolor('#0F131C')
    bp = ax2.boxplot([ic_series.dropna()], vert=True, patch_artist=True,
                      widths=0.5, showfliers=True,
                      flierprops=dict(marker='o', markerfacecolor='#E9A568',
                                     markersize=4, alpha=0.5),
                      boxprops=dict(facecolor='#38BDF8', alpha=0.6,
                                   edgecolor='#6EE7B7', linewidth=1.5),
                      medianprops=dict(color='#6EE7B7', linewidth=2),
                      whiskerprops=dict(color='#9CA3AF', linewidth=1.5),
                      capprops=dict(color='#9CA3AF', linewidth=1.5))

    ax2.axhline(y=0, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('IC Value', color='#E5E7EB', fontsize=11)
    ax2.set_title('IC Box Plot', color='#F9FAFB', fontsize=12, pad=10)
    ax2.set_xticklabels(['IC'], color='#E5E7EB')
    ax2.grid(True, alpha=0.2, color='#1E2636', axis='y')
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.spines['bottom'].set_color('#1E2636')
    ax2.spines['left'].set_color('#1E2636')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    if show_stats:
        stats_text = (
            f'Mean: {ic_series.mean():.4f}\n'
            f'Std: {ic_series.std():.4f}\n'
            f'Skew: {ic_series.skew():.4f}\n'
            f'Kurt: {ic_series.kurtosis():.4f}\n'
            f'IR: {ic_series.mean() / ic_series.std():.4f}'
        )
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#161D2B',
                         alpha=0.9, edgecolor='#1E2636'),
                color='#E5E7EB', fontsize=9, family='monospace')

    plt.tight_layout()
    return fig


def plot_ic_heatmap(
    ic_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "IC Heatmap Across Factors",
    annot: bool = False,
    cmap: str = 'RdBu_r',
    vmin: float = -0.1,
    vmax: float = 0.1,
) -> plt.Figure:
    """
    Plot heatmap of IC values across multiple factors and time periods.

    Parameters
    ----------
    ic_df : pd.DataFrame
        DataFrame with factors as columns and dates as index
    figsize : tuple
        Figure size
    title : str
        Plot title
    annot : bool
        Whether to annotate cells with values
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0A0D12')
    ax.set_facecolor('#0F131C')

    # Create heatmap
    sns.heatmap(ic_df.T, cmap=cmap, center=0, vmin=vmin, vmax=vmax,
                annot=annot, fmt='.3f', linewidths=0.5, linecolor='#0A0D12',
                cbar_kws={'label': 'IC', 'shrink': 0.8}, ax=ax)

    ax.set_xlabel('Date', color='#E5E7EB', fontsize=11)
    ax.set_ylabel('Factor', color='#E5E7EB', fontsize=11)
    ax.set_title(title, color='#F9FAFB', fontsize=13, pad=15)
    ax.tick_params(colors='#9CA3AF', labelsize=9)

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#9CA3AF', labelsize=9)
    cbar.ax.yaxis.label.set_color('#E5E7EB')
    cbar.outline.set_edgecolor('#1E2636')

    plt.tight_layout()
    return fig
