"""
Multi-factor comparison visualization utilities.

Provides plotting functions for comparing multiple factors,
correlation analysis, IC decay, and performance benchmarking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict


def plot_multi_factor_ic(
    ic_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Multi-Factor IC Comparison",
    show_mean: bool = True,
    rolling_window: Optional[int] = 20,
) -> plt.Figure:
    """
    Plot IC time series for multiple factors with rolling averages.

    Parameters
    ----------
    ic_df : pd.DataFrame
        IC values with factors as columns, dates as index
    figsize : tuple
        Figure size
    title : str
        Plot title
    show_mean : bool
        Whether to show mean IC for each factor
    rolling_window : int, optional
        Window for rolling average

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig = plt.figure(figsize=figsize, facecolor='#0A0D12')
    gs = fig.add_gridspec(2, 1, hspace=0.3, height_ratios=[3, 1])

    # IC time series
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0F131C')

    colors = ['#38BDF8', '#6EE7B7', '#E9A568', '#3B6DFF', '#F472B6',
              '#A78BFA', '#FCD34D', '#34D399']

    for i, col in enumerate(ic_df.columns):
        color = colors[i % len(colors)]
        if rolling_window:
            ic_smooth = ic_df[col].rolling(window=rolling_window, min_periods=1).mean()
            ax1.plot(ic_smooth.index, ic_smooth.values, color=color,
                    linewidth=2, alpha=0.8, label=col)
        else:
            ax1.plot(ic_df.index, ic_df[col].values, color=color,
                    linewidth=1.5, alpha=0.7, label=col)

    ax1.axhline(y=0, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('IC', color='#E5E7EB', fontsize=11)
    ax1.set_title(title, color='#F9FAFB', fontsize=13, pad=15)
    ax1.legend(loc='upper left', framealpha=0.9, facecolor='#161D2B',
               edgecolor='#1E2636', labelcolor='#E5E7EB', ncol=min(4, len(ic_df.columns)))
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Bar chart of mean IC and IR
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0F131C')

    mean_ic = ic_df.mean()
    ir = ic_df.mean() / ic_df.std()

    x = np.arange(len(ic_df.columns))
    width = 0.35

    bars1 = ax2.bar(x - width/2, mean_ic, width, color='#38BDF8',
                    alpha=0.7, label='Mean IC', edgecolor='#161D2B', linewidth=1)

    ax2_twin = ax2.twinx()
    ax2_twin.set_facecolor('none')
    bars2 = ax2_twin.bar(x + width/2, ir, width, color='#6EE7B7',
                         alpha=0.7, label='IR', edgecolor='#161D2B', linewidth=1)

    ax2.set_ylabel('Mean IC', color='#38BDF8', fontsize=11)
    ax2_twin.set_ylabel('IR', color='#6EE7B7', fontsize=11)
    ax2.set_xlabel('Factor', color='#E5E7EB', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ic_df.columns, rotation=45, ha='right')
    ax2.axhline(y=0, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.5)
    ax2_twin.axhline(y=0, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.5)

    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               framealpha=0.9, facecolor='#161D2B', edgecolor='#1E2636',
               labelcolor='#E5E7EB')

    ax2.grid(True, alpha=0.2, color='#1E2636', axis='y')
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.tick_params(axis='y', labelcolor='#38BDF8')
    ax2_twin.tick_params(colors='#9CA3AF', labelsize=9)
    ax2_twin.tick_params(axis='y', labelcolor='#6EE7B7')
    ax2.spines['bottom'].set_color('#1E2636')
    ax2.spines['left'].set_color('#1E2636')
    ax2.spines['right'].set_color('#1E2636')
    ax2.spines['top'].set_visible(False)
    ax2_twin.spines['top'].set_visible(False)
    ax2_twin.spines['right'].set_color('#1E2636')

    plt.tight_layout()
    return fig


def plot_factor_correlation_matrix(
    factor_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Factor Correlation Matrix",
    method: str = 'pearson',
    annot: bool = True,
) -> plt.Figure:
    """
    Plot correlation matrix heatmap for multiple factors.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Factor values with factors as columns
    figsize : tuple
        Figure size
    title : str
        Plot title
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
    annot : bool
        Whether to annotate cells

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0A0D12')
    ax.set_facecolor('#0F131C')

    corr_matrix = factor_df.corr(method=method)

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, annot=annot, fmt='.2f',
                linewidths=1, linecolor='#0A0D12',
                square=True, cbar_kws={'label': 'Correlation', 'shrink': 0.8},
                ax=ax)

    ax.set_title(title, color='#F9FAFB', fontsize=13, pad=15)
    ax.tick_params(colors='#9CA3AF', labelsize=9)

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#9CA3AF', labelsize=9)
    cbar.ax.yaxis.label.set_color('#E5E7EB')
    cbar.outline.set_edgecolor('#1E2636')

    plt.tight_layout()
    return fig


def plot_ic_decay(
    ic_decay_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "IC Decay Analysis",
    lags: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Plot IC decay over multiple forward periods for multiple factors.

    Parameters
    ----------
    ic_decay_df : pd.DataFrame
        IC values by lag period, factors as columns, lags as index
    figsize : tuple
        Figure size
    title : str
        Plot title
    lags : list, optional
        Specific lag periods to highlight

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='#0A0D12')

    colors = ['#38BDF8', '#6EE7B7', '#E9A568', '#3B6DFF', '#F472B6',
              '#A78BFA', '#FCD34D', '#34D399']

    # IC decay line plot
    ax1.set_facecolor('#0F131C')
    for i, col in enumerate(ic_decay_df.columns):
        color = colors[i % len(colors)]
        ax1.plot(ic_decay_df.index, ic_decay_df[col].values, color=color,
                linewidth=2, marker='o', markersize=5, alpha=0.8, label=col)

    ax1.axhline(y=0, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Forward Period (days)', color='#E5E7EB', fontsize=11)
    ax1.set_ylabel('IC', color='#E5E7EB', fontsize=11)
    ax1.set_title(title, color='#F9FAFB', fontsize=12, pad=10)
    ax1.legend(loc='upper right', framealpha=0.9, facecolor='#161D2B',
               edgecolor='#1E2636', labelcolor='#E5E7EB')
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Heatmap of IC decay
    ax2.set_facecolor('#0F131C')
    sns.heatmap(ic_decay_df.T, cmap='RdBu_r', center=0,
                annot=True, fmt='.3f', linewidths=0.5, linecolor='#0A0D12',
                cbar_kws={'label': 'IC', 'shrink': 0.8}, ax=ax2)

    ax2.set_xlabel('Forward Period (days)', color='#E5E7EB', fontsize=11)
    ax2.set_ylabel('Factor', color='#E5E7EB', fontsize=11)
    ax2.set_title('IC Decay Heatmap', color='#F9FAFB', fontsize=12, pad=10)
    ax2.tick_params(colors='#9CA3AF', labelsize=9)

    # Style colorbar
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(colors='#9CA3AF', labelsize=9)
    cbar.ax.yaxis.label.set_color('#E5E7EB')
    cbar.outline.set_edgecolor('#1E2636')

    plt.tight_layout()
    return fig


def plot_performance_comparison(
    performance_df: pd.DataFrame,
    metrics: List[str] = ['IC_mean', 'IR', 'Sharpe', 'Max_DD'],
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Factor Performance Comparison",
) -> plt.Figure:
    """
    Plot comprehensive performance comparison across multiple factors.

    Parameters
    ----------
    performance_df : pd.DataFrame
        Performance metrics, factors as index, metrics as columns
    metrics : list
        Specific metrics to display
    figsize : tuple
        Figure size
    title : str
        Overall title

    Returns
    -------
    plt.Figure
        The figure object
    """
    available_metrics = [m for m in metrics if m in performance_df.columns]
    n_metrics = len(available_metrics)

    fig = plt.figure(figsize=figsize, facecolor='#0A0D12')
    gs = fig.add_gridspec(n_metrics, 2, hspace=0.4, wspace=0.3)

    colors = ['#38BDF8', '#6EE7B7', '#E9A568', '#3B6DFF', '#F472B6',
              '#A78BFA', '#FCD34D', '#34D399']

    for idx, metric in enumerate(available_metrics):
        # Bar chart
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.set_facecolor('#0F131C')

        values = performance_df[metric].sort_values(ascending=False)
        bars = ax1.barh(range(len(values)), values.values,
                        color=[colors[i % len(colors)] for i in range(len(values))],
                        alpha=0.7, edgecolor='#161D2B', linewidth=1)

        ax1.set_yticks(range(len(values)))
        ax1.set_yticklabels(values.index)
        ax1.set_xlabel(metric.replace('_', ' '), color='#E5E7EB', fontsize=10)
        ax1.set_title(f'{metric.replace("_", " ")} by Factor',
                     color='#F9FAFB', fontsize=11, pad=8)
        ax1.grid(True, alpha=0.2, color='#1E2636', axis='x')
        ax1.tick_params(colors='#9CA3AF', labelsize=9)
        ax1.spines['bottom'].set_color('#1E2636')
        ax1.spines['left'].set_color('#1E2636')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values.values)):
            label_x = val if val >= 0 else 0
            ha = 'left' if val >= 0 else 'right'
            ax1.text(label_x, i, f' {val:.3f}', va='center', ha=ha,
                    color='#E5E7EB', fontsize=8, fontweight='bold')

        # Distribution
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.set_facecolor('#0F131C')

        ax2.hist(performance_df[metric].dropna(), bins=15, color='#38BDF8',
                alpha=0.6, edgecolor='#161D2B')

        mean_val = performance_df[metric].mean()
        median_val = performance_df[metric].median()

        ax2.axvline(x=mean_val, color='#6EE7B7', linestyle='--',
                   linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax2.axvline(x=median_val, color='#E9A568', linestyle=':',
                   linewidth=2, alpha=0.8, label=f'Median: {median_val:.3f}')

        ax2.set_xlabel(metric.replace('_', ' '), color='#E5E7EB', fontsize=10)
        ax2.set_ylabel('Count', color='#E5E7EB', fontsize=10)
        ax2.set_title(f'{metric.replace("_", " ")} Distribution',
                     color='#F9FAFB', fontsize=11, pad=8)
        ax2.legend(loc='upper right', framealpha=0.9, facecolor='#161D2B',
                   edgecolor='#1E2636', labelcolor='#E5E7EB', fontsize=8)
        ax2.grid(True, alpha=0.2, color='#1E2636', axis='y')
        ax2.tick_params(colors='#9CA3AF', labelsize=9)
        ax2.spines['bottom'].set_color('#1E2636')
        ax2.spines['left'].set_color('#1E2636')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    fig.suptitle(title, color='#F9FAFB', fontsize=14, y=0.995)
    plt.tight_layout()
    return fig
