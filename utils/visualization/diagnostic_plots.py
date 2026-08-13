"""
Statistical diagnostic visualization utilities.

Provides plotting functions for residual analysis, normality tests,
autocorrelation, and heteroskedasticity detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from scipy import stats


def plot_residual_analysis(
    residuals: pd.Series,
    fitted_values: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Residual Analysis",
) -> plt.Figure:
    """
    Comprehensive residual diagnostic plots.

    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    fitted_values : pd.Series, optional
        Fitted values for residual vs fitted plot
    figsize : tuple
        Figure size
    title : str
        Overall title

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig = plt.figure(figsize=figsize, facecolor='#0A0D12')
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    residuals_clean = residuals.dropna()

    # 1. Residuals vs Fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0F131C')

    if fitted_values is not None:
        fitted_clean = fitted_values.loc[residuals_clean.index]
        ax1.scatter(fitted_clean, residuals_clean, color='#38BDF8',
                   alpha=0.5, s=20, edgecolors='none')

        # Add lowess smoothing line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals_clean, fitted_clean, frac=0.3)
            ax1.plot(smoothed[:, 0], smoothed[:, 1], color='#E9A568',
                    linewidth=2, label='LOWESS')
        except ImportError:
            pass

        ax1.axhline(y=0, color='#6EE7B7', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Fitted Values', color='#E5E7EB', fontsize=10)
    else:
        ax1.plot(residuals_clean.index, residuals_clean.values,
                color='#38BDF8', alpha=0.6, linewidth=1)
        ax1.axhline(y=0, color='#6EE7B7', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Index', color='#E5E7EB', fontsize=10)

    ax1.set_ylabel('Residuals', color='#E5E7EB', fontsize=10)
    ax1.set_title('Residuals vs Fitted', color='#F9FAFB', fontsize=11, pad=10)
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0F131C')

    stats.probplot(residuals_clean, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_color('#38BDF8')
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[1].set_color('#E9A568')
    ax2.get_lines()[1].set_linewidth(2)

    ax2.set_xlabel('Theoretical Quantiles', color='#E5E7EB', fontsize=10)
    ax2.set_ylabel('Sample Quantiles', color='#E5E7EB', fontsize=10)
    ax2.set_title('Normal Q-Q Plot', color='#F9FAFB', fontsize=11, pad=10)
    ax2.grid(True, alpha=0.2, color='#1E2636')
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.spines['bottom'].set_color('#1E2636')
    ax2.spines['left'].set_color('#1E2636')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. Scale-Location (sqrt of standardized residuals)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#0F131C')

    standardized_resid = residuals_clean / residuals_clean.std()
    sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))

    if fitted_values is not None:
        ax3.scatter(fitted_clean, sqrt_abs_resid, color='#38BDF8',
                   alpha=0.5, s=20, edgecolors='none')

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(sqrt_abs_resid, fitted_clean, frac=0.3)
            ax3.plot(smoothed[:, 0], smoothed[:, 1], color='#E9A568',
                    linewidth=2, label='LOWESS')
        except ImportError:
            pass

        ax3.set_xlabel('Fitted Values', color='#E5E7EB', fontsize=10)
    else:
        ax3.plot(sqrt_abs_resid.index, sqrt_abs_resid.values,
                color='#38BDF8', alpha=0.6, linewidth=1)
        ax3.set_xlabel('Index', color='#E5E7EB', fontsize=10)

    ax3.set_ylabel('√|Standardized Residuals|', color='#E5E7EB', fontsize=10)
    ax3.set_title('Scale-Location', color='#F9FAFB', fontsize=11, pad=10)
    ax3.grid(True, alpha=0.2, color='#1E2636')
    ax3.tick_params(colors='#9CA3AF', labelsize=9)
    ax3.spines['bottom'].set_color('#1E2636')
    ax3.spines['left'].set_color('#1E2636')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 4. Histogram with normality test
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#0F131C')

    ax4.hist(residuals_clean, bins=30, color='#38BDF8', alpha=0.6,
            edgecolor='#161D2B', density=True)

    # Overlay normal distribution
    mu, sigma = residuals_clean.mean(), residuals_clean.std()
    x_range = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
    ax4.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
            color='#E9A568', linewidth=2, label='Normal')

    # KDE
    kde = stats.gaussian_kde(residuals_clean)
    ax4.plot(x_range, kde(x_range), color='#6EE7B7',
            linewidth=2, linestyle='--', label='KDE')

    ax4.set_xlabel('Residuals', color='#E5E7EB', fontsize=10)
    ax4.set_ylabel('Density', color='#E5E7EB', fontsize=10)
    ax4.set_title('Residual Distribution', color='#F9FAFB', fontsize=11, pad=10)
    ax4.legend(framealpha=0.9, facecolor='#161D2B', edgecolor='#1E2636',
               labelcolor='#E5E7EB', fontsize=8)
    ax4.grid(True, alpha=0.2, color='#1E2636')
    ax4.tick_params(colors='#9CA3AF', labelsize=9)
    ax4.spines['bottom'].set_color('#1E2636')
    ax4.spines['left'].set_color('#1E2636')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Normality tests
    _, shapiro_p = stats.shapiro(residuals_clean[:min(5000, len(residuals_clean))])
    _, ks_p = stats.kstest(residuals_clean, 'norm', args=(mu, sigma))

    stats_text = (
        f'Mean: {mu:.4f}\n'
        f'Std: {sigma:.4f}\n'
        f'Skew: {residuals_clean.skew():.4f}\n'
        f'Kurt: {residuals_clean.kurtosis():.4f}\n'
        f'Shapiro p: {shapiro_p:.4f}\n'
        f'KS p: {ks_p:.4f}'
    )
    ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#161D2B', alpha=0.9,
                     edgecolor='#1E2636'),
            color='#E5E7EB', fontsize=8, family='monospace')

    fig.suptitle(title, color='#F9FAFB', fontsize=13, y=0.995)
    plt.tight_layout()
    return fig


def plot_qq_plot(
    data: pd.Series,
    distribution: str = 'norm',
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Q-Q Plot",
) -> plt.Figure:
    """
    Quantile-Quantile plot against theoretical distribution.

    Parameters
    ----------
    data : pd.Series
        Data to test
    distribution : str
        Distribution name ('norm', 't', 'uniform', etc.)
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0A0D12')
    ax.set_facecolor('#0F131C')

    data_clean = data.dropna()

    stats.probplot(data_clean, dist=distribution, plot=ax)
    ax.get_lines()[0].set_color('#38BDF8')
    ax.get_lines()[0].set_markersize(5)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('#E9A568')
    ax.get_lines()[1].set_linewidth(2.5)

    ax.set_xlabel('Theoretical Quantiles', color='#E5E7EB', fontsize=11)
    ax.set_ylabel('Sample Quantiles', color='#E5E7EB', fontsize=11)
    ax.set_title(f'{title} ({distribution})', color='#F9FAFB', fontsize=13, pad=15)
    ax.grid(True, alpha=0.2, color='#1E2636')
    ax.tick_params(colors='#9CA3AF', labelsize=9)
    ax.spines['bottom'].set_color('#1E2636')
    ax.spines['left'].set_color('#1E2636')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Statistical tests
    if distribution == 'norm':
        _, shapiro_p = stats.shapiro(data_clean[:min(5000, len(data_clean))])
        _, ks_p = stats.kstest(data_clean, 'norm')
        stats_text = (
            f'Shapiro-Wilk p: {shapiro_p:.4f}\n'
            f'KS test p: {ks_p:.4f}\n'
            f'Skewness: {data_clean.skew():.4f}\n'
            f'Kurtosis: {data_clean.kurtosis():.4f}'
        )
    else:
        stats_text = (
            f'Skewness: {data_clean.skew():.4f}\n'
            f'Kurtosis: {data_clean.kurtosis():.4f}'
        )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='#161D2B', alpha=0.9,
                     edgecolor='#1E2636'),
            color='#E5E7EB', fontsize=9, family='monospace')

    plt.tight_layout()
    return fig


def plot_autocorrelation(
    data: pd.Series,
    lags: int = 40,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Autocorrelation Analysis",
    alpha: float = 0.05,
) -> plt.Figure:
    """
    Plot ACF and PACF for time series data.

    Parameters
    ----------
    data : pd.Series
        Time series data
    lags : int
        Number of lags to plot
    figsize : tuple
        Figure size
    title : str
        Overall title
    alpha : float
        Significance level for confidence intervals

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='#0A0D12')

    data_clean = data.dropna()
    n = len(data_clean)

    # Calculate ACF
    acf_vals = np.zeros(lags + 1)
    acf_vals[0] = 1.0
    for lag in range(1, lags + 1):
        c0 = np.dot(data_clean - data_clean.mean(),
                    data_clean - data_clean.mean())
        c_lag = np.dot(data_clean[:-lag] - data_clean.mean(),
                       data_clean[lag:] - data_clean.mean())
        acf_vals[lag] = c_lag / c0 if c0 > 0 else 0

    # ACF plot
    ax1.set_facecolor('#0F131C')
    ax1.vlines(range(lags + 1), 0, acf_vals, colors='#38BDF8',
               linewidth=2, alpha=0.8)
    ax1.scatter(range(lags + 1), acf_vals, color='#38BDF8',
                s=30, zorder=3, alpha=0.8)
    ax1.axhline(y=0, color='#9CA3AF', linestyle='-', linewidth=1, alpha=0.5)

    # Confidence intervals
    ci = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    ax1.axhline(y=ci, color='#E9A568', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=-ci, color='#E9A568', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.fill_between(range(lags + 1), ci, -ci, color='#E9A568', alpha=0.1)

    ax1.set_xlabel('Lag', color='#E5E7EB', fontsize=11)
    ax1.set_ylabel('ACF', color='#E5E7EB', fontsize=11)
    ax1.set_title('Autocorrelation Function', color='#F9FAFB',
                  fontsize=12, pad=10)
    ax1.set_xlim(-1, lags + 1)
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Calculate PACF (simplified Yule-Walker)
    pacf_vals = np.zeros(lags + 1)
    pacf_vals[0] = 1.0
    for lag in range(1, min(lags + 1, len(data_clean) // 2)):
        try:
            ar_coeffs = np.zeros(lag)
            for i in range(lag):
                ar_coeffs[i] = acf_vals[i + 1]
            R = np.zeros((lag, lag))
            for i in range(lag):
                for j in range(lag):
                    R[i, j] = acf_vals[abs(i - j)]
            if np.linalg.det(R) != 0:
                pacf_vals[lag] = np.linalg.solve(R, ar_coeffs)[-1]
        except:
            pacf_vals[lag] = 0

    # PACF plot
    ax2.set_facecolor('#0F131C')
    ax2.vlines(range(lags + 1), 0, pacf_vals, colors='#6EE7B7',
               linewidth=2, alpha=0.8)
    ax2.scatter(range(lags + 1), pacf_vals, color='#6EE7B7',
                s=30, zorder=3, alpha=0.8)
    ax2.axhline(y=0, color='#9CA3AF', linestyle='-', linewidth=1, alpha=0.5)

    # Confidence intervals
    ax2.axhline(y=ci, color='#E9A568', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-ci, color='#E9A568', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.fill_between(range(lags + 1), ci, -ci, color='#E9A568', alpha=0.1)

    ax2.set_xlabel('Lag', color='#E5E7EB', fontsize=11)
    ax2.set_ylabel('PACF', color='#E5E7EB', fontsize=11)
    ax2.set_title('Partial Autocorrelation Function', color='#F9FAFB',
                  fontsize=12, pad=10)
    ax2.set_xlim(-1, lags + 1)
    ax2.grid(True, alpha=0.2, color='#1E2636')
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.spines['bottom'].set_color('#1E2636')
    ax2.spines['left'].set_color('#1E2636')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(title, color='#F9FAFB', fontsize=13, y=0.995)
    plt.tight_layout()
    return fig


def plot_heteroskedasticity(
    residuals: pd.Series,
    fitted_values: pd.Series,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Heteroskedasticity Analysis",
) -> plt.Figure:
    """
    Analyze heteroskedasticity in residuals.

    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    fitted_values : pd.Series
        Fitted values
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='#0A0D12')

    # Align indices
    common_idx = residuals.dropna().index.intersection(fitted_values.dropna().index)
    resid_clean = residuals.loc[common_idx]
    fitted_clean = fitted_values.loc[common_idx]

    # Squared residuals vs fitted
    ax1.set_facecolor('#0F131C')
    squared_resid = resid_clean ** 2

    ax1.scatter(fitted_clean, squared_resid, color='#38BDF8',
               alpha=0.4, s=20, edgecolors='none')

    # Add trend line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(squared_resid, fitted_clean, frac=0.3)
        ax1.plot(smoothed[:, 0], smoothed[:, 1], color='#E9A568',
                linewidth=2.5, label='LOWESS', zorder=5)
    except ImportError:
        z = np.polyfit(fitted_clean, squared_resid, 2)
        p = np.poly1d(z)
        x_sorted = np.sort(fitted_clean)
        ax1.plot(x_sorted, p(x_sorted), color='#E9A568',
                linewidth=2.5, label='Polynomial Fit', zorder=5)

    ax1.set_xlabel('Fitted Values', color='#E5E7EB', fontsize=11)
    ax1.set_ylabel('Squared Residuals', color='#E5E7EB', fontsize=11)
    ax1.set_title('Residuals² vs Fitted', color='#F9FAFB', fontsize=12, pad=10)
    ax1.legend(framealpha=0.9, facecolor='#161D2B', edgecolor='#1E2636',
               labelcolor='#E5E7EB')
    ax1.grid(True, alpha=0.2, color='#1E2636')
    ax1.tick_params(colors='#9CA3AF', labelsize=9)
    ax1.spines['bottom'].set_color('#1E2636')
    ax1.spines['left'].set_color('#1E2636')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Absolute residuals vs fitted
    ax2.set_facecolor('#0F131C')
    abs_resid = np.abs(resid_clean)

    ax2.scatter(fitted_clean, abs_resid, color='#6EE7B7',
               alpha=0.4, s=20, edgecolors='none')

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(abs_resid, fitted_clean, frac=0.3)
        ax2.plot(smoothed[:, 0], smoothed[:, 1], color='#E9A568',
                linewidth=2.5, label='LOWESS', zorder=5)
    except ImportError:
        z = np.polyfit(fitted_clean, abs_resid, 2)
        p = np.poly1d(z)
        x_sorted = np.sort(fitted_clean)
        ax2.plot(x_sorted, p(x_sorted), color='#E9A568',
                linewidth=2.5, label='Polynomial Fit', zorder=5)

    ax2.set_xlabel('Fitted Values', color='#E5E7EB', fontsize=11)
    ax2.set_ylabel('|Residuals|', color='#E5E7EB', fontsize=11)
    ax2.set_title('|Residuals| vs Fitted', color='#F9FAFB', fontsize=12, pad=10)
    ax2.legend(framealpha=0.9, facecolor='#161D2B', edgecolor='#1E2636',
               labelcolor='#E5E7EB')
    ax2.grid(True, alpha=0.2, color='#1E2636')
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.spines['bottom'].set_color('#1E2636')
    ax2.spines['left'].set_color('#1E2636')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Breusch-Pagan test approximation
    corr = np.corrcoef(fitted_clean, squared_resid)[0, 1]
    stats_text = (
        f'Corr(Fitted, Resid²): {corr:.4f}\n'
        f'Mean |Resid|: {abs_resid.mean():.4f}\n'
        f'Std |Resid|: {abs_resid.std():.4f}'
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='#161D2B', alpha=0.9,
                     edgecolor='#1E2636'),
            color='#E5E7EB', fontsize=9, family='monospace')

    fig.suptitle(title, color='#F9FAFB', fontsize=13, y=0.995)
    plt.tight_layout()
    return fig
