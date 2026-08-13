"""
Portfolio statistics for long/short backtesting.

Provides Sharpe ratio, drawdown analysis, and portfolio return metrics.
"""

from typing import Tuple, Optional
import numpy as np


def compute_long_short_returns(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    long_threshold: float = 0.8,
    short_threshold: float = 0.2,
    validity_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute long/short portfolio returns based on factor quantiles.

    Args:
        factor_values: Factor values (T, N) or (T, N, F)
        forward_returns: Forward returns (T, N)
        long_threshold: Quantile threshold for long positions (default 0.8 = top 20%)
        short_threshold: Quantile threshold for short positions (default 0.2 = bottom 20%)
        validity_mask: Optional boolean mask (T, N) or (T, N, F)

    Returns:
        (long_returns, short_returns, long_short_returns)
        Each shape (T,) or (T, F) for time series of portfolio returns
    """
    # Handle 3D factor values
    if factor_values.ndim == 3:
        T, N, F = factor_values.shape
        long_rets = np.full((T, F), np.nan)
        short_rets = np.full((T, F), np.nan)
        ls_rets = np.full((T, F), np.nan)

        for f in range(F):
            fv = factor_values[:, :, f]
            vm = validity_mask[:, :, f] if validity_mask is not None else None
            long_rets[:, f], short_rets[:, f], ls_rets[:, f] = compute_long_short_returns(
                fv, forward_returns, long_threshold, short_threshold, vm
            )
        return long_rets, short_rets, ls_rets

    # 2D case
    T, N = factor_values.shape
    long_returns = np.full(T, np.nan)
    short_returns = np.full(T, np.nan)
    long_short_returns = np.full(T, np.nan)

    for t in range(T):
        factor_t = factor_values[t, :]
        ret_t = forward_returns[t, :]

        # Apply validity mask
        if validity_mask is not None:
            valid = validity_mask[t, :]
            factor_t = np.where(valid, factor_t, np.nan)

        # Filter finite values
        finite_mask = np.isfinite(factor_t) & np.isfinite(ret_t)
        if np.sum(finite_mask) < 2:
            continue

        factor_valid = factor_t[finite_mask]
        ret_valid = ret_t[finite_mask]

        # Compute quantiles
        long_cutoff = np.quantile(factor_valid, long_threshold)
        short_cutoff = np.quantile(factor_valid, short_threshold)

        # Select long/short positions
        long_mask = factor_valid >= long_cutoff
        short_mask = factor_valid <= short_cutoff

        if np.sum(long_mask) > 0:
            long_returns[t] = np.mean(ret_valid[long_mask])

        if np.sum(short_mask) > 0:
            short_returns[t] = np.mean(ret_valid[short_mask])

        if np.sum(long_mask) > 0 and np.sum(short_mask) > 0:
            long_short_returns[t] = long_returns[t] - short_returns[t]

    return long_returns, short_returns, long_short_returns


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Return series (T,) or (T, F)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        min_periods: Minimum periods required

    Returns:
        Sharpe ratio, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    sharpe = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Compute excess returns
        rf_per_period = risk_free_rate / periods_per_year
        excess_ret = ret_valid - rf_per_period

        mean_excess = np.mean(excess_ret)
        std_excess = np.std(excess_ret, ddof=1)

        if std_excess == 0 or std_excess < 1e-10 or not np.isfinite(std_excess):
            continue

        # Annualize
        sharpe[f] = mean_excess / std_excess * np.sqrt(periods_per_year)

    return sharpe[0] if squeeze else sharpe


def compute_maximum_drawdown(
    returns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute maximum drawdown from return series.

    Args:
        returns: Return series (T,) or (T, F)

    Returns:
        (max_drawdown, drawdown_series, peak_indices)
        max_drawdown: Maximum drawdown magnitude (positive), shape () or (F,)
        drawdown_series: Drawdown at each time step, shape (T,) or (T, F)
        peak_indices: Index of peak before maximum drawdown, shape () or (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape

    # Replace NaN with 0 for cumulative product
    returns_filled = np.where(np.isfinite(returns), returns, 0.0)

    # Compute cumulative returns (wealth curve)
    cum_returns = np.cumprod(1.0 + returns_filled, axis=0)

    # Compute running maximum
    running_max = np.maximum.accumulate(cum_returns, axis=0)

    # Drawdown series
    drawdown_series = (cum_returns - running_max) / running_max

    # Maximum drawdown per factor
    max_dd = np.min(drawdown_series, axis=0)  # Most negative
    max_dd = -max_dd  # Convert to positive magnitude

    # Find peak indices
    peak_indices = np.argmin(drawdown_series, axis=0)

    if squeeze:
        return max_dd[0], drawdown_series[:, 0], peak_indices[0]
    else:
        return max_dd, drawdown_series, peak_indices


def compute_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Calmar ratio (annualized return / maximum drawdown).

    Args:
        returns: Return series (T,) or (T, F)
        periods_per_year: Number of periods per year
        min_periods: Minimum periods required

    Returns:
        Calmar ratio, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    calmar = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Annualized return
        mean_ret = np.mean(ret_valid)
        ann_ret = mean_ret * periods_per_year

        # Maximum drawdown
        max_dd, _, _ = compute_maximum_drawdown(ret_valid)

        if max_dd == 0 or not np.isfinite(max_dd):
            continue

        calmar[f] = ann_ret / max_dd

    return calmar[0] if squeeze else calmar


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Sortino ratio (excess return / downside deviation).

    Args:
        returns: Return series (T,) or (T, F)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        min_periods: Minimum periods required

    Returns:
        Sortino ratio, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    sortino = np.full(F, np.nan)

    rf_per_period = risk_free_rate / periods_per_year

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]
        excess_ret = ret_valid - rf_per_period

        mean_excess = np.mean(excess_ret)

        # Downside deviation (only negative excess returns)
        downside_ret = excess_ret[excess_ret < 0]
        if len(downside_ret) == 0:
            continue

        downside_std = np.sqrt(np.mean(downside_ret ** 2))

        if downside_std == 0 or not np.isfinite(downside_std):
            continue

        # Annualize
        sortino[f] = mean_excess / downside_std * np.sqrt(periods_per_year)

    return sortino[0] if squeeze else sortino


def compute_win_rate(
    returns: np.ndarray,
) -> np.ndarray:
    """
    Compute win rate (fraction of positive returns).

    Args:
        returns: Return series (T,) or (T, F)

    Returns:
        Win rate in [0, 1], scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    valid = np.isfinite(returns)
    n_valid = np.sum(valid, axis=0)

    wins = np.sum((returns > 0) & valid, axis=0)
    win_rate = wins / np.maximum(n_valid, 1)

    # Set to NaN if no valid observations
    win_rate = np.where(n_valid > 0, win_rate, np.nan)

    return win_rate[0] if squeeze else win_rate
