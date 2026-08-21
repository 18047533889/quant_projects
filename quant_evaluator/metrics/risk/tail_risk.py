"""
Tail risk measures for portfolio analysis.

Provides tail ratio, gain/loss ratio, upside potential ratio, omega ratio,
expected shortfall ratio, and tail dependence metrics.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats


def compute_tail_ratio(
    returns: np.ndarray,
    upper_percentile: float = 0.95,
    lower_percentile: float = 0.05,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute tail ratio (ratio of right tail to left tail).

    Tail ratio = |95th percentile| / |5th percentile|

    Args:
        returns: Return series (T,) or (T, F)
        upper_percentile: Upper tail percentile (default 0.95)
        lower_percentile: Lower tail percentile (default 0.05)
        min_periods: Minimum periods required

    Returns:
        Tail ratio, scalar or shape (F,). Higher is better (more upside vs downside).
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    tail_ratio = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        upper_tail = np.percentile(ret_valid, upper_percentile * 100.0)
        lower_tail = np.percentile(ret_valid, lower_percentile * 100.0)

        # Avoid division by zero
        if abs(lower_tail) < 1e-10:
            continue

        tail_ratio[f] = abs(upper_tail) / abs(lower_tail)

    return tail_ratio[0] if squeeze else tail_ratio


def compute_gain_loss_ratio(
    returns: np.ndarray,
    threshold: float = 0.0,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute gain/loss ratio (average gain / average loss).

    Args:
        returns: Return series (T,) or (T, F)
        threshold: Threshold for gains/losses (default 0.0)
        min_periods: Minimum periods required

    Returns:
        Gain/loss ratio, scalar or shape (F,). Higher is better.
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    gl_ratio = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        gains = ret_valid[ret_valid > threshold]
        losses = ret_valid[ret_valid <= threshold]

        if len(gains) == 0 or len(losses) == 0:
            continue

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        # Avoid division by zero
        if abs(avg_loss) < 1e-10:
            continue

        gl_ratio[f] = avg_gain / abs(avg_loss)

    return gl_ratio[0] if squeeze else gl_ratio


def compute_upside_potential_ratio(
    returns: np.ndarray,
    minimum_acceptable_return: float = 0.0,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Upside Potential Ratio (UPR).

    UPR = E[max(R - MAR, 0)] / sqrt(E[min(R - MAR, 0)^2])

    Args:
        returns: Return series (T,) or (T, F)
        minimum_acceptable_return: MAR threshold (default 0.0)
        min_periods: Minimum periods required

    Returns:
        Upside Potential Ratio, scalar or shape (F,). Higher is better.
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    upr = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Excess returns over MAR
        excess = ret_valid - minimum_acceptable_return

        # Upside potential (positive excess)
        upside = excess[excess > 0]
        upside_potential = np.mean(upside) if len(upside) > 0 else 0.0

        # Downside deviation (negative excess)
        downside = excess[excess < 0]
        downside_deviation = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0.0

        if downside_deviation < 1e-10:
            continue

        upr[f] = upside_potential / downside_deviation

    return upr[0] if squeeze else upr


def compute_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Omega ratio (probability-weighted gains / probability-weighted losses).

    Omega = sum(max(R - threshold, 0)) / sum(max(threshold - R, 0))

    Args:
        returns: Return series (T,) or (T, F)
        threshold: Return threshold (default 0.0)
        min_periods: Minimum periods required

    Returns:
        Omega ratio, scalar or shape (F,). Higher is better (>1 is good).
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    omega = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Gains and losses relative to threshold
        gains = np.maximum(ret_valid - threshold, 0.0)
        losses = np.maximum(threshold - ret_valid, 0.0)

        sum_gains = np.sum(gains)
        sum_losses = np.sum(losses)

        if sum_losses < 1e-10:
            continue

        omega[f] = sum_gains / sum_losses

    return omega[0] if squeeze else omega


def compute_expected_shortfall_ratio(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 30,
) -> np.ndarray:
    """
    Compute Expected Shortfall Ratio (excess return / CVaR).

    Similar to Sharpe but uses CVaR instead of volatility.

    Args:
        returns: Return series (T,) or (T, F)
        confidence_level: Confidence level for CVaR
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year for annualization
        min_periods: Minimum periods required

    Returns:
        ES Ratio, scalar or shape (F,). Higher is better.
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    es_ratio = np.full(F, np.nan)

    rf_per_period = risk_free_rate / periods_per_year
    quantile = 1.0 - confidence_level

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Excess return
        excess_ret = ret_valid - rf_per_period
        mean_excess = np.mean(excess_ret)

        # CVaR (Expected Shortfall)
        var_threshold = np.quantile(ret_valid, quantile)
        tail_returns = ret_valid[ret_valid <= var_threshold]

        if len(tail_returns) == 0:
            continue

        cvar = -np.mean(tail_returns)

        if cvar < 1e-10:
            continue

        # Annualize
        es_ratio[f] = mean_excess / cvar * np.sqrt(periods_per_year)

    return es_ratio[0] if squeeze else es_ratio


def compute_tail_dependence(
    returns_x: np.ndarray,
    returns_y: np.ndarray,
    quantile: float = 0.05,
    min_periods: int = 30,
) -> Tuple[float, float]:
    """
    Compute lower and upper tail dependence between two return series.

    Measures correlation in extreme events (left and right tails).

    Args:
        returns_x: First return series (T,)
        returns_y: Second return series (T,)
        quantile: Quantile threshold for tails (default 0.05 for 5%)
        min_periods: Minimum periods required

    Returns:
        (lower_tail_dep, upper_tail_dep)
        Lower tail dependence: correlation when both have extreme losses
        Upper tail dependence: correlation when both have extreme gains
    """
    if returns_x.ndim != 1 or returns_y.ndim != 1:
        raise ValueError("compute_tail_dependence requires 1D returns")

    if len(returns_x) != len(returns_y):
        raise ValueError("returns_x and returns_y must have same length")

    # Filter finite values
    valid = np.isfinite(returns_x) & np.isfinite(returns_y)
    n_valid = np.sum(valid)

    if n_valid < min_periods:
        return np.nan, np.nan

    ret_x = returns_x[valid]
    ret_y = returns_y[valid]

    # Lower tail dependence (extreme losses)
    threshold_x_lower = np.quantile(ret_x, quantile)
    threshold_y_lower = np.quantile(ret_y, quantile)

    both_lower = (ret_x <= threshold_x_lower) & (ret_y <= threshold_y_lower)
    x_lower = ret_x <= threshold_x_lower

    prob_both_lower = np.sum(both_lower) / len(ret_x)
    prob_x_lower = np.sum(x_lower) / len(ret_x)

    if prob_x_lower < 1e-10:
        lower_tail_dep = np.nan
    else:
        lower_tail_dep = prob_both_lower / prob_x_lower

    # Upper tail dependence (extreme gains)
    threshold_x_upper = np.quantile(ret_x, 1.0 - quantile)
    threshold_y_upper = np.quantile(ret_y, 1.0 - quantile)

    both_upper = (ret_x >= threshold_x_upper) & (ret_y >= threshold_y_upper)
    x_upper = ret_x >= threshold_x_upper

    prob_both_upper = np.sum(both_upper) / len(ret_x)
    prob_x_upper = np.sum(x_upper) / len(ret_x)

    if prob_x_upper < 1e-10:
        upper_tail_dep = np.nan
    else:
        upper_tail_dep = prob_both_upper / prob_x_upper

    return lower_tail_dep, upper_tail_dep
