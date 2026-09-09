"""
Tail risk measures for portfolio analysis.

Provides tail ratio, gain/loss ratio, upside potential ratio, omega ratio,
expected shortfall ratio, and tail dependence metrics.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats
from .var_cvar import empirical_expected_shortfall


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
        upside_potential = np.mean(np.maximum(excess, 0.))

        # Downside deviation (negative excess)
        downside_deviation = np.sqrt(np.mean(np.minimum(excess, 0.) ** 2))

        if not np.isfinite(downside_deviation) or downside_deviation <= 0:
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
        periods_per_year: Converts annual risk-free rate only; does not scale ES
        min_periods: Minimum periods required

    Returns:
        ES Ratio, scalar or shape (F,). Higher is better.
    """
    if (isinstance(periods_per_year, (bool, np.bool_)) or not isinstance(periods_per_year, (int, np.integer))
            or periods_per_year < 1 or isinstance(risk_free_rate, (bool, np.bool_))
            or not np.isfinite(risk_free_rate)):
        raise ValueError("valid annual risk-free rate and periods_per_year required")
    returns = np.asarray(returns, dtype=float)
    if returns.ndim not in (1, 2):
        raise ValueError("ES ratio requires (T,) or (T,F) returns")
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    es_ratio = np.full(F, np.nan)

    rf_per_period = risk_free_rate / periods_per_year

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

        cvar = empirical_expected_shortfall(ret_valid, confidence_level)

        if not np.isfinite(cvar) or cvar <= 0:
            continue

        # Same-period mean/ES; there is no universal sqrt(time) ES scaling.
        es_ratio[f] = mean_excess / cvar

    return es_ratio[0] if squeeze else es_ratio


def compute_tail_dependence(
    returns_x: np.ndarray,
    returns_y: np.ndarray,
    quantile: float = 0.05,
    min_periods: int = 30,
) -> Tuple[float, float]:
    """
    Compute lower and upper tail dependence between two return series.

    Compatibility diagnostic: finite-q P(Y tail | X tail), NOT correlation
    or an asymptotic tail-dependence coefficient. Use
    compute_finite_q_coexceedance for counts, both directions and status.

    Args:
        returns_x: First return series (T,)
        returns_y: Second return series (T,)
        quantile: Quantile threshold for tails (default 0.05 for 5%)
        min_periods: Minimum periods required

    Returns:
        (lower_tail_dep, upper_tail_dep)
        Directional finite-q conditional probabilities P(Y-tail | X-tail).
        These are neither correlations nor asymptotic tail coefficients.
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


def compute_finite_q_coexceedance(returns_x, returns_y, quantile=.05, min_periods=30,
                                  min_tail_observations=5):
    """Directional finite-q event counts, inclusive empirical quantile ties."""
    x, y = np.asarray(returns_x, dtype=float), np.asarray(returns_y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("coexceedance requires aligned one-dimensional samples")
    if isinstance(quantile, bool) or not np.isfinite(quantile) or not 0 < quantile < .5:
        raise ValueError("tail quantile must be in (0, .5)")
    for name, value in (("min_periods", min_periods), ("min_tail_observations", min_tail_observations)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    result = {"metric_id": "finite_q_coexceedance.v1", "quantile": quantile,
              "n_valid": len(x), "qualification": "DESCRIPTIVE_ONLY", "tails": {}}
    for name, q in (("lower", quantile), ("upper", 1-quantile)):
        if len(x) < min_periods:
            result["tails"][name] = {"status": "INSUFFICIENT_DATA"}
            continue
        tx, ty = np.quantile(x, q), np.quantile(y, q)
        ex, ey = (x <= tx, y <= ty) if name == "lower" else (x >= tx, y >= ty)
        nx, ny, joint = int(ex.sum()), int(ey.sum()), int((ex & ey).sum())
        degenerate=np.ptp(x)==0 or np.ptp(y)==0
        enough=nx>=min_tail_observations and ny>=min_tail_observations
        status="DEGENERATE_MARGIN" if degenerate else ("INSUFFICIENT_TAIL_EVENTS" if not enough else "DESCRIPTIVE_ONLY")
        result["tails"][name] = {"status": status,
            "x_count": nx, "y_count": ny, "joint_count": joint,
            "p_y_given_x": joint/nx if nx else np.nan,
            "p_x_given_y": joint/ny if ny else np.nan,
            "joint_probability": joint/len(x), "threshold_x": float(tx), "threshold_y": float(ty),
            "ties_rule":"inclusive_empirical_quantile","marginal_quality":"ADEQUATE" if enough and not degenerate else "LOW"}
    return result
