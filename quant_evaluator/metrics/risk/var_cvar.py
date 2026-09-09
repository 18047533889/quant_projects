"""
Value at Risk (VaR) and Conditional Value at Risk (CVaR) computation.

Implements multiple VaR methodologies: historical, parametric, and Cornish-Fisher.
Historical Expected Shortfall integrates a fixed lower-tail probability mass,
including fractional boundary observations; it is not a threshold mean.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats


def _validate_confidence_level(confidence_level: float) -> None:
    """Fail closed: confidence_level must be strictly inside (0, 1).

    confidence_level == 1.0 would divide by zero in parametric CVaR
    (inf VaR), and <= 0 is meaningless. Raise ValueError otherwise.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )


def compute_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical",
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Value at Risk (VaR) at specified confidence level.

    Args:
        returns: Return series (T,) or (T, F)
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR); must be in (0, 1)
        method: "historical", "parametric", or "cornish_fisher"
        min_periods: Minimum periods required

    Returns:
        VaR (positive value representing loss), scalar or shape (F,)
    """
    _validate_confidence_level(confidence_level)
    if method == "historical":
        return compute_var_historical(returns, confidence_level, min_periods)
    elif method == "parametric":
        return compute_var_parametric(returns, confidence_level, min_periods)
    elif method == "cornish_fisher":
        return compute_var_cornish_fisher(returns, confidence_level, min_periods)
    else:
        raise ValueError(f"Unknown VaR method: {method}")


def compute_var_historical(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Historical VaR using empirical quantile.

    Args:
        returns: Return series (T,) or (T, F)
        confidence_level: Confidence level
        min_periods: Minimum periods required

    Returns:
        VaR (positive loss value), scalar or shape (F,)
    """
    _validate_confidence_level(confidence_level)

    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    var = np.full(F, np.nan)

    # VaR is at (1 - confidence_level) quantile of loss distribution
    # For returns, this is negative quantile (losses are negative returns)
    quantile = 1.0 - confidence_level

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Compute quantile (negative return = loss)
        var_value = np.quantile(ret_valid, quantile)
        # Return as positive loss magnitude
        var[f] = -var_value if var_value < 0 else 0.0

    return var[0] if squeeze else var


def compute_var_parametric(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Parametric VaR assuming normal distribution.

    Args:
        returns: Return series (T,) or (T, F)
        confidence_level: Confidence level
        min_periods: Minimum periods required

    Returns:
        VaR (positive loss value), scalar or shape (F,)
    """
    _validate_confidence_level(confidence_level)

    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    var = np.full(F, np.nan)

    # Z-score for confidence level
    z_score = stats.norm.ppf(1.0 - confidence_level)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        mean_ret = np.mean(ret_valid)
        std_ret = np.std(ret_valid, ddof=1)

        if not np.isfinite(std_ret) or std_ret <= 0:
            continue

        # VaR = mean + z * std (z is negative for losses)
        var_value = mean_ret + z_score * std_ret
        var[f] = -var_value if var_value < 0 else 0.0

    return var[0] if squeeze else var


def compute_var_cornish_fisher(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    min_periods: int = 30,
) -> np.ndarray:
    """
    Cornish-Fisher VaR adjusting for skewness and kurtosis.

    Args:
        returns: Return series (T,) or (T, F)
        confidence_level: Confidence level
        min_periods: Minimum periods required (higher due to moment estimation)

    Returns:
        VaR (positive loss value), scalar or shape (F,)
    """
    _validate_confidence_level(confidence_level)

    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    var = np.full(F, np.nan)

    z = stats.norm.ppf(1.0 - confidence_level)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        mean_ret = np.mean(ret_valid)
        std_ret = np.std(ret_valid, ddof=1)
        skew = stats.skew(ret_valid, bias=False)
        kurt = stats.kurtosis(ret_valid, bias=False, fisher=True)  # Excess kurtosis

        if not np.isfinite(std_ret) or std_ret <= 0:
            continue
        if not np.isfinite(skew) or not np.isfinite(kurt):
            # An unavailable CF estimate is not a Gaussian estimate.
            continue

        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z**2 - 1) * skew / 6.0
            + (z**3 - 3 * z) * kurt / 24.0
            - (2 * z**3 - 5 * z) * skew**2 / 36.0
        )

        var_value = mean_ret + z_cf * std_ret
        var[f] = -var_value if var_value < 0 else 0.0

    return var[0] if squeeze else var


def compute_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical",
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that loss exceeds VaR threshold.

    Args:
        returns: Return series (T,) or (T, F)
        confidence_level: Confidence level
        method: "historical" or "parametric"
        min_periods: Minimum periods required

    Returns:
        CVaR (positive loss value), scalar or shape (F,)
    """
    _validate_confidence_level(confidence_level)

    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    cvar = np.full(F, np.nan)

    if method == "historical":
        # Historical CVaR: mean of returns below VaR threshold
        quantile = 1.0 - confidence_level

        for f in range(F):
            ret_f = returns[:, f]
            valid = np.isfinite(ret_f)
            n_valid = np.sum(valid)

            if n_valid < min_periods:
                continue

            ret_valid = ret_f[valid]

            cvar[f] = max(0.0, empirical_expected_shortfall(ret_valid, confidence_level))

    elif method == "parametric":
        # Parametric CVaR for normal distribution
        z = stats.norm.ppf(1.0 - confidence_level)
        pdf_at_z = stats.norm.pdf(z)

        for f in range(F):
            ret_f = returns[:, f]
            valid = np.isfinite(ret_f)
            n_valid = np.sum(valid)

            if n_valid < min_periods:
                continue

            ret_valid = ret_f[valid]

            mean_ret = np.mean(ret_valid)
            std_ret = np.std(ret_valid, ddof=1)

            if not np.isfinite(std_ret) or std_ret <= 0:
                continue

            # CVaR = mean + std * E[Z | Z < z] = mean - std * pdf(z) / (1 - CL)
            cvar_value = mean_ret - std_ret * pdf_at_z / (1.0 - confidence_level)
            cvar[f] = -cvar_value if cvar_value < 0 else 0.0

    else:
        raise ValueError(f"Unknown CVaR method: {method}")

    return cvar[0] if squeeze else cvar


def compute_var_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical",
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both VaR and CVaR efficiently.

    Args:
        returns: Return series (T,) or (T, F)
        confidence_level: Confidence level
        method: "historical", "parametric", or "cornish_fisher"
        min_periods: Minimum periods required

    Returns:
        (VaR, CVaR) both as positive loss values, scalar or shape (F,)
    """
    _validate_confidence_level(confidence_level)
    var = compute_var(returns, confidence_level, method, min_periods)

    if method == "cornish_fisher":
        raise ValueError("Cornish-Fisher ES is unsupported; request historical ES separately")
    cvar = compute_cvar(returns, confidence_level, method, min_periods)

    return var, cvar


def empirical_expected_shortfall(returns, confidence_level=0.95):
    """Signed empirical loss ES v2: fixed tail mass, fractional boundary.

    Equal observation weights; finite observations only. This is a descriptive
    same-period estimate, not an inference/adequate-tail-sample certificate.
    compute_cvar is the separately documented positive-loss-clamped display.
    """
    _validate_confidence_level(confidence_level)
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("empirical ES requires a one-dimensional return sample")
    losses = np.sort(-values[np.isfinite(values)])[::-1]
    if not len(losses):
        return float("nan")
    mass = (1.0 - confidence_level) * len(losses)
    full = int(np.floor(mass))
    fraction = mass - full
    return float((losses[:full].sum() + (fraction * losses[full] if full < len(losses) else 0.)) / mass)
