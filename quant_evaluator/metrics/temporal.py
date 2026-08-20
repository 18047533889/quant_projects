"""
Temporal stability metrics: autocorrelation and factor persistence analysis.

Reference implementation for measuring factor temporal properties and stability.

True-time-axis policy: all lag-based estimators in this module
(``compute_autocorrelation``, ``compute_half_life``) use pairwise-finite lag
alignment — for lag k, only pairs (t, t-k) where BOTH original positions are
finite contribute. Observations are never NaN-compressed before lagging:
compression destroys calendar alignment and fabricates pairs across gaps.
"""

from typing import Tuple
import numpy as np
from scipy import stats


def compute_autocorrelation(
    series: np.ndarray,
    max_lag: int = 20,
    min_obs: int = 30,
) -> np.ndarray:
    """
    Compute autocorrelation function (ACF) for time series.

    True-time-axis semantics: for lag k, only pairs (t, t-k) where BOTH
    original positions are finite contribute. Missing observations are never
    compressed out before lagging — calendar gaps destroy real lag alignment
    and this implementation keeps it intact.

    The estimator is the standard correlation between the lag-k pair samples
    (x_t, x_{t-k}), computed with pair-specific means, so it is unaffected by
    where the NaNs sit.

    Args:
        series: Time series (T,) or (T, F) for multiple factors
        max_lag: Maximum lag to compute
        min_obs: Minimum observations required

    Returns:
        acf: shape (max_lag+1,) or (max_lag+1, F)
        acf[0] is always 1.0 (correlation with self)
    """
    one_dimensional = series.ndim == 1
    if one_dimensional:
        series = series.reshape(-1, 1)

    T, F = series.shape

    if T < min_obs:
        acf = np.full((max_lag + 1, F), np.nan, dtype=np.float64)
        return acf[:, 0] if one_dimensional else acf

    acf = np.full((max_lag + 1, F), np.nan, dtype=np.float64)

    for f in range(F):
        ts = series[:, f]
        finite = np.isfinite(ts)
        n_finite = int(np.sum(finite))

        if n_finite < min_obs:
            continue

        # Lag 0: correlation of the finite values with themselves.
        acf[0, f] = 1.0

        for lag in range(1, max_lag + 1):
            # Pairs (t, t-lag) where BOTH original positions are finite.
            pair_mask = finite[lag:] & finite[:-lag]  # length T - lag
            n_pairs = int(np.sum(pair_mask))
            if n_pairs < min_obs:
                break

            x_prev = ts[:-lag][pair_mask]  # x_{t-lag}
            x_curr = ts[lag:][pair_mask]   # x_t

            # Correlation between the two aligned pair samples.
            x_prev_c = x_prev - np.mean(x_prev)
            x_curr_c = x_curr - np.mean(x_curr)

            denom = np.sqrt(np.sum(x_prev_c ** 2) * np.sum(x_curr_c ** 2))
            if denom <= 0 or not np.isfinite(denom):
                # Constant pair sample(s): correlation undefined.
                continue

            acf[lag, f] = float(np.sum(x_prev_c * x_curr_c) / denom)

    return acf[:, 0] if one_dimensional else acf


def compute_ic_autocorrelation(
    ic_series: np.ndarray,
    max_lag: int = 20,
    min_obs: int = 30,
) -> np.ndarray:
    """
    Compute autocorrelation of IC series.

    High IC autocorrelation indicates persistent factor performance.

    Args:
        ic_series: Daily IC series (T, F)
        max_lag: Maximum lag
        min_obs: Minimum observations

    Returns:
        ic_acf: shape (max_lag+1, F)
    """
    return compute_autocorrelation(ic_series, max_lag=max_lag, min_obs=min_obs)


def compute_rank_stability(
    factor_values: np.ndarray,
    lag: int = 1,
    method: str = "spearman",
) -> np.ndarray:
    """
    Compute rank stability across time.

    Measures correlation of factor ranks between t and t+lag.
    High rank stability indicates persistent factor ordering.

    Args:
        factor_values: Factor values (T, N, F)
        lag: Time lag for stability measurement
        method: "spearman" or "pearson"

    Returns:
        stability: shape (T-lag, F) - rank correlation at each time
    """
    T, N, F = factor_values.shape

    if lag >= T:
        raise ValueError(f"lag ({lag}) must be less than T ({T})")

    stability = np.full((T - lag, F), np.nan, dtype=np.float64)

    corr_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

    for f in range(F):
        for t in range(T - lag):
            factor_t = factor_values[t, :, f]
            factor_t_lag = factor_values[t + lag, :, f]

            # Valid observations in both periods
            valid_mask = np.isfinite(factor_t) & np.isfinite(factor_t_lag)

            if np.sum(valid_mask) < 10:
                continue

            factor_t_valid = factor_t[valid_mask]
            factor_t_lag_valid = factor_t_lag[valid_mask]

            # Check for constants
            if len(np.unique(factor_t_valid)) == 1 or len(np.unique(factor_t_lag_valid)) == 1:
                continue

            if method == "spearman":
                corr, _ = stats.spearmanr(factor_t_valid, factor_t_lag_valid)
            else:
                corr, _ = stats.pearsonr(factor_t_valid, factor_t_lag_valid)

            stability[t, f] = corr

    return stability


def compute_mean_rank_stability(
    factor_values: np.ndarray,
    lag: int = 1,
    method: str = "spearman",
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute time-averaged rank stability.

    Args:
        factor_values: Factor values (T, N, F)
        lag: Time lag
        method: "spearman" or "pearson"
        min_periods: Minimum valid periods

    Returns:
        mean_stability: shape (F,)
    """
    stability = compute_rank_stability(factor_values, lag=lag, method=method)

    valid_periods = np.sum(~np.isnan(stability), axis=0)

    with np.errstate(invalid='ignore'):
        mean_stability = np.nanmean(stability, axis=0)

    insufficient = valid_periods < min_periods
    mean_stability = np.where(insufficient, np.nan, mean_stability)

    return mean_stability


def compute_factor_turnover_rate(
    factor_values: np.ndarray,
    quantile: float = 0.9,
) -> np.ndarray:
    """
    Compute turnover rate of top/bottom quantile membership.

    Measures how frequently assets enter/exit extreme factor quantiles.
    High turnover indicates unstable factor ordering.

    Args:
        factor_values: Factor values (T, N, F)
        quantile: Quantile threshold (0.9 = top 10%, 0.1 = bottom 10%)

    Returns:
        turnover_rate: shape (T-1, F) - fraction of positions changed
    """
    T, N, F = factor_values.shape

    if quantile <= 0 or quantile >= 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    turnover_rate = np.full((T - 1, F), np.nan, dtype=np.float64)

    for f in range(F):
        for t in range(T - 1):
            factor_t = factor_values[t, :, f]
            factor_t1 = factor_values[t + 1, :, f]

            # Valid in both periods
            valid_mask = np.isfinite(factor_t) & np.isfinite(factor_t1)

            if np.sum(valid_mask) < 10:
                continue

            factor_t_valid = factor_t[valid_mask]
            factor_t1_valid = factor_t1[valid_mask]

            # Determine quantile membership
            if quantile > 0.5:
                threshold_t = np.nanquantile(factor_t_valid, quantile)
                threshold_t1 = np.nanquantile(factor_t1_valid, quantile)
                in_quantile_t = factor_t_valid >= threshold_t
                in_quantile_t1 = factor_t1_valid >= threshold_t1
            else:
                threshold_t = np.nanquantile(factor_t_valid, quantile)
                threshold_t1 = np.nanquantile(factor_t1_valid, quantile)
                in_quantile_t = factor_t_valid <= threshold_t
                in_quantile_t1 = factor_t1_valid <= threshold_t1

            # Count changes
            changed = in_quantile_t != in_quantile_t1
            turnover_rate[t, f] = np.mean(changed)

    return turnover_rate


def compute_half_life(
    ic_series: np.ndarray,
    min_periods: int = 60,
) -> np.ndarray:
    """
    Estimate IC half-life using AR(1) model.

    Half-life = -log(2) / log(phi) where phi is AR(1) coefficient.
    Measures how quickly factor predictive power decays.

    True-time-axis semantics: phi is estimated on (t, t-1) pairs where BOTH
    original positions are finite. NaNs are never compressed out before
    lagging, so calendar gaps do not fabricate adjacent pairs.

    Args:
        ic_series: Daily IC series (T, F)
        min_periods: Minimum periods for AR estimation

    Returns:
        half_life: shape (F,) - half-life in periods (NaN if phi >= 1 or phi <= 0)
    """
    T, F = ic_series.shape
    half_life = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = ic_series[:, f]
        finite = np.isfinite(ic_f)
        n_finite = int(np.sum(finite))

        if n_finite < min_periods:
            continue

        # AR(1): IC_t = phi * IC_{t-1} + epsilon, estimated on pairs
        # (t, t-1) where BOTH original positions are finite. Missing
        # observations are never compressed out before lagging — that would
        # fabricate adjacent pairs across calendar gaps.
        pair_mask = finite[1:] & finite[:-1]  # length T - 1
        x = ic_f[:-1][pair_mask]  # IC_{t-1}
        y = ic_f[1:][pair_mask]   # IC_t

        if len(y) < min_periods - 1:
            continue

        # OLS estimate of phi (no intercept, matching the original model)
        denom = np.sum(x * x)
        if not np.isfinite(denom) or denom <= 0:
            continue

        phi = np.sum(x * y) / denom

        # Half-life is only meaningful for 0 < phi < 1
        if 0 < phi < 1:
            half_life[f] = -np.log(2) / np.log(phi)

    return half_life
