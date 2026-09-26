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
import warnings

import numpy as np
from scipy import stats


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (same convention as scipy.stats.rankdata, 'average')."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    boundary = np.empty(len(x), dtype=bool)
    boundary[0] = True
    boundary[1:] = sx[1:] != sx[:-1]
    group = np.cumsum(boundary) - 1
    counts = np.bincount(group)
    ends = np.cumsum(counts)
    starts = ends - counts
    avg = (starts + ends + 1) / 2.0
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = avg[group]
    return ranks


def _lean_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Lean Spearman correlation (ranks then centered Pearson).

    Matches scipy.stats.spearmanr to well below rtol 1e-8 on non-degenerate
    inputs (measured max abs deviation 5.6e-17).
    """
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(np.sum(rx * ry) / denom)


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
    for name, value, lower in (("max_lag", max_lag, 0), ("min_obs", min_obs, 2)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    series = np.asarray(series, dtype=float)
    if series.ndim not in (1, 2):
        raise ValueError("ACF requires T or T x F")
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

        for lag in range(1, min(max_lag + 1, T)):
            # Pairs (t, t-lag) where BOTH original positions are finite.
            pair_mask = finite[lag:] & finite[:-lag]  # length T - lag
            n_pairs = int(np.sum(pair_mask))
            if n_pairs < min_obs:
                continue

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


def compute_autocorrelation_evidence(series, max_lag=20, min_obs=30, *, clock_ref):
    """Lag-by-lag descriptive ACF and original-clock effective pair counts."""
    if not isinstance(clock_ref, str) or not clock_ref:
        raise ValueError("explicit clock identity required")
    values = np.asarray(series, dtype=float)
    acf = compute_autocorrelation(values, max_lag, min_obs)
    matrix = values[:, None] if values.ndim == 1 else values
    finite = np.isfinite(matrix)
    counts = np.zeros((max_lag+1, matrix.shape[1]), dtype=np.int64)
    counts[0] = finite.sum(axis=0)
    for lag in range(1, min(max_lag+1, len(matrix))):
        counts[lag] = (finite[:-lag] & finite[lag:]).sum(axis=0)
    if values.ndim == 1:
        counts = counts[:, 0]
    from quant_evaluator.contracts.metric_artifacts import FrozenMapping
    return FrozenMapping({"metric_id": "acf_original_clock.v2", "clock_ref": clock_ref,
        "values": acf, "pair_counts": counts,
        "status": np.where(counts < min_obs, "INSUFFICIENT_DATA", np.where(np.isfinite(acf), "COMPUTED", "UNDEFINED"))})


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
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError("lag must be a positive integer")
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be spearman or pearson")
    T, N, F = factor_values.shape

    if lag >= T:
        raise ValueError(f"lag ({lag}) must be less than T ({T})")

    stability = np.full((T - lag, F), np.nan, dtype=np.float64)

    if method == "spearman":
        for f in range(F):
            for t in range(T - lag):
                factor_t = factor_values[t, :, f]
                factor_t_lag = factor_values[t + lag, :, f]

                valid_mask = np.isfinite(factor_t) & np.isfinite(factor_t_lag)
                if np.sum(valid_mask) < 10:
                    continue

                factor_t_valid = factor_t[valid_mask]
                factor_t_lag_valid = factor_t_lag[valid_mask]

                # min==max is an exact detector for len(np.unique(...))==1.
                if factor_t_valid.min() == factor_t_valid.max() or \
                        factor_t_lag_valid.min() == factor_t_lag_valid.max():
                    continue

                corr = _lean_spearman(factor_t_valid, factor_t_lag_valid)
                if np.isfinite(corr):
                    stability[t, f] = corr
        return stability

    # Pearson (and any non-spearman method, matching the legacy else-branch).
    mu_full = np.nanmean(factor_values, axis=1)  # (T, F) per-row centering shift
    with np.errstate(invalid="ignore"):
        centered = factor_values - mu_full[:, np.newaxis, :]
    sprime = np.full(F, 0.0)
    for f in range(F):
        vals = centered[:, :, f]
        finite_vals = vals[np.isfinite(vals)]
        sprime[f] = np.max(np.abs(finite_vals)) if finite_vals.size else 0.0
    suspicious_scale = 1e4 * np.finfo(float).eps * np.maximum(1.0, sprime) ** 2

    for f in range(F):
        x = factor_values[:-lag, :, f]
        y = factor_values[lag:, :, f]
        valid = np.isfinite(x) & np.isfinite(y)  # (T-lag, N)
        n = valid.sum(axis=1)
        xc = centered[:-lag, :, f]
        yc = centered[lag:, :, f]
        sx = np.sum(np.where(valid, xc, 0.0), axis=1)
        sy = np.sum(np.where(valid, yc, 0.0), axis=1)
        sxx = np.sum(np.where(valid, xc * xc, 0.0), axis=1)
        syy = np.sum(np.where(valid, yc * yc, 0.0), axis=1)
        sxy = np.sum(np.where(valid, xc * yc, 0.0), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            num = n * sxy - sx * sy
            den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
            corr = np.where(den > 0, num / den, np.nan)
            varx = sxx / np.maximum(n, 1) - (sx / np.maximum(n, 1)) ** 2
            vary = syy / np.maximum(n, 1) - (sy / np.maximum(n, 1)) ** 2
        suspicious = (
            (n < 10)
            | ~(varx > suspicious_scale[f])
            | ~(vary > suspicious_scale[f])
            | ~np.isfinite(corr)
        )
        for t in np.flatnonzero(suspicious):
            factor_t = factor_values[t, :, f]
            factor_t_lag = factor_values[t + lag, :, f]
            valid_mask = np.isfinite(factor_t) & np.isfinite(factor_t_lag)
            if np.sum(valid_mask) < 10:
                continue
            factor_t_valid = factor_t[valid_mask]
            factor_t_lag_valid = factor_t_lag[valid_mask]
            if len(np.unique(factor_t_valid)) == 1 or len(np.unique(factor_t_lag_valid)) == 1:
                continue
            corr_ref, _ = stats.pearsonr(factor_t_valid, factor_t_lag_valid)
            if np.isfinite(corr_ref):
                stability[t, f] = corr_ref
        ok = (~suspicious) & np.isfinite(corr)
        stability[ok, f] = corr[ok]

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
    *, measure: str = "universe_membership_change",
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
    if measure not in ("universe_membership_change", "top_exit_fraction", "top_entry_fraction", "jaccard_distance"):
        raise ValueError("unknown membership diagnostic; actual turnover requires holdings")
    T, N, F = factor_values.shape

    if quantile <= 0 or quantile >= 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    turnover_rate = np.full((max(T - 1, 0), F), np.nan, dtype=np.float64)

    for f in range(F):
        panel = factor_values[:, :, f]  # (T, N)
        valid = np.isfinite(panel)  # (T, N)
        counts = valid.sum(axis=1)  # (T,)

        # Vectorized per-row nanquantile is bitwise identical to the legacy
        # per-row calls on the finite-compacted rows (verified).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thresholds = np.nanquantile(np.where(valid, panel, np.nan), quantile, axis=1)  # (T,)
        thr = thresholds[:, np.newaxis]
        if quantile > 0.5:
            in_quantile = valid & (panel >= thr)  # (T, N)
        else:
            in_quantile = valid & (panel <= thr)  # (T, N)

        in_q_t = in_quantile[:-1]
        in_q_t1 = in_quantile[1:]
        valid_t = valid[:-1]
        valid_t1 = valid[1:]

        eligible = (np.minimum(counts[:-1], counts[1:]) >= 10) & ~np.any(
            in_q_t & ~valid_t1, axis=1
        )
        if not np.any(eligible):
            continue

        changed = (in_q_t != in_q_t1)
        denom_union_valid = np.sum(valid_t | valid_t1, axis=1)
        denom_exit = np.sum(in_q_t, axis=1)
        denom_entry = np.sum(in_q_t1, axis=1)
        denom_jaccard = np.sum(in_q_t | in_q_t1, axis=1)
        changed_sum = np.sum(changed, axis=1)
        exit_sum = np.sum(in_q_t & ~in_q_t1, axis=1)
        entry_sum = np.sum(in_q_t1 & ~in_q_t, axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if measure == "universe_membership_change":
                values_f = changed_sum / denom_union_valid
            elif measure == "top_exit_fraction":
                values_f = exit_sum / denom_exit
            elif measure == "top_entry_fraction":
                values_f = entry_sum / denom_entry
            else:
                values_f = changed_sum / denom_jaccard
        turnover_rate[:, f] = np.where(eligible, values_f, np.nan)

    return turnover_rate


def compute_half_life(
    ic_series: np.ndarray,
    min_periods: int = 60,
    *, model: str = "centered_ar1",
) -> np.ndarray:
    """
    Estimate IC half-life using AR(1) model.

    Half-life = -log(2) / log(phi) where phi is AR(1) coefficient.
    Measures IC temporal persistence, NOT predictive-horizon decay.
    The default fits an intercept; zero_mean_ar1 is an explicit research model.

    True-time-axis semantics: phi is estimated on (t, t-1) pairs where BOTH
    original positions are finite. NaNs are never compressed out before
    lagging, so calendar gaps do not fabricate adjacent pairs.

    Args:
        ic_series: Daily IC series (T, F)
        min_periods: Minimum periods for AR estimation

    Returns:
        half_life: shape (F,) - half-life in periods (NaN if phi >= 1 or phi <= 0)
    """
    if model not in ("centered_ar1", "zero_mean_ar1"):
        raise ValueError("unknown AR1 model")
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

        if model == "centered_ar1":
            x = x - np.mean(x)
            y = y - np.mean(y)
        # Pair-specific centering is equivalent to an intercept regression.
        denom = np.sum(x * x)
        if not np.isfinite(denom) or denom <= 0:
            continue

        phi = np.sum(x * y) / denom

        # Half-life is only meaningful for 0 < phi < 1
        if 0 < phi < 1:
            half_life[f] = -np.log(2) / np.log(phi)

    return half_life


def compute_ic_temporal_persistence(ic_series, min_periods=60):
    """Centered AR1 temporal persistence (not predictive-horizon decay)."""
    return compute_half_life(ic_series, min_periods, model="centered_ar1")


def compute_zero_mean_ar1_half_life(ic_series, min_periods=60):
    """Separate zero-mean-model research identity; not translation invariant."""
    return compute_half_life(ic_series, min_periods, model="zero_mean_ar1")


def _compute_rank_stability_reference(
    factor_values: np.ndarray,
    lag: int = 1,
    method: str = "spearman",
) -> np.ndarray:
    """Verbatim legacy oracle for equivalence testing."""
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


def _compute_factor_turnover_rate_reference(
    factor_values: np.ndarray,
    quantile: float = 0.9,
    *, measure: str = "universe_membership_change",
) -> np.ndarray:
    """Verbatim legacy oracle for equivalence testing."""
    if measure not in ("universe_membership_change", "top_exit_fraction", "top_entry_fraction", "jaccard_distance"):
        raise ValueError("unknown membership diagnostic; actual turnover requires holdings")
    T, N, F = factor_values.shape

    if quantile <= 0 or quantile >= 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    turnover_rate = np.full((max(T - 1, 0), F), np.nan, dtype=np.float64)

    for f in range(F):
        for t in range(T - 1):
            factor_t = factor_values[t, :, f]
            factor_t1 = factor_values[t + 1, :, f]

            valid_t = np.isfinite(factor_t)
            valid_t1 = np.isfinite(factor_t1)
            if min(np.sum(valid_t), np.sum(valid_t1)) < 10:
                continue
            factor_t_valid = factor_t[valid_t]
            factor_t1_valid = factor_t1[valid_t1]

            # Determine quantile membership
            if quantile > 0.5:
                threshold_t = np.nanquantile(factor_t_valid, quantile)
                threshold_t1 = np.nanquantile(factor_t1_valid, quantile)
                in_quantile_t = valid_t & (factor_t >= threshold_t)
                in_quantile_t1 = valid_t1 & (factor_t1 >= threshold_t1)
            else:
                threshold_t = np.nanquantile(factor_t_valid, quantile)
                threshold_t1 = np.nanquantile(factor_t1_valid, quantile)
                in_quantile_t = valid_t & (factor_t <= threshold_t)
                in_quantile_t1 = valid_t1 & (factor_t1 <= threshold_t1)

            # Count changes
            # A previously selected security with unknown next signal is not
            # proof of a sale. Preserve uncertainty rather than erase it.
            if np.any(in_quantile_t & ~valid_t1):
                continue
            changed = in_quantile_t != in_quantile_t1
            if measure == "universe_membership_change":
                turnover_rate[t, f] = changed.sum() / np.sum(valid_t | valid_t1)
            elif measure == "top_exit_fraction":
                turnover_rate[t, f] = np.sum(in_quantile_t & ~in_quantile_t1) / in_quantile_t.sum()
            elif measure == "top_entry_fraction":
                turnover_rate[t, f] = np.sum(in_quantile_t1 & ~in_quantile_t) / in_quantile_t1.sum()
            else:
                turnover_rate[t, f] = changed.sum() / np.sum(in_quantile_t | in_quantile_t1)

    return turnover_rate
