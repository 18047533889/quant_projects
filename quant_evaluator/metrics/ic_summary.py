"""
IC summary statistics: ICIR, t-statistics, and decay analysis.

Reference implementation for IC-based performance metrics with proper
statistical inference and temporal decay patterns.
"""

from typing import Tuple
import warnings

import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, _reject_boolean_ic_series


def compute_icir(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Information Coefficient Information Ratio (ICIR).

    ICIR = mean(IC) / std(IC), measures consistency of IC signal.

    Args:
        ic_series: Daily IC series (T, F)
        min_periods: Minimum periods required

    Returns:
        ICIR array of shape (F,), NaN if insufficient periods or zero std
    """
    values = np.asarray(ic_series)
    _reject_boolean_ic_series(values)
    if values.ndim != 2:
        raise ValueError("ic_series must have shape (T, F)")
    if isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer)):
        raise TypeError("min_periods must be an integer")
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2 for sample standard deviation")

    values = values.astype(np.float64, copy=False)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for factor_index in range(values.shape[1]):
        finite = values[np.isfinite(values[:, factor_index]), factor_index]
        if finite.size < min_periods:
            continue
        # Raw ICIR is mean / sample std.  Only exact zero variance is
        # undefined; a small but genuine dispersion must not be thresholded
        # away because that changes the economic statistic by scale.
        if np.all(finite == finite[0]):
            continue
        std_ic = float(np.std(finite, ddof=1))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            value = float(np.mean(finite) / std_ic)
        if np.isfinite(value):
            result[factor_index] = value
    return result


def compute_ic_tstat(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute t-statistic for IC series testing H0: mean(IC) = 0.

    Args:
        ic_series: Daily IC series (T, F)
        min_periods: Minimum periods required

    Returns:
        (t_stat, p_value) arrays of shape (F,)
        Two-tailed p-value for t-test

    Raises:
        ValueError: If ic_series is boolean (or contains Python bools)
    """
    _reject_boolean_ic_series(np.asarray(ic_series))

    T, F = ic_series.shape
    t_stats = np.full(F, np.nan, dtype=np.float64)
    p_values = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = ic_series[:, f]
        valid_ic = ic_f[~np.isnan(ic_f)]

        if len(valid_ic) < min_periods:
            continue

        # Constant IC series (std=0) makes scipy return a degenerate
        # t/p pair (t ~ 1e16, p = 0) — fake ultra-significance.
        std_ic = np.std(valid_ic)
        if not np.isfinite(std_ic) or std_ic < 1e-10:
            continue

        # One-sample t-test against H0: mean = 0
        t_stat, p_val = stats.ttest_1samp(valid_ic, 0.0)
        t_stats[f] = t_stat
        p_values[f] = p_val

    return t_stats, p_values


def compute_ic_decay(
    factor_batch: FactorBatch,
    label_bundles: Tuple[LabelBundle, ...],
    method: str = "pearson",
    min_assets: int = 10,
) -> np.ndarray:
    """
    Compute IC decay across multiple horizons.

    Measures how factor predictive power decays over time by computing
    IC for each horizon in label_bundles.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundles: Tuple of labels at different horizons (1d, 5d, 20d, etc.)
        method: "pearson" or "spearman"
        min_assets: Minimum valid assets per day

    Returns:
        decay_matrix: shape (num_horizons, F)
        Each row is mean IC for that horizon
    """
    num_horizons = len(label_bundles)
    F = factor_batch.num_factors

    decay_matrix = np.full((num_horizons, F), np.nan, dtype=np.float64)

    for h_idx, label_bundle in enumerate(label_bundles):
        ic_series, _ = compute_daily_ic(
            factor_batch, label_bundle, method=method, min_assets=min_assets
        )

        # Mean IC for this horizon
        with np.errstate(invalid='ignore'):
            mean_ic = np.nanmean(ic_series, axis=0)

        decay_matrix[h_idx, :] = mean_ic

    return decay_matrix


def compute_ic_stability(
    ic_series: np.ndarray,
    window_size: int = 60,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling IC stability metrics.

    .. deprecated-ish / EXPERIMENTAL ::
        :status experimental:
        This first-half-vs-second-half correlation within a rolling window is
        a WEAK stability proxy: it is noisy for short windows, insensitive to
        sign persistence, and says nothing about regime change. Prefer the
        richer stability family in this module:
        ``compute_rolling_ic_stats``, ``compute_yearly_quarterly_dispersion``
        and ``compute_change_point_score``. This function is retained for
        backward compatibility and is NOT registered as a stable metric.

    Stability = correlation between first-half and second-half IC within rolling windows.
    High stability indicates consistent factor performance.

    Args:
        ic_series: Daily IC series (T, F)
        window_size: Rolling window size (must be even, >= 40)
        min_periods: Minimum valid periods in each half

    Returns:
        (stability_series, valid_windows)
        stability_series: shape (num_windows, F)
        valid_windows: count of valid windows per factor (F,)
    """
    if window_size < 40 or window_size % 2 != 0:
        raise ValueError(f"window_size must be even and >= 40, got {window_size}")

    T, F = ic_series.shape
    half_window = window_size // 2

    num_windows = T - window_size + 1
    if num_windows <= 0:
        return np.full((0, F), np.nan), np.zeros(F, dtype=np.int32)

    arr = np.asarray(ic_series, dtype=np.float64)
    swv = sliding_window_view(arr, window_size, axis=0)  # (num_windows, F, window_size)
    A = swv[:, :, :half_window]  # first halves
    B = swv[:, :, half_window:]  # second halves

    valid = np.isfinite(A) & np.isfinite(B)  # (num_windows, F, half)
    n = valid.sum(axis=2)
    s1 = np.sum(np.where(valid, A, 0.0), axis=2)
    s2 = np.sum(np.where(valid, B, 0.0), axis=2)
    s11 = np.sum(np.where(valid, A * A, 0.0), axis=2)
    s22 = np.sum(np.where(valid, B * B, 0.0), axis=2)
    s12 = np.sum(np.where(valid, A * B, 0.0), axis=2)

    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * s12 - s1 * s2
        den = np.sqrt((n * s11 - s1 * s1) * (n * s22 - s2 * s2))
        corr = np.where(den > 0, num / den, np.nan)

    # Suspicious cells (constant halves, insufficient data, or numerically
    # degenerate) fall back to the verbatim legacy computation so behavior
    # is bit-identical wherever the fast path's precision could waver.
    sprime = np.full(F, 0.0)
    for f in range(F):
        col = arr[:, f]
        fin = col[np.isfinite(col)]
        sprime[f] = np.max(np.abs(fin)) if fin.size else 0.0
    scale = 1e4 * np.finfo(float).eps * np.maximum(1.0, sprime) ** 2
    with np.errstate(invalid="ignore"):
        var1 = s11 / np.maximum(n, 1) - (s1 / np.maximum(n, 1)) ** 2
        var2 = s22 / np.maximum(n, 1) - (s2 / np.maximum(n, 1)) ** 2
    suspicious = (
        (n < min_periods)
        | ~(var1 > scale[None, :])
        | ~(var2 > scale[None, :])
        | ~np.isfinite(corr)
    )

    stability_series = np.full((num_windows, F), np.nan, dtype=np.float64)
    ok = (~suspicious) & np.isfinite(corr)
    stability_series[ok] = corr[ok]

    if np.any(suspicious):
        for f in range(F):
            for w in np.flatnonzero(suspicious[:, f]):
                first_half = ic_series[w:w+half_window, f]
                second_half = ic_series[w+half_window:w+window_size, f]

                # Correlate the same time periods across halves: drop a period
                # only when EITHER half is NaN.  Dropping per half independently
                # misaligns the series (and crashes corrcoef on length mismatch).
                v = ~np.isnan(first_half) & ~np.isnan(second_half)
                first_valid = first_half[v]
                second_valid = second_half[v]

                if len(first_valid) < min_periods:
                    continue

                # Pearson correlation between halves
                if np.std(first_valid) == 0 or np.std(second_valid) == 0:
                    continue

                stability_series[w, f] = np.corrcoef(first_valid, second_valid)[0, 1]

    valid_windows = np.sum(~np.isnan(stability_series), axis=0)

    return stability_series, valid_windows


def _rolling_mean_nan(series: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """NaN-aware rolling mean without pandas. Returns NaN where insufficient.

    Vectorized form; arithmetic (cumulative sums of finite values / counts)
    is identical to the legacy per-end loop, so results are bitwise equal.
    """
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(series)
    x = np.where(finite, series, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    fsum = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
    ends = np.arange(n)
    starts = np.maximum(ends - window + 1, 0)
    n_fin = fsum[ends + 1] - fsum[starts]
    ok = n_fin >= min_periods
    out[ok] = (csum[ends[ok] + 1] - csum[starts[ok]]) / n_fin[ok]
    return out


def _rolling_mean_nan_reference(series: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Verbatim legacy oracle for equivalence testing."""
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float64)
    csum = np.concatenate([[0.0], np.nancumsum(series)])
    finite = np.isfinite(series).astype(np.float64)
    fsum = np.concatenate([[0.0], np.cumsum(finite)])
    for end in range(n):
        start = max(0, end - window + 1)
        n_fin = fsum[end + 1] - fsum[start]
        if n_fin >= min_periods:
            out[end] = (csum[end + 1] - csum[start]) / n_fin
    return out


def compute_rolling_ic_stats(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> dict:
    """
    Rich rolling-window IC stability statistics (EXPERIMENTAL family).

    Computes, per factor:
        - rolling_ic_mean: (T,) rolling mean IC over ``window`` periods
          (NaN before ``min_periods`` finite observations accumulate)
        - rolling_ic_ir: (T,) rolling mean / rolling std (ddof=1) of IC
        - positive_ic_ratio: () fraction of finite daily IC values > 0
        - sign_survival: () mean run-length of consistent-sign IC (a run is a
          maximal streak of same-sign finite IC values; longer runs = more
          persistent signal)
        - worst_rolling_ic: () minimum rolling mean IC (most negative regime)
        - recent_vs_full: () mean of the last ``window`` finite IC values
          minus the full-sample mean (positive = recent improvement)

    All outputs are NaN-aware and fail-closed: constant or insufficient
    input yields NaN for the affected statistics.

    Args:
        ic_series: Daily IC series (T, F)
        window: Rolling window length
        min_periods: Minimum finite observations per window

    Returns:
        dict mapping statistic name -> np.ndarray of shape (T,), (T,) or ()
        depending on the statistic (per factor arrays are stacked when F > 1:
        rolling stats have shape (T, F), scalars have shape (F,))
    """
    arr = np.asarray(ic_series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    shape = arr.shape
    T = shape[0]; F = shape[1]

    rolling_ic_mean = np.full((T, F), np.nan, dtype=np.float64)
    rolling_ic_ir = np.full((T, F), np.nan, dtype=np.float64)
    positive_ic_ratio = np.full(F, np.nan, dtype=np.float64)
    sign_survival = np.full(F, np.nan, dtype=np.float64)
    worst_rolling_ic = np.full(F, np.nan, dtype=np.float64)
    recent_vs_full = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = arr[:, f]
        finite = np.isfinite(ic_f)
        n_finite = int(np.sum(finite))

        if n_finite < min_periods:
            continue

        # Rolling mean via cumulative sums over expanding windows.
        roll_mean = _rolling_mean_nan(ic_f, window, min_periods)
        rolling_ic_mean[:, f] = roll_mean

        # Rolling std (ddof=1) and rolling IR, vectorized over all window ends
        # via cumulative sums of the globally centered series. Cells whose
        # std falls in the numerically delicate zone are recomputed verbatim.
        finite_w = np.isfinite(ic_f)
        vals_f = ic_f[finite_w]
        mu_glob = float(np.mean(vals_f))
        sprime = float(np.max(np.abs(vals_f))) if vals_f.size else 0.0
        x = np.where(finite_w, ic_f - mu_glob, 0.0)
        c1 = np.concatenate([[0.0], np.cumsum(x)])
        c2 = np.concatenate([[0.0], np.cumsum(x * x)])
        cf = np.concatenate([[0.0], np.cumsum(finite_w.astype(np.float64))])
        ends = np.arange(T)
        starts = np.maximum(ends - window + 1, 0)
        cnt = cf[ends + 1] - cf[starts]
        ssum = c1[ends + 1] - c1[starts]
        ssq = c2[ends + 1] - c2[starts]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_c = ssum / np.maximum(cnt, 1.0)
            # Sample variance (ddof=1), matching np.std(seg, ddof=1).
            var = (ssq / np.maximum(cnt, 1.0) - mean_c * mean_c) * (
                np.maximum(cnt, 1.0) / np.maximum(cnt - 1.0, 1.0)
            )
            std = np.sqrt(np.maximum(var, 0.0))
            mean_raw = mu_glob + mean_c
            ir = mean_raw / std
        eligible = cnt >= max(2, min_periods)
        fast_ok = eligible & (std > 1e-3 * max(1.0, sprime)) & (std > 1e-12)
        rolling_ic_ir[fast_ok, f] = ir[fast_ok]
        slow = np.flatnonzero(eligible & ~fast_ok)
        for end in slow:
            start = max(0, end - window + 1)
            seg = ic_f[start:end + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= max(2, min_periods):
                std_ref = np.std(seg, ddof=1)
                if np.isfinite(std_ref) and std_ref > 1e-12:
                    rolling_ic_ir[end, f] = np.mean(seg) / std_ref

        valid_mean = roll_mean[np.isfinite(roll_mean)]
        if valid_mean.size > 0:
            worst_rolling_ic[f] = float(np.min(valid_mean))

        # Positive IC ratio.
        vals = ic_f[finite]
        positive_ic_ratio[f] = float(np.mean(vals > 0))

        # Sign survival: mean run-length of consistent-sign finite IC.
        # Vectorized run-length detection yields the identical runs array,
        # so the mean is bitwise equal to the legacy loop's.
        signs = np.sign(vals)
        if len(signs) <= 1:
            runs = np.array([1], dtype=np.float64)
        else:
            change_pos = np.flatnonzero(np.diff(signs) != 0)
            cut = np.concatenate(([-1], change_pos, [len(signs) - 1]))
            runs = np.diff(cut).astype(np.float64)
        if len(runs) > 0:
            sign_survival[f] = float(np.mean(runs))

        # Recent vs full: mean of last `window` finite values vs full mean.
        if n_finite >= min_periods:
            recent = vals[-window:]
            if len(recent) >= min_periods:
                recent_vs_full[f] = float(np.mean(recent) - np.mean(vals))

    return {
        "rolling_ic_mean": rolling_ic_mean,
        "rolling_ic_ir": rolling_ic_ir,
        "positive_ic_ratio": positive_ic_ratio,
        "sign_survival": sign_survival,
        "worst_rolling_ic": worst_rolling_ic,
        "recent_vs_full": recent_vs_full,
    }


def compute_yearly_quarterly_dispersion(
    ic_series: np.ndarray,
    time_index=None,
) -> dict:
    """
    Per-factor dispersion of mean IC across calendar years and quarters.

    :status experimental:

    If ``time_index`` is provided (sequence of datetime-like or integer
    labels, one per period), periods are grouped by year and by (year,
    quarter) derived from the labels. Otherwise grouping is position-based:
    quarters of 63 periods and years of 252 periods.

    Returns a dict with per-factor arrays:
        - yearly_mean_ic_std: std (ddof=1) across yearly mean ICs; NaN if
          fewer than 2 complete years
        - quarterly_mean_ic_std: std across quarterly mean ICs; NaN if
          fewer than 2 complete quarters
        - n_years: number of year groups with at least one finite IC
        - n_quarters: number of quarter groups with at least one finite IC

    Fails closed: insufficient data -> NaN.
    """
    arr = np.asarray(ic_series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    shape = arr.shape
    T = shape[0]; F = shape[1]

    # Build group labels.
    if time_index is not None:
        ti = np.asarray(time_index)
        if ti.shape[0] != T:
            raise ValueError(
                f"time_index has length {ti.shape[0]}, expected {T}"
            )
        try:
            dt = np.asarray(ti, dtype="datetime64[D]")
            years = dt.astype("datetime64[Y]").astype(int)
            months = (dt.astype("datetime64[M]").astype(int) % 12) + 1
            quarters = ((months - 1) // 3) + 1
        except (TypeError, ValueError):
            # Integer position labels.
            pos = np.asarray(ti, dtype=np.int64)
            years = pos // 252
            quarters = pos // 63
        year_keys = years
        quarter_keys = years * 10 + quarters
    else:
        pos = np.arange(T)
        year_keys = pos // 252
        quarter_keys = pos // 63

    yearly_std = np.full(F, np.nan, dtype=np.float64)
    quarterly_std = np.full(F, np.nan, dtype=np.float64)
    n_years_arr = np.zeros(F, dtype=np.int64)
    n_quarters_arr = np.zeros(F, dtype=np.int64)

    for f in range(F):
        ic_f = arr[:, f]

        def _group_std(keys):
            means = []
            for k in np.unique(keys):
                sel = ic_f[keys == k]
                sel = sel[np.isfinite(sel)]
                if sel.size > 0:
                    means.append(np.mean(sel))
            means = np.asarray(means, dtype=np.float64)
            if means.size >= 2:
                return float(np.std(means, ddof=1)), int(means.size)
            return np.nan, int(means.size)

        y_std, y_n = _group_std(year_keys)
        q_std, q_n = _group_std(quarter_keys)
        yearly_std[f] = y_std
        quarterly_std[f] = q_std
        n_years_arr[f] = y_n
        n_quarters_arr[f] = q_n

    return {
        "yearly_mean_ic_std": yearly_std,
        "quarterly_mean_ic_std": quarterly_std,
        "n_years": n_years_arr,
        "n_quarters": n_quarters_arr,
    }


def compute_change_point_score(
    ic_series: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Maximum absolute mean shift between adjacent (non-overlapping) windows.

    :status experimental:

    For each factor and each split point t, compares the mean IC of the
    ``window`` periods ENDING at t with the mean IC of the ``window`` periods
    STARTING at t (adjacent non-overlapping windows, NaN-aware), and returns

        max_t |mean(window ending at t) - mean(window starting at t)|
            / full_sample_std

    i.e. the largest adjacent-window mean shift, normalized by the
    full-sample standard deviation of finite IC values. High values flag
    regime changes (e.g. an IC that flips sign or magnitude).

    Fails closed: constant series (zero std) or insufficient data -> NaN.

    Args:
        ic_series: Daily IC series (T, F)
        window: Rolling window length

    Returns:
        change_point_score: shape (F,)
    """
    arr = np.asarray(ic_series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    shape = arr.shape
    T = shape[0]; F = shape[1]

    if window < 2:
        window = 2

    scores = np.full(F, np.nan, dtype=np.float64)
    min_periods = max(2, window // 2)

    for f in range(F):
        ic_f = arr[:, f]
        vals = ic_f[np.isfinite(ic_f)]
        if vals.size < min_periods or vals.size < 2:
            continue

        full_std = np.std(vals, ddof=1)
        if not np.isfinite(full_std) or full_std < 1e-12:
            # Constant series: no meaningful shift normalization.
            continue

        # Split points where both adjacent windows exist in range.
        finite = np.isfinite(ic_f)
        x = np.where(finite, ic_f, 0.0)
        csum = np.concatenate([[0.0], np.cumsum(x)])
        fsum = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
        t_arr = np.arange(window, T - window + 1)
        left_n = fsum[t_arr] - fsum[t_arr - window]
        right_n = fsum[t_arr + window] - fsum[t_arr]
        with np.errstate(invalid="ignore", divide="ignore"):
            left_mean = (csum[t_arr] - csum[t_arr - window]) / left_n
            right_mean = (csum[t_arr + window] - csum[t_arr]) / right_n
        ok = (left_n >= min_periods) & (right_n >= min_periods)
        if not np.any(ok):
            continue
        shifts = np.abs(left_mean - right_mean)
        max_shift = float(np.max(shifts[ok]))
        scores[f] = float(max_shift / full_std)

    return scores


def _compute_ic_stability_reference(
    ic_series: np.ndarray,
    window_size: int = 60,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Verbatim legacy oracle for equivalence testing."""
    if window_size < 40 or window_size % 2 != 0:
        raise ValueError(f"window_size must be even and >= 40, got {window_size}")

    T, F = ic_series.shape
    half_window = window_size // 2

    num_windows = T - window_size + 1
    if num_windows <= 0:
        return np.full((0, F), np.nan), np.zeros(F, dtype=np.int32)

    stability_series = np.full((num_windows, F), np.nan, dtype=np.float64)

    for f in range(F):
        for w in range(num_windows):
            first_half = ic_series[w:w+half_window, f]
            second_half = ic_series[w+half_window:w+window_size, f]

            # Correlate the same time periods across halves: drop a period
            # only when EITHER half is NaN.  Dropping per half independently
            # misaligns the series (and crashes corrcoef on length mismatch).
            valid = ~np.isnan(first_half) & ~np.isnan(second_half)
            first_valid = first_half[valid]
            second_valid = second_half[valid]

            if len(first_valid) < min_periods:
                continue

            # Pearson correlation between halves
            if np.std(first_valid) == 0 or np.std(second_valid) == 0:
                continue

            corr = np.corrcoef(first_valid, second_valid)[0, 1]
            stability_series[w, f] = corr

    valid_windows = np.sum(~np.isnan(stability_series), axis=0)

    return stability_series, valid_windows


def _compute_rolling_ic_stats_reference(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> dict:
    """Verbatim legacy oracle for equivalence testing."""
    arr = np.asarray(ic_series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    shape = arr.shape
    T = shape[0]; F = shape[1]

    rolling_ic_mean = np.full((T, F), np.nan, dtype=np.float64)
    rolling_ic_ir = np.full((T, F), np.nan, dtype=np.float64)
    positive_ic_ratio = np.full(F, np.nan, dtype=np.float64)
    sign_survival = np.full(F, np.nan, dtype=np.float64)
    worst_rolling_ic = np.full(F, np.nan, dtype=np.float64)
    recent_vs_full = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = arr[:, f]
        finite = np.isfinite(ic_f)
        n_finite = int(np.sum(finite))

        if n_finite < min_periods:
            continue

        roll_mean = _rolling_mean_nan_reference(ic_f, window, min_periods)
        rolling_ic_mean[:, f] = roll_mean

        for end in range(T):
            start = max(0, end - window + 1)
            seg = ic_f[start:end + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= max(2, min_periods):
                std = np.std(seg, ddof=1)
                if np.isfinite(std) and std > 1e-12:
                    rolling_ic_ir[end, f] = np.mean(seg) / std

        valid_mean = roll_mean[np.isfinite(roll_mean)]
        if valid_mean.size > 0:
            worst_rolling_ic[f] = float(np.min(valid_mean))

        vals = ic_f[finite]
        positive_ic_ratio[f] = float(np.mean(vals > 0))

        signs = np.sign(vals)
        runs = []
        run_len = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i - 1]:
                run_len += 1
            else:
                runs.append(run_len)
                run_len = 1
        runs.append(run_len)
        if len(runs) > 0:
            sign_survival[f] = float(np.mean(runs))

        if n_finite >= min_periods:
            recent = vals[-window:]
            if len(recent) >= min_periods:
                recent_vs_full[f] = float(np.mean(recent) - np.mean(vals))

    return {
        "rolling_ic_mean": rolling_ic_mean,
        "rolling_ic_ir": rolling_ic_ir,
        "positive_ic_ratio": positive_ic_ratio,
        "sign_survival": sign_survival,
        "worst_rolling_ic": worst_rolling_ic,
        "recent_vs_full": recent_vs_full,
    }


def _compute_change_point_score_reference(
    ic_series: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """Verbatim legacy oracle for equivalence testing."""
    arr = np.asarray(ic_series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    shape = arr.shape
    T = shape[0]; F = shape[1]

    if window < 2:
        window = 2

    scores = np.full(F, np.nan, dtype=np.float64)
    min_periods = max(2, window // 2)

    for f in range(F):
        ic_f = arr[:, f]
        vals = ic_f[np.isfinite(ic_f)]
        if vals.size < min_periods or vals.size < 2:
            continue

        full_std = np.std(vals, ddof=1)
        if not np.isfinite(full_std) or full_std < 1e-12:
            continue

        max_shift = 0.0
        found = False
        for t in range(window, T - window + 1):
            left = ic_f[t - window:t]
            right = ic_f[t:t + window]
            left_v = left[np.isfinite(left)]
            right_v = right[np.isfinite(right)]
            if len(left_v) < min_periods or len(right_v) < min_periods:
                continue
            shift = abs(float(np.mean(left_v)) - float(np.mean(right_v)))
            if not found or shift > max_shift:
                max_shift = shift
                found = True

        if found:
            scores[f] = float(max_shift / full_std)

    return scores


# ---------------------------------------------------------------------------
# Missing-kernel round (2026-09-24): registry kernels for the ic_summary,
# ic_stability, quantile_stability and ic_decay catalog ids.
#
# All kernels are pure numpy/scipy functions: explicit NaN semantics, integer
# ``min_periods`` gates, dtype rejection for bool/complex/object inputs, and
# deterministic accumulation order (documented per function) so repeated
# calls are bit-identical.
# ---------------------------------------------------------------------------


def _coerce_real_float_array(values, name: str) -> np.ndarray:
    """Reject bool / complex / object inputs and coerce to float64.

    Boolean panels are a semantic error (a bool is not an IC, quantile return
    or turnover), complex values have no ordering, and object arrays may hide
    either.  String dtypes fail in the float64 cast.
    """
    arr = np.asarray(values)
    if arr.dtype == object:
        raise TypeError(f"{name} must be a real numeric array, got object dtype")
    if arr.dtype == bool:
        raise TypeError(f"{name} must be a real numeric array, got bool dtype")
    if np.iscomplexobj(arr):
        raise TypeError(f"{name} must be a real numeric array, got complex dtype")
    return arr.astype(np.float64, copy=False)


def _check_min_periods(min_periods, minimum: int, name: str = "min_periods") -> int:
    if isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if min_periods < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(min_periods)


def compute_ic_summary_stats(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> dict:
    """Distribution summary of a daily IC series (T, F), per factor (F,).

    Doc metric-ic_summary: over FINITE daily IC values only,

        mean  = bar{IC}_f                          (documented primary scalar)
        std   = s_{IC,f}                           (sample std, ddof=1)
        icir  = mean / std                         (NaN for constant series)
        t     = mean / (std / sqrt(n))             (H0: mean = 0)
        p     = 2 * F_{t, n-1}(-|t|)               (two-sided)

    ``icir``/``t``/``p`` reuse the existing :func:`compute_icir` and
    :func:`compute_ic_tstat` kernels (single numeric authority); ``mean`` and
    ``std`` accumulate with ``np.mean``/``np.std(ddof=1)`` over the finite
    values in time order, so repeated calls are bit-identical.  Fewer than
    ``min_periods`` finite values leave every field NaN for that factor; a
    constant series has std 0.0 with NaN icir/t/p.

    Returns:
        dict with keys ``"mean"``, ``"std"``, ``"icir"``, ``"t_stat"``,
        ``"p_value"`` mapping to (F,) float64 arrays.
    """
    arr = np.asarray(ic_series)
    _reject_boolean_ic_series(arr)
    arr = _coerce_real_float_array(arr, "ic_series")
    if arr.ndim != 2:
        raise ValueError("ic_series must have shape (T, F)")
    min_periods = _check_min_periods(min_periods, 2)

    F = arr.shape[1]
    mean = np.full(F, np.nan, dtype=np.float64)
    std = np.full(F, np.nan, dtype=np.float64)
    for f in range(F):
        finite = arr[np.isfinite(arr[:, f]), f]
        if finite.size < min_periods:
            continue
        mean[f] = float(np.mean(finite))
        if finite.size >= 2:
            std[f] = float(np.std(finite, ddof=1))

    icir = compute_icir(arr, min_periods=min_periods)
    t_stat, p_value = compute_ic_tstat(arr, min_periods=min_periods)
    return {
        "mean": mean,
        "std": std,
        "icir": icir,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def compute_ic_stability_scalar(
    ic_series: np.ndarray,
    window_size: int = 60,
    min_periods: int = 20,
    min_windows: int = 1,
) -> np.ndarray:
    """Scalar adaptation of the rolling IC stability windows, (F,).

    Doc metric-ic_stability defines the window statistic
    ``S_{w,f} = Corr(first-half IC, second-half IC)`` and defers the single
    scalar interpretation to the registry adapter.  The interpretation bound
    with the 2026-09-24 missing-kernel round is the mean of the FINITE window
    values, accumulated in window order via ``np.mean`` (bit-identical across
    calls).  Factors with fewer than ``min_windows`` valid windows stay NaN.
    """
    arr = np.asarray(ic_series)
    _reject_boolean_ic_series(arr)
    arr = _coerce_real_float_array(arr, "ic_series")
    if arr.ndim != 2:
        raise ValueError("ic_series must have shape (T, F)")
    min_windows = _check_min_periods(min_windows, 1, "min_windows")

    series, _ = compute_ic_stability(arr, window_size=window_size, min_periods=min_periods)
    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for f in range(series.shape[1]):
        finite = series[np.isfinite(series[:, f]), f]
        if finite.size >= min_windows:
            out[f] = float(np.mean(finite))
    return out


def compute_quantile_rank_stability(
    daily_quantile_returns: np.ndarray,
    min_periods: int = 2,
    min_common: int = 3,
) -> np.ndarray:
    """Cross-time stability of daily quantile-return rankings, (F,).

    Doc metric-quantile_stability ("Stability of quantile return rankings
    across time"); the formula below is the interpretation bound with the
    2026-09-24 missing-kernel round.  For each factor, consider every pair of
    valid days ``t1 < t2`` (lexicographic order).  A day is valid when it has
    at least ``min_common`` finite quantile buckets.  On each pair's joint
    finite support the two day vectors are re-ranked with average ties
    (``scipy.stats.rankdata(method="average")``) and correlated with Pearson
    correlation, i.e. the Spearman correlation on that common support; pairs
    with a zero-variance rank vector yield no value.  The metric is the mean
    of the finite pairwise correlations, accumulated in lexicographic pair
    order (bit-identical across calls).  Fewer than ``min_periods`` valid
    days leave the factor NaN.

    Args:
        daily_quantile_returns: (T, Q) or (T, Q, F) daily quantile-return
            matrix (e.g. from ``compute_quantile_returns_fast``).
        min_periods: minimum number of valid days required per factor.
        min_common: minimum jointly finite buckets per day / per pair.

    Returns:
        (F,) float64 array; NaN where the gate fails.
    """
    arr = _coerce_real_float_array(daily_quantile_returns, "daily_quantile_returns")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("daily_quantile_returns must be (T, Q) or (T, Q, F)")
    min_periods = _check_min_periods(min_periods, 2)
    min_common = _check_min_periods(min_common, 2, "min_common")

    from scipy.stats import rankdata

    T, Q, F = arr.shape
    out = np.full(F, np.nan, dtype=np.float64)
    for f in range(F):
        panel = arr[:, :, f]
        finite = np.isfinite(panel)
        valid_days = np.flatnonzero(np.sum(finite, axis=1) >= min_common)
        if valid_days.size < min_periods:
            continue
        sub = panel[valid_days]
        sub_finite = finite[valid_days]

        m = valid_days.size
        # All pairs, batched: per-pair joint-support re-rank is preserved
        # exactly (rankdata on the pair's masked rows via +inf sentinels),
        # but the O(m^2) Python loop becomes chunked vectorized numpy
        # (2026-09-25: T=1250 loop was >60s; batched ~seconds).
        ii, jj = np.triu_indices(m, k=1)
        corr_arr = np.full(ii.size, np.nan, dtype=np.float64)

        # Gather only the current pair tile. Expanding all O(T^2) pairs
        # before this loop defeated the bounded-batch design on multi-year
        # histories, even though the reductions below were chunked.
        chunk = 65536
        for lo in range(0, ii.size, chunk):
            hi = min(lo + chunk, ii.size)
            a = sub[ii[lo:hi]]
            b = sub[jj[lo:hi]]
            common = sub_finite[ii[lo:hi]] & sub_finite[jj[lo:hi]]
            n = common.sum(axis=1)
            A = np.where(common, a, np.inf)
            B = np.where(common, b, np.inf)
            RA = rankdata(A, axis=1, method="average")
            RB = rankdata(B, axis=1, method="average")
            RA = np.where(common, RA, np.nan)
            RB = np.where(common, RB, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mx = np.nanmean(RA, axis=1)
                my = np.nanmean(RB, axis=1)
                xc = RA - mx[:, None]
                yc = RB - my[:, None]
                sxx = np.nansum(xc * xc, axis=1)
                syy = np.nansum(yc * yc, axis=1)
                sxy = np.nansum(xc * yc, axis=1)
            ok = (n >= min_common) & (sxx > 0) & (syy > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr_arr[lo:hi] = np.where(
                    ok, sxy / np.sqrt(np.where(sxx * syy > 0, sxx * syy, 1.0)), np.nan
                )

        finite_pairs = np.isfinite(corr_arr)
        if finite_pairs.any():
            out[f] = float(np.mean(corr_arr[finite_pairs]))

        # Pair order: np.triu_indices is row-major == the historical
        # (a, b) a<b ordering; np.mean over the compacted finite values
        # reproduces the historical accumulation sequence bit-for-bit.
    return out


def compute_ic_decay_from_mean_ics(
    mean_ics: np.ndarray,
    horizons=None,
) -> np.ndarray:
    """IC decay matrix from pre-computed per-horizon mean ICs, (n_horizons, F).

    Doc metric-ic_decay: ``Decay_{h,f} = mean_{t: IC finite} IC_{t,f}^{(h)}``
    — the per-horizon finite mean IC.  This kernel consumes the already
    reduced (n_horizons, F) mean-IC matrix (one row per forward horizon) and
    returns exactly that matrix (non-finite entries normalised to NaN); it is
    NOT an exponential-decay fit.  Keeping the reduction upstream preserves
    the documented semantics: the facade cannot synthesise multiple
    LabelBundles, so the registry spec declares ``requires=["HorizonMeanIC"]``
    and the id is not auto-dispatchable through ``evaluate()``; explicit
    callers (e.g. the api/horizons ``summarize_horizons`` path) supply the
    per-horizon means.

    Args:
        mean_ics: (n_horizons, F) per-horizon mean IC matrix.
        horizons: optional 1-D sequence of positive, strictly increasing
            forward horizons matching the rows (validated only).

    Returns:
        (n_horizons, F) float64 decay matrix (a copy; input never mutated).
    """
    arr = _coerce_real_float_array(mean_ics, "mean_ics")
    if arr.ndim != 2:
        raise ValueError("mean_ics must have shape (n_horizons, F)")
    if horizons is not None:
        h = np.asarray(horizons)
        if h.ndim != 1 or h.shape[0] != arr.shape[0]:
            raise ValueError(
                "horizons must be a 1-D sequence matching the mean_ics rows"
            )
        if h.size and (not np.all(np.diff(h) > 0) or not np.all(h > 0)):
            raise ValueError("horizons must be positive and strictly increasing")
    return np.where(np.isfinite(arr), arr, np.nan)
