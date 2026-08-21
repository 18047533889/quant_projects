"""
IC summary statistics: ICIR, t-statistics, and decay analysis.

Reference implementation for IC-based performance metrics with proper
statistical inference and temporal decay patterns.
"""

from typing import Tuple
import numpy as np
from scipy import stats

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
    valid_periods = np.sum(~np.isnan(ic_series), axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_ic = np.nanmean(ic_series, axis=0)
        std_ic = np.nanstd(ic_series, axis=0, ddof=1)
        icir = mean_ic / std_ic

    insufficient = (
        (valid_periods < min_periods)
        | (std_ic < 1e-10)
        | np.isnan(std_ic)
        | ~np.isfinite(icir)
    )
    icir = np.where(insufficient, np.nan, icir)

    return icir


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


def _rolling_mean_nan(series: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """NaN-aware rolling mean without pandas. Returns NaN where insufficient."""
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

        # Rolling std (ddof=1) and rolling IR.
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

        # Positive IC ratio.
        vals = ic_f[finite]
        positive_ic_ratio[f] = float(np.mean(vals > 0))

        # Sign survival: mean run-length of consistent-sign finite IC.
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

        max_shift = 0.0
        found = False
        # Split points where both adjacent windows exist in range.
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
