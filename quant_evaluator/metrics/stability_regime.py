"""Stability / regime metrics derived from a daily IC series (spec §30).

All functions consume an ``ICSeriesArtifact``'s ``values`` array of shape
``(T, F)`` (daily IC per factor) and return a per-factor scalar array of
shape ``(F,)``.

Regime metrics split the IC series into two halves (early vs late) and
compare their mean IC / dispersion / sign behaviour to quantify how stable
the factor's predictive power is across time regimes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "compute_year_consistency",
    "compute_quarter_consistency",
    "compute_month_consistency",
    "compute_rolling_ic_volatility",
    "compute_rolling_ic_drawdown",
    "compute_ic_sign_flip_rate",
    "compute_change_point_score",
    "compute_cusum_break_score",
    "compute_recent_degradation_score",
    "compute_regime_conditional_ic",
    "compute_regime_worst_ic",
    "compute_regime_dispersion",
    "compute_regime_sign_consistency",
]

_BLOCK_DAYS = {"year": 252, "month": 21, "quarter": 63}


def _as_series(ic_series: np.ndarray) -> np.ndarray:
    s = np.asarray(ic_series, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    return s


def _valid_counts(s: np.ndarray) -> np.ndarray:
    return np.sum(np.isfinite(s), axis=0)


def _period_means(s: np.ndarray, period: str, time_index=None) -> np.ndarray:
    if time_index is not None and len(time_index) == s.shape[0]:
        try:
            import pandas as pd

            idx = pd.to_datetime(list(time_index))
            df = pd.DataFrame(s, index=idx)
            if period == "year":
                grouped = df.groupby(df.index.year)
            elif period == "month":
                grouped = df.groupby([df.index.year, df.index.month])
            else:
                grouped = df.groupby(df.index.to_period("Q"))
            return grouped.mean().to_numpy(dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass
    block = _BLOCK_DAYS.get(period, 63)
    T, F = s.shape
    n_blocks = max(1, int(np.ceil(T / block)))
    pad = n_blocks * block - T
    if pad > 0:
        s = np.vstack([s, np.full((pad, F), np.nan)])
    blocks = s.reshape(n_blocks, block, F)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=1)


def _period_consistency(s: np.ndarray, period: str, time_index=None) -> np.ndarray:
    """Fraction of period means sharing the sign of the overall mean IC, (F,)."""
    means = _period_means(s, period, time_index)
    with np.errstate(invalid="ignore"):
        overall = np.nanmean(s, axis=0)
    sign = np.sign(overall)
    finite = np.isfinite(means)
    n = np.sum(finite, axis=0)
    same = np.sum((np.sign(means) == sign[None, :]) & finite, axis=0)
    ratio = same / np.maximum(n, 1)
    return np.where(n >= 1, ratio, np.nan)


def compute_year_consistency(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Fraction of years whose mean IC matches the overall IC sign, (F,)."""
    s = _as_series(ic_series)
    out = _period_consistency(s, "year", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_quarter_consistency(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Fraction of quarters whose mean IC matches the overall IC sign, (F,)."""
    s = _as_series(ic_series)
    out = _period_consistency(s, "quarter", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_month_consistency(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Fraction of months whose mean IC matches the overall IC sign, (F,)."""
    s = _as_series(ic_series)
    out = _period_consistency(s, "month", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_rolling_ic_volatility(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Time-mean of the rolling-window IC standard deviation, (F,).

    Lower values indicate a more stable IC series.
    """
    s = _as_series(ic_series)
    T, F = s.shape
    finite = np.isfinite(s)
    x = np.where(finite, s, 0.0)
    pref = np.concatenate([np.zeros((1, F)), np.cumsum(x, axis=0)], axis=0)
    fpref = np.concatenate([np.zeros((1, F)), np.cumsum(finite, axis=0)], axis=0)
    pref_sq = np.concatenate([np.zeros((1, F)), np.cumsum(x * x, axis=0)], axis=0)
    ends = np.arange(1, T + 1)
    start = np.maximum(ends - window, 0)
    cnt = fpref[ends] - fpref[start]
    ssum = pref[ends] - pref[start]
    ssq = pref_sq[ends] - pref_sq[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / np.maximum(cnt, 1.0)
        var = ssq - cnt * mean * mean
        std = np.sqrt(np.maximum(var, 0.0) / np.maximum(cnt - 1, 1.0))
    std = np.where(cnt >= min_periods, std, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(std, axis=0)


def compute_rolling_ic_drawdown(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Mean of the rolling-window IC drawdown (peak-to-trough), (F,).

    Computed on the cumulative sum of the rolling mean IC within each window.
    """
    s = _as_series(ic_series)
    T, F = s.shape
    finite = np.isfinite(s)
    x = np.where(finite, s, 0.0)
    pref = np.concatenate([np.zeros((1, F)), np.cumsum(x, axis=0)], axis=0)
    fpref = np.concatenate([np.zeros((1, F)), np.cumsum(finite, axis=0)], axis=0)
    ends = np.arange(1, T + 1)
    start = np.maximum(ends - window, 0)
    cnt = fpref[ends] - fpref[start]
    ssum = pref[ends] - pref[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / np.maximum(cnt, 1.0)
    mean = np.where(cnt >= min_periods, mean, np.nan)
    # cumulative sum of the rolling mean (NaN -> 0), then peak-to-trough
    cum = np.cumsum(np.where(np.isfinite(mean), mean, 0.0), axis=0)
    running_max = np.maximum.accumulate(cum, axis=0)
    dd = cum - running_max
    with np.errstate(invalid="ignore"):
        return np.nanmean(dd, axis=0)


def compute_ic_sign_flip_rate(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Fraction of adjacent finite IC pairs whose sign flips, (F,).

    Lower values indicate a more persistent (stable) IC sign.
    """
    s = _as_series(ic_series)
    finite = np.isfinite(s)
    pair = finite[1:, :] & finite[:-1, :]
    n = np.sum(pair, axis=0)
    flip = np.sum((np.sign(s[1:, :]) != np.sign(s[:-1, :])) & pair, axis=0)
    rate = flip / np.maximum(n, 1)
    return np.where(n >= min_periods - 1, rate, np.nan)


def compute_change_point_score(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """CUSUM-based change-point score: max |cumulative deviation|, (F,).

    A large score indicates a structural break in the IC level.
    """
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(s, axis=0)
    dev = np.where(valid, s - mean[None, :], 0.0)
    cum = np.cumsum(dev, axis=0)
    score = np.max(np.abs(cum), axis=0)
    return np.where(n >= min_periods, score, np.nan)


def compute_cusum_break_score(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """CUSUM break score: max cumulative deviation normalised by std, (F,).

    ``max|cumsum(ic - mean)| / (std * sqrt(T))`` — a standardised change-point
    statistic.  Larger values flag a stronger break.
    """
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(s, axis=0)
        std = np.nanstd(s, axis=0, ddof=1)
    dev = np.where(valid, s - mean[None, :], 0.0)
    cum = np.cumsum(dev, axis=0)
    score = np.max(np.abs(cum), axis=0) / np.maximum(std * np.sqrt(n), 1e-12)
    score = np.where(std > 1e-12, score, np.nan)
    return np.where(n >= min_periods, score, np.nan)


def compute_recent_degradation_score(
    ic_series: np.ndarray,
    recent_days: int = 63,
    min_periods: int = 20,
) -> np.ndarray:
    """Recent degradation: (recent mean IC - full mean IC) / full std, (F,).

    Negative values indicate the factor's recent IC is weaker than its
    historical average (degradation).  This is the negative of the
    ``ic_recent_vs_history_delta`` predictive metric.
    """
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        hist_mean = np.nanmean(s, axis=0)
        hist_std = np.nanstd(s, axis=0, ddof=1)
        recent_mean = np.nanmean(s[-recent_days:, :], axis=0)
    score = (hist_mean - recent_mean) / np.maximum(hist_std, 1e-12)
    score = np.where(hist_std > 1e-12, score, np.nan)
    return np.where(n >= min_periods, score, np.nan)


def _regime_split(s: np.ndarray):
    """Split the IC series into early/late halves, returning (early, late)."""
    T, F = s.shape
    mid = T // 2
    return s[:mid, :], s[mid:, :]


def compute_regime_conditional_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Mean IC in the late (recent) regime, (F,)."""
    s = _as_series(ic_series)
    _, late = _regime_split(s)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(late, axis=0)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_regime_worst_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Minimum of the early/late regime mean IC, (F,)."""
    s = _as_series(ic_series)
    early, late = _regime_split(s)
    with np.errstate(invalid="ignore"):
        e = np.nanmean(early, axis=0)
        l = np.nanmean(late, axis=0)
    out = np.minimum(e, l)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_regime_dispersion(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Absolute difference between early and late regime mean IC, (F,).

    Larger values indicate the factor's predictive power changed across
    regimes (instability).
    """
    s = _as_series(ic_series)
    early, late = _regime_split(s)
    with np.errstate(invalid="ignore"):
        e = np.nanmean(early, axis=0)
        l = np.nanmean(late, axis=0)
    out = np.abs(e - l)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_regime_sign_consistency(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """1.0 if early and late regime mean IC share the same sign, else 0.0, (F,)."""
    s = _as_series(ic_series)
    early, late = _regime_split(s)
    with np.errstate(invalid="ignore"):
        e = np.nanmean(early, axis=0)
        l = np.nanmean(late, axis=0)
    out = np.where(np.sign(e) == np.sign(l), 1.0, 0.0)
    out = np.where(np.isfinite(e) & np.isfinite(l), out, np.nan)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
