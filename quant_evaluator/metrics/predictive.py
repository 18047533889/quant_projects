"""Predictive metrics derived from a daily IC series (spec §28).

All functions consume an ``ICSeriesArtifact``'s ``values`` array of shape
``(T, F)`` (daily IC per factor) and return a per-factor scalar array of
shape ``(F,)``.  They are *derived* metrics: the IC series is the input, not
the raw factor/label panel.

Calendar grouping: yearly / monthly / quarterly metrics group the IC series
into contiguous blocks of ``block_size`` trading days (252 / 21 / 63) when no
``time_index`` is supplied.  When a ``time_index`` (a sequence of
datetime-like values) is provided, grouping uses the calendar period instead
so real trading calendars are honoured.  The block fallback keeps the metrics
deterministic and testable without a calendar.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "compute_ic_positive_ratio",
    "compute_rank_ic_positive_ratio",
    "compute_yearly_rank_ic",
    "compute_monthly_rank_ic",
    "compute_quarterly_rank_ic",
    "compute_rolling_rank_ic_mean",
    "compute_rolling_rank_ic_ir",
    "compute_recent_3m_rank_ic",
    "compute_recent_6m_rank_ic",
    "compute_recent_12m_rank_ic",
    "compute_worst_year_rank_ic",
    "compute_worst_quarter_rank_ic",
    "compute_rank_ic_decay",
    "compute_ic_sign_consistency",
    "compute_ic_recent_vs_history_delta",
]

_BLOCK_DAYS = {"year": 252, "month": 21, "quarter": 63}


def _as_series(ic_series: np.ndarray) -> np.ndarray:
    """Coerce to a float64 (T, F) array."""
    s = np.asarray(ic_series, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    return s


def _valid_counts(s: np.ndarray) -> np.ndarray:
    return np.sum(np.isfinite(s), axis=0)


def _period_means(
    s: np.ndarray,
    period: str,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Per-period mean IC, shape (n_periods, F).

    Uses the calendar period when ``time_index`` is provided, otherwise
    contiguous blocks of ``_BLOCK_DAYS[period]`` trading days.
    """
    if time_index is not None and len(time_index) == s.shape[0]:
        try:
            import pandas as pd

            idx = pd.to_datetime(list(time_index))
            df = pd.DataFrame(s, index=idx)
            if period == "year":
                grouped = df.groupby(df.index.year)
            elif period == "month":
                grouped = df.groupby([df.index.year, df.index.month])
            else:  # quarter
                grouped = df.groupby(df.index.to_period("Q"))
            return grouped.mean().to_numpy(dtype=np.float64)  # (n_periods, F)
        except Exception:  # noqa: BLE001 - fall back to block grouping
            pass
    block = _BLOCK_DAYS.get(period, 63)
    T, F = s.shape
    n_blocks = max(1, int(np.ceil(T / block)))
    pad = n_blocks * block - T
    if pad > 0:
        s = np.vstack([s, np.full((pad, F), np.nan)])
    blocks = s.reshape(n_blocks, block, F)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=1)  # (n_blocks, F)


def _mean_of_period_means(s: np.ndarray, period: str, time_index=None) -> np.ndarray:
    means = _period_means(s, period, time_index)
    with np.errstate(invalid="ignore"):
        return np.nanmean(means, axis=0)


def _worst_period(s: np.ndarray, period: str, time_index=None) -> np.ndarray:
    means = _period_means(s, period, time_index)
    with np.errstate(invalid="ignore"):
        return np.nanmin(means, axis=0)


def _recent_mean(s: np.ndarray, n_days: int) -> np.ndarray:
    tail = s[-n_days:, :]
    with np.errstate(invalid="ignore"):
        return np.nanmean(tail, axis=0)


def _rolling_mean_ir(s: np.ndarray, window: int, min_periods: int):
    """Rolling mean and IR (mean/std) over a trailing window, (T, F) each."""
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
        ir = mean / std
    mean = np.where(cnt >= min_periods, mean, np.nan)
    ir = np.where((cnt >= max(2, min_periods)) & (std > 1e-12), ir, np.nan)
    return mean, ir


def _autocorr_lag(s: np.ndarray, lag: int) -> np.ndarray:
    """Pairwise-finite autocorrelation at ``lag`` on the original axis, (F,)."""
    T, F = s.shape
    if lag >= T:
        return np.full(F, np.nan)
    finite = np.isfinite(s)
    pair = finite[lag:, :] & finite[:-lag, :]
    n = np.sum(pair, axis=0).astype(np.float64)
    x_prev = np.where(pair, s[:-lag, :], 0.0)
    x_curr = np.where(pair, s[lag:, :], 0.0)
    s_prev = np.sum(x_prev, axis=0)
    s_curr = np.sum(x_curr, axis=0)
    s_pp = np.sum(x_prev * x_prev, axis=0)
    s_cc = np.sum(x_curr * x_curr, axis=0)
    s_pc = np.sum(x_prev * x_curr, axis=0)
    num = n * s_pc - s_prev * s_curr
    denom = np.sqrt((n * s_pp - s_prev * s_prev) * (n * s_cc - s_curr * s_curr))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / denom
    corr = np.where((denom <= 0) | (~np.isfinite(denom)) | (n < 2), np.nan, corr)
    return corr


def compute_ic_positive_ratio(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Fraction of finite daily IC values that are strictly positive, (F,)."""
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    pos = np.sum((s > 0) & valid, axis=0)
    ratio = pos / np.maximum(n, 1)
    return np.where(n >= min_periods, ratio, np.nan)


def compute_rank_ic_positive_ratio(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Fraction of finite daily rank-IC values that are strictly positive, (F,)."""
    return compute_ic_positive_ratio(ic_series, min_periods=min_periods)


def compute_yearly_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Mean of the per-year mean rank IC, (F,)."""
    s = _as_series(ic_series)
    out = _mean_of_period_means(s, "year", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_monthly_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Mean of the per-month mean rank IC, (F,)."""
    s = _as_series(ic_series)
    out = _mean_of_period_means(s, "month", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_quarterly_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Mean of the per-quarter mean rank IC, (F,)."""
    s = _as_series(ic_series)
    out = _mean_of_period_means(s, "quarter", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_rolling_rank_ic_mean(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Time-mean of the rolling-window mean rank IC, (F,)."""
    s = _as_series(ic_series)
    mean, _ = _rolling_mean_ir(s, window, min_periods)
    with np.errstate(invalid="ignore"):
        return np.nanmean(mean, axis=0)


def compute_rolling_rank_ic_ir(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Time-mean of the rolling-window IC information ratio, (F,)."""
    s = _as_series(ic_series)
    _, ir = _rolling_mean_ir(s, window, min_periods)
    with np.errstate(invalid="ignore"):
        return np.nanmean(ir, axis=0)


def compute_recent_3m_rank_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Mean rank IC over the most recent ~3 months (63 trading days), (F,)."""
    s = _as_series(ic_series)
    out = _recent_mean(s, 63)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_recent_6m_rank_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Mean rank IC over the most recent ~6 months (126 trading days), (F,)."""
    s = _as_series(ic_series)
    out = _recent_mean(s, 126)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_recent_12m_rank_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Mean rank IC over the most recent ~12 months (252 trading days), (F,)."""
    s = _as_series(ic_series)
    out = _recent_mean(s, 252)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_worst_year_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Minimum per-year mean rank IC (worst year), (F,)."""
    s = _as_series(ic_series)
    out = _worst_period(s, "year", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_worst_quarter_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Minimum per-quarter mean rank IC (worst quarter), (F,)."""
    s = _as_series(ic_series)
    out = _worst_period(s, "quarter", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_rank_ic_decay(
    ic_series: np.ndarray,
    horizons=(1, 5, 10, 20),
    min_periods: int = 20,
) -> np.ndarray:
    """Mean IC autocorrelation across the decay horizons {1,5,10,20}, (F,).

    A single scalar per factor summarising how quickly the IC series loses
    autocorrelation (decays) at increasing lags.
    """
    s = _as_series(ic_series)
    acfs = [_autocorr_lag(s, h) for h in horizons]
    with np.errstate(invalid="ignore"):
        out = np.nanmean(np.stack(acfs, axis=0), axis=0)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)


def compute_ic_sign_consistency(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Fraction of finite daily IC values sharing the sign of the mean IC, (F,)."""
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(s, axis=0)
    sign = np.sign(mean)
    same = np.sum((np.sign(s) == sign[None, :]) & valid, axis=0)
    ratio = same / np.maximum(n, 1)
    return np.where(n >= min_periods, ratio, np.nan)


def compute_ic_recent_vs_history_delta(
    ic_series: np.ndarray,
    recent_days: int = 63,
    min_periods: int = 20,
) -> np.ndarray:
    """(recent mean IC - full-history mean IC) / full-history std, (F,).

    Positive means recent IC is stronger than the historical average; a
    strongly negative value flags recent degradation.
    """
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        hist_mean = np.nanmean(s, axis=0)
        hist_std = np.nanstd(s, axis=0, ddof=1)
        recent_mean = _recent_mean(s, recent_days)
    delta = (recent_mean - hist_mean) / np.maximum(hist_std, 1e-12)
    delta = np.where(hist_std > 1e-12, delta, np.nan)
    return np.where(n >= min_periods, delta, np.nan)
