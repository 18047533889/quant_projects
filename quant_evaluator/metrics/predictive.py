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

import numbers
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
    """Rolling mean and IR over stable, centered windows in bounded chunks."""
    T, F = s.shape
    if (isinstance(window, (bool, np.bool_))
            or not isinstance(window, numbers.Integral)
            or window < 1):
        raise ValueError("window must be a positive integer")
    if (isinstance(min_periods, (bool, np.bool_))
            or not isinstance(min_periods, numbers.Integral)
            or min_periods < 1):
        raise ValueError("min_periods must be a positive integer")
    window = int(window)
    min_periods = int(min_periods)
    mean = np.full((T, F), np.nan, dtype=np.float64)
    ir = np.full((T, F), np.nan, dtype=np.float64)
    workspace_elements = 65_536
    factor_chunk = max(1, min(F, workspace_elements // window))

    def assign_windows(start, factor_start, factor_stop, blocks):
        finite = np.isfinite(blocks)
        count = np.sum(finite, axis=1)
        safe = np.where(finite, blocks, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            window_mean = np.sum(safe, axis=1) / np.maximum(count, 1)
            centered = np.where(finite, blocks - window_mean[:, None, :], 0.0)
            squared_deviation = np.sum(centered * centered, axis=1)
            std = np.sqrt(squared_deviation / np.maximum(count - 1, 1))
            window_ir = window_mean / std
        output_slice = slice(start, start + len(blocks))
        factor_slice = slice(factor_start, factor_stop)
        mean[output_slice, factor_slice] = np.where(
            count >= min_periods, window_mean, np.nan)
        ir[output_slice, factor_slice] = np.where(
            (count >= max(2, min_periods)) & (std > 1e-12),
            window_ir, np.nan)

    for factor_start in range(0, F, factor_chunk):
        factor_stop = min(F, factor_start + factor_chunk)
        factor_width = factor_stop - factor_start
        early_stop = min(T, window - 1)
        for end in range(1, early_stop + 1):
            blocks = s[None, max(0, end - window):end, factor_start:factor_stop]
            assign_windows(end - 1, factor_start, factor_stop, blocks)

        if T >= window:
            rows_per_chunk = max(
                1, workspace_elements // (window * factor_width))
            windows = np.lib.stride_tricks.sliding_window_view(
                s[:, factor_start:factor_stop], window, axis=0)
            for row_start in range(0, len(windows), rows_per_chunk):
                row_stop = min(len(windows), row_start + rows_per_chunk)
                blocks = windows[row_start:row_stop].transpose(0, 2, 1)
                output_start = row_start + window - 1
                assign_windows(output_start, factor_start, factor_stop, blocks)
    return mean, ir


def _autocorr_lag(s: np.ndarray, lag: int) -> np.ndarray:
    """Pairwise-finite autocorrelation at ``lag`` on the original axis, (F,)."""
    T, F = s.shape
    if lag >= T:
        return np.full(F, np.nan)
    finite = np.isfinite(s)
    pair = finite[lag:, :] & finite[:-lag, :]
    n = np.sum(pair, axis=0)
    x_prev = np.where(pair, s[:-lag, :], 0.0)
    x_curr = np.where(pair, s[lag:, :], 0.0)
    anchor_index = np.argmax(pair, axis=0)
    anchor_prev = x_prev[anchor_index, np.arange(F)]
    anchor_curr = x_curr[anchor_index, np.arange(F)]
    delta_prev = np.where(pair, x_prev - anchor_prev[None, :], 0.0)
    delta_curr = np.where(pair, x_curr - anchor_curr[None, :], 0.0)
    safe_n = np.maximum(n, 1)
    mean_prev = np.sum(delta_prev, axis=0) / safe_n
    mean_curr = np.sum(delta_curr, axis=0) / safe_n
    centered_prev = np.where(pair, delta_prev - mean_prev[None, :], 0.0)
    centered_curr = np.where(pair, delta_curr - mean_curr[None, :], 0.0)
    covariance = np.sum(centered_prev * centered_curr, axis=0)
    variance_prev = np.sum(centered_prev * centered_prev, axis=0)
    variance_curr = np.sum(centered_curr * centered_curr, axis=0)
    denom = np.sqrt(variance_prev * variance_curr)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = covariance / denom
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
