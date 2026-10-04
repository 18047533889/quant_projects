"""GPU predictive metrics derived from a daily IC series (spec §28).

All inputs are daily IC series of shape (T, F) — small relative to raw
factor tensors — so the point is cross-factor batch vectorization over F,
matching the CPU reference in :mod:`quant_evaluator.metrics.predictive`.

CuPy is imported lazily so the module loads on a CPU-only environment.
"""

from __future__ import annotations

import numpy as np


def _import_cp():
    import cupy as cp
    return cp


def _as_series(ic_series):
    cp = _import_cp()
    ic = cp.asarray(ic_series, dtype=cp.float64)
    if ic.ndim == 1:
        ic = ic[:, None]
    return ic


def _valid_counts(ic):
    cp = _import_cp()
    return cp.sum(cp.isfinite(ic), axis=0)


def _period_means(ic, block):
    """Per-block mean IC, (n_blocks, F), using contiguous blocks of ``block`` days."""
    cp = _import_cp()
    T, F = ic.shape
    n_blocks = max(1, int(np.ceil(T / block)))
    pad = n_blocks * block - T
    if pad > 0:
        ic = cp.vstack([ic, cp.full((pad, F), cp.nan)])
    blocks = ic.reshape(n_blocks, block, F)
    finite = cp.isfinite(blocks).astype(cp.float64)
    s = cp.where(cp.isfinite(blocks), blocks, 0.0)
    cnt = cp.sum(finite, axis=1)
    means = cp.sum(s, axis=1) / cp.maximum(cnt, 1.0)
    return means  # (n_blocks, F)


def _mean_of_period_means(ic, block):
    cp = _import_cp()
    means = _period_means(ic, block)
    return cp.nanmean(means, axis=0)


def _worst_period(ic, block):
    cp = _import_cp()
    means = _period_means(ic, block)
    return cp.nanmin(means, axis=0)


def _recent_mean(ic, n_days):
    cp = _import_cp()
    tail = ic[-n_days:, :]
    return cp.nanmean(tail, axis=0)


def _rolling_mean_ir(ic, window, min_periods):
    """Rolling mean and IR using stable centered variance in bounded chunks."""
    cp = _import_cp()
    from quant_evaluator.kernels.gpu.rolling_ic_statistics import rolling_ic_mean_ir
    T, F = ic.shape
    return rolling_ic_mean_ir(ic, window, min_periods)


def _autocorr_lag(ic, lag):
    """Pairwise-finite autocorrelation at ``lag`` on the original axis, (F,)."""
    cp = _import_cp()
    T, F = ic.shape
    if lag >= T:
        return cp.full(F, cp.nan)
    if lag <= 0:
        raise ValueError("lag must be a positive integer")
    finite = cp.isfinite(ic)
    pair = finite[lag:, :] & finite[:-lag, :]
    n = cp.sum(pair, axis=0).astype(cp.float64)
    first_pair = cp.argmax(pair, axis=0)
    left_anchor = cp.take_along_axis(ic[:-lag, :], first_pair[None, :], axis=0)[0]
    right_anchor = cp.take_along_axis(ic[lag:, :], first_pair[None, :], axis=0)[0]
    left_delta = cp.where(pair, ic[:-lag, :] - left_anchor, 0.0)
    right_delta = cp.where(pair, ic[lag:, :] - right_anchor, 0.0)
    left_mean = cp.sum(left_delta, axis=0) / cp.maximum(n, 1.0)
    right_mean = cp.sum(right_delta, axis=0) / cp.maximum(n, 1.0)
    left_centered = cp.where(pair, left_delta - left_mean, 0.0)
    right_centered = cp.where(pair, right_delta - right_mean, 0.0)
    covariance = cp.sum(left_centered * right_centered, axis=0)
    left_ss = cp.sum(left_centered * left_centered, axis=0)
    right_ss = cp.sum(right_centered * right_centered, axis=0)
    denom = cp.sqrt(left_ss * right_ss)
    corr = covariance / denom
    corr = cp.where((denom <= 0) | (~cp.isfinite(denom)) | (n < 2), cp.nan, corr)
    return corr


def ic_positive_ratio(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    valid = cp.isfinite(ic)
    n = cp.sum(valid, axis=0)
    pos = cp.sum((ic > 0) & valid, axis=0)
    ratio = pos / cp.maximum(n, 1)
    return cp.where(n >= min_periods, ratio, cp.nan).get()


def rank_ic_positive_ratio(ic_series, min_periods=20):
    return ic_positive_ratio(ic_series, min_periods=min_periods)


def yearly_rank_ic(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    out = _mean_of_period_means(ic, 252)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def monthly_rank_ic(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    out = _mean_of_period_means(ic, 21)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def quarterly_rank_ic(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    out = _mean_of_period_means(ic, 63)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def rolling_rank_ic_mean(ic_series, window=60, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    mean, _ = _rolling_mean_ir(ic, window, min_periods)
    return cp.nanmean(mean, axis=0).get()


def rolling_rank_ic_ir(ic_series, window=60, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    _, ir = _rolling_mean_ir(ic, window, min_periods)
    return cp.nanmean(ir, axis=0).get()


def recent_3m_rank_ic(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    out = _recent_mean(ic, 63)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def recent_6m_rank_ic(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    out = _recent_mean(ic, 126)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def recent_12m_rank_ic(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    out = _recent_mean(ic, 252)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def worst_year_rank_ic(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    out = _worst_period(ic, 252)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def worst_quarter_rank_ic(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    out = _worst_period(ic, 63)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def rank_ic_decay(ic_series, horizons=(1, 5, 10, 20), min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    acfs = [_autocorr_lag(ic, h) for h in horizons]
    out = cp.nanmean(cp.stack(acfs, axis=0), axis=0)
    return cp.where(_valid_counts(ic) >= min_periods, out, cp.nan).get()


def ic_sign_consistency(ic_series, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    valid = cp.isfinite(ic)
    n = cp.sum(valid, axis=0)
    mean = cp.nanmean(ic, axis=0)
    sign = cp.sign(mean)
    same = cp.sum((cp.sign(ic) == sign[None, :]) & valid, axis=0)
    ratio = same / cp.maximum(n, 1)
    return cp.where(n >= min_periods, ratio, cp.nan).get()


def ic_recent_vs_history_delta(ic_series, recent_days=63, min_periods=20):
    cp = _import_cp()
    ic = _as_series(ic_series)
    valid = cp.isfinite(ic)
    n = cp.sum(valid, axis=0)
    hist_mean = cp.nanmean(ic, axis=0)
    hist_std = cp.nanstd(ic, axis=0, ddof=1)
    recent_mean = _recent_mean(ic, recent_days)
    delta = (recent_mean - hist_mean) / cp.maximum(hist_std, 1e-12)
    delta = cp.where(hist_std > 1e-12, delta, cp.nan)
    return cp.where(n >= min_periods, delta, cp.nan).get()
