# -*- coding: utf-8 -*-
"""Polars backends for next-stage time-series model operators (genuine).

Rolling regression / AR kernels use Polars ``rolling_map`` with NumPy vector
kernels shared with the pandas reference, so they remain genuine Polars
expressions (no pandas delegation).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common._polars_bridge import align_cols


def _cols(df, *others):
    cols = [c for c in df.columns if c != "date"]
    for o in others:
        cols = [c for c in cols if c in o.columns]
    return cols


def _meta(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name, category="time_series_regression", description=description, param_names=params,
        return_type="series", tags=["time_series_regression", "polars", "native", "typed_v2"],
    )


def _register(name: str, description: str, params: list[str], fn):
    @register_operator(
        name=name, category="time_series_regression", business_category="time_series_regression",
        canonical=name, source="ts_model.polars_regression",
    )
    class _TsPolars(SeriesOperator):
        metadata = _meta(name, description, params)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    return _TsPolars


def _ols_beta(vals):
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 3:
        return float("nan")
    t = np.arange(len(finite), dtype=float)
    if np.var(t) <= 1e-12 or np.var(finite) <= 1e-12:
        return float("nan")
    return float(np.cov(t, finite)[0, 1] / np.var(t))


def _rolling_slope(x, window, min_periods):
    w = max(3, int(window))
    mp = max(3, int(min_periods))
    return x.with_columns(
        [x[c].rolling_map(_ols_beta, window_size=w, min_samples=mp).alias(c) for c in _cols(x)]
    )


def _variance_ratio_slope(vals, max_q):
    arr = np.asarray(vals, dtype=float)
    # 与 pandas 参考一致：只用窗口内**尾部连续有限块**（P0-008，禁止把缺口
    # 两侧的观测桥接成相邻），而非全部有限值。
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return float("nan")
    # R35-P0-M11: current-slot NaN parity.  The pandas reference
    # (``trailing_contiguous_finite``) yields an EMPTY block when the last row is
    # non-finite -> output NaN.  This Polars callback used to back off to the
    # previous finite block (stale backoff), manufacturing a value in a slot
    # where the reference says NaN — a dangerous current-slot semantics
    # divergence.  Match the reference exactly: last row non-finite => NaN.
    last = int(len(arr) - 1)
    if not finite_mask[last]:
        return float("nan")
    j = last
    while j >= 0 and finite_mask[j]:
        j -= 1
    finite = arr[j + 1 : last + 1]
    if len(finite) < max(12, int(max_q) + 3):
        return float("nan")
    rets = np.diff(finite)
    var1 = float(np.var(rets))
    if var1 <= 1e-12:
        return float("nan")
    logq, vr = [], []
    for q in range(2, int(max_q) + 1):
        if len(finite) < q + 2:
            continue
        qrets = finite[q:] - finite[:-q]
        varq = float(np.var(qrets))
        vr.append(varq / (q * var1) - 1.0)
        logq.append(np.log(float(q)))
    if len(logq) < 2 or np.var(logq) <= 1e-12:
        return float("nan")
    # P1-90: centered dot product (ddof=0 numerator and denominator) — matches
    # the pandas reference and removes the n/(n-1) bias of cov(ddof=1)/var(ddof=0).
    lx = np.asarray(logq, dtype=float) - float(np.mean(logq))
    ly = np.asarray(vr, dtype=float) - float(np.mean(vr))
    denom = float(np.dot(lx, lx))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(lx, ly) / denom)


def _vr_slope(x, window, max_q, min_periods):
    w = max(20, int(window))
    # P1-90: a ``max_q`` that the rolling window can never support would make the
    # kernel return NaN (or a silently truncated q-range) for every cell — fail
    # loudly instead of manufacturing identical outputs for different max_q.
    if int(max_q) + 3 > w:
        raise ValueError(
            f"ts_variance_ratio_slope: max_q={max_q} needs at least "
            f"max_q+3={int(max_q) + 3} rows in the window, but window={window}"
        )
    out = []
    for c in _cols(x):
        out.append(x[c].rolling_map(lambda s: _variance_ratio_slope(s, int(max_q)), window_size=w, min_samples=max(int(min_periods), 8)).alias(c))
    return x.with_columns(out)


def _mean_reversion_half_life(vals, min_periods):
    arr = np.asarray(vals, dtype=float)
    x = arr[:-1]
    y = np.diff(arr)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < max(int(min_periods), 4) or np.var(x[valid]) <= 1e-12:
        return float("nan")
    # OLS slope: sample cov / sample var (ddof-consistent), matching lstsq
    denom = float(np.cov(x[valid], x[valid])[0, 1])
    if denom <= 1e-12:
        return float("nan")
    beta = float(np.cov(y[valid], x[valid])[0, 1] / denom)
    # Exact discrete AR(1) half-life (audit P1-E): phi = 1 + beta, valid only
    # for 0 < phi < 1; half_life = ln(0.5)/ln(phi).  Matches the pandas
    # reference in ts_model.ar_meanrev.
    phi = 1.0 + beta
    if not np.isfinite(phi) or not (0.0 < phi < 1.0):
        return float("nan")
    return float(np.log(0.5) / np.log(phi))


def _half_life(x, window, min_periods):
    w = max(20, int(window))
    out = []
    for c in _cols(x):
        out.append(x[c].rolling_map(lambda s: _mean_reversion_half_life(s, int(min_periods)), window_size=w, min_samples=max(int(min_periods), 4)).alias(c))
    return x.with_columns(out)


def _pairwise_rolling(y, x, window, min_periods):
    """Rolling slope via the covariance identity (matches the pandas
    reference ddof convention: population cov / population var).

    Both series are first masked to the *pairwise-valid* set (rows where both
    y and x are finite), so the four moments are computed over the same rows as
    the pandas reference.  The covariance identity with per-series means would
    otherwise diverge whenever the null patterns of the two inputs differ (e.g.
    a liquidity-delta panel whose first row is NaN).  R35-P0-M12: this kernel
    used ``sample_cov = pop_cov * n/(n-1)`` — a ddof=1 numerator against a
    ddof=0 denominator — which scaled every beta by n/(n-1) relative to the
    ``np.mean`` / ``np.sum((b-xbar)**2)`` pandas reference (which is ddof=0/0).
    Both must be population (ddof=0) so the identity reproduces the reference
    exactly.
    """
    w = max(3, int(window))
    mp = max(3, int(min_periods))
    cols = align_cols(y, x)
    out = []
    for c in cols:
        # R35-P0-M12: FINITE-only pairwise mask, matching the pandas reference
        # (``np.isfinite``).  ``is_not_null()`` alone admits NaN/±Inf rows that
        # the reference excludes, so ``ts_market_liquidity_beta`` /
        # ``ts_industry_liquidity_beta`` diverged on hostile inputs.  ``±Inf``
        # standardizes to NaN in ``rolling_mean`` / ``rolling_sum`` and then the
        # moment identity silently drops the row, breaking the "same rows as the
        # reference" invariant.
        both = y[c].is_finite() & x[c].is_finite()
        ym = pl.when(both).then(y[c]).otherwise(None)
        xm = pl.when(both).then(x[c]).otherwise(None)
        mean_ab = (ym * xm).rolling_mean(window_size=w, min_samples=mp)
        mean_a = ym.rolling_mean(window_size=w, min_samples=mp)
        mean_b = xm.rolling_mean(window_size=w, min_samples=mp)
        mean_b2 = (xm * xm).rolling_mean(window_size=w, min_samples=mp)
        n = ym.is_not_null().cast(pl.Float64).rolling_sum(window_size=w, min_samples=mp)
        pop_cov = mean_ab - mean_a * mean_b
        pop_var = mean_b2 - mean_b * mean_b
        # R35-P0-M12: population cov/var (ddof=0/0), exactly matching the
        # ``np.mean`` / ``np.sum((b-xbar)**2)`` pandas reference.  ``n`` guards
        # the degenerate 1-row window only.
        slope = pl.when(n > 1).then(pop_cov / pop_var).otherwise(None)
        out.append(slope.alias(c))
    return y.with_columns(out)


_register("ts_mean_reversion_half_life", "均值回复半衰期（Polars rolling_map）。", ["x", "window", "min_periods"],
          lambda x, window=120, min_periods=20: _half_life(x, int(window), int(min_periods)))
_register("ts_variance_ratio_slope", "方差比斜率（Polars rolling_map）。", ["x", "window", "max_q", "min_periods"],
          lambda x, window=120, max_q=10, min_periods=20: _vr_slope(x, int(window), int(max_q), int(min_periods)))
def _liquidity_delta(x):
    """Regress on the *change* in liquidity to match the pandas reference."""
    return x.with_columns([x[c].diff(1).alias(c) for c in _cols(x)])


_register("ts_market_liquidity_beta", "收益对市场流动性变化 Beta（Polars rolling_cov/var）。", ["own_return", "market_liquidity", "window"],
          lambda y, m, window=60: _pairwise_rolling(y, _liquidity_delta(m), int(window), max(3, int(window) // 5)))
_register("ts_industry_liquidity_beta", "收益对行业流动性变化 Beta（Polars rolling_cov/var）。", ["own_return", "industry_liquidity", "window"],
          lambda y, m, window=60: _pairwise_rolling(y, _liquidity_delta(m), int(window), max(3, int(window) // 5)))
