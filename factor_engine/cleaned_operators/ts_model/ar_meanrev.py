# -*- coding: utf-8 -*-
"""AR forecasting / innovation and mean-reversion operators (P0).

Daily panels in, daily panels out; causal rolling kernels per (instrument,
date).  ``ts_ar_coefficient`` and ``ts_variance_ratio`` already exist in
``regression_models``; this module adds the forecast / innovation forms and the
multi-horizon variance-ratio slope.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.ts_model._rolling_core import (
    frame_like,
    metadata,
    ols_fit,
    trailing_contiguous_finite,
)

_CANONICALS: list[str] = []


def _ar_fit(seg: np.ndarray, order: int) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit AR(order) on a 1-D segment; return (beta, lagged design rows).

    ``beta`` has length order+1 (intercept first).  ``design`` rows correspond
    to the same time indexes as the segment, with NaN for rows without enough
    lag history (used for the current-row forecast).
    """
    n = len(seg)
    design = np.full((n, order + 1), np.nan, dtype=float)
    for t in range(order, n):
        xv = seg[t - order : t][::-1]  # x_{t-1}, ..., x_{t-order}
        if np.all(np.isfinite(xv)):
            design[t] = np.concatenate([[1.0], xv])
    valid = np.all(np.isfinite(design), axis=1) & np.isfinite(seg)
    if valid.sum() < order + 2:
        return None, design
    X = design[valid]
    y = seg[valid]
    beta = ols_fit(X, y)
    return beta, design


def _ar_apply(vals: np.ndarray, window: int, order: int, stat: str, *, fit_lag: int = 0, stability_k: int = 0) -> np.ndarray:
    """Causal AR(order) rolling kernel.

    ``fit_lag=0`` (legacy) fits the AR model on the window *including* the
    current row and reports the in-sample fitted value.  ``fit_lag>=1`` fits on
    rows strictly before the current one and reports the genuine one-step-ahead
    forecast / innovation at the current row.  ``stability_k>0`` (with
    ``stat='coeff'``) reports the standard deviation of the AR slope over the
    last ``stability_k`` consecutive fits.
    """
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    o = max(1, int(order))
    w = max(o + 2, int(window))
    lag = max(0, int(fit_lag))
    k = max(0, int(stability_k))
    for row in range(n):
        fit_end = row - lag
        if fit_end < 0:
            continue
        start = max(0, fit_end - w + 1)
        seg = vals[start : fit_end + 1]
        beta, design = _ar_fit(seg, o)
        if beta is None:
            continue
        if row < o:
            continue
        cur_xv = vals[row - o : row][::-1]  # x_{t-1}, ..., x_{t-order}
        if not np.all(np.isfinite(cur_xv)):
            continue
        cur = np.concatenate([[1.0], cur_xv])
        with np.errstate(over="ignore", invalid="ignore"):
            pred = float(np.dot(cur, beta))
        innov = float(vals[row] - pred) if np.isfinite(vals[row]) else np.nan
        if stat == "forecast":
            out[row] = pred
        elif stat == "innovation":
            out[row] = innov
        elif stat == "innovation_z":
            resid = np.full(len(seg), np.nan, dtype=float)
            valid = np.all(np.isfinite(design), axis=1) & np.isfinite(seg)
            resid[valid] = seg[valid] - design[valid] @ beta
            sd = float(np.nanstd(resid))
            if sd is not None and np.isfinite(sd) and sd > 0.0:
                out[row] = innov / sd
        elif stat == "coeff":
            out[row] = float(beta[1])  # first lag coefficient (index 0 is the intercept)
        elif stat == "coeff_stability":
            coeffs: list[float] = []
            for j in range(k):
                fe = row - lag - j
                if fe < 0:
                    break
                s2 = max(0, fe - w + 1)
                b2, _ = _ar_fit(vals[s2 : fe + 1], o)
                if b2 is None:
                    break
                coeffs.append(float(b2[1]))
            if len(coeffs) >= 2:
                out[row] = float(np.std(coeffs))
    return out


def _ar_op(name: str, description: str, unit: str, stat: str, *, fit_lag: int = 0, stability_k: int = 0, cost: int = 4, diagnostic_only: bool = False):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.ar_meanrev",
        backend="pandas_numpy",
        status="experimental",
    )
    class _ArOp(SeriesOperator):
        metadata = metadata(name, description, ["x", "window", "order"], unit=unit, cost=cost, diagnostic_only=diagnostic_only)

        def _calculate_series(self, x, window=60, order=1, **_):
            xv = x.to_numpy(dtype=float)
            rows, cols = xv.shape
            out = np.full((rows, cols), np.nan, dtype=float)
            for col in range(cols):
                out[:, col] = _ar_apply(xv[:, col], int(window), int(order), stat,
                                        fit_lag=int(fit_lag), stability_k=int(stability_k))
            return frame_like(x, out)

    return _ArOp


# In-sample (fitted-value) AR forms: fit on the window *including* the current
# row and report the fitted value / residual.  These are the honest names; the
# legacy ``ts_ar_forecast`` / ``ts_ar_innovation`` are aliases of the same
# kernels (fit_lag=0 is NOT a one-step-ahead forecast).
_ar_op("ts_ar_fitted_value", "AR(order) 窗口内拟合值(fit_lag=0, in-sample)。", "level", "forecast", diagnostic_only=True)
_ar_op("ts_ar_in_sample_resid", "当前实际值减窗口内 AR 拟合值(in-sample resid)。", "level", "innovation", diagnostic_only=True)
_ar_op("ts_ar_forecast", "AR(order) 当前值预测（legacy 名称——实为窗口内拟合值，见 ts_ar_fitted_value）。", "level", "forecast", diagnostic_only=True)
_ar_op("ts_ar_innovation", "当前实际值减 AR 预测（legacy 名称——实为样本内残差，见 ts_ar_in_sample_resid）。", "level", "innovation", diagnostic_only=True)
_ar_op("ts_ar_innovation_z", "AR 创新标准化。", "level", "innovation_z", diagnostic_only=True)

# Prior-window (out-of-sample) AR forms: fit on t-window..t-1, forecast t.
_ar_op("ts_ar_prior_forecast", "AR(order) 截至 t-1 训练的一步预测。", "level", "forecast", fit_lag=1)
_ar_op("ts_ar_prior_innovation", "当前实际值减截至 t-1 训练的 AR 预测。", "level", "innovation", fit_lag=1)
_ar_op("ts_ar_prior_innovation_z", "AR 样本外创新 / 历史残差标准差。", "level", "innovation_z", fit_lag=1)
_ar_op("ts_ar_prior_coeff", "AR(order) 截至 t-1 训练的一阶滞后系数。", "level", "coeff", fit_lag=1)
_ar_op("ts_ar_coeff_stability", "AR 一阶滞后系数在最近 K 个窗口的标准差。", "level", "coeff_stability", fit_lag=1, stability_k=5, cost=6)


def _mean_reversion_half_life(vals: np.ndarray, window: int, min_periods: int) -> float:
    n = len(vals)
    start = max(0, n - window)
    seg = vals[start:]
    xprev = seg[:-1]
    d = np.diff(seg)
    valid = np.isfinite(xprev) & np.isfinite(d)
    x, y = xprev[valid], d[valid]
    if len(x) < max(min_periods, 4) or np.std(x) <= 0.0:
        return np.nan
    design = np.column_stack([np.ones(len(x)), x])
    beta = ols_fit(design, y)
    if beta is None:
        return np.nan
    b = float(beta[1])
    # Exact discrete AR(1) half-life (audit P1-E): x_t = a + phi*x_{t-1} with
    # phi = 1 + b.  Only 0 < phi < 1 is mean-reverting; half_life =
    # ln(0.5)/ln(phi).  The old OU approximation -ln(2)/b is exposed separately
    # as ts_mean_reversion_ou_approx_half_life.
    phi = 1.0 + b
    if not np.isfinite(phi) or not (0.0 < phi < 1.0):
        return np.nan
    return float(np.log(0.5) / np.log(phi))


@register_operator(
    name="ts_mean_reversion_half_life",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_mean_reversion_half_life",
    source="ts_model.ar_meanrev",
    backend="pandas_numpy",
    status="experimental",
)
class TsMeanReversionHalfLife(SeriesOperator):
    """均值回复半衰期 ln(0.5)/ln(1+beta)（离散 AR(1) 精确解）。"""

    metadata = metadata(
        "ts_mean_reversion_half_life", "均值回复半衰期（AR(1) 精确离散）。", ["x", "window", "min_periods"], unit="count", cost=3,
    )

    def _calculate_series(self, x, window=120, min_periods=20, **_):
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                out[row, col] = _mean_reversion_half_life(xv[: row + 1, col], int(window), int(min_periods))
        return frame_like(x, out)


def _mean_reversion_ou_half_life(vals: np.ndarray, window: int, min_periods: int) -> float:
    n = len(vals)
    start = max(0, n - window)
    seg = vals[start:]
    xprev = seg[:-1]
    d = np.diff(seg)
    valid = np.isfinite(xprev) & np.isfinite(d)
    x, y = xprev[valid], d[valid]
    if len(x) < max(min_periods, 4) or np.std(x) <= 0.0:
        return np.nan
    design = np.column_stack([np.ones(len(x)), x])
    beta = ols_fit(design, y)
    if beta is None:
        return np.nan
    b = float(beta[1])
    if not np.isfinite(b) or b >= 0.0:
        return np.nan
    # OU continuous-time approximation -ln(2)/b (audit P1-E: kept under an
    # explicit _ou_approx name; the production canonical is the exact discrete
    # AR(1) half-life above).
    return float(-np.log(2.0) / b)


@register_operator(
    name="ts_mean_reversion_ou_approx_half_life",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_mean_reversion_ou_approx_half_life",
    source="ts_model.ar_meanrev",
    backend="pandas_numpy",
    status="experimental",
)
class TsMeanReversionOuApproxHalfLife(SeriesOperator):
    """均值回复半衰期 OU 近似 -log(2)/beta（beta<0 时定义）。"""

    metadata = metadata(
        "ts_mean_reversion_ou_approx_half_life",
        "均值回复半衰期（OU 连续近似 -ln2/beta）。",
        ["x", "window", "min_periods"],
        unit="count",
        cost=3,
    )

    def _calculate_series(self, x, window=120, min_periods=20, **_):
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                out[row, col] = _mean_reversion_ou_half_life(
                    xv[: row + 1, col], int(window), int(min_periods)
                )
        return frame_like(x, out)


def _variance_ratio_slope(vals: np.ndarray, window: int, max_q: int, min_periods: int) -> float:
    maxq = max(2, int(max_q))
    # P1-90: when ``max_q`` exceeds what the window can ever support, every q in
    # the range 2..max_q cannot be evaluated and the operator would silently
    # fall back to a shorter q-range — manufacturing identical outputs for
    # different ``max_q`` parameters.  Fail loudly (independent of data
    # availability) instead of silently skipping.
    if maxq + 2 > int(window):
        raise ValueError(
            f"ts_variance_ratio_slope: max_q={max_q} needs at least "
            f"max_q+2={maxq + 2} rows in the window, but window={window}"
        )
    n = len(vals)
    start = max(0, n - window)
    seg = vals[start:]
    # P0-008: never bridge suspensions/gaps into adjacent observations — the
    # variance ratio runs over the most recent contiguous finite block only.
    finite = trailing_contiguous_finite(seg)
    if finite.size < max(min_periods, 6):
        return np.nan
    rets = np.diff(finite)
    if rets.size < max(min_periods, 3):
        return np.nan
    var1 = float(np.var(rets))
    if var1 <= 0.0:
        return np.nan
    logq = []
    vr = []
    for q in range(2, maxq + 1):
        if finite.size < q + 2:
            continue
        qrets = finite[q:] - finite[:-q]
        varq = float(np.var(qrets))
        vr.append(varq / (q * var1) - 1.0)
        logq.append(np.log(float(q)))
    if len(logq) < 2:
        return np.nan
    # P1-90: use a centered dot product (ddof=0 in both numerator and
    # denominator) instead of ``np.cov(logq, vr)[0, 1] / np.var(logq)``, whose
    # ddof=1 covariance / ddof=0 variance ratio biased the slope by n/(n-1).
    lx = np.asarray(logq, dtype=float) - float(np.mean(logq))
    ly = np.asarray(vr, dtype=float) - float(np.mean(vr))
    denom = float(np.dot(lx, lx))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(lx, ly) / denom)


@register_operator(
    name="ts_variance_ratio_slope",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_variance_ratio_slope",
    source="ts_model.ar_meanrev",
    backend="pandas_numpy",
    status="experimental",
)
class TsVarianceRatioSlope(SeriesOperator):
    """多持有期方差比相对 log(q) 的斜率（趋势/随机游走/均值回复判别）。"""

    metadata = metadata(
        "ts_variance_ratio_slope", "方差比斜率。", ["x", "window", "max_q", "min_periods"], unit="level", cost=4,
    )

    def _calculate_series(self, x, window=120, max_q=10, min_periods=20, **_):
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                out[row, col] = _variance_ratio_slope(xv[: row + 1, col], int(window), int(max_q), int(min_periods))
        return frame_like(x, out)


_CANONICALS.extend(
    [
        "ts_ar_fitted_value",
        "ts_ar_in_sample_resid",
        "ts_ar_forecast",
        "ts_ar_innovation",
        "ts_ar_innovation_z",
        "ts_mean_reversion_half_life",
        "ts_mean_reversion_ou_approx_half_life",
        "ts_variance_ratio_slope",
        "ts_ar_prior_forecast",
        "ts_ar_prior_innovation",
        "ts_ar_prior_innovation_z",
        "ts_ar_prior_coeff",
        "ts_ar_coeff_stability",
    ]
)

import cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.extend_extended_only(set(_CANONICALS))
