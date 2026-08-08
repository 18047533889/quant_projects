# -*- coding: utf-8 -*-
"""GARCH / GJR-GARCH and HAR-RV volatility operators (P2, experimental).

GARCH parameters are estimated per window by quasi-maximum likelihood
(variance targeting with a small numeric optimisation over alpha/beta).  These
are high-cost operators and must not enter default random factor search.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.ts_model._rolling_core import fit_linear_model_checked, frame_like, metadata

_CANONICALS: list[str] = []

try:
    from scipy.optimize import minimize as _minimize
except Exception:  # pragma: no cover
    _minimize = None


def _register(name: str, description: str, params: list[str], unit: str, fn):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.volatility",
        backend="pandas_numpy",
        status="experimental",
    )
    class _VolOp(SeriesOperator):
        metadata = metadata(name, description, params, unit=unit, cost=8)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {name}
    )
    return _VolOp


def _fit_garch(rets: np.ndarray) -> tuple[float, float, float] | None:
    """Return (omega, alpha, beta) for GARCH(1,1) via variance-targeting MLE.

    The optimisation result is only accepted when the solver reports success
    and the parameters lie in the stationary region (alpha, beta >= 0 and
    alpha + beta < 1).  A failed / non-converged fit returns ``None`` so the
    caller emits NaN rather than the solver's last (possibly invalid) iterate.
    """
    if _minimize is None or len(rets) < 10 or np.std(rets) <= 1e-12:
        return None
    long_var = float(np.var(rets))

    def _nll(params):
        a, b = params
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return 1e12
        w = max(long_var * (1 - a - b), 1e-12)
        # Unify the likelihood's initial variance with the output recursion's
        # seed (_variance_path callers pass var(fit_seg)): both backcast from the
        # unconditional variance long_var = omega/(1-alpha-beta), not the
        # constant term omega alone.
        h = np.full(len(rets), long_var, dtype=float)
        for t in range(1, len(rets)):
            h[t] = w + a * rets[t - 1] ** 2 + b * h[t - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(h) + rets ** 2 / h))
    try:
        res = _minimize(_nll, np.array([0.05, 0.9]), method="Nelder-Mead",
                        options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6})
        if not getattr(res, "success", False):
            return None
        a, b = float(res.x[0]), float(res.x[1])
        if not (np.isfinite(a) and np.isfinite(b)):
            return None
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return None
        w = max(long_var * (1 - a - b), 1e-12)
        return w, a, b
    except Exception:
        return None


def _variance_path(
    seg: np.ndarray,
    w: float,
    a: float,
    b: float,
    h_init: float,
    gamma: float = 0.0,
    *,
    asymmetric: bool = False,
) -> float:
    """Recurse the GARCH/GJR conditional variance over ``seg``.

    ``h_init`` seeds the recursion and the result is the conditional variance
    governing the LAST observation of ``seg`` (information strictly before it).
    The caller chooses ``h_init`` so the current return can never seed its own
    standardisation denominator (audit P0: the init must exclude r_t).
    """
    h = h_init
    for i in range(1, len(seg)):
        prev_r = seg[i - 1]
        if asymmetric:
            lev = gamma if prev_r < 0 else 0.0
            h = w + (a + lev) * prev_r ** 2 + b * h
        else:
            h = w + a * prev_r ** 2 + b * h
    return h


def _garch_path(rets: np.ndarray, window: int, stat: str, asymmetric: bool, r: float) -> float:
    """Rolling GARCH(1,1) / GJR statistic.

    Timing convention: the returned conditional variance ``h_last`` governs the
    *last* observed return (computed from information strictly before it), and
    ``h_next = w + a*r_t^2 + b*h_last`` is the one-step-ahead forecast for the
    next period.  The standardised shock divides ``r_t`` by ``sqrt(h_last)`` —
    the variance that actually governed it.
    """
    if len(rets) < max(window, 12):
        return np.nan
    seg = rets[-int(window):]
    # P0-040 / P0 (this audit): the standardised shock / surprise of the current
    # return must not be standardised by parameters fitted on that same return,
    # and must not leak the current return into its own denominator through the
    # *initial* variance either.  Fit on the window excluding the current
    # observation (<= t-1) AND seed the variance recursion from the variance of
    # that same fit segment.  Forecast / persistence stats may use the current
    # return (they legitimately forecast the *next* period).
    fit_seg = seg[:-1] if stat == "shock" else seg
    if len(fit_seg) < 12:
        return np.nan
    if asymmetric:
        # GJR: h_t = w + (a + gamma*I(r<0))*r^2 + b*h
        params = _fit_gjr(fit_seg)
    else:
        params = _fit_garch(fit_seg)
    if params is None:
        return np.nan
    gamma = 0.0
    if asymmetric:
        w, a, gamma, b = params
    else:
        w, a, b = params
    if stat == "persistence":
        return float(a + b) if not asymmetric else float(a + 0.5 * gamma + b)
    # Seed the variance recursion from the fit segment only (excludes the
    # current return for the shock stat — no self-leak through the init).
    h_last = _variance_path(
        seg, w, a, b, float(np.var(fit_seg)), gamma, asymmetric=asymmetric
    )
    if stat == "forecast":
        # h_last governs the current return; the next-period forecast conditions
        # on it.
        h_next = w + a * seg[-1] ** 2 + b * h_last
        return float(np.sqrt(max(h_next, 1e-12)))
    # standardized shock of the last return uses the variance that governed it
    if not np.isfinite(seg[-1]):
        return np.nan
    return float(seg[-1] / np.sqrt(max(h_last, 1e-12)))


def _fit_gjr(rets: np.ndarray) -> tuple[float, float, float, float] | None:
    if _minimize is None or len(rets) < 12 or np.std(rets) <= 1e-12:
        return None
    long_var = float(np.var(rets))

    def _nll(p):
        a, g, b = p
        if min(a, b, g) < 1e-6 or a + 0.5 * g + b >= 0.999:
            return 1e12
        w = max(long_var * (1 - a - 0.5 * g - b), 1e-12)
        # Unify with the output recursion seed: backcast from the unconditional
        # variance long_var (= omega/(1-a-0.5g-b)), matching _variance_path.
        h = np.full(len(rets), long_var, dtype=float)
        for t in range(1, len(rets)):
            lev = g if rets[t - 1] < 0 else 0.0
            h[t] = w + (a + lev) * rets[t - 1] ** 2 + b * h[t - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(h) + rets ** 2 / h))
    try:
        res = _minimize(_nll, np.array([0.03, 0.05, 0.9]), method="Nelder-Mead",
                        options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-6})
        if not getattr(res, "success", False):
            return None
        a, g, b = float(res.x[0]), float(res.x[1]), float(res.x[2])
        if not (np.isfinite(a) and np.isfinite(g) and np.isfinite(b)):
            return None
        if min(a, b, g) < 1e-6 or a + 0.5 * g + b >= 0.999:
            return None
        w = max(long_var * (1 - a - 0.5 * g - b), 1e-12)
        return w, a, g, b
    except Exception:
        return None


def _apply(x: pd.DataFrame, fn) -> pd.DataFrame:
    xv = x.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = fn(xv[: row + 1, col])
    return frame_like(x, out)


_register("ts_garch_next_vol_forecast", "GARCH(1,1) 下一期条件波动率（观测最后收益之后）。", ["x", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "forecast", False, 0.0)))
_register("ts_garch_vol_surprise", "GARCH 波动率意外：最近收益平方 / 条件方差 - 1。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _garch_vol_surprise(v, int(window))))
_register("ts_garch_persistence", "GARCH(1,1) 波动持续 alpha+beta。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "persistence", False, 0.0)))
_register("ts_garch_standardized_shock", "GARCH 标准化冲击 return/条件波动率（参数于 t-1 及以前拟合）。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "shock", False, 0.0)))
# Deprecated alias for the next-period forecast: a registry compatibility
# alias, NOT a separate canonical — mining must not double-search the same
# kernel under two independent research candidates.
OperatorRegistry.register_compat_alias(
    "ts_garch_vol_forecast",
    "ts_garch_next_vol_forecast",
    migration_reason="legacy name for the identical next-period GARCH forecast",
    deprecated_since="2026-08",
    removal_version="1.0",
)
_register("ts_gjr_garch_vol_forecast", "GJR-GARCH 波动预测（杠杆效应）。", ["x", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "forecast", True, 0.0)))
_register("ts_gjr_leverage", "GJR 负收益冲击系数。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _gjr_leverage(v, int(window))))


def _garch_vol_surprise(vals: np.ndarray, window: int) -> float:
    """Realised variance of the last return relative to its conditional variance.

    ``rv_t / h_t - 1`` where ``h_t`` is the GARCH variance that governed the
    last observed return (information strictly before it).
    """
    if len(vals) < max(window, 12):
        return np.nan
    seg = vals[-int(window):]
    # P0-040 / P0 (this audit): params fit on <= t-1 (exclude the current
    # return) AND the variance recursion is seeded from that same fit segment,
    # so rv_t/h_t is a genuine out-of-sample surprise and the current return
    # cannot leak into its own denominator through the initial variance.
    if len(seg) < 13:
        return np.nan
    params = _fit_garch(seg[:-1])
    if params is None:
        return np.nan
    w, a, b = params
    h_last = _variance_path(seg, w, a, b, float(np.var(seg[:-1])))
    if not np.isfinite(seg[-1]) or h_last <= 1e-12:
        return np.nan
    return float(seg[-1] ** 2 / h_last - 1.0)


def _gjr_leverage(vals: np.ndarray, window: int) -> float:
    if len(vals) < max(window, 12):
        return np.nan
    params = _fit_gjr(vals[-int(window):])
    return np.nan if params is None else float(params[2])


def _har_rv(rv: np.ndarray, window: int, stat: str) -> float:
    """HAR model whose input is a *realized variance* series (not squared here).

    Features are RV_t, weekly mean and monthly mean of RV; the target is
    RV_{t+1}, so the model genuinely forecasts next-period RV from today's
    components.  This is the ``ts_har_rv_*`` / ``ts_har_from_return_*`` family
    contract.  ``ts_har_from_return_*`` squares its return input before calling
    this kernel.
    """
    seg = rv[-int(window):]
    n = len(seg)
    if n < 30:
        return np.nan
    daily = seg
    weekly = pd.Series(seg).rolling(5).mean().to_numpy()
    monthly = pd.Series(seg).rolling(22).mean().to_numpy()
    X = np.column_stack([np.ones(n), daily, weekly, monthly])
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(seg)
    if stat == "forecast":
        # Next-period forecast RV_{t+1}: target = next RV, features today.
        target = np.concatenate([seg[1:], [np.nan]])
        valid_t = np.isfinite(target) & valid
        if valid_t.sum() < 25:
            return np.nan
        Xs = X[valid_t]
        y = target[valid_t]
        # ModelDesignGate: reject rank-deficient / ill-conditioned HAR design.
        beta = fit_linear_model_checked(Xs, y)
        if beta is None:
            return np.nan
        pred = float(np.dot(X[-1], beta))
        return float(np.sqrt(max(pred, 0.0)))
    # Current-period surprise RV_t - forecast(RV_t | t-1).  P0-039: the
    # training window is capped strictly before the last observation (feature
    # rows 0..n-3, targets rv[1..n-2]), and the forecast is made from the
    # t-1 feature row — never from X_t, and never mixing a t+1 forecast with a
    # current residual.
    fit_rows = np.arange(n - 2)  # 0 .. n-3
    fit_valid = valid[fit_rows]
    Xs = X[fit_rows][fit_valid]
    y = seg[fit_rows + 1][fit_valid]
    if Xs.shape[0] < 25 or Xs.shape[0] <= Xs.shape[1]:
        return np.nan
    # ModelDesignGate: an ill-conditioned HAR design fails closed (NaN).
    beta = fit_linear_model_checked(Xs, y)
    if beta is None:
        return np.nan
    pred_t = float(np.dot(X[n - 2], beta))  # forecast RV_{n-1} from t-1 info
    sd = float(np.std(y - Xs @ beta))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.nan
    return float((seg[-1] - pred_t) / sd)


def _har_from_return(rets: np.ndarray, window: int, stat: str) -> float:
    """HAR over daily returns: squares them into an RV series internally."""
    return _har_rv(rets ** 2, window, stat)


# The historic operators accept a realized-variance panel (parameter named
# ``rv``); they no longer square it a second time.
_register("ts_har_rv_next_forecast", "HAR-RV 下一期已实现方差预测（平方根，输入已实现方差）。", ["rv", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "forecast")))
_register("ts_har_rv_forecast_error_z", "RV 相对 HAR 预测的标准化偏差（输入已实现方差）。", ["rv", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "innovation_z")))
# The from-return variants square the daily return panel internally.
_register("ts_har_from_return_next_vol", "HAR 基于日收益的下一期波动预测（内部平方为 RV）。", ["ret", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _har_from_return(v, int(window), "forecast")))
_register("ts_har_from_return_forecast_error_z", "日收益平方 RV 相对 HAR 预测的标准化偏差。", ["ret", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _har_from_return(v, int(window), "innovation_z")))
# Deprecated aliases (kept registered); use the *_next_forecast names.
_register("ts_har_rv_forecast", "HAR-RV 已实现方差预测（平方根，deprecated 别名）。", ["rv", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "forecast")))
_register("ts_har_rv_innovation_z", "RV 相对 HAR 预测的标准化偏差（deprecated 别名）。", ["rv", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "innovation_z")))
