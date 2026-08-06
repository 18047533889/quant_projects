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
from cleaned_operators.ts_model._rolling_core import frame_like, metadata

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
    """Return (omega, alpha, beta) for GARCH(1,1) via variance-targeting MLE."""
    if _minimize is None or len(rets) < 10 or np.std(rets) <= 1e-12:
        return None
    long_var = float(np.var(rets))

    def _nll(params):
        a, b = params
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return 1e12
        w = max(long_var * (1 - a - b), 1e-12)
        h = np.full(len(rets), w, dtype=float)
        for t in range(1, len(rets)):
            h[t] = w + a * rets[t - 1] ** 2 + b * h[t - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(h) + rets ** 2 / h))
    try:
        res = _minimize(_nll, np.array([0.05, 0.9]), method="Nelder-Mead",
                        options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6})
        a, b = float(res.x[0]), float(res.x[1])
        w = max(long_var * (1 - a - b), 1e-12)
        return w, a, b
    except Exception:
        return None


def _garch_path(rets: np.ndarray, window: int, stat: str, asymmetric: bool, r: float) -> float:
    if len(rets) < max(window, 12):
        return np.nan
    seg = rets[-int(window):]
    if asymmetric:
        # GJR: h_t = w + (a + gamma*I(r<0))*r^2 + b*h
        params = _fit_gjr(seg)
    else:
        params = _fit_garch(seg)
    if params is None:
        return np.nan
    gamma = 0.0
    if asymmetric:
        w, a, gamma, b = params
    else:
        w, a, b = params
    h = float(np.var(seg))
    for i in range(1, len(seg)):
        prev_r = seg[i - 1]
        if asymmetric:
            lev = gamma if prev_r < 0 else 0.0
            h = w + (a + lev) * prev_r ** 2 + b * h
        else:
            h = w + a * prev_r ** 2 + b * h
    if stat == "forecast":
        return float(np.sqrt(max(h, 1e-12)))
    if stat == "persistence":
        return float(a + b) if not asymmetric else float(a + 0.5 * gamma + b)
    # standardized shock of the last return
    if not np.isfinite(seg[-1]):
        return np.nan
    return float(seg[-1] / np.sqrt(max(h, 1e-12)))


def _fit_gjr(rets: np.ndarray) -> tuple[float, float, float, float] | None:
    if _minimize is None or len(rets) < 12 or np.std(rets) <= 1e-12:
        return None
    long_var = float(np.var(rets))

    def _nll(p):
        a, g, b = p
        if min(a, b, g) < 1e-6 or a + 0.5 * g + b >= 0.999:
            return 1e12
        w = max(long_var * (1 - a - 0.5 * g - b), 1e-12)
        h = np.full(len(rets), w, dtype=float)
        for t in range(1, len(rets)):
            lev = g if rets[t - 1] < 0 else 0.0
            h[t] = w + (a + lev) * rets[t - 1] ** 2 + b * h[t - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(h) + rets ** 2 / h))
    try:
        res = _minimize(_nll, np.array([0.03, 0.05, 0.9]), method="Nelder-Mead",
                        options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-6})
        a, g, b = float(res.x[0]), float(res.x[1]), float(res.x[2])
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


_register("ts_garch_vol_forecast", "GARCH(1,1) 下一期条件波动率。", ["x", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "forecast", False, 0.0)))
_register("ts_garch_persistence", "GARCH(1,1) 波动持续 alpha+beta。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "persistence", False, 0.0)))
_register("ts_garch_standardized_shock", "GARCH 标准化冲击 return/条件波动率。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "shock", False, 0.0)))
_register("ts_gjr_garch_vol_forecast", "GJR-GARCH 波动预测（杠杆效应）。", ["x", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "forecast", True, 0.0)))
_register("ts_gjr_leverage", "GJR 负收益冲击系数。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _gjr_leverage(v, int(window))))


def _gjr_leverage(vals: np.ndarray, window: int) -> float:
    if len(vals) < max(window, 12):
        return np.nan
    params = _fit_gjr(vals[-int(window):])
    return np.nan if params is None else float(params[2])


def _har_rv(rets: np.ndarray, window: int, stat: str) -> float:
    seg = rets[-int(window):]
    if len(seg) < 30:
        return np.nan
    rv = seg ** 2
    daily = rv
    weekly = pd.Series(rv).rolling(5).mean().to_numpy()
    monthly = pd.Series(rv).rolling(22).mean().to_numpy()
    X = np.column_stack([np.ones(len(seg)), daily, weekly, monthly])
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(seg)
    # HAR predicts future RV using today's components
    target = np.concatenate([seg[1:] ** 2, [np.nan]])
    valid_t = np.isfinite(target) & valid
    if valid_t.sum() < 25:
        return np.nan
    Xs = X[valid_t]
    y = target[valid_t]
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    pred = float(np.dot(X[-1], beta))
    if stat == "forecast":
        return float(np.sqrt(max(pred, 0.0)))
    # innovation z on current RV
    sd = float(np.std(y - Xs @ beta))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.nan
    return float((seg[-1] ** 2 - pred) / sd)


_register("ts_har_rv_forecast", "HAR-RV 已实现方差预测（平方根）。", ["rv", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "forecast")))
_register("ts_har_rv_innovation_z", "RV 相对 HAR 预测的标准化偏差。", ["rv", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "innovation_z")))
