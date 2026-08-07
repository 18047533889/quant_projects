# -*- coding: utf-8 -*-
"""Kalman-filter state-space operators (P2, experimental).

Deterministic initialisation (first finite observation), explicit noise
parameters, and a causal one-pass filter so segmented / full-history execution
agree.  Outputs are the filtered level / trend / beta state, standardised
innovations and state uncertainty.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.ts_model._rolling_core import aligned, frame_like, metadata

_CANONICALS: list[str] = []


def _register(name: str, description: str, params: list[str], unit: str, fn):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.state_space",
        backend="pandas_numpy",
        status="experimental",
    )
    class _StateOp(SeriesOperator):
        metadata = metadata(name, description, params, unit=unit, cost=8)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {name}
    )
    return _StateOp


def _kalman_level(vals: np.ndarray, q: float, r: float, out_stat: str) -> np.ndarray:
    n = len(vals)
    mu = np.full(n, np.nan, dtype=float)
    p = np.full(n, np.nan, dtype=float)
    innov_z = np.full(n, np.nan, dtype=float)
    mu_prev = np.nan
    p_prev = 1.0
    for t in range(n):
        x = vals[t]
        if not np.isfinite(x):
            if np.isfinite(mu_prev):
                mu[t] = mu_prev
                p[t] = p_prev + q
            continue
        if not np.isfinite(mu_prev):
            mu_prev = x
            p_prev = r
            mu[t] = x
            p[t] = r
            continue
        p_pred = p_prev + q
        k = p_pred / (p_pred + r)
        # Innovation is the difference between the observation and the
        # *predicted* state (mu_prev before the Kalman update), not the filtered
        # state.  Computing it after the update would shrink every innovation by
        # (1 - k) and corrupt the standardised innovation.
        innov = x - mu_prev
        mu_prev = mu_prev + k * innov
        p_prev = (1.0 - k) * p_pred
        mu[t] = mu_prev
        p[t] = p_prev
        innov_z[t] = innov / np.sqrt(max(p_pred + r, 1e-12))
    if out_stat == "level":
        return mu
    if out_stat == "innovation_z":
        return innov_z
    return p


_register("ts_kalman_level", "局部水平模型的过滤水平估计。", ["x", "q", "r"], "level",
           lambda x, q=1e-4, r=1.0: _apply_col(x, lambda v: _kalman_level(v, float(q), float(r), "level")))
_register("ts_kalman_innovation_z", "观测值相对 Kalman 预测的标准化创新。", ["x", "q", "r"], "level",
           lambda x, q=1e-4, r=1.0: _apply_col(x, lambda v: _kalman_level(v, float(q), float(r), "innovation_z")))
_register("ts_kalman_beta_uncertainty", "Beta 状态协方差/标准误。", ["y", "x", "q", "r"], "level",
           lambda y, x, q=1e-3, r=1.0: _apply_two(y, x, lambda a, b: _kalman_beta(a, b, float(q), float(r), "uncertainty")))


def _kalman_trend_slope(vals: np.ndarray, q_level: float, q_trend: float, r: float) -> np.ndarray:
    """Local linear trend: level and slope states."""
    n = len(vals)
    slope = np.full(n, np.nan, dtype=float)
    level = np.nan
    trend = 0.0
    p11 = p12 = p22 = 1.0
    for t in range(n):
        x = vals[t]
        if not np.isfinite(x):
            continue
        if not np.isfinite(level):
            level = x
            trend = 0.0
            slope[t] = 0.0
            continue
        # predict
        l_pred = level + trend
        t_pred = trend
        p11_p = p11 + q_level + 2 * p12 + p22
        p12_p = p12 + p22
        p22_p = p22 + q_trend
        # update (scalar observation with H = [1, 0])
        k1 = p11_p / (p11_p + r)
        k2 = p12_p / (p11_p + r)
        innov = x - l_pred
        level = l_pred + k1 * innov
        trend = t_pred + k2 * innov
        p11 = (1 - k1) * p11_p
        p12 = (1 - k1) * p12_p
        p22 = p22_p - k2 * p12_p
        slope[t] = trend
    return slope


_register("ts_kalman_trend", "局部线性趋势模型的潜在斜率。", ["x", "q_level", "q_trend", "r"], "level",
           lambda x, q_level=1e-5, q_trend=1e-5, r=1.0: _apply_col(x, lambda v: _kalman_trend_slope(v, float(q_level), float(q_trend), float(r))))


def _kalman_beta(y: np.ndarray, x: np.ndarray, q: float, r: float, out_stat: str) -> np.ndarray:
    n = len(y)
    beta = np.full(n, np.nan, dtype=float)
    change = np.full(n, np.nan, dtype=float)
    unc = np.full(n, np.nan, dtype=float)
    b = np.nan
    p = 1.0
    b_prev = np.nan
    for t in range(n):
        yv, xv = y[t], x[t]
        if not (np.isfinite(yv) and np.isfinite(xv)):
            continue
        if not np.isfinite(b):
            b = yv / xv if abs(xv) > 1e-12 else 0.0
            b_prev = b
            p = 1.0
            beta[t] = b
            unc[t] = p
            continue
        p_pred = p + q
        denom = p_pred * xv * xv + r
        if denom <= 0.0:
            continue
        k = p_pred * xv / denom
        innov = yv - b * xv
        b_new = b + k * innov
        p = (1.0 - k * xv) * p_pred
        beta[t] = b_new
        change[t] = b_new - b_prev
        unc[t] = p
        b_prev = b_new
        b = b_new
    if out_stat == "beta":
        return beta
    if out_stat == "beta_change":
        return change
    return unc


_register("ts_kalman_beta", "动态市场 Beta 状态。", ["y", "x", "q", "r"], "level",
           lambda y, x, q=1e-3, r=1.0: _apply_two(y, x, lambda a, b: _kalman_beta(a, b, float(q), float(r), "beta")))
_register("ts_kalman_beta_change", "动态 Beta 变化。", ["y", "x", "q", "r"], "level",
           lambda y, x, q=1e-3, r=1.0: _apply_two(y, x, lambda a, b: _kalman_beta(a, b, float(q), float(r), "beta_change")))


def _apply_col(x: pd.DataFrame, fn) -> pd.DataFrame:
    xv = x.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = fn(xv[:, col])
    return frame_like(x, out)


def _apply_two(y: pd.DataFrame, x: pd.DataFrame, fn) -> pd.DataFrame:
    # P0-041: model inputs must be index/column aligned before the per-column
    # recursion, so a reordered ``x`` panel can never silently mispair stocks.
    y, x = aligned(y, x)
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = fn(yv[:, col], xv[:, col])
    return frame_like(y, out)
