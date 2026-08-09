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

    _surface.extend_research_only({name})
    return _StateOp


def _kalman_level(vals: np.ndarray, q: float, r: float, out_stat: str) -> np.ndarray:
    # P0-15: the noise parameters must be well-typed — a negative process-noise
    # would shrink uncertainty, a non-positive observation-noise breaks the
    # Kalman update.  Fail fast instead of silently producing nonsense.
    if not (q >= 0.0 and r > 0.0):
        raise ValueError("q must be >= 0 and r must be > 0")
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
                # P0-15: a missing observation is predict-only — the filtered
                # state is unchanged but the covariance really advances by Q.
                # Previously p_prev was left behind and the same stale P was
                # re-emitted for every missing row, so K consecutive gaps did
                # not accumulate P + K*Q as the theory requires.
                mu[t] = mu_prev
                p_prev = p_prev + q
                p[t] = p_prev
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
_register("ts_kalman_beta_uncertainty", "Beta 状态滤波协方差 P(标准误为 sqrt(P), 此处输出 P)。", ["y", "x", "q", "r"], "level",
           lambda y, x, q=1e-3, r=1.0: _apply_two(y, x, lambda a, b: _kalman_beta(a, b, float(q), float(r), "uncertainty")))


def _kalman_trend_slope(vals: np.ndarray, q_level: float, q_trend: float, r: float) -> np.ndarray:
    """Local linear trend: level and slope states."""
    if not (q_level >= 0.0 and q_trend >= 0.0 and r > 0.0):
        raise ValueError("q_level/q_trend must be >= 0 and r must be > 0")
    n = len(vals)
    slope = np.full(n, np.nan, dtype=float)
    level = np.nan
    trend = 0.0
    p11 = p12 = p22 = 1.0
    for t in range(n):
        x = vals[t]
        if not np.isfinite(x):
            if np.isfinite(level):
                # Predict-only on a missing observation (audit P0-N): propagate
                # the state AND the covariance — never silently drop the
                # uncertainty growth.
                level = level + trend
                p11_p = p11 + q_level + 2 * p12 + p22
                p12_p = p12 + p22
                p22_p = p22 + q_trend
                p11, p12, p22 = p11_p, p12_p, p22_p
                slope[t] = trend
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


_BETA_WARMUP = 5


def _kalman_beta(y: np.ndarray, x: np.ndarray, q: float, r: float, out_stat: str) -> np.ndarray:
    if not (q >= 0.0 and r > 0.0):
        raise ValueError("q must be >= 0 and r must be > 0")
    n = len(y)
    beta = np.full(n, np.nan, dtype=float)
    change = np.full(n, np.nan, dtype=float)
    unc = np.full(n, np.nan, dtype=float)
    b = np.nan
    p = 1.0
    b_prev = np.nan
    warmup_y: list[float] = []
    warmup_x: list[float] = []
    for t in range(n):
        yv, xv = y[t], x[t]
        if not (np.isfinite(yv) and np.isfinite(xv)):
            if np.isfinite(b):
                # Predict-only on a missing observation (audit P0-N): the random
                # walk beta keeps its covariance growth (F P F' + Q with F=1).
                p = p + q
                unc[t] = p
            continue
        if not np.isfinite(b):
            # Audit P0-N: never initialise beta from the unstable single ratio
            # y/x (it explodes when x ~ 0).  Use a trailing warmup OLS through
            # the origin; fall back to a diffuse prior when it is degenerate.
            warmup_y.append(yv)
            warmup_x.append(xv)
            if len(warmup_y) >= _BETA_WARMUP:
                wy = np.asarray(warmup_y, dtype=float)
                wx = np.asarray(warmup_x, dtype=float)
                denom = float(np.dot(wx, wx))
                if denom > 1e-12:
                    b = float(np.dot(wx, wy) / denom)
                    p = r / max(denom, 1e-12)
                else:
                    b = 0.0  # diffuse prior fallback
                    p = 1e3
                b_prev = b
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
