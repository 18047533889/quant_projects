# -*- coding: utf-8 -*-
"""Rough-path / path-signature operators (P2, experimental).

Two-input causal operators over aligned daily panels.  The level-2 signature of
a 2-D path is computed from the discrete iterated integrals over the window.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.ts_model._rolling_core import frame_like, metadata

_CANONICALS: list[str] = []


def _register(name: str, description: str, params: list[str], unit: str, fn, cost: int = 8):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.path_signature",
        backend="pandas_numpy",
        status="experimental",
    )
    class _SigOp(SeriesOperator):
        metadata = metadata(name, description, params, unit=unit, cost=cost)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {name}
    )
    return _SigOp


def _apply_two(x: pd.DataFrame, y: pd.DataFrame, fn) -> pd.DataFrame:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = fn(xv[: row + 1, col], yv[: row + 1, col])
    return frame_like(x, out)


def _sig_level2(x: np.ndarray, y: np.ndarray, window: int):
    seg_x = x[-int(window):]
    seg_y = y[-int(window):]
    finite = np.isfinite(seg_x) & np.isfinite(seg_y)
    sx, sy = seg_x[finite], seg_y[finite]
    if len(sx) < 3:
        return None
    dx = np.diff(sx)
    dy = np.diff(sy)
    ax = sx[:-1]
    ay = sy[:-1]
    # iterated integrals: ∫ X dY etc. (discrete rectangle rule)
    s_xdx = float(np.sum(ax * dx))
    s_xdy = float(np.sum(ax * dy))
    s_ydx = float(np.sum(ay * dx))
    s_ydy = float(np.sum(ay * dy))
    area = 0.5 * (s_xdy - s_ydx)
    return s_xdx, s_xdy, s_ydx, s_ydy, area


def _sig_area(x: np.ndarray, y: np.ndarray, window: int) -> float:
    out = _sig_level2(x, y, window)
    return np.nan if out is None else float(out[4])


_register("ts_path_signature_area", "二维路径的有向面积（Levy 面积）。", ["x", "y", "window"], "level",
           lambda x, y, window=60: _apply_two(x, y, lambda a, b: _sig_area(a, b, int(window))))


def _sig_depth2_norm(x: np.ndarray, y: np.ndarray, window: int) -> float:
    out = _sig_level2(x, y, window)
    if out is None:
        return np.nan
    return float(np.sqrt(sum(c * c for c in out[:4])))


_register("ts_path_signature_depth2_norm", "二阶路径签名各项 L2 范数。", ["x", "y", "window"], "level",
           lambda x, y, window=60: _apply_two(x, y, lambda a, b: _sig_depth2_norm(a, b, int(window))))


def _leadlag_area(x: np.ndarray, y: np.ndarray, window: int, lag: int) -> float:
    l = max(1, int(lag))
    seg_x = x[-int(window):]
    seg_y = y[-int(window):]
    if len(seg_x) <= l:
        return np.nan
    return _sig_area(seg_x[:-l], seg_y[l:], len(seg_x) - l)


_register("ts_path_leadlag_area", "lead-lag 变换后的路径有向面积。", ["x", "y", "window", "lag"], "level",
           lambda x, y, window=60, lag=1: _apply_two(x, y, lambda a, b: _leadlag_area(a, b, int(window), int(lag))))
