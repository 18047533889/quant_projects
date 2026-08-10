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

    _surface.extend_research_only({name})
    return _SigOp


def _apply_two(x: pd.DataFrame, y: pd.DataFrame, fn) -> pd.DataFrame:
    # Multi-input axes alignment (audit P0): identical index AND columns, so a
    # reordered secondary panel can never pair A's data with B's path.
    if not x.index.equals(y.index) or not x.columns.equals(y.columns):
        raise ValueError(
            "path_signature inputs must share identical index and columns"
        )
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = fn(xv[: row + 1, col], yv[: row + 1, col])
    return frame_like(x, out)


def _trailing_contiguous_xy(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Longest trailing run where BOTH x and y are finite.

    Gap no-compress (audit P0): a missing point splits the path — data on the
    other side of a gap is never re-joined into the same signature.

    R28-P0-001: when the CURRENT row is non-finite, fail closed (return None)
    instead of substituting a stale historical run as the current output.  A
    signature at decision time ``t`` is only defined when the joint input at
    ``t`` is observable.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.shape[0]
    if n == 0 or not valid[-1]:
        return None
    end = n
    start = end
    while start > 0 and valid[start - 1]:
        start -= 1
    if end - start < 1:
        return None
    return x[start:end], y[start:end]


def _sig_level2(x: np.ndarray, y: np.ndarray, window: int):
    seg_x = x[-int(window):]
    seg_y = y[-int(window):]
    pair = _trailing_contiguous_xy(seg_x, seg_y)
    if pair is None:
        return None
    sx, sy = pair
    if len(sx) < 3:
        return None
    # Anchor the path to its first point (translation invariance): Xbar(t) = X(t)
    # - X(0).  Level-2 iterated integrals are then the Chen integrals over the
    # anchored path — the absolute level no longer enters the statistic.
    ax = sx - sx[0]
    ay = sy - sy[0]
    dx = np.diff(sx)
    dy = np.diff(sy)
    s_xdx = float(np.sum(ax[:-1] * dx))  # Σ Xbar_x ΔX  (strictly-lower part)
    s_xdy = float(np.sum(ax[:-1] * dy))  # Σ Xbar_x ΔY   (cross term)
    s_ydx = float(np.sum(ay[:-1] * dx))  # Σ Xbar_y ΔX   (cross term)
    s_ydy = float(np.sum(ay[:-1] * dy))  # Σ Xbar_y ΔY
    # P1-92: for a piecewise-linear path the exact level-2 iterated integral is
    #   S^(2)_{jk} = Σ_i [ Xbar_j(i) ΔS_k(i) + 0.5 ΔS_j(i) ΔS_k(i) ]
    # — the discrete sums above are only the strictly-lower part; the diagonal
    # (1/2)ΔX⊗ΔX term was missing, so the depth-2 norm was not the standard
    # signature norm.  The Levy area is unaffected (the diagonal contribution
    # to ∫∫dX dY equals that of ∫∫dY dX, so it cancels exactly in area).
    s_xdx += 0.5 * float(np.sum(dx * dx))
    s_xdy += 0.5 * float(np.sum(dx * dy))
    s_ydx += 0.5 * float(np.sum(dy * dx))
    s_ydy += 0.5 * float(np.sum(dy * dy))
    area = 0.5 * (s_xdy - s_ydx)  # Levy area (translation-invariant)
    return s_xdx, s_xdy, s_ydx, s_ydy, area


def _sig_area(x: np.ndarray, y: np.ndarray, window: int) -> float:
    out = _sig_level2(x, y, window)
    return np.nan if out is None else float(out[4])


_register("ts_path_signature_area", "二维路径的有向面积（Levy 面积）。", ["x", "y", "window"], "level",
           lambda x, y, window=60: _apply_two(x, y, lambda a, b: _sig_area(a, b, int(window))))


def _robust_scale(vals: np.ndarray) -> float:
    """Past robust scale (MAD) of a series; std / 1.0 fallback.

    Used to make two series dimensionless before forming the depth-2 signature
    norm, so a single series with a much larger scale cannot dominate the norm.
    """
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if np.isfinite(mad) and mad > 1e-12:
        return mad
    sd = float(np.std(vals))
    if np.isfinite(sd) and sd > 1e-12:
        return sd
    return 1.0


def _sig_depth2_norm(x: np.ndarray, y: np.ndarray, window: int) -> float:
    """Depth-2 signature norm over *dimensionless* inputs.

    The raw level-2 terms mix units (Sxx ~ unit(x)^2, Sxy ~ unit(x)*unit(y)), so
    a series with the larger scale would dominate the Euclidean norm.  Each input
    is standardised by its past robust scale (MAD over the trailing window)
    before forming the signature, so every series contributes on equal footing.
    The raw signature statistics remain available via the ``*_area`` / raw
    kernels; only this factor-facing norm uses dimensionless inputs.

    R28 §五十二: the robust scale is computed on the SAME joint finite trailing
    run that feeds the signature (never on the raw NaN-containing suffix), so the
    normalisation footprint and the signature footprint always agree.
    """
    seg_x = x[-int(window):]
    seg_y = y[-int(window):]
    if len(seg_x) == 0 or len(seg_y) == 0:
        return np.nan
    pair = _trailing_contiguous_xy(seg_x, seg_y)
    if pair is None:
        return np.nan
    rx, ry = pair
    sx = _robust_scale(rx)
    sy = _robust_scale(ry)
    out = _sig_level2(rx / sx, ry / sy, len(rx))
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
