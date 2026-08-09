# -*- coding: utf-8 -*-
"""Multiscale trend term structure (2026-08 geometry/math expansion).

A trend is not one number: the same series read at different look-backs yields
different slopes, and the *term structure* of those slopes is informative.

* ``ts_multiscale_trend_consensus``  — fraction of scales that agree on trend sign
  (mean of ``sign(T_s)`` in [-1, 1]).
* ``ts_multiscale_trend_dispersion`` — spread of the per-scale trend statistics
  ``T_s`` (MAD across scales); disagreement between horizons.
* ``ts_multiscale_trend_curvature``  — quadratic fit ``T_s = a + b*log(s) +
  c*log(s)^2``; the ``c`` coefficient captures horizon-dependent trend
  acceleration (upward trend steepening / decaying with horizon).

Shared kernel — *per-scale trend statistic*.  For scale ``s``, OLS slope ``b_s``
is fit over the last ``s`` rows and normalised by the residual scale:
``rs_s = sqrt(mean squared OLS residual)`` and ``T_s = b_s / (rs_s + eps)``.  A
scale requires all ``s`` rows to be finite (the trailing window is never time-
compressed); otherwise ``T_s`` is NaN for that row.

All operators are trailing-window per-column, prefix-causal, deterministic,
NaN-safe and reject invalid parameters (``window``, each scale >= 2 and
``scale <= window``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="multiscale_trend",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "multiscale_trend", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:trend",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _normalise_scales(scales: Any) -> list[int]:
    if isinstance(scales, (int, np.integer, float, np.floating)):
        scales = (scales,)
    if not isinstance(scales, (tuple, list)):
        raise ValueError("scales must be a tuple/list of integers >= 2")
    out: list[int] = []
    for s in scales:
        if isinstance(s, (bool, np.bool_)):
            raise ValueError("scales must be integers >= 2, not bool")
        fv = float(s)
        if not np.isfinite(fv) or fv != float(int(fv)):
            raise ValueError("scales must be integers >= 2")
        iv = int(fv)
        if iv < 2:
            raise ValueError("scales must be integers >= 2")
        out.append(iv)
    if not out:
        raise ValueError("scales must contain at least one scale")
    # Duplicate scales would double-weight one horizon and silently change the
    # statistic — reject them (P1-43).
    if len(set(out)) != len(out):
        raise ValueError("scales must be unique (duplicate scales are rejected)")
    return out


def _scale_trend(chunk: np.ndarray, s: int) -> tuple[float, float]:
    """OLS slope and residual scale over the last ``s`` rows of ``chunk``."""
    ys = chunk[-int(s):]
    if not np.isfinite(ys).all():
        return np.nan, np.nan
    xs = np.arange(int(s), dtype=float)
    xbar = xs.mean()
    ybar = ys.mean()
    denom = float(np.dot(xs - xbar, xs - xbar))
    if denom <= 0.0:
        return np.nan, np.nan
    slope = float(np.dot(xs - xbar, ys - ybar) / denom)
    resid = ys - (ybar + slope * (xs - xbar))
    rs = float(np.sqrt(np.mean(resid ** 2)))
    return slope, rs


def _trend_pairs(x: np.ndarray, t: int, scales: list[int]) -> list[tuple[int, float]]:
    """``(scale, T_s)`` pairs valid at row ``t`` (NaN-invalid ones dropped)."""
    pairs: list[tuple[int, float]] = []
    for s in scales:
        if t + 1 < s:
            continue
        slope, rs = _scale_trend(x[t - s + 1 : t + 1], s)
        if np.isfinite(slope) and np.isfinite(rs):
            pairs.append((s, slope / (rs + _EPS)))
    return pairs


def _consensus_series(x2d: np.ndarray, window: int, scales: list[int]) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        x = x2d[:, c]
        for t in range(rows):
            pairs = _trend_pairs(x, t, scales)
            if not pairs:
                continue
            out[t, c] = float(np.mean(np.sign([T for _s, T in pairs])))
    return out


def _dispersion_series(x2d: np.ndarray, window: int, scales: list[int]) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        x = x2d[:, c]
        for t in range(rows):
            pairs = _trend_pairs(x, t, scales)
            if len(pairs) < 2:
                continue
            ts = np.asarray([T for _s, T in pairs], dtype=float)
            med = float(np.median(ts))
            out[t, c] = float(np.median(np.abs(ts - med)))
    return out


def _curvature_series(x2d: np.ndarray, window: int, scales: list[int]) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        x = x2d[:, c]
        for t in range(rows):
            pairs = _trend_pairs(x, t, scales)
            if len(pairs) < 3:
                continue
            ls = np.asarray([np.log(float(s)) for s, _T in pairs], dtype=float)
            y = np.asarray([T for _s, T in pairs], dtype=float)
            if len(np.unique(np.round(ls, 10))) < 3:
                continue
            X = np.column_stack([np.ones(len(ls)), ls, ls ** 2])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            coef = float(beta[2])
            if np.isfinite(coef):
                out[t, c] = coef
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_multiscale_trend_consensus",
    category="multiscale_trend",
    business_category="multiscale_trend",
    canonical="ts_multiscale_trend_consensus",
    source="multiscale_trend",
)
class TsMultiscaleTrendConsensus(SeriesOperator):
    """多尺度趋势方向一致性：各尺度 T_s 符号均值 ∈ [-1,1]。

    接近 +1 → 所有短/中/长尺度一致上行；接近 -1 → 一致下行；接近 0 →
    尺度间方向分歧。P1。
    """

    metadata = _metadata(
        "ts_multiscale_trend_consensus",
        "各尺度趋势统计量 T_s 符号的均值（多尺度方向一致度）。",
        ["x", "window", "scales"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, scales: tuple[int, ...] = (5, 10, 20, 40), **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        ss = _normalise_scales(scales)
        for s in ss:
            if s > w:
                raise ValueError("each scale must be <= window")
        return frame_like(x, _consensus_series(x.to_numpy(dtype=float), w, ss))


@register_operator(
    name="ts_multiscale_trend_dispersion",
    category="multiscale_trend",
    business_category="multiscale_trend",
    canonical="ts_multiscale_trend_dispersion",
    source="multiscale_trend",
)
class TsMultiscaleTrendDispersion(SeriesOperator):
    """多尺度趋势分散度：{T_s} 的 MAD。

    高 → 短中长期趋势强度/方向差异大（regime 切换、尺度间背离）。P1。
    """

    metadata = _metadata(
        "ts_multiscale_trend_dispersion",
        "各尺度趋势统计量 T_s 的中位数绝对偏差（尺度间分歧度）。",
        ["x", "window", "scales"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, scales: tuple[int, ...] = (5, 10, 20, 40), **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        ss = _normalise_scales(scales)
        for s in ss:
            if s > w:
                raise ValueError("each scale must be <= window")
        return frame_like(x, _dispersion_series(x.to_numpy(dtype=float), w, ss))


@register_operator(
    name="ts_multiscale_trend_curvature",
    category="multiscale_trend",
    business_category="multiscale_trend",
    canonical="ts_multiscale_trend_curvature",
    source="multiscale_trend",
)
class TsMultiscaleTrendCurvature(SeriesOperator):
    """多尺度趋势曲率：T_s ~ a + b*log(s) + c*log(s)^2 的二次项系数 c。

    c>0 → 趋势随尺度加速（越长越强）；c<0 → 长尺度衰减。需要 >= 3 个尺度，
    否则输出 NaN。P2。
    """

    metadata = _metadata(
        "ts_multiscale_trend_curvature",
        "T_s 对 log(s) 二次回归的二次项系数（趋势随尺度的曲率）。",
        ["x", "window", "scales"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, scales: tuple[int, ...] = (5, 10, 20, 40), **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        ss = _normalise_scales(scales)
        for s in ss:
            if s > w:
                raise ValueError("each scale must be <= window")
        return frame_like(x, _curvature_series(x.to_numpy(dtype=float), w, ss))


_NEW_CANONICALS = (
    "ts_multiscale_trend_consensus",
    "ts_multiscale_trend_dispersion",
    "ts_multiscale_trend_curvature",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
