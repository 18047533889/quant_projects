# -*- coding: utf-8 -*-
"""Binned response-curve operators (2026-08 geometry/math expansion).

Characterise the *conditional response curve* ``y = f(x)`` locally by binning
``x`` over a trailing window and inspecting the per-bin medians ``m_b`` of
``y``:

* ``ts_binned_response_monotonicity`` — Spearman rank correlation between the
  quantile-group index ``1..B`` and the per-bin median of ``y`` (how monotone
  the conditional response is).
* ``ts_binned_response_curvature`` — quadratic fit ``m_b = a + b q + c q^2``
  over quantile centres ``q_b``; the curvature coefficient ``c`` is reported
  relative to the dispersion of the bin medians.
* ``ts_response_slope_asymmetry`` — signed gap between the OLS slope of
  ``y ~ x`` on the low side and the high side of ``x`` (positive → convex
  response, negative → concave).

All operators are trailing-window, prefix-causal and deterministic.  Only
same-position finite ``(x, y)`` pairs are used; windows with too few pairs or
too few nonempty bins emit NaN (fail-closed).  Invalid parameters raise
``ValueError``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    aligned_pairs,
    check_window,
    frame_like,
    register_polars_bridge,
)

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="binned_response",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "binned_response", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:response_shape",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# shared kernels (private to this module; 2D panel = TradeDate x Symbol)
# ---------------------------------------------------------------------------
def _rankdata(v: np.ndarray) -> np.ndarray:
    """Average ranks (1-based); ties share the mean rank."""
    n = v.size
    order = np.argsort(v, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(a) - _rankdata(a).mean()
    rb = _rankdata(b) - _rankdata(b).mean()
    denom = float(np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))
    if denom <= _EPS:
        return np.nan
    return float(np.dot(ra, rb) / denom)


def _quantile_groups(xv: np.ndarray, bins: int) -> np.ndarray:
    """Window-local quantile buckets over the values of ``x``."""
    cuts = np.quantile(xv, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    bucket = np.searchsorted(cuts, xv, side="right")
    return np.clip(bucket.astype(np.int64), 0, bins - 1)


def _binned_medians(yv: np.ndarray, xv: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin median of ``y`` by quantile group of ``x``. Returns (medians, counts)."""
    b = _quantile_groups(xv, bins)
    medians = np.full(bins, np.nan, dtype=float)
    counts = np.zeros(bins, dtype=float)
    for g in range(bins):
        sel = b == g
        if sel.sum() > 0:
            medians[g] = float(np.median(yv[sel]))
            counts[g] = float(sel.sum())
    return medians, counts


def _monotonicity_series(y2d: np.ndarray, x2d: np.ndarray, window: int, bins: int) -> np.ndarray:
    rows, cols = y2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, b = int(window), int(bins)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            yv, xv = aligned_pairs(y2d[i0 : r + 1, c], x2d[i0 : r + 1, c])
            if yv.size < b:
                continue
            medians, counts = _binned_medians(yv, xv, b)
            nz = counts > 0
            if int(nz.sum()) < 3:
                continue
            gi = np.arange(1, b + 1, dtype=float)[nz]
            out[r, c] = _spearman(gi, medians[nz])
    return out


def _curvature_series(y2d: np.ndarray, x2d: np.ndarray, window: int, bins: int) -> np.ndarray:
    rows, cols = y2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, b = int(window), int(bins)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            yv, xv = aligned_pairs(y2d[i0 : r + 1, c], x2d[i0 : r + 1, c])
            if yv.size < b:
                continue
            medians, counts = _binned_medians(yv, xv, b)
            nz = counts > 0
            if int(nz.sum()) < 3:
                continue
            q = (np.arange(b) + 0.5) / b
            qq, mm = q[nz], medians[nz]
            X = np.column_stack([np.ones_like(qq), qq, qq ** 2])
            coeff, *_ = np.linalg.lstsq(X, mm, rcond=None)
            sd = float(np.std(mm))
            out[r, c] = float(coeff[2]) / (sd + _EPS)
    return out


def _ols_slope_resid(xv: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
    n = xv.size
    if n < 2:
        return np.nan, np.nan
    xm, ym = float(xv.mean()), float(yv.mean())
    sxx = float(np.dot(xv - xm, xv - xm))
    if sxx <= _EPS:
        return np.nan, np.nan
    beta = float(np.dot(xv - xm, yv - ym) / sxx)
    alpha = ym - beta * xm
    resid = yv - (alpha + beta * xv)
    scale = float(np.sqrt(np.mean(resid ** 2)))
    return beta, scale


def _slope_asymmetry_series(y2d: np.ndarray, x2d: np.ndarray, window: int, split_quantile: float) -> np.ndarray:
    rows, cols = y2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    qq = float(split_quantile)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            yv, xv = aligned_pairs(y2d[i0 : r + 1, c], x2d[i0 : r + 1, c])
            if yv.size < 4:
                continue
            thr = float(np.quantile(xv, qq))
            lo = xv <= thr
            hi = ~lo
            if lo.sum() < 2 or hi.sum() < 2:
                continue
            bL, sL = _ols_slope_resid(xv[lo], yv[lo])
            bH, sH = _ols_slope_resid(xv[hi], yv[hi])
            if not (np.isfinite(bL) and np.isfinite(bH)):
                continue
            bLn = bL / (sL + _EPS)
            bHn = bH / (sH + _EPS)
            denom = abs(bHn) + abs(bLn) + _EPS
            out[r, c] = (bHn - bLn) / denom
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_binned_response_monotonicity",
    category="binned_response",
    business_category="binned_response",
    canonical="ts_binned_response_monotonicity",
    source="binned_response",
)
class TsBinnedResponseMonotonicity(SeriesOperator):
    """条件响应单调性：x 分位组索引 1..B 与组内 y 中位数的 Spearman 相关。

    接近 +1 → y 随 x 单调上升；接近 -1 → 单调下降；0 → 无单调结构。
    要求 ≥B 个有效配对且 ≥3 个非空分位组,否则 NaN。P2。
    """

    metadata = _metadata(
        "ts_binned_response_monotonicity",
        "x 分位组索引与组内 y 中位数的 Spearman 秩相关（单调性）。",
        ["y", "x", "window", "bins"],
        unit="corr",
        cost=4,
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 120, bins: int = 5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        b = int(bins)
        if b < 2:
            raise ValueError("bins must be >= 2")
        return frame_like(y, _monotonicity_series(y.to_numpy(dtype=float), x.to_numpy(dtype=float), w, b))


@register_operator(
    name="ts_binned_response_curvature",
    category="binned_response",
    business_category="binned_response",
    canonical="ts_binned_response_curvature",
    source="binned_response",
)
class TsBinnedResponseCurvature(SeriesOperator):
    """条件响应曲率：m_b 对分位中心 q_b 的二次拟合系数 c,按中位数离散度归一。

    正 → 响应曲线凹向上（加速）；负 → 凸向上（减速）。P2。
    """

    metadata = _metadata(
        "ts_binned_response_curvature",
        "分位响应曲线二次项系数,按 std(m_b) 归一（曲率）。",
        ["y", "x", "window", "bins"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 120, bins: int = 5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        b = int(bins)
        if b < 3:
            raise ValueError("bins must be >= 3 for a quadratic fit")
        return frame_like(y, _curvature_series(y.to_numpy(dtype=float), x.to_numpy(dtype=float), w, b))


@register_operator(
    name="ts_response_slope_asymmetry",
    category="binned_response",
    business_category="binned_response",
    canonical="ts_response_slope_asymmetry",
    source="binned_response",
)
class TsResponseSlopeAsymmetry(SeriesOperator):
    """响应斜率不对称：x 高位侧与低位侧 OLS 斜率差的归一化符号。

    A=(β_H-β_L)/(|β_H|+|β_L|+eps),斜率各自按残差尺度归一。正 → 高位斜率更大
    （凸响应）；负 → 凹响应。split_quantile 必须落在 (0,1)。P2。
    """

    metadata = _metadata(
        "ts_response_slope_asymmetry",
        "x 高低侧 OLS 斜率差的归一化不对称度（凸/凹响应）。",
        ["y", "x", "window", "split_quantile"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 120, split_quantile: float = 0.5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        q = float(split_quantile)
        if not 0.0 < q < 1.0:
            raise ValueError("split_quantile must be in (0, 1)")
        return frame_like(y, _slope_asymmetry_series(y.to_numpy(dtype=float), x.to_numpy(dtype=float), w, q))


_NEW_CANONICALS = (
    "ts_binned_response_monotonicity",
    "ts_binned_response_curvature",
    "ts_response_slope_asymmetry",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
