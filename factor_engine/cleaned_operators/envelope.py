# -*- coding: utf-8 -*-
"""Envelope quality operators (2026-08 geometry/math expansion).

An envelope (upper/lower bands around a mid line) is a range a price is expected
to respect.  This module measures how the price actually interacts with the
envelope:

* ``ts_envelope_compression``        — 1 minus the trailing percentile rank of the
  current band width; near 1 means the envelope has compressed to its historical
  extreme (volatility squeeze).
* ``ts_envelope_pressure``           — recency-weighted mean normalised position of
  the price inside the bands (``p_j`` in [-1, 1] at the lower/upper band,
  > 1 / < -1 outside the envelope).
* ``ts_envelope_boundary_dwell``     — fraction of the window where the price is
  near/at the bands (``|p_j| >= quantile``).

Shared kernel — *normalised envelope position* ``p_j = 2*(x_j - lower_j) /
(upper_j - lower_j + eps) - 1``.  A row with a *degenerate* zero-width or
inverted envelope (``upper_j - lower_j <= 0``) is handled with a clip to [-1, 1]
(so it cannot blow up the denominator) and, if the *current* row is degenerate,
the operator emits NaN for that row.

All operators are trailing-window per-column, prefix-causal, deterministic,
NaN-safe and reject invalid parameters.
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
        category="envelope",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "envelope", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:envelope",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _clip(x: float) -> float:
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)


def _normalised_position(x: float, upper: float, lower: float) -> float:
    """p_j for one row; degenerate width is clipped to [-1, 1]."""
    wd = upper - lower
    if wd <= 0.0:
        return _clip(2.0 * (x - lower) / _EPS - 1.0)
    return 2.0 * (x - lower) / (wd + _EPS) - 1.0


def _valid_triple(x: float, upper: float, lower: float) -> bool:
    return np.isfinite(x) and np.isfinite(upper) and np.isfinite(lower)


def _compression_series(upper2d: np.ndarray, lower2d: np.ndarray, mid2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = upper2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        u, l, m = upper2d[:, c], lower2d[:, c], mid2d[:, c]
        width = np.full(rows, np.nan, dtype=float)
        for t in range(rows):
            if (
                np.isfinite(u[t]) and np.isfinite(l[t]) and np.isfinite(m[t])
                and u[t] - l[t] > 0.0
            ):
                width[t] = (u[t] - l[t]) / (abs(m[t]) + _EPS)
        for t in range(rows):
            if not np.isfinite(width[t]):
                continue
            i0 = max(0, t - w + 1)
            chunk = width[i0 : t + 1]
            v = chunk[np.isfinite(chunk)]
            if v.size == 0:
                continue
            rank = float(np.sum(v <= width[t])) / v.size
            out[t, c] = 1.0 - rank
    return out


def _pressure_series(x2d: np.ndarray, upper2d: np.ndarray, lower2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        x, u, l = x2d[:, c], upper2d[:, c], lower2d[:, c]
        for t in range(rows):
            if not (_valid_triple(x[t], u[t], l[t]) and u[t] - l[t] > 0.0):
                continue  # current row degenerate/non-finite -> NaN
            i0 = max(0, t - w + 1)
            num = 0.0
            den = 0.0
            for j in range(i0, t + 1):
                if not _valid_triple(x[j], u[j], l[j]):
                    continue
                pj = _normalised_position(x[j], u[j], l[j])
                wj = float(j - i0 + 1)
                num += wj * pj
                den += wj
            if den <= 0.0:
                continue
            out[t, c] = num / den
    return out


def _boundary_dwell_series(
    x2d: np.ndarray, upper2d: np.ndarray, lower2d: np.ndarray, window: int, quantile: float
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    q = float(quantile)
    for c in range(cols):
        x, u, l = x2d[:, c], upper2d[:, c], lower2d[:, c]
        for t in range(rows):
            if not (_valid_triple(x[t], u[t], l[t]) and u[t] - l[t] > 0.0):
                continue  # current row degenerate/non-finite -> NaN
            i0 = max(0, t - w + 1)
            total = 0
            near = 0
            for j in range(i0, t + 1):
                if not _valid_triple(x[j], u[j], l[j]):
                    continue
                pj = _normalised_position(x[j], u[j], l[j])
                total += 1
                if abs(pj) >= q:
                    near += 1
            if total > 0:
                out[t, c] = near / total
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_envelope_compression",
    category="envelope",
    business_category="envelope",
    canonical="ts_envelope_compression",
    source="envelope",
)
class TsEnvelopeCompression(SeriesOperator):
    """包络带宽压缩度：1 - 当前带宽在窗口内的分位排名。

    接近 1 → 当前包络收窄至历史极端（波动压缩/蓄势）；接近 0 → 扩张。P1。
    """

    metadata = _metadata(
        "ts_envelope_compression",
        "1 - 当前包络宽度的滚动分位排名（带宽极端压缩≈1）。",
        ["upper", "lower", "mid", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, upper: pd.DataFrame, lower: pd.DataFrame, mid: pd.DataFrame, window: int = 20, **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        return frame_like(
            upper,
            _compression_series(
                upper.to_numpy(dtype=float), lower.to_numpy(dtype=float), mid.to_numpy(dtype=float), w
            ),
        )


@register_operator(
    name="ts_envelope_pressure",
    category="envelope",
    business_category="envelope",
    canonical="ts_envelope_pressure",
    source="envelope",
)
class TsEnvelopePressure(SeriesOperator):
    """包络内价格压力：窗口内线性新近加权的 p_j 均值。

    p_j∈[-1,1] 对应贴着下/上轨；>1/< -1 表示价格在包络之外（极端压力）。P1。
    """

    metadata = _metadata(
        "ts_envelope_pressure",
        "价格在包络内的新近加权归一化位置（线性 recency 权重）。",
        ["x", "upper", "lower", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, x: pd.DataFrame, upper: pd.DataFrame, lower: pd.DataFrame, window: int = 20, **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        return frame_like(
            x,
            _pressure_series(
                x.to_numpy(dtype=float), upper.to_numpy(dtype=float), lower.to_numpy(dtype=float), w
            ),
        )


@register_operator(
    name="ts_envelope_boundary_dwell",
    category="envelope",
    business_category="envelope",
    canonical="ts_envelope_boundary_dwell",
    source="envelope",
)
class TsEnvelopeBoundaryDwell(SeriesOperator):
    """包络边界驻留比例：|p_j| >= quantile 的窗口占比。

    高 → 价格长时间贴近/超出上下轨（趋势/挤压边缘）；低 → 价格沿中枢游走。P1。
    """

    metadata = _metadata(
        "ts_envelope_boundary_dwell",
        "窗口内 |p_j| >= quantile 的比例（贴轨/出界驻留）。",
        ["x", "upper", "lower", "window", "quantile"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        upper: pd.DataFrame,
        lower: pd.DataFrame,
        window: int = 20,
        quantile: float = 0.8,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        q = float(quantile)
        if not np.isfinite(q) or not (0.0 < q < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        return frame_like(
            x,
            _boundary_dwell_series(
                x.to_numpy(dtype=float), upper.to_numpy(dtype=float), lower.to_numpy(dtype=float), w, q
            ),
        )


_NEW_CANONICALS = (
    "ts_envelope_compression",
    "ts_envelope_pressure",
    "ts_envelope_boundary_dwell",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
