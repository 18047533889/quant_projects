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
(upper_j - lower_j) - 1`` (already price-level independent: a dimensionless
position inside the band, not an absolute width).  A *degenerate* zero-width or
inverted envelope (``upper_j - lower_j <= 0``) is NaN — no ``+ eps`` denominator,
no clip to [-1, 1] — so a constant/zero-width envelope fails closed instead of
blowing up (2026-08 round-3).  The compression width ratio is
``(upper-lower)/|mid|``: a %-based, price-level normalised width, so a 100-yuan
stock and a 10-yuan stock with the same *relative* envelope produce the same
value (comparable across price levels).

All operators are trailing-window per-column, prefix-causal, deterministic,
NaN-safe and reject invalid parameters.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="envelope",
        description=description,
        param_names=params,
        panel_params=tuple(p for p in params if p in {"x", "upper", "lower", "mid"}),
        scalar_params=tuple(p for p in params if p in {"window", "quantile"}),
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, param_role=ParamRole.HORIZON),
            **({"quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.8,
                                     param_role=ParamRole.STATE_THRESHOLD)}
               if "quantile" in params else {}),
        },
        return_type="series",
        tags=[
            "envelope", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:envelope",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _normalised_position(x: float, upper: float, lower: float) -> float:
    """p_j for one row; a degenerate (zero-width/inverted) envelope is NaN.

    No ``+ eps`` floor on the width denominator and no clip: a constant
    envelope fails closed rather than producing a fabricated ±1.
    """
    wd = upper - lower
    if wd <= 0.0:
        return np.nan
    return 2.0 * (x - lower) / wd - 1.0


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
                and u[t] - l[t] > 0.0 and m[t] != 0.0
            ):
                # %-based width: price-level normalised by |mid| (no EPS floor).
                width[t] = (u[t] - l[t]) / abs(m[t])
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
                if not np.isfinite(pj):
                    continue  # degenerate past envelope -> skip, never clip
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
                if not np.isfinite(pj):
                    continue  # degenerate past envelope -> skip, never clip
                total += 1
                if abs(pj) >= q:
                    near += 1
            if total > 0:
                out[t, c] = near / total
    return out


def _polars_envelope(kind, panels, window=20, quantile=0.8):
    """Native expression implementation of the authored three-panel formulas."""
    import polars as pl
    from factor_engine.cleaned_operators.common.strict_params import strict_int, strict_float

    w = strict_int(window, "window", minimum=2)
    q = strict_float(quantile, "quantile")
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    first, second, third = panels
    if not all(isinstance(p, pl.DataFrame) for p in panels):
        raise TypeError("envelope inputs must be Polars wide DataFrame panels")
    if any(p.shape != first.shape or p.columns != first.columns for p in panels[1:]):
        raise ValueError("envelope panel axes must match")
    output = []
    for column in first.columns:
        if column in {"date", "stock_code"}:
            output.append(first[column])
            continue
        # A bounded three-column workspace per instrument, never a whole dataset copy.
        frame = pl.DataFrame({"a": first[column], "b": second[column], "c": third[column]})
        a, b, c = (pl.col(k).cast(pl.Float64) for k in ("a", "b", "c"))
        finite = a.is_finite() & b.is_finite() & c.is_finite()
        if kind == "compression":
            width = (a-b)/c.abs()
            value = pl.when(finite & (a>b) & (c!=0) & width.is_finite()).then(width).otherwise(None)
        else:
            position = 2*(a-c)/(b-c)-1
            value = pl.when(finite & (b>c) & position.is_finite()).then(position).otherwise(None)
        frame = frame.with_columns(value.alias("v"))
        v = pl.col("v")
        count = v.is_not_null().cast(pl.Float64).rolling_sum(w, min_samples=1)
        if kind == "compression":
            result = 1-v.rolling_rank(w, method="max", min_samples=1)/count
        elif kind == "dwell":
            near = pl.when(v.is_not_null()).then(v.abs()>=q).otherwise(None).cast(pl.Float64)
            result = near.rolling_sum(w, min_samples=1)/count
        elif kind == "pressure":
            ordinal = pl.int_range(0, pl.len()).cast(pl.Float64)
            start = (ordinal-w+1).clip(lower_bound=0)
            numerator = (v*(ordinal+1)).rolling_sum(w, min_samples=1)-start*v.rolling_sum(w,min_samples=1)
            weights = pl.when(v.is_not_null()).then(ordinal+1).otherwise(None)
            denominator = weights.rolling_sum(w,min_samples=1)-start*count
            result = numerator/denominator
        else:
            raise ValueError(f"unknown envelope formula: {kind}")
        result = pl.when(v.is_not_null() & result.is_finite()).then(result).otherwise(None)
        output.append(frame.select(result.alias(column)).to_series())
    return pl.DataFrame(output)


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
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
