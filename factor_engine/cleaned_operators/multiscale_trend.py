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
``rs_s = sqrt(mean squared OLS residual)`` and ``T_s = b_s / rs_s`` (a t-stat-like
measure).  A scale requires all ``s`` rows to be finite (the trailing window is
never time-compressed); a zero residual scale (``rs_s == 0``) is degenerate and
fails closed.

*All-scales contract* — a row is emitted only when EVERY declared scale is
present (contiguous-finite, ``rs_s > 0``).  When the data is shorter than the
largest scale, or any one scale is non-finite/degenerate, the row is NaN — the
operator never silently recomputes on a subset of the declared scales.

All operators are trailing-window per-column, prefix-causal, deterministic,
NaN-safe and reject invalid parameters (each scale >= 2, ``scale <= window``).
The ``window`` parameter does not participate in the computation (declared
POLICY / non-searchable); the effective warm-up is ``max(scales)`` rows.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge

# The ``window`` parameter does NOT participate in the computation: the per-scale
# OLS windows ARE the scales (each scale s fits over the last s rows).  It is
# kept in the signature only for back-compat with early recipes and declared
# POLICY / non-searchable (same treatment as ``group_decay_linear``), so a
# search/GP grammar never treats it as an alpha dimension.
_DEAD_WINDOW_SPEC = ParamSpec(
    dtype=int,
    min=2,
    default=60,
    searchable=False,
    param_role=ParamRole.POLICY,
)


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    min_scales: int,
) -> OperatorMetadata:
    scale_item = ParamSpec(dtype=int, min=2)
    scales_spec = ParamSpec(
        alternatives=(
            ParamSpec(dtype=tuple, items=scale_item, min_items=min_scales),
            ParamSpec(dtype=list, items=scale_item, min_items=min_scales),
        ),
        default=(5, 10, 20, 40),
        searchable=False,
        param_role=ParamRole.ESTIMATOR_RESOLUTION,
    )
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
        input_units={"x": "price_level_or_log_price_level"},
        output_unit="dimensionless",
        panel_params=("x",),
        panel_arity=1,
        scalar_params=("window", "scales"),
        param_specs={"window": _DEAD_WINDOW_SPEC, "scales": scales_spec},
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
    amplitude = float(np.max(np.abs(ys)))
    if not np.isfinite(amplitude) or amplitude == 0.0:
        return np.nan, np.nan
    # T_s is invariant to a non-zero multiplicative rescaling.  Normalising
    # first avoids overflow/underflow for otherwise equivalent tiny/huge data.
    ys = ys / amplitude
    xs = np.arange(int(s), dtype=float)
    xbar = xs.mean()
    ybar = ys.mean()
    denom = float(np.dot(xs - xbar, xs - xbar))
    if denom <= 0.0:
        return np.nan, np.nan
    slope = float(np.dot(xs - xbar, ys - ybar) / denom)
    resid = ys - (ybar + slope * (xs - xbar))
    rs = float(np.sqrt(np.mean(resid ** 2)))
    if rs <= np.finfo(float).eps * 8.0:
        return np.nan, np.nan
    return slope, rs


def _trend_pairs(x: np.ndarray, t: int, scales: list[int]) -> list[tuple[int, float]]:
    """``(scale, T_s)`` pairs valid at row ``t``.

    All-scales contract: returns the FULL pair set only when every declared scale
    is present (enough rows, contiguous-finite, non-degenerate residual scale);
    returns ``[]`` otherwise so the caller emits NaN.  No partial-scale recompute
    and no ``+ eps`` floor on the residual-scale denominator.
    """
    pairs: list[tuple[int, float]] = []
    for s in scales:
        if t + 1 < s:
            return []  # data shorter than this scale -> whole row NaN
        slope, rs = _scale_trend(x[t - s + 1 : t + 1], s)
        if not (np.isfinite(slope) and np.isfinite(rs)) or rs <= 0.0:
            return []  # non-finite / degenerate residual -> whole row NaN
        pairs.append((s, slope / rs))
    return pairs




# ---------------------------------------------------------------------------
# R62 vectorised per-scale trend statistics.  ``_scale_trend`` above stays the
# readable reference; the batch version performs the SAME operations on every
# row of one column (all-``s``-rows-finite requirement, ``amplitude``
# normalisation, centred OLS slope, residual scale floor) with one sliding
# window matrix per scale.
# ---------------------------------------------------------------------------
_SCALE_RS_FLOOR = np.finfo(float).eps * 8.0


def _scale_trend_vec(col: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """``_scale_trend`` for every row of one column: (slope, rs)."""
    n = int(col.size)
    slope = np.full(n, np.nan, dtype=float)
    rs = np.full(n, np.nan, dtype=float)
    xs = np.arange(int(s), dtype=float)
    dx = xs - xs.mean()
    sxx = float(np.dot(dx, dx))
    if n < int(s) or sxx <= 0.0:
        return slope, rs
    win = sliding_window_view(col, int(s))              # row j ends at row j+s-1
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.max(np.abs(win), axis=1)
        ys = win / amp[:, None]
        ybar = ys.mean(axis=1)
        sl = (ys - ybar[:, None]) @ dx / sxx
        resid = ys - (ybar[:, None] + sl[:, None] * dx[None, :])
        r = np.sqrt(np.mean(resid * resid, axis=1))
    ok = (np.isfinite(win).all(axis=1) & np.isfinite(amp) & (amp != 0.0)
          & np.isfinite(sl) & np.isfinite(r) & (r > _SCALE_RS_FLOOR))
    end = np.arange(int(s) - 1, n)
    slope[end[ok]] = sl[ok]
    rs[end[ok]] = r[ok]
    return slope, rs


def _scale_t_stats(x2d: np.ndarray, scales: list[int]) -> np.ndarray:
    """``(rows, cols, len(scales))`` of ``T_s = slope / rs`` (NaN when invalid)."""
    rows, cols = x2d.shape
    out = np.full((rows, cols, len(scales)), np.nan, dtype=float)
    for c in range(cols):
        for si, s in enumerate(scales):
            sl, rs = _scale_trend_vec(x2d[:, c], s)
            good = np.isfinite(sl) & np.isfinite(rs) & (rs > 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, c, si] = np.where(good, sl / np.where(rs > 0.0, rs, 1.0), np.nan)
    return out

def _consensus_series(x2d: np.ndarray, scales: list[int]) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if not scales:
        return out
    tstats = _scale_t_stats(x2d, scales)
    all_ok = np.isfinite(tstats).all(axis=2)
    mean_sign = np.sign(tstats).mean(axis=2)
    out[all_ok] = mean_sign[all_ok]
    return out


def _dispersion_series(x2d: np.ndarray, scales: list[int]) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if len(scales) < 2:
        return out
    tstats = _scale_t_stats(x2d, scales)
    all_ok = np.isfinite(tstats).all(axis=2)
    med = np.median(tstats, axis=2)
    mad = np.median(np.abs(tstats - med[:, :, None]), axis=2)
    good = all_ok & np.isfinite(mad)
    out[good] = mad[good]
    return out


def _curvature_series(x2d: np.ndarray, scales: list[int]) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if len(scales) < 3:
        return out
    ls = np.asarray([np.log(float(s)) for s in scales], dtype=float)
    if len(np.unique(np.round(ls, 10))) < 3:
        return out
    tstats = _scale_t_stats(x2d, scales)
    all_ok = np.isfinite(tstats).all(axis=2)
    X = np.column_stack([np.ones(len(ls)), ls, ls ** 2])
    Y = tstats.reshape(rows * cols, len(scales)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    coef = beta[2]
    good = (all_ok.reshape(-1) & np.isfinite(coef))
    flat = out.reshape(-1)
    flat[good] = coef[good]
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
    尺度间方向分歧。所有声明尺度必须齐全（contiguous-finite）才输出，
    否则 NaN（无部分尺度重算）；预热 = max(scales)。P1。
    """

    metadata = _metadata(
        "ts_multiscale_trend_consensus",
        "各尺度趋势统计量 T_s 符号的均值（多尺度方向一致度）。",
        ["x", "window", "scales"],
        unit="ratio",
        cost=5,
        min_scales=1,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, scales: tuple[int, ...] = (5, 10, 20, 40), **_: Any
    ) -> pd.DataFrame:
        # ``window`` is a POLICY back-compat knob that does not participate in the
        # computation (the scales ARE the windows); it only bounds the scales.
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        ss = _normalise_scales(scales)
        for s in ss:
            if s > w:
                raise ValueError("each scale must be <= window")
        return frame_like(x, _consensus_series(x.to_numpy(dtype=float), ss))


@register_operator(
    name="ts_multiscale_trend_dispersion",
    category="multiscale_trend",
    business_category="multiscale_trend",
    canonical="ts_multiscale_trend_dispersion",
    source="multiscale_trend",
)
class TsMultiscaleTrendDispersion(SeriesOperator):
    """多尺度趋势分散度：{T_s} 的 MAD。

    高 → 短中长期趋势强度/方向差异大（regime 切换、尺度间背离）。所有声明
    尺度必须齐全才输出，否则 NaN；预热 = max(scales)。P1。
    """

    metadata = _metadata(
        "ts_multiscale_trend_dispersion",
        "各尺度趋势统计量 T_s 的中位数绝对偏差（尺度间分歧度）。",
        ["x", "window", "scales"],
        unit="ratio",
        cost=5,
        min_scales=2,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, scales: tuple[int, ...] = (5, 10, 20, 40), **_: Any
    ) -> pd.DataFrame:
        # ``window`` is a POLICY back-compat knob that does not participate in the
        # computation (the scales ARE the windows); it only bounds the scales.
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        ss = _normalise_scales(scales)
        for s in ss:
            if s > w:
                raise ValueError("each scale must be <= window")
        return frame_like(x, _dispersion_series(x.to_numpy(dtype=float), ss))


@register_operator(
    name="ts_multiscale_trend_curvature",
    category="multiscale_trend",
    business_category="multiscale_trend",
    canonical="ts_multiscale_trend_curvature",
    source="multiscale_trend",
)
class TsMultiscaleTrendCurvature(SeriesOperator):
    """多尺度趋势曲率：T_s ~ a + b*log(s) + c*log(s)^2 的二次项系数 c。

    c>0 → 趋势随尺度加速（越长越强）；c<0 → 长尺度衰减。需要 >= 3 个声明
    尺度，且所有声明尺度必须齐全才输出，否则 NaN；预热 = max(scales)。P2。
    """

    metadata = _metadata(
        "ts_multiscale_trend_curvature",
        "T_s 对 log(s) 二次回归的二次项系数（趋势随尺度的曲率）。",
        ["x", "window", "scales"],
        unit="ratio",
        cost=6,
        min_scales=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, scales: tuple[int, ...] = (5, 10, 20, 40), **_: Any
    ) -> pd.DataFrame:
        # ``window`` is a POLICY back-compat knob that does not participate in the
        # computation (the scales ARE the windows); it only bounds the scales.
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        ss = _normalise_scales(scales)
        for s in ss:
            if s > w:
                raise ValueError("each scale must be <= window")
        return frame_like(x, _curvature_series(x.to_numpy(dtype=float), ss))


_NEW_CANONICALS = (
    "ts_multiscale_trend_consensus",
    "ts_multiscale_trend_dispersion",
    "ts_multiscale_trend_curvature",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
