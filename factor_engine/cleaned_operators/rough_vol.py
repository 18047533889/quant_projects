# -*- coding: utf-8 -*-
"""Volatility roughness operators (2026-08 geometry/math expansion).

Rough-volatility models describe how the *increment variation* of a price path
scales with the sampling lag.  For ``V_p(Delta) = sum |Delta_Delta x|^p`` a pure
fractional process obeys ``log V_p(Delta) ~ a + p*H*log Delta``, so the slope of
``log V_p`` vs ``log Delta`` divided by ``p`` estimates the Hurst/roughness
parameter ``H`` (``H`` close to 0.5 = Brownian; ``H < 0.5`` = rougher).

* ``ts_vol_pvariation_roughness`` — Hurst ``H`` from a single slope fit over the
  user-supplied scale grid (``scales=(1,2,4)``).
* ``ts_vol_scaling_break`` — short-lag Hurst minus long-lag Hurst
  (``H(1,2) - H(8,16)``); a large positive value means the process is rough at
  short lags but smooth at long lags (volatility-regime/scale break).

Both are trailing-window, prefix-causal and deterministic.  The original time
axis is preserved (review R4-59): lag increments are taken at their true lag
positions and an increment whose endpoint is NaN contributes nothing to
``V_p`` — values are never dropped and re-connected, which would change the
real time meaning of the lags (1/2/4/8/16).  Windows with too few finite
observations to form the required increments emit NaN, never Inf.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import check_window, frame_like, register_polars_bridge

_EPS = 1e-12
_MIN_FINITE = 5
_LONG_LAG = 16  # ts_vol_scaling_break needs N >= LONG_LAG+1 increments


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_volatility",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_volatility", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:rough_volatility",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _rolling_values(values: np.ndarray, window: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """Trailing-window per-column reduction over a price/level panel.

    Review R4-59: the raw window (missing values kept at their true positions)
    is passed to ``fn``; lag increments are computed at their true lag offsets
    and NaN-endpoint increments are skipped — never drop-and-reconnect.
    """
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = values[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            v = col[i0 : r + 1]
            if np.isfinite(v).sum() < _MIN_FINITE:
                continue
            out[r, c] = fn(v)
    return out


def _validate_p(p: float) -> float:
    pp = float(p)
    if not np.isfinite(pp) or pp <= 0.0:
        raise ValueError("p must be a positive finite number")
    return pp


def _validate_scales(scales: Any) -> tuple[int, ...]:
    """Strict positive-integer scale grid (review R4-60).

    A float that is not an exact integer (e.g. ``1.9``) is rejected instead of
    being truncated to ``1`` and silently merged with a duplicate grid.
    """
    if isinstance(scales, (int, float, np.integer, np.floating)) and not isinstance(scales, bool):
        scales = (scales,)
    if not isinstance(scales, (tuple, list)) or len(scales) < 2:
        raise ValueError("scales must be a sequence of at least two positive integers")
    out: list[int] = []
    for s in scales:
        if isinstance(s, (bool, np.bool_)):
            raise ValueError("scales must contain only positive integers")
        if isinstance(s, (int, np.integer)):
            si = int(s)
        elif isinstance(s, (float, np.floating)):
            if not np.isfinite(float(s)) or float(s) != float(int(s)):
                raise ValueError(f"scale {s!r} must be a positive integer")
            si = int(s)
        else:
            try:
                si = int(s)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"scale {s!r} must be a positive integer") from exc
            if float(s) != float(si):
                raise ValueError(f"scale {s!r} must be a positive integer")
        if si < 1:
            raise ValueError("scales must contain only positive integers")
        out.append(si)
    cleaned = tuple(sorted(set(out)))
    if len(cleaned) < 2:
        raise ValueError("scales must contain at least two distinct values")
    return cleaned


def _increment_variation(v: np.ndarray, delta: int, p: float) -> float:
    """``sum |x_{t+delta} - x_t|^p`` on the ORIGINAL time axis (review R4-59).

    The lag-delta increment at position ``t`` is ``v[t+delta]-v[t]`` regardless
    of gaps elsewhere; an increment whose either endpoint is NaN contributes
    nothing to ``V_p``.  Values are never dropped and re-connected.
    """
    if int(v.size) <= int(delta):
        return np.nan
    inc = v[delta:] - v[:-delta]
    finite = np.isfinite(inc)
    if not np.any(finite):
        return np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        var = float(np.sum(np.abs(inc[finite]) ** p))
    return var


def _pv_roughness(v: np.ndarray, p: float, scales: tuple[int, ...]) -> float:
    pts: list[tuple[float, float]] = []
    for d in scales:
        var = _increment_variation(v, d, p)
        if np.isfinite(var) and var > _EPS:
            pts.append((float(np.log(d)), float(np.log(var))))
    if len(pts) < 2:
        return np.nan
    xs = np.asarray([a for a, _ in pts], dtype=float)
    ys = np.asarray([b for _, b in pts], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0])
    return float(slope / p)


def _scaling_break(v: np.ndarray, p: float) -> float:
    n = int(v.size)
    if n < _LONG_LAG + 1:
        return np.nan
    v1 = _increment_variation(v, 1, p)
    v2 = _increment_variation(v, 2, p)
    v8 = _increment_variation(v, 8, p)
    v16 = _increment_variation(v, _LONG_LAG, p)
    if not all(np.isfinite(x) and x > _EPS for x in (v1, v2, v8, v16)):
        return np.nan
    # slope_short = (log V(2) - log V(1)) / (log 2 - log 1); log 1 = 0.
    h_short = float((np.log(v2) - np.log(v1)) / np.log(2.0) / p)
    # slope_long = (log V(16) - log V(8)) / (log 16 - log 8) = ... / log 2.
    h_long = float((np.log(v16) - np.log(v8)) / np.log(2.0) / p)
    return float(h_short - h_long)


@register_operator(
    name="ts_vol_pvariation_roughness",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_vol_pvariation_roughness",
    source="rough_vol",
)
class TsVolPvariationRoughness(SeriesOperator):
    """p-变差尺度指数：``log V_p(Δ)`` 对 ``log Δ`` 的斜率 / p → Hurst/roughness ``H``。

    这是**通用 p-变差 scaling 指数**（generic p-variation scaling exponent）：
    输入任意 level 序列 ``x``（价格、波动率、对数波动率……），估计的是该序列
    增量变差的尺度行为，并非特指「波动率过程的 H」。``V_p(Δ)=Σ|Δ_Δ x|^p``
    （滞后 Δ 增量，保留原时间轴，R4-59）。在 ``scales`` 网格上最小二乘拟合
    ``log V_p(Δ)=a+b·log Δ``，``H=b/p``。``H≈0.5`` = 布朗运动，``H<0.5`` = 粗糙。
    不足 2 个有效 scale → NaN。单位 ratio。P1。精确泛用名
    ``ts_pvariation_scaling_exponent`` 为本算子的别名。
    """

    metadata = _metadata(
        "ts_vol_pvariation_roughness",
        "p-变差尺度指数（Hurst/roughness H；通用 scaling exponent）。",
        ["x", "window", "p", "scales"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120, p: float = 2.0, scales: tuple[int, ...] = (1, 2, 4), **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        pp = _validate_p(p)
        sc = _validate_scales(scales)
        return frame_like(x, _rolling_values(x.to_numpy(dtype=float), w, lambda v: _pv_roughness(v, pp, sc)))


@register_operator(
    name="ts_vol_scaling_break",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_vol_scaling_break",
    source="rough_vol",
)
class TsVolScalingBreak(SeriesOperator):
    """尺度断裂：短滞后 Hurst 与长滞后 Hurst 之差 ``H(1,2) - H(8,16)``。

    分别用滞后 (1,2) 与 (8,16) 的 ``log V_p`` 斜率 / p 估 H。正值 = 短程粗糙、长程
    平滑（波动尺度断裂/regime 结构）；≈0 = 单一尺度行为。有效值不足（滞后 16 需要
    ≥17 个有限观测）→ NaN。单位 ratio。P2。
    """

    metadata = _metadata(
        "ts_vol_scaling_break",
        "短程 vs 长程 Hurst 指数之差。",
        ["x", "window", "p"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120, p: float = 2.0, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        pp = _validate_p(p)
        return frame_like(x, _rolling_values(x.to_numpy(dtype=float), w, lambda v: _scaling_break(v, pp)))


_NEW_CANONICALS = (
    "ts_vol_pvariation_roughness",
    "ts_vol_scaling_break",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


def _register_precise_alias() -> None:
    """R4-61: expose the precise generic name as an alias.

    The canonical stays ``ts_vol_pvariation_roughness`` (polars_geometry_math
    hard-codes that name in its polars-backend list, so a canonical rename is
    not loadable without editing that module).  The generic spelling
    ``ts_pvariation_scaling_exponent`` is registered as an alias so new DSL
    expressions can use the precise name; old expressions keep working.
    """
    from cleaned_operators.registry import OperatorRegistry

    OperatorRegistry.register_alias(
        "ts_pvariation_scaling_exponent",
        "ts_vol_pvariation_roughness",
    )


_register_surface()
_register_precise_alias()
