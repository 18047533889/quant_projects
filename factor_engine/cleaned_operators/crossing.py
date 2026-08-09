# -*- coding: utf-8 -*-
"""Crossing-event quality operators (2026-08 geometry/math expansion).

A *crossing* of ``x`` through ``y`` (``z = x - y``) is a categorical event, but
not every crossing is equal: some slice through quietly, some explode through.
These operators score the *geometry* of a crossing bar:

* ``ts_crossing_speed``        — signed jump size across the crossing bar,
  normalised by the rolling volatility of ``z``.  Up-cross: +|dz|/(scale_z+eps);
  down-cross: -|dz|/(scale_z+eps); non-crossing bars are 0.
* ``ts_crossing_acceleration`` — second difference of ``z`` on crossing bars only
  (``a_t = (dz_t - dz_{t-1})/(scale_z+eps)``), 0 elsewhere.

Shared kernel — ``scale_z`` is the rolling standard deviation of ``z`` over the
trailing window (at least two finite values; otherwise non-finite -> 0 output).
Crossing detection is exact: up-cross requires ``z_{t-1} <= 0 < z_t`` and down-
cross ``z_{t-1} >= 0 > z_t``; bars with non-finite ``z`` never cross.

All operators are trailing-window per-column, prefix-causal, deterministic,
NaN-safe (non-crossing bars emit 0.0) and reject invalid parameters.
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
        category="crossing",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "crossing", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:crossing",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _crossing_parts(x2d: np.ndarray, y2d: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(z, dz, scale_z)`` 2D panels for all columns."""
    rows, cols = x2d.shape
    z = np.full((rows, cols), np.nan, dtype=float)
    dz = np.full((rows, cols), np.nan, dtype=float)
    scale = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        x, y = x2d[:, c], y2d[:, c]
        zc = np.full(rows, np.nan, dtype=float)
        for t in range(rows):
            if np.isfinite(x[t]) and np.isfinite(y[t]):
                zc[t] = x[t] - y[t]
        for t in range(1, rows):
            if np.isfinite(zc[t]) and np.isfinite(zc[t - 1]):
                dz[t, c] = zc[t] - zc[t - 1]
        sc = np.full(rows, np.nan, dtype=float)
        for t in range(rows):
            i0 = max(0, t - w + 1)
            v = zc[i0 : t + 1]
            v = v[np.isfinite(v)]
            if v.size >= 2:
                sc[t] = float(np.std(v))
        z[:, c] = zc
        scale[:, c] = sc
    return z, dz, scale


def _crossing_flags(z: np.ndarray) -> np.ndarray:
    """+1 up-cross, -1 down-cross, 0 otherwise, per row (aligned to ``z``)."""
    rows, cols = z.shape
    flags = np.zeros((rows, cols), dtype=float)
    for t in range(1, rows):
        for c in range(cols):
            zt, ztm = z[t, c], z[t - 1, c]
            if not (np.isfinite(zt) and np.isfinite(ztm)):
                continue
            if ztm <= 0.0 < zt:
                flags[t, c] = 1.0
            elif ztm >= 0.0 > zt:
                flags[t, c] = -1.0
    return flags


def _crossing_speed_series(x2d: np.ndarray, y2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    z, dz, scale = _crossing_parts(x2d, y2d, window)
    flags = _crossing_flags(z)
    out = np.zeros((rows, cols), dtype=float)
    for t in range(1, rows):
        for c in range(cols):
            f = flags[t, c]
            if f == 0.0 or not np.isfinite(scale[t, c]):
                continue  # non-crossing bar / non-finite scale -> 0.0
            mag = abs(dz[t, c]) if np.isfinite(dz[t, c]) else 0.0
            out[t, c] = (mag / (scale[t, c] + _EPS)) * f
    return out


def _crossing_acceleration_series(x2d: np.ndarray, y2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    z, dz, scale = _crossing_parts(x2d, y2d, window)
    flags = _crossing_flags(z)
    out = np.zeros((rows, cols), dtype=float)
    for t in range(2, rows):
        for c in range(cols):
            if flags[t, c] == 0.0 or not np.isfinite(scale[t, c]):
                continue  # non-crossing bar / non-finite scale -> 0.0
            if not (np.isfinite(dz[t, c]) and np.isfinite(dz[t - 1, c])):
                continue
            a = (dz[t, c] - dz[t - 1, c]) / (scale[t, c] + _EPS)
            if np.isfinite(a):
                out[t, c] = a
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_crossing_speed",
    category="crossing",
    business_category="crossing",
    canonical="ts_crossing_speed",
    source="crossing",
)
class TsCrossingSpeed(SeriesOperator):
    """穿越速度：穿越当根 z 跳变幅度 / 滚动波动率（上穿为正、下穿为负）。

    非穿越 bar 输出 0。高 |值| → 价格以显著动量刺穿均线/阈值。P1。
    """

    metadata = _metadata(
        "ts_crossing_speed",
        "穿越 bar 的带符号 z 跳变幅度 / 滚动 std（上穿+/下穿-）。",
        ["x", "y", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        return frame_like(
            x,
            _crossing_speed_series(x.to_numpy(dtype=float), y.to_numpy(dtype=float), w),
        )


@register_operator(
    name="ts_crossing_acceleration",
    category="crossing",
    business_category="crossing",
    canonical="ts_crossing_acceleration",
    source="crossing",
)
class TsCrossingAcceleration(SeriesOperator):
    """穿越加速度：仅在穿越 bar 上输出 (dz_t - dz_{t-1})/滚动 std，其余为 0。

    正值 → 穿越在加速（动量增强）；负值 → 穿越在减速（动能衰竭）。P2。
    """

    metadata = _metadata(
        "ts_crossing_acceleration",
        "穿越 bar 的 z 二阶差分 / 滚动 std（仅穿越处非零）。",
        ["x", "y", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        return frame_like(
            x,
            _crossing_acceleration_series(x.to_numpy(dtype=float), y.to_numpy(dtype=float), w),
        )


_NEW_CANONICALS = (
    "ts_crossing_speed",
    "ts_crossing_acceleration",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
