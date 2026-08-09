# -*- coding: utf-8 -*-
"""Crossing-event quality operators (2026-08 geometry/math expansion).

A *crossing* of ``x`` through ``y`` (``z = x - y``) is a categorical event, but
not every crossing is equal: some slice through quietly, some explode through.
These operators score the *geometry* of a crossing bar:

* ``ts_crossing_speed``        — signed jump size across the crossing bar,
  normalised by the rolling volatility of ``z``.  Up-cross: +|dz|/scale_z;
  down-cross: -|dz|/scale_z; genuine non-crossing bars are 0.
* ``ts_crossing_acceleration`` — second difference of ``z`` on crossing bars only
  (``a_t = (dz_t - dz_{t-1})/scale_z``), 0 elsewhere.

Shared kernel — ``scale_z`` is the rolling (population) standard deviation of
``z`` over the trailing window (at least two finite values).  Crossing detection
is exact: up-cross requires ``z_{t-1} <= 0 < z_t`` and down-cross
``z_{t-1} >= 0 > z_t``.

Fail-closed rules (2026-08 round-3): no ``+ eps`` floor on the denominator — a
crossing bar whose volatility scale is non-finite or ``<= 0`` (sample too small
/ zero spread) emits NaN, never a fabricated number; a row whose *current*
``z`` is NaN emits NaN, never a stale 0/trailing value.  A finite non-crossing
bar emits 0.0 (there was genuinely no crossing event).

All operators are trailing-window per-column, prefix-causal, deterministic,
NaN-safe and reject invalid parameters.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge


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
    for t in range(rows):
        for c in range(cols):
            # Fail closed on a NaN CURRENT row (no stale 0/trailing value).
            if not np.isfinite(z[t, c]):
                out[t, c] = np.nan
                continue
            if t == 0:
                continue  # first bar: no previous z -> no crossing -> 0.0
            f = flags[t, c]
            if f == 0.0:
                continue  # genuine non-crossing finite bar -> 0.0
            # Sample-size aware, no EPS floor: the volatility scale must be
            # finite and strictly positive, otherwise fail closed to NaN.
            if not np.isfinite(scale[t, c]) or scale[t, c] <= 0.0:
                out[t, c] = np.nan
                continue
            if not np.isfinite(dz[t, c]):
                out[t, c] = np.nan
                continue
            out[t, c] = (abs(dz[t, c]) / scale[t, c]) * f
    return out


def _crossing_acceleration_series(x2d: np.ndarray, y2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    z, dz, scale = _crossing_parts(x2d, y2d, window)
    flags = _crossing_flags(z)
    out = np.zeros((rows, cols), dtype=float)
    for t in range(rows):
        for c in range(cols):
            # Fail closed on a NaN CURRENT row (no stale 0/trailing value).
            if not np.isfinite(z[t, c]):
                out[t, c] = np.nan
                continue
            if t < 2:
                continue  # need two diffs -> rows 0,1 -> 0.0
            f = flags[t, c]
            if f == 0.0:
                continue  # genuine non-crossing finite bar -> 0.0
            # Sample-size aware, no EPS floor: fail closed when the volatility
            # scale is not finite/positive.
            if not np.isfinite(scale[t, c]) or scale[t, c] <= 0.0:
                out[t, c] = np.nan
                continue
            if not (np.isfinite(dz[t, c]) and np.isfinite(dz[t - 1, c])):
                out[t, c] = np.nan
                continue
            a = (dz[t, c] - dz[t - 1, c]) / scale[t, c]
            if np.isfinite(a):
                out[t, c] = a
            else:
                out[t, c] = np.nan
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

    非穿越 bar 输出 0；当前行 NaN 或波动率尺度不可靠(<=0)时输出 NaN(fail
    closed，无 EPS 分母)。高 |值| → 价格以显著动量刺穿均线/阈值。P1。
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

    当前行 NaN 或波动率尺度不可靠(<=0)时输出 NaN(fail closed，无 EPS 分母)。
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
