# -*- coding: utf-8 -*-
"""2D joint-trajectory path geometry operators (2026-08 geometry/math expansion).

Treat the pair ``(f1, f2)`` as a 2D joint trajectory over a trailing window and
characterise the *shape* of the drawn path:

* ``ts_vector_path_efficiency``       — net displacement / total path length
  (how directional the 2D path is).  Uses rolling-standardized coordinates.
* ``ts_vector_turning_coherence``     — mean cosine between consecutive velocity
  vectors (how smoothly the direction evolves).  Uses rolling-standardized
  coordinates.
* ``ts_vector_path_curvature``        — robust median of the 2D curvature
  ``|x'y''-y'x''|/((x'^2+y'^2)^(3/2))`` built from first-difference derivatives.
  Uses raw coordinates.
* ``ts_vector_self_intersection_rate``— fraction of segment pairs that properly
  intersect (2D segment-intersection test).  Uses raw coordinates.

All operators are trailing-window, prefix-causal and deterministic.  Only
same-position finite ``(f1, f2)`` pairs are used; degenerate windows (too few
points, zero rolling std) emit NaN (fail-closed).  Invalid parameters raise
``ValueError``.
"""
from __future__ import annotations

from typing import Any, Callable

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
        category="vector_path",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "vector_path", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:path_geometry",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# shared kernels (private to this module; 2D panel = TradeDate x Symbol)
# ---------------------------------------------------------------------------
def _standardize(f1: np.ndarray, f2: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Rolling-window z-standardization of each field (mean/std over the window)."""
    if f1.size < 2:
        return None
    s1, s2 = float(np.std(f1)), float(np.std(f2))
    if s1 <= _EPS or s2 <= _EPS:
        return None
    return (f1 - f1.mean()) / s1, (f2 - f2.mean()) / s2


def _path_efficiency(f1: np.ndarray, f2: np.ndarray) -> float:
    n = f1.size
    if n < 2:
        return np.nan
    z = _standardize(f1, f2)
    if z is None:
        return np.nan
    z1, z2 = z
    d = float(np.hypot(z1[-1] - z1[0], z2[-1] - z2[0]))
    length = 0.0
    for k in range(1, n):
        length += float(np.hypot(z1[k] - z1[k - 1], z2[k] - z2[k - 1]))
    return float(d / (length + _EPS))


def _turning_coherence(f1: np.ndarray, f2: np.ndarray) -> float:
    n = f1.size
    if n < 3:
        return np.nan
    z = _standardize(f1, f2)
    if z is None:
        return np.nan
    z1, z2 = z
    vx = np.diff(z1)
    vy = np.diff(z2)
    cosines = []
    for k in range(1, vx.size):
        n1 = float(np.hypot(vx[k], vy[k]))
        n2 = float(np.hypot(vx[k - 1], vy[k - 1]))
        if n1 <= _EPS or n2 <= _EPS:
            continue
        cosines.append(float((vx[k] * vx[k - 1] + vy[k] * vy[k - 1]) / (n1 * n2)))
    if not cosines:
        return np.nan
    return float(np.mean(cosines))


def _path_curvature(f1: np.ndarray, f2: np.ndarray) -> float:
    n = f1.size
    if n < 3:
        return np.nan
    dx = np.diff(f1)
    dy = np.diff(f2)
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    kappas = []
    for k in range(ddx.size):
        xp, yp = dx[k], dy[k]
        xpp, ypp = ddx[k], ddy[k]
        num = abs(xp * ypp - yp * xpp)
        denom = (xp * xp + yp * yp) ** 1.5 + _EPS
        kappas.append(num / denom)
    if not kappas:
        return np.nan
    return float(np.median(kappas))


def _orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def _segments_properly_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _self_intersection_rate(f1: np.ndarray, f2: np.ndarray) -> float:
    n = f1.size
    if n < 3:
        return np.nan
    pts = np.column_stack([f1, f2])
    m = n - 1  # number of segments
    if m < 2:
        return np.nan
    total = m * (m - 1) // 2
    inter = 0
    for i in range(m):
        p1, p2 = pts[i], pts[i + 1]
        for j in range(i + 1, m):
            p3, p4 = pts[j], pts[j + 1]
            if _segments_properly_intersect(p1, p2, p3, p4):
                inter += 1
    return float(inter) / float(total)


def _pair_series(a2d: np.ndarray, b2d: np.ndarray, window: int, kernel: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    rows, cols = a2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            f1, f2 = aligned_pairs(a2d[i0 : r + 1, c], b2d[i0 : r + 1, c])
            if f1.size < 2:
                continue
            out[r, c] = kernel(f1, f2)
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_vector_path_efficiency",
    category="vector_path",
    business_category="vector_path",
    canonical="ts_vector_path_efficiency",
    source="vector_path",
)
class TsVectorPathEfficiency(SeriesOperator):
    """2D 路径效率：窗口端点净位移 / 实际路径总长,∈[0,1]。

    高 → (f1,f2) 联合轨迹方向性一致；低 → 来回震荡。坐标按窗口滚动 std 标准化。P1。
    """

    metadata = _metadata(
        "ts_vector_path_efficiency",
        "2D 轨迹净位移 / 路径总长（方向效率）。",
        ["f1", "f2", "window"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, f1: pd.DataFrame, f2: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(f1, _pair_series(f1.to_numpy(dtype=float), f2.to_numpy(dtype=float), w, _path_efficiency))


@register_operator(
    name="ts_vector_turning_coherence",
    category="vector_path",
    business_category="vector_path",
    canonical="ts_vector_turning_coherence",
    source="vector_path",
)
class TsVectorTurningCoherence(SeriesOperator):
    """2D 转向一致性：相邻速度向量夹角的平均余弦。

    高 → 轨迹平滑沿同一方向推进；低/负 → 频繁急转弯。跳过退化零向量。P1。
    """

    metadata = _metadata(
        "ts_vector_turning_coherence",
        "相邻速度向量夹角平均余弦（转向平滑度）。",
        ["f1", "f2", "window"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, f1: pd.DataFrame, f2: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(f1, _pair_series(f1.to_numpy(dtype=float), f2.to_numpy(dtype=float), w, _turning_coherence))


@register_operator(
    name="ts_vector_path_curvature",
    category="vector_path",
    business_category="vector_path",
    canonical="ts_vector_path_curvature",
    source="vector_path",
)
class TsVectorPathCurvature(SeriesOperator):
    """2D 路径曲率：一阶差分导数下的 κ 的窗口稳健中位数。

    高 → 轨迹弯曲剧烈；低 → 接近直线。使用原始坐标。P2。
    """

    metadata = _metadata(
        "ts_vector_path_curvature",
        "2D 曲率 κ 的稳健中位数（一阶差分导数）。",
        ["f1", "f2", "window"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, f1: pd.DataFrame, f2: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(f1, _pair_series(f1.to_numpy(dtype=float), f2.to_numpy(dtype=float), w, _path_curvature))


@register_operator(
    name="ts_vector_self_intersection_rate",
    category="vector_path",
    business_category="vector_path",
    canonical="ts_vector_self_intersection_rate",
    source="vector_path",
)
class TsVectorSelfIntersectionRate(SeriesOperator):
    """2D 轨迹自交率：正确相交的线段对占全部线段对的比例,∈[0,1]。

    高 → 轨迹反复缠绕穿越自身（chop）；低 → 大致单调前进。使用原始坐标。P2。
    """

    metadata = _metadata(
        "ts_vector_self_intersection_rate",
        "轨迹线段对正确相交比例（自交率）。",
        ["f1", "f2", "window"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(self, f1: pd.DataFrame, f2: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(f1, _pair_series(f1.to_numpy(dtype=float), f2.to_numpy(dtype=float), w, _self_intersection_rate))


_NEW_CANONICALS = (
    "ts_vector_path_efficiency",
    "ts_vector_turning_coherence",
    "ts_vector_path_curvature",
    "ts_vector_self_intersection_rate",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
