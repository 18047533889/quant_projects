# -*- coding: utf-8 -*-
"""Alpha-language cross-sectional locality and group ex-self operators (2026-08).

Locality family (rank-based, evaluated independently per trading row):

* neighbor gap: symmetric value-space gap between the k-th higher and k-th
  lower neighbor (one-sided at the edge of the cross-section);
* isolation: geometric mean of the up / down neighbor gaps (how far from peers);
* local curvature: up-gap vs down-gap imbalance ([-1, 1]).

Group ex-self family (leave-one-out within a group label):

* ex-self std / mad / quantile of the peer set.

Relation-weighted dispersion: weighted peer dispersion with self excluded and
weights row-normalised within a group (engine-native ``(x, weight, group)`` form).

All operators are causal, evaluated per trading row only (no forward
information).  Edge cross-sections and degenerate groups return NaN, never Inf.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common.daily_panel import _aligned
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="cross_section",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "cross_section", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


def _row_local_gaps(x_row: np.ndarray, g_row: Any, k: int, finite: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-stock (up-gap, down-gap, scale) within a single cross-sectional row.

    up-gap   = value of the k-th higher neighbour - own value  (>= 0)
    down-gap = own value - value of the k-th lower neighbour   (>= 0)
    scale    = MAD of the local cross-section (group or whole row).
    """
    n = len(x_row)
    g_plus = np.full(n, np.nan, dtype=float)
    g_minus = np.full(n, np.nan, dtype=float)
    scale = np.full(n, np.nan, dtype=float)
    labels = [None] if g_row is None else pd.unique(g_row)
    for label in labels:
        idx = finite if label is None else ((g_row == label) & finite)
        positions = np.flatnonzero(idx)
        m = positions.size
        if m == 0:
            continue
        vals = x_row[positions]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) if m >= 2 else 0.0
        order = np.argsort(vals)
        for rank, stock_pos in enumerate(order):
            stock = positions[stock_pos]
            if rank + k < m:
                g_plus[stock] = float(vals[order[rank + k]] - vals[stock_pos])
            if rank - k >= 0:
                g_minus[stock] = float(vals[stock_pos] - vals[order[rank - k]])
            scale[stock] = mad
    return g_plus, g_minus, scale


def _one_sided(g_plus: np.ndarray, g_minus: np.ndarray) -> tuple[np.ndarray, bool]:
    gp_ok = np.isfinite(g_plus)
    gm_ok = np.isfinite(g_minus)
    val = np.where(gp_ok & gm_ok, g_plus + g_minus,
                   np.where(gp_ok, g_plus,
                            np.where(gm_ok, g_minus, np.nan)))
    return val, bool(np.any(gp_ok | gm_ok))


def _symmetric_gap(gp: np.ndarray, gm: np.ndarray) -> np.ndarray:
    """Symmetric k-th-neighbour value gap (one-sided at the cross-section edge)."""
    out = np.full(gp.shape, np.nan, dtype=float)
    both = np.isfinite(gp) & np.isfinite(gm)
    only_up = np.isfinite(gp) & ~np.isfinite(gm)
    only_down = ~np.isfinite(gp) & np.isfinite(gm)
    out[both] = gp[both] + gm[both]
    out[only_up] = gp[only_up]
    out[only_down] = gm[only_down]
    return out


@register_operator(
    name="cs_neighbor_gap",
    category="cross_section",
    business_category="cross_section",
    canonical="cs_neighbor_gap",
    source="alpha_language_cross",
)
class CsNeighborGap(SeriesOperator):
    """邻域缺口(恒非负宽度): (x_{r+k} - x_{r-k}) / (横截面 MAD + eps), 按日 rank。

    边缘(仅一侧有邻居)用单侧缺口; 双侧都无 -> NaN。输出恒 >= 0: 它度量 k 阶上
    下邻居之间的局部数值间距宽度, **不是**带符号的方向性缺口。上下方向失衡由
    ``cs_local_curvature`` (G+ - G-)/(G+ + G-) 捕捉。
    """

    metadata = _metadata(
        "cs_neighbor_gap",
        "k 阶上下邻居间的局部数值间距宽度(恒非负, 按截面 MAD 归一)。",
        ["x", "k", "group"],
        domain="cross_section",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, k: int = 5, group: Any = None, **_: Any) -> pd.DataFrame:
        kk = int(k)
        if kk < 1:
            raise ValueError("k must be >= 1")
        if group is not None:
            x, group = _aligned(x, group)
            gv = group.to_numpy()
        else:
            gv = None
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        finite = np.isfinite(xv)
        for row in range(rows):
            gp, gm, scale = _row_local_gaps(xv[row], None if gv is None else gv[row], kk, finite[row])
            gap = _symmetric_gap(gp, gm)
            for j in range(cols):
                if not np.isfinite(gap[j]):
                    continue
                # P0-010: a degenerate local scale (MAD ~ 0, e.g. a limit-up
                # board or a discrete fundamental field) is mathematically
                # undefined — return NaN, never a huge ``gap/eps`` alpha.
                if not np.isfinite(scale[j]) or scale[j] <= _EPS:
                    continue
                out[row, j] = gap[j] / scale[j]
        return frame_like(x, out)


@register_operator(
    name="cs_local_density",
    category="cross_section",
    business_category="cross_section",
    canonical="cs_local_density",
    source="alpha_language_cross",
)
class CsLocalDensity(SeriesOperator):
    """局部密度: 2k / ((x_{r+k} - x_{r-k}) / (截面 MAD + eps) + eps) + eps。

    高 = 附近很多股票挤在一起; 低 = 附近稀疏。单特征版, k 为位置参数
    (既有 ``cs_local_density_score`` 是多特征 + k 关键字签名, 不做别名)。
    """

    metadata = _metadata(
        "cs_local_density",
        "k 邻域对称缺口倒数(局部密度)。",
        ["x", "k", "group"],
        domain="cross_section",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, k: int = 5, group: Any = None, **_: Any) -> pd.DataFrame:
        kk = int(k)
        if kk < 1:
            raise ValueError("k must be >= 1")
        if group is not None:
            x, group = _aligned(x, group)
            gv = group.to_numpy()
        else:
            gv = None
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        finite = np.isfinite(xv)
        for row in range(rows):
            gp, gm, scale = _row_local_gaps(xv[row], None if gv is None else gv[row], kk, finite[row])
            gap = _symmetric_gap(gp, gm)
            for j in range(cols):
                if not np.isfinite(gap[j]):
                    continue
                # P0-009: degenerate cross-section (MAD ~ 0 / local gap ~ 0,
                # e.g. an all-equal limit board) is mathematically undefined.
                # Without this guard ``2k / (0/eps + eps) ~ 2k/eps`` would
                # surface as a huge finite "density" alpha.
                if not np.isfinite(scale[j]) or scale[j] <= _EPS:
                    continue
                if not np.isfinite(gap[j]) or abs(gap[j]) <= _EPS:
                    continue
                density = 2.0 * kk / (gap[j] / scale[j] + _EPS) + _EPS
                out[row, j] = density
        return frame_like(x, out)


@register_operator(
    name="cs_isolation",
    category="cross_section",
    business_category="cross_section",
    canonical="cs_isolation",
    source="alpha_language_cross",
)
class CsIsolation(SeriesOperator):
    """孤立度: sqrt(G+ * G-), 双侧都有邻居时; 单侧用该侧缺口; 双侧无 -> NaN。

    高 = 与上下邻居在数值空间都远(孤立)。
    """

    metadata = _metadata(
        "cs_isolation",
        "上下邻居缺口几何均值。",
        ["x", "k", "group"],
        domain="cross_section",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, k: int = 5, group: Any = None, **_: Any) -> pd.DataFrame:
        kk = int(k)
        if kk < 1:
            raise ValueError("k must be >= 1")
        if group is not None:
            x, group = _aligned(x, group)
            gv = group.to_numpy()
        else:
            gv = None
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        finite = np.isfinite(xv)
        for row in range(rows):
            gp, gm, _ = _row_local_gaps(xv[row], None if gv is None else gv[row], kk, finite[row])
            gp_ok = np.isfinite(gp)
            gm_ok = np.isfinite(gm)
            both = gp_ok & gm_ok
            out[row, both] = np.sqrt(gp[both] * gm[both])
            only_up = gp_ok & ~gm_ok
            out[row, only_up] = gp[only_up]
            only_down = ~gp_ok & gm_ok
            out[row, only_down] = gm[only_down]
        return frame_like(x, out)


@register_operator(
    name="cs_local_curvature",
    category="cross_section",
    business_category="cross_section",
    canonical="cs_local_curvature",
    source="alpha_language_cross",
)
class CsLocalCurvature(SeriesOperator):
    """局部曲率: (G+ - G-) / (G+ + G- + eps), 范围 [-1,1]。

    +1 = 向上拉伸, -1 = 向下拉伸, 0 = 左右对称; 单侧(边缘) -> NaN。
    """

    metadata = _metadata(
        "cs_local_curvature",
        "上下邻居缺口失衡。",
        ["x", "k", "group"],
        domain="cross_section",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, k: int = 5, group: Any = None, **_: Any) -> pd.DataFrame:
        kk = int(k)
        if kk < 1:
            raise ValueError("k must be >= 1")
        if group is not None:
            x, group = _aligned(x, group)
            gv = group.to_numpy()
        else:
            gv = None
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        finite = np.isfinite(xv)
        for row in range(rows):
            gp, gm, _ = _row_local_gaps(xv[row], None if gv is None else gv[row], kk, finite[row])
            both = np.isfinite(gp) & np.isfinite(gm)
            num = gp[both] - gm[both]
            den = gp[both] + gm[both]
            out[row, both] = np.where(den < _EPS, np.nan, num / (den + _EPS))
        return frame_like(x, out)


def _group_peers_indices(g_row: np.ndarray, label: Any, finite: np.ndarray) -> np.ndarray:
    return np.flatnonzero((g_row == label) & finite)


def _group_ex_self_std_row(x_row: np.ndarray, g_row: np.ndarray, finite: np.ndarray, min_peers: int) -> np.ndarray:
    out = np.full(x_row.shape, np.nan, dtype=float)
    for label in pd.unique(g_row):
        peers = _group_peers_indices(g_row, label, finite)
        if peers.size <= min_peers:
            continue
        for j in peers:
            others = peers[peers != j]
            if others.size < min_peers:
                continue
            out[j] = float(np.std(x_row[others]))
    return out


@register_operator(
    name="group_ex_self_std",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="group_ex_self_std",
    source="alpha_language_cross",
)
class GroupExSelfStd(SeriesOperator):
    """组内除自身外其余成员的 std(leave-one-out peer dispersion)。"""

    metadata = _metadata(
        "group_ex_self_std",
        "组内除自身外其余成员的 std。",
        ["x", "group", "min_peers"],
        domain="price_volume",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame, min_peers: int = 2, **_: Any) -> pd.DataFrame:
        x, group = _aligned(x, group)
        xv = x.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        finite = np.isfinite(xv)
        for row in range(rows):
            out[row] = _group_ex_self_std_row(xv[row], gv[row], finite[row], max(2, int(min_peers)))
        return frame_like(x, out)


@register_operator(
    name="group_ex_self_mad",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="group_ex_self_mad",
    source="alpha_language_cross",
)
class GroupExSelfMad(SeriesOperator):
    """组内除自身外其余成员的 MAD(比 std 更稳健的 peer dispersion)。"""

    metadata = _metadata(
        "group_ex_self_mad",
        "组内除自身外其余成员的 MAD。",
        ["x", "group", "min_peers"],
        domain="price_volume",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame, min_peers: int = 3, **_: Any) -> pd.DataFrame:
        x, group = _aligned(x, group)
        xv = x.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        finite = np.isfinite(xv)
        minp = max(3, int(min_peers))
        for row in range(rows):
            for label in pd.unique(gv[row]):
                peers = _group_peers_indices(gv[row], label, finite[row])
                if peers.size <= minp:
                    continue
                for j in peers:
                    others = peers[peers != j]
                    if others.size < minp:
                        continue
                    vals = xv[row, others]
                    med = float(np.median(vals))
                    out[row, j] = float(np.median(np.abs(vals - med)))
        return frame_like(x, out)


@register_operator(
    name="group_ex_self_quantile",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="group_ex_self_quantile",
    source="alpha_language_cross",
)
class GroupExSelfQuantile(SeriesOperator):
    """组内除自身外其余成员的 q 分位数。"""

    metadata = _metadata(
        "group_ex_self_quantile",
        "组内除自身外其余成员的 q 分位数。",
        ["x", "group", "q", "min_peers"],
        domain="price_volume",
        unit="level",
    )

    def _calculate_series(
        self, x: pd.DataFrame, group: pd.DataFrame, q: float = 0.5, min_peers: int = 2, **_: Any
    ) -> pd.DataFrame:
        qq = float(q)
        if not 0.0 < qq < 1.0:
            raise ValueError("q must be in (0, 1)")
        x, group = _aligned(x, group)
        xv = x.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        finite = np.isfinite(xv)
        minp = max(2, int(min_peers))
        for row in range(rows):
            for label in pd.unique(gv[row]):
                peers = _group_peers_indices(gv[row], label, finite[row])
                if peers.size <= minp:
                    continue
                for j in peers:
                    others = peers[peers != j]
                    if others.size < minp:
                        continue
                    out[row, j] = float(np.quantile(xv[row, others], qq))
        return frame_like(x, out)


@register_operator(
    name="relation_weighted_std_ex_self",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="relation_weighted_std_ex_self",
    source="alpha_language_cross",
)
class RelationWeightedStdExSelf(SeriesOperator):
    """组内排除自身、按节点权重加权的 peer std(非严格 relation-weighted)。

    mu_i = Σ_{j≠i} w_j x_j / Σ_{j≠i} w_j,
    sigma_i = sqrt(Σ_{j≠i} w_j (x_j - mu_i)² / Σ_{j≠i} w_j)。
    输入只有单 weight 列(各 peer 自身权重), **不是** focal×peer 的 w_ij relation
    weight; 严格说是 group 内按节点权重加权的分散度(leave-one-out)。
    组内权重行归一化, 自身排除, weight 要求非负。
    """

    metadata = _metadata(
        "relation_weighted_std_ex_self",
        "组内按各 peer 自身权重加权的排除自身 std(单 weight 输入, 非 w_ij relation weight)。",
        ["x", "weight", "group", "min_peers"],
        domain="price_volume",
        unit="level",
    )

    def _calculate_series(
        self, x: pd.DataFrame, weight: pd.DataFrame, group: pd.DataFrame, min_peers: int = 2, **_: Any
    ) -> pd.DataFrame:
        x, weight, group = _aligned(x, weight, group)
        xv = x.to_numpy(dtype=float)
        wv = weight.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        minp = max(2, int(min_peers))
        for row in range(rows):
            g_row = gv[row]
            for label in pd.unique(g_row):
                peers = _group_peers_indices(g_row, label, np.isfinite(xv[row]) & np.isfinite(wv[row]))
                if peers.size <= minp:
                    continue
                for j in peers:
                    others = peers[peers != j]
                    w = wv[row, others]
                    if np.any(w < 0.0):
                        continue
                    total_w = float(w.sum())
                    if total_w <= 0.0:
                        continue
                    w = w / total_w
                    vals = xv[row, others]
                    mu = float(np.sum(w * vals))
                    var = float(np.sum(w * (vals - mu) ** 2))
                    out[row, j] = float(np.sqrt(max(var, 0.0)))
        return frame_like(x, out)


_NEW_CANONICALS = (
    "cs_neighbor_gap",
    "cs_local_density",
    "cs_isolation",
    "cs_local_curvature",
    "group_ex_self_std",
    "group_ex_self_mad",
    "group_ex_self_quantile",
    "relation_weighted_std_ex_self",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
