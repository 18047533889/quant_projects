# -*- coding: utf-8 -*-
"""Dynamic cross-sectional KNN peer geometry (2026-08 V3, P1).

Trading "peers" are often not the same industry — high-turnover / high-vol /
small-cap / strong-momentum names co-move across industries.  These operators
build, every date, a KNN graph on rank-standardised style features (2..4) and
answer:

* ``cs_knn_peer_mean_ex_self``   — mean of a target over the k most similar
  names today (uniform weights; a residual recipe uses ``target - peer_mean``).
* ``cs_knn_neighbor_retention``  — Jaccard overlap between today's neighbor set
  and the set ``lag`` days ago: a stable style cluster vs rapid style drift.

The peer group is endogenously re-derived every day (unlike fixed relation /
industry groups).  Rank-transform makes the L2 metric scale-free; ties are
broken deterministically by stable argsort.  Deterministic and prefix-causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="cross_sectional",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "cross_sectional", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _avg_tie_ranks(vals: np.ndarray) -> np.ndarray:
    """Average tie ranks of ``vals`` (0-based).

    A double stable argsort gives tied values *distinct* ranks in
    column-arrival order, silently making the rank transform depend on the
    stock-column ordering (review R4-13).  Average ranks canonicalise ties.
    """
    order = np.argsort(vals, kind="mergesort")
    s = vals[order]
    ranks = np.empty(len(vals), dtype=float)
    n = len(vals)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def _rank_features(feats: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Date t feature matrix (n,d) -> rank-standardised U + per-stock validity."""
    n, d = feats[t].shape
    U = np.full((n, d), np.nan, dtype=float)
    for j in range(d):
        col = feats[t, :, j]
        fin = np.isfinite(col)
        m = int(fin.sum())
        if m < 2:
            continue
        ranks = _avg_tie_ranks(col[fin])
        U[fin, j] = (ranks + 0.5) / m
    valid = np.all(np.isfinite(U), axis=1)
    return U, valid


def _neighbors(U: np.ndarray, valid: np.ndarray, k: int, i: int) -> np.ndarray:
    dist = np.sqrt(np.sum((U - U[i]) ** 2, axis=1))
    dist = np.where(valid, dist, np.inf)
    dist[i] = np.inf
    count = int(valid.sum())
    k = max(1, int(k))
    if count - 1 < k:
        # Fail-closed (audit F03): never silently average fewer than k genuine
        # neighbors and call it kNN-k.
        return np.array([], dtype=int)
    finite = np.flatnonzero(valid)
    finite = finite[finite != i]
    ds = dist[finite]
    order = np.argsort(ds, kind="stable")
    kth = ds[order[k - 1]]
    # Tie-inclusive selection (audit F01 / review R4-83): the neighbour set is
    # the *kth-distance radius* — every name with distance <= the k-th distance
    # enters, so a tie at the boundary is never broken by column position
    # (permuting stock columns leaves the neighbour identity unchanged).  The
    # returned set therefore contains >= k names when boundary ties exist.
    return finite[ds <= kth]


def _peer_mean_series(target: np.ndarray, feats: np.ndarray, k: int) -> np.ndarray:
    rows, n, d = feats.shape
    out = np.full((rows, n), np.nan, dtype=float)
    for t in range(rows):
        U, valid = _rank_features(feats, t)
        for i in range(n):
            if not valid[i]:
                continue
            if not np.isfinite(target[t, i]):
                continue  # R4-82: target itself missing -> cannot judge peers
            nbrs = _neighbors(U, valid, k, i)
            if nbrs.size < k:
                continue
            vals = target[t, nbrs]
            vals = vals[np.isfinite(vals)]
            if vals.size < k:
                continue  # R4-82: never call a <k-peer mean a k-NN mean
            out[t, i] = float(np.mean(vals))
    return out


@register_operator(
    name="cs_knn_peer_mean_ex_self",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_knn_peer_mean_ex_self",
    source="dynamic_knn",
)
class CsKnnPeerMeanExSelf(SeriesOperator):
    """今天和我交易风格最相似的一群股票在做什么（k-NN 同行均值）。

    特征（市值/换手/波动/动量等 2..4 个）逐日截面秩标准化后取 L2 距离，最近
    ``k`` 个同行（排除自己）的 ``target`` 均匀均值。``target - peer_mean``
    可作 Recipe：动态同行残差。同行组每天内生改变，区别于固定行业/relation。
    P1。
    """

    metadata = _metadata(
        "cs_knn_peer_mean_ex_self",
        "动态 k-NN 同行均值（风格相似股，排除自身）。",
        ["target", "f1", "f2", "f3", "k"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(
        self, target: pd.DataFrame, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, k: int = 5, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(target, _peer_mean_series(target.to_numpy(dtype=float), feats, k))


def _retention_series(feats: np.ndarray, k: int, lag: int) -> np.ndarray:
    rows, n, d = feats.shape
    out = np.full((rows, n), np.nan, dtype=float)
    neighbor_sets: list[list[np.ndarray]] = []
    for t in range(rows):
        U, valid = _rank_features(feats, t)
        sets: list[np.ndarray] = []
        for i in range(n):
            if not valid[i]:
                sets.append(np.array([], dtype=int))
            else:
                sets.append(_neighbors(U, valid, k, i))
        neighbor_sets.append(sets)
        if t < lag:
            continue
        prev_sets = neighbor_sets[t - lag]
        for i in range(n):
            cur = sets[i]
            prev = prev_sets[i]
            if cur.size == 0 or prev.size == 0:
                continue
            inter = float(np.intersect1d(cur, prev).size)
            union = float(np.union1d(cur, prev).size)
            if union <= 0.0:
                continue
            out[t, i] = float(inter / union)
    return out


@register_operator(
    name="cs_knn_neighbor_retention",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_knn_neighbor_retention",
    source="dynamic_knn",
)
class CsKnnNeighborRetention(SeriesOperator):
    """KNN 同行集合的时间保持率（Jaccard，今天 vs lag 天前）。

    高 = 长期属于稳定 style cluster；低 = 正在快速切换交易风格/资金群体
    （风格漂移 / 主题切换 / 筹码属性变化）。与 ``cs_knn_peer_mean_ex_self``
    共享同一 KNN 图。P1。
    """

    metadata = _metadata(
        "cs_knn_neighbor_retention",
        "KNN 同行集合保持率 Jaccard（今天 vs lag 天前）。",
        ["f1", "f2", "f3", "k", "lag"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, k: int = 5, lag: int = 20, **_: Any
    ) -> pd.DataFrame:
        lg = max(1, int(lag))
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(f1, _retention_series(feats, k, lg))


def _dirichlet_energy_series(target: np.ndarray, feats: np.ndarray, k: int) -> np.ndarray:
    """Per-node Dirichlet energy over the style-kNN graph: mean squared gap of
    the *target* to its k most style-similar names.

    Low = similar names actually move together (factor smooth on the style
    graph); high = style similarity has decoupled from realized behaviour
    (regime divergence).  Zero/undefined neighbour sets fail closed to NaN."""
    rows, n, d = feats.shape
    out = np.full((rows, n), np.nan, dtype=float)
    for t in range(rows):
        U, valid = _rank_features(feats, t)
        for i in range(n):
            if not valid[i]:
                continue
            nbrs = _neighbors(U, valid, k, i)
            if nbrs.size < k:
                continue
            x_i = target[t, i]
            if not np.isfinite(x_i):
                continue
            vals = target[t, nbrs]
            vals = vals[np.isfinite(vals)]
            if vals.size < k:
                continue  # R4-82: never call a <k-peer energy a k-NN energy
            out[t, i] = float(np.mean((vals - x_i) ** 2))
    return out


@register_operator(
    name="cs_knn_graph_dirichlet_energy",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_knn_graph_dirichlet_energy",
    source="dynamic_knn",
)
class CsKnnGraphDirichletEnergy(SeriesOperator):
    """风格 k-NN 图上的目标 Dirichlet 能量 ``mean_{j∈NN(i)} (x_i - x_j)²``。

    每日期截面：特征（风格）图用秩标准化 L2 建 kNN，目标在图上不平滑程度 =
    每个节点相对其同行的均方差。低 = 相似股票确实一起走（因子在图上平滑）；
    高 = 风格相似与实际行为发生解耦（regime 切换 / 分化）。对 regime 检测和
    "同行 residual" 配方都有用。PIT 安全（当天截面，无前视）。
    """

    metadata = _metadata(
        "cs_knn_graph_dirichlet_energy",
        "目标在风格 kNN 图上的 Dirichlet energy（低=平滑）。",
        ["target", "f1", "f2", "f3", "k"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(
        self, target: pd.DataFrame, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, k: int = 5, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(target, _dirichlet_energy_series(target.to_numpy(dtype=float), feats, k))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "cs_knn_peer_mean_ex_self",
            "cs_knn_neighbor_retention",
            "cs_knn_graph_dirichlet_energy",
        }
    )


_register_surface()
