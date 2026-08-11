# -*- coding: utf-8 -*-
"""Dynamic cross-sectional KNN peer geometry (2026-08 V3, P1).

Trading "peers" are often not the same industry — high-turnover / high-vol /
small-cap / strong-momentum names co-move across industries.  These operators
build, every date, a KNN graph on rank-standardised style features (exactly 3:
``f1``/``f2``/``f3``) and answer:

* ``cs_knn_peer_mean_ex_self``   — mean of a target over the k most similar
  names today (uniform weights; a residual recipe uses ``target - peer_mean``).
* ``cs_knn_neighbor_retention``  — Jaccard overlap between today's neighbor set
  and the set ``lag`` days ago: a stable style cluster vs rapid style drift.

The peer group is endogenously re-derived every day (unlike fixed relation /
industry groups).  Rank-transform makes the L2 metric scale-free; ties are
broken deterministically by stable argsort.  Deterministic and prefix-causal.

SameTimeCrossSection semantics (audit M-110 / M-111)
----------------------------------------------------
These operators are date-``t`` **same-time** peer cross-sections, NOT
fit-through-``t-1`` models: the graph is built on date-``t`` features and the
peer target values read are date-``t`` (``as_of=0`` same time slice;
``self_excluded=True``; ``peer_feature_available=0``; ``peer_target_available=0``).
The reconciler's timing contract for the KNN family is
``fit_cutoff_offset=0`` / ``SAME_TIME_CROSS_SECTIONAL``.  Consequences that must
be documented, not silently assumed:

* ``cs_knn_peer_mean_ex_self`` uses same-day peer ``target_t``.  A same-day
  target is only known after the close — the factor cannot be backtested as a
  same-day VWAP signal; a pre-open variant needs a prior-label (``target_{t-1}``)
  recipe.
* The as-of universe / tradable mask is NOT invented as a parameter here: the
  operator operates on the given panel.  Production pipelines must apply the
  universe/tradable membership BEFORE calling these operators and the semantic
  identity of any derived factor must include universe membership (M-116).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.closure.strict_scalar import strict_int, strict_float
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    param_specs: dict | None = None,
) -> OperatorMetadata:
    # R11 §37-D unit-algebra honesty: algebraic units (``same_as:`` /
    # ``unit(...)`` / ``dimensionless``) are propagated to ``output_unit`` so
    # the catalog / typed search see the real output dimension instead of an
    # opaque ``ratio`` tag (mirrors conditional_ext / group_ext / relation).
    output_unit = unit if (unit.startswith("same_as:") or unit.startswith("unit(") or unit == "dimensionless") else None
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
        output_unit=output_unit,
        # M-115: ``k`` / ``lag`` are estimator-resolution knobs of the KNN graph
        # (neighbour-count / lookback to a past graph), never freely-searchable
        # economic alphas.  Declared explicitly so certification does not have to
        # infer a search role from the name.
        param_specs=dict(param_specs) if param_specs else {},
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
    k = strict_int(k, "k", lower=1)  # A-5: k=0/-1 is a contract violation, never clamped
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

    恰好 3 个特征（``f1``/``f2``/``f3``：市值/换手/波动/动量等）逐日截面秩标准化
    后取 L2 距离，最近 ``k`` 个同行（排除自己）的 ``target`` 均匀均值。
    ``target - peer_mean`` 可作 Recipe：动态同行残差。同行组每天内生改变，区别于
    固定行业/relation。

    SameTimeCrossSection：同日截面（``as_of=0``）、排除自身（``self_excluded=True``）、
    同行特征与目标均取当日（``peer_feature_available=0``、``peer_target_available=0``）。
    同一交易日 target 仅收盘后可知——本因子不可回测同日 VWAP；如需盘前可用需
    prior-label 变体（``target_{t-1}``）。as-of universe/tradable mask 由调用方在面板
    上应用（算子不引入 universe 参数）。

    输出是 ``target`` 的无权均值，单位与 ``target`` 相同（``same_as:target``）——
    不是无量纲的 ratio；把它标成 ratio 会让 typed algebra 把同单位均值当作
    无量纲（typed-factor-composition 污染）。P1。
    """

    metadata = _metadata(
        "cs_knn_peer_mean_ex_self",
        "动态 k-NN 同行均值（风格相似股，排除自身；恰好 3 个特征 f1/f2/f3；"
        "SameTimeCrossSection：as_of=0 / self_excluded / peer_feature_available=0 / "
        "peer_target_available=0）。k 是"
        "最小邻居数/kth-distance radius（tie-inclusive，边界平局全收，可>k）。同一"
        "交易日 target 仅收盘后可知——不可回测同日 VWAP，盘前需 prior-label 变体。"
        "输出单位 same_as:target（同行 target 的均值，非 ratio）。",
        ["target", "f1", "f2", "f3", "k"],
        unit="same_as:target",
        cost=7,
        param_specs={
            "k": ParamSpec(
                dtype=int, min=1,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=5,
                searchable=False,
            ),
        },
    )

    def _calculate_series(
        self, target: pd.DataFrame, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, k: int = 5, **_: Any
    ) -> pd.DataFrame:
        # M-113 / M-240: strict operator-boundary validation — never silently
        # clamp / truncate k (rejects <1, fractional, NaN, Inf, bool).
        kk = strict_int(k, "k", lower=1)
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(target, _peer_mean_series(target.to_numpy(dtype=float), feats, kk))


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

    SameTimeCrossSection：每一天的集合都是在当天特征上内生重建的
    （``as_of=0``、``self_excluded=True``）；比较的是今天 vs ``lag`` 天前
    的同日截面集合。恰好 3 个特征（``f1``/``f2``/``f3``）。
    """

    metadata = _metadata(
        "cs_knn_neighbor_retention",
        "KNN 同行集合保持率 Jaccard（今天 vs lag 天前；kth-radius tie-inclusive，"
        "k 是最小邻居数，边界平局全收）。SameTimeCrossSection：as_of=0 / "
        "self_excluded。恰好 3 个特征 f1/f2/f3。",
        ["f1", "f2", "f3", "k", "lag"],
        unit="ratio",
        cost=7,
        param_specs={
            "k": ParamSpec(
                dtype=int, min=1,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=5,
                searchable=False,
            ),
            "lag": ParamSpec(
                dtype=int, min=1,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=20,
                searchable=False,
            ),
        },
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, k: int = 5, lag: int = 20, **_: Any
    ) -> pd.DataFrame:
        # M-113 / M-240: strict operator-boundary validation — ``lag`` must be a
        # genuine positive integer.  The old ``max(1, int(lag))`` silently
        # clamped ``lag=0``/fractional/``True`` to a valid window, manufacturing
        # a false search space; reject instead of clamp.
        kk = strict_int(k, "k", lower=1)
        lg = strict_int(lag, "lag", lower=1)
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(f1, _retention_series(feats, kk, lg))


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

    SameTimeCrossSection：同日截面（``as_of=0``）、排除自身（``self_excluded=True``）、
    同行特征与目标均取当日。恰好 3 个特征（``f1``/``f2``/``f3``）。

    输出是 ``(target 差)²`` 的均值，维度为 ``unit(target)²``。tag 代数不直接
    表达上标平方，使用等价的乘积形式 ``unit(target)*unit(target)``（与 relation
    加权输出的 ``unit(value)*unit(weight)`` 同一套乘积语法）；这不是 ratio。
    """

    metadata = _metadata(
        "cs_knn_graph_dirichlet_energy",
        "目标在风格 kNN 图上的 Dirichlet energy（低=平滑；SameTimeCrossSection："
        "as_of=0 / self_excluded；k 是最小邻居数/kth-distance radius，tie-inclusive）。"
        "恰好 3 个特征 f1/f2/f3。"
        "输出单位 unit(target)*unit(target)（即 unit(target)²，非 ratio）。",
        ["target", "f1", "f2", "f3", "k"],
        unit="unit(target)*unit(target)",
        cost=7,
        param_specs={
            "k": ParamSpec(
                dtype=int, min=1,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=5,
                searchable=False,
            ),
        },
    )

    def _calculate_series(
        self, target: pd.DataFrame, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, k: int = 5, **_: Any
    ) -> pd.DataFrame:
        # M-113 / M-240: strict operator-boundary validation of ``k``.
        kk = strict_int(k, "k", lower=1)
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(target, _dirichlet_energy_series(target.to_numpy(dtype=float), feats, kk))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "cs_knn_peer_mean_ex_self",
            "cs_knn_neighbor_retention",
            "cs_knn_graph_dirichlet_energy",
        })


_register_surface()
