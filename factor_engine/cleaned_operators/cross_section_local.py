# -*- coding: utf-8 -*-
"""Local non-linear cross-section geometry (2026-08 market language, P1/P2).

The next step after "mean of similar names": instead of asking what the peers
are doing, ask **what the normal target value should be in my region of feature
space**.

* ``cs_knn_local_linear_residual`` — residual of ``target`` against a local
  ridge regression of ``target ~ f1+f2+f3`` fit on the k style-neighbours,
  normalised by the neighbour-residual MAD (peer-relative mispricing).
* ``cs_knn_tangent_residual``     — distance of a name from the local tangent
  plane (top-2 PCA) of its neighbours' feature cloud (off-manifold names).
* ``cs_knn_local_gradient_norm``  — ||beta|| of that local regression (how
  sensitive the target is to the features *in that region*).
* ``cs_rank_copula_mi`` / ``cs_rank_copula_entropy`` — cross-sectional rank
  copula mutual information / entropy of two fields (Jeffreys-smoothed plug-in
  estimator, single definition — no Miller–Madow layer, R26-011).

Features are rank-standardised per day (scale-free L2); all kernels are
per-day cross-sections (prefix-causal, no future stocks/days), deterministic,
fail-closed to NaN on degenerate neighbourhoods.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.rolling_pack import frame_like, register_polars_udf

_EPS = 1e-12
_ALPHA = 0.5  # Jeffreys smoothing for copula histograms (deterministic).
# R11 round-3 P1-A (item 95): a top-2 tangent plane needs a genuine second
# principal direction.  When s2/s1 < this ratio the "plane" is one dominant
# direction padded with numerical noise -> fail closed to NaN.
_TANGENT_MIN_S2_S1 = 0.05

# R15-INC-171/172: declared local-fit support policy.  A local regression on
# ``d`` features + intercept is near-interpolation below
# ``max(floor, 5*(d+1))`` peers; the effective neighbourhood is ALWAYS
# ``max(min_neigh, k)`` (no silent k degradation).  Visible as a named constant
# so the DOF multiplier is a reviewed policy, not a magic number inside the
# kernels.
_MIN_NEIGH_FLOOR = 10
_MIN_NEIGH_DOF_MULTIPLIER = 5
# These cross-section-local operators ALWAYS take exactly 3 features (f1/f2/f3),
# so the DOF floor for their local regression is a FIXED value
# ``max(floor, 5*(3+1)) == 20``.  The searchable ``k`` range starts there —
# any k below the floor would silently map to the same effective neighbourhood
# (a dead-parameter region, R15-INC-263).
_KNN_MIN_K = max(_MIN_NEIGH_FLOOR, _MIN_NEIGH_DOF_MULTIPLIER * 4)


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
    param_specs: dict | None = None,
) -> OperatorMetadata:
    # R11 §37-D unit-algebra honesty: algebraic units (``same_as:`` /
    # ``unit(...)`` / ``dimensionless``) propagate to ``output_unit`` so typed
    # algebra / the catalog see the real output dimension instead of an opaque
    # tag (mirrors dynamic_knn / cross_section_ext).
    output_unit = unit if (unit.startswith("same_as:") or unit.startswith("unit(") or unit == "dimensionless") else None
    return OperatorMetadata(
        name=name,
        category="cross_sectional",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "cross_sectional", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
        output_unit=output_unit,
        param_specs=dict(param_specs) if param_specs else {},
    )


def _avg_tie_ranks(vals: np.ndarray) -> np.ndarray:
    """Average tie ranks of ``vals`` (0-based).  A double stable argsort gives
    tied values *distinct* ranks in column-arrival order, silently making the
    rank transform depend on the stock-column ordering (P1-42/45)."""
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
    """Date-t feature matrix (n,d) -> rank-standardised U + validity mask."""
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


def _neighbors(U: np.ndarray, valid: np.ndarray, k: int, i: int, peer_mask: np.ndarray | None = None) -> np.ndarray:
    """Tie-inclusive nearest neighbours of query ``i`` (R4-83).

    ``peer_mask`` restricts which stocks may serve as peers (feature-valid AND
    target-observed for the regression ops — R4-82 k-peer eligibility); for the
    tangent operator it is the feature-valid mask.  ``k`` nearest are selected;
    when several peers tie at the k-th distance **all** of them are included so
    the neighbourhood does not depend on the stock-column ordering (a plain
    ``argsort(...)[:k]`` would silently pick an arbitrary subset of a tied group
    and make the result column-order-dependent).  Self is always excluded.
    """
    mask = valid if peer_mask is None else peer_mask
    dist = np.sqrt(np.sum((U - U[i]) ** 2, axis=1))
    dist = np.where(mask, dist, np.inf)
    dist[i] = np.inf
    peer_count = int(mask.sum())
    if peer_count < 2:
        return np.array([], dtype=int)
    k_eff = min(max(1, int(k)), peer_count - 1)
    if k_eff < 1:
        return np.array([], dtype=int)
    kth = float(np.partition(dist, k_eff - 1)[k_eff - 1])
    if not np.isfinite(kth):
        return np.array([], dtype=int)
    return np.where((dist <= kth) & np.isfinite(dist))[0]


def _local_linear_series(target: np.ndarray, feats: np.ndarray, k: int, ridge: float) -> np.ndarray:
    rows, n, d = feats.shape
    out = np.full((rows, n), np.nan, dtype=float)
    # R11 round-3 P1-A (item 93): 3 features + intercept = 4 parameters; k≈4-5
    # almost interpolates and collapses the residual scale -> huge z-scores.
    # Enforce a minimum effective neighbourhood = max(10, 5*(d+1)); below the
    # DOF floor the fit is near-interpolation and emits NaN (fail-closed, never
    # a spurious z-score).
    # R15-INC-171/172: the neighbourhood target is the EXPLICIT
    # ``max(min_neigh, k)`` — a ``k`` smaller than the DOF floor is never a
    # silent degradation into a smaller fit; the kernel fits on the true
    # effective neighbourhood.  The DOF multiplier ``5`` is the declared
    # ``_MIN_NEIGH_DOF_MULTIPLIER`` support policy, visible in metadata.
    min_neigh = max(_MIN_NEIGH_FLOOR, _MIN_NEIGH_DOF_MULTIPLIER * (d + 1))
    target_k = max(min_neigh, int(k))
    for t in range(rows):
        U, valid = _rank_features(feats, t)
        y_t = target[t]
        target_fin = np.isfinite(y_t)
        # R4-82: peers for the local regression must have an *observed target* —
        # a feature-near stock with a missing target cannot contribute a y value,
        # so counting it toward k would silently shrink the effective regression
        # sample below k.
        peer_mask = valid & target_fin
        for i in range(n):
            if not valid[i]:
                continue
            y_i = y_t[i]
            if not np.isfinite(y_i):
                continue
            nbrs = _neighbors(U, valid, target_k, i, peer_mask)
            # P1-45(a): fail-close when fewer than the requested k peers exist —
            # a k=10 request must not silently degrade into a 4-peer fit.  The
            # round-3 DOF floor is applied on top of the requested k.
            if nbrs.size < target_k:
                continue
            Z = U[nbrs]
            y = y_t[nbrs]
            # all neighbours are target-finite by construction; keep the guard.
            fin = np.isfinite(y)
            if int(fin.sum()) < target_k:
                continue
            Z = Z[fin]
            y = y[fin]
            # P1-45(b): fit with an intercept.  Rank-standardised features live in
            # [0,1], so a no-intercept fit forces the regression through the
            # origin and biases the residual for off-centre targets.  The ridge
            # penalty applies to the feature slopes only, never the intercept.
            Xd = np.column_stack([np.ones(len(y)), Z])
            pen = np.zeros((d + 1, d + 1))
            pen[1:, 1:] = float(ridge) * np.eye(d)
            try:
                beta = np.linalg.solve(Xd.T @ Xd + pen, Xd.T @ y)
            except np.linalg.LinAlgError:
                continue
            resid = y - Xd @ beta
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med)))
            # P1-45(c): robust scale fallback — MAD, then std, then NaN.
            if mad > _EPS:
                scale = 1.4826 * mad
            else:
                sd_resid = float(np.std(resid))
                if sd_resid > _EPS:
                    scale = sd_resid
                else:
                    continue
            yhat = float(beta[0] + np.dot(U[i], beta[1:]))
            out[t, i] = float((y_i - yhat) / scale)
    return out


def _local_gradient_series(target: np.ndarray, feats: np.ndarray, k: int, ridge: float) -> np.ndarray:
    rows, n, d = feats.shape
    out = np.full((rows, n), np.nan, dtype=float)
    # R11 round-3 P1-A (item 93): same DOF floor as the linear residual — a
    # 4-parameter fit needs max(10, 5*(d+1)) neighbours to be more than
    # interpolation; below the floor -> NaN.
    # R15-INC-171/172: explicit ``target_k = max(min_neigh, k)`` (see the linear
    # kernel); the DOF multiplier is the shared declared support policy.
    min_neigh = max(_MIN_NEIGH_FLOOR, _MIN_NEIGH_DOF_MULTIPLIER * (d + 1))
    target_k = max(min_neigh, int(k))
    for t in range(rows):
        U, valid = _rank_features(feats, t)
        y_t = target[t]
        target_fin = np.isfinite(y_t)
        # R4-82: same target-observed peer eligibility as the linear residual.
        peer_mask = valid & target_fin
        for i in range(n):
            if not valid[i]:
                continue
            nbrs = _neighbors(U, valid, target_k, i, peer_mask)
            # P1-45(a): fail-close when fewer than the requested k peers exist.
            if nbrs.size < target_k:
                continue
            Z = U[nbrs]
            y = y_t[nbrs]
            fin = np.isfinite(y)
            if int(fin.sum()) < target_k:
                continue
            Z = Z[fin]
            y = y[fin]
            # P1-45(b): fit with an intercept (ridge on feature slopes only); the
            # reported gradient is the feature-slope norm, intercept excluded.
            Xd = np.column_stack([np.ones(len(y)), Z])
            pen = np.zeros((d + 1, d + 1))
            pen[1:, 1:] = float(ridge) * np.eye(d)
            try:
                beta = np.linalg.solve(Xd.T @ Xd + pen, Xd.T @ y)
            except np.linalg.LinAlgError:
                continue
            out[t, i] = float(np.linalg.norm(beta[1:]))
    return out


def _tangent_series(feats: np.ndarray, k: int) -> np.ndarray:
    rows, n, d = feats.shape
    out = np.full((rows, n), np.nan, dtype=float)
    for t in range(rows):
        U, valid = _rank_features(feats, t)
        for i in range(n):
            if not valid[i]:
                continue
            # R15-INC-171: same explicit effective-neighbourhood contract as the
            # regression kernels — the tangent plane also needs the DOF floor,
            # never a silent smaller-k fit.
            target_k = max(_MIN_NEIGH_FLOOR, int(k))
            nbrs = _neighbors(U, valid, target_k, i)
            # P1-45(a): fail-close when fewer than the requested k peers exist.
            if nbrs.size < target_k:
                continue
            cloud = U[nbrs]
            mu = cloud.mean(axis=0)
            Zc = cloud - mu
            try:
                _, s, vt = np.linalg.svd(Zc, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            # R11 round-3 P1-A (item 95): numerical-rank gate.  A top-2 tangent
            # plane when s1 >> s2 ≈ 0 is one noisy direction padded with noise.
            # Require numerical rank >= 2 AND a minimum s2/s1 ratio; below ->
            # NaN (no off-manifold distance against a rank-1 fit).
            if s.size < 2 or s[0] <= _EPS:
                continue
            if s[1] <= _EPS or s[1] / s[0] < _TANGENT_MIN_S2_S1:
                continue
            tangent_dim = 2
            V = vt[:tangent_dim].T  # (d, tangent_dim)
            # R16-080: a completely-coincident peer cloud has NO local scale —
            # ``+ _EPS`` fabricated a finite denominator for an unidentifiable
            # tangent.  A zero / sub-tolerance local scale is NaN (the distance
            # is undefined), never a machine-constant-normalized value.
            raw_scale = float(np.sqrt(np.mean(np.sum(Zc ** 2, axis=1))))
            identity_scale = float(np.sqrt(np.mean(np.sum(U ** 2, axis=1))))
            if raw_scale <= _EPS * max(1.0, identity_scale):
                continue
            local_scale = raw_scale
            off = U[i] - mu
            proj = off - off @ V @ V.T
            out[t, i] = float(np.linalg.norm(proj) / local_scale)
    return out


def _stack_feats(f1, f2, f3) -> np.ndarray:
    return np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)


def _register_knn_op(canonical: str, description: str, unit: str, cost: int, fn) -> SeriesOperator:
    def _calculate_series(
        self,
        target: pd.DataFrame,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        k: int = 20,
        ridge: float = 1e-3,
        **_: Any,
    ) -> pd.DataFrame:
        kk = int(k)
        if kk < _KNN_MIN_K:
            raise ValueError(f"{canonical} requires k >= {_KNN_MIN_K}")
        rg = float(ridge)
        if rg < 0.0:
            raise ValueError(f"{canonical} requires ridge >= 0")
        return frame_like(
            target,
            fn(target.to_numpy(dtype=float), _stack_feats(f1, f2, f3), kk, rg),
        )

    metadata = _metadata(
        canonical, description,
        ["target", "f1", "f2", "f3", "k", "ridge"],
        unit=unit, cost=cost,
        # R15-INC-171/173: the default ``k`` is raised to the DOF floor
        # (max(10, 5*(d+1)) == 20 for the 3-feature default) so the default is
        # not a guaranteed-NaN dead region.  ``k`` and ``ridge`` are estimator
        # resolution knobs, never freely-searchable economic alphas.
        param_specs={
            # R16-077: ``k`` / ``ridge`` are estimator-resolution knobs that
            # must NEVER be freely searched — they expand the AST without
            # economic signal and their injectivity on a small cross-section is
            # a fixture artifact (k > breadth -> all-NaN regardless of value).
            # Explicitly ``searchable=False``: audited preset / robustness lane
            # only, removed from the default mining search set.
            "k": ParamSpec(
                dtype=int, min=_KNN_MIN_K,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=20,
                searchable=False,
            ),
            "ridge": ParamSpec(
                dtype=float, min=0.0,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=1e-3,
                searchable=False,
            ),
        },
    )
    return register_operator(
        name=canonical,
        category="cross_sectional",
        business_category="cross_sectional",
        canonical=canonical,
        source="cross_section_local",
    )(
        type(
            canonical.replace("_", " ").title().replace(" ", "") + "Op",
            (SeriesOperator,),
            {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
        )
    )


CsKnnLocalLinearResidual = _register_knn_op(
    "cs_knn_local_linear_residual",
    "KNN 局部线性回归残差（peer-relative mispricing）。",
    "ratio",
    8,
    _local_linear_series,
)
CsKnnLocalGradientNorm = _register_knn_op(
    "cs_knn_local_gradient_norm",
    "KNN 局部回归梯度范数 ||beta||（局部响应灵敏度）。"
    "预测因子是秩坐标（无量纲），故 ||beta|| 单位与 target 相同（same_as:target），"
    "不是 generic norm。",
    "same_as:target",
    8,
    _local_gradient_series,
)


@register_operator(
    name="cs_knn_tangent_residual",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_knn_tangent_residual",
    source="cross_section_local",
    status="experimental",
)
class CsKnnTangentResidual(SeriesOperator):
    """KNN 局部切平面残差（偏离正常"股票状态流形"的程度）。

    每日对特征云做 KNN，邻居云局部 PCA 取 top-2 切空间，输出个股到切平面的
    距离除以邻居云局部尺度。与 local density/isolation 不同：即使附近邻居
    很多，偏离正常流形方向仍会被捕获。P2 / Research。

    R11 round-3 P1-A (item 95)：切平面需满足数值秩 >= 2 且 s2/s1 >= 0.05，
    否则视为单主方向+噪声，输出 NaN（不报告基于 rank-1 拟合的伪 off-manifold
    距离）。
    """

    metadata = _metadata(
        "cs_knn_tangent_residual",
        "KNN 局部切平面距离（off-manifold 程度）。",
        ["f1", "f2", "f3", "k"],
        unit="distance",
        cost=8,
        # R15-INC-171: default k raised to the DOF floor; estimator resolution.
        param_specs={
            "k": ParamSpec(
                dtype=int, min=_KNN_MIN_K,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, default=20,
            ),
        },
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, k: int = 20, **_: Any
    ) -> pd.DataFrame:
        kk = int(k)
        if kk < _KNN_MIN_K:
            raise ValueError(f"cs_knn_tangent_residual requires k >= {_KNN_MIN_K}")
        return frame_like(f1, _tangent_series(_stack_feats(f1, f2, f3), kk))


# --------------------------------------------------------------------------
# rank copula cross-section
# --------------------------------------------------------------------------
def _rank_transform(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for c in range(values.shape[1]):
        col = values[:, c]
        fin = np.isfinite(col)
        m = int(fin.sum())
        if m < 2:
            continue
        ranks = _avg_tie_ranks(col[fin])
        out[fin, c] = (ranks + 0.5) / m
    return out


def _copula_cross_series(a: np.ndarray, b: np.ndarray, grid: int, entropy: bool) -> np.ndarray:
    rows, n = a.shape
    out = np.full((rows, n), np.nan, dtype=float)
    g = int(grid)
    for t in range(rows):
        U = _rank_transform(np.stack([a[t], b[t]], axis=1))
        fin = np.isfinite(U).all(axis=1)
        N = int(fin.sum())
        if N < max(2 * g * g, 16):  # P1-21: estimator resolution requires N >> grid²
            continue
        u = np.clip((U[fin, 0] * g).astype(int), 0, g - 1)
        v = np.clip((U[fin, 1] * g).astype(int), 0, g - 1)
        raw = np.zeros((g, g), dtype=np.float64)
        for i in range(u.shape[0]):
            raw[u[i], v[i]] += 1.0
        # R26-009..011: single estimator definition — Jeffreys/Dirichlet
        # smoothed plug-in, NO Miller–Madow on top.  Miller–Madow is a bias
        # correction for the RAW plug-in estimator; layering it over a
        # pseudo-count smoothing double-corrects (R26-010), driving MI negative
        # on near-independent data and adding a mechanical N-drift.  The
        # smoothed joint is a proper probability distribution, so the KL MI is
        # non-negative and the normalized entropy stays in [0,1] by construction.
        joint = raw + _ALPHA  # Jeffreys smoothing, deterministic
        joint /= joint.sum()
        if entropy:
            ent = float(-np.sum(joint * np.log(joint)))
            # Normalize by log(grid²) so entropy is a resolution-independent
            # [0,1] concentration measure (P1-21).  No MM term (R26-011).
            out[t, :] = ent / np.log(float(g * g))
        else:
            pu = joint.sum(axis=1)
            pv = joint.sum(axis=0)
            mi = 0.0
            for i in range(g):
                for j in range(g):
                    p = joint[i, j]
                    denom = pu[i] * pv[j]
                    if p <= _EPS or denom <= _EPS:
                        continue
                    mi += p * np.log(p / denom)
            # KL(MI) of the smoothed joint — always >= 0.  No MM term (R26-011).
            out[t, :] = float(mi)
    return out


def _register_copula_op(canonical: str, description: str, entropy: bool) -> SeriesOperator:
    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, grid: int = 8, **_: Any) -> pd.DataFrame:
        g = int(grid)
        # P1-21: ``grid`` is an *estimator resolution*, not a searchable alpha
        # parameter — vary it and you change plug-in bias / finite-sample noise
        # rather than market structure.  Only a small verified set is allowed.
        if g not in (4, 8, 16):
            raise ValueError(f"{canonical} requires grid in {{4, 8, 16}} (verified estimator resolutions)")
        return frame_like(
            a,
            _copula_cross_series(a.to_numpy(dtype=float), b.to_numpy(dtype=float), g, entropy),
        )

    # R11 #28: ``cs_rank_copula_entropy`` is normalised by log(grid^2) to [0,1]
    # — the unit is DIMENSIONLESS, not nats.  ``cs_rank_copula_mi`` stays nats.
    # R26-011: both use the Jeffreys-smoothed plug-in estimator (NO Miller–Madow
    # layer) — single definition, MI >= 0 by construction.
    unit = "dimensionless" if entropy else "nats"
    metadata = _metadata(
        canonical, description, ["a", "b", "grid"], unit=unit, cost=6,
        # P1-44: the copula MI/entropy is a per-day *market-wide scalar* broadcast
        # to every stock — it is a GLOBAL_STATE / regime feature for ``where`` /
        # ``trade_when`` / state conditioning, NOT a per-stock Numeric Alpha.  The
        # tag routes it out of the per-stock alpha pool.
        extra_tags=("global_state",),
    )
    return register_operator(
        name=canonical,
        category="cross_sectional",
        business_category="cross_sectional",
        canonical=canonical,
        source="cross_section_local",
        status="experimental",
    )(
        type(
            canonical.replace("_", " ").title().replace(" ", "") + "Op",
            (SeriesOperator,),
            {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
        )
    )


CsRankCopulaMi = _register_copula_op(
    "cs_rank_copula_mi",
    "横截面 rank copula 互信息（两字段同/异向信息量）。",
    False,
)
CsRankCopulaEntropy = _register_copula_op(
    "cs_rank_copula_entropy",
    "横截面 rank copula 熵（联合秩依赖的分散度）。",
    True,
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"cs_knn_local_linear_residual"})
    _surface.extend_research_only(
        {
            "cs_knn_tangent_residual",
            "cs_knn_local_gradient_norm",
            "cs_rank_copula_mi",
            "cs_rank_copula_entropy",
        }
    )
    for _canon in (
        "cs_knn_local_linear_residual",
        "cs_knn_tangent_residual",
        "cs_knn_local_gradient_norm",
        "cs_rank_copula_mi",
        "cs_rank_copula_entropy",
    ):
        register_polars_udf(_canon)


_register_surface()
