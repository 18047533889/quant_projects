# -*- coding: utf-8 -*-
"""Extended dependence operators (2026-08 geometry/math expansion).

Nonlinear / directed dependence measures beyond plain correlation:

* ``ts_chatterjee_xi``          — Chatterjee's directed rank correlation
  (functional dependence of ``y`` on ``x``; 1 when ``y = f(x)`` is deterministic).
* ``ts_hsic``                   — normalized Hilbert-Schmidt independence
  criterion with RBF kernels and the median *positive* pairwise-distance
  bandwidth (audit #53: a zero median under heavy ties must not fall back to a
  near-identity kernel).
* ``ts_conditional_mutual_information`` — CMI(X;Y|Z) from quantile-discretized
  joint counts, normalized by ``log(effective_bins)`` (base-e entropy; audit
  #54: after ties collapse the effective state space, the denominator uses the
  effective cell counts, never the requested ``bins``).
* ``ts_distance_correlation_partial_proxy`` — the Pearson-combination of
  distance correlations ``(r_xy - r_xz·r_yz) / sqrt((1-r_xz²)(1-r_yz²))`` built
  from double-centered Euclidean distance matrices.  This is a *proxy*, NOT the
  strict Székely/Rizzo partial distance correlation (which requires U-centered
  distance matrices plus a Hilbert-space projection); the name is deliberately
  honest (audit #52).  The legacy ``ts_partial_distance_correlation`` resolves
  as a deprecated alias.

All operators are trailing-window, prefix-causal and deterministic.  Only
same-position finite aligned triples/pairs are used; degenerate windows (too
few points, constant marginals, zero kernel bandwidth) emit NaN (fail-closed).
Invalid parameters raise ``ValueError``.

Timing honesty: ``ts_hsic`` is a trailing-window descriptive dependence
statistic, not PRIOR_FIT and not a BLOCKED_HISTORICAL_EVALUATION forecast.
The blocked historical train/test diagnostic is ``ts_kernel_granger_score`` in
``research_spectral.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    aligned_pairs,
    check_window,
    frame_like,
    map_pair_rolling,
    register_polars_bridge,
)

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="dependence_ext",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "dependence_ext", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:dependence",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# shared kernels (private to this module; 2D panel = TradeDate x Symbol)
# ---------------------------------------------------------------------------
def _rankdata(v: np.ndarray) -> np.ndarray:
    """Average ranks (1-based); ties share the mean rank."""
    n = v.size
    order = np.argsort(v, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _value_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Window-local quantile buckets over the values (side='right', clip to [0,bins-1])."""
    cuts = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    bucket = np.searchsorted(cuts, values, side="right")
    return np.clip(bucket.astype(np.int64), 0, bins - 1)


def _entropy_from_counts(counts: np.ndarray, total: int) -> float:
    p = counts[counts > 0] / float(total)
    return float(-np.sum(p * np.log(p)))


def _chatterjee_xi(xv: np.ndarray, yv: np.ndarray) -> float:
    n = xv.size
    if n < 3:
        return np.nan
    if float(np.std(xv) <= _EPS or float(np.std(yv) <= _EPS:
        return np.nan
    # Tie-aware Chatterjee ξ (review R4-49, R26-005..007).  The no-tie formula
    # 1 - 3·Σ|Δr|/(n²-1) assumes continuous data; A-share series have many
    # zero-return / limit / discrete values.  The general formula uses
    # r_i = #{j: Y_j ≤ Y_i} (max-rank) and l_i = #{j: Y_j ≥ Y_i}:
    #   ξ = 1 - n·Σ_{i<n}|r_{i+1}-r_i| / (2·Σ_i l_i(n-l_i)).
    #
    # R26-005/006: X ties MUST NOT be resolved by sorting on Y.  Sorting tied-X
    # rows by the *response* makes the within-group Y-sequence monotone, which
    # mechanically shrinks |r_{i+1}-r_i| inside every tie group and inflates ξ —
    # on independent data with heavy X ties this reaches ~0.9 (measured).  We
    # use a stable sort on X only: tied-X rows keep their input (chronological)
    # order, which is deterministic and completely independent of y (R26-007).
    order = np.argsort(xv, kind="stable")
    sy = yv[order]
    s_sorted = np.sort(sy)
    r = np.searchsorted(s_sorted, sy, side="right").astype(float)  # max-rank
    l = (n - np.searchsorted(s_sorted, sy, side="left")).astype(float)  # #{≥}
    denom = 2.0 * float(np.sum(l * (n - l)))
    if denom <= _EPS:
        return np.nan
    return float(1.0 - n * np.sum(np.abs(np.diff(r))) / denom)


def _rbf_kernel(v: np.ndarray) -> np.ndarray | None:
    n = v.size
    d = np.abs(v[:, None] - v[None, :])
    tri = d[np.triu_indices(n, 1)]
    if tri.size == 0:
        return None
    # Audit #53: bandwidth is the median of the POSITIVE pairwise distances.
    # With heavy ties (e.g. 0,0,0,0,0,1,2) more than half the pairwise distances
    # are zero, so ``median(all)`` = 0; the old ``sigma = max(median, EPS)``
    # fallback made the kernel ≈ identity and HSIC mostly reflected the tie
    # rate, not genuine dependence.  Positive distances give the real scale of
    # the non-constant structure.  If too few positive distances exist the
    # series is (near-)constant -> fail closed to NaN.
    pos = tri[tri > _EPS]
    if pos.size < max(2, int(np.sqrt(n))):
        return None
    sigma = float(np.median(pos))
    if sigma <= _EPS:
        return None
    return np.exp(-(d * d) / (2.0 * sigma * sigma))


def _hsic(xv: np.ndarray, yv: np.ndarray) -> float:
    n = xv.size
    if n < 5:
        return np.nan
    K = _rbf_kernel(xv)
    L = _rbf_kernel(yv)
    if K is None or L is None:
        return np.nan
    H = np.eye(n) - np.ones((n, n)) / n
    # Centered kernel alignment (review R4-14): Kc = H K H, Lc = H L H and
    # HSIC_norm = <Kc,Lc>_F / (||Kc||_F·||Lc||_F).  For symmetric kernels,
    # <Kc,Lc>_F = tr(K H L H) and ||Kc||_F² = tr(K H K H) — the old denominator
    # tr(K H K)·tr(L H L) was missing one centering H on each side.
    num = float(np.trace(K @ H @ L @ H))
    denom = float(np.sqrt(np.trace(K @ H @ K @ H) * np.trace(L @ H @ L @ H)))
    if denom <= _EPS:
        return np.nan
    return float(num / denom)


def _cmi(xv: np.ndarray, yv: np.ndarray, zv: np.ndarray, bins: int) -> float:
    n = xv.size
    # Plug-in empirical entropy over bins^3 cells has huge finite-sample bias;
    # require N >= 2·bins^3 aligned triples (review R4-50).
    if n < 2 * bins * bins * bins:
        return np.nan
    bx = _value_bins(xv, bins)
    by = _value_bins(yv, bins)
    bz = _value_bins(zv, bins)
    cz = np.bincount(bz, minlength=bins).astype(float)
    if cz.sum() == 0:
        return np.nan
    hz = _entropy_from_counts(cz, n)
    cxz = np.zeros((bins, bins), dtype=float)
    np.add.at(cxz, (bx, bz), 1.0)
    hxz = _entropy_from_counts(cxz.ravel(), n)
    cyz = np.zeros((bins, bins), dtype=float)
    np.add.at(cyz, (by, bz), 1.0)
    hyz = _entropy_from_counts(cyz.ravel(), n)
    cxyz = np.zeros((bins, bins, bins), dtype=float)
    np.add.at(cxyz, (bx, by, bz), 1.0)
    hxyz = _entropy_from_counts(cxyz.ravel(), n)
    cmi = hxz + hyz - hz - hxyz
    # Audit #54: after ties collapse the effective state space, the normalized
    # denominator must use the EFFECTIVE occupied cell counts — CMI <=
    # min(H(X), H(Y) <= min(log n_x_eff, log n_y_eff) — never log(requested
    # bins).  A marginal that degenerated to a single occupied cell carries no
    # conditional information -> fail closed.
    n_x_eff = int(np.unique(bx).size)
    n_y_eff = int(np.unique(by).size)
    eff = min(n_x_eff, n_y_eff)
    if eff < 2:
        return np.nan
    return float(cmi / np.log(eff))


def _double_center(v: np.ndarray) -> np.ndarray:
    d = np.abs(v[:, None] - v[None, :])
    return d - d.mean(axis=0, keepdims=True) - d.mean(axis=1, keepdims=True) + d.mean()


def _normalized_dcov(A: np.ndarray, B: np.ndarray) -> float:
    """Normalized distance covariance: dcov/sqrt(dvar_x * dvar_y) (distance corr)."""
    n = A.shape[0]
    dcov2 = float(np.sum(A * B)) / float(n * n)
    dvar2_a = float(np.sum(A * A)) / float(n * n)
    dvar2_b = float(np.sum(B * B)) / float(n * n)
    if dvar2_a <= _EPS or dvar2_b <= _EPS:
        return np.nan
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvar2_a * dvar2_b)))


def _partial_dcor_proxy(xv: np.ndarray, yv: np.ndarray, zv: np.ndarray) -> float:
    """Pearson-formula proxy for partial distance correlation (audit #52).

    This computes ``(r_xy - r_xz·r_yz) / sqrt((1 - r_xz²)(1 - r_yz²))`` over the
    plain double-centered distance correlations.  That is the standard partial
    CORRELATION combination, NOT the strict Székely/Rizzo partial distance
    correlation (which requires U-centered distance matrices plus a Hilbert
    space projection of the distance-covariance operator).  It is kept and
    RENAMED honestly as ``ts_distance_correlation_partial_proxy`` — a fast,
    interpretable control-for-Z dependence proxy, but not the certified pdCor.
    """
    n = xv.size
    if n < 5:
        return np.nan
    A = _double_center(xv)
    B = _double_center(yv)
    C = _double_center(zv)
    r_xy = _normalized_dcov(A, B)
    r_xz = _normalized_dcov(A, C)
    r_yz = _normalized_dcov(B, C)
    if not (np.isfinite(r_xy) and np.isfinite(r_xz) and np.isfinite(r_yz)):
        return np.nan
    num = r_xy - r_xz * r_yz
    # Partial-correlation combination (review R4-15): the denominator needs the
    # *squared* conditioning correlations, sqrt((1-r_xz²)(1-r_yz²)), not
    # sqrt((1-r_xz)(1-r_yz)).
    raw_denom2 = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
    # Check the unregularized geometry before applying any numerical floor:
    # singular controls are unidentified, not merely numerically unstable.
    if not np.isfinite(raw_denom2) or raw_denom2 <= _EPS:
        return np.nan
    denom = float(np.sqrt(raw_denom2))
    if not np.isfinite(denom) or denom <= _EPS:
        return np.nan
    return float(num / denom)


def _triple_series(a2d: np.ndarray, b2d: np.ndarray, c2d: np.ndarray, window: int, kernel: Any) -> np.ndarray:
    rows, cols = a2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            a, b, cc = a2d[i0 : r + 1, c], b2d[i0 : r + 1, c], c2d[i0 : r + 1, c]
            m = np.isfinite(a) & np.isfinite(b) & np.isfinite(cc)
            if m.sum() < 2:
                continue
            out[r, c] = kernel(a[m].astype(float), b[m].astype(float), cc[m].astype(float))
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_chatterjee_xi",
    category="dependence_ext",
    business_category="dependence_ext",
    canonical="ts_chatterjee_xi",
    source="dependence_ext",
)
class TsChatterjeeXi(SeriesOperator):
    """Chatterjee 秩相关:按 x 排序后相邻 y 秩差的归一化,测量函数型依赖。

    接近 1 → y 几乎由 x 决定（任意单调/非单调函数均捕捉）。P2。
    R26-007: tie-aware 估计量；x 并列内部按输入（时序）序确定，与 y 无关。
    """

    metadata = _metadata(
        "ts_chatterjee_xi",
        "Chatterjee 方向函数依赖度 ξ（秩差归一化）。",
        ["x", "y", "window"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = check_window(window)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            return _chatterjee_xi(pa, pb)

        return frame_like(x, map_pair_rolling(x.to_numpy(dtype=float), y.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_hsic",
    category="dependence_ext",
    business_category="dependence_ext",
    canonical="ts_hsic",
    source="dependence_ext",
)
class TsHsic(SeriesOperator):
    """归一化 HSIC:RBF 核（中位数距离带宽）+ 双中心化后的核对齐。

    捕捉任意非线性依赖,输出 [0,1]。P2。

    **Timing honesty（M-9xx）**：``ts_hsic`` 是 trailing-window 的 *描述性*
    依赖统计量——在同一窗口 ``[t-window, t]`` 内同时读 x 与 y 并计算核对齐，
    **不是** t-1 fit→forecast t 的预测诊断分（与 ``ts_kernel_granger_score``
    的 BLOCKED_HISTORICAL_EVALUATION 不同；本算子不做 train/test 划分）。它
    描述"最近 window 内 x 与 y 的联合非线性依赖有多强"，不冒充 PRIOR_FIT。
    """

    metadata = _metadata(
        "ts_hsic",
        "归一化 Hilbert-Schmidt 独立性准则（RBF 核,中位数带宽；trailing-window 描述性依赖，"
        "非预测分）。",
        ["x", "y", "window"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = check_window(window)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            return _hsic(pa, pb)

        return frame_like(x, map_pair_rolling(x.to_numpy(dtype=float), y.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_conditional_mutual_information",
    category="dependence_ext",
    business_category="dependence_ext",
    canonical="ts_conditional_mutual_information",
    source="dependence_ext",
)
class TsConditionalMutualInformation(SeriesOperator):
    """条件互信息 CMI(X;Y|Z):分位数离散化 + 联合计数经验熵,按 log(bins) 归一。

    衡量控制 Z 后 X 与 Y 的剩余依赖。P2。
    """

    metadata = _metadata(
        "ts_conditional_mutual_information",
        "条件互信息 CMI(X;Y|Z),经验基-e 熵,按 log(bins) 归一 → [0,1] 无量纲比值。",
        ["x", "y", "z", "window", "bins"],
        unit="ratio",
        cost=6,
    )
    metadata.param_specs = {
        # R4-50: bins^3 plug-in cells need large N; production grid is {2, 3}.
        "bins": ParamSpec(dtype=int, min=2, max=3, choices=(2, 3), param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "window": ParamSpec(dtype=int, min=2),
    }

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, window: int = 120, bins: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        b = int(bins)
        if b not in (2, 3):
            raise ValueError("bins must be 2 or 3 (production grid, review R4-50)")
        return frame_like(
            x,
            _triple_series(
                x.to_numpy(dtype=float), y.to_numpy(dtype=float), z.to_numpy(dtype=float), w,
                lambda a, bv, cv: _cmi(a, bv, cv, b),
            ),
        )


@register_operator(
    name="ts_distance_correlation_partial_proxy",
    category="dependence_ext",
    business_category="dependence_ext",
    canonical="ts_distance_correlation_partial_proxy",
    source="dependence_ext",
)
class TsPartialDistanceCorrelation(SeriesOperator):
    """偏距离相关代理:控制 Z 后 X-Y 的 Pearson 组合距离相关（非严格 pdCor）。

    用双中心化欧氏距离矩阵构造,全部使用归一化 dCov
    (= dcov/sqrt(dvar_x·dvar_y)),再套用偏相关组合式
    ``(r_xy - r_xz·r_yz) / sqrt((1-r_xz²)(1-r_yz²))``。审计 #52：这是
    **proxy**，不是 Székely/Rizzo 的严格偏距离相关（后者需要 U-center 距离
    矩阵 + Hilbert 空间投影）；命名已诚实改为 ``ts_distance_correlation_
    partial_proxy``，旧名 ``ts_partial_distance_correlation`` 作为弃用别名。
    P2。
    """

    metadata = _metadata(
        "ts_distance_correlation_partial_proxy",
        "控制 Z 后的 X-Y 距离相关代理（Pearson 组合，非严格 Székely/Rizzo pdCor）。",
        ["x", "y", "z", "window"],
        unit="corr",
        cost=7,
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(
            x,
            _triple_series(
                x.to_numpy(dtype=float), y.to_numpy(dtype=float), z.to_numpy(dtype=float), w,
                _partial_dcor_proxy,
            ),
        )


_NEW_CANONICALS = (
    "ts_chatterjee_xi",
    "ts_hsic",
    "ts_conditional_mutual_information",
    "ts_distance_correlation_partial_proxy",
)
# Audit #52 honest rename: the legacy name computed the Pearson combination of
# distance correlations, NOT the strict Székely/Rizzo partial distance
# correlation.  It resolves as a deprecated alias so existing recipes load.
_DEPRECATED_ALIASES = {
    "ts_partial_distance_correlation": "ts_distance_correlation_partial_proxy",
}


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface
    from cleaned_operators.registry import OperatorRegistry

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _old, _new in _DEPRECATED_ALIASES.items():
        try:
            OperatorRegistry.register_alias(_old, _new)
        except (KeyError, ValueError):
            pass  # already registered
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
