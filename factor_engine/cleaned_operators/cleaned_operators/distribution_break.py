# -*- coding: utf-8 -*-
"""Multivariate full-distribution break detection (2026-08 V2).

Univariate location/scale/KS/Wasserstein shifts miss joint changes: each
marginal may barely move while the *joint* distribution of (return, turnover,
volatility) shifts.  Energy distance is a parameter-light omnibus distance
between empirical multivariate samples.

* ``ts_joint_energy_shift``        — energy distance between a recent and a prior
  window of **3 features** (fixed 3-feature canonical, ``f1,f2,f3``),
  robust-standardised on the prior window (P1).
* ``ts_energy_break_score``        — robust z of the current joint-energy shift
  against its own trailing history (regime-break score) (P1).
* ``ts_copula_central_asymmetry``  — squared deviation of the empirical copula
  from central symmetry (P2 research).

Deterministic, prefix-causal: both windows end at ``t-1`` for the shift; the
break score then compares the shift at ``t`` against strictly-past shifts.

Timing classification: PRIOR_REFERENCE_CURRENT_QUERY / PRE_T_STATE.  The
current row is a query point only; neither the prior nor recent reference
window includes it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(
    name: str, description: str, params: list[str], *, unit: str, cost: int,
    extra_tags: tuple[str, ...] = (),
    param_specs: dict | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="distribution_shift",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "distribution_shift", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
        param_specs=dict(param_specs) if param_specs else {},
    )


def _stack_feats(features: list[pd.DataFrame]) -> np.ndarray:
    return np.stack([f.to_numpy(dtype=float) for f in features], axis=2)


def _euclidean_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(m,d) vs (n,d) -> (m,n) pairwise Euclidean distances.

    No ``+ EPS`` inside the radical (P0-04 review): an additive floor makes the
    self-distance ``d(x_i, x_i) = sqrt(EPS) > 0`` and systematically biases the
    energy distance ``2 E|X-Y| - E|X-X'| - E|Y-Y'|``.  A distance is a distance;
    degenerate zero-distance cases are handled at the caller (``max(ed, 0)``).
    """
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """U-stat energy distance between two multivariate samples (R3-139).

    The classical V-statistic ``2 E|X-Y| - E|X-X'| - E|Y-Y'|`` includes the
    diagonal zeros and divides the within-sample terms by ``m^2`` / ``n^2``, so
    its finite-N bias differs between the two sides whenever ``recent_window !=
    prior_window`` — a purely mechanical bias that changes the baseline.  This
    implementation uses the UNBIASED U-stat form: the within-sample terms
    exclude the diagonal and divide by ``m*(m-1)`` / ``n*(n-1)``, so the
    estimator is unbiased at every finite sample size on both sides.
    """
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        return np.nan
    dxy = _euclidean_pairs(X, Y)
    dxx = _euclidean_pairs(X, X)
    dyy = _euclidean_pairs(Y, Y)
    first = float(np.sum(dxy)) / (m * n)
    # U-stat: exclude the diagonal (self-distances); denominator m*(m-1).
    second = float(np.sum(dxx) - np.trace(dxx)) / (m * (m - 1))
    third = float(np.sum(dyy) - np.trace(dyy)) / (n * (n - 1))
    ed = 2.0 * first - second - third
    return float(max(ed, 0.0))


def _joint_shift_cell(block: np.ndarray, recent: int, prior: int) -> float:
    """Energy distance between recent and prior windows (robust-std on prior).

    R6-122 (strict-past contract): both windows end at ``t-1`` — the current
    row ``t`` is ONLY the query point, never part of either block.  The caller
    passes ``[.., t]`` (inclusive); slicing here stops at ``n-1`` so the recent
    window is ``[t-r, t-1]`` and the prior is ``[t-r-p, t-r-1]``.  The old code
    sliced ``block[n-r:n]`` which re-included the current row.
    """
    r = int(recent)
    p = int(prior)
    n = block.shape[0]
    if n < r + p + 1:
        return np.nan
    prior_block = block[n - r - p - 1 : n - r - 1]      # [t-r-p, t-r-1]
    recent_block = block[n - r - 1 : n - 1]             # [t-r, t-1]
    d = prior_block.shape[1]
    med = np.median(prior_block, axis=0)
    mad = 1.4826 * np.median(np.abs(prior_block - med), axis=0)
    scale = np.where(mad > _EPS, mad, np.std(prior_block, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= _EPS):
        return np.nan
    X = (recent_block - med) / scale
    Y = (prior_block - med) / scale
    return _energy_distance(X, Y)


def _joint_shift_series(feats: np.ndarray, recent: int, prior: int) -> np.ndarray:
    rows, cols, _ = feats.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for t in range(rows):
            # R9-OP-001 (off-by-one): ``_joint_shift_cell`` needs ``p + r + 1``
            # rows (prior p + recent r + current-row query).  The old bound
            # sliced ``[t-r-p, t]`` with only ``p + r`` rows, so the cell's
            # ``n < r + p + 1`` guard was ALWAYS true and the whole series was
            # NaN.  The inclusive ``t`` row is the query only: it never enters
            # either reference block (see ``_joint_shift_cell`` slicing).
            lo = max(0, t - (recent + prior))
            if t - lo + 1 < recent + prior + 1:
                continue
            out[t, c] = _joint_shift_cell(feats[lo : t + 1, c, :], recent, prior)
    return out


def _robust_z(series: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    mp = max(2, int(min_periods))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w)
        past = series[lo:t]
        finite = past[np.isfinite(past)]
        if finite.size < mp:
            continue
        cur = series[t]
        if not np.isfinite(cur):
            continue
        med = float(np.median(finite))
        mad = 1.4826 * float(np.median(np.abs(finite - med)))
        scale = mad if mad > _EPS else float(np.std(finite))
        if scale <= _EPS:
            continue
        out[t] = float((cur - med) / scale)
    return out


@register_operator(
    name="ts_joint_energy_shift",
    category="distribution_shift",
    business_category="distribution_shift",
    canonical="ts_joint_energy_shift",
    source="distribution_break",
)
class TsJointEnergyShift(SeriesOperator):
    """多变量联合分布 Energy distance 漂移（recent vs prior 窗口）。

    **3-feature canonical**（``f1,f2,f3``，固定 d=3；R9-P1-003 明确
    machine contract，不再宣称 "2..4 features"）。特征堆成 3 维样本；只用
    prior 窗口的 median/MAD 做稳健标准化，然后计算 recent 与 prior 之间的
    energy distance（omnibus 全分布距离，clip≥0）。可发现单变量都变化不大、
    corr 也不明显但联合分布改变的情形。PIT 安全（两个窗口都止于 t-1）。

    R3-140 (pre-t state): the value emitted at time ``t`` is computed ONLY from
    rows ``[t-recent-prior, t-1]`` — the current row ``t`` is the query point
    and never enters either block.  The output is therefore an
    ``asof_previous_observation`` pre-t state: it is known at the START of
    period ``t`` and can be used in a same-day decision without look-ahead.

    **真实 timing 归类 = PRIOR_REFERENCE_CURRENT_QUERY / PRE_T_STATE（M-6xx）**：
    prior 窗口与 recent 窗口都结束于 t-1（current t 只是查询点，不参与任一
    reference 窗口）。实现 ``_joint_shift_cell`` 对传入块 ``[.., t]`` 切片
    ``recent = [t-r, t-1]``、``prior = [t-r-p, t-r-1]``——当前行 t 从不泄漏进
    reference。
    """

    metadata = _metadata(
        "ts_joint_energy_shift",
        "多变量 Energy distance 联合分布漂移（recent vs prior，固定 3-feature；"
        "asof t-1；timing=PRIOR_REFERENCE_CURRENT_QUERY/PRE_T_STATE）。",
        ["f1", "f2", "f3", "recent_window", "prior_window"],
        unit="distance",
        cost=6,
        extra_tags=("asof_previous_observation",),
    )

    def _calculate_series(
        self,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        recent_window: int = 20,
        prior_window: int = 60,
        **_: Any,
    ) -> pd.DataFrame:
        r = max(2, int(recent_window))
        p = max(2, int(prior_window))
        feats = _stack_feats([f1, f2, f3])
        return frame_like(f1, _joint_shift_series(feats, r, p))


@register_operator(
    name="ts_energy_break_score",
    category="distribution_shift",
    business_category="distribution_shift",
    canonical="ts_energy_break_score",
    source="distribution_break",
)
class TsEnergyBreakScore(SeriesOperator):
    """联合分布 regime-break 得分：当前 energy shift 相对自身历史的多稳健 z。

    先算每期的 joint energy shift（与 ``ts_joint_energy_shift`` 同核），再对
    过去 ``window`` 期的 shift 序列做 robust z：高值 = 当前联合分布位移在自身
    历史上极不寻常（状态切换/崩坍）。PIT 安全、确定性。

    R3-140 (pre-t state): like ``ts_joint_energy_shift`` the per-period shift
    uses only rows up to ``t-1``, so the break score at ``t`` is an
    ``asof_previous_observation`` pre-t state usable same-day without
    look-ahead.

    **真实 timing 归类 = PRIOR_REFERENCE_CURRENT_QUERY / PRE_T_STATE（M-6xx）**：
    每个周期的 joint energy shift 的 prior/recent 窗口都止于 t-1；当前 t 只是
    查询点，不参与 reference。break score 是把该 pre-t shift 与更早的
    shift 历史做稳健 z。
    """

    metadata = _metadata(
        "ts_energy_break_score",
        "联合分布 break 得分（energy shift 的稳健 z；asof t-1）。",
        ["f1", "f2", "f3", "window", "recent_window", "prior_window"],
        unit="zscore",
        cost=6,
        extra_tags=("asof_previous_observation",),
    )

    def _calculate_series(
        self,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        window: int = 60,
        recent_window: int = 20,
        prior_window: int = 60,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(2, int(window))
        r = max(2, int(recent_window))
        p = max(2, int(prior_window))
        feats = _stack_feats([f1, f2, f3])
        shifts = _joint_shift_series(feats, r, p)
        rows, cols = shifts.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _robust_z(shifts[:, c], w, 3)
        return frame_like(f1, out)


# ---------------------------------------------------------------------------
# ts_copula_central_asymmetry (P2 research)
# ---------------------------------------------------------------------------

def _midranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (midranks) of ``a``, ties sharing the mean rank.

    ``np.argsort(np.argsort(x))`` assigns arbitrary *competition* ranks to ties
    (ordered by original index), so a factor's copula pseudo-observations become
    ticker/row-order sensitive on discrete fields (0 returns, rounded ratios,
    limit states).  Average ranks are invariant to the input's original order
    (round-7 P0, review §35).
    """
    n = a.size
    if n == 0:
        return a.astype(float)
    order = np.argsort(a, kind="stable")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[i:j] = (i + 1.0 + j) / 2.0  # mean of positions i+1..j
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _copula_central_asymmetry_cell(xv: np.ndarray, yv: np.ndarray, grid: int) -> float:
    finite = np.isfinite(xv) & np.isfinite(yv)
    x = xv[finite]
    y = yv[finite]
    if x.size < 10:
        return np.nan
    G = max(4, int(grid))
    u = (_midranks(x) - 0.5) / x.size
    v = (_midranks(y) - 0.5) / y.size
    g = np.linspace(0.5 / G, 1.0 - 0.5 / G, G)
    C = np.empty((G, G), dtype=float)
    for a in range(G):
        for b in range(G):
            C[a, b] = float(np.mean((u <= g[a]) & (v <= g[b])))
    asym = 0.0
    for a in range(G):
        for b in range(G):
            # C(1-u,1-v) lands on the symmetric grid cell.
            term = C[a, b] - g[a] - g[b] + 1.0 - C[G - 1 - a, G - 1 - b]
            asym += term * term
    return float(asym / (G * G))


def _copula_series(xv: np.ndarray, yv: np.ndarray, window: int, grid: int) -> np.ndarray:
    rows, cols = xv.shape
    w = max(2, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for t in range(rows):
            lo = max(0, t - w + 1)
            if t - lo + 1 < w:
                continue
            out[t, c] = _copula_central_asymmetry_cell(xv[lo : t + 1, c], yv[lo : t + 1, c], grid)
    return out


@register_operator(
    name="ts_copula_central_asymmetry",
    category="distribution_shift",
    business_category="distribution_shift",
    canonical="ts_copula_central_asymmetry",
    source="distribution_break",
    status="experimental",
)
class TsCopulaCentralAsymmetry(SeriesOperator):
    """经验 copula 中心非对称度 ``(1/G²)Σ[C(u,v)-u-v+1-C(1-u,1-v)]²``。

    不是简单的 upper/lower 尾依赖差，而是比较整个联合 dependence structure
    正负方向是否不对称（downturn 中尤其值得关注）。P2 / Research。

    R3-141 (null calibration): this RAW statistic has an N/grid-dependent
    baseline — the empirical copula's deviation from central symmetry is
    mechanically inflated by small sample sizes (``window``), a coarse grid, and
    tie density, independent of any real asymmetric dependence.  It is therefore
    a RESEARCH statistic only: interpret it against a matched permutation / IID
    null at the SAME ``(window, grid)`` before calling a value "asymmetric".
    The nuisance parameters ``window`` and ``grid`` are declared
    ``ESTIMATOR_RESOLUTION`` + ``searchable=False`` so a role-aware search
    grammar cannot optimise the raw baseline itself (R3-141).
    """

    metadata = _metadata(
        "ts_copula_central_asymmetry",
        "经验 copula 中心非对称度（联合依赖方向不对称性；raw，需 N/grid null 校准）。",
        ["x", "y", "window", "grid"],
        unit="distance",
        cost=7,
        param_specs={
            "window": ParamSpec(dtype=int, min=4, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False),
            "grid": ParamSpec(dtype=int, min=4, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, y: pd.DataFrame, window: int = 120, grid: int = 8, **_: Any
    ) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        return frame_like(x, _copula_series(xv, yv, window, grid))


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"ts_joint_energy_shift", "ts_energy_break_score"})
    _surface.extend_research_only({"ts_copula_central_asymmetry"})


_register_surface()
