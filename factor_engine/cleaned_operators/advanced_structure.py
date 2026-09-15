# -*- coding: utf-8 -*-
"""Advanced dependence-structure operators (2026-08 Gemini round, P1).

Deterministic, prefix-causal pandas-numpy references returning
``(TradeDate x Symbol)`` panels.

* ``ts_bures_corr_shift``                    — Bures-Wasserstein distance between
  two non-overlapping correlation windows (2x2, 5% shrinkage) — a PIT-safe
  price-volume / return-volatility relationship break detector.
* ``ts_kramers_moyal_drift`` / ``diffusion`` — state-conditioned first / second
  Kramers-Moyal coefficients (local drift / diffusion) evaluated at the current
  state. Deterministic historical quantile state bins; expanding history with
  at least 10 finite past observations.
* ``cs_sliced_wasserstein_copula_shift``     — Sliced-Wasserstein distance
  between today's cross-sectional rank copula (fixed seeded directions) and the
  trailing reference — a market/group regime state, broadcast to every stock.
* ``group_spd_feature_structure_shift``      — per-group log-Euclidean SPD
  correlation-structure shift (exactly 3 features), one value per stock.
* ``holder_class_js_shift``                  — Jensen-Shannon distance between
  current and previous holder-class share distributions (5 class slots).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator, ParamSpec, ParamRole
from factor_engine.cleaned_operators.parameter_validation import strict_integer

_EPS = 1e-12
_LAM = 0.05              # fixed correlation shrinkage / SPD regularization.
_SEED = 20260807         # fixed seed -> deterministic projection directions.
_LN2 = float(np.log(2.0))


_STRUCTURE_TARGETS={
    "ts_bures_corr_shift","ts_kramers_moyal_drift","ts_kramers_moyal_diffusion",
    "cs_sliced_wasserstein_copula_shift","group_spd_feature_structure_shift",
}
def _structure_specs(params):
    declared={
        "recent_window":ParamSpec(dtype=int,min=2,default=20,param_role=ParamRole.HORIZON),
        "prior_window":ParamSpec(dtype=int,min=2,default=60,param_role=ParamRole.HORIZON),
        "min_pairs":ParamSpec(dtype=int,min=2,default=None,param_role=ParamRole.SUPPORT_POLICY,searchable=False),
        "window":ParamSpec(dtype=int,min=10 if "bins" in params else 2,default=60,param_role=ParamRole.HORIZON),
        "bins":ParamSpec(dtype=int,min=2,default=8,param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "lag":ParamSpec(dtype=int,min=1,default=1,param_role=ParamRole.HORIZON),
        "min_bin_count":ParamSpec(dtype=int,min=1,default=None,param_role=ParamRole.SUPPORT_POLICY,searchable=False),
        "directions":ParamSpec(dtype=int,min=4,default=32,param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "reference_window":ParamSpec(dtype=int,min=2,default=60,param_role=ParamRole.HORIZON),
        "min_peers":ParamSpec(dtype=int,min=3,default=None,param_role=ParamRole.SUPPORT_POLICY,searchable=False),
        "min_reference_days":ParamSpec(dtype=int,min=2,default=None,param_role=ParamRole.SUPPORT_POLICY,searchable=False),
        "composition_policy":ParamSpec(dtype=str,choices=("current","intersection"),default="current",param_role=ParamRole.POLICY,searchable=False),
    }
    return {key:declared[key] for key in params if key in declared}

def _valid_group_label(label):
    if label is None or pd.isna(label):
        return False
    if isinstance(label,str):
        return bool(label.strip())
    if isinstance(label,(int,float,np.number)):
        return bool(np.isfinite(label))
    return True

def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="structure_shift",
        description=description,
        param_names=params,
        panel_params=tuple(p for p in params if p not in _structure_specs(params)) if name in _STRUCTURE_TARGETS else (),
        scalar_params=tuple(_structure_specs(params)) if name in _STRUCTURE_TARGETS else (),
        param_specs=_structure_specs(params) if name in _STRUCTURE_TARGETS else {},
        return_type="series",
        tags=[
            "structure_shift", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _bures2(a: np.ndarray, b: np.ndarray) -> float:
    """Bures-Wasserstein^2 between two 2x2 symmetric PSD matrices."""
    wa, va = np.linalg.eigh(a)
    wa = np.clip(wa, 1e-12, None)
    a_half = (va * np.sqrt(wa)) @ va.T
    m = a_half @ b @ a_half
    wm, _ = np.linalg.eigh(m)
    wm = np.clip(wm, 1e-12, None)
    return float(np.trace(a) + np.trace(b) - 2.0 * np.sum(np.sqrt(wm)))


def _corr_window(xv: np.ndarray, yv: np.ndarray, min_pairs: int) -> float | None:
    finite = np.isfinite(xv) & np.isfinite(yv)
    if int(finite.sum()) < min_pairs:
        return None
    x = xv[finite]
    y = yv[finite]
    sx,sy=np.max(np.abs(x)),np.max(np.abs(y))
    if sx==0 or sy==0:
        return None
    x,y=x/sx,y/sy
    if np.var(x)<=0 or np.var(y)<=0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _bures_shift_series(xv: np.ndarray, yv: np.ndarray, recent: int, prior: int, min_pairs: int) -> float:
    n = xv.shape[0]
    r = int(recent)
    p = int(prior)
    if r < 2 or p < 2 or n < r + p:
        return np.nan
    rho_r = _corr_window(xv[n - r :], yv[n - r :], min_pairs)
    rho_p = _corr_window(xv[n - r - p : n - r], yv[n - r - p : n - r], min_pairs)
    if rho_r is None or rho_p is None:
        return np.nan
    rr = np.array([[1.0, rho_r], [rho_r, 1.0]], dtype=float)
    rp = np.array([[1.0, rho_p], [rho_p, 1.0]], dtype=float)
    rr = (1.0 - _LAM) * rr + _LAM * np.eye(2)
    rp = (1.0 - _LAM) * rp + _LAM * np.eye(2)
    d2 = _bures2(rr, rp)
    return float(np.sqrt(max(d2, 0.0)))


@register_operator(
    name="ts_bures_corr_shift",
    category="structure_shift",
    business_category="structure_shift",
    canonical="ts_bures_corr_shift",
    source="advanced_structure",
)
class TsBuresCorrShift(SeriesOperator):
    """Bures-Wasserstein 相关结构漂移（不重叠 recent/prior 窗口）。

    ``recent = [t-rw+1, t]``、``prior = [t-rw-pw+1, t-rw]`` 两个完全不重叠的
    窗口各估计一个 2x2 相关系数矩阵（5% shrinkage 到单位阵），输出两者在
    PSD 流形上的 Bures 距离。检测价量/收益-波动/基本面-价格关系突变。
    """

    metadata = _metadata(
        "ts_bures_corr_shift",
        "Bures-Wasserstein 相关结构漂移（recent vs prior 不重叠窗口）。",
        ["x", "y", "recent_window", "prior_window", "min_pairs"],
        domain="price_volume",
        unit="distance",
        cost=5,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        recent_window: int = 20,
        prior_window: int = 60,
        min_pairs: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        r = strict_integer(recent_window,"recent_window",minimum=2)
        p = strict_integer(prior_window,"prior_window",minimum=2)
        if r < 2 or p < 2:
            raise ValueError("ts_bures_corr_shift requires recent_window, prior_window >= 2")
        # A 2-point correlation is ±1 almost surely -> spurious "breaks"; require a
        # statistically meaningful paired-sample count in both windows.
        if min_pairs is None:
            mp = max(5, int(min(r, p) // 2))
        else:
            mp = strict_integer(min_pairs,"min_pairs",minimum=2)
        if mp>min(r,p):
            raise ValueError("min_pairs must fit both recent and prior windows")
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - (r + p) + 1)
                if row - start + 1 < r + p:
                    continue
                out[row, col] = _bures_shift_series(xv[start : row + 1, col], yv[start : row + 1, col], r, p, mp)
        return _frame_like(x, out)


def _km_series(vals_2d: np.ndarray, window: int, bins: int, lag: int, order: int, min_bin_count: int) -> np.ndarray:
    """Kramers-Moyal D1/D2 evaluated at the current state (shared kernel).

    2026-08 V3 completion (spec §三): the coefficients come from the shared
    ``DiscreteStateDynamicsKernel`` — deterministic *quantile* state bins over the
    strictly-past window ``[t-W, t-1]``, a per-bin ``min_bin_count`` gate, lagged
    increments ``dx = x[s+lag]-x[s]``, and the strict Kramers-Moyal scaling
    ``D2 = mean(dx^2)/(2*lag)``.  The current value only selects its state bin;
    it never enters the historical estimates.
    """
    from factor_engine.cleaned_operators.markov_dynamics import _state_dynamics_series

    rows, cols = vals_2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        res = _state_dynamics_series(
            vals_2d[:, col], int(window), int(bins), int(lag), int(min_bin_count),
            min_state_support=int(min_bin_count)
        )
        d = res["D1"] if order == 1 else res["D2"]
        for t in range(rows):
            k = res["state"][t]
            if not np.isfinite(k):
                continue
            k = int(k)
            v = d[t, k]
            if np.isfinite(v):
                out[t, col] = v
    return out


@register_operator(
    name="ts_kramers_moyal_drift",
    category="structure_shift",
    business_category="structure_shift",
    canonical="ts_kramers_moyal_drift",
    source="advanced_structure",
)
class TsKramersMoyalDrift(SeriesOperator):
    """Kramers-Moyal 一阶系数（局部漂移）D1，在当前状态下取值。

    用共享状态动力学核：严格过去窗口 ``[t-W, t-1]`` 的分位数状态分箱，滞后
    ``lag`` 增量 ``dx = x[s+lag]-x[s]``，每 bin ``min_bin_count`` 门槛；D1(bin)
    = mean(dx)/lag。输出当前状态所在 bin 的 D1。确定性。P1。
    """

    metadata = _metadata(
        "ts_kramers_moyal_drift",
        "Kramers-Moyal D1 漂移（分位数状态 bin + lag 增量）。",
        ["x", "window", "bins", "lag", "min_bin_count"],
        domain="price_volume",
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 8, lag: int = 1, min_bin_count: Any = None, **_: Any
    ) -> pd.DataFrame:
        w = strict_integer(window,"window",minimum=10)
        b = strict_integer(bins,"bins",minimum=2)
        lg = strict_integer(lag,"lag",minimum=1)
        if b < 2:
            raise ValueError("ts_kramers_moyal_drift requires bins >= 2")
        if w <= lg:
            raise ValueError("ts_kramers_moyal_drift requires window > lag")
        if min_bin_count is None:
            mbc = max(3, int(np.ceil(w / b * 0.25)))
        else:
            mbc = strict_integer(min_bin_count,"min_bin_count",minimum=1)
        if mbc>w-lg:
            raise ValueError("min_bin_count must fit available lagged transitions")
        return _frame_like(x, _km_series(x.to_numpy(dtype=float), w, b, lg, 1, mbc))


@register_operator(
    name="ts_kramers_moyal_diffusion",
    category="structure_shift",
    business_category="structure_shift",
    canonical="ts_kramers_moyal_diffusion",
    source="advanced_structure",
)
class TsKramersMoyalDiffusion(SeriesOperator):
    """Kramers-Moyal 二阶系数（局部扩散）D2，在当前状态下取值。

    与 drift 共享同一分位数状态估计与 lag 增量；严格 KM 缩放
    ``D2 = mean(dx^2)/(2*lag)``。输出当前状态所在 bin 的 D2。确定性。P1。
    """

    metadata = _metadata(
        "ts_kramers_moyal_diffusion",
        "Kramers-Moyal D2 扩散（mean(dx^2)/(2*lag)）。",
        ["x", "window", "bins", "lag", "min_bin_count"],
        domain="price_volume",
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bins: int = 8, lag: int = 1, min_bin_count: Any = None, **_: Any
    ) -> pd.DataFrame:
        w = strict_integer(window,"window",minimum=10)
        b = strict_integer(bins,"bins",minimum=2)
        if b < 2:
            raise ValueError("ts_kramers_moyal_diffusion requires bins >= 2")
        lg = strict_integer(lag,"lag",minimum=1)
        if w <= lg:
            raise ValueError("ts_kramers_moyal_diffusion requires window > lag")
        if min_bin_count is None:
            mbc = max(3, int(np.ceil(w / b * 0.25)))
        else:
            mbc = strict_integer(min_bin_count,"min_bin_count",minimum=1)
        if mbc>w-lg:
            raise ValueError("min_bin_count must fit available lagged transitions")
        return _frame_like(x, _km_series(x.to_numpy(dtype=float), w, b, lg, 2, mbc))


def _average_rank_masked(col: np.ndarray) -> tuple[np.ndarray, int]:
    """Average-tie ranks (rankdata 'average') over the finite subset.

    Competition ranks (``argsort(argsort(...))``) assign arbitrary distinct ranks
    to tied values, which makes the copula sensitive to the stock-column order —
    a stock's rank would depend on which of its equal-valued peers sorts first.
    """
    fin = np.isfinite(col)
    m = int(fin.sum())
    ranks = np.full(col.shape[0], np.nan, dtype=float)
    if m == 0:
        return ranks, 0
    sub = col[fin]
    order = np.argsort(sub, kind="mergesort")
    sorted_sub = sub[order]
    avg = np.empty(m, dtype=float)
    i = 0
    while i < m:
        j = i
        while j + 1 < m and sorted_sub[j + 1] == sorted_sub[i]:
            j += 1
        avg[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    ranks[fin] = avg
    return ranks, m


def _rank_copula_proj(row: np.ndarray, thetas: np.ndarray) -> np.ndarray | None:
    """Per-date cross-sectional rank copula -> fixed-direction projections."""
    n, d = row.shape
    u = np.full((n, d), np.nan, dtype=float)
    for k in range(d):
        col = row[:, k]
        ranks, m = _average_rank_masked(col)
        if m < 2:
            continue
        u[np.isfinite(col), k] = (ranks[np.isfinite(col)] + 0.5) / m
    ok = np.isfinite(u).all(axis=1)
    if int(ok.sum()) < 2:
        return None
    return u[ok] @ thetas.T


def _sw_copula_shift(projs: list[np.ndarray | None], t: int, window: int, directions: int) -> float:
    cur = projs[t]
    ref: list[np.ndarray] = [p for p in projs[max(0, t - window) : t] if p is not None]
    if cur is None or not ref:
        return np.nan
    if cur.shape[0] < 2:
        return np.nan
    qgrid = np.linspace(0.01, 0.99, 99)
    dists: list[float] = []
    for l in range(directions):
        qc = np.quantile(cur[:, l], qgrid)
        # Date-equal-weight reference: average of each historical day's own
        # quantile curve, NOT the pooled-stocks quantile — pooling weights days
        # with many listed stocks (e.g. 5000) far more than sparse early days
        # (e.g. 2000), which fabricates regime shifts as the universe grows.
        curves = [np.quantile(p[:, l], qgrid) for p in ref if p.shape[0] >= 2]
        if not curves:
            return np.nan
        qr = np.mean(np.stack(curves), axis=0)
        dists.append(float(np.mean(np.abs(qc - qr))))
    return float(np.mean(dists)) if dists else np.nan


@register_operator(
    name="cs_sliced_wasserstein_copula_shift",
    category="structure_shift",
    business_category="structure_shift",
    canonical="cs_sliced_wasserstein_copula_shift",
    source="advanced_structure",
)
class CsSlicedWassersteinCopulaShift(SeriesOperator):
    """Sliced-Wasserstein 秩 copula 结构漂移（市场/行业 regime state）。

    每个特征做截面秩变换（marginal->uniform），用固定种子方向的单位向量投影，
    比较当日投影分布与过去 ``window`` 日参考投影分布的 Sliced-Wasserstein
    距离（W1 在均匀分位网格上，参考取各历史日分位曲线的等权平均）。当日所有
    股票得到同一 regime 值（适合作为 state gate / where / multiply 输入）。
    这是 GLOBAL_STATE：当天截面完全无区分度，**禁止当普通个股横截面 alpha
    单独挖**，只能用于 regime 交互。确定性（固定种子 + average-tie rank，
    对股票列顺序不变）。P1。
    """

    metadata = _metadata(
        "cs_sliced_wasserstein_copula_shift",
        "Sliced-Wasserstein 秩 copula 结构漂移（GLOBAL_STATE regime，全市场同值）。",
        ["f1", "f2", "f3", "window", "directions"],
        domain="price_volume",
        unit="distance",
        cost=4,
        extra_tags=("global_state",),
    )

    def _calculate_series(
        self,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        window: int = 60,
        directions: int = 32,
        **_: Any,
    ) -> pd.DataFrame:
        w = strict_integer(window,"window",minimum=2)
        d_l = strict_integer(directions,"directions",minimum=4)
        if w < 2 or d_l < 4:
            raise ValueError("cs_sliced_wasserstein_copula_shift requires window >= 2, directions >= 4")
        feats = np.stack(
            [f1.to_numpy(dtype=float), f2.to_numpy(dtype=float), f3.to_numpy(dtype=float)], axis=2
        )
        rng = np.random.default_rng(_SEED)
        thetas = rng.normal(0.0, 1.0, size=(d_l, feats.shape[2]))
        thetas /= np.maximum(np.linalg.norm(thetas, axis=1, keepdims=True), _EPS)
        projs: list[np.ndarray | None] = [_rank_copula_proj(feats[t], thetas) for t in range(feats.shape[0])]
        out = np.full(feats.shape[0], np.nan, dtype=float)
        for t in range(feats.shape[0]):
            out[t] = _sw_copula_shift(projs, t, w, d_l)
        return _frame_like(f1, np.repeat(out[:, None], f1.shape[1], axis=1))


def _robust_z_col(col: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(col)
    if int(finite.sum()) < 2:
        return None
    values=col[finite]
    scale=np.max(np.abs(values))
    if scale==0:
        return None
    values=values/scale
    med=float(np.median(values))
    mad=float(np.median(np.abs(values-med)))*1.4826
    spread=mad if mad>0 else float(np.std(values))
    if spread<=0:
        return None
    out=np.full_like(col,np.nan,dtype=float)
    out[finite]=(values-med)/spread
    return out


def _spd_log_matrix(feat_block: np.ndarray, d: int, lam: float, min_rows: int) -> np.ndarray | None:
    """Robust-z features -> regularized correlation -> log-Euclidean 3x3."""
    z: list[np.ndarray] = []
    for k in range(d):
        zk = _robust_z_col(feat_block[:, k])
        if zk is None:
            return None
        z.append(zk)
    Z = np.stack(z, axis=1)
    # Complete-case: one stock's one missing feature must not NaN the whole
    # group's covariance.  Drop incomplete rows first, then require enough peers.
    complete = np.isfinite(Z).all(axis=1)
    if int(complete.sum()) < min_rows:
        return None
    Z = Z[complete]
    cov = np.cov(Z, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    sd = np.sqrt(np.maximum(np.diag(cov), _EPS))
    corr = cov / np.outer(sd, sd)
    corr = np.clip(corr, -1.0, 1.0)
    r = (1.0 - lam) * corr + lam * np.eye(d)
    w, v = np.linalg.eigh(r)
    w = np.clip(w, 1e-10, None)
    return (v * np.log(w)) @ v.T


def _group_spd_shift(
    feats: np.ndarray,
    group: np.ndarray,
    t: int,
    ref_window: int,
    log_by_date: list[dict[Any, tuple[np.ndarray, np.ndarray]]],
    d: int,
    lam: float,
    min_peers: int,
    min_reference_days: int,
    composition_policy: str,
) -> np.ndarray:
    row = feats[t]
    g_row = group[t]
    labels = pd.unique(g_row)
    out = np.full(row.shape[0], np.nan, dtype=float)
    for label in labels:
        if not _valid_group_label(label):
            continue
        idx = np.flatnonzero(g_row == label)
        if idx.size < min_peers:
            continue
        entry = log_by_date[t].get(label)
        if entry is None:
            continue
        ref_days = [
            s for s in range(max(0, t - ref_window), t)
            if log_by_date[s].get(label) is not None
        ]
        if len(ref_days) < min_reference_days:
            continue
        if composition_policy == "intersection":
            # Restrict to members present on the current day AND every reference
            # day, recomputing both matrices on that common member set, so a pure
            # membership change cannot masquerade as a dependence-structure shift.
            shared = set(int(i) for i in idx)
            for s in ref_days:
                shared &= set(int(i) for i in log_by_date[s][label][1])
            if len(shared) < min_peers:
                continue
            cur_idx = np.array(sorted(shared), dtype=int)
            cur = _spd_log_matrix(row[cur_idx], d, lam, min_peers)
            refs = [
                _spd_log_matrix(feats[s][list(shared)], d, lam, min_peers)
                for s in ref_days
            ]
            refs = [m for m in refs if m is not None]
            if len(refs) < min_reference_days:
                continue
            ref = np.mean(np.stack(refs), axis=0)
        else:  # "current": the group as observed today (default).
            cur = entry[0]
            refs = [log_by_date[s][label][0] for s in ref_days]
            ref = np.mean(np.stack(refs), axis=0)
            cur_idx = idx
        if cur is None or not np.isfinite(cur).all():
            continue
        out[cur_idx] = float(np.linalg.norm(cur - ref, ord="fro"))
    return out


@register_operator(
    name="group_spd_feature_structure_shift",
    category="structure_shift",
    business_category="structure_shift",
    canonical="group_spd_feature_structure_shift",
    source="advanced_structure",
)
class GroupSpdFeatureStructureShift(SeriesOperator):
    """组内 SPD 相关结构漂移（log-Euclidean Frobenius 距离，恰好 3 个特征）。

    每组股票在组内对**恰好 3 个**特征（f1,f2,f3）做稳健 z（complete-case，
    剔除缺测行），估计特征相关矩阵（5% 正则化），取 log-Euclidean，与过去
    ``reference_window`` 日同组的历史平均 log 矩阵比较 Frobenius 距离。组内
    股票得到该组的结构漂移值。``composition_policy="intersection"`` 时只在与
    每个参考日共同存在的成员上重算矩阵，剔除纯成员变化效应。确定性。
    """

    metadata = _metadata(
        "group_spd_feature_structure_shift",
        "组内 SPD(log-Euclidean) 特征相关结构漂移（恰好 3 特征）。",
        ["f1", "f2", "f3", "group", "reference_window", "min_peers", "min_reference_days", "composition_policy"],
        domain="price_volume",
        unit="distance",
        cost=5,
    )

    def _calculate_series(
        self,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        group: pd.DataFrame,
        reference_window: int = 60,
        min_peers: Any = None,
        min_reference_days: Any = None,
        composition_policy: str = "current",
        **_: Any,
    ) -> pd.DataFrame:
        rw = strict_integer(reference_window,"reference_window",minimum=2)
        if rw < 2:
            raise ValueError("group_spd_feature_structure_shift requires reference_window >= 2")
        if composition_policy not in ("current", "intersection"):
            raise ValueError("group_spd_feature_structure_shift composition_policy must be 'current' or 'intersection'")
        # A 3x3 correlation needs >> 3 peers; 2 stocks estimating 3 features is a
        # degenerate covariance (the 5% shrinkage cannot create information).
        if min_peers is None:
            mp = 5
        else:
            mp = strict_integer(min_peers,"min_peers",minimum=3)
        if min_reference_days is None:
            mrd = 5
        else:
            mrd = strict_integer(min_reference_days,"min_reference_days",minimum=2)
        if mrd>rw:
            raise ValueError("min_reference_days must be <= reference_window")
        feats = np.stack(
            [f1.to_numpy(dtype=float), f2.to_numpy(dtype=float), f3.to_numpy(dtype=float)], axis=2
        )
        gv = group.to_numpy()
        d = feats.shape[2]
        log_by_date: list[dict[Any, tuple[np.ndarray, np.ndarray]]] = []
        for t in range(feats.shape[0]):
            row = feats[t]
            g_row = gv[t]
            day: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
            for label in pd.unique(g_row):
                if not _valid_group_label(label):
                    continue
                idx = np.flatnonzero(g_row == label)
                if idx.size < mp:
                    continue
                m = _spd_log_matrix(row[idx], d, _LAM, mp)
                if m is not None:
                    day[label] = (m, idx)
            log_by_date.append(day)
        out = np.full_like(feats[:, :, 0], np.nan, dtype=float)
        for t in range(feats.shape[0]):
            out[t] = _group_spd_shift(
                feats, gv, t, rw, log_by_date, d, _LAM, mp, mrd, composition_policy
            )
        return _frame_like(f1, out)


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    ok = m > 0.0
    kl_pm = 0.0
    kl_qm = 0.0
    if ok.any():
        # KL with the 0 log 0 = 0 convention; evaluated only where m > 0 so the
        # log never sees an all-zero denominator (no eager -inf*0 warnings).
        pm = p[ok]
        qm = q[ok]
        mm = m[ok]
        # 0 log 0 = 0; compute log only where the numerator is positive so the
        # eager log never sees a zero (no -inf*0 warnings).
        rp = np.divide(pm, mm, out=np.ones_like(pm), where=pm > 0.0)
        rq = np.divide(qm, mm, out=np.ones_like(qm), where=qm > 0.0)
        kl_pm = float(np.sum(pm * np.log(rp)))
        kl_qm = float(np.sum(qm * np.log(rq)))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(max(js, 0.0) / _LN2))


def _holder_js_cell(current: np.ndarray, previous: np.ndarray) -> float:
    """JS distance over the *fixed* 5 class slots (identity preserved).

    A slot is ``0`` when that class holds no shares; ``NaN`` means the class is
    unknown for this disclosure.  Unknown is never silently treated as 0: if a
    slot is NaN while the distribution otherwise carries known mass, the cell
    fails closed (NaN) rather than shifting the remaining slots left — the
    original implementation compressed ``current[np.isfinite(current)]``, which
    changed class identity (class3 -> class2) and made the distance meaningless
    whenever the two disclosures had different missing patterns.
    """
    cur = np.asarray(current, dtype=float)
    prev = np.asarray(previous, dtype=float)
    if cur.shape[0] < 5 or prev.shape[0] < 5:
        return np.nan
    if (cur < 0.0).any() or (prev < 0.0).any():
        return np.nan  # negative shares are invalid data -> fail closed.
    if (cur > 0.0).any() and not np.isfinite(cur).all():
        return np.nan  # partial unknown class list -> not comparable.
    if (prev > 0.0).any() and not np.isfinite(prev).all():
        return np.nan
    cu = np.where(np.isfinite(cur), cur, 0.0)
    pv = np.where(np.isfinite(prev), prev, 0.0)
    sc = float(cu.sum())
    sp = float(pv.sum())
    if sc <= _EPS or sp <= _EPS:
        return np.nan
    return _js_distance(cu / sc, pv / sp)


@register_operator(
    name="holder_class_js_shift",
    category="structure_shift",
    business_category="structure_shift",
    canonical="holder_class_js_shift",
    source="advanced_structure",
)
class HolderClassJsShift(SeriesOperator):
    """股东类别分布 JS 漂移（当前 vs 上一有效披露）。

    输入为已按股东类别聚合的日频面板（当前类别的 5 个份额槽 s1..s5、上一披露
    的 5 个份额槽 ps1..ps5，来源层 materialize）。逐格归一化后计算
    JS 距离并归一化到 [0,1]。仅依赖 PubDate 可见快照，PIT 安全。
    """

    metadata = _metadata(
        "holder_class_js_shift",
        "股东类别分布 JS 距离 [0,1]（当前 vs 上一披露快照）。",
        ["s1", "s2", "s3", "s4", "s5", "ps1", "ps2", "ps3", "ps4", "ps5"],
        domain="shareholder",
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self,
        s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, s4: pd.DataFrame, s5: pd.DataFrame,
        ps1: pd.DataFrame, ps2: pd.DataFrame, ps3: pd.DataFrame, ps4: pd.DataFrame, ps5: pd.DataFrame,
        **_: Any,
    ) -> pd.DataFrame:
        cur = np.stack([s.to_numpy(dtype=float) for s in (s1, s2, s3, s4, s5)], axis=2)
        prev = np.stack([p.to_numpy(dtype=float) for p in (ps1, ps2, ps3, ps4, ps5)], axis=2)
        rows, cols, _ = cur.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for i in range(rows):
            for j in range(cols):
                out[i, j] = _holder_js_cell(cur[i, j], prev[i, j])
        return _frame_like(s1, out)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_bures_corr_shift",
            "ts_kramers_moyal_drift",
            "ts_kramers_moyal_diffusion",
            "cs_sliced_wasserstein_copula_shift",
            "group_spd_feature_structure_shift",
            "holder_class_js_shift",
        })


_register_surface()
