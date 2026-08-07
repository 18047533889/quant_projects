# -*- coding: utf-8 -*-
"""Advanced dependence-structure operators (2026-08 Gemini round, P1).

Deterministic, prefix-causal pandas-numpy references returning
``(TradeDate x Symbol)`` panels.

* ``ts_bures_corr_shift``                    — Bures-Wasserstein distance between
  two non-overlapping correlation windows (2x2, 5% shrinkage) — a PIT-safe
  price-volume / return-volatility relationship break detector.
* ``ts_kramers_moyal_drift`` / ``diffusion`` — state-conditioned first / second
  Kramers-Moyal coefficients (local drift / diffusion) evaluated at the current
  state.  Deterministic equal-width state bins.
* ``cs_sliced_wasserstein_copula_shift``     — Sliced-Wasserstein distance
  between today's cross-sectional rank copula (fixed seeded directions) and the
  trailing reference — a market/group regime state, broadcast to every stock.
* ``group_spd_feature_structure_shift``      — per-group log-Euclidean SPD
  correlation-structure shift (2..4 features), one value per stock.
* ``holder_class_js_shift``                  — Jensen-Shannon distance between
  current and previous holder-class share distributions (5 class slots).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_LAM = 0.05              # fixed correlation shrinkage / SPD regularization.
_SEED = 20260807         # fixed seed -> deterministic projection directions.
_LN2 = float(np.log(2.0))


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="structure_shift",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "structure_shift", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
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


def _corr_window(xv: np.ndarray, yv: np.ndarray) -> float | None:
    finite = np.isfinite(xv) & np.isfinite(yv)
    if int(finite.sum()) < 2:
        return None
    x = xv[finite]
    y = yv[finite]
    vx = float(np.var(x))
    vy = float(np.var(y))
    if vx <= _EPS or vy <= _EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _bures_shift_series(xv: np.ndarray, yv: np.ndarray, recent: int, prior: int) -> float:
    n = xv.shape[0]
    r = int(recent)
    p = int(prior)
    if r < 2 or p < 2 or n < r + p:
        return np.nan
    rho_r = _corr_window(xv[n - r :], yv[n - r :])
    rho_p = _corr_window(xv[n - r - p : n - r], yv[n - r - p : n - r])
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
        ["x", "y", "recent_window", "prior_window"],
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
        **_: Any,
    ) -> pd.DataFrame:
        r = int(recent_window)
        p = int(prior_window)
        if r < 2 or p < 2:
            raise ValueError("ts_bures_corr_shift requires recent_window, prior_window >= 2")
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - (r + p) + 1)
                if row - start + 1 < r + p:
                    continue
                out[row, col] = _bures_shift_series(xv[start : row + 1, col], yv[start : row + 1, col], r, p)
        return _frame_like(x, out)


def _km_coeff(vals: np.ndarray, bins: int, order: int) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size < 3:
        return np.nan
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo <= _EPS:
        return np.nan
    edges = np.linspace(lo, hi, int(bins) + 1)
    cur = float(vals[-1])
    if not np.isfinite(cur):
        return np.nan
    cb = int(np.clip(np.searchsorted(edges, cur, side="right") - 1, 0, int(bins) - 1))
    inc = np.diff(vals)  # inc[s] = x[s+1] - x[s], base state x[s]
    bases = vals[:-1]
    base_bin = np.clip(np.searchsorted(edges, bases, side="right") - 1, 0, int(bins) - 1)
    ok = np.isfinite(inc)
    sel = ok & (base_bin == cb)
    if not sel.any():
        return np.nan
    if order == 1:
        return float(np.mean(inc[sel]))
    return float(np.mean(inc[sel] ** 2))


def _km_series(vals_2d: np.ndarray, window: int, bins: int, order: int) -> np.ndarray:
    rows, cols = vals_2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            chunk = vals_2d[start : row + 1, col]
            if chunk.shape[0] < 3:
                continue
            out[row, col] = _km_coeff(chunk, bins, order)
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

    窗口内将状态值等宽分箱为 ``bins`` 档，对每个状态 bin 收集条件增量
    ``dx = x[s+1]-x[s]``，D1(bin) = mean(dx)。输出当前状态所在 bin 的 D1。
    确定性（等宽分箱、无随机）。P1。
    """

    metadata = _metadata(
        "ts_kramers_moyal_drift",
        "Kramers-Moyal D1 漂移（当前状态的局部条件增量均值）。",
        ["x", "window", "bins"],
        domain="price_volume",
        unit="ratio",
        cost=5,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, bins: int = 8, **_: Any) -> pd.DataFrame:
        w = int(window)
        b = int(bins)
        if b < 2:
            raise ValueError("ts_kramers_moyal_drift requires bins >= 2")
        return _frame_like(x, _km_series(x.to_numpy(dtype=float), w, b, 1))


@register_operator(
    name="ts_kramers_moyal_diffusion",
    category="structure_shift",
    business_category="structure_shift",
    canonical="ts_kramers_moyal_diffusion",
    source="advanced_structure",
)
class TsKramersMoyalDiffusion(SeriesOperator):
    """Kramers-Moyal 二阶系数（局部扩散）D2，在当前状态下取值。

    与 drift 同分箱；D2(bin) = mean(dx^2)。输出当前状态所在 bin 的 D2。
    确定性。P1。
    """

    metadata = _metadata(
        "ts_kramers_moyal_diffusion",
        "Kramers-Moyal D2 扩散（当前状态的局部条件增量平方均值）。",
        ["x", "window", "bins"],
        domain="price_volume",
        unit="ratio",
        cost=5,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, bins: int = 8, **_: Any) -> pd.DataFrame:
        w = int(window)
        b = int(bins)
        if b < 2:
            raise ValueError("ts_kramers_moyal_diffusion requires bins >= 2")
        return _frame_like(x, _km_series(x.to_numpy(dtype=float), w, b, 2))


def _rank_copula_proj(row: np.ndarray, thetas: np.ndarray) -> np.ndarray | None:
    """Per-date cross-sectional rank copula -> fixed-direction projections."""
    n, d = row.shape
    u = np.full((n, d), np.nan, dtype=float)
    for k in range(d):
        col = row[:, k]
        fin = np.isfinite(col)
        m = int(fin.sum())
        if m < 2:
            continue
        ranks = np.argsort(np.argsort(col[fin])).astype(np.float64)
        u[fin, k] = (ranks + 0.5) / m
    ok = np.isfinite(u).all(axis=1)
    if int(ok.sum()) < 2:
        return None
    return u[ok] @ thetas.T


def _sw_copula_shift(projs: list[np.ndarray | None], t: int, window: int, directions: int) -> float:
    cur = projs[t]
    ref: list[np.ndarray] = [p for p in projs[max(0, t - window) : t] if p is not None]
    if cur is None or not ref:
        return np.nan
    refall = np.concatenate(ref, axis=0)
    if refall.shape[0] < 5 or cur.shape[0] < 2:
        return np.nan
    qgrid = np.linspace(0.01, 0.99, 99)
    dists: list[float] = []
    for l in range(directions):
        qc = np.quantile(cur[:, l], qgrid)
        qr = np.quantile(refall[:, l], qgrid)
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
    距离（W1 在均匀分位网格上）。当日所有股票得到同一 regime 值（适合作为
    state gate / where / multiply 输入）。确定性（固定种子）。
    """

    metadata = _metadata(
        "cs_sliced_wasserstein_copula_shift",
        "Sliced-Wasserstein 秩 copula 结构漂移（regime state，全市场同值）。",
        ["f1", "f2", "f3", "window", "directions"],
        domain="price_volume",
        unit="distance",
        cost=4,
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
        w = int(window)
        d_l = int(directions)
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
    med = float(np.median(col[finite]))
    mad = float(np.median(np.abs(col[finite] - med))) * 1.4826
    spread = mad if mad > _EPS else float(np.std(col[finite]))
    if spread <= _EPS:
        return None
    out = np.full_like(col, np.nan, dtype=float)
    out[finite] = (col[finite] - med) / spread
    return out


def _spd_log_matrix(feat_block: np.ndarray, d: int, lam: float) -> np.ndarray | None:
    """Robust-z features -> regularized correlation -> log-Euclidean 3x3."""
    z: list[np.ndarray] = []
    for k in range(d):
        zk = _robust_z_col(feat_block[:, k])
        if zk is None:
            return None
        z.append(zk)
    Z = np.stack(z, axis=1)
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
    log_by_date: list[dict[Any, np.ndarray]],
    d: int,
) -> np.ndarray:
    row = feats[t]
    g_row = group[t]
    labels = pd.unique(g_row)
    out = np.full(row.shape[0], np.nan, dtype=float)
    for label in labels:
        idx = np.flatnonzero(g_row == label)
        cur = log_by_date[t].get(label)
        if cur is None:
            continue
        refs = [
            log_by_date[s].get(label)
            for s in range(max(0, t - ref_window), t)
            if log_by_date[s].get(label) is not None
        ]
        if not refs:
            continue
        ref = np.mean(np.stack(refs), axis=0)
        out[idx] = float(np.linalg.norm(cur - ref, ord="fro"))
    return out


@register_operator(
    name="group_spd_feature_structure_shift",
    category="structure_shift",
    business_category="structure_shift",
    canonical="group_spd_feature_structure_shift",
    source="advanced_structure",
)
class GroupSpdFeatureStructureShift(SeriesOperator):
    """组内 SPD 相关结构漂移（log-Euclidean Frobenius 距离）。

    每组股票在组内对 2..4 个特征做稳健 z，估计特征相关矩阵（5% 正则化），
    取 log-Euclidean，与过去 ``reference_window`` 日同组的历史平均 log 矩阵
    比较 Frobenius 距离。组内股票得到该组的结构漂移值。确定性。
    """

    metadata = _metadata(
        "group_spd_feature_structure_shift",
        "组内 SPD(log-Euclidean) 特征相关结构漂移。",
        ["f1", "f2", "f3", "group", "reference_window"],
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
        **_: Any,
    ) -> pd.DataFrame:
        rw = int(reference_window)
        if rw < 2:
            raise ValueError("group_spd_feature_structure_shift requires reference_window >= 2")
        feats = np.stack(
            [f1.to_numpy(dtype=float), f2.to_numpy(dtype=float), f3.to_numpy(dtype=float)], axis=2
        )
        gv = group.to_numpy()
        d = feats.shape[2]
        log_by_date: list[dict[Any, np.ndarray]] = []
        for t in range(feats.shape[0]):
            row = feats[t]
            g_row = gv[t]
            day: dict[Any, np.ndarray] = {}
            for label in pd.unique(g_row):
                idx = np.flatnonzero(g_row == label)
                if idx.size < 2:
                    continue
                m = _spd_log_matrix(row[idx], d, _LAM)
                if m is not None:
                    day[label] = m
            log_by_date.append(day)
        out = np.full_like(feats[:, :, 0], np.nan, dtype=float)
        for t in range(feats.shape[0]):
            out[t] = _group_spd_shift(feats, gv, t, rw, log_by_date, d)
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
    cur = current[np.isfinite(current)]
    prev = previous[np.isfinite(previous)]
    if cur.size == 0 or prev.size == 0:
        return np.nan
    sc = float(cur.sum())
    sp = float(prev.sum())
    if sc <= _EPS or sp <= _EPS:
        return np.nan
    p = np.zeros(5, dtype=float)
    q = np.zeros(5, dtype=float)
    for k in range(min(5, cur.size)):
        p[k] = cur[k] / sc
    for k in range(min(5, prev.size)):
        q[k] = prev[k] / sp
    return _js_distance(p, q)


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
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "ts_bures_corr_shift",
            "ts_kramers_moyal_drift",
            "ts_kramers_moyal_diffusion",
            "cs_sliced_wasserstein_copula_shift",
            "group_spd_feature_structure_shift",
            "holder_class_js_shift",
        }
    )


_register_surface()
