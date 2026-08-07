# -*- coding: utf-8 -*-
"""Topological / information-geometry operators (2026-08 Gemini round, P2).

All operators are deterministic, prefix-causal pandas-numpy references.
``gudhi``/``ripser`` are not runtime dependencies: the Vietoris-Rips H1
persistence used here is a compact pure-numpy reduction over F2 (bitmask
columns) with a hard deterministic point-cloud cap, so repeated evaluation is
bit-identical and cost stays bounded for automatic search.

* ``ts_betti_1_max_persistence``  — max H1 persistence of the Takens-embedded
  rolling window (dimensionless, median-distance normalised).  P2/Research.
* ``ts_persistence_diagram_shift`` — 1D Wasserstein distance between the H1
  persistence multisets of the current and the previous window (a topology
  regime-change proxy, cheaper than a full CROCKER).  P2/Research.
* ``ts_student_t_fisher_shift``   — empirical-Fisher (information-geometry)
  distance between two non-overlapping windows under a Student-t family,
  parameterised by ``(mu, log sigma, log(nu-2))``.  P2/Research.
"""
from __future__ import annotations

from math import lgamma, log as mlog, sqrt as msqrt
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_MAX_POINTS = 12              # hard deterministic cap on the Rips point cloud.
_DF_GRID = (2.1, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)
_DF_STEP = 0.05


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="topology",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "topology", "daily", "pit_safe", "causal", "typed_v2", "deterministic",
            "research_only",
            f"signature:{','.join(params)}->series", f"unit:{unit}", "cost:9",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# Vietoris-Rips H1 persistence (pure numpy, F2 bitmask reduction)
# ---------------------------------------------------------------------------

def _rips_h1_pairs(points: np.ndarray) -> list[tuple[float, float]]:
    """H1 persistence pairs ``(birth, death)`` of a Rips complex on ``points``."""
    n = points.shape[0]
    if n < 4:
        return []
    d = np.sqrt(np.maximum(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1), 0.0))
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(d[i, j]), i, j))
    edges.sort()
    edge_len: list[float] = [e[0] for e in edges]
    edge_idx: dict[tuple[int, int], int] = {}
    for idx, (_, i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    triangles: list[tuple[float, int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                flt = max(d[i, j], d[i, k], d[j, k])
                triangles.append((float(flt), i, j, k))
    triangles.sort()
    pivots: dict[int, int] = {}
    pairs: list[tuple[float, float]] = []
    for flt, i, j, k in triangles:
        col = (1 << edge_idx[(i, j)]) | (1 << edge_idx[(i, k)]) | (1 << edge_idx[(j, k)])
        while True:
            low = col & -col
            if low == 0:
                break
            p = low.bit_length() - 1
            prev = pivots.get(p)
            if prev is None:
                pivots[p] = col
                break
            col ^= prev
        if col == 0:
            continue  # region already filled -> no H1 death here.
        p = (col & -col).bit_length() - 1
        pairs.append((edge_len[p], flt))
    return pairs


def _max_persistence(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    return float(max(death - birth for birth, death in pairs))


def _takens_points(vals: np.ndarray, tau: int, dim: int) -> np.ndarray | None:
    """Robust-normalised Takens embedding, deduplicated and deterministically
    decimated to at most ``_MAX_POINTS`` points."""
    finite = vals[np.isfinite(vals)]
    if finite.size < 4:
        return None
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * 1.4826
    spread = mad if mad > _EPS else float(np.std(finite))
    if spread <= _EPS:
        return None
    z = (vals - med) / spread
    n = vals.shape[0]
    lag = tau * (dim - 1)
    if n < lag + 3:
        return None
    pts = np.stack([z[s - (dim - 1) * tau : s + 1 : tau] for s in range(lag, n)], axis=0)
    pts = np.round(pts, decimals=9)
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 4:
        return None
    if pts.shape[0] > _MAX_POINTS:
        keep = np.unique(
            np.round(np.linspace(0, pts.shape[0] - 1, _MAX_POINTS)).astype(np.int64)
        )
        pts = pts[keep]
    return pts


def _betti_series(vals_2d: np.ndarray, window: int, tau: int, dim: int) -> np.ndarray:
    rows, cols = vals_2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            chunk = vals_2d[start : row + 1, col]
            pts = _takens_points(chunk, tau, dim)
            if pts is None:
                continue
            pd_pairwise = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
            finite = pd_pairwise[np.isfinite(pd_pairwise)]
            if finite.size == 0 or float(np.median(finite)) <= _EPS:
                continue
            pairs = _rips_h1_pairs(pts)
            out[row, col] = _max_persistence(pairs) / float(np.median(finite))
    return out


@register_operator(
    name="ts_betti_1_max_persistence",
    category="topology",
    business_category="topology",
    canonical="ts_betti_1_max_persistence",
    source="advanced_topology",
    status="experimental",
)
class TsBetti1MaxPersistence(SeriesOperator):
    """滚动 Takens 嵌入的 H1 最大持久性（Rips filtration）。

    窗口稳健归一化（median/MAD）后做 Takens 嵌入（tau、embedding_dim），去重并
    确定性抽稀到 ≤12 点，纯 numpy 计算 Rips H1 持久对，输出
    ``max(death-birth)`` 再除以点云中位成对距离（无量纲）。无 H1 -> 0。
    绝不使用 random jitter。P2 / Research。
    """

    metadata = _metadata(
        "ts_betti_1_max_persistence",
        "滚动 H1 最大持久性（Takens+Rips，中位距离归一化，无量纲）。",
        ["x", "window", "tau", "embedding_dim"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        tau: int = 1,
        embedding_dim: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        t = int(tau)
        m = int(embedding_dim)
        if w < 6 or t < 1 or not (2 <= m <= 6):
            raise ValueError("ts_betti_1_max_persistence requires window>=6, tau>=1, 2<=embedding_dim<=6")
        return _frame_like(x, _betti_series(x.to_numpy(dtype=float), w, t, m))


def _diagram_persistence(pairs: list[tuple[float, float]]) -> np.ndarray:
    return np.array([death - birth for birth, death in pairs], dtype=float)


def _diagram_w1(p_current: np.ndarray, p_prior: np.ndarray) -> float:
    """1D Wasserstein between two H1 persistence multisets (pad with zeros)."""
    if p_current.size == 0 and p_prior.size == 0:
        return 0.0
    if p_current.size == 0:
        return float(np.mean(np.abs(p_prior)))
    if p_prior.size == 0:
        return float(np.mean(np.abs(p_current)))
    a = np.sort(np.abs(p_current))
    b = np.sort(np.abs(p_prior))
    k = min(a.size, b.size)
    total = float(np.sum(np.abs(a[:k] - b[:k])))
    if a.size > b.size:
        total += float(np.sum(np.abs(a[k:])))
    else:
        total += float(np.sum(np.abs(b[k:])))
    return total / max(a.size, b.size)


def _persistence_shift_series(vals_2d: np.ndarray, window: int, tau: int, dim: int) -> np.ndarray:
    rows, cols = vals_2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            cur_lo = max(0, row - window + 1)
            pri_lo = max(0, row - 2 * window + 1)
            cur_chunk = vals_2d[cur_lo : row + 1, col]
            pri_chunk = vals_2d[pri_lo:cur_lo, col]
            if pri_chunk.shape[0] < 6 or cur_chunk.shape[0] < 6:
                continue
            cur_pts = _takens_points(cur_chunk, tau, dim)
            pri_pts = _takens_points(pri_chunk, tau, dim)
            if cur_pts is None or pri_pts is None:
                continue
            cur_pairs = _rips_h1_pairs(cur_pts)
            pri_pairs = _rips_h1_pairs(pri_pts)
            out[row, col] = _diagram_w1(
                _diagram_persistence(cur_pairs), _diagram_persistence(pri_pairs)
            )
    return out


@register_operator(
    name="ts_persistence_diagram_shift",
    category="topology",
    business_category="topology",
    canonical="ts_persistence_diagram_shift",
    source="advanced_topology",
    status="experimental",
)
class TsPersistenceDiagramShift(SeriesOperator):
    """相空间拓扑 regime 漂移：当前窗口 H1 持久值与上一窗口的 1D Wasserstein。

    两个窗口均 Takens 嵌入 + Rips H1；输出两 H1 持久值多重集的 1D Wasserstein
    距离（短者补零，即匹配到对角线的代价）。这是 CROCKER 的低成本替代——
    回答"拓扑结构变了多少"，而非逐层 Betti 曲线。P2 / Research。
    """

    metadata = _metadata(
        "ts_persistence_diagram_shift",
        "当前 vs 上一窗口 H1 持久值多重集的 1D Wasserstein 距离。",
        ["x", "window", "tau", "embedding_dim"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        tau: int = 1,
        embedding_dim: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        t = int(tau)
        m = int(embedding_dim)
        if w < 6 or t < 1 or not (2 <= m <= 6):
            raise ValueError("ts_persistence_diagram_shift requires window>=6, tau>=1, 2<=embedding_dim<=6")
        return _frame_like(x, _persistence_shift_series(x.to_numpy(dtype=float), w, t, m))


# ---------------------------------------------------------------------------
# Student-t empirical Fisher shift
# ---------------------------------------------------------------------------

def _t_logpdf_const(df: float) -> float:
    return lgamma(0.5 * (df + 1.0)) - 0.5 * mlog(df * np.pi) - lgamma(0.5 * df)


def _t_logpdf_vec(v: np.ndarray, df: float, mu: float, scale: float) -> np.ndarray:
    z = (v - mu) / scale
    return _t_logpdf_const(df) - mlog(scale) - 0.5 * (df + 1.0) * np.log1p(z * z / df)


def _t_fit(vals: np.ndarray) -> tuple[float, float, float] | None:
    """Fast profile-likelihood Student-t fit over a fixed df grid (deterministic)."""
    v = vals[np.isfinite(vals)]
    if v.size < 8:
        return None
    best: tuple[float, float, float, float] | None = None
    for df in _DF_GRID:
        mu = float(np.median(v))
        mad = float(np.median(np.abs(v - mu))) * 1.4826
        scale = mad if mad > _EPS else float(np.std(v))
        if not (np.isfinite(scale) and scale > _EPS):
            return None
        for _ in range(15):
            z = (v - mu) / scale
            w = (df + 1.0) / (df + z * z)
            den = float(np.sum(w))
            if den <= _EPS:
                break
            mu = float(np.sum(w * v) / den)
            scale = msqrt(max(float(np.mean(w * (v - mu) ** 2)), 1e-12))
            if not (np.isfinite(scale) and scale > _EPS):
                return None
        ll = float(np.sum(_t_logpdf_vec(v, df, mu, scale)))
        if best is None or ll > best[0]:
            best = (ll, df, mu, scale)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _empirical_fisher(v: np.ndarray, df: float, mu: float, scale: float) -> np.ndarray:
    z = (v - mu) / scale
    w = (df + 1.0) / (df + z * z)
    g_mu = w * z / scale
    g_ls = df * (z * z - 1.0) / (df + z * z)          # d/d(log scale)
    llp = _t_logpdf_vec(v, df + _DF_STEP, mu, scale)
    llm = _t_logpdf_vec(v, df - _DF_STEP, mu, scale)
    g_df = ((llp - llm) / (2.0 * _DF_STEP)) * (df - 2.0)  # d/d(log(df-2))
    g = np.stack([g_mu, g_ls, g_df], axis=1)
    return (g.T @ g) / v.size + 1e-8 * np.eye(3)


def _matrix_log(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (m + m.T))
    w = np.clip(w, 1e-10, None)
    return (v * np.log(w)) @ v.T


def _fisher_shift_series(vals_2d: np.ndarray, recent: int, prior: int) -> np.ndarray:
    rows, cols = vals_2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            r = int(recent)
            p = int(prior)
            if row < r + p:  # keep both prior and recent windows non-negative.
                continue
            cur = vals_2d[row - r : row + 1, col]
            pri = vals_2d[row - r - p : row - r, col]
            fit_r = _t_fit(cur)
            fit_p = _t_fit(pri)
            if fit_r is None or fit_p is None:
                continue
            i_r = _empirical_fisher(cur, *fit_r)
            i_p = _empirical_fisher(pri, *fit_p)
            lr = _matrix_log(i_r)
            lp = _matrix_log(i_p)
            out[row, col] = float(np.linalg.norm(lr - lp, ord="fro"))
    return out


@register_operator(
    name="ts_student_t_fisher_shift",
    category="topology",
    business_category="topology",
    canonical="ts_student_t_fisher_shift",
    source="advanced_topology",
    status="experimental",
)
class TsStudentTFisherShift(SeriesOperator):
    """Student-t 参数化下的经验 Fisher 信息几何距离。

    两个不重叠窗口分别做 Student-t 快速 profile MLE（``2.1 <= nu <= 30``，
    参数化 ``(mu, log sigma, log(nu-2))``），估计经验 Fisher 信息阵
    ``I = mean(g g^T) + eps I``，输出 ``|log I_recent - log I_prior|_F``。
    确定性（固定 df 网格 + 固定中心差分步长）。P2 / Research。
    """

    metadata = _metadata(
        "ts_student_t_fisher_shift",
        "Student-t 经验 Fisher 信息几何距离（log 矩阵 Frobenius）。",
        ["x", "recent_window", "prior_window"],
        unit="distance",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        recent_window: int = 60,
        prior_window: int = 120,
        **_: Any,
    ) -> pd.DataFrame:
        r = int(recent_window)
        p = int(prior_window)
        if r < 8 or p < 8:
            raise ValueError("ts_student_t_fisher_shift requires recent_window, prior_window >= 8")
        return _frame_like(x, _fisher_shift_series(x.to_numpy(dtype=float), r, p))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        | {
            "ts_betti_1_max_persistence",
            "ts_persistence_diagram_shift",
            "ts_student_t_fisher_shift",
        }
    )


_register_surface()
