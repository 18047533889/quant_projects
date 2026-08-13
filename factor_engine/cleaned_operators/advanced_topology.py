# -*- coding: utf-8 -*-
"""Topological / information-geometry operators (2026-08 Gemini round, P2).

All operators are deterministic, prefix-causal pandas-numpy references.
``gudhi``/``ripser`` are not runtime dependencies: the Vietoris-Rips H1
persistence used here is a compact pure-numpy reduction over F2 (bitmask
columns) with a hard deterministic point-cloud cap, so repeated evaluation is
bit-identical and cost stays bounded for automatic search.

* ``ts_betti_1_max_persistence``  — max H1 persistence of the Takens-embedded
  rolling window (dimensionless, median-distance normalised).  P2/Research.
* ``ts_persistence_diagram_shift`` — Wasserstein-1 distance between the H1
  persistence *diagrams* of the current and the previous window, with diagonal
  matching solved exactly by the Hungarian algorithm (a topology regime-change
  proxy, cheaper than a full CROCKER).  P2/Research.
* ``ts_fisher_information_shift`` — Frobenius distance between the log
  empirical Fisher information matrices of two non-overlapping windows under a
  Student-t family, parameterised by ``(mu, log sigma, log(nu-2))``.
  P2/Research.

Missing-value policy — HISTORICAL_STATE_ALLOWED (M-4xx)
--------------------------------------------------------
Every operator in this module is a trailing-window *state* read: the Takens /
Rips / Fisher kernel consumes the finite history and a missing CURRENT row does
NOT force NaN — the window is allowed to emit a value built from the finite
past (this is the semantic opposite of ``topology_ext``'s
``ts_persistence_entropy_*``, which are CURRENT_ROW_REQUIRED and emit NaN when
the current observation is missing).  The two policies are explicitly
distinguished across the topology family so a stale-history factor is never
silently mis-classified.

Deterministic sampling policy (M-3xx)
-------------------------------------
The Takens point cloud is deduplicated by STABLE unique preserving first
occurrence (time order) and decimated by time-order skeleton sampling to at
most ``_MAX_POINTS`` points (``_SAMPLING_POLICY``) — never by lexicographic
``np.unique`` ordering.  Identical input -> bit-identical output.
"""
from __future__ import annotations

from math import lgamma, log as mlog, sqrt as msqrt
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.closure.strict_scalar import strict_int

_EPS = 1e-12
_MAX_POINTS = 12              # hard deterministic cap on the Rips point cloud.
_DF_GRID = (2.1, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)
_DF_STEP = 0.05
# M-3xx: versioned sampling policy for the Takens point cloud.
#
#   "stable_unique_first_occurrence + time_decimation"
#
# Duplicate embedding vectors are deduplicated by STABLE unique preserving the
# FIRST occurrence (i.e. the earliest time row), never ``np.unique(axis=0)``
# which sorts lexicographically and destroys the time/geometry ordering; the
# decimation to ``_MAX_POINTS`` then samples evenly in that time/first-occurrence
# order (a deterministic temporal skeleton), not in lexicographic state-space
# order.  Same input -> bit-identical output on every run (no RNG, no
# hash-order dependence).
_SAMPLING_POLICY = "stable_unique_first_occurrence + time_decimation"


def _stable_unique_first(pts: np.ndarray) -> np.ndarray:
    """Deduplicate rows preserving FIRST-occurrence (time) order.

    ``np.unique(pts, axis=0)`` returns the unique rows SORTED lexicographically,
    so a later decimation step sampled from a state-space-sorted point cloud —
    the sample was determined by lexicographic order, not by time or geometry.
    ``np.unique(..., return_index=True)`` returns, for each unique row, the index
    of its FIRST occurrence; sorting those indices restores the original
    (chronological) row order, so the surviving point cloud keeps the temporal
    structure of the embedding.
    """
    if pts.shape[0] == 0:
        return pts
    _, first_idx = np.unique(pts, axis=0, return_index=True)
    return pts[np.sort(first_idx)]


def _decimate_time_order(pts: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministic decimation to at most ``max_points`` in time order.

    Sample ``max_points`` evenly spaced positions in the *first-occurrence
    (time) order* of the point cloud.  This is a temporal skeleton — a
    deterministic subsample of the time-indexed trajectory — NOT a
    lexicographic-state-space sample.  ``np.unique`` on the chosen indices only
    removes the duplicate boundary index (e.g. two indices rounding to the same
    position); the remaining positions are still time-ordered.
    """
    if pts.shape[0] <= max_points:
        return pts
    keep = np.unique(
        np.round(np.linspace(0, pts.shape[0] - 1, max_points)).astype(np.int64)
    )
    return pts[keep]


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    param_specs: dict | None = None,
    relational_specs: list | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="topology",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "topology", "daily", "pit_safe", "causal", "typed_v2", "deterministic",
            "research_only", "historical_state_allowed",
            f"signature:{','.join(params)}->series", f"unit:{unit}", "cost:9",
        ],
        param_specs=dict(param_specs) if param_specs else {},
        relational_specs=list(relational_specs) if relational_specs else [],
    )


# M-2xx: Takens embedding resolution (tau / embedding_dim) is an ESTIMATOR knob
# (searchable=False); window is the HORIZON.  Relational feasibility: a window
# must hold at least 3 delay embeddings for the point cloud to be non-empty.
_TOPOLOGY_TAKENS_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=6, param_role=ParamRole.HORIZON, searchable=True),
    "tau": ParamSpec(
        dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=1
    ),
    "embedding_dim": ParamSpec(
        dtype=int, min=2, max=6,
        param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=3,
    ),
}
_TOPOLOGY_TAKENS_RELATIONAL: list[RelationalParamSpec] = [
    RelationalParamSpec(
        "window - (embedding_dim - 1) * tau >= 3",
        "requires window-(embedding_dim-1)*tau >= 3 delay embeddings "
        "(window={window}, embedding_dim={embedding_dim}, tau={tau})",
    ),
]


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# Vietoris-Rips H1 persistence (pure numpy, F2 bitmask reduction)
# ---------------------------------------------------------------------------

def _rips_h1_pairs(points: np.ndarray) -> list[tuple[float, float]]:
    """H1 persistence pairs ``(birth, death)`` of a Rips complex on ``points``.

    Zomorodian–Carlsson boundary-matrix reduction over F2: triangle columns are
    processed in ascending filtration value and reduced against the pivot column
    sharing the same *highest* edge row (latest filtration).  The pivot row must
    be the highest edge index — mirroring the pivot onto the lowest edge (the
    earlier implementation) invents spurious H1 classes on collinear clouds,
    where the correct barcode is empty.
    """
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
        while col:
            hi = col.bit_length() - 1  # highest edge index (latest filtration) = pivot row.
            prev = pivots.get(hi)
            if prev is None:
                pivots[hi] = col
                break
            col ^= prev
        if col == 0:
            continue  # boundary already spanned -> no new H1 class to kill.
        hi = col.bit_length() - 1
        pairs.append((edge_len[hi], flt))
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
    z = (vals - med) / spread if spread != 0 else np.nan
    n = vals.shape[0]
    lag = tau * (dim - 1)
    if n < lag + 3:
        return None
    pts = np.stack([z[s - (dim - 1) * tau : s + 1 : tau] for s in range(lag, n)], axis=0)
    # Drop any embedding vector that contains a NaN (a NaN in one lagged
    # coordinate would otherwise poison every pairwise distance in the Rips
    # complex below).
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 4:
        return None
    pts = np.round(pts, decimals=9)
    # M-3xx: stable unique preserving FIRST occurrence (time order) — the old
    # ``np.unique(axis=0)`` sorted the cloud lexicographically, so the decimation
    # below sampled from a state-space-sorted order.  First-occurrence unique +
    # time-order decimation keeps the sample a deterministic temporal skeleton.
    pts = _stable_unique_first(pts)
    if pts.shape[0] < 4:
        return None
    pts = _decimate_time_order(pts, _MAX_POINTS)
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
            # R6-121: the distance scale must come from the OFF-DIAGONAL
            # distances (i<j) only — the full matrix median mechanically includes
            # the zero diagonal and biases the scale low.  Use the strictly
            # upper-triangular pairwise distances.
            triu = pd_pairwise[np.triu_indices(pd_pairwise.shape[0], k=1)]
            finite = triu[np.isfinite(triu)]
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
    确定性抽稀到 ≤12 点（M-3xx：stable-unique 保留首次出现 + 时间顺序骨架抽样，
    不是字典序状态空间抽样），纯 numpy 计算 Rips H1 持久对，输出
    ``max(death-birth)`` 再除以点云中位成对距离（无量纲）。无 H1 -> 0。
    绝不使用 random jitter。**Missing policy：HISTORICAL_STATE_ALLOWED**——
    当前行缺失时窗口可用有限过去发射状态（与 topology_ext 的
    CURRENT_ROW_REQUIRED 相反）。M-2xx：window/tau/embedding_dim 全部 ParamSpec，
    ``window-(embedding_dim-1)*tau >= 3`` 关系可行在绑定期强制。P2 / Research。
    """

    metadata = _metadata(
        "ts_betti_1_max_persistence",
        "滚动 H1 最大持久性（Takens+Rips，中位距离归一化，无量纲；"
        "HISTORICAL_STATE_ALLOWED）。",
        ["x", "window", "tau", "embedding_dim"],
        unit="ratio",
        param_specs=_TOPOLOGY_TAKENS_SPECS,
        relational_specs=_TOPOLOGY_TAKENS_RELATIONAL,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        tau: int = 1,
        embedding_dim: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        w = strict_int(window, "window", lower=6)
        t = strict_int(tau, "tau", lower=1)
        m = strict_int(embedding_dim, "embedding_dim", lower=2, upper=6)
        if w - (m - 1) * t < 3:
            raise ValueError(
                "ts_betti_1_max_persistence requires window-(embedding_dim-1)*tau >= 3 "
                f"(window={w}, embedding_dim={m}, tau={t})"
            )
        return _frame_like(x, _betti_series(x.to_numpy(dtype=float), w, t, m))


def _diag_dist(p: tuple[float, float]) -> float:
    """L-infinity distance from point ``(birth, death)`` to the diagonal."""
    return (p[1] - p[0]) / 2.0


def _diagram_w1(pairs_a: list[tuple[float, float]], pairs_b: list[tuple[float, float]]) -> float:
    """1-Wasserstein distance between two persistence diagrams (diagonal matching).

    Points may match a point of the other diagram (L-infinity cost) or the
    diagonal (cost ``(death-birth)/2``); the optimal bipartite matching is solved
    exactly with the Hungarian algorithm.  This is the genuine persistence-diagram
    distance the operator name promises — the earlier implementation compared only
    the 1D ``death-birth`` lifetimes, discarding birth locations.
    """
    m1, m2 = len(pairs_a), len(pairs_b)
    if m1 == 0 and m2 == 0:
        return 0.0
    # Empty-vs-nonempty is the same metric as the general case: every point of
    # the non-empty diagram matches to the diagonal, each contributing its
    # diagonal distance.  Summing (not averaging) keeps W1(empty, B) on the
    # same total-cost scale as the assignment branch below — a mean here would
    # make an empty diagram look closer for larger diagrams, breaking the
    # metric (and the total-vs-mean choice made in the R6-120 fix).
    if m1 == 0:
        return float(sum(_diag_dist(p) for p in pairs_b))
    if m2 == 0:
        return float(sum(_diag_dist(p) for p in pairs_a))
    from scipy.optimize import linear_sum_assignment

    # Standard extended-diagram construction.  A point may match a point of the
    # other diagram (L-infinity cost) or the diagonal (cost (death-birth)/2).
    # Rows  = m1 real D1 points + m2 D1-diagonal copies.
    # Cols  = m2 real D2 points + m1 D2-diagonal copies.
    #   real A_i -> real B_j        : d_inf(A_i, B_j)
    #   real A_i -> any B-diagonal  : diag(A_i)
    #   any A-diagonal -> real B_j  : diag(B_j)
    #   A-diagonal -> B-diagonal    : 0   (leftover)
    # All other cells are illegal and carry a huge cost.  The earlier code left
    # the unassigned cells at 0 (B1 -> B2's diagonal slot plus B2 -> A1's slot
    # resolved the matching at cost 0), and the "each point owns one slot"
    # variant that replaced it double-counted diagonal costs even on identical
    # diagrams.
    n = m1 + m2
    BIG = 1e12
    cost = np.full((n, n), BIG, dtype=float)
    for i, p in enumerate(pairs_a):
        for j, q in enumerate(pairs_b):
            cost[i, j] = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
    for i in range(m1):
        cost[i, m2:] = _diag_dist(pairs_a[i])
    for j in range(m2):
        cost[m1:, j] = _diag_dist(pairs_b[j])
    cost[m1:, m2:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    # R6-120: the previous code divided the total assignment cost by
    # max(m1, m2), turning it into a MEAN matching cost — not the standard
    # persistence-diagram W1.  True W1 keeps the TOTAL assignment cost (the
    # diagonal copies give it the metric interpretation).  Removed the division;
    # the output is now the genuine W1 the operator name and description claim.
    return float(cost[rows, cols].sum())


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
            out[row, col] = _diagram_w1(cur_pairs, pri_pairs)
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
    """相空间拓扑 regime 漂移：当前窗口与上一窗口 H1 persistence diagram 的 W1。

    两个窗口均 Takens 嵌入 + Rips H1；输出两个 ``(birth, death)`` diagram 的
    Wasserstein-1 距离，未匹配点按到对角线的 L-infinity 代价计入（Hungarian
    精确匹配）。这是 CROCKER 的低成本替代——回答"拓扑结构变了多少"，
    而非逐层 Betti 曲线。**Missing policy：HISTORICAL_STATE_ALLOWED**（当前行
    缺失时可用有限过去发射窗口状态）。M-2xx：window/tau/embedding_dim 全部
    ParamSpec + 关系可行绑定期强制。P2 / Research。
    """

    metadata = _metadata(
        "ts_persistence_diagram_shift",
        "当前 vs 上一窗口 H1 persistence diagram 的 Wasserstein-1 距离"
        "（HISTORICAL_STATE_ALLOWED）。",
        ["x", "window", "tau", "embedding_dim"],
        unit="ratio",
        param_specs=_TOPOLOGY_TAKENS_SPECS,
        relational_specs=_TOPOLOGY_TAKENS_RELATIONAL,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        tau: int = 1,
        embedding_dim: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        w = strict_int(window, "window", lower=6)
        t = strict_int(tau, "tau", lower=1)
        m = strict_int(embedding_dim, "embedding_dim", lower=2, upper=6)
        if w - (m - 1) * t < 3:
            raise ValueError(
                "ts_persistence_diagram_shift requires window-(embedding_dim-1)*tau >= 3 "
                f"(window={w}, embedding_dim={m}, tau={t})"
            )
        return _frame_like(x, _persistence_shift_series(x.to_numpy(dtype=float), w, t, m))


# ---------------------------------------------------------------------------
# Student-t empirical Fisher shift
# ---------------------------------------------------------------------------

def _t_logpdf_const(df: float) -> float:
    return lgamma(0.5 * (df + 1.0)) - 0.5 * mlog(df * np.pi) - lgamma(0.5 * df)


def _t_logpdf_vec(v: np.ndarray, df: float, mu: float, scale: float) -> np.ndarray:
    z = (v - mu) / scale
    return np.where(df) != 0, _t_logpdf_const(df) - mlog(scale) - 0.5 * (df + 1.0) * np.log1p(z * z / df), np.nan)


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
            w = np.where((df + z * z) != 0, (df + 1.0) / (df + z * z), np.nan)
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
    w = np.where((df + z * z) != 0, (df + 1.0) / (df + z * z), np.nan)
    g_mu = w * z / scale if scale != 0 else np.nan
    g_ls = df * (z * z - 1.0) / (df + z * z)          #np.where(d(log scale) != 0, (d) / (d(log scale)), np.nan)
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
            if row < r + p - 1:  # need exactly ``prior`` then exactly ``recent`` rows.
                continue
            cur = vals_2d[row - r + 1 : row + 1, col]            # exactly r observations.
            pri = vals_2d[row - r - p + 1 : row - r + 1, col]    # exactly p observations.
            # R6-119: fit and empirical-Fisher MUST run on the SAME exact
            # finite cohort.  ``_t_fit`` filters NaN internally; passing the raw
            # block to ``_empirical_fisher`` after a NaN-filtered fit mixed
            # sample sizes and let a NaN leak into the Fisher matrix (which then
            # went NaN / eigh failed instead of failing closed).  Filter once,
            # use the same arrays for both.
            cur_f = cur[np.isfinite(cur)]
            pri_f = pri[np.isfinite(pri)]
            fit_r = _t_fit(cur_f)
            fit_p = _t_fit(pri_f)
            if fit_r is None or fit_p is None:
                continue
            i_r = _empirical_fisher(cur_f, *fit_r)
            i_p = _empirical_fisher(pri_f, *fit_p)
            lr = _matrix_log(i_r)
            lp = _matrix_log(i_p)
            out[row, col] = float(np.linalg.norm(lr - lp, ord="fro"))
    return out


@register_operator(
    name="ts_fisher_information_shift",
    category="topology",
    business_category="topology",
    canonical="ts_fisher_information_shift",
    source="advanced_topology",
    status="experimental",
)
class TsFisherInformationShift(SeriesOperator):
    """Student-t 参数化下的经验 Fisher 信息阵结构漂移。

    两个不重叠窗口分别做 Student-t 快速 profile MLE（``2.1 <= nu <= 30``，
    参数化 ``(mu, log sigma, log(nu-2))``），估计经验 Fisher 信息阵
    ``I = mean(g g^T) + eps I``，输出 ``|log I_recent - log I_prior|_F``。
    注意：这是 Fisher *信息阵*的结构漂移，不是两个 Student-t 参数点之间的
    Fisher-Rao 测地距离。确定性（固定 df 网格 + 固定中心差分步长）。
    **Missing policy：HISTORICAL_STATE_ALLOWED**（当前行缺失时可用有限过去的
    两段窗口发射 Fisher 结构状态）。recent_window/prior_window 为
    ESTIMATOR_RESOLUTION（searchable=False）。P2 / Research。
    """

    metadata = _metadata(
        "ts_fisher_information_shift",
        "Student-t 经验 Fisher 信息阵结构漂移（log 矩阵 Frobenius；"
        "HISTORICAL_STATE_ALLOWED）。",
        ["x", "recent_window", "prior_window"],
        unit="distance",
        param_specs={
            "recent_window": ParamSpec(
                dtype=int, min=8,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=60,
            ),
            "prior_window": ParamSpec(
                dtype=int, min=8,
                param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=120,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        recent_window: int = 60,
        prior_window: int = 120,
        **_: Any,
    ) -> pd.DataFrame:
        r = strict_int(recent_window, "recent_window", lower=8)
        p = strict_int(prior_window, "prior_window", lower=8)
        return _frame_like(x, _fisher_shift_series(x.to_numpy(dtype=float), r, p))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_research_only(
        {
            "ts_betti_1_max_persistence",
            "ts_persistence_diagram_shift",
            "ts_fisher_information_shift",
        }
    )


_register_surface()
