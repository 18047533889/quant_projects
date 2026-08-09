# -*- coding: utf-8 -*-
"""Cross-sectional extension operators: graph / monotone-fit / group tail geometry.

This module extends the 2026-08 geometry/math expansion with four new
per-date cross-sectional or per-group operators:

* ``cs_knn_local_moran``          — Local Moran I over a daily k-NN style graph
  (spatial-autocorrelation view of a target against its style peers).
* ``cs_isotonic_residual``        — cross-sectional residual of ``y`` on ``x``
  after an isotonic (monotone, pool-adjacent-violators) regression whose
  direction is chosen from the sign of the daily Spearman rank correlation.
* ``group_current_members_tail_coexceedance`` — within-group pairwise probability
  that both members are in their own extreme tail (exceedance density), net of a
  per-pair empirical independence baseline.  The group is defined by TODAY's
  membership (``current_members_retrospective``); the legacy name
  ``group_tail_coexceedance_density`` resolves as a deprecated alias.
* ``group_corr_mst_length``       — mean edge length of the Minimum Spanning
  Tree of the within-group pairwise correlation graph (a "how tightly is the
  group wired together" gauge).

Shared kernels (private to this module, deterministic):
``_rank_features`` / ``_neighbors`` (rank-standardised k-NN, from dynamic_knn),
``_zscore_cross``, ``_spearman`` / ``_pava_weighted`` (monotone fit, ties
share one fitted level),
``_pearson``, and ``_prim_mean`` (Prim MST).

PIT / causal / deterministic contract: every value at row ``t`` uses only rows
``<= t`` (prefix-causal).  The cs_* operators are computed on the date-t
cross-section only; the group_* operators use a trailing aligned window that
ends at ``t``.  No randomness anywhere; ties are broken with stable argsorts.
NaN inputs are dropped per aligned pair; a degenerate window (too few valid
rows / too few group members) emits NaN rather than a fabricated value.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge
# R6-133: one canonical k-NN implementation.  This module previously duplicated
# the competition-rank (nested argsort) kNN with a stable ``order[:k_eff]`` cut
# — ties were broken by column arrival order, so a feature tie made the factor
# depend on the stock-column ordering even after dynamic_knn was fixed.  Import
# the canonical helpers (average tie ranks + kth-distance-radius tie-inclusive
# selection) and drop the local copies.
from cleaned_operators.dynamic_knn import _avg_tie_ranks, _neighbors

_EPS = 1e-12
_RHO_EPS = 0.05  # |Spearman| below this → try both monotone fits, keep the better.
_MIN_Q_ROWS = 3  # trailing rows needed to compute a stock's own quantile threshold.
_MIN_ALIGNED = 3  # aligned trailing rows needed for a valid pairwise joint stat.
# R9-OP-018: the group coexceedance factor is the mean excess over *valid*
# pairs; if fewer than this fraction of the group's pairs can be estimated at a
# date, the group sample is too sparse and the factor emits NaN rather than a
# denominator-biased value.
_MIN_PAIR_COVERAGE = 0.5
# R6-136: a pairwise Pearson on ~3 aligned samples lands near ±1 trivially and
# then flows straight into the MST.  Require a real sample: max(20, window/2)
# aligned rows, or the edge is unknown and excluded.
_MIN_PAIR_ROWS = 20
# R6-137: isotonic fit over 2-3 stocks is near-perfect by construction.  The
# cross-sectional fit needs minimum breadth (>=10 names) to be meaningful.
_MIN_ISOTONIC_BREADTH = 10


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="cross_section_ext",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "cross_section_ext", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# shared k-NN kernels (rank-standardised feature graph, self excluded)
# ---------------------------------------------------------------------------
def _rank_features(feats: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Date t feature matrix (n,d) -> rank-standardised U + per-stock validity.

    R6-133: rank transform delegates to the canonical ``_avg_tie_ranks``
    (average tie ranks), never the nested-argsort competition ranks that made
    tied feature values depend on stock-column ordering.
    """
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


def _zscore_cross(vals: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score of one row; degenerate (constant) row → NaN."""
    z = np.full(vals.shape, np.nan, dtype=float)
    fin = np.isfinite(vals)
    m = int(fin.sum())
    if m < 2:
        return z
    mean = float(vals[fin].mean())
    sd = float(vals[fin].std(ddof=0))
    if not np.isfinite(sd) or sd <= _EPS:
        return z
    z[fin] = (vals[fin] - mean) / sd
    return z


def _local_moran_series(target: np.ndarray, feats: np.ndarray, k: int) -> np.ndarray:
    rows, n, d = feats.shape
    out = np.full((rows, n), np.nan, dtype=float)
    kk = int(k)
    for t in range(rows):
        U, valid = _rank_features(feats, t)
        z = _zscore_cross(target[t])
        # R6-134: neighbour candidates must be feature-valid AND target-valid
        # from the start.  Selecting k feature-neighbours and then dropping
        # target-NaN members would silently leave k-3 neighbours while a
        # farther target-valid name was never considered.  ``feat_valid`` is
        # the feature mask ANDed with the target mask before the kth-distance
        # selection.
        feat_valid = valid & np.isfinite(z)
        for i in range(n):
            if not valid[i] or not np.isfinite(z[i]):
                continue
            nbrs = _neighbors(U, feat_valid, kk, i)
            if nbrs.size == 0:
                continue
            nz = z[nbrs]
            out[t, i] = float(z[i] * np.mean(nz))
    return out


# ---------------------------------------------------------------------------
# isotonic (monotone) cross-sectional residual kernels
# ---------------------------------------------------------------------------
def _rankdata(arr: np.ndarray) -> np.ndarray:
    """1-based average ranks (standard tie handling)."""
    n = arr.size
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length 1D arrays."""
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = np.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if denom <= _EPS:
        return 0.0
    return float((dx * dy).sum() / denom)


def _pava_weighted(vals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit (non-decreasing) over *levels*.

    Each ``(vals[i], weights[i])`` is one x-level with weight = number of
    observations sharing that level.  A violator pair ``m_i > m_{i+1}`` is
    pooled into its weighted mean (with backtracking).  The returned array has
    ONE fitted value per INPUT level, so a caller can index it by
    ``return_inverse``; every level absorbed into a pooled block receives the
    block's common fitted value (the whole point of isotonic regression on ties).
    """
    means: list[float] = [float(v) for v in vals]
    counts: list[float] = [float(w) for w in weights]
    members: list[list[int]] = [[i] for i in range(int(vals.size))]
    i = 0
    while i < len(means) - 1:
        if means[i] > means[i + 1] + _EPS:  # violation → pool i and i+1
            total = counts[i] + counts[i + 1]
            means[i] = (means[i] * counts[i] + means[i + 1] * counts[i + 1]) / total
            counts[i] = total
            members[i] = members[i] + members[i + 1]
            del means[i + 1]
            del counts[i + 1]
            del members[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = np.empty(int(vals.size), dtype=float)
    for b, block in enumerate(members):
        for lv in block:
            out[lv] = means[b]
    return out


def _iso_fit(y_vals: np.ndarray, x_vals: np.ndarray, *, decreasing: bool) -> np.ndarray:
    """Isotonic regression fit of y on x — ties share ONE fitted level.

    R9-OP-017 (tied-x correctness): the old code stable-sorted x and ran PAVA
    over the raw per-stock sequence, so two stocks with the SAME x could get
    different fitted values and the result depended on stock-column order via
    the stable sort.  Standard isotonic regression assigns every observation at
    a given x-level the same fitted value.  We aggregate to unique x levels
    (weighted mean of y, count), run WEIGHTED PAVA on the levels, then map each
    observation back to its level's fitted value — deterministic and invariant
    to column permutation.  ``inv`` is the position->level map, so the output is
    in the caller's original (unsorted) order.
    """
    order = np.argsort(x_vals, kind="stable")
    xs = x_vals[order]
    ys = y_vals[order]
    unique_x, inv, counts = np.unique(xs, return_inverse=True, return_counts=True)
    wsum = np.zeros(unique_x.size, dtype=float)
    np.add.at(wsum, inv, ys)
    y_level = wsum / counts.astype(float)
    sign = -1.0 if decreasing else 1.0
    fit_level = _pava_weighted(sign * y_level, counts.astype(float))
    if decreasing:
        fit_level = -fit_level
    fitted_sorted = fit_level[inv]
    inv_idx = np.empty_like(order)
    inv_idx[order] = np.arange(order.size)
    return fitted_sorted[inv_idx]


def _isotonic_residual_series(y2d: np.ndarray, x2d: np.ndarray) -> np.ndarray:
    rows, cols = y2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for t in range(rows):
        xr, yr = x2d[t], y2d[t]
        m = np.isfinite(xr) & np.isfinite(yr)
        xs = xr[m].astype(float)
        ys = yr[m].astype(float)
        # R6-137: 2-3 stocks fit an isotonic regression almost perfectly.  The
        # cross-sectional fit needs minimum breadth to be a meaningful residual.
        if xs.size < _MIN_ISOTONIC_BREADTH:
            continue
        rho = _spearman(xs, ys)
        if rho > _RHO_EPS:
            fitted = _iso_fit(ys, xs, decreasing=False)
        elif rho < -_RHO_EPS:
            fitted = _iso_fit(ys, xs, decreasing=True)
        else:
            up = _iso_fit(ys, xs, decreasing=False)
            dn = _iso_fit(ys, xs, decreasing=True)
            sse_up = float(np.sum((ys - up) ** 2))
            sse_dn = float(np.sum((ys - dn) ** 2))
            fitted = up if sse_up <= sse_dn else dn
        out[t, m] = ys - fitted
    return out


# ---------------------------------------------------------------------------
# group kernels (extra `group`/`group_id` panel of per-date group ids)
# ---------------------------------------------------------------------------
def _group_labels(g_row: np.ndarray) -> list[Any]:
    valid_g = ~pd.isna(g_row)
    if not np.any(valid_g):
        return []
    return list(pd.unique(g_row[valid_g]))


def _tail_coexceedance_series(x2d: np.ndarray, g2d: np.ndarray, window: int, quantile: float, side: str) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    q = float(quantile)
    level = q if side == "upper" else (1.0 - q)
    # trailing own-quantile threshold per stock, prefix-causal at each row r
    thresh = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            # R6-132: the threshold must come from the STRICT prior history
            # [t-W, t-1] — the current row cannot participate in defining the
            # anomaly threshold it is then tested against (a big x_t would raise
            # its own bar).  Same prior-prefix rule as the impact-decay shock
            # threshold (#194).
            vals = col[lo:r]
            vals = vals[np.isfinite(vals)]
            if vals.size < _MIN_Q_ROWS:
                continue
            thresh[r, c] = float(np.quantile(vals, level))
    # tail indicator series
    E = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for c in range(cols):
            xv = x2d[r, c]
            tv = thresh[r, c]
            if not np.isfinite(xv) or not np.isfinite(tv):
                continue
            if side == "upper":
                E[r, c] = 1.0 if xv > tv else 0.0
            else:
                E[r, c] = 1.0 if xv < tv else 0.0
    for t in range(rows):
        lo = max(0, t - w + 1)
        for label in _group_labels(g2d[t]):
            idx = np.flatnonzero(g2d[t] == label)
            N = int(idx.size)
            if N < 3:
                continue
            total_pairs = 0.5 * N * (N - 1)
            pair_excess: list[float] = []
            for a in range(N):
                i = int(idx[a])
                for b in range(a + 1, N):
                    j = int(idx[b])
                    xa = x2d[lo : t + 1, i]
                    xb = x2d[lo : t + 1, j]
                    ta = thresh[lo : t + 1, i]
                    tb = thresh[lo : t + 1, j]
                    aligned = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(ta) & np.isfinite(tb)
                    cnt = int(aligned.sum())
                    if cnt < _MIN_ALIGNED:
                        continue
                    ei = (E[lo : t + 1, i] == 1.0) & aligned
                    ej = (E[lo : t + 1, j] == 1.0) & aligned
                    p_i = float(ei.sum()) / cnt
                    p_j = float(ej.sum()) / cnt
                    p_ij = float((ei & ej).sum()) / cnt
                    # R9-OP-019 (empirical independence baseline): a fixed
                    # ``(1-q)²`` assumes P(tail) == 1-q for BOTH stocks, which
                    # breaks with ties / discrete values / idiosyncratic tail
                    # rates.  The honest excess is per-pair ``p_ij - p_i·p_j``
                    # on the SAME aligned cohort; averaging it over valid pairs
                    # removes each pair's own marginal tail probability.
                    pair_excess.append(float(p_ij - p_i * p_j))
            if not pair_excess:
                continue
            # R9-OP-018 (missing-pair normalisation): the old code averaged over
            # ALL ``N(N-1)/2`` pairs but accumulated only the VALID ones, so a
            # pair with no aligned data silently counted as 0 in the denominator
            # and depressed the factor as data went missing.  Average over the
            # valid pairs only, and gate on the pair-coverage ratio: if too many
            # pairs cannot be estimated, emit NaN instead of a biased value.
            pair_coverage = len(pair_excess) / total_pairs
            if pair_coverage < _MIN_PAIR_COVERAGE:
                continue
            tc = float(np.mean(pair_excess))
            out[t, idx] = tc
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m].astype(float)
    b = b[m].astype(float)
    # R6-136: 3 aligned samples trivially give |rho| ~ 1 and then corrupt the
    # MST.  Require at least 20 aligned rows for a pairwise correlation edge.
    if a.size < _MIN_PAIR_ROWS:
        return np.nan
    da = a - a.mean()
    db = b - b.mean()
    denom = np.sqrt(float((da * da).sum()) * float((db * db).sum()))
    if denom <= _EPS:
        return np.nan
    return float((da * db).sum() / denom)


def _prim_mean(D: np.ndarray) -> float:
    """Mean edge length of the MST of a (symmetric) distance matrix D.

    A zero-length edge (perfectly correlated pair, d_ij = sqrt(2(1-ρ)) = 0) is a
    legal MST edge and must be counted.  Edge existence is determined by the
    selected-root / parent markers, never by ``edge_length > 0`` — 3rd-round
    audit P0-07.
    """
    n = D.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf, dtype=float)
    min_edge[0] = 0.0
    total = 0.0
    edges = 0
    for _ in range(n):
        cand = np.where(~in_tree & np.isfinite(min_edge), min_edge, np.inf)
        u = int(np.argmin(cand))
        if not np.isfinite(cand[u]):
            return np.nan  # disconnected graph → no spanning tree
        in_tree[u] = True
        if u != 0:  # every non-root node enters the tree through exactly one edge
            total += min_edge[u]
            edges += 1
        for v in range(n):
            if not in_tree[v] and D[u, v] < min_edge[v]:
                min_edge[v] = D[u, v]
    if edges != n - 1:
        return np.nan
    return total / (n - 1)


def _mst_length_series(x2d: np.ndarray, g2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for t in range(rows):
        lo = max(0, t - w + 1)
        for label in _group_labels(g2d[t]):
            idx = np.flatnonzero(g2d[t] == label)
            N = int(idx.size)
            if N < 3:
                continue
            D = np.full((N, N), np.inf, dtype=float)
            for a in range(N):
                i = int(idx[a])
                for b in range(a + 1, N):
                    j = int(idx[b])
                    rho = _pearson(x2d[lo : t + 1, i], x2d[lo : t + 1, j])
                    if not np.isfinite(rho):
                        continue
                    d = float(np.sqrt(2.0 * (1.0 - rho)))
                    D[a, b] = d
                    D[b, a] = d
            mst = _prim_mean(D)
            if np.isfinite(mst):
                out[t, idx] = mst
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="cs_knn_local_moran",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_knn_local_moran",
    source="cross_section_ext",
)
class CsKnnLocalMoran(SeriesOperator):
    """局部 Moran I：目标在风格 k-NN 图上的空间自相关。

    每日期截面：f1..f3 秩标准化后建 kNN 图（排除自身），target 截面 z-score 为
    z_i，Local Moran I_i = z_i · mean(z of neighbors)。高正 = 我和风格近邻同步
    极值（局部抱团）；高负 = 我相对风格近邻反向（风格内 alpha）。P1。
    """

    metadata = _metadata(
        "cs_knn_local_moran",
        "目标在 kNN 风格图上的 Local Moran I（局部空间自相关）。",
        ["target", "f1", "f2", "f3", "k"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(
        self, target: pd.DataFrame, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, k: int = 5, **_: Any
    ) -> pd.DataFrame:
        kk = int(k)
        if kk < 1:
            raise ValueError("k must be >= 1")
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(target, _local_moran_series(target.to_numpy(dtype=float), feats, kk))


@register_operator(
    name="cs_isotonic_residual",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_isotonic_residual",
    source="cross_section_ext",
)
class CsIsotonicResidual(SeriesOperator):
    """截面等渗回归残差：y 对 x 的单调拟合残差。

    每日期截面按 Spearman 秩相关符号决定单调方向（增/减），用 pool-adjacent-
    violators 拟合；|rho| 接近 0 时同时拟合升/降两条并取残差平方和更小者。
    残差 = y - ŷ。捕捉与 x 单调关系正交的截面 alpha。P1。
    """

    metadata = _metadata(
        "cs_isotonic_residual",
        "y 对 x 的截面等渗（单调）回归残差。",
        ["y", "x"],
        unit="residual",
        cost=6,
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return frame_like(y, _isotonic_residual_series(y.to_numpy(dtype=float), x.to_numpy(dtype=float)))


@register_operator(
    name="group_tail_coexceedance_density",
    category="group_structure",
    business_category="group_structure",
    canonical="group_current_members_tail_coexceedance",
    source="cross_section_ext",
)
class GroupTailCoexceedanceDensity(SeriesOperator):
    """组内尾部共同超越密度：两只股票同时处于各自极值尾部的概率。

    **current-members-retrospective canonical**（R9-OP-020）：组由今日
    membership 定义，每个当前成员的 trailing 对齐窗口参与组样本；成员的
    pre-reclassification 历史计入新组。这不是 historical-contemporaneous
    membership（那需要 PIT group 面板），canonical 名显式带
    ``current_members``。每个成员用自己的 trailing-window 分位数定义极值事件
    E_i（upper: x_i > Q_i(q)；lower: x_i < Q_i(1-q)），对组内每对股票取对齐
    窗口内同时为 1 的频率，净独立基线为逐 pair 的 ``p_ij - p_i·p_j``（而非固定
    (1-q)²），仅对有效 pair 平均并受 pair-coverage 门控。组内成员 <3、
    pair-coverage 不足或窗口过短 → NaN。P1。
    """

    metadata = _metadata(
        "group_current_members_tail_coexceedance",
        "组内两只股票同时处于各自极值尾部的概率密度（current-members-retrospective，净逐 pair 独立基线）。",
        ["x", "group_id", "window", "quantile", "side"],
        unit="ratio",
        cost=7,
    )
    metadata.param_specs = {
        # window must hold >= _MIN_Q_ROWS prior rows for a per-stock tail
        # threshold PLUS the current query row, so window >= _MIN_Q_ROWS + 1.
        "window": ParamSpec(dtype=int, min=_MIN_Q_ROWS + 1),
        "quantile": ParamSpec(dtype=float, min=0.5, max=0.99),
        "side": ParamSpec(dtype=str, choices=("upper", "lower")),
    }
    # R6-135 (membership vintage — current_members_retrospective): the group is
    # defined by TODAY's group labels, and each *current* member's trailing
    # aligned window is used as its "group sample".  This is the
    # ``current_members_retrospective`` canonical — a stock reclassified into
    # industry A yesterday contributes its pre-reclassification history to A's
    # sample.  That is an explicit, documented choice (not a silent mixing with
    # ``historical_contemporaneous_membership``, which would need PIT group_id
    # per day); operators mixing both without a flag are NOT accepted.

    def _calculate_series(
        self, x: pd.DataFrame, group_id: pd.DataFrame, window: int = 120, quantile: float = 0.9, side: str = "upper", **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < _MIN_Q_ROWS + 1:
            raise ValueError(f"window must be >= {_MIN_Q_ROWS + 1} (threshold needs {_MIN_Q_ROWS} prior rows + query)")
        if not (0.0 < float(quantile) < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        if side not in ("upper", "lower"):
            raise ValueError("side must be 'upper' or 'lower'")
        return frame_like(x, _tail_coexceedance_series(
            x.to_numpy(dtype=float), group_id.to_numpy(), w, float(quantile), side))


@register_operator(
    name="group_corr_mst_length",
    category="group_structure",
    business_category="group_structure",
    canonical="group_corr_mst_length",
    source="cross_section_ext",
)
class GroupCorrMstLength(SeriesOperator):
    """组内相关图最小生成树的平均边长。

    每组每日期：成员两两的 trailing Pearson 相关 ρ → 距离 d_ij = sqrt(2(1-ρ))，
    用 Prim 算法求最小生成树，输出平均 MST 边长。组内相关性高且一致 → 边长短
    （组被拧成一股绳）；相关性低/分裂 → 边长长。组内成员 <3 或重叠数据不足 →
    NaN。P1。
    """

    metadata = _metadata(
        "group_corr_mst_length",
        "组内相关图最小生成树（Prim）的平均边长。",
        ["x", "group_id", "window"],
        unit="ratio",
        cost=8,
    )
    # R9-OP-021 (feasibility): ``_pearson`` requires >= _MIN_PAIR_ROWS aligned
    # rows per edge; window < _MIN_PAIR_ROWS made the whole canonical a
    # guaranteed-NaN parameter region (window=2..19 legal but every edge NaN).
    # The declared contract now uses the SAME constant as the kernel guard, so
    # compile-valid == runtime-feasible.
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=_MIN_PAIR_ROWS),
    }
    # R6-135: same ``current_members_retrospective`` contract as
    # group_tail_coexceedance_density — today's group membership + each current
    # member's trailing aligned window.  Documented, not silent.

    def _calculate_series(self, x: pd.DataFrame, group_id: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = int(window)
        if w < _MIN_PAIR_ROWS:
            raise ValueError(f"window must be >= {_MIN_PAIR_ROWS} (a Pearson edge needs {_MIN_PAIR_ROWS} aligned rows)")
        return frame_like(x, _mst_length_series(x.to_numpy(dtype=float), group_id.to_numpy(), w))


_NEW_CANONICALS = (
    "cs_knn_local_moran",
    "cs_isotonic_residual",
    "group_current_members_tail_coexceedance",
    "group_corr_mst_length",
)
# R9-OP-020: honest rename — the legacy name did not say whether membership was
# current- or historical-contemporaneous.  ``group_tail_coexceedance_density``
# is kept as a deprecated resolving alias so existing recipes keep loading.
_DEPRECATED_ALIASES = {
    "group_tail_coexceedance_density": "group_current_members_tail_coexceedance",
}


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface
    from cleaned_operators.registry import OperatorRegistry

    # R9-P1-045: live-extend mutator — NEVER rebind the frozenset.  A rebind
    # snapshots a module-local copy that other modules do not see, so different
    # modules diverge on what is "extended-only".  ``extend_extended_only``
    # mutates the canonical surface object in place.
    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _old, _new in _DEPRECATED_ALIASES.items():
        try:
            OperatorRegistry.register_alias(_old, _new)
        except (KeyError, ValueError):
            pass  # already registered
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
