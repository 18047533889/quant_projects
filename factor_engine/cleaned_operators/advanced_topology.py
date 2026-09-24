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

from contextvars import ContextVar
from math import lgamma, log as mlog, sqrt as msqrt
from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.closure.strict_scalar import strict_int

_EPS = 1e-12
_MAX_POINTS = 12              # hard deterministic cap on the Rips point cloud.
_DF_GRID = (2.1, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)
_DF_STEP = 0.05
_LAST_T_FIT_STATUS: ContextVar[dict[str, Any]] = ContextVar(
    "advanced_topology_t_fit_status", default={"reason": "not_run"}
)
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
    z = (vals - med) / spread
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


# ---------------------------------------------------------------------------
# R66-perf: batched Takens + Vietoris-Rips H1 kernels
#
# 逐 (row, col) 的 Python 循环（_takens_points + _rips_h1_pairs）改为面板级
# 批量计算。语义逐点复刻：round-9、stable-unique 保留首次出现（时间序）、
# 时间骨架抽稀到 ≤ _MAX_POINTS、边按 (dist, i, j) / 三角按 (flt, i, j, k)
# 排序、F2 位掩码（双 word）最高位主元的 Zomorodian–Carlsson 消元。
# ---------------------------------------------------------------------------

_RIPS_TABLES_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

# W1(Hungarian)结果精确缓存：key 为两 diagram 的 (birth, death) 字节串。
# 纯函数缓存，相同输入必得相同输出；超上限时整体清空。
_W1_ASSIGN_CACHE: dict = {}


def _rips_tables(n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(edges_ij, triangles_ijk, triangle_edge_ids)`` of the ``n_max`` simplex."""
    tab = _RIPS_TABLES_CACHE.get(n_max)
    if tab is None:
        pairs = [(i, j) for i in range(n_max) for j in range(i + 1, n_max)]
        eid = {p: k for k, p in enumerate(pairs)}
        tris = [
            (i, j, k)
            for i in range(n_max)
            for j in range(i + 1, n_max)
            for k in range(j + 1, n_max)
        ]
        tab = (
            np.asarray(pairs, dtype=np.int64),
            np.asarray(tris, dtype=np.int64),
            np.asarray(
                [[eid[(i, j)], eid[(i, k)], eid[(j, k)]] for (i, j, k) in tris],
                dtype=np.int64,
            ),
        )
        _RIPS_TABLES_CACHE[n_max] = tab
    return tab


def _high_bit_u64(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Highest set bit index of a two-word (lo, hi) F2 bitmask."""
    hi_nz = hi != 0
    word = np.where(hi_nz, hi, lo)
    _, expo = np.frexp(word.astype(np.float64))
    bit = expo.astype(np.int64) - 1
    shift = np.clip(bit, 0, 63).astype(np.uint64)
    set_ = ((word >> shift) & np.uint64(1)) != 0
    bit = np.where(set_, bit, bit - 1)
    return np.where(hi_nz, bit + 64, bit)


def _robust_z_local(win2: np.ndarray):
    """逐行 median/MAD*1.4826（std 回退）稳健 z，复刻 _takens_points 归一。"""
    finite = np.isfinite(win2)
    n_fin = finite.sum(axis=1)
    pf = np.where(finite, win2, np.nan)
    guarded = np.where((n_fin > 0)[:, None], pf, 0.0)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(guarded, axis=1)
        mad = np.nanmedian(
            np.where(n_fin[:, None] > 0, np.abs(pf - med[:, None]), 0.0), axis=1
        ) * 1.4826
    spread = np.where(mad > _EPS, mad, np.nan)
    use_std = ~(mad > _EPS) & (n_fin >= 2)
    if use_std.any():
        with np.errstate(invalid="ignore"):
            spread[use_std] = np.nanstd(guarded[use_std], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (win2 - med[:, None]) / spread[:, None]
    return z, spread, n_fin


def _trailing_windows(vals_2d: np.ndarray, w: int) -> np.ndarray:
    """(rows, w, cols) 左对齐 trailing 窗口，行内不足补 NaN。"""
    rows, cols = vals_2d.shape
    ridx = np.arange(rows)[:, None]
    widx = np.maximum(0, ridx - w + 1) + np.arange(w)[None, :]
    gather = vals_2d[np.clip(widx, 0, rows - 1)]
    return np.where((widx <= ridx)[:, :, None], gather, np.nan)


def _takens_clouds_batch(win: np.ndarray, lag: int, dim: int, tau: int, cap: int):
    """批量化 ``_takens_points``：返回 (pts, vmask, n_uniq)。

    ``win`` 为 (cells, w) 的稳健 z 填充窗口；dedup 用 lexsort（时间索引为
    末位 tie-break → 首次出现保序），抽稀为 round(linspace(0, n-1, cap))
    去重的时间骨架。
    """
    cells, w = win.shape
    n_pts = w - lag
    posidx = np.arange(n_pts)[:, None] + tau * np.arange(dim)[None, :]
    cloud = win[:, posidx]
    valid = np.isfinite(cloud).all(axis=2)
    rnd = np.round(cloud, 9)
    keys = (np.broadcast_to(np.arange(n_pts, dtype=float), (cells, n_pts)),) + tuple(
        np.where(valid, rnd[:, :, c], np.nan) for c in range(dim - 1, -1, -1)
    )
    perm = np.lexsort(keys, axis=1)
    masked = np.where(valid[..., None], rnd, np.nan)
    srt = np.take_along_axis(
        masked, np.broadcast_to(perm[:, :, None], masked.shape), axis=1
    )
    pos = np.arange(n_pts, dtype=np.int64)[None, :]
    starts = np.ones((cells, n_pts), dtype=bool)
    starts[:, 1:] = ~(srt[:, 1:, :] == srt[:, :-1, :]).all(axis=2)
    gstart = starts & (pos < valid.sum(axis=1)[:, None])
    fidx = np.where(gstart, perm, n_pts)
    order2 = np.argsort(fidx, axis=1, kind="stable")
    found = np.take_along_axis(gstart, order2, axis=1)
    n_uniq = found.sum(axis=1)
    uni = np.take_along_axis(
        srt, np.broadcast_to(order2[:, :, None], srt.shape), axis=1
    )
    span = np.maximum(n_uniq - 1, 0).astype(np.float64)
    node = np.arange(cap, dtype=np.float64)[None, :] * (span / (cap - 1.0))[:, None]
    node[:, -1] = span
    last = np.maximum(n_uniq - 1, 0)[:, None]
    ksrt = np.sort(np.clip(np.rint(node).astype(np.int64), 0, last), axis=1)
    kmask = np.ones_like(ksrt, dtype=bool)
    kmask[:, 1:] = ksrt[:, 1:] != ksrt[:, :-1]
    pts = np.full((cells, cap, dim), np.nan)
    vmask = np.zeros((cells, cap), dtype=bool)
    flat = kmask.ravel()
    rsel = np.repeat(np.arange(cells), cap)[flat]
    csel = (np.cumsum(kmask, axis=1) - 1).ravel()[flat]
    ssel = ksrt.ravel()[flat]
    pts[rsel, csel] = uni[rsel, ssel]
    vmask[rsel, csel] = True
    vmask &= (n_uniq >= 4)[:, None]
    return pts, vmask, n_uniq


def _rips_h1_pairs_batch(points: np.ndarray, pmask: np.ndarray):
    """批量化 ``_rips_h1_pairs``：返回 ``(birth, death, has)``。

    has[c, t] = 第 c 个点云的第 t 个（filtration 序）三角消元后存活的 H1 对；
    birth/death 为对应边长/三角 filtration。padding 产生的 inf 简单形已剔除。
    """
    rows, n_max, _ = points.shape
    pairs_ij, tris, tri_edges = _rips_tables(n_max)
    ei, ej = pairs_ij[:, 0], pairs_ij[:, 1]
    ti, tj, tk = tris[:, 0], tris[:, 1], tris[:, 2]
    diff = points[:, :, None, :] - points[:, None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    ok_pair = pmask[:, :, None] & pmask[:, None, :]
    dist = np.where(ok_pair, dist, np.inf)
    edge = dist[:, ei, ej]
    tri_flt = np.maximum(np.maximum(dist[:, ti, tj], dist[:, ti, tk]), dist[:, tj, tk])
    n_edges = edge.shape[1]
    shp_e = edge.shape
    # 边表按 (ei, ej) 字典序生成，故 lexsort((ej, ei, edge)) 的次键 tie-break
    # 恰为表序 —— 等价于对 edge 的单键 stable argsort。
    eorder = np.argsort(edge, axis=1, kind="stable")
    e_sorted = np.take_along_axis(edge, eorder, axis=1)
    erank = np.empty(shp_e, dtype=np.int64)
    np.put_along_axis(erank, eorder, np.arange(n_edges, dtype=np.int64)[None, :], axis=1)
    shp_t = tri_flt.shape
    # 同理：三角表按 (ti, tj, tk) 字典序生成，lexsort 的三个次键 tie-break
    # 恰为表序 —— 等价于对 tri_flt 的单键 stable argsort。
    torder = np.argsort(tri_flt, axis=1, kind="stable")
    tri_rank = erank[:, tri_edges]
    # R66-perf-4：初始边界掩码循环前一次算完（(n_tris, rows) 布局，每轮取
    # 连续行切片），并转置 torder/tri_flt；消元链本身与原实现逐步一致。
    # 同一格内三条边 rank 互异（erank 是排列），位的 OR 等价于 XOR。
    rix = np.arange(rows)
    # 按每格 filtration 序（torder）gather 三角：输出列 t 对应 torder[:, t]。
    rr_all = tri_rank[rix[:, None], torder]  # (rows, n_tris, 3)
    low_all = rr_all < 64
    blo = np.left_shift(np.uint64(1), (rr_all & np.int64(63)).astype(np.uint64))
    blo = np.where(low_all, blo, np.uint64(0))
    bhi = np.left_shift(
        np.uint64(1), np.clip(rr_all - np.int64(64), 0, 63).astype(np.uint64)
    )
    bhi = np.where(low_all, np.uint64(0), bhi)
    c0_lo = np.ascontiguousarray((blo[:, :, 0] | blo[:, :, 1] | blo[:, :, 2]).T)
    c0_hi = np.ascontiguousarray((bhi[:, :, 0] | bhi[:, :, 1] | bhi[:, :, 2]).T)
    flt_T = np.ascontiguousarray(tri_flt[rix[:, None], torder].T)
    piv_lo = np.zeros((rows, n_edges), dtype=np.uint64)
    piv_hi = np.zeros((rows, n_edges), dtype=np.uint64)
    birth = np.full((rows, shp_t[1]), np.nan, dtype=float)
    death = np.full((rows, shp_t[1]), np.nan, dtype=float)
    has = np.zeros((rows, shp_t[1]), dtype=bool)
    _zero = np.uint64(0)
    for t in range(shp_t[1]):
        clo = c0_lo[t].copy()
        chi = c0_hi[t].copy()
        flt = flt_T[t]
        act = np.flatnonzero((clo | chi) != 0)
        while act.size:
            alo = clo[act]
            ahi = chi[act]
            hi = _high_bit_u64(alo, ahi)
            # 占据过的 piv 位形必非零（act 只含非零掩码），零判定等价 piv_ok。
            free = (piv_lo[act, hi] == _zero) & (piv_hi[act, hi] == _zero)
            if free.any():
                st = act[free]
                hst = hi[free]
                piv_lo[st, hst] = alo[free]
                piv_hi[st, hst] = ahi[free]
                # 同一轮内每格至多 free 一次，直接写等价于 stored 暂存。
                birth[st, t] = e_sorted[st, hst]
                death[st, t] = flt[st]
                has[st, t] = True
            keep = ~free
            if not keep.any():
                break
            ct = act[keep]
            phi = hi[keep]
            clo[ct] = alo[keep] ^ piv_lo[ct, phi]
            chi[ct] = ahi[keep] ^ piv_hi[ct, phi]
            act = ct[(clo[ct] | chi[ct]) != 0]
    has &= np.isfinite(birth) & np.isfinite(death)
    return birth, death, has


def _betti_series(vals_2d: np.ndarray, window: int, tau: int, dim: int) -> np.ndarray:
    """R66-perf 批量化：每格 trailing 窗口 → Takens 云 → Rips H1 → max 持久
    / 中位成对距离。语义与逐格参考实现一致。

    R66-perf-4：robust-z / Takens / Rips 全量单批；消元内核见
    ``_rips_h1_pairs_batch`` 的 R66-perf-4 注。
    """
    rows, cols = vals_2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    lag = tau * (dim - 1)
    if rows == 0 or cols == 0 or w - lag <= 0 or w < lag + 3:
        return out
    win = _trailing_windows(vals_2d, w).transpose(0, 2, 1).reshape(rows * cols, w)
    # 注：曾尝试 np.unique(axis=0) 去重窗口，实测合成面板各列窗口几乎不重复
    # 且 unique 本身有固定开销，得不偿失，回退为全量批量。
    z, spread, n_fin = _robust_z_local(win)
    pts, vmask, n_uniq = _takens_clouds_batch(z, lag, dim, tau, _MAX_POINTS)
    birth, death, has = _rips_h1_pairs_batch(pts, vmask)
    life = np.where(has, death - birth, -np.inf)
    maxp = life.max(axis=1)
    maxp = np.where(np.isfinite(maxp), maxp, 0.0)   # 无 H1 对 → 0.0
    iu0, iu1 = np.triu_indices(_MAX_POINTS, k=1)
    # 直接在 66 个上三角点对上算距离（避免 12x12 全矩阵广播再 gather）。
    pa = pts[:, iu0]
    pb = pts[:, iu1]
    pd2 = ((pa - pb) ** 2).sum(-1)
    pd2 = np.where(vmask[:, iu0] & vmask[:, iu1], pd2, np.nan)
    triu_d = np.sqrt(pd2)
    with np.errstate(invalid="ignore"):
        med_d = np.nanmedian(triu_d, axis=1)
    length = np.minimum(np.arange(rows) + 1, w)
    ok = (
        (n_fin >= 4)
        & (np.repeat(length, cols) >= lag + 3)
        & (spread > _EPS)
        & (n_uniq >= 4)
        & np.isfinite(med_d)
        & (med_d > _EPS)
    )
    val = np.where(ok, maxp / np.where(med_d > _EPS, med_d, 1.0), np.nan)
    return val.reshape(rows, cols)


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
    """R66-perf 批量化：当前/上一窗口的 H1 diagram 的 W1（逐格 Hungarian）。

    语义与逐格参考实现一致：两窗口各 ≥6 行、Takens 云有效、diagram 为空时
    按对角距离求和（_diagram_w1 的三分支）。

    R66-perf-2：上一窗口与当前窗口共享同一条序列（prior[r] == cur[r-w]），
    robust-z / Takens / Rips 只算一份，prior 侧按 ``w*cols`` 平移索引复用；
    逐格 Hungarian 结果以 diagram 字节串为 key 精确缓存（纯函数，相同输入
    必得相同输出，不改变语义）。
    """
    from scipy.optimize import linear_sum_assignment

    rows, cols = vals_2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    lag = tau * (dim - 1)
    if rows == 0 or cols == 0 or rows <= w or w - lag <= 0 or w < lag + 3:
        return out
    win = _trailing_windows(vals_2d, w).transpose(0, 2, 1).reshape(rows * cols, w)
    zc, spread_c, n_fin_c = _robust_z_local(win)
    length_cur = np.minimum(np.arange(rows) + 1, w)
    okc = (
        (n_fin_c >= 4)
        & (np.repeat(length_cur, cols) >= 6)
        & (spread_c > _EPS)
    )
    okp = np.zeros_like(okc)
    okp[w * cols:] = okc[: (rows - w) * cols]
    pts_c, vmask_c, n_uniq_c = _takens_clouds_batch(zc, lag, dim, tau, _MAX_POINTS)
    n_uniq_p = np.zeros_like(n_uniq_c)
    n_uniq_p[w * cols:] = n_uniq_c[: (rows - w) * cols]
    ok = okc & okp & (n_uniq_c >= 4) & (n_uniq_p >= 4)
    if not ok.any():
        return out
    bc, dc, has_c = _rips_h1_pairs_batch(pts_c, vmask_c)
    shift = w * cols
    flat_out = np.full(rows * cols, np.nan, dtype=float)
    BIG = 1e12
    cache = _W1_ASSIGN_CACHE
    if len(cache) > 200000:
        cache.clear()
    for cidx in np.flatnonzero(ok):
        sc = cidx - shift
        m1m = has_c[cidx]
        m2m = has_c[sc]
        b1 = bc[cidx][m1m]
        d1 = dc[cidx][m1m]
        b2 = bc[sc][m2m]
        d2 = dc[sc][m2m]
        m1 = b1.size
        m2 = b2.size
        if m1 == 0 and m2 == 0:
            flat_out[cidx] = 0.0
            continue
        if m2 == 0:
            flat_out[cidx] = float(np.sum((d1 - b1) / 2.0))
            continue
        if m1 == 0:
            flat_out[cidx] = float(np.sum((d2 - b2) / 2.0))
            continue
        key = (b1.tobytes(), d1.tobytes(), b2.tobytes(), d2.tobytes())
        val = cache.get(key)
        if val is None:
            n = m1 + m2
            cost = np.full((n, n), BIG, dtype=float)
            cost[:m1, :m2] = np.maximum(
                np.abs(b1[:, None] - b2[None, :]), np.abs(d1[:, None] - d2[None, :])
            )
            cost[:m1, m2:] = ((d1 - b1) / 2.0)[:, None]
            cost[m1:, :m2] = (d2 - b2) / 2.0
            cost[m1:, m2:] = 0.0
            rr, cc = linear_sum_assignment(cost)
            val = float(cost[rr, cc].sum())
            cache[key] = val
        flat_out[cidx] = val
    return flat_out.reshape(rows, cols)


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
    return _t_logpdf_const(df) - mlog(scale) - 0.5 * (df + 1.0) * np.log1p(z * z / df)


def last_t_fit_status() -> dict[str, Any]:
    """Return a copy of the context-local diagnostic for the latest t fit."""
    return dict(_LAST_T_FIT_STATUS.get())


def _t_fit(
    vals: np.ndarray,
    *,
    max_iter: int = 200,
    score_tol: float = 1e-7,
) -> tuple[float, float, float] | None:
    """Profile Student-t MLE on a fixed df grid in block-normalized units."""
    v = vals[np.isfinite(vals)]
    if v.size < 8:
        _LAST_T_FIT_STATUS.set({"reason": "insufficient_sample", "n": int(v.size)})
        return None
    center = float(np.median(v))
    centered = v - center
    spread = float(np.median(np.abs(centered))) * 1.4826
    if not np.isfinite(spread) or spread == 0.0:
        spread = float(np.std(centered))
    if not np.isfinite(spread) or spread == 0.0:
        _LAST_T_FIT_STATUS.set({"reason": "degenerate_scale", "n": int(v.size)})
        return None
    q = centered / spread
    from scipy.optimize import minimize

    best: tuple[float, float, float, float] | None = None
    for df in _DF_GRID:
        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            mu_n, log_scale_n = float(theta[0]), float(theta[1])
            scale_n = float(np.exp(log_scale_n))
            if not np.isfinite(scale_n) or scale_n == 0.0:
                return float("inf"), np.array([np.nan, np.nan])
            z = (q - mu_n) / scale_n
            ll_vec = _t_logpdf_vec(q, df, mu_n, scale_n)
            score_mu = np.sum((df + 1.0) * z / (df + z * z)) / scale_n
            score_ls = np.sum(df * (z * z - 1.0) / (df + z * z))
            return -float(np.sum(ll_vec)), -np.array([score_mu, score_ls])

        starts = (
            np.array([float(np.median(q)), 0.0]),
            np.array([float(np.mean(q)), mlog(float(np.std(q)))]),
        )
        for start in starts:
            result = minimize(
                objective,
                start,
                method="BFGS",
                jac=True,
                options={"gtol": score_tol * v.size, "maxiter": int(max_iter)},
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                continue
            nll, gradient = objective(result.x)
            mean_score = float(np.max(np.abs(gradient))) / v.size
            if not np.isfinite(nll) or mean_score > score_tol:
                continue
            mu_n, log_scale_n = map(float, result.x)
            scale_n = float(np.exp(log_scale_n))
            mu = center + spread * mu_n
            scale = spread * scale_n
            if not (np.isfinite(mu) and np.isfinite(scale) and scale > 0.0):
                continue
            ll = -nll - v.size * mlog(spread)
            if best is None or ll > best[0]:
                best = (ll, df, mu, scale)
    if best is None:
        _LAST_T_FIT_STATUS.set(
            {"reason": "NON_CONVERGED", "n": int(v.size), "max_iter": int(max_iter)}
        )
        return None
    _LAST_T_FIT_STATUS.set(
        {"reason": "converged", "n": int(v.size), "df": best[1], "log_likelihood": best[0]}
    )
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
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_research_only(
        {
            "ts_betti_1_max_persistence",
            "ts_persistence_diagram_shift",
            "ts_fisher_information_shift",
        }
    )


_register_surface()
