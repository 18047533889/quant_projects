# -*- coding: utf-8 -*-
"""Topological persistence-entropy operators (2026-08 geometry/math expansion).

Each operator builds the Takens delay embedding of the trailing window, computes
a persistence diagram, and returns the normalised Shannon entropy of the
lifetime distribution ``ell = death - birth``:

    p = ell / sum(ell),   H = -sum(p log p) / log(#pairs).

A high persistence entropy means the topological features have a broad range
of lifetimes (heterogeneous structure); a low value means the signal is
dominated by a single long-lived feature (e.g. one persistent cycle).

Homology degree is a **semantic part of the canonical** — 3rd-round audit P0-03.
``ts_persistence_entropy_h1`` uses the pure-numpy Vietoris-Rips H1 reduction
(``cleaned_operators.advanced_topology``); ``ts_persistence_entropy_h0`` uses a
single-linkage union-find over the points sorted by ascending pairwise distance
(every point born at 0, dying at the merge distance).  There is **no silent
H1→H0 fallback**: the two canonicals are environment-independent and mean the
same thing on every machine.  If the H1 kernel is unavailable the H1 operator
raises ``RuntimeError`` (production unsupported) instead of quietly switching
homology degree.

Both operators are trailing-window, prefix-causal and deterministic.  A window
that yields no persistence pairs emits NaN.  The CURRENT bar is always required
(P0-7): a missing current observation emits NaN — the persistence diagram must
never be built from the finite past alone and emit a stale-history factor.

Missing-value policy — CURRENT_ROW_REQUIRED (M-4xx)
----------------------------------------------------
These two canonicals belong to the CURRENT_ROW_REQUIRED class of the topology
family: the current observation must be finite or the output is NaN (enforced
in ``_persistence_entropy_series``).  This is the semantic opposite of
``cleaned_operators.advanced_topology`` (``ts_betti_1_max_persistence`` /
``ts_persistence_diagram_shift`` / ``ts_fisher_information_shift``), which are
HISTORICAL_STATE_ALLOWED and may emit a window state from the finite past.  The
two classes are explicitly distinguished so a stale-history factor is never
silently mis-classified.

Parameter governance (M-2xx)
----------------------------
``window`` / ``tau`` / ``dim`` are fully ParamSpec'd (tau/dim are
ESTIMATOR_RESOLUTION, searchable=False) with the relational feasibility
``window - (dim-1)*tau >= 3`` enforced at binding — a combination that cannot
form a non-empty delay embedding is rejected before running, never
``int()``-truncated.
"""
from __future__ import annotations

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
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12

# M-2xx: Takens embedding resolution (tau / dim) is an ESTIMATOR knob
# (searchable=False); window is the HORIZON.  ``window - (dim-1)*tau >= 3`` is
# the feasibility floor for a non-empty delay embedding — rejected at binding.
_PERSISTENCE_ENTROPY_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=6, param_role=ParamRole.HORIZON, searchable=True),
    "tau": ParamSpec(
        dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=1
    ),
    "dim": ParamSpec(
        dtype=int, min=2, max=6,
        param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=3,
    ),
}
_PERSISTENCE_ENTROPY_RELATIONAL: list[RelationalParamSpec] = [
    RelationalParamSpec(
        "window - (dim - 1) * tau >= 3",
        "ts_persistence_entropy requires window-(dim-1)*tau >= 3 delay embeddings "
        "(window={window}, dim={dim}, tau={tau})",
    ),
]

try:  # H1 Rips persistence kernel from the topology reference module.
    from factor_engine.cleaned_operators.advanced_topology import _rips_h1_pairs, _takens_points

    _HAVE_H1 = True
except Exception:  # pragma: no cover - optional kernel, no silent fallback.
    _rips_h1_pairs = None  # type: ignore
    _takens_points = None  # type: ignore
    _HAVE_H1 = False


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
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
            "topology", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", "current_row_required",
            f"signature:{','.join(params)}->series", "domain:topology",
            f"unit:{unit}", f"cost:{cost}",
        ],
        param_specs=dict(param_specs) if param_specs else {},
        relational_specs=list(relational_specs) if relational_specs else [],
    )


def _h0_pairs_union_find(points: np.ndarray) -> list[tuple[float, float]]:
    """H0 persistence pairs via single-linkage union-find."""
    n = int(points.shape[0])
    if n < 2:
        return []
    d = np.sqrt(np.maximum(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1), 0.0))
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(d[i, j]), i, j))
    edges.sort(key=lambda e: (e[0], e[1], e[2]))
    parent = list(range(n))
    size = [1] * n
    comps = n
    pairs: list[tuple[float, float]] = []

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for dist, i, j in edges:
        if comps <= 1:
            break
        ri, rj = _find(i), _find(j)
        if ri == rj:
            continue
        if size[ri] < size[rj]:
            ri, rj = rj, ri
        # component rj dies at this merge distance; it was born at 0.
        pairs.append((0.0, dist))
        parent[rj] = ri
        size[ri] += size[rj]
        comps -= 1
    return pairs


def _h0_persistence_pairs(chunk: np.ndarray, tau: int, dim: int) -> list[tuple[float, float]]:
    """H0 persistence pairs of the robust-normalised Takens embedding."""
    finite = chunk[np.isfinite(chunk)]
    if finite.size < 4:
        return []
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * 1.4826
    spread = mad if mad > 1e-12 else float(np.std(finite))
    if spread <= 1e-12:
        return []
    z = (chunk - med) / spread
    n = int(chunk.shape[0])
    lag = tau * (dim - 1)
    if n < lag + 3:
        return []
    pts = np.stack([z[s - lag : s + 1 : tau] for s in range(lag, n)], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 4:
        return []
    return _h0_pairs_union_find(pts)


def _h1_persistence_pairs(chunk: np.ndarray, tau: int, dim: int) -> list[tuple[float, float]]:
    """H1 Rips persistence pairs.  Raises if the H1 kernel is unavailable —
    the canonical must not silently change homology degree (P0-03)."""
    if not _HAVE_H1:
        raise RuntimeError(
            "ts_persistence_entropy_h1 requires the H1 Rips kernel "
            "(cleaned_operators.advanced_topology) which is unavailable in this "
            "environment. Use ts_persistence_entropy_h0, or install the kernel."
        )
    pts = _takens_points(chunk, tau, dim)
    if pts is None:
        return []
    return _rips_h1_pairs(pts)


def _persistence_pairs(chunk: np.ndarray, tau: int, dim: int, h0: bool) -> list[tuple[float, float]]:
    if h0:
        return _h0_persistence_pairs(chunk, tau, dim)
    return _h1_persistence_pairs(chunk, tau, dim)


def _persistence_entropy(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return np.nan
    lifetimes = np.asarray([death - birth for birth, death in pairs], dtype=float)
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if lifetimes.size == 0:
        return np.nan
    # R6-118: zero-lifetime features (birth == death) contribute zero to the
    # probability mass but still increase ``count``, silently changing the
    # entropy normalisation of the features that DO persist.  Drop them BEFORE
    # computing p / count / entropy so the measure reflects the genuine
    # persistence structure only.
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > _EPS)]
    if lifetimes.size == 0:
        return np.nan
    total = float(lifetimes.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.nan
    p = np.clip(lifetimes / total, 0.0, 1.0)
    ent = -float(np.sum(p * np.log(np.clip(p, 1e-15, 1.0))))
    count = int(lifetimes.size)
    if count > 1:
        ent = ent / np.log(count)
    return float(max(ent, 0.0))


def _trailing_rows(x: np.ndarray, w: int) -> np.ndarray:
    """Left-aligned trailing-window matrix ``(rows, w)``, NaN padded.

    Row ``t`` holds ``x[max(0, t-w+1) : t+1]`` in its first ``min(t+1, w)``
    slots — exactly the chunk ``_persistence_entropy_series`` used to slice out
    per row, so the delay cloud of every row is one gather with the
    row-independent index map ``point + coord*tau``.
    """
    rows = x.shape[0]
    tcol = np.arange(rows, dtype=np.int64)[:, None]
    widx = np.maximum(0, tcol - w + 1) + np.arange(w, dtype=np.int64)[None, :]
    return np.where(widx <= tcol, x[np.clip(widx, 0, rows - 1)], np.nan)


def _robust_z_rows(zp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row robust z of a padded window matrix.

    Mirrors ``median / MAD*1.4826 (std fallback) / z = (x-med)/spread`` of the
    per-window reference, batched; returns ``(z, spread, n_finite, finite)``.
    Rows whose window is empty/length-1 are given the degenerate values the
    reference would have produced (spread 0 / NaN) and are masked by the caller.
    """
    finite = np.isfinite(zp)
    n_fin = finite.sum(axis=1)
    pf = np.where(finite, zp, np.nan)
    guarded = np.where((n_fin > 0)[:, None], pf, 0.0)
    med = np.nanmedian(guarded, axis=1)
    mad = np.nanmedian(np.where(n_fin[:, None] > 0, np.abs(pf - med[:, None]), 0.0), axis=1) * 1.4826
    spread = np.where(mad > _EPS, mad, np.nan)
    fallback = ~(mad > _EPS)          # also catches a NaN MAD
    use_std = fallback & (n_fin >= 2)
    if use_std.any():
        spread[use_std] = np.nanstd(guarded[use_std], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (zp - med[:, None]) / spread[:, None]
    return z, spread, n_fin, finite


def _entropy_from_lifetimes(life: np.ndarray) -> np.ndarray:
    """Batched :func:`_persistence_entropy` for an already-collected lifetime grid.

    ``life`` holds the persistence lifetimes of one window per row (non-lifetime
    slots are non-finite).  Reproduces the reference exactly: finite filter, then
    the ``> _EPS`` zero-lifetime drop *before* the probability normalisation and
    the count, the ``clip(p, 1e-15, 1)`` log guard and the ``log(count)`` scale.
    """
    keep = np.isfinite(life) & (life > _EPS)
    count = keep.sum(axis=1)
    vals = np.where(keep, life, 0.0)
    total = vals.sum(axis=1)
    safe = np.where((total > 0.0) & np.isfinite(total), total, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.clip(vals / safe[:, None], 0.0, 1.0)
        ent = -np.sum(p * np.log(np.clip(p, 1e-15, 1.0)), axis=1)
        ent = np.where(count > 1, ent / np.log(np.where(count > 1, count, 2)), ent)
    ent = np.maximum(ent, 0.0)
    good = (count > 0) & np.isfinite(total) & (total > 0.0)
    return np.where(good, ent, np.nan)


def _h0_entropy_batch(z: np.ndarray, valid: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """Batched H0 persistence entropy: single-linkage merge distances per row.

    H0 with every point born at height 0 means the pair ``(0, d)`` recorded at
    each single-linkage merge, so the lifetime multiset is exactly the multiset
    of MST edge weights (a sorted-weight invariant of the graph: Kruskal and Prim
    trees of the same cloud share it, so the union-find scan over the
    ``(dist, i, j)``-sorted edge list can be replaced by Prim without changing
    the entropy).  Prim runs *batched over rows*: one ``argmin`` picks the next
    point, one column gather of the precomputed ``(rows, W, W)`` squared-distance
    block relaxes the frontier, and ``np.fmin`` keeps the NaN padding inert.
    """
    rows, n_pts = valid.shape
    out = np.full(rows, np.nan, dtype=float)
    if n_pts < 2:
        return out
    idx = np.arange(rows)
    chunk = max(1, int(2.0e6 // max(n_pts * n_pts, 1)))
    for c0 in range(0, rows, chunk):
        c1 = min(c0 + chunk, rows)
        zc = z[c0:c1]
        vc = valid[c0:c1]
        cw = c1 - c0
        # ||z_k - z_a||^2 = sum_c (z[k + c*tau] - z[a + c*tau])^2 from one
        # (cw, n_pts + lag, n_pts + lag) squared-difference block: every
        # coordinate of the delay cloud is the same 1-D window shifted by c*tau.
        width = n_pts + tau * (dim - 1)
        a2 = (zc[:, :width, None] - zc[:, None, :width]) ** 2
        d2 = a2[:, 0:n_pts, 0:n_pts].copy()
        for c in range(1, dim):
            d2 += a2[:, c * tau : c * tau + n_pts, c * tau : c * tau + n_pts]
        np.sqrt(d2, out=d2)
        rix = np.arange(cw)
        keys = np.full((cw, n_pts), np.inf)
        intree = ~vc                       # padded points are never selectable
        first = np.argmax(vc, axis=1)
        anyvalid = vc.any(axis=1)
        # Prim with the first observation of each row as the root: its own key is
        # 0 (not a merge), so the n_pts recorded keys are
        # [0, merge_1, ..., merge_{n_pts-1}] and the merge weights are [1:].
        keys[rix[anyvalid], first[anyvalid]] = 0.0
        picked = np.full((cw, n_pts), np.inf)
        for step in range(n_pts):
            pick = np.argmin(keys, axis=1)
            picked[:, step] = keys[rix, pick]
            intree[rix, pick] = True
            np.fmin(keys, d2[rix, :, pick], out=keys)
            keys[intree] = np.inf
        out[c0:c1] = _entropy_from_lifetimes(picked[:, 1:])
    return out


try:  # stay in lock-step with the reference cap (M-3xx _MAX_POINTS).
    from factor_engine.cleaned_operators.advanced_topology import _MAX_POINTS as _H1_POINT_CAP
except Exception:  # pragma: no cover - reference module unavailable (H1 already raises)
    _H1_POINT_CAP = 12

_RIPS_TABLE_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _rips_tables(n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(edges_ij, triangles_ijk, triangle_edge_ids)`` of the ``n_max`` simplex.

    Cached: the padded H1 cloud always has ``_H1_POINT_CAP`` points, so the
    combinatorial tables are built once per process instead of per window.
    """
    tab = _RIPS_TABLE_CACHE.get(n_max)
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
        _RIPS_TABLE_CACHE[n_max] = tab
    return tab


def _high_bit_u64(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Highest set bit index of a two-word (lo, hi) F2 bitmask.

    Exact in the presence of a zero high word: ``frexp`` yields a
    ``floor(log2)`` candidate that float rounding can only overshoot by one
    position, and a single membership test repairs it — cheaper than a
    bit-length binary search, and without the false positives of ``log2`` on
    masks whose top bit sits at 2**62.  Undefined for an all-zero mask (the
    caller filters those out).
    """
    hi_nz = hi != 0
    word = np.where(hi_nz, hi, lo)
    _, expo = np.frexp(word.astype(np.float64))
    bit = expo.astype(np.int64) - 1
    shift = np.clip(bit, 0, 63).astype(np.uint64)
    set_ = ((word >> shift) & np.uint64(1)) != 0
    bit = np.where(set_, bit, bit - 1)
    return np.where(hi_nz, bit + 64, bit)


def _rips_h1_entropy(points: np.ndarray, pmask: np.ndarray) -> np.ndarray:
    """Batched ``_rips_h1_pairs`` + ``_persistence_entropy`` over padded clouds.

    ``points`` is ``(rows, cap, dim)`` with the valid points of each row in its
    first ``pmask.sum(1)`` slots.  The Zomorodian–Carlsson reduction is data
    dependent, so instead of one Python ``while`` per window it runs as a
    *batched* reduction: every row advances through its own triangle order in
    lock-step (one pass per triangle position), with the still-unreduced rows
    compacted after each XOR so the pivot search only touches them.  Columns are
    two-word F2 bitmasks over the edge ranks, so the pivot (highest set bit) is
    computed exactly.

    Padded points are given ``inf`` edge lengths, hence their edges rank after
    every real edge and their triangles filter last: they can never steal a real
    pivot row, and the (necessarily non-finite) lifetimes they produce are
    dropped by :func:`_entropy_from_lifetimes`.
    """
    rows, n_max, _ = points.shape
    out = np.full(rows, np.nan, dtype=float)
    if n_max < 4 or rows == 0:
        return out
    pairs_ij, tris, tri_edges = _rips_tables(n_max)
    ei, ej = pairs_ij[:, 0], pairs_ij[:, 1]
    ti, tj, tk = tris[:, 0], tris[:, 1], tris[:, 2]
    diff = points[:, :, None, :] - points[:, None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    ok_pair = pmask[:, :, None] & pmask[:, None, :]
    dist = np.where(ok_pair, dist, np.inf)           # padded simplexes filter last
    edge = dist[:, ei, ej]                           # (rows, n_edges)
    tri_flt = np.maximum(np.maximum(dist[:, ti, tj], dist[:, ti, tk]), dist[:, tj, tk])
    n_edges = edge.shape[1]
    shp_e = edge.shape
    # edges sorted by (distance, i, j): the reference's ``edges.sort()``.
    eorder = np.lexsort(
        (np.broadcast_to(ej, shp_e), np.broadcast_to(ei, shp_e), edge), axis=1
    )
    e_sorted = np.take_along_axis(edge, eorder, axis=1)
    erank = np.empty(shp_e, dtype=np.int64)
    np.put_along_axis(erank, eorder, np.arange(n_edges, dtype=np.int64)[None, :], axis=1)
    # triangles sorted by (filtration, i, j, k): the reference's ``triangles.sort()``.
    shp_t = tri_flt.shape
    torder = np.lexsort(
        (
            np.broadcast_to(tk, shp_t),
            np.broadcast_to(tj, shp_t),
            np.broadcast_to(ti, shp_t),
            tri_flt,
        ),
        axis=1,
    )
    tri_rank = erank[:, tri_edges]                   # (rows, n_tri, 3)
    rix = np.arange(rows)
    piv_lo = np.zeros((rows, n_edges), dtype=np.uint64)
    piv_hi = np.zeros((rows, n_edges), dtype=np.uint64)
    piv_ok = np.zeros((rows, n_edges), dtype=bool)
    life = np.full((rows, shp_t[1]), np.nan, dtype=float)
    for t in range(shp_t[1]):
        tri = torder[:, t]
        rr = tri_rank[rix, tri]
        clo = np.zeros(rows, dtype=np.uint64)
        chi = np.zeros(rows, dtype=np.uint64)
        for s in range(3):
            r = rr[:, s]
            low = r < 64
            clo |= np.where(
                low,
                np.left_shift(np.uint64(1), (r & np.int64(63)).astype(np.uint64)),
                np.uint64(0),
            )
            chi |= np.where(
                low,
                np.uint64(0),
                np.left_shift(
                    np.uint64(1), np.clip(r - np.int64(64), 0, 63).astype(np.uint64)
                ),
            )
        flt = tri_flt[rix, tri]
        stored_hi = np.full(rows, -1, dtype=np.int64)
        act = np.flatnonzero((clo | chi) != 0)
        while act.size:
            alo = clo[act]
            ahi = chi[act]
            hi = _high_bit_u64(alo, ahi)
            free = ~piv_ok[act, hi]                  # pivot row still unclaimed
            if free.any():
                st = act[free]
                hst = hi[free]
                piv_lo[st, hst] = alo[free]
                piv_hi[st, hst] = ahi[free]
                piv_ok[st, hst] = True
                stored_hi[st] = hst                  # born at this edge, dies at flt
            keep = ~free
            if not keep.any():
                break
            ct = act[keep]
            phi = hi[keep]
            clo[ct] = alo[keep] ^ piv_lo[ct, phi]
            chi[ct] = ahi[keep] ^ piv_hi[ct, phi]
            act = ct[(clo[ct] | chi[ct]) != 0]        # reduced to zero -> no H1 class
        st = np.flatnonzero(stored_hi >= 0)
        if st.size:
            life[st, t] = flt[st] - e_sorted[st, stored_hi[st]]
    return _entropy_from_lifetimes(life)


def _h1_entropy_batch(cloud: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Batched H1 persistence entropy of the Rips complex of every row's cloud.

    Reproduces ``_takens_points`` — round to 9 decimals, stable unique keeping
    the FIRST occurrence (time order), deterministic time-order decimation to
    ``_H1_POINT_CAP`` — and then ``_rips_h1_pairs`` for all rows at once.  The
    deduplication is one ``lexsort`` with the time index as the final tie-break
    (numpy's row ``unique`` sorts lexicographically and reports the first
    occurrence, so re-sorting those occurrences by index restores time order),
    and the decimation is the reference's ``round(linspace(0, n-1, cap))``
    skeleton written as a batched gather.
    """
    rows, m_pts, dim = cloud.shape
    out = np.full(rows, np.nan, dtype=float)
    if m_pts < 4 or rows == 0:
        return out
    cap = _H1_POINT_CAP
    rnd = np.round(cloud, 9)
    keys = (np.broadcast_to(np.arange(m_pts, dtype=float), (rows, m_pts)),) + tuple(
        np.where(valid, rnd[:, :, c], np.nan) for c in range(dim - 1, -1, -1)
    )
    perm = np.lexsort(keys, axis=1)
    masked = np.where(valid[..., None], rnd, np.nan)
    srt = np.take_along_axis(
        masked, np.broadcast_to(perm[:, :, None], masked.shape), axis=1
    )
    pos = np.arange(m_pts, dtype=np.int64)[None, :]
    n_pts = valid.sum(axis=1)
    # M-3xx: stable unique preserving FIRST occurrence = lexicographic order with
    # the time index as the final tie-break, then re-sort the survivors by that
    # (minimum) index.  Invalid points have all-NaN keys, so they sort last and
    # never enter the compacted cloud.
    starts = np.ones((rows, m_pts), dtype=bool)
    starts[:, 1:] = ~(srt[:, 1:, :] == srt[:, :-1, :]).all(axis=2)
    gstart = starts & (pos < n_pts[:, None])
    # ``perm[j]`` is the cloud index of the j-th lexicographic slot and ``starts``
    # marks the slot holding each unique row's FIRST occurrence, so ordering the
    # start slots by that index restores time order.
    fidx = np.where(gstart, perm, m_pts)
    order2 = np.argsort(fidx, axis=1, kind="stable")
    found = np.take_along_axis(gstart, order2, axis=1)
    n_uniq = found.sum(axis=1)
    uni = np.take_along_axis(
        srt, np.broadcast_to(order2[:, :, None], srt.shape), axis=1
    )                                                         # time-ordered uniques
    # _decimate_time_order: keep round(linspace(0, n-1, cap]), then unique them.
    span = np.maximum(n_uniq - 1, 0).astype(np.float64)
    node = np.arange(cap, dtype=np.float64)[None, :] * (span / (cap - 1.0))[:, None]
    node[:, -1] = span                                        # linspace endpoint
    last = np.maximum(n_uniq - 1, 0)[:, None]
    ksrt = np.sort(np.clip(np.rint(node).astype(np.int64), 0, last), axis=1)
    kmask = np.ones_like(ksrt, dtype=bool)
    kmask[:, 1:] = ksrt[:, 1:] != ksrt[:, :-1]
    pts = np.full((rows, cap, dim), np.nan)
    vmask = np.zeros((rows, cap), dtype=bool)
    flat = kmask.ravel()
    rsel = np.repeat(np.arange(rows), cap)[flat]
    csel = (np.cumsum(kmask, axis=1) - 1).ravel()[flat]
    ssel = ksrt.ravel()[flat]
    pts[rsel, csel] = uni[rsel, ssel]
    vmask[rsel, csel] = True
    vmask &= (n_uniq >= 4)[:, None]                  # _takens_points: >= 4 points
    return _rips_h1_entropy(pts, vmask)


def _persistence_entropy_series(x2d: np.ndarray, window: int, tau: int, dim: int, h0: bool) -> np.ndarray:
    """Persistence-lifetime entropies for every trailing window, fully batched.

    R62: the per-row Python loop, the ``(W choose 2)`` edge tuple list and its
    sort (H0) and the per-window Vietoris-Rips reduction loop (H1) are gone.  The
    shared prefix — trailing window matrix, robust z, delay cloud, CURRENT_ROW_
    REQUIRED gate — is one batched pass, and each homology degree has its own
    batched kernel (``_h0_entropy_batch`` / ``_h1_entropy_batch``).  The H1
    point-cloud sampling (stable-unique + time decimation) lives in
    ``_h1_entropy_batch``; the reduction in ``_rips_h1_entropy``.
    """
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    lag = tau * (dim - 1)
    n_pts = w - lag
    if rows == 0 or cols == 0 or n_pts <= 0 or w < lag + 3:
        return out
    if not h0 and not _HAVE_H1:
        # P0-03: never silently downgrade the homology degree.
        raise RuntimeError(
            "ts_persistence_entropy_h1 requires the H1 Rips kernel "
            "(cleaned_operators.advanced_topology) which is unavailable in this "
            "environment. Use ts_persistence_entropy_h0, or install the kernel."
        )
    posidx = np.arange(n_pts)[:, None] + tau * np.arange(dim)[None, :]
    length = np.minimum(np.arange(rows) + 1, w)
    for c in range(cols):
        x = np.asarray(x2d[:, c], dtype=float)
        z, spread, n_fin, _finite = _robust_z_rows(_trailing_rows(x, w))
        cloud = z[:, posidx]
        valid = np.isfinite(cloud).all(axis=2)
        n_cloud = valid.sum(axis=1)
        # CURRENT_ROW_REQUIRED (P0-7) + the reference's feasibility gates: at
        # least 4 finite observations, a non-degenerate robust spread, at least
        # `lag + 3` observations in the window and 4 finite embedding rows.
        gate = (
            np.isfinite(x) & (n_fin >= 4) & (length >= lag + 3)
            & (spread > _EPS) & (n_cloud >= 4)
        )
        if not gate.any():
            continue
        if h0:
            ent = _h0_entropy_batch(z, valid, dim, tau)
        else:
            ent = _h1_entropy_batch(cloud, valid)
        out[:, c] = np.where(gate, ent, np.nan)
    return out


def _register(name: str, description: str, *, h0: bool) -> SeriesOperator:
    suffix = "h0" if h0 else "h1"
    canonical = f"ts_persistence_entropy_{suffix}"

    @register_operator(
        name=canonical,
        category="topology",
        business_category="topology",
        canonical=canonical,
        source="topology_ext",
    )
    class _PersistenceEntropy(SeriesOperator):
        metadata = _metadata(
            canonical,
            description,
            ["x", "window", "tau", "dim"],
            unit="entropy",
            cost=8,
            param_specs=_PERSISTENCE_ENTROPY_SPECS,
            relational_specs=_PERSISTENCE_ENTROPY_RELATIONAL,
        )

        def _calculate_series(
            self, x: pd.DataFrame, window: int = 120, tau: int = 1, dim: int = 3, **_: Any
        ) -> pd.DataFrame:
            # M-2xx: strict validation — fractional/NaN/bool rejected, never
            # ``int()``-truncated; relational feasibility enforced at binding.
            w = strict_int(window, "window", lower=6)
            t = strict_int(tau, "tau", lower=1)
            d = strict_int(dim, "dim", lower=2, upper=6)
            if w - (d - 1) * t < 3:
                raise ValueError(
                    "ts_persistence_entropy requires window-(dim-1)*tau >= 3 "
                    f"(window={w}, dim={d}, tau={t})"
                )
            return frame_like(x, _persistence_entropy_series(x.to_numpy(dtype=float), w, t, d, h0))

    return _PersistenceEntropy


_register(
    "ts_persistence_entropy_h0",
    "持久性寿命熵（H0：单链 union-find，零维连接分量的合并距离）。环境无关。",
    h0=True,
)
_register(
    "ts_persistence_entropy_h1",
    "持久性寿命熵（H1：Vietoris-Rips 一维环；内核不可用时抛错，不静默降级）。",
    h0=False,
)

_NEW_CANONICALS = (
    "ts_persistence_entropy_h0",
    "ts_persistence_entropy_h1",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    # R5-50: live extend mutator, never a frozenset reassignment.
    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
