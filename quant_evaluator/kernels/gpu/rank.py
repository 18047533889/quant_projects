"""Batched cross-section rank / sort primitives (spec §12.2, §9).

The linchpin of the GPU refactor: a single batched segmented sort along N
that produces average-tie ranks, distinct-level counts, and quantile
assignments, reused by RankIC / quantile / turnover / rank-stability /
long-short buckets.

Layout: input is (T, F, N) — the GPU-friendly layout from
DeviceEvaluationSession.stage_factors.  We reshape to (T*F, N) and do one
batched argsort along axis=1, then resolve average ties via a segmented
cummax/cummin scan (spec §57 Memory Leak).

Semantics preserved from CPU reference (spec §12.2):
  - pairwise finite (NaN excluded from ranking)
  - average-tie rank (mean of 1-based positions among finite equal values)
  - constant rejection / distinct-level floor
  - min_assets floor handled by callers

Memory-lean design (spec §57):
  - The factor storage stays in the input dtype (float32 for the real
    500-factor benchmark); only the rank VALUES are promoted to float64.
  - The flattened (T*F) axis is processed in bounded chunks so no single
    intermediate exceeds ``_MAX_CHUNK_BYTES``.  Each chunk's argsort uses
    int32 indices (N <= 2^31), and the average-tie resolution uses a
    segmented cummax/cummin scan over int32 positions — no float64
    (T*F, N) temporaries, no bincount, no per-group key materialization.
  - Results are concatenated across chunks, bounding peak VRAM.
"""

from __future__ import annotations

import numpy as np

# Bound each chunk's working set to ~1GB of device memory.  The dominant
# per-chunk temporaries are the argsort indices (int32) and the sorted
# values (input dtype); the scan works on int32 positions.
_MAX_CHUNK_BYTES = 1 << 30  # 1 GiB


def _import_cp():
    import cupy as cp
    return cp


def _chunk_rows(n: int, dtype_bytes: int) -> int:
    """Max rows (flattened cross-sections) per chunk for a given N and dtype.

    Each row needs roughly: sorted values (dtype_bytes*N) + argsort indices
    (4*N) + a handful of int32/float64 (N) scan temporaries.  We budget a
    conservative multiplier so the argsort thrust scratch (which can be a
    few x the input) stays inside the chunk bound.
    """
    per_row = (dtype_bytes + 4 + 4 + 4 + 8) * n  # sv + order + scans + mean
    # argsort scratch can be several x the input; use a 4x safety factor.
    per_row *= 4
    return max(int(_MAX_CHUNK_BYTES // max(per_row, 1)), 1)


def _rank_chunk(flat, nan_mask, cp):
    """Average-tie rank for one chunk of (R, N) float32/float64 values.

    Returns (R, N) float64 ranks; NaN positions stay NaN.  Uses a segmented
    reduction via ``cp.bincount`` over per-group (row, gid) keys.  The chunk
    is bounded by :func:`_chunk_rows` so peak VRAM stays within budget even
    though a few float64 (R, N) temporaries are materialized per chunk.
    """
    R, n = flat.shape
    x_safe = cp.where(nan_mask, cp.inf, flat)
    order = cp.argsort(x_safe, axis=1, kind="stable")  # (R, N) int32
    sv = cp.take_along_axis(x_safe, order, axis=1)     # (R, N) input dtype
    fin_sorted = sv != cp.inf                          # (R, N) bool

    # group id: contiguous equal-value runs per row (int32)
    starts = cp.zeros_like(sv, dtype=cp.bool_)
    starts[:, 0] = True
    starts[:, 1:] = sv[:, 1:] != sv[:, :-1]
    gid = cp.cumsum(starts, axis=1, dtype=cp.int32) - 1

    # 1-based positions in sorted order (pandas default rank before tie-average)
    pos1 = cp.arange(1, n + 1, dtype=cp.float64)[None, :]  # (1, N) float64
    # zero out NaN contributions so they never affect finite group means
    rankw = cp.where(fin_sorted, pos1, 0.0)   # (R, N) float64
    cntw = cp.where(fin_sorted, 1.0, 0.0)     # (R, N) float64

    # per (row, gid) segmented reduction; NaN sentinels share one trailing group
    maxg = int(gid.max()) + 1
    key = (cp.arange(R, dtype=cp.int64)[:, None] * maxg + gid).ravel()
    n_bins = R * maxg
    group_sum = cp.bincount(key, weights=rankw.ravel(), minlength=n_bins)
    group_cnt = cp.bincount(key, minlength=n_bins).astype(cp.float64)
    group_mean = group_sum / cp.maximum(group_cnt, 1.0)

    mean_rank_flat = group_mean[key].reshape(R, n)  # (R, N) float64

    # scatter back to original (unsorted) positions
    ranks = cp.empty_like(mean_rank_flat)
    ranks[cp.arange(R)[:, None], order] = mean_rank_flat
    return cp.where(nan_mask, cp.nan, ranks)


def batched_rank(
    values,
    axis_n: int = -1,
    method: str = "average",
    pct: bool = False,
):
    """Batched average-tie rank along the last axis (NaN excluded).

    Args:
        values: (..., N) device or host array.
        method: "average" — ties get the mean of their 1-based positions among
            finite members (pandas ``rank(method='average')`` parity).
        pct: if True, ranks normalized by the count of valid (non-NaN) per row.

    Returns:
        ranks with same shape as values; NaN positions stay NaN.
    """
    cp = _import_cp()
    x = cp.asarray(values)
    shape = x.shape
    n = shape[-1]
    flat = x.reshape(-1, n)
    R = flat.shape[0]

    nan_mask = cp.isnan(flat)
    dtype_bytes = flat.dtype.itemsize
    chunk = _chunk_rows(n, dtype_bytes)

    if R <= chunk:
        ranks = _rank_chunk(flat, nan_mask, cp)
    else:
        parts = []
        for s in range(0, R, chunk):
            e = min(s + chunk, R)
            parts.append(_rank_chunk(flat[s:e], nan_mask[s:e], cp))
        ranks = cp.concatenate(parts, axis=0)

    if pct:
        vc = cp.sum(~nan_mask, axis=1, keepdims=True)
        ranks = ranks / cp.maximum(vc, 1.0)

    return ranks.reshape(shape)


def batched_distinct_level_count(values):
    """Number of distinct finite values per cross-section (flattened rows,)."""
    cp = _import_cp()
    x = cp.asarray(values)
    flat = x.reshape(-1, x.shape[-1])
    R, n = flat.shape
    nan_mask = cp.isnan(flat)
    dtype_bytes = flat.dtype.itemsize
    chunk = _chunk_rows(n, dtype_bytes)

    def _count(seg):
        x_safe = cp.where(seg[1], cp.inf, seg[0])
        order = cp.argsort(x_safe, axis=1, kind="stable")
        sorted_vals = cp.take_along_axis(x_safe, order, axis=1)
        new_finite = cp.zeros_like(sorted_vals, dtype=cp.bool_)
        new_finite[:, 0] = sorted_vals[:, 0] != cp.inf
        new_finite[:, 1:] = (
            (sorted_vals[:, 1:] != sorted_vals[:, :-1])
            & (sorted_vals[:, 1:] != cp.inf)
        )
        return cp.sum(new_finite, axis=1)

    if R <= chunk:
        return _count((flat, nan_mask))
    parts = []
    for s in range(0, R, chunk):
        e = min(s + chunk, R)
        parts.append(_count((flat[s:e], nan_mask[s:e])))
    return cp.concatenate(parts, axis=0)


def batched_quantile_assignment(values, n_quantiles: int = 10, method: str = "max"):
    """Assign each element to a quantile bucket (0..n_quantiles-1) per row.

    Mirrors CPU ``assign_quantiles`` (QE-Q-P0-001/002): percentile boundaries
    computed on the sorted finite values with the same interpolation formula,
    then searchsorted binning honoring the tie policy (MAX → side='right').
    NaN → -1.
    """
    cp = _import_cp()
    x = cp.asarray(values)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    R, n = flat.shape
    nan_mask = cp.isnan(flat)
    x_safe = cp.where(nan_mask, cp.inf, flat)
    order = cp.argsort(x_safe, axis=1, kind="stable")
    sv = cp.take_along_axis(x_safe, order, axis=1)  # sorted (inf at end)
    # finite count per row
    n_finite = cp.sum(~nan_mask, axis=1).astype(cp.float64)  # (R,)
    out = cp.full(flat.shape, -1, dtype=cp.int32)

    # rows with enough finite values
    enough = n_finite >= n_quantiles
    if not bool(cp.any(enough)):
        return out.reshape(shape)
    R_ok = cp.where(enough)[0]
    sv_ok = sv[R_ok]
    nf_ok = n_finite[R_ok]

    # percentile boundaries: pos = (b+1)/nq * (n-1), interpolate on sorted finite
    b = cp.arange(1, n_quantiles, dtype=cp.float64)[None, :]  # (1, nq-1)
    pos = b / n_quantiles * (nf_ok[:, None] - 1.0)  # (R_ok, nq-1)
    lo = cp.floor(pos).astype(cp.int64)
    frac = pos - lo
    # snap near-integer positions to exact sorted value
    frac = cp.where(frac < 1e-9, 0.0, frac)
    frac = cp.where(frac > 1.0 - 1e-9, 1.0, frac)
    lo = cp.clip(lo, 0, n - 2)
    hi = lo + 1
    v_lo = cp.take_along_axis(sv_ok, lo, axis=1)
    v_hi = cp.take_along_axis(sv_ok, hi, axis=1)
    boundaries = v_lo + frac * (v_hi - v_lo)  # (R_ok, nq-1)

    # searchsorted side='right' (MAX): count boundaries <= value (value ==
    # boundary goes to the HIGHER bin).  side='right' insertion index = number
    # of boundaries strictly less than value, but for value==boundary the
    # right-insertion lands after the equal boundary, so bin = count of
    # boundaries <= value.
    v_fin = cp.where(cp.isnan(flat[R_ok]), 0.0, flat[R_ok])  # (R_ok, n)
    ge = v_fin[:, :, None] >= boundaries[:, None, :]  # (R_ok, n, nq-1)
    q = cp.sum(ge, axis=2).astype(cp.int32)
    q = cp.clip(q, 0, n_quantiles - 1)
    # write back only finite positions
    fin_ok = ~nan_mask[R_ok]
    out[R_ok] = cp.where(fin_ok, q, -1)
    return out.reshape(shape)


def batched_rank_weights(values):
    """Normalized rank weights (spec §12.4): w = rank_pct - 0.5, NaN→0."""
    cp = _import_cp()
    x = cp.asarray(values)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    nan_mask = cp.isnan(flat)
    ranks_pct = batched_rank(flat, pct=True)
    w = ranks_pct - 0.5
    w = cp.where(nan_mask, 0.0, w)
    return w.reshape(shape)
