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
    actual int64 sort indices and group keys. Average ties use bincount
    with float64 reductions; these are included in the workspace estimate.
  - Rank results are written into one preallocated output across chunks.
"""

from __future__ import annotations

import numpy as np

# Bound each chunk's working set to ~1GB of device memory.  The dominant
# per-chunk temporaries include int64 argsort indices/keys, sorted values,
# and float64 segmented reduction arrays.
_MAX_CHUNK_BYTES = 1 << 30  # 1 GiB


def _import_cp():
    import cupy as cp
    return cp


def _chunk_rows(n: int, dtype_bytes: int) -> int:
    """Max rows (flattened cross-sections) per chunk for a given N and dtype.

    Each row needs roughly: sorted values (dtype_bytes*N) + argsort indices
    (8*N) + int64/float64 (N) keys and reduction temporaries. We budget a
    conservative multiplier so the argsort thrust scratch (which can be a
    few x the input) stays inside the chunk bound.
    """
    per_row = (dtype_bytes + 8 + 8 + 8 + 16) * n  # values, int64 indices/keys and reductions
    # argsort scratch can be several x the input; use a 4x safety factor.
    per_row *= 4
    if per_row > _MAX_CHUNK_BYTES:
        raise MemoryError("single rank cross-section exceeds workspace budget")
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
    order = cp.argsort(x_safe, axis=1, kind="stable")  # (R, N), actual backend index dtype
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
    if method != "average":
        raise ValueError("batched_rank supports only method='average'")
    cp = _import_cp()
    x = cp.asarray(values)
    if (isinstance(axis_n, (bool, np.bool_)) or not isinstance(axis_n, (int, np.integer))
            or not -x.ndim <= axis_n < x.ndim):
        raise ValueError("batched_rank axis_n is outside input axes")
    original_axis = int(axis_n) % x.ndim
    x = cp.moveaxis(x, original_axis, -1)
    shape = x.shape
    n = shape[-1]
    if x.size == 0:
        return cp.moveaxis(cp.empty(shape, dtype=cp.float64), -1, original_axis)
    flat = x.reshape(-1, n)
    R = flat.shape[0]

    nan_mask = ~cp.isfinite(flat)
    dtype_bytes = flat.dtype.itemsize
    chunk = _chunk_rows(n, dtype_bytes)

    if R <= chunk:
        ranks = _rank_chunk(flat, nan_mask, cp)
    else:
        ranks = cp.empty(flat.shape, dtype=cp.float64)
        for s in range(0, R, chunk):
            e = min(s + chunk, R)
            ranks[s:e] = _rank_chunk(flat[s:e], nan_mask[s:e], cp)

    if pct:
        vc = cp.sum(~nan_mask, axis=1, keepdims=True)
        ranks = ranks / cp.maximum(vc, 1.0)

    return cp.moveaxis(ranks.reshape(shape), -1, original_axis)


def batched_distinct_level_count(values):
    """Number of distinct finite values per cross-section (flattened rows,)."""
    cp = _import_cp()
    x = cp.asarray(values)
    if x.ndim < 1:
        raise ValueError("distinct-level input requires an asset axis")
    if x.size == 0:
        return cp.zeros(int(np.prod(x.shape[:-1])), dtype=cp.int32)
    flat = x.reshape(-1, x.shape[-1])
    R, n = flat.shape
    nan_mask = ~cp.isfinite(flat)
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


def batched_quantile_assignment(values, n_quantiles: int = 10, method: str = "max", *, workspace_bytes=None):
    """Assign each element to a quantile bucket (0..n_quantiles-1) per row.

    Mirrors CPU ``assign_quantiles`` (QE-Q-P0-001/002): percentile boundaries
    computed on the sorted finite values with the same interpolation formula,
    then searchsorted binning honoring the tie policy (MAX → side='right').
    NaN → -1.
    """
    if method not in ("min", "max"):
        raise ValueError("batched_quantile_assignment method supports min/max only")
    if isinstance(n_quantiles, (bool, np.bool_)) or not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 2:
        raise ValueError("n_quantiles must be an integer >= 2")
    cp = _import_cp()
    x = cp.asarray(values)
    if x.ndim < 1:
        raise ValueError("quantile input requires an asset axis")
    shape = x.shape
    if x.size == 0:
        return cp.full(shape, -1, dtype=cp.int32)
    flat = x.reshape(-1, shape[-1])
    R, n = flat.shape
    budget = _MAX_CHUNK_BYTES if workspace_bytes is None else workspace_bytes
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("workspace_bytes must be a positive integer")
    # Includes actual int64 sort indices, values, masks, boundary arrays,
    # scatter results and conservative sorting scratch, excludes caller-owned
    # input/output (admitted separately by the device session).
    row_bytes = 4 * ((x.dtype.itemsize + 8 + 8 + 8 + 4 + 4) * n + 64 * n_quantiles)
    if row_bytes > budget:
        raise MemoryError("single quantile cross-section exceeds workspace budget")
    chunk = max(1, int(budget // row_bytes))
    if R > chunk:
        result = cp.empty(flat.shape, dtype=cp.int32)
        for start in range(0, R, chunk):
            result[start:start+chunk] = batched_quantile_assignment(flat[start:start+chunk], n_quantiles, method, workspace_bytes=budget)
        return result.reshape(shape)
    nan_mask = ~cp.isfinite(flat)
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
    # Stream boundaries; never materialize R x N x Q.
    q = cp.zeros(v_fin.shape, dtype=cp.int32)
    for boundary in range(n_quantiles - 1):
        edge = boundaries[:, boundary, None]
        q += v_fin >= edge if method == "max" else v_fin > edge
    q = cp.clip(q, 0, n_quantiles - 1)
    # write back only finite positions
    fin_ok = ~nan_mask[R_ok]
    out[R_ok] = cp.where(fin_ok, q, -1)
    return out.reshape(shape)


def batched_rank_weights(values):
    """Average-tie rank proxy weights normalized to sum one, CPU authority."""
    cp = _import_cp()
    x = cp.asarray(values)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    finite = cp.isfinite(flat)
    ranks = batched_rank(cp.where(finite, flat, cp.nan), pct=False)
    total = cp.nansum(ranks, axis=1, keepdims=True)
    w = cp.where(finite & (finite.sum(axis=1, keepdims=True) >= 2), ranks / total, cp.nan)
    return w.reshape(shape)
