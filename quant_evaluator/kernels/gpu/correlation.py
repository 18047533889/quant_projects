"""Batched Pearson / Spearman IC kernels (spec §12.1, §12.2).

Both operate on the GPU-friendly (T, F, N) layout and reuse the shared
batched rank primitive from :mod:`kernels.gpu.rank`.  Semantics preserved
from the CPU reference:
  - pairwise finite
  - min_assets floor
  - zero-variance rejection
  - Spearman: average-tie rank, distinct-level floor, constant rejection
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from quant_evaluator.kernels.gpu.rank import batched_rank


def _import_cp():
    import cupy as cp
    return cp


def _pairwise_finite_sums(x, y, min_obs: int):
    """Compute Pearson sums over pairwise-finite (T,F,N) x and y.

    y may be (T,N) (broadcast to (T,1,N)) or (T,F,N).  Returns
    (ic, valid_counts) with NaN where insufficient/zero-variance.
    """
    cp = _import_cp()
    if y.ndim == 2:
        yb = y[:, None, :]  # (T,1,N)
    else:
        yb = y  # (T,F,N)
    finite = cp.isfinite(x) & cp.isfinite(yb)
    n = cp.sum(finite, axis=2, dtype=cp.float64)  # (T, F)
    # Reduce in float64, then center before forming products. Raw moments in
    # float32 lose the variance of a large cross-section with a common offset;
    # even float64 raw moments can cancel when the offset is sufficiently large.
    sx = cp.sum(cp.where(finite, x, 0.0), axis=2, dtype=cp.float64)
    sy = cp.sum(cp.where(finite, yb, 0.0), axis=2, dtype=cp.float64)
    mx = sx / cp.maximum(n, 1.0)
    my = sy / cp.maximum(n, 1.0)
    dx = cp.where(finite, x - mx[:, :, None], 0.0)
    dy = cp.where(finite, yb - my[:, :, None], 0.0)
    vx = cp.sum(dx * dx, axis=2, dtype=cp.float64)
    vy = cp.sum(dy * dy, axis=2, dtype=cp.float64)
    cov = cp.sum(dx * dy, axis=2, dtype=cp.float64)
    denom = cp.sqrt(vx * vy)
    ic = cov / denom
    ic = cp.where((n < min_obs) | (vx <= 0) | (vy <= 0), cp.nan, ic)
    return ic, n.astype(cp.int32)


def batched_pearson_ic(
    factor_values,
    label_values,
    min_obs: int = 20,
    factor_validity=None,
    label_validity=None,
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """Batched Pearson IC over (T, F, N) factors vs (T, N) labels.

    Returns (ic (T,F), valid_counts (T,F)) on device.
    """
    cp = _import_cp()
    x = cp.asarray(factor_values)
    y = cp.asarray(label_values)
    if x.ndim == 3 and x.shape[1] != 1 and y.ndim == 2:
        pass  # (T,F,N) vs (T,N)
    elif x.ndim == 2:
        x = x[:, None, :]  # (T,1,N)
    if factor_validity is not None:
        fv = cp.asarray(factor_validity)
        x = cp.where(fv, x, cp.nan)
    if label_validity is not None:
        lv = cp.asarray(label_validity)
        if lv.ndim == 1:
            lv = lv[:, None]
        y = cp.where(lv, y, cp.nan)
    return _pairwise_finite_sums(x, y, min_obs)


def batched_spearman_ic(
    factor_values,
    label_values,
    min_obs: int = 20,
    factor_validity=None,
    label_validity=None,
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """Batched Spearman RankIC over (T, F, N) factors vs (T, N) labels.

    Uses the shared batched average-tie rank; rejects constant cross-sections
    (distinct-level floor) and insufficient observations.
    """
    cp = _import_cp()
    x = cp.asarray(factor_values)
    y = cp.asarray(label_values)
    if x.ndim == 2:
        x = x[:, None, :]
    if factor_validity is not None:
        fv = cp.asarray(factor_validity)
        x = cp.where(fv, x, cp.nan)
    if label_validity is not None:
        lv = cp.asarray(label_validity)
        if lv.ndim == 1:
            lv = lv[:, None]
        y = cp.where(lv, y, cp.nan)

    # rank factors (T,F,N) and labels per (t,f) over the PAIRWISE-FINITE set.
    # CPU reference ranks over finite(x)&finite(y) for each (t,f) separately,
    # so the label rank tensor is per-factor (T,F,N).
    yb = y[:, None, :]  # (T,1,N)
    pairwise = cp.isfinite(x) & cp.isfinite(yb)  # (T,F,N)
    x_rank = cp.where(pairwise, x, cp.nan)
    y_rank = cp.where(pairwise, yb, cp.nan)  # (T,F,N)
    rx, dlx = batched_rank(x_rank, pct=False, return_distinct=True)  # (T,F,N), (T,F)
    ry, dly = batched_rank(y_rank, pct=False, return_distinct=True)  # (T,F,N), (T,F)

    # Distinct-level counts were collected during each rank's existing sort.
    T, F, N = x.shape
    # Mathematical definition, not a quality gate. Binary/ordinal signals
    # retain average-tie Spearman; confidence/applicability is separate.
    min_levels = 2
    # rx and ry are (T,F,N); _pairwise_finite_sums handles y.ndim==3
    ic, n = _pairwise_finite_sums(rx, ry, min_obs)
    low_levels = (dlx < min_levels) | (dly < min_levels)
    ic = cp.where(low_levels, cp.nan, ic)
    return ic, n
