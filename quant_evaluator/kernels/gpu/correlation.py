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
    x0 = cp.where(finite, x, 0.0)
    y0 = cp.where(finite, yb, 0.0)
    sx = cp.sum(x0, axis=2)
    sy = cp.sum(y0, axis=2)
    sxx = cp.sum(x0 * x0, axis=2)
    syy = cp.sum(y0 * y0, axis=2)
    sxy = cp.sum(x0 * y0, axis=2)
    # division by zero yields inf/nan; we mask those via zero_var / insufficient below
    num = n * sxy - sx * sy
    denom = (n * sxx - sx * sx) * (n * syy - sy * sy)
    denom_safe = cp.maximum(denom, 0.0)
    ic = num / cp.sqrt(denom_safe)
    insufficient = n < min_obs
    zero_var = (n * sxx - sx * sx <= 0) | (n * syy - sy * sy <= 0)
    ic = cp.where(insufficient | zero_var | (denom_safe == 0.0), cp.nan, ic)
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
    rx = batched_rank(x_rank, pct=False)          # (T,F,N)
    ry = batched_rank(y_rank, pct=False)          # (T,F,N)

    # distinct-level floor over the PAIRWISE-FINITE set (CPU reference)
    from quant_evaluator.kernels.gpu.rank import batched_distinct_level_count
    dlx = batched_distinct_level_count(x_rank)  # (T*F,)
    dly = batched_distinct_level_count(y_rank)  # (T*F,)
    T, F, N = x.shape
    dlx = dlx.reshape(T, F)
    dly = dly.reshape(T, F)
    # Mathematical definition, not a quality gate. Binary/ordinal signals
    # retain average-tie Spearman; confidence/applicability is separate.
    min_levels = 2
    # rx and ry are (T,F,N); _pairwise_finite_sums handles y.ndim==3
    ic, n = _pairwise_finite_sums(rx, ry, min_obs)
    low_levels = (dlx < min_levels) | (dly < min_levels)
    ic = cp.where(low_levels, cp.nan, ic)
    return ic, n
