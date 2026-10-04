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
from quant_evaluator.kernels.gpu.pearson_repair import repair_unsafe_pearson_rows
from quant_evaluator.kernels.gpu.pearson_centered_fused import fused_centered_sums
from quant_evaluator.kernels.gpu.pearson_endpoint import certify_gpu_pearson_endpoints


def _import_cp():
    import cupy as cp
    return cp




def _pairwise_finite_sums(x, y, min_obs: int, *, bounded: bool = False,
                          exact_endpoints: bool = False):
    """Compute Pearson sums over pairwise-finite (T,F,N) x and y.

    y may be (T,N) (broadcast to (T,1,N)) or (T,F,N). Returns
    (ic, valid_counts) with NaN where insufficient/zero-variance.
    """
    cp = _import_cp()
    if x.ndim != 3:
        raise ValueError("Pearson factor input must have shape (T,F,N)")
    if y.ndim == 2:
        if y.shape != (x.shape[0], x.shape[2]):
            raise ValueError("Pearson label shape must match factor time and asset axes")
        yb = y[:, None, :]  # (T,1,N)
    elif y.ndim == 3:
        if (y.shape[0] != x.shape[0] or y.shape[2] != x.shape[2]
                or y.shape[1] not in (1, x.shape[1])):
            raise ValueError("Pearson label tensor must have shape (T,1,N) or (T,F,N)")
        yb = y
    else:
        raise ValueError("Pearson labels must have shape (T,N) or (T,1,N)/(T,F,N)")
    if x.shape[2] == 0:
        return (
            cp.full(x.shape[:2], cp.nan, dtype=cp.float64),
            cp.zeros(x.shape[:2], dtype=cp.int32),
        )
    finite = cp.isfinite(x) & cp.isfinite(yb)
    supported_dtype = (
        x.dtype in (cp.dtype(cp.float32), cp.dtype(cp.float64))
        and yb.dtype in (cp.dtype(cp.float32), cp.dtype(cp.float64))
    )
    if not bounded and supported_dtype:
        # Fused two-pass row reduction avoids materializing dx, dy, dx*dx,
        # dy*dy and dx*dy. bounded=True is Spearman's rank path and deliberately
        # keeps the legacy reduction order below.
        n_count, sx, sy, mx, my, vx, vy, cov = fused_centered_sums(x, yb, finite)
        n = n_count.astype(cp.float64)
    else:
        n = cp.sum(finite, axis=2, dtype=cp.float64)  # (T, F)
        # Keep the legacy arithmetic for Spearman and unsupported dtypes.
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

    # Float16/float32 inputs stay inside float64's squared-norm range, including
    # their smallest subnormals. The arithmetic path avoids unsafe-row repair;
    # Pearson endpoint candidate discovery still introduces a rare sync below.
    if bounded or (
        x.dtype in (cp.float16, cp.float32)
        and yb.dtype in (cp.float16, cp.float32)
    ):
        ic = cp.where((n < min_obs) | (vx <= 0) | (vy <= 0), cp.nan, ic)
        if exact_endpoints:
            ic = certify_gpu_pearson_endpoints(ic, x, yb, finite)
        return ic, n.astype(cp.int32)

    # The direct product may overflow/underflow even when each variance is
    # representable. Route those rows through scale-normalized arithmetic.
    tiny = np.finfo(np.float64).tiny
    large = np.sqrt(np.finfo(np.float64).max)
    unsafe = (
        (vx <= 0) | (vy <= 0) | (~cp.isfinite(vx)) | (~cp.isfinite(vy))
        | (vx < tiny) | (vy < tiny) | (vx > large) | (vy > large)
        | (~cp.isfinite(denom)) | (denom <= 0) | (~cp.isfinite(ic))
        # Centering around a rounded mean loses low bits when the offset is
        # enormous compared with the cross-sectional standard deviation.
        | (cp.abs(mx) > cp.sqrt(vx / cp.maximum(n - 1.0, 1.0)) * 1e6)
        | (cp.abs(my) > cp.sqrt(vy / cp.maximum(n - 1.0, 1.0)) * 1e6)
    )
    ic = repair_unsafe_pearson_rows(
        x, yb, finite, unsafe, ic,
        n_factors=x.shape[1], y_is_broadcast=(yb.shape[1] == 1),
    )
    if exact_endpoints:
        ic = certify_gpu_pearson_endpoints(ic, x, yb, finite)

    ic = cp.where(n < min_obs, cp.nan, ic)
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
    return _pairwise_finite_sums(x, y, min_obs, exact_endpoints=True)


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
    ic, n = _pairwise_finite_sums(rx, ry, min_obs, bounded=True)
    low_levels = (dlx < min_levels) | (dly < min_levels)
    ic = cp.where(low_levels, cp.nan, ic)
    return ic, n
