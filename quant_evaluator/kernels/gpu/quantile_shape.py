"""GPU quantile-shape metrics derived from per-quantile returns (spec §29).

Input is a ``(n_quantiles, F)`` matrix of time-averaged per-quantile returns
per factor — small relative to raw factor tensors — so the point is
cross-factor batch vectorization over F, matching the CPU reference in
:mod:`quant_evaluator.metrics.quantile_shape`.

CuPy is imported lazily so the module loads on a CPU-only environment.
"""

from __future__ import annotations

import numpy as np
from numbers import Integral


def _import_cp():
    import cupy as cp
    return cp


def _as_matrix(qr):
    cp = _import_cp()
    if isinstance(qr, cp.ndarray) and qr.device.id != cp.cuda.Device().id:
        raise ValueError("quantile profile must be on the current CUDA device")
    m = cp.asarray(qr, dtype=cp.float64)
    if m.ndim == 1:
        m = m[:, None]
    return cp.ascontiguousarray(m)


def _repair_linear(values, result, metric, workspace_bytes):
    from quant_evaluator.kernels.gpu.shape_linear_numeric import repair_shape_linear_gpu
    repaired = repair_shape_linear_gpu(
        values, result, metric, workspace_bytes=workspace_bytes,
    )
    return repaired.get()


def _finite_columns(m):
    cp = _import_cp()
    return cp.sum(cp.isfinite(m), axis=0) >= 2


def quantile_monotonicity(qr, *, return_device=False):
    cp = _import_cp()
    m = _as_matrix(qr)
    nq, F = m.shape
    out = cp.full(F, cp.nan)
    if nq < 2:
        return out if return_device else out.get()
    finite = cp.isfinite(m)
    pairs = finite[:-1, :] & finite[1:, :]
    n = cp.sum(pairs, axis=0).astype(cp.float64)
    inc = cp.sum((m[1:, :] > m[:-1, :]) & pairs, axis=0)
    ratio = inc / cp.maximum(n, 1)
    out = cp.where(n >= 1, ratio, cp.nan)
    return out if return_device else out.get()


def daily_quantile_monotonicity(daily_qr, *, return_device=False):
    """Per-date finite-adjacent monotonicity fractions over ``(T,Q,F)``."""
    cp = _import_cp()
    values = cp.asarray(daily_qr, dtype=cp.float64)
    if values.ndim != 3:
        raise ValueError("daily quantile returns must have shape (T,Q,F)")
    pairs = cp.isfinite(values[:, :-1, :]) & cp.isfinite(values[:, 1:, :])
    denominator = cp.sum(pairs, axis=1)
    increasing = cp.sum((values[:, 1:, :] > values[:, :-1, :]) & pairs, axis=1)
    out = cp.where(denominator > 0, increasing / cp.maximum(denominator, 1), cp.nan)
    return out if return_device else out.get()


def quantile_curvature(qr, *, workspace_bytes: int = 1 << 30):
    workspace_bytes = _validate_workspace_bytes(workspace_bytes)
    cp = _import_cp()
    m = _as_matrix(qr)
    nq, F = m.shape
    out = cp.full(F, cp.nan)
    if nq < 3:
        return out.get()
    finite = cp.isfinite(m)
    interior = finite[1:-1, :] & finite[:-2, :] & finite[2:, :]
    d2 = m[2:, :] - 2.0 * m[1:-1, :] + m[:-2, :]
    cnt = cp.sum(interior, axis=0).astype(cp.float64)
    s = cp.sum(cp.where(interior, d2, 0.0), axis=0)
    out = cp.where(cnt >= 1, s / cnt, cp.nan)
    return _repair_linear(m, out, "curvature", workspace_bytes)

def quantile_tail_asymmetry(qr, *, workspace_bytes: int = 1 << 30):
    workspace_bytes = _validate_workspace_bytes(workspace_bytes)
    cp = _import_cp()
    m = _as_matrix(qr)
    nq, F = m.shape
    out = cp.full(F, cp.nan)
    if nq < 3:
        return out.get()
    mid = nq // 2
    ok = cp.isfinite(m[0, :]) & cp.isfinite(m[-1, :]) & cp.isfinite(m[mid, :])
    val = (m[-1, :] - m[mid, :]) - (m[mid, :] - m[0, :])
    out = cp.where(ok, val, cp.nan)
    return _repair_linear(m, out, "tail_asymmetry", workspace_bytes)

def quantile_adjacent_spread(qr, *, workspace_bytes: int = 1 << 30):
    workspace_bytes = _validate_workspace_bytes(workspace_bytes)
    cp = _import_cp()
    m = _as_matrix(qr)
    nq, F = m.shape
    out = cp.full(F, cp.nan)
    if nq < 2:
        return out.get()
    finite = cp.isfinite(m)
    pairs = finite[:-1, :] & finite[1:, :]
    n = cp.sum(pairs, axis=0).astype(cp.float64)
    s = cp.sum(cp.where(pairs, cp.abs(m[1:, :] - m[:-1, :]), 0.0), axis=0)
    out = cp.where(n >= 1, s / n, cp.nan)
    return _repair_linear(m, out, "spread", workspace_bytes)

def quantile_extreme_cliff(qr, *, workspace_bytes: int = 1 << 30):
    workspace_bytes = _validate_workspace_bytes(workspace_bytes)
    cp = _import_cp()
    m = _as_matrix(qr)
    nq, F = m.shape
    out = cp.full(F, cp.nan)
    if nq < 2:
        return out.get()
    ok = (
        cp.isfinite(m[0, :]) & cp.isfinite(m[1, :])
        & cp.isfinite(m[-1, :]) & cp.isfinite(m[-2, :])
    )
    cliff_top = m[-1, :] - m[-2, :]
    cliff_bottom = m[1, :] - m[0, :]
    val = (cliff_top + cliff_bottom) / 2.0
    out = cp.where(ok, val, cp.nan)
    return _repair_linear(m, out, "extreme_cliff", workspace_bytes)

def top_quantile_cliff(qr, *, workspace_bytes: int = 1 << 30):
    workspace_bytes = _validate_workspace_bytes(workspace_bytes)
    cp = _import_cp()
    m = _as_matrix(qr)
    nq, F = m.shape
    out = cp.full(F, cp.nan)
    if nq < 2:
        return out.get()
    ok = cp.isfinite(m[-1, :]) & cp.isfinite(m[-2, :])
    val = m[-1, :] - m[-2, :]
    out = cp.where(ok, val, cp.nan)
    return _repair_linear(m, out, "top_cliff", workspace_bytes)

def bottom_quantile_cliff(qr, *, workspace_bytes: int = 1 << 30):
    workspace_bytes = _validate_workspace_bytes(workspace_bytes)
    cp = _import_cp()
    m = _as_matrix(qr)
    nq, F = m.shape
    out = cp.full(F, cp.nan)
    if nq < 2:
        return out.get()
    ok = cp.isfinite(m[1, :]) & cp.isfinite(m[0, :])
    val = m[1, :] - m[0, :]
    out = cp.where(ok, val, cp.nan)
    return _repair_linear(m, out, "bottom_cliff", workspace_bytes)


def _validate_workspace_bytes(workspace_bytes):
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, Integral) or workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be a positive integer")
    return int(workspace_bytes)
