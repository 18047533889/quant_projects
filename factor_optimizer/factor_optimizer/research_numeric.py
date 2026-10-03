"""Numerically guarded reductions for research diagnostics."""
from __future__ import annotations

import math

import numpy as np


def guarded_finite_mean(values, axis=None, keepdims=False):
    """Use NumPy's ordinary mean, repairing only overflow of finite slices.

    Nonfinite source slices retain NumPy's NaN/Inf result semantics. The slower
    QE exact-rational mean is imported lazily and used only when an otherwise
    all-finite reduction overflowed.
    """
    array = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.mean(array, axis=axis, keepdims=keepdims)
    if array.size == 0 or np.all(np.isfinite(result)):
        return result

    if axis is None:
        slices = array.reshape(1, -1)
        reduced_axes = tuple(range(array.ndim))
        output_axes = ()
    else:
        axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
        reduced_axes = tuple(sorted(a % array.ndim for a in axes))
        output_axes = tuple(a for a in range(array.ndim) if a not in reduced_axes)
        moved = np.transpose(array, output_axes + reduced_axes)
        slice_count = math.prod(array.shape[a] for a in output_axes) if output_axes else 1
        reduced_count = math.prod(array.shape[a] for a in reduced_axes)
        slices = moved.reshape(slice_count, reduced_count)

    reduced_result = np.asarray(result)
    if keepdims and axis is not None:
        reduced_result = np.squeeze(reduced_result, axis=reduced_axes)
    flat_result = reduced_result.reshape(-1).copy()
    bad = np.flatnonzero(~np.isfinite(flat_result))
    if bad.size:
        from quant_evaluator.metrics.quantile_numeric import stable_finite_mean
        for row_index in bad:
            row = slices[row_index]
            if row.size and np.isfinite(row).all():
                flat_result[row_index] = stable_finite_mean(row)
    repaired = flat_result.reshape(reduced_result.shape)
    if keepdims and axis is not None:
        for reduced_axis in reduced_axes:
            repaired = np.expand_dims(repaired, axis=reduced_axis)
    if axis is None and keepdims:
        repaired = repaired.reshape((1,) * array.ndim)
    if axis is None and not keepdims:
        return repaired.reshape(()).item()
    return repaired
