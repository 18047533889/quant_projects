"""Finite temporal means for GPU quantile metrics.

Ordinary inputs use CuPy reductions. Rows whose Float64 sum is unsafe are
repaired with the exact GPU quantile-mean kernel; values stay on device.
"""
from __future__ import annotations

from typing import Any


def finite_mean_axis0(values: Any, *, workspace_bytes: int):
    """Return Float64 finite means and counts along axis 0 on the GPU.

    ``workspace_bytes`` is the available working-set admission after the
    caller reserves result/count arrays through its device session. The helper
    checks its complete conservative bound before allocating.
    """
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - host without CUDA
        raise RuntimeError("CuPy is required for GPU temporal means") from exc

    if not isinstance(values, cp.ndarray):
        raise TypeError("values must be a CuPy device array; CPU fallback is unsupported")
    if values.ndim < 1 or values.dtype != cp.float64:
        raise TypeError("values must have at least one axis and dtype float64")
    if values.device.id != cp.cuda.Device().id:
        raise ValueError("values must be on the current CUDA device")
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int) or workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be a positive integer")

    time_count = int(values.shape[0])
    if time_count > 2147483647:
        raise ValueError("time axis exceeds the int32max admission bound")
    result_shape = tuple(int(dim) for dim in values.shape[1:])
    columns = 1
    for dim in result_shape:
        columns *= dim
    if columns == 0:
        return (cp.empty(result_shape, dtype=cp.float64),
                cp.empty(result_shape, dtype=cp.int64))

    required_working_set = 40 * time_count * columns + 16 * columns
    if workspace_bytes < required_working_set:
        raise MemoryError(
            f"GPU temporal mean requires {required_working_set} working-set bytes; "
            f"only {workspace_bytes} admitted"
        )

    flat = values.reshape((time_count, columns))
    finite = cp.isfinite(flat)
    counts = cp.sum(finite, axis=0, dtype=cp.int64)
    safe = cp.where(finite, flat, cp.float64(0.0))
    sums = cp.sum(safe, axis=0, dtype=cp.float64)
    means = cp.where(counts > 0, sums / cp.maximum(counts, 1), cp.nan)
    means = cp.asarray(means, dtype=cp.float64)

    labels = cp.ascontiguousarray(flat.T, dtype=cp.float64)
    label_finite = cp.isfinite(labels)
    bucket = cp.where(label_finite, 0, -1).astype(cp.int32, copy=False)
    means_2d = means.reshape((columns, 1))
    counts_2d = counts.reshape((columns, 1))

    del finite, safe, sums, label_finite, flat
    live_bytes = int(labels.nbytes + bucket.nbytes + means.nbytes + counts.nbytes)
    remaining = int(workspace_bytes) - live_bytes
    if remaining <= 0:
        raise MemoryError(
            f"GPU temporal mean has no scratch left after {live_bytes} live bytes"
        )

    from quant_evaluator.kernels.gpu.quantile_numeric import repair_quantile_means_gpu

    repair_quantile_means_gpu(
        bucket, labels, means_2d, counts_2d, 1,
        workspace_bytes=remaining,
    )
    return means.reshape(result_shape), counts.reshape(result_shape)


__all__ = ["finite_mean_axis0"]
