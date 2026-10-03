"""Device-only dispatch for the six linear quantile-profile summaries.

Daily grouping and temporal averaging belong to GPUExecutor's request-local
cache. This module maps registry IDs to canonical CUDA kernels and packs
small tile outputs for one host transfer. It never creates a CPU profile.
"""
from __future__ import annotations

from types import MappingProxyType

_KERNEL_NAMES = MappingProxyType({
    "quantile_curvature": "quantile_curvature",
    "quantile_tail_asymmetry": "quantile_tail_asymmetry",
    "quantile_adjacent_spread": "quantile_adjacent_spread",
    "quantile_extreme_cliff": "quantile_extreme_cliff",
    "top_quantile_cliff": "top_quantile_cliff",
    "bottom_quantile_cliff": "bottom_quantile_cliff",
})
GPU_LINEAR_QUANTILE_METRICS = frozenset(_KERNEL_NAMES)
GPU_PROFILE_QUANTILE_METRICS = GPU_LINEAR_QUANTILE_METRICS | {"quantile_monotonicity"}


def compute_linear_quantile_shape_device(metric_id, profile, *, workspace_bytes):
    """Compute one admitted registry metric, retaining its result on device.

    The caller admits workspace through DeviceEvaluationSession and owns the
    final host materialization. Unknown IDs fail before CUDA is imported.
    Kernels may read bounded scalar error/admission flags, never fall back to
    CPU numerical computation or transfer a full input/result panel here.
    """
    if metric_id not in _KERNEL_NAMES:
        from quant_evaluator.contracts.errors import UnsupportedMetricError
        raise UnsupportedMetricError("metric has no linear CUDA quantile-shape adapter")
    from quant_evaluator.kernels.gpu import quantile_shape
    kernel = getattr(quantile_shape, _KERNEL_NAMES[metric_id])
    return kernel(profile, workspace_bytes=workspace_bytes, return_device=True)


def materialize_linear_quantile_shape_results(results, *, to_host):
    """Pack finite-scalar values/counts into one bounded transfer per tile.

    Caller admits packing workspace before entering. Counts are zero/one and
    exactly representable as float64; the host cast only restores their typed
    metadata representation, not a CPU numerical metric implementation.
    Mapping insertion order is preserved. No CUDA import for an empty input.
    """
    if not results:
        return {}, {}
    if set(results) - GPU_LINEAR_QUANTILE_METRICS:
        raise ValueError("unknown linear CUDA quantile-shape result")
    import cupy as cp
    import numpy as np
    names = tuple(results)
    arrays = tuple(results[name] for name in names)
    width = arrays[0].shape if isinstance(arrays[0], cp.ndarray) else None
    if width is None or len(width) != 1 or any(
        not isinstance(a, cp.ndarray) or a.shape != width
        or a.dtype != cp.float64 or a.device.id != arrays[0].device.id
        for a in arrays
    ):
        raise ValueError("shape outputs must be same-device float64 vectors")
    values = cp.stack(arrays, axis=0)
    packed = cp.concatenate((values, cp.isfinite(values).astype(cp.float64)), axis=0)
    host = np.asarray(to_host(packed))
    expected = (2 * len(names), width[0])
    if host.shape != expected or host.dtype != np.float64:
        raise ValueError("invalid packed CUDA shape output materialization")
    scalars = {name: host[i].copy() for i, name in enumerate(names)}
    counts = {name: host[len(names) + i].astype(np.int64)
              for i, name in enumerate(names)}
    return scalars, counts
