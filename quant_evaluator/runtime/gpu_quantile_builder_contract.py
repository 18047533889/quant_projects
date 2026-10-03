"""Admission contract for GPU quantile builder parameters.

This validates explicit CUDA requests only.  It does not certify or influence
the public ``auto`` backend policy.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from quant_evaluator.contracts.errors import (
    InvalidContractError,
    UnsupportedMetricError,
)

_ALLOWED_PARAMETERS = frozenset({"n_quantiles", "min_assets", "window_size"})
_MINIMUMS = {"n_quantiles": 2, "min_assets": 2}


def normalize_gpu_quantile_builder_parameters(parameters, canonical_metrics) -> dict:
    """Validate and normalize optional GPU quantile-builder parameters.

    Empty input is a no-op and does not import the optional GPU adapter.  A
    non-empty request must contain a supported shape-consumer metric; other
    metrics may coexist in the request and continue to use their own routing.
    """
    if parameters is None or (isinstance(parameters, Mapping) and not parameters):
        return {}
    if not isinstance(parameters, Mapping):
        raise InvalidContractError("GPU quantile builder parameters must be a mapping")

    keys = set(parameters)
    if "window_size" in keys:
        raise UnsupportedMetricError("GPU quantile builder does not support window_size")
    unknown = keys - _ALLOWED_PARAMETERS
    if unknown:
        raise InvalidContractError(
            "Unknown GPU quantile builder parameter(s): "
            + ", ".join(sorted(map(str, unknown)))
        )

    normalized = {}
    for name, minimum in _MINIMUMS.items():
        if name not in parameters:
            continue
        value = parameters[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < minimum):
            raise InvalidContractError(
                f"{name} must be an integer greater than or equal to {minimum}"
            )
        normalized[name] = int(value)

    if normalized:
        # Keep the quantile backend optional at import time.  In particular,
        # empty/default CPU requests never initialize the CUDA adapter.
        from quant_evaluator.runtime.gpu_quantile_shape_adapter import (
            GPU_PROFILE_QUANTILE_METRICS,
        )

        if not set(canonical_metrics).intersection(GPU_PROFILE_QUANTILE_METRICS):
            raise UnsupportedMetricError(
                "GPU quantile builder parameters require a GPU-profile quantile metric"
            )
    return normalized
