"""
Automatic backend selection for factor_preprocess.

Detects available acceleration backends (numba, cupy, polars) and selects
the optimal backend based on data size and operation type.
"""

from factor_preprocess.backends.selector import (
    BackendSelector,
    BackendType,
    BackendCapabilities,
    get_backend_capabilities,
    benchmark_backends,
)
from factor_preprocess.backends.registry import (
    BackendRegistry,
    register_backend,
    get_backend,
)

__all__ = [
    "BackendSelector",
    "BackendType",
    "BackendCapabilities",
    "get_backend_capabilities",
    "benchmark_backends",
    "BackendRegistry",
    "register_backend",
    "get_backend",
    "polars_backend",
]
