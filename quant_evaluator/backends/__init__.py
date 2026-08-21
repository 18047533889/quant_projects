"""
Backend implementations for quant_evaluator.

Provides CPU and GPU computation backends with automatic fallback handling.
"""

from typing import Optional

# Backend availability flags
_cupy_available = False
_gpu_available = False

try:
    import cupy as cp
    _cupy_available = True
    # Check if GPU is actually available
    try:
        cp.cuda.Device(0).compute_capability
        _gpu_available = True
    except (cp.cuda.runtime.CUDARuntimeError, AttributeError):
        _gpu_available = False
except ImportError:
    pass


def is_cupy_available() -> bool:
    """Check if CuPy is installed."""
    return _cupy_available


def is_gpu_available() -> bool:
    """Check if GPU is available for computation."""
    return _gpu_available


def get_available_backends() -> list:
    """Get list of available computation backends."""
    backends = ["numpy"]
    if _cupy_available:
        backends.append("cupy")
    return backends


def get_default_backend() -> str:
    """Get the default backend based on availability."""
    return "cupy" if _gpu_available else "numpy"


class OptionalDependencyMissing(ImportError):
    """Raised when optional dependency is missing."""

    def __init__(self, package: str, feature: str):
        self.package = package
        self.feature = feature
        super().__init__(
            f"Optional dependency '{package}' is required for {feature}. "
            f"Install with: pip install {package}"
        )


# Import submodules for convenience
try:
    from quant_evaluator.backends.selector import (
        BackendSelector,
        BackendType,
        BackendCapabilities,
        get_backend_capabilities,
        benchmark_backends,
    )
    from quant_evaluator.backends.registry import (
        BackendRegistry,
        register_backend,
        get_backend,
    )

    __all__ = [
        "is_cupy_available",
        "is_gpu_available",
        "get_available_backends",
        "get_default_backend",
        "OptionalDependencyMissing",
        "BackendSelector",
        "BackendType",
        "BackendCapabilities",
        "get_backend_capabilities",
        "benchmark_backends",
        "BackendRegistry",
        "register_backend",
        "get_backend",
    ]
except ImportError:
    # Graceful degradation if submodules fail to import
    __all__ = [
        "is_cupy_available",
        "is_gpu_available",
        "get_available_backends",
        "get_default_backend",
        "OptionalDependencyMissing",
    ]
