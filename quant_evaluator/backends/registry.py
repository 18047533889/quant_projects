"""
Backend registry for managing computational kernels across backends.

Allows registration of backend-specific implementations and automatic fallback.
"""

from typing import Dict, Callable, Optional, Any
import functools

# Import BackendType from selector to ensure single source of truth
from quant_evaluator.backends.selector import BackendType


class BackendRegistry:
    """
    Registry for backend-specific kernel implementations.

    Manages multiple implementations of the same operation across different
    backends with automatic fallback to reference implementations.

    Examples
    --------
    >>> registry = BackendRegistry()
    >>> registry.register("ic_batch", BackendType.NUMPY, numpy_ic_impl)
    >>> registry.register("ic_batch", BackendType.NUMBA, numba_ic_impl)
    >>>
    >>> # Get best available implementation
    >>> impl = registry.get("ic_batch", preferred=BackendType.NUMBA)
    """

    def __init__(self):
        """Initialize empty registry."""
        self._registry: Dict[str, Dict[BackendType, Callable]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        operation: str,
        backend: BackendType,
        implementation: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Register a backend implementation for an operation.

        Parameters
        ----------
        operation : str
            Operation name (e.g., "ic_batch", "quantile_binning")
        backend : BackendType
            Backend type
        implementation : callable
            Implementation function
        metadata : dict, optional
            Additional metadata (version, requirements, etc.)
        """
        if operation not in self._registry:
            self._registry[operation] = {}

        self._registry[operation][backend] = implementation

        if metadata is not None:
            if operation not in self._metadata:
                self._metadata[operation] = {}
            self._metadata[operation][backend.value] = metadata

    def get(
        self,
        operation: str,
        preferred: Optional[BackendType] = None,
        fallback_order: Optional[list[BackendType]] = None,
    ) -> Optional[Callable]:
        """
        Get implementation for an operation.

        Parameters
        ----------
        operation : str
            Operation name
        preferred : BackendType, optional
            Preferred backend
        fallback_order : list of BackendType, optional
            Custom fallback order. Defaults to [NUMBA, NUMPY]

        Returns
        -------
        callable or None
            Implementation function, or None if not found

        Examples
        --------
        >>> impl = registry.get("ic_batch", preferred=BackendType.CUPY)
        >>> # Falls back to NUMBA or NUMPY if CUPY not available
        """
        if operation not in self._registry:
            return None

        available = self._registry[operation]

        # Try preferred backend first
        if preferred is not None and preferred in available:
            return available[preferred]

        # Fallback order
        if fallback_order is None:
            fallback_order = [
                BackendType.CUPY,
                BackendType.NUMBA,
                BackendType.POLARS,
                BackendType.NUMPY,
            ]

        for backend in fallback_order:
            if backend in available:
                return available[backend]

        return None

    def list_operations(self) -> list[str]:
        """List all registered operations."""
        return list(self._registry.keys())

    def list_backends(self, operation: str) -> list[BackendType]:
        """
        List available backends for an operation.

        Parameters
        ----------
        operation : str
            Operation name

        Returns
        -------
        list of BackendType
            Available backends
        """
        if operation not in self._registry:
            return []
        return list(self._registry[operation].keys())

    def get_metadata(self, operation: str, backend: BackendType) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific implementation.

        Parameters
        ----------
        operation : str
            Operation name
        backend : BackendType
            Backend type

        Returns
        -------
        dict or None
            Metadata, or None if not found
        """
        if operation in self._metadata and backend.value in self._metadata[operation]:
            return self._metadata[operation][backend.value].copy()
        return None


# Global registry instance
_global_registry = BackendRegistry()


def register_backend(
    operation: str,
    backend: BackendType,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to register a backend implementation.

    Parameters
    ----------
    operation : str
        Operation name
    backend : BackendType
        Backend type
    metadata : dict, optional
        Additional metadata

    Examples
    --------
    >>> @register_backend("ic_batch", BackendType.NUMBA)
    ... def numba_ic_batch(factors, labels):
    ...     # Numba implementation
    ...     pass
    """
    def decorator(func: Callable) -> Callable:
        _global_registry.register(operation, backend, func, metadata)
        return func
    return decorator


def get_backend(
    operation: str,
    preferred: Optional[BackendType] = None,
) -> Optional[Callable]:
    """
    Get implementation from global registry.

    Parameters
    ----------
    operation : str
        Operation name
    preferred : BackendType, optional
        Preferred backend

    Returns
    -------
    callable or None
        Implementation function

    Examples
    --------
    >>> ic_impl = get_backend("ic_batch", preferred=BackendType.NUMBA)
    >>> if ic_impl is not None:
    ...     result = ic_impl(factors, labels)
    """
    return _global_registry.get(operation, preferred)


def auto_backend(operation: str):
    """
    Decorator that automatically selects backend based on input size.

    Uses BackendSelector to choose optimal backend dynamically.

    Examples
    --------
    >>> @auto_backend("ic_batch")
    ... def ic_batch_auto(factors, labels, **kwargs):
    ...     # This will automatically select and call the best backend
    ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from quant_evaluator.backends.selector import BackendSelector

            # Extract shape from first argument (assumed to be factor_values)
            if len(args) > 0 and hasattr(args[0], 'shape'):
                shape = args[0].shape
                if len(shape) == 3:  # (T, N, F)
                    T, N, F = shape
                    selector = BackendSelector()

                    if operation == "ic_batch":
                        backend = selector.select_for_ic(T, N, F)
                    elif operation == "quantile_binning":
                        backend = selector.select_for_quantile(T, N, F)
                    else:
                        backend = BackendType.NUMPY

                    # Get implementation
                    impl = _global_registry.get(operation, preferred=backend)
                    if impl is not None:
                        return impl(*args, **kwargs)

            # Fallback to original function
            return func(*args, **kwargs)

        return wrapper
    return decorator
