"""
Fast kernel implementations for factor preprocessing.

This module provides optimized implementations using:
- bottleneck for fast rank operations
- Numba JIT compilation for hot loops
- NumPy stride tricks for efficient rolling windows

All fast implementations must match reference_bridge.py exactly.
"""
import numpy as np
from typing import Literal, Optional

# Optional fast dependencies
try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False

try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ============================================================================
# Cross-sectional operations
# ============================================================================

def fast_cs_rank(
    values: np.ndarray,
    axis: int = -1,
    method: Literal["average", "min", "max", "dense", "ordinal"] = "average",
    pct: bool = False,
) -> np.ndarray:
    """
    Fast cross-sectional rank using bottleneck.

    Parameters
    ----------
    values : np.ndarray
        Input array, typically shape (dates, assets) or (assets,)
    axis : int
        Axis to rank along (default -1 for last axis)
    method : str
        Tie-breaking method. Only "average" is optimized.
    pct : bool
        If True, return percentile ranks [0, 1]

    Returns
    -------
    np.ndarray
        Ranked values. NaN inputs produce NaN outputs.

    Notes
    -----
    Uses bottleneck.rankdata for 10-100x speedup on large arrays.
    Falls back to scipy for non-average methods.
    """
    if values.size == 0:
        return values.copy()

    # Fast path: use bottleneck for average ranking
    if HAS_BOTTLENECK and method == "average":
        if values.ndim == 1:
            # 1D case
            mask = np.isfinite(values)
            if not np.any(mask):
                return np.full_like(values, np.nan, dtype=np.float64)

            result = np.full_like(values, np.nan, dtype=np.float64)
            # bottleneck.rankdata returns 1-based ranks
            result[mask] = bn.rankdata(values[mask])

            if pct:
                n_finite = np.sum(mask)
                if n_finite > 1:
                    result[mask] = (result[mask] - 1.0) / (n_finite - 1.0)
                else:
                    result[mask] = 0.5

            return result
        else:
            # Multi-dimensional: apply along axis
            return np.apply_along_axis(
                lambda arr: fast_cs_rank(arr, axis=-1, method=method, pct=pct),
                axis,
                values
            )
    else:
        # Fallback to reference implementation
        from factor_preprocess.kernels.reference_bridge import reference_cs_rank
        return reference_cs_rank(values, axis=axis, method=method, pct=pct)


def fast_cs_zscore(
    values: np.ndarray,
    axis: int = -1,
    ddof: int = 1,
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Fast cross-sectional z-score normalization.

    Parameters
    ----------
    values : np.ndarray
        Input array
    axis : int
        Axis to normalize along
    ddof : int
        Delta degrees of freedom
    constant_value : float
        Value to return for constant slices

    Returns
    -------
    np.ndarray
        Z-scored values. NaN inputs produce NaN outputs.

    Notes
    -----
    Uses vectorized NumPy operations for maximum speed.
    Typically 2-5x faster than naive loops due to memory layout.
    """
    if values.size == 0:
        return values.copy()

    mean = np.nanmean(values, axis=axis, keepdims=True)
    std = np.nanstd(values, axis=axis, keepdims=True, ddof=ddof)

    result = np.where(
        std > 0,
        (values - mean) / std,
        constant_value,
    )

    result = np.where(np.isnan(values), np.nan, result)

    return result


def fast_cs_demean(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Fast cross-sectional demean.

    Parameters
    ----------
    values : np.ndarray
        Input array
    axis : int
        Axis to demean along

    Returns
    -------
    np.ndarray
        Demeaned values. NaN inputs produce NaN outputs.

    Notes
    -----
    Simple vectorized operation, minimal overhead.
    """
    if values.size == 0:
        return values.copy()

    mean = np.nanmean(values, axis=axis, keepdims=True)
    return values - mean


# ============================================================================
# Rolling operations with stride tricks
# ============================================================================

def _strided_window_view(arr: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    Create a strided view for rolling window operations.

    Parameters
    ----------
    arr : np.ndarray
        Input array, shape (T, ...)
    window : int
        Window size
    axis : int
        Time axis

    Returns
    -------
    np.ndarray
        View of shape (T - window + 1, window, ...) with no data copy

    Notes
    -----
    Uses np.lib.stride_tricks.as_strided for zero-copy window views.
    Result shares memory with input - do not modify!
    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 supported")

    shape = arr.shape
    strides = arr.strides

    T = shape[0]
    if window > T:
        raise ValueError(f"window {window} > array length {T}")

    # New shape: (num_windows, window, *rest)
    new_shape = (T - window + 1, window) + shape[1:]
    # New strides: time stride for both first two dims
    new_strides = (strides[0], strides[0]) + strides[1:]

    return np.lib.stride_tricks.as_strided(
        arr,
        shape=new_shape,
        strides=new_strides,
        writeable=False
    )


def fast_rolling_mean(
    values: np.ndarray,
    window: int,
    axis: int = 0,
) -> np.ndarray:
    """
    Fast rolling mean using stride tricks.

    Parameters
    ----------
    values : np.ndarray
        Input array, shape (T, N) for time x assets
    window : int
        Window size
    axis : int
        Time axis (default 0)

    Returns
    -------
    np.ndarray
        Rolling mean. First `window-1` rows are NaN.

    Notes
    -----
    Uses stride tricks to avoid memory allocation.
    10-50x faster than naive loop for large arrays.
    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 supported for fast kernels")
    if window <= 0:
        raise ValueError("window must be positive")

    if values.size == 0:
        return values.copy()

    T = values.shape[0]
    if window > T:
        return np.full_like(values, np.nan, dtype=np.float64)

    result = np.full_like(values, np.nan, dtype=np.float64)

    # Create strided view: shape (T-window+1, window, *rest)
    windowed = _strided_window_view(values, window, axis=0)

    # Compute mean along window axis (axis=1)
    result[window - 1:] = np.nanmean(windowed, axis=1)

    return result


def fast_rolling_std(
    values: np.ndarray,
    window: int,
    axis: int = 0,
    ddof: int = 1,
) -> np.ndarray:
    """
    Fast rolling standard deviation using stride tricks.

    Parameters
    ----------
    values : np.ndarray
        Input array, shape (T, N) for time x assets
    window : int
        Window size
    axis : int
        Time axis (default 0)
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    np.ndarray
        Rolling std. First `window-1` rows are NaN.

    Notes
    -----
    Uses stride tricks for efficient window computation.
    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 supported for fast kernels")
    if window <= 0:
        raise ValueError("window must be positive")

    if values.size == 0:
        return values.copy()

    T = values.shape[0]
    if window > T:
        return np.full_like(values, np.nan, dtype=np.float64)

    result = np.full_like(values, np.nan, dtype=np.float64)

    # Create strided view
    windowed = _strided_window_view(values, window, axis=0)

    # Compute std along window axis (axis=1)
    result[window - 1:] = np.nanstd(windowed, axis=1, ddof=ddof)

    return result


# ============================================================================
# Numba-accelerated kernels
# ============================================================================

if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=False)
    def _numba_rolling_mean_2d(values: np.ndarray, window: int) -> np.ndarray:
        """
        Numba-accelerated rolling mean for 2D arrays.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N)
        window : int
            Window size

        Returns
        -------
        np.ndarray
            Rolling mean, shape (T, N)
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        for n in prange(N):
            for t in range(window - 1, T):
                window_sum = 0.0
                count = 0
                for i in range(t - window + 1, t + 1):
                    val = values[i, n]
                    if not np.isnan(val):
                        window_sum += val
                        count += 1

                if count > 0:
                    result[t, n] = window_sum / count
                # else remains NaN

        return result

    @jit(nopython=True, parallel=True, cache=False)
    def _numba_rolling_std_2d(
        values: np.ndarray,
        window: int,
        ddof: int = 1
    ) -> np.ndarray:
        """
        Numba-accelerated rolling std for 2D arrays.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N)
        window : int
            Window size
        ddof : int
            Delta degrees of freedom

        Returns
        -------
        np.ndarray
            Rolling std, shape (T, N)
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        for n in prange(N):
            for t in range(window - 1, T):
                # Two-pass algorithm for numerical stability
                window_sum = 0.0
                count = 0
                for i in range(t - window + 1, t + 1):
                    val = values[i, n]
                    if not np.isnan(val):
                        window_sum += val
                        count += 1

                if count > ddof:
                    mean = window_sum / count
                    var_sum = 0.0
                    for i in range(t - window + 1, t + 1):
                        val = values[i, n]
                        if not np.isnan(val):
                            diff = val - mean
                            var_sum += diff * diff

                    result[t, n] = np.sqrt(var_sum / (count - ddof))
                # else remains NaN

        return result


def numba_rolling_mean(
    values: np.ndarray,
    window: int,
    axis: int = 0,
) -> np.ndarray:
    """
    Numba-accelerated rolling mean.

    Parameters
    ----------
    values : np.ndarray
        Input array, shape (T, N)
    window : int
        Window size
    axis : int
        Time axis (must be 0)

    Returns
    -------
    np.ndarray
        Rolling mean

    Notes
    -----
    Uses Numba JIT compilation and parallelization.
    Can be 50-200x faster than naive Python loops for large N.
    """
    if not HAS_NUMBA:
        # Fallback to stride-based version
        return fast_rolling_mean(values, window, axis)

    if window <= 0:
        raise ValueError("window must be positive")

    if axis != 0:
        raise NotImplementedError("Only axis=0 supported")

    if values.ndim == 1:
        values = values.reshape(-1, 1)
        result = _numba_rolling_mean_2d(values, window)
        return result.ravel()
    elif values.ndim == 2:
        return _numba_rolling_mean_2d(values, window)
    else:
        raise NotImplementedError("Only 1D and 2D arrays supported")


def numba_rolling_std(
    values: np.ndarray,
    window: int,
    axis: int = 0,
    ddof: int = 1,
) -> np.ndarray:
    """
    Numba-accelerated rolling std.

    Parameters
    ----------
    values : np.ndarray
        Input array, shape (T, N)
    window : int
        Window size
    axis : int
        Time axis (must be 0)
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    np.ndarray
        Rolling std

    Notes
    -----
    Uses Numba JIT compilation and parallelization.
    """
    if not HAS_NUMBA:
        # Fallback to stride-based version
        return fast_rolling_std(values, window, axis, ddof)

    if window <= 0:
        raise ValueError("window must be positive")

    if axis != 0:
        raise NotImplementedError("Only axis=0 supported")

    if values.ndim == 1:
        values = values.reshape(-1, 1)
        result = _numba_rolling_std_2d(values, window, ddof)
        return result.ravel()
    elif values.ndim == 2:
        return _numba_rolling_std_2d(values, window, ddof)
    else:
        raise NotImplementedError("Only 1D and 2D arrays supported")


# ============================================================================
# Capability reporting
# ============================================================================

def get_capabilities() -> dict:
    """
    Return available fast kernel capabilities.

    Returns
    -------
    dict
        Keys: capability names
        Values: True if available, False otherwise
    """
    return {
        "bottleneck": HAS_BOTTLENECK,
        "numba": HAS_NUMBA,
        "stride_tricks": True,  # Always available with NumPy
    }
