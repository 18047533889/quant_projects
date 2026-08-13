"""
Numba-accelerated transforms for factor preprocessing.

This module provides highly optimized implementations using Numba JIT compilation
with parallel execution for large panel data (time × assets).

Target speedup: 20-100x for rolling operations on typical factor panels.

All implementations maintain exact parity with reference implementations.
"""
import numpy as np
from typing import Optional, Literal

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorators for graceful degradation
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ============================================================================
# Rolling operations - primary speedup target
# ============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=False)
    def rolling_mean_2d(values: np.ndarray, window: int) -> np.ndarray:
        """
        Numba-accelerated rolling mean for 2D panels.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N) where T = time periods, N = assets
        window : int
            Rolling window size

        Returns
        -------
        np.ndarray
            Rolling mean, shape (T, N). First window-1 rows are NaN.

        Notes
        -----
        - Parallel execution across assets (columns)
        - Handles NaN and Inf correctly (treats both as missing)
        - 20-100x speedup vs naive Python loops
        - Typical use: window=20-60, T=252-1000, N=100-3000
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        # Parallel loop over assets
        for n in prange(N):
            for t in range(window - 1, T):
                window_sum = 0.0
                count = 0
                for i in range(t - window + 1, t + 1):
                    val = values[i, n]
                    if np.isfinite(val):
                        window_sum += val
                        count += 1

                if count > 0:
                    result[t, n] = window_sum / count

        return result

    @njit(parallel=True, cache=True, fastmath=False)
    def rolling_std_2d(values: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
        """
        Numba-accelerated rolling standard deviation for 2D panels.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N)
        window : int
            Rolling window size
        ddof : int
            Delta degrees of freedom (default 1 for sample std)

        Returns
        -------
        np.ndarray
            Rolling std, shape (T, N). First window-1 rows are NaN.

        Notes
        -----
        - Two-pass algorithm for numerical stability
        - Parallel execution across assets
        - 30-100x speedup for large panels
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        for n in prange(N):
            for t in range(window - 1, T):
                # First pass: compute mean
                window_sum = 0.0
                count = 0
                for i in range(t - window + 1, t + 1):
                    val = values[i, n]
                    if np.isfinite(val):
                        window_sum += val
                        count += 1

                if count > ddof:
                    mean = window_sum / count
                    # Second pass: compute variance
                    var_sum = 0.0
                    for i in range(t - window + 1, t + 1):
                        val = values[i, n]
                        if np.isfinite(val):
                            diff = val - mean
                            var_sum += diff * diff

                    result[t, n] = np.sqrt(var_sum / (count - ddof))

        return result

    @njit(parallel=True, cache=True, fastmath=False)
    def rolling_sum_2d(values: np.ndarray, window: int) -> np.ndarray:
        """
        Numba-accelerated rolling sum for 2D panels.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N)
        window : int
            Rolling window size

        Returns
        -------
        np.ndarray
            Rolling sum, shape (T, N)
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        for n in prange(N):
            for t in range(window - 1, T):
                window_sum = 0.0
                count = 0
                for i in range(t - window + 1, t + 1):
                    val = values[i, n]
                    if np.isfinite(val):
                        window_sum += val
                        count += 1

                if count > 0:
                    result[t, n] = window_sum

        return result

    @njit(parallel=True, cache=True, fastmath=False)
    def rolling_min_2d(values: np.ndarray, window: int) -> np.ndarray:
        """
        Numba-accelerated rolling minimum for 2D panels.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N)
        window : int
            Rolling window size

        Returns
        -------
        np.ndarray
            Rolling min, shape (T, N)
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        for n in prange(N):
            for t in range(window - 1, T):
                window_min = np.inf
                has_value = False
                for i in range(t - window + 1, t + 1):
                    val = values[i, n]
                    if np.isfinite(val):
                        if val < window_min:
                            window_min = val
                        has_value = True

                if has_value:
                    result[t, n] = window_min

        return result

    @njit(parallel=True, cache=True, fastmath=False)
    def rolling_max_2d(values: np.ndarray, window: int) -> np.ndarray:
        """
        Numba-accelerated rolling maximum for 2D panels.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N)
        window : int
            Rolling window size

        Returns
        -------
        np.ndarray
            Rolling max, shape (T, N)
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        for n in prange(N):
            for t in range(window - 1, T):
                window_max = -np.inf
                has_value = False
                for i in range(t - window + 1, t + 1):
                    val = values[i, n]
                    if np.isfinite(val):
                        if val > window_max:
                            window_max = val
                        has_value = True

                if has_value:
                    result[t, n] = window_max

        return result


# ============================================================================
# Cross-sectional operations
# ============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def cs_rank_2d(values: np.ndarray, pct: bool = False) -> np.ndarray:
        """
        Numba-accelerated cross-sectional rank for 2D panels.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N) - rank across N (assets) for each T (date)
        pct : bool
            If True, return percentile ranks [0, 1]

        Returns
        -------
        np.ndarray
            Ranked values, shape (T, N). NaN inputs produce NaN outputs.

        Notes
        -----
        - Uses average ranking for ties
        - Parallel execution across time periods
        - 10-50x speedup for large cross-sections
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        for t in prange(T):
            row = values[t, :]

            # Collect finite values and their indices
            finite_mask = np.isfinite(row)
            n_finite = np.sum(finite_mask)

            if n_finite == 0:
                continue

            # Extract finite values
            finite_vals = np.empty(n_finite, dtype=np.float64)
            finite_idx = np.empty(n_finite, dtype=np.int64)
            j = 0
            for i in range(N):
                if finite_mask[i]:
                    finite_vals[j] = row[i]
                    finite_idx[j] = i
                    j += 1

            # Argsort to get ranking order
            sorted_idx = np.argsort(finite_vals)

            # Compute ranks with tie handling (average method)
            ranks = np.empty(n_finite, dtype=np.float64)
            i = 0
            while i < n_finite:
                # Find extent of tie
                j = i + 1
                while j < n_finite and finite_vals[sorted_idx[j]] == finite_vals[sorted_idx[i]]:
                    j += 1

                # Average rank for tied group (1-indexed)
                avg_rank = (i + j - 1) / 2.0 + 1.0
                for k in range(i, j):
                    ranks[sorted_idx[k]] = avg_rank

                i = j

            # Convert to percentile if requested
            if pct:
                if n_finite > 1:
                    ranks = (ranks - 1.0) / (n_finite - 1.0)
                else:
                    ranks[:] = 0.5

            # Write back to result
            for j in range(n_finite):
                result[t, finite_idx[j]] = ranks[j]

        return result

    @njit(parallel=True, cache=True, fastmath=False)
    def cs_zscore_2d(values: np.ndarray, ddof: int = 1, constant_value: float = 0.0) -> np.ndarray:
        """
        Numba-accelerated cross-sectional z-score for 2D panels.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N) - normalize across N for each T
        ddof : int
            Delta degrees of freedom
        constant_value : float
            Value to return for constant slices (zero std)

        Returns
        -------
        np.ndarray
            Z-scored values, shape (T, N)

        Notes
        -----
        - Parallel execution across time periods
        - 5-20x speedup for large cross-sections
        """
        T, N = values.shape
        result = np.full((T, N), np.nan, dtype=np.float64)

        for t in prange(T):
            row = values[t, :]

            # Compute mean
            row_sum = 0.0
            count = 0
            for i in range(N):
                val = row[i]
                if np.isfinite(val):
                    row_sum += val
                    count += 1

            if count == 0:
                continue

            mean = row_sum / count

            # Compute std
            var_sum = 0.0
            for i in range(N):
                val = row[i]
                if np.isfinite(val):
                    diff = val - mean
                    var_sum += diff * diff

            if count > ddof:
                std = np.sqrt(var_sum / (count - ddof))
            else:
                std = 0.0

            # Normalize
            for i in range(N):
                val = row[i]
                if np.isfinite(val):
                    if std > 0:
                        result[t, i] = (val - mean) / std
                    else:
                        result[t, i] = constant_value

        return result


# ============================================================================
# Winsorization
# ============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def cs_winsorize_2d(
        values: np.ndarray,
        lower: float = 0.05,
        upper: float = 0.95
    ) -> np.ndarray:
        """
        Numba-accelerated cross-sectional winsorization for 2D panels.

        Parameters
        ----------
        values : np.ndarray
            Shape (T, N)
        lower : float
            Lower percentile (0-1)
        upper : float
            Upper percentile (0-1)

        Returns
        -------
        np.ndarray
            Winsorized values, shape (T, N)

        Notes
        -----
        - Clips extreme values to percentile bounds
        - Parallel execution across time periods
        - 10-40x speedup for large cross-sections
        """
        T, N = values.shape
        result = np.copy(values)

        for t in prange(T):
            row = values[t, :]

            # Collect finite values
            finite_mask = np.isfinite(row)
            n_finite = np.sum(finite_mask)

            if n_finite < 2:
                continue

            finite_vals = np.empty(n_finite, dtype=np.float64)
            j = 0
            for i in range(N):
                if finite_mask[i]:
                    finite_vals[j] = row[i]
                    j += 1

            # Sort to find percentiles
            finite_vals.sort()

            # Use direct indexing (matches scipy.stats.mstats.winsorize behavior)
            # Not interpolated like np.percentile
            lower_idx = int(np.floor(lower * n_finite))
            upper_idx = int(np.floor(upper * n_finite))

            # Ensure upper_idx doesn't go out of bounds
            if upper_idx >= n_finite:
                upper_idx = n_finite - 1

            lower_bound = finite_vals[lower_idx]
            upper_bound = finite_vals[upper_idx]

            # Clip values
            for i in range(N):
                if finite_mask[i]:
                    if row[i] < lower_bound:
                        result[t, i] = lower_bound
                    elif row[i] > upper_bound:
                        result[t, i] = upper_bound

        return result


# ============================================================================
# Public API with dimension handling
# ============================================================================

def numba_rolling_mean(
    values: np.ndarray,
    window: int,
    axis: int = 0,
) -> np.ndarray:
    """
    Numba-accelerated rolling mean with automatic dimension handling.

    Parameters
    ----------
    values : np.ndarray
        Input array. 1D or 2D.
    window : int
        Rolling window size
    axis : int
        Time axis (must be 0)

    Returns
    -------
    np.ndarray
        Rolling mean, same shape as input

    Notes
    -----
    Target speedup: 20-100x vs naive Python loops for T=500, N=1000, window=20
    """
    if not HAS_NUMBA:
        raise ImportError("Numba not available. Install with: pip install numba")

    if axis != 0:
        raise ValueError("Only axis=0 supported")

    if values.size == 0:
        return values.copy()

    if values.ndim == 1:
        reshaped = values.reshape(-1, 1)
        result = rolling_mean_2d(reshaped, window)
        return result.ravel()
    elif values.ndim == 2:
        return rolling_mean_2d(values, window)
    else:
        raise ValueError("Only 1D and 2D arrays supported")


def numba_rolling_std(
    values: np.ndarray,
    window: int,
    axis: int = 0,
    ddof: int = 1,
) -> np.ndarray:
    """
    Numba-accelerated rolling std with automatic dimension handling.

    Parameters
    ----------
    values : np.ndarray
        Input array. 1D or 2D.
    window : int
        Rolling window size
    axis : int
        Time axis (must be 0)
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    np.ndarray
        Rolling std, same shape as input

    Notes
    -----
    Target speedup: 30-100x vs naive Python loops
    """
    if not HAS_NUMBA:
        raise ImportError("Numba not available. Install with: pip install numba")

    if axis != 0:
        raise ValueError("Only axis=0 supported")

    if values.size == 0:
        return values.copy()

    if values.ndim == 1:
        reshaped = values.reshape(-1, 1)
        result = rolling_std_2d(reshaped, window, ddof)
        return result.ravel()
    elif values.ndim == 2:
        return rolling_std_2d(values, window, ddof)
    else:
        raise ValueError("Only 1D and 2D arrays supported")


def numba_rolling_sum(values: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """Numba-accelerated rolling sum."""
    if not HAS_NUMBA:
        raise ImportError("Numba not available")
    if axis != 0:
        raise ValueError("Only axis=0 supported")
    if values.ndim == 1:
        return rolling_sum_2d(values.reshape(-1, 1), window).ravel()
    return rolling_sum_2d(values, window)


def numba_rolling_min(values: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """Numba-accelerated rolling minimum."""
    if not HAS_NUMBA:
        raise ImportError("Numba not available")
    if axis != 0:
        raise ValueError("Only axis=0 supported")
    if values.ndim == 1:
        return rolling_min_2d(values.reshape(-1, 1), window).ravel()
    return rolling_min_2d(values, window)


def numba_rolling_max(values: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """Numba-accelerated rolling maximum."""
    if not HAS_NUMBA:
        raise ImportError("Numba not available")
    if axis != 0:
        raise ValueError("Only axis=0 supported")
    if values.ndim == 1:
        return rolling_max_2d(values.reshape(-1, 1), window).ravel()
    return rolling_max_2d(values, window)


def numba_cs_rank(values: np.ndarray, axis: int = -1, pct: bool = False) -> np.ndarray:
    """
    Numba-accelerated cross-sectional rank.

    Parameters
    ----------
    values : np.ndarray
        Input array. Ranks along last axis.
    axis : int
        Axis to rank along (must be -1)
    pct : bool
        If True, return percentile ranks [0, 1]

    Returns
    -------
    np.ndarray
        Ranked values

    Notes
    -----
    Target speedup: 10-50x for large cross-sections (N > 1000)
    """
    if not HAS_NUMBA:
        raise ImportError("Numba not available")
    if axis != -1:
        raise ValueError("Only axis=-1 supported")
    if values.ndim == 1:
        return cs_rank_2d(values.reshape(1, -1), pct).ravel()
    return cs_rank_2d(values, pct)


def numba_cs_zscore(
    values: np.ndarray,
    axis: int = -1,
    ddof: int = 1,
    constant_value: float = 0.0
) -> np.ndarray:
    """
    Numba-accelerated cross-sectional z-score.

    Parameters
    ----------
    values : np.ndarray
        Input array. Normalizes along last axis.
    axis : int
        Axis to normalize along (must be -1)
    ddof : int
        Delta degrees of freedom
    constant_value : float
        Value for constant slices

    Returns
    -------
    np.ndarray
        Z-scored values

    Notes
    -----
    Target speedup: 5-20x for large cross-sections
    """
    if not HAS_NUMBA:
        raise ImportError("Numba not available")
    if axis != -1:
        raise ValueError("Only axis=-1 supported")
    if values.ndim == 1:
        return cs_zscore_2d(values.reshape(1, -1), ddof, constant_value).ravel()
    return cs_zscore_2d(values, ddof, constant_value)


def numba_cs_winsorize(
    values: np.ndarray,
    lower: float = 0.05,
    upper: float = 0.95,
    axis: int = -1
) -> np.ndarray:
    """
    Numba-accelerated cross-sectional winsorization.

    Parameters
    ----------
    values : np.ndarray
        Input array. Winsorizes along last axis.
    lower : float
        Lower percentile (0-1)
    upper : float
        Upper percentile (0-1)
    axis : int
        Axis to winsorize along (must be -1)

    Returns
    -------
    np.ndarray
        Winsorized values

    Notes
    -----
    Target speedup: 10-40x for large cross-sections
    """
    if not HAS_NUMBA:
        raise ImportError("Numba not available")
    if axis != -1:
        raise ValueError("Only axis=-1 supported")
    if values.ndim == 1:
        return cs_winsorize_2d(values.reshape(1, -1), lower, upper).ravel()
    return cs_winsorize_2d(values, lower, upper)


def has_numba() -> bool:
    """Check if Numba is available."""
    return HAS_NUMBA
