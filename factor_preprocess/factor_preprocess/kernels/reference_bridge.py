"""
Reference implementation bridge for parity checking.

This module provides reference implementations that fast kernels must match.
All reference implementations are pure NumPy/SciPy with explicit NaN handling.
"""
import numpy as np
from typing import Literal
from scipy.stats import rankdata


def reference_cs_rank(
    values: np.ndarray,
    axis: int = -1,
    method: Literal["average", "min", "max", "dense", "ordinal"] = "average",
    pct: bool = False,
) -> np.ndarray:
    """
    Reference cross-sectional rank implementation.

    Parameters
    ----------
    values : np.ndarray
        Input array, typically shape (dates, assets) or (assets,)
    axis : int
        Axis to rank along (default -1 for last axis)
    method : str
        Tie-breaking method
    pct : bool
        If True, return percentile ranks [0, 1]

    Returns
    -------
    np.ndarray
        Ranked values. NaN inputs produce NaN outputs.
    """
    if values.size == 0:
        return values.copy()

    def rank_1d(arr):
        """Rank a single vector."""
        if np.all(np.isnan(arr)):
            return arr.copy()

        mask = np.isfinite(arr)
        if not np.any(mask):
            return np.full_like(arr, np.nan)

        result = np.full_like(arr, np.nan, dtype=np.float64)
        ranks = rankdata(arr[mask], method=method)
        result[mask] = ranks

        if pct:
            n_finite = np.sum(mask)
            if n_finite > 1:
                result[mask] = (result[mask] - 1.0) / (n_finite - 1.0)
            else:
                result[mask] = 0.5

        return result

    return np.apply_along_axis(rank_1d, axis, values)


def reference_cs_zscore(
    values: np.ndarray,
    axis: int = -1,
    ddof: int = 1,
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Reference cross-sectional z-score implementation.

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
        Non-finite inputs (±inf) produce NaN outputs — mirrors
        transforms.cross_sectional.cs_zscore (FP-P0-12), so an inf
        can never masquerade as a perfectly average (0.0) factor value.
    """
    if values.size == 0:
        return values.copy()

    # nanmean/nanstd do not ignore ±inf, so an inf would push std to NaN
    # and silently route the slice into the constant branch.  Mask
    # non-finite inputs to NaN first — parity with fast_cs_zscore.
    values = np.where(np.isfinite(values), values, np.nan)

    mean = np.nanmean(values, axis=axis, keepdims=True)
    std = np.nanstd(values, axis=axis, keepdims=True, ddof=ddof)

    result = np.where(
        std > 0,
        (values - mean) / std,
        constant_value,
    )

    result = np.where(np.isnan(values), np.nan, result)

    return result


def reference_cs_demean(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Reference cross-sectional demean implementation.

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
        Non-finite inputs (±inf) produce NaN outputs — mirrors
        transforms.cross_sectional.cs_demean, so one inf cannot turn
        every finite value into ∓inf via the slice mean.
    """
    if values.size == 0:
        return values.copy()

    # nanmean does not ignore ±inf: one inf makes the slice mean inf and
    # every finite entry becomes -inf.  Mask non-finite inputs to NaN
    # first (parity with transforms.cs_demean, FP-P2-6).
    values = np.where(np.isfinite(values), values, np.nan)

    mean = np.nanmean(values, axis=axis, keepdims=True)
    return values - mean


def reference_rolling_mean(
    values: np.ndarray,
    window: int,
    axis: int = 0,
) -> np.ndarray:
    """
    Reference rolling mean (simple stride-based implementation).

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
    This is a naive implementation for reference.
    Fast kernel should use stride tricks or numba.
    """
    if axis != 0:
        raise NotImplementedError("Reference only supports axis=0")

    if values.size == 0:
        return values.copy()

    T = values.shape[0]
    result = np.full_like(values, np.nan, dtype=np.float64)

    for t in range(window - 1, T):
        window_data = values[t - window + 1 : t + 1]
        result[t] = np.nanmean(window_data, axis=0)

    return result


def reference_rolling_std(
    values: np.ndarray,
    window: int,
    axis: int = 0,
    ddof: int = 1,
) -> np.ndarray:
    """
    Reference rolling standard deviation.

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
    """
    if axis != 0:
        raise NotImplementedError("Reference only supports axis=0")

    if values.size == 0:
        return values.copy()

    T = values.shape[0]
    result = np.full_like(values, np.nan, dtype=np.float64)

    for t in range(window - 1, T):
        window_data = values[t - window + 1 : t + 1]
        result[t] = np.nanstd(window_data, axis=0, ddof=ddof)

    return result
