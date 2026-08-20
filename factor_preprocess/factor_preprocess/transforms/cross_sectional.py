"""
Cross-sectional transforms: rank, zscore, demean, winsor.

These transforms operate on each time slice independently.
All transforms handle NaN explicitly and use average-tie ranking.
"""
import numpy as np
from typing import Optional, Literal


def cs_rank(
    values: np.ndarray,
    axis: int = -1,
    method: Literal["average", "min", "max", "dense", "ordinal"] = "average",
    pct: bool = False,
) -> np.ndarray:
    """
    Cross-sectional rank with explicit tie handling.

    Parameters
    ----------
    values : np.ndarray
        Input array, typically shape (dates, assets) or (assets,)
    axis : int
        Axis to rank along (default -1 for last axis)
    method : str
        Tie-breaking method. "average" is production default.
    pct : bool
        If True, return percentile ranks [0, 1]

    Returns
    -------
    np.ndarray
        Ranked values. NaN inputs produce NaN outputs.

    Notes
    -----
    Uses average-tie ranking for stable, permutation-invariant results.
    Constants (all same value) produce mid-rank.
    """
    if values.size == 0:
        return values.copy()

    # scipy.stats.rankdata doesn't handle multi-dimensional arrays natively
    # Work along specified axis
    from scipy.stats import rankdata

    def rank_1d(arr):
        """Rank a single vector."""
        # Handle all-NaN
        if np.all(np.isnan(arr)):
            return arr.copy()

        # rankdata with nan_policy='omit' returns ranks for finite values
        # We need to preserve NaN positions
        mask = np.isfinite(arr)
        if not np.any(mask):
            return np.full_like(arr, np.nan)

        result = np.full_like(arr, np.nan, dtype=np.float64)
        ranks = rankdata(arr[mask], method=method)
        result[mask] = ranks

        if pct:
            # Convert to percentile: (rank - 1) / (n - 1)
            n_finite = np.sum(mask)
            if n_finite > 1:
                result[mask] = (result[mask] - 1.0) / (n_finite - 1.0)
            else:
                # Single value -> 0.5
                result[mask] = 0.5

        return result

    # Apply along axis
    return np.apply_along_axis(rank_1d, axis, values)


def cs_zscore(
    values: np.ndarray,
    axis: int = -1,
    ddof: int = 1,
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Cross-sectional z-score normalization.

    Parameters
    ----------
    values : np.ndarray
        Input array
    axis : int
        Axis to normalize along
    ddof : int
        Delta degrees of freedom for std calculation
    constant_value : float
        Value to return for constant slices (zero std)

    Returns
    -------
    np.ndarray
        Z-scored values. NaN inputs produce NaN outputs.

    Notes
    -----
    (x - mean) / std with explicit handling of zero std.
    """
    if values.size == 0:
        return values.copy()

    # Use nanmean and nanstd to handle NaN
    mean = np.nanmean(values, axis=axis, keepdims=True)
    std = np.nanstd(values, axis=axis, keepdims=True, ddof=ddof)

    # Avoid division by zero
    result = np.where(
        std > 0,
        (values - mean) / std,
        constant_value,
    )

    # Preserve NaN
    result = np.where(np.isnan(values), np.nan, result)

    return result


def cs_demean(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Cross-sectional demean.

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
    """
    if values.size == 0:
        return values.copy()

    mean = np.nanmean(values, axis=axis, keepdims=True)
    return values - mean


def cs_winsor(
    values: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
    axis: int = -1,
) -> np.ndarray:
    """
    Cross-sectional winsorization by quantiles.

    Parameters
    ----------
    values : np.ndarray
        Input array
    lower : float
        Lower quantile [0, 1]
    upper : float
        Upper quantile [0, 1]
    axis : int
        Axis to winsorize along

    Returns
    -------
    np.ndarray
        Winsorized values. NaN inputs produce NaN outputs.

    Notes
    -----
    Clips extreme values to quantile boundaries.
    Uses linear interpolation for quantile estimation.
    """
    if values.size == 0:
        return values.copy()

    if not (0 <= lower < upper <= 1):
        raise ValueError(f"Invalid quantiles: lower={lower}, upper={upper}")

    # Compute quantiles along axis, ignoring NaN
    lower_bound = np.nanquantile(values, lower, axis=axis, keepdims=True)
    upper_bound = np.nanquantile(values, upper, axis=axis, keepdims=True)

    # Clip
    result = np.clip(values, lower_bound, upper_bound)

    # Preserve NaN
    result = np.where(np.isnan(values), np.nan, result)

    return result


def cs_scale(
    values: np.ndarray,
    axis: int = -1,
    target_std: float = 1.0,
    ddof: int = 1,
) -> np.ndarray:
    """
    Cross-sectional scaling to target standard deviation.

    Parameters
    ----------
    values : np.ndarray
        Input array
    axis : int
        Axis to scale along
    target_std : float
        Target standard deviation
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    np.ndarray
        Scaled values. NaN inputs produce NaN outputs.
    """
    if values.size == 0:
        return values.copy()

    std = np.nanstd(values, axis=axis, keepdims=True, ddof=ddof)

    # Avoid division by zero
    result = np.where(
        std > 0,
        values * (target_std / std),
        values,  # Keep original if constant
    )

    return result
