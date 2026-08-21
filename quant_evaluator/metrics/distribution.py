"""
Distribution metrics for factor analysis.

Provides skewness, kurtosis, and outlier detection metrics.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats


def compute_skewness(
    values: np.ndarray,
    axis: int = 0,
    min_obs: int = 10,
) -> np.ndarray:
    """
    Compute skewness along specified axis.

    Args:
        values: Input array, typically (T, N, F) or (T, F)
        axis: Axis along which to compute (0=time, 1=cross-section)
        min_obs: Minimum observations required

    Returns:
        Skewness array with axis reduced. NaN for insufficient data.
    """
    # Count finite observations
    finite_mask = np.isfinite(values)
    n_obs = np.sum(finite_mask, axis=axis)

    # Compute skewness using scipy (handles NaN)
    with np.errstate(invalid='ignore'):
        skew = stats.skew(values, axis=axis, nan_policy='omit', bias=False)

    # Mask insufficient observations
    skew = np.where(n_obs >= min_obs, skew, np.nan)

    return skew


def compute_kurtosis(
    values: np.ndarray,
    axis: int = 0,
    min_obs: int = 10,
    excess: bool = True,
) -> np.ndarray:
    """
    Compute kurtosis along specified axis.

    Args:
        values: Input array, typically (T, N, F) or (T, F)
        axis: Axis along which to compute
        min_obs: Minimum observations required
        excess: If True, return excess kurtosis (subtract 3 from normal kurtosis)

    Returns:
        Kurtosis array with axis reduced. NaN for insufficient data.
    """
    # Count finite observations
    finite_mask = np.isfinite(values)
    n_obs = np.sum(finite_mask, axis=axis)

    # Compute kurtosis using scipy
    with np.errstate(invalid='ignore'):
        kurt = stats.kurtosis(values, axis=axis, nan_policy='omit', bias=False, fisher=excess)

    # Mask insufficient observations
    kurt = np.where(n_obs >= min_obs, kurt, np.nan)

    return kurt


def detect_outliers_iqr(
    values: np.ndarray,
    axis: int = 0,
    multiplier: float = 1.5,
    min_obs: int = 10,
) -> np.ndarray:
    """
    Detect outliers using IQR method.

    Args:
        values: Input array (T, N, F) or (T, F)
        axis: Axis along which to compute IQR (0=time, 1=cross-section)
        multiplier: IQR multiplier for outlier bounds (default 1.5)
        min_obs: Minimum observations to compute quartiles

    Returns:
        Boolean mask same shape as values, True where outlier detected
    """
    # Count finite observations
    finite_mask = np.isfinite(values)
    n_obs = np.sum(finite_mask, axis=axis, keepdims=True)

    # Compute quartiles
    with np.errstate(invalid='ignore'):
        q1 = np.nanpercentile(values, 25, axis=axis, keepdims=True)
        q3 = np.nanpercentile(values, 75, axis=axis, keepdims=True)

    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Mark outliers
    outliers = (values < lower_bound) | (values > upper_bound)

    # Mask insufficient observations: if not enough data, mark nothing as outlier
    insufficient = n_obs < min_obs
    outliers = outliers & ~insufficient

    # Non-finite values are not outliers (they're already invalid)
    outliers = outliers & finite_mask

    return outliers


def detect_outliers_zscore(
    values: np.ndarray,
    axis: int = 0,
    threshold: float = 3.0,
    min_obs: int = 10,
) -> np.ndarray:
    """
    Detect outliers using z-score method.

    Args:
        values: Input array (T, N, F) or (T, F)
        axis: Axis along which to compute mean/std
        threshold: Z-score threshold (default 3.0)
        min_obs: Minimum observations to compute statistics

    Returns:
        Boolean mask same shape as values, True where outlier detected
    """
    # Count finite observations
    finite_mask = np.isfinite(values)
    n_obs = np.sum(finite_mask, axis=axis, keepdims=True)

    # Compute mean and std
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(values, axis=axis, keepdims=True)
        std = np.nanstd(values, axis=axis, keepdims=True, ddof=1)

    # Compute absolute z-scores
    z_scores = np.abs((values - mean) / std)

    # Mark outliers
    outliers = z_scores > threshold

    # Mask insufficient observations or zero std
    insufficient = (n_obs < min_obs) | (std == 0) | ~np.isfinite(std)
    outliers = outliers & ~insufficient

    # Non-finite values are not outliers
    outliers = outliers & finite_mask

    return outliers


def compute_outlier_ratio(
    outlier_mask: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """
    Compute ratio of outliers along axis.

    Args:
        outlier_mask: Boolean mask from detect_outliers_*
        axis: Axis along which to compute ratio

    Returns:
        Ratio of outliers, shape with axis reduced
    """
    total = outlier_mask.shape[axis]
    count = np.sum(outlier_mask, axis=axis)
    ratio = count / total

    return ratio


def compute_higher_moments(
    values: np.ndarray,
    axis: int = 0,
    min_obs: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, std, skewness, and kurtosis together.

    Args:
        values: Input array (T, N, F) or (T, F)
        axis: Axis along which to compute
        min_obs: Minimum observations required

    Returns:
        (mean, std, skewness, excess_kurtosis) with axis reduced
    """
    finite_mask = np.isfinite(values)
    n_obs = np.sum(finite_mask, axis=axis)

    with np.errstate(invalid='ignore'):
        mean = np.nanmean(values, axis=axis)
        std = np.nanstd(values, axis=axis, ddof=1)
        skew = stats.skew(values, axis=axis, nan_policy='omit', bias=False)
        kurt = stats.kurtosis(values, axis=axis, nan_policy='omit', bias=False, fisher=True)

    # Mask insufficient observations
    insufficient = n_obs < min_obs
    mean = np.where(insufficient, np.nan, mean)
    std = np.where(insufficient, np.nan, std)
    skew = np.where(insufficient, np.nan, skew)
    kurt = np.where(insufficient, np.nan, kurt)

    return mean, std, skew, kurt
