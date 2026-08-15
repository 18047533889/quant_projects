"""
Stateless transforms for modeling_adapters.

These transforms do not require fitting and can be applied independently
to each cross-section or time period.
"""
from typing import Optional
import numpy as np

from modeling_adapters.errors import InsufficientDataError


def rank_transform(
    values: np.ndarray,
    axis: int = 0,
    method: str = "average",
    pct: bool = True,
) -> np.ndarray:
    """
    Rank transform along specified axis.

    Args:
        values: Input array (T, N) where T=time, N=assets
        axis: Axis along which to rank (0=time, 1=cross-section)
        method: Ranking method ('average', 'min', 'max', 'dense', 'ordinal')
        pct: If True, return percentile ranks [0, 1], else ranks [1, N]

    Returns:
        Ranked array with same shape as input

    Note:
        This is a minimal reference implementation. For production,
        use factor_preprocess via adapter which has optimized kernels.
    """
    if values.size == 0:
        raise InsufficientDataError("Empty array provided")

    # Handle 1D case
    if values.ndim == 1:
        values = values.reshape(-1, 1) if axis == 0 else values.reshape(1, -1)
        result = _rank_1d(values.ravel(), method=method, pct=pct)
        return result

    # 2D case: rank along specified axis
    if axis == 1:
        # Rank cross-sectionally (each row independently)
        ranked = np.apply_along_axis(
            lambda x: _rank_1d(x, method=method, pct=pct),
            axis=1,
            arr=values,
        )
    else:
        # Rank time-serially (each column independently)
        ranked = np.apply_along_axis(
            lambda x: _rank_1d(x, method=method, pct=pct),
            axis=0,
            arr=values,
        )

    return ranked


def _rank_1d(values: np.ndarray, method: str, pct: bool) -> np.ndarray:
    """Rank 1D array handling NaN."""
    if np.all(np.isnan(values)):
        return np.full_like(values, np.nan, dtype=np.float64)

    # Get valid mask
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.full_like(values, np.nan, dtype=np.float64)

    result = np.full_like(values, np.nan, dtype=np.float64)
    valid_values = values[valid_mask]

    # Use scipy-style ranking
    from scipy.stats import rankdata
    ranks = rankdata(valid_values, method=method)

    if pct:
        # Convert to percentile [0, 1]
        n_valid = np.sum(valid_mask)
        ranks = (ranks - 1) / (n_valid - 1) if n_valid > 1 else np.full_like(ranks, 0.5)

    result[valid_mask] = ranks
    return result


def zscore_transform(
    values: np.ndarray,
    axis: int = 1,
    ddof: int = 1,
    clip: Optional[float] = None,
) -> np.ndarray:
    """
    Z-score standardization along specified axis.

    Args:
        values: Input array (T, N)
        axis: Axis along which to standardize (0=time, 1=cross-section)
        ddof: Degrees of freedom for std calculation
        clip: Optional clipping threshold (in std units)

    Returns:
        Standardized array with same shape
    """
    if values.size == 0:
        raise InsufficientDataError("Empty array provided")

    # Compute mean and std along axis, keeping dims
    mean = np.nanmean(values, axis=axis, keepdims=True)
    std = np.nanstd(values, axis=axis, ddof=ddof, keepdims=True)

    # Avoid division by zero
    std = np.where(std < 1e-10, np.nan, std)

    # Standardize
    z = (values - mean) / std

    # Optional clipping
    if clip is not None:
        z = np.clip(z, -clip, clip)

    return z


def winsorize(
    values: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
    axis: int = 1,
) -> np.ndarray:
    """
    Winsorize values to specified quantiles.

    Args:
        values: Input array (T, N)
        lower: Lower quantile (e.g., 0.01 for 1st percentile)
        upper: Upper quantile (e.g., 0.99 for 99th percentile)
        axis: Axis along which to compute quantiles

    Returns:
        Winsorized array
    """
    if values.size == 0:
        raise InsufficientDataError("Empty array provided")

    if not (0 <= lower < upper <= 1):
        raise ValueError(f"Invalid quantiles: lower={lower}, upper={upper}")

    # Compute quantiles along axis
    q_lower = np.nanquantile(values, lower, axis=axis, keepdims=True)
    q_upper = np.nanquantile(values, upper, axis=axis, keepdims=True)

    # Clip values
    winsorized = np.clip(values, q_lower, q_upper)

    return winsorized
