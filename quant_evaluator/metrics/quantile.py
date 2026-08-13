"""
Quantile-based metrics and summary statistics.

Optimized implementations using numpy SIMD operations, np.partition for O(n) binning.
"""

from typing import Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def assign_quantiles(
    values: np.ndarray,
    n_quantiles: int = 5,
    method: str = "average",
) -> np.ndarray:
    """
    Assign quantile IDs to values with tie handling.

    Args:
        values: Input values (1D or 2D)
        n_quantiles: Number of quantiles
        method: Tie-breaking method ("average" for average ranks)

    Returns:
        Quantile assignments (0 to n_quantiles-1), or -1 for NaN
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    T, N = values.shape
    quantiles = np.full((T, N), -1, dtype=np.int32)

    for t in range(T):
        v = values[t, :]
        finite_mask = np.isfinite(v)

        n_finite = np.sum(finite_mask)
        if n_finite < n_quantiles:
            continue

        v_finite = v[finite_mask]

        # Fast path: use np.partition to find quantile boundaries in O(n)
        # Only need n_quantiles-1 partition points
        boundary_indices = np.linspace(0, n_finite - 1, n_quantiles + 1, dtype=np.int64)
        boundaries = np.partition(v_finite, boundary_indices[1:-1])[boundary_indices[1:-1]]

        # Use searchsorted for O(n log k) binning instead of O(n log n) sorting
        # Add small epsilon to boundaries to handle ties consistently
        boundaries_sorted = np.sort(boundaries)
        q_bins = np.searchsorted(boundaries_sorted, v_finite, side='right')
        q_bins = np.clip(q_bins, 0, n_quantiles - 1)

        quantiles[t, finite_mask] = q_bins

    return quantiles


def assign_quantiles_fast(
    values: np.ndarray,
    n_quantiles: int = 5,
) -> np.ndarray:
    """
    Fast quantile assignment using percentile-based boundaries.

    Uses np.percentile for boundary computation (single pass) and
    vectorized comparison for binning. Significantly faster for large arrays.

    Args:
        values: Input values shape (T, N) or (T, N, F)
        n_quantiles: Number of quantiles

    Returns:
        Quantile assignments (0 to n_quantiles-1), or -1 for NaN
    """
    if values.ndim == 2:
        T, N = values.shape
        F = 1
        values = values.reshape(T, N, 1)
    else:
        T, N, F = values.shape

    quantiles = np.full((T, N, F), -1, dtype=np.int32)

    # Compute percentile boundaries for each time-factor slice
    percentiles = np.linspace(0, 100, n_quantiles + 1)[1:-1]  # Exclude 0 and 100

    for t in range(T):
        for f in range(F):
            v = values[t, :, f]
            finite_mask = np.isfinite(v)
            n_finite = np.sum(finite_mask)

            if n_finite < n_quantiles:
                continue

            v_finite = v[finite_mask]

            # Compute quantile boundaries using percentile (fast, approximate)
            boundaries = np.percentile(v_finite, percentiles)

            # Vectorized binning: count how many boundaries each value exceeds
            # This is O(n * k) where k is number of quantiles
            q_bins = np.searchsorted(boundaries, v_finite, side='right')
            q_bins = np.clip(q_bins, 0, n_quantiles - 1)

            quantiles[t, finite_mask, f] = q_bins

    if F == 1:
        return quantiles.reshape(T, N)
    return quantiles


def assign_quantiles_batch(
    values: np.ndarray,
    n_quantiles: int = 5,
) -> np.ndarray:
    """
    Ultra-fast batch quantile assignment with minimal Python loops.

    Processes entire time×asset×factor tensor with vectorized operations.
    Best performance for large batches.

    Args:
        values: Input values shape (T, N, F)
        n_quantiles: Number of quantiles

    Returns:
        Quantile assignments shape (T, N, F), -1 for NaN
    """
    if values.ndim == 2:
        values = values[:, :, np.newaxis]

    T, N, F = values.shape
    quantiles = np.full((T, N, F), -1, dtype=np.int32)

    # Process all time-factor pairs efficiently
    for t in range(T):
        # Vectorize across all factors at once
        v_t = values[t, :, :]  # (N, F)
        finite_mask_t = np.isfinite(v_t)  # (N, F)

        for f in range(F):
            mask = finite_mask_t[:, f]
            n_finite = np.sum(mask)

            if n_finite < n_quantiles:
                continue

            v_finite = v_t[mask, f]

            # Use nanpercentile for robustness
            percentiles = np.linspace(0, 100, n_quantiles + 1)[1:-1]
            boundaries = np.percentile(v_finite, percentiles)

            # Vectorized binning with searchsorted
            q_bins = np.searchsorted(boundaries, v_finite, side='right')
            quantiles[t, mask, f] = q_bins

    return quantiles.squeeze() if quantiles.shape[2] == 1 else quantiles



def compute_quantile_returns(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average returns per quantile per period.

    Optimized with vectorized quantile assignment and bincount aggregation.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundle: Forward returns (T, N)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    values = factor_batch.values  # (T, N, F)
    labels = label_bundle.values

    if labels.ndim == 1:
        labels = np.broadcast_to(labels[:, np.newaxis], (factor_batch.num_times, factor_batch.num_assets))

    T, N, F = values.shape
    quantile_returns = np.full((T, n_quantiles, F), np.nan, dtype=np.float64)
    quantile_counts = np.zeros((T, n_quantiles, F), dtype=np.int32)

    # Vectorized quantile assignment for all factors at once
    q_assignments = assign_quantiles_batch(values, n_quantiles=n_quantiles)  # (T, N, F)

    # Vectorized aggregation per time-factor pair
    for t in range(T):
        label_t = labels[t, :]
        valid_labels = np.isfinite(label_t)

        for f in range(F):
            q_t_f = q_assignments[t, :, f]

            # Combined mask: valid quantile assignment AND valid label
            valid_mask = (q_t_f >= 0) & valid_labels

            if not np.any(valid_mask):
                continue

            q_valid = q_t_f[valid_mask]
            label_valid = label_t[valid_mask]

            # Use bincount for fast aggregation - much faster than loop over quantiles
            # bincount sums, so we sum labels and divide by counts
            counts = np.bincount(q_valid, minlength=n_quantiles)
            sums = np.bincount(q_valid, weights=label_valid, minlength=n_quantiles)

            # Apply min_assets filter and compute means
            sufficient_mask = counts >= min_assets
            quantile_counts[t, :, f] = counts
            quantile_returns[t, sufficient_mask, f] = sums[sufficient_mask] / counts[sufficient_mask]

    return quantile_returns, quantile_counts


def compute_quantile_returns_optimized(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Highly optimized quantile return computation.

    Uses fast quantile assignment and np.bincount for aggregation.
    Target: 5-10x faster than original nested loop implementation.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundle: Forward returns (T, N)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    values = factor_batch.values  # (T, N, F)
    labels = label_bundle.values

    if labels.ndim == 1:
        labels = np.broadcast_to(labels[:, np.newaxis], (factor_batch.num_times, factor_batch.num_assets))

    T, N, F = values.shape
    quantile_returns = np.full((T, n_quantiles, F), np.nan, dtype=np.float64)
    quantile_counts = np.zeros((T, n_quantiles, F), dtype=np.int32)

    # Use fast quantile assignment
    q_assignments = assign_quantiles_fast(values, n_quantiles=n_quantiles)  # (T, N, F)

    # Pre-compute valid labels mask once
    valid_labels = np.isfinite(labels)

    # Vectorized aggregation with bincount
    for t in range(T):
        label_t = labels[t, :]
        valid_t = valid_labels[t, :]
        q_t = q_assignments[t, :, :]  # (N, F)

        for f in range(F):
            q_f = q_t[:, f]

            # Combined mask: valid quantile AND valid label
            mask = (q_f >= 0) & valid_t

            if not np.any(mask):
                continue

            q_valid = q_f[mask]
            label_valid = label_t[mask]

            # Fast aggregation with bincount
            counts = np.bincount(q_valid, minlength=n_quantiles)
            sums = np.bincount(q_valid, weights=label_valid, minlength=n_quantiles)

            # Apply threshold and compute means
            sufficient = counts >= min_assets
            quantile_counts[t, :, f] = counts

            if np.any(sufficient):
                quantile_returns[t, sufficient, f] = sums[sufficient] / counts[sufficient]

    return quantile_returns, quantile_counts


def compute_top_bottom_spread(
    quantile_returns: np.ndarray,
    top_q: int = 0,
    bottom_q: int = -1,
) -> np.ndarray:
    """
    Compute top-bottom quantile spread.

    Args:
        quantile_returns: Quantile returns (T, n_quantiles, F)
        top_q: Index of top quantile (default 0 = highest)
        bottom_q: Index of bottom quantile (default -1 = lowest)

    Returns:
        Spread series (T, F)
    """
    top_returns = quantile_returns[:, top_q, :]
    bottom_returns = quantile_returns[:, bottom_q, :]

    spread = top_returns - bottom_returns

    return spread
