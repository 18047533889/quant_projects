"""
Multiple testing corrections for factor analysis.

Provides Bonferroni and Benjamini-Hochberg FDR corrections.
"""

from typing import Tuple
import numpy as np


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction for multiple testing.

    Args:
        p_values: Array of p-values, any shape
        alpha: Family-wise error rate (default 0.05)

    Returns:
        (adjusted_p_values, reject_mask)
        adjusted_p_values: Bonferroni-adjusted p-values (min(p * n_tests, 1.0))
        reject_mask: Boolean mask where null hypothesis is rejected
    """
    # Flatten for processing
    original_shape = p_values.shape
    p_flat = p_values.ravel()

    # Count valid (non-NaN) p-values
    valid_mask = np.isfinite(p_flat)
    n_tests = np.sum(valid_mask)

    if n_tests == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, dtype=bool)

    # Adjust p-values
    adjusted = np.full_like(p_flat, np.nan)
    adjusted[valid_mask] = np.minimum(p_flat[valid_mask] * n_tests, 1.0)

    # Rejection mask
    reject = np.zeros_like(p_flat, dtype=bool)
    reject[valid_mask] = adjusted[valid_mask] <= alpha

    return adjusted.reshape(original_shape), reject.reshape(original_shape)


def benjamini_hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.

    Args:
        p_values: Array of p-values, any shape
        alpha: False discovery rate (default 0.05)

    Returns:
        (adjusted_p_values, reject_mask, n_discoveries)
        adjusted_p_values: BH-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected
        n_discoveries: Number of discoveries (rejections)
    """
    # Flatten for processing
    original_shape = p_values.shape
    p_flat = p_values.ravel()

    # Extract valid p-values with their indices
    valid_mask = np.isfinite(p_flat)
    valid_indices = np.where(valid_mask)[0]
    p_valid = p_flat[valid_mask]

    n_tests = len(p_valid)

    if n_tests == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, dtype=bool), 0

    # Sort p-values and track original indices
    sort_idx = np.argsort(p_valid)
    p_sorted = p_valid[sort_idx]
    original_idx = valid_indices[sort_idx]

    # Compute BH critical values
    ranks = np.arange(1, n_tests + 1)
    bh_critical = (ranks / n_tests) * alpha

    # Find largest i where p[i] <= (i/m) * alpha
    comparisons = p_sorted <= bh_critical
    if np.any(comparisons):
        max_idx = np.where(comparisons)[0][-1]
        n_discoveries = max_idx + 1
    else:
        n_discoveries = 0

    # Compute adjusted p-values (step-down)
    adjusted_sorted = np.minimum.accumulate(
        (p_sorted * n_tests / ranks)[::-1]
    )[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Map back to original positions
    adjusted_flat = np.full_like(p_flat, np.nan)
    adjusted_flat[original_idx[sort_idx]] = adjusted_sorted

    # Rejection mask
    reject_flat = np.zeros_like(p_flat, dtype=bool)
    if n_discoveries > 0:
        reject_indices = original_idx[sort_idx[:n_discoveries]]
        reject_flat[reject_indices] = True

    return (
        adjusted_flat.reshape(original_shape),
        reject_flat.reshape(original_shape),
        n_discoveries,
    )


def holm_bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply Holm-Bonferroni step-down correction.

    More powerful than Bonferroni, controls family-wise error rate.

    Args:
        p_values: Array of p-values, any shape
        alpha: Family-wise error rate (default 0.05)

    Returns:
        (adjusted_p_values, reject_mask, n_discoveries)
        adjusted_p_values: Holm-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected
        n_discoveries: Number of discoveries (rejections)
    """
    # Flatten for processing
    original_shape = p_values.shape
    p_flat = p_values.ravel()

    # Extract valid p-values
    valid_mask = np.isfinite(p_flat)
    valid_indices = np.where(valid_mask)[0]
    p_valid = p_flat[valid_mask]

    n_tests = len(p_valid)

    if n_tests == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, dtype=bool), 0

    # Sort p-values
    sort_idx = np.argsort(p_valid)
    p_sorted = p_valid[sort_idx]
    original_idx = valid_indices[sort_idx]

    # Compute Holm critical values (step-down)
    ranks = np.arange(1, n_tests + 1)
    holm_critical = alpha / (n_tests - ranks + 1)

    # Find rejections (sequential)
    n_discoveries = 0
    for i in range(n_tests):
        if p_sorted[i] <= holm_critical[i]:
            n_discoveries = i + 1
        else:
            break

    # Compute adjusted p-values (step-down maximum)
    adjusted_sorted = p_sorted * (n_tests - ranks + 1)
    adjusted_sorted = np.minimum.accumulate(np.maximum.accumulate(adjusted_sorted))
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Map back to original positions
    adjusted_flat = np.full_like(p_flat, np.nan)
    adjusted_flat[original_idx[sort_idx]] = adjusted_sorted

    # Rejection mask
    reject_flat = np.zeros_like(p_flat, dtype=bool)
    if n_discoveries > 0:
        reject_indices = original_idx[sort_idx[:n_discoveries]]
        reject_flat[reject_indices] = True

    return (
        adjusted_flat.reshape(original_shape),
        reject_flat.reshape(original_shape),
        n_discoveries,
    )


def sidak_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Šidák correction for multiple testing.

    Assumes independence, slightly less conservative than Bonferroni.

    Args:
        p_values: Array of p-values, any shape
        alpha: Family-wise error rate (default 0.05)

    Returns:
        (adjusted_p_values, reject_mask)
        adjusted_p_values: Šidák-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected
    """
    # Flatten for processing
    original_shape = p_values.shape
    p_flat = p_values.ravel()

    # Count valid p-values
    valid_mask = np.isfinite(p_flat)
    n_tests = np.sum(valid_mask)

    if n_tests == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, dtype=bool)

    # Adjusted alpha for Šidák
    alpha_sidak = 1.0 - (1.0 - alpha) ** (1.0 / n_tests)

    # Adjust p-values: 1 - (1 - p)^m
    adjusted = np.full_like(p_flat, np.nan)
    adjusted[valid_mask] = 1.0 - (1.0 - p_flat[valid_mask]) ** n_tests
    adjusted[valid_mask] = np.minimum(adjusted[valid_mask], 1.0)

    # Rejection mask
    reject = np.zeros_like(p_flat, dtype=bool)
    reject[valid_mask] = p_flat[valid_mask] <= alpha_sidak

    return adjusted.reshape(original_shape), reject.reshape(original_shape)


def compute_fdr(
    p_values: np.ndarray,
    reject_mask: np.ndarray,
) -> float:
    """
    Compute empirical false discovery rate.

    Args:
        p_values: Array of p-values
        reject_mask: Boolean mask of rejections

    Returns:
        FDR estimate (expected proportion of false discoveries)
    """
    n_rejections = np.sum(reject_mask)

    if n_rejections == 0:
        return 0.0

    # Count valid tests
    valid_mask = np.isfinite(p_values.ravel())
    n_tests = np.sum(valid_mask)

    if n_tests == 0:
        return np.nan

    # FDR = E[V/R] where V = false discoveries, R = rejections
    # Under null, expected false discoveries = alpha * n_tests
    # Conservative estimate: use mean p-value of rejected as alpha
    rejected_p = p_values.ravel()[reject_mask.ravel()]
    alpha_est = np.mean(rejected_p)

    fdr = (alpha_est * n_tests) / n_rejections

    return min(fdr, 1.0)
