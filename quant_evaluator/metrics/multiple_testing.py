"""
Multiple testing corrections for factor analysis.

Provides Bonferroni and Benjamini-Hochberg FDR corrections.

QE-METRIC P0-14 — input validation:

All correction functions fail closed (ValueError) on inputs that would
otherwise produce silently wrong results:

- empty ``p_values`` arrays;
- finite p-values outside [0, 1] (e.g. 1.5 or -0.01 — these are not
  p-values and would corrupt every downstream adjusted-p formula);
- ``alpha`` that is non-finite or not strictly inside (0, 1).

Non-finite p-values (NaN, +inf, -inf) are NOT rejected: they are treated
as *missing* tests, excluded from the correction, and their output slots
stay NaN with a False rejection bit. This documented missing-value
convention is relied upon by the statsmodels-parity tests (invalid
entries must be skipped while valid positions are preserved).
"""

from typing import Tuple
import numpy as np


def _validate_p_values(p_values: np.ndarray) -> np.ndarray:
    """QE-METRIC P0-14: fail-closed validation for p-value inputs.

    Rejects empty arrays and finite p-values outside [0, 1]. Non-finite
    entries (NaN, +-inf) are allowed and treated as missing downstream.

    Returns the flattened array (a read view) for processing.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        raise ValueError(
            "p_values must be non-empty; got an empty array "
            f"(shape {p.shape})"
        )
    finite = np.isfinite(p)
    out_of_range = finite & ((p < 0.0) | (p > 1.0))
    n_bad = int(np.sum(out_of_range))
    if n_bad > 0:
        first_idx = int(np.nonzero(out_of_range.ravel())[0][0])
        raise ValueError(
            f"p_values contains {n_bad} finite value(s) outside [0, 1] "
            f"(first at flat index {first_idx}: "
            f"{p.ravel()[first_idx]!r}); these are not valid p-values"
        )
    return p


def _validate_alpha(alpha: float) -> float:
    """QE-METRIC P0-14: fail-closed validation for the alpha level."""
    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError):
        raise ValueError(f"alpha must be a finite float in (0, 1), got {alpha!r}")
    if not np.isfinite(alpha_val) or not (0.0 < alpha_val < 1.0):
        raise ValueError(
            f"alpha must be strictly inside (0, 1), got {alpha_val!r}"
        )
    return alpha_val


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction for multiple testing.

    Args:
        p_values: Array of p-values, any shape. Non-finite entries are
            treated as missing tests (see module docstring); finite
            entries must lie in [0, 1].
        alpha: Family-wise error rate, strictly inside (0, 1)

    Returns:
        (adjusted_p_values, reject_mask)
        adjusted_p_values: Bonferroni-adjusted p-values (min(p * n_tests, 1.0))
        reject_mask: Boolean mask where null hypothesis is rejected

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``alpha`` is not strictly inside (0, 1)
    """
    alpha = _validate_alpha(alpha)
    p_values = _validate_p_values(p_values)

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
        p_values: Array of p-values, any shape. Non-finite entries are
            treated as missing tests (see module docstring); finite
            entries must lie in [0, 1].
        alpha: False discovery rate, strictly inside (0, 1)

    Returns:
        (adjusted_p_values, reject_mask, n_discoveries)
        adjusted_p_values: BH-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected
        n_discoveries: Number of discoveries (rejections)

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``alpha`` is not strictly inside (0, 1)
    """
    alpha = _validate_alpha(alpha)
    p_values = _validate_p_values(p_values)

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

    # Compute BH adjusted p-values. In sorted order, they must be
    # non-decreasing, so propagate each smaller value toward lower ranks.
    adjusted_sorted = np.minimum.accumulate(
        (p_sorted * n_tests / ranks)[::-1]
    )[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Map back to original positions
    adjusted_flat = np.full_like(p_flat, np.nan)
    adjusted_flat[original_idx] = adjusted_sorted

    # Rejection mask
    reject_flat = np.zeros_like(p_flat, dtype=bool)
    if n_discoveries > 0:
        reject_indices = original_idx[:n_discoveries]
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
        p_values: Array of p-values, any shape. Non-finite entries are
            treated as missing tests (see module docstring); finite
            entries must lie in [0, 1].
        alpha: Family-wise error rate, strictly inside (0, 1)

    Returns:
        (adjusted_p_values, reject_mask, n_discoveries)
        adjusted_p_values: Holm-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected
        n_discoveries: Number of discoveries (rejections)

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``alpha`` is not strictly inside (0, 1)
    """
    alpha = _validate_alpha(alpha)
    p_values = _validate_p_values(p_values)

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

    # Compute Holm adjusted p-values. In sorted order, adjusted values must
    # be non-decreasing, so each rank inherits the largest prior value.
    adjusted_sorted = np.minimum(
        np.maximum.accumulate(p_sorted * (n_tests - ranks + 1)),
        1.0,
    )

    # Map back to original positions
    adjusted_flat = np.full_like(p_flat, np.nan)
    adjusted_flat[original_idx] = adjusted_sorted

    # Rejection mask
    reject_flat = np.zeros_like(p_flat, dtype=bool)
    if n_discoveries > 0:
        reject_indices = original_idx[:n_discoveries]
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
        p_values: Array of p-values, any shape. Non-finite entries are
            treated as missing tests (see module docstring); finite
            entries must lie in [0, 1].
        alpha: Family-wise error rate, strictly inside (0, 1)

    Returns:
        (adjusted_p_values, reject_mask)
        adjusted_p_values: Šidák-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``alpha`` is not strictly inside (0, 1)
    """
    alpha = _validate_alpha(alpha)
    p_values = _validate_p_values(p_values)

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

    .. warning::
        **EXPERIMENTAL** (QE-METRIC P0-14 demotion). This estimator uses
        the mean p-value of the *rejected* tests as a plug-in alpha
        ("conservative estimate"), which is a crude heuristic — it does
        not correspond to any standard FDR estimator (BH, BY, Storey
        q-values) and can be badly biased when rejections are few or the
        p-value distribution is non-uniform. It is NOT registered in the
        metric registry (``registry/metrics.py``) and should not be used
        for reported results without independent validation. Prefer
        ``benjamini_hochberg_correction`` for standard FDR control.

    Args:
        p_values: Array of p-values. Non-finite entries are treated as
            missing tests; finite entries must lie in [0, 1].
        reject_mask: Boolean mask of rejections, same shape as
            ``p_values``

    Returns:
        FDR estimate (expected proportion of false discoveries)

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``reject_mask`` is not boolean with a
            shape matching ``p_values``
    """
    p_values = _validate_p_values(p_values)

    reject_mask = np.asarray(reject_mask)
    if reject_mask.dtype != np.bool_:
        raise ValueError(
            f"reject_mask must be a boolean array, got dtype "
            f"{reject_mask.dtype}"
        )
    if reject_mask.shape != p_values.shape:
        raise ValueError(
            f"reject_mask shape {reject_mask.shape} does not match "
            f"p_values shape {p_values.shape}"
        )

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
