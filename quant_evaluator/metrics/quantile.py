"""
Quantile-based metrics and summary statistics.

Optimized implementations:
- Standard: Pure numpy with bincount aggregation
- Numba JIT: 2-5x faster using JIT compilation (optional, requires numba)

Performance:
- For 252 days × 3000 assets × 200 factors:
  - Standard: ~26s
  - Numba: ~18s (1.5x speedup)
- Speedup increases with more factors due to better parallelization

QE-Q-P0-001: Quantile tie policy enforcement
QE-Q-P0-002: NumPy-Numba boundary parity validation
"""

from typing import Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.label_panel import normalize_label_panel
from quant_evaluator.contracts.quantile_policy import (
    QuantileTiePolicy,
    validate_tie_policy,
)

# Import Numba-optimized version if available
try:
    from quant_evaluator.metrics.quantile_numba import (
        compute_quantile_returns_numba,
        is_numba_available,
    )
    _NUMBA_AVAILABLE = is_numba_available()
except ImportError:
    _NUMBA_AVAILABLE = False
    compute_quantile_returns_numba = None


def compute_quantile_returns_fast(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
    use_numba: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute quantile returns with automatic selection of fastest implementation.

    Automatically uses Numba JIT if available (2-5x faster), otherwise falls back
    to optimized numpy implementation.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundle: Forward returns (T, N)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile
        use_numba: If True and numba available, use JIT version (default: True)

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    if use_numba and _NUMBA_AVAILABLE:
        return compute_quantile_returns_numba(
            factor_batch, label_bundle, n_quantiles, min_assets
        )
    else:
        return compute_quantile_returns(
            factor_batch, label_bundle, n_quantiles, min_assets
        )


def _validate_quantile_count(n_quantiles):
    if isinstance(n_quantiles, (bool, np.bool_)) or not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 1:
        raise ValueError("n_quantiles must be a positive integer")


def assign_quantiles(
    values: np.ndarray,
    n_quantiles: int = 5,
    method: str = "max",
) -> np.ndarray:
    """
    Assign quantile IDs to values with tie handling.

    QE-Q-P0-001: Enforces tie-breaking policy contract.

    Args:
        values: Input values (1D or 2D)
        n_quantiles: Number of quantiles
        method: Tie-breaking policy ('min' or 'max'). Default 'max'.
                - 'min': ties at boundary get LOWER bin (searchsorted side='left')
                - 'max': ties at boundary get HIGHER bin (searchsorted side='right')
                'average' and 'first' are not implemented.

    Returns:
        Quantile assignments (0 to n_quantiles-1), or -1 for NaN

    Raises:
        ValueError: If method is invalid
        NotImplementedError: If method is 'average' or 'first'
    """
    # QE-Q-P0-001: Validate that method parameter actually affects behavior
    policy = validate_tie_policy(method)
    _validate_quantile_count(n_quantiles)
    values = np.asarray(values)
    original_ndim = values.ndim
    if original_ndim not in (1, 2):
        raise ValueError("assign_quantiles accepts N or T x N")
    if original_ndim == 1:
        values = values[None, :]

    T, N = values.shape
    quantiles = np.full((T, N), -1, dtype=np.int32)

    for t in range(T):
        v = values[t, :]
        finite_mask = np.isfinite(v)

        n_finite = np.sum(finite_mask)
        if n_finite < n_quantiles:
            continue

        v_finite = v[finite_mask]

        # QE-Q-P0-002: percentile boundaries + tie-policy searchsorted,
        # shared with the fast/Numba/Polars implementations.
        boundaries = _percentile_boundaries(v_finite, n_quantiles)
        q_bins = _searchsorted_bins(boundaries, v_finite, n_quantiles, policy)

        quantiles[t, finite_mask] = q_bins

    return quantiles[0] if original_ndim == 1 else quantiles


def assign_quantiles_fast(
    values: np.ndarray,
    n_quantiles: int = 5,
    method: str = "max",
) -> np.ndarray:
    """
    Fast quantile assignment using percentile-based boundaries.

    QE-Q-P0-001: Enforces tie-breaking policy.

    Uses np.percentile for boundary computation (single pass) and
    vectorized comparison for binning. Significantly faster for large arrays.

    Args:
        values: Input values shape (T, N) or (T, N, F)
        n_quantiles: Number of quantiles
        method: Tie-breaking policy ('min' or 'max'). Default 'max'.

    Returns:
        Quantile assignments (0 to n_quantiles-1), or -1 for NaN
    """
    policy = validate_tie_policy(method)
    _validate_quantile_count(n_quantiles)
    original_ndim = values.ndim
    if original_ndim not in (2, 3):
        raise ValueError("quantile input must be T x N or T x N x F")
    if values.ndim == 2:
        T, N = values.shape
        F = 1
        values = values.reshape(T, N, 1)
    else:
        T, N, F = values.shape

    quantiles = np.full((T, N, F), -1, dtype=np.int32)

    for t in range(T):
        for f in range(F):
            v = values[t, :, f]
            finite_mask = np.isfinite(v)
            n_finite = np.sum(finite_mask)

            if n_finite < n_quantiles:
                continue

            v_finite = v[finite_mask]

            # QE-Q-P0-002: same percentile boundaries + tie policy as the
            # reference implementation (shared helpers).
            boundaries = _percentile_boundaries(v_finite, n_quantiles)
            q_bins = _searchsorted_bins(boundaries, v_finite, n_quantiles, policy)

            quantiles[t, finite_mask, f] = q_bins

    if original_ndim == 2:
        return quantiles.reshape(T, N)
    return quantiles


def _percentile_boundaries(
    v_finite: np.ndarray,
    n_quantiles: int,
) -> np.ndarray:
    """Internal boundary computation shared by every assign_quantiles* path.

    QE-Q-P0-002: all quantile binning implementations (NumPy reference, fast,
    Numba, Polars, CuPy) must derive bins from the same percentile boundaries,
    so the boundary values themselves are computed in exactly one place.

    Boundaries are interpolated on the sorted values with the formula
    ``sv[lo] + frac * (sv[lo+1] - sv[lo])`` at position
    ``pos = (b+1)/n_quantiles * (n-1)``.  Positions within 1e-9 of an integer
    snap to the exact sorted value: np.percentile can land one ulp off the
    data value there (its percentile/100 rounding), which would silently flip
    the tie policy for the value sitting exactly on the boundary.  The snap
    keeps every backend bit-identical at the only positions where exact ties
    are possible.
    """
    sv = np.sort(v_finite)
    n = sv.shape[0]
    boundaries = np.empty(n_quantiles - 1, dtype=np.float64)
    for b in range(n_quantiles - 1):
        pos = (b + 1) / n_quantiles * (n - 1)
        lo = int(pos)
        frac = pos - lo
        if lo >= n - 1:
            boundaries[b] = sv[n - 1]
        elif frac < 1e-9:
            boundaries[b] = sv[lo]
        elif frac > 1.0 - 1e-9:
            boundaries[b] = sv[lo + 1]
        else:
            boundaries[b] = sv[lo] + frac * (sv[lo + 1] - sv[lo])
    return boundaries


def _searchsorted_bins(
    boundaries: np.ndarray,
    v_finite: np.ndarray,
    n_quantiles: int,
    policy: QuantileTiePolicy,
) -> np.ndarray:
    """Internal searchsorted binning honoring the tie policy.

    MIN -> side='left' (value == boundary goes to the LOWER bin)
    MAX -> side='right' (value == boundary goes to the HIGHER bin)
    """
    if policy == QuantileTiePolicy.MIN:
        q_bins = np.searchsorted(boundaries, v_finite, side='left')
    else:  # QuantileTiePolicy.MAX
        q_bins = np.searchsorted(boundaries, v_finite, side='right')
    return np.clip(q_bins, 0, n_quantiles - 1)


def assign_quantiles_batch(
    values: np.ndarray,
    n_quantiles: int = 5,
    method: str = "max",
) -> np.ndarray:
    """
    Ultra-fast batch quantile assignment with minimal Python loops.

    QE-Q-P0-001: Enforces tie-breaking policy.

    Processes entire time×asset×factor tensor with vectorized operations.
    Best performance for large batches.

    Args:
        values: Input values shape (T, N, F)
        n_quantiles: Number of quantiles
        method: Tie-breaking policy ('min' or 'max'). Default 'max'.

    Returns:
        Quantile assignments shape (T, N, F), -1 for NaN
    """
    policy = validate_tie_policy(method)
    _validate_quantile_count(n_quantiles)
    original_ndim = values.ndim
    if original_ndim not in (2, 3):
        raise ValueError("quantile input must be T x N or T x N x F")
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

            # QE-Q-P0-002: same percentile boundaries + tie policy as the
            # reference implementation (shared helpers).
            boundaries = _percentile_boundaries(v_finite, n_quantiles)
            q_bins = _searchsorted_bins(boundaries, v_finite, n_quantiles, policy)

            quantiles[t, mask, f] = q_bins

    return quantiles[:, :, 0] if original_ndim == 2 else quantiles



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
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    labels, label_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )
    if label_validity is not None:
        labels = np.where(label_validity, labels, np.nan)

    T, N, F = values.shape
    quantile_returns = np.full((T, n_quantiles, F), np.nan, dtype=np.float64)
    quantile_counts = np.zeros((T, n_quantiles, F), dtype=np.int32)

    # Process each factor independently
    for f in range(F):
        # Assign quantiles for this factor across all time periods
        q_assignments_f = assign_quantiles_batch(values[:, :, f], n_quantiles=n_quantiles)  # (T, N)

        # Aggregate per time period
        for t in range(T):
            label_t = labels[t, :]
            valid_labels = np.isfinite(label_t)
            q_t_f = q_assignments_f[t, :]  # (N,)

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
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    labels, label_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )
    if label_validity is not None:
        labels = np.where(label_validity, labels, np.nan)

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
    top_q: Optional[int] = None,
    bottom_q: int = 0,
) -> np.ndarray:
    """
    Compute top-bottom quantile spread (highest group minus lowest group).

    Quantile encoding convention (QE2-P0-001):
        quantile 0   = LOWEST factor value group  (bottom)
        quantile N-1 = HIGHEST factor value group (top)

    Spread = top_group_return - bottom_group_return

    For a positively predictive factor (higher value → higher return), the spread
    is positive. Reversing this sign reverses Sharpe ratio sign and all directional
    judgments.

    Args:
        quantile_returns: Quantile returns array of shape (T, n_quantiles, F).
        top_q: Index of the top (highest) quantile group.
               Default None → uses quantile_returns.shape[1] - 1 (i.e. the last
               bin, which is the highest-value group under standard searchsorted
               assignment). Negative indices are accepted (e.g. -1 is equivalent
               to n_quantiles - 1).
        bottom_q: Index of the bottom (lowest) quantile group.
                  Default 0 = the lowest-value group.

    Returns:
        Spread series of shape (T, F): top_returns - bottom_returns.

    Bug history:
        Prior to QE2-P0-001 the defaults were top_q=0, bottom_q=-1, which
        computed Low - High instead of High - Low, reversing the spread sign.
    """
    n_quantiles = quantile_returns.shape[1]
    if top_q is None:
        top_q = n_quantiles - 1

    top_returns = quantile_returns[:, top_q, :]
    bottom_returns = quantile_returns[:, bottom_q, :]

    spread = top_returns - bottom_returns

    return spread
