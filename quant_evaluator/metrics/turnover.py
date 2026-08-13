"""
Turnover metrics for factor portfolios.

Measures stability and trading cost implications.
"""

from typing import Optional
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch

# Try to import numba for JIT compilation
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # No-op decorator if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])


@njit(cache=True)
def _compute_turnover_series_numba(weights: np.ndarray) -> np.ndarray:
    """
    Numba-optimized turnover series computation.

    Note: fastmath disabled to ensure correct NaN handling.

    Args:
        weights: Weight matrix (T, N)

    Returns:
        Turnover series (T,)
    """
    T, N = weights.shape
    turnover = np.empty(T, dtype=np.float64)
    turnover[0] = np.nan

    for t in range(1, T):
        sum_abs_delta = 0.0
        n_valid = 0

        for n in range(N):
            w0 = weights[t-1, n]
            w1 = weights[t, n]

            if np.isfinite(w0) and np.isfinite(w1):
                sum_abs_delta += abs(w1 - w0)
                n_valid += 1

        if n_valid > 0:
            turnover[t] = 0.5 * sum_abs_delta
        else:
            turnover[t] = np.nan

    return turnover


def compute_turnover(
    weights_t0: np.ndarray,
    weights_t1: np.ndarray,
    method: str = "half_sum_abs",
) -> float:
    """
    Compute turnover between two weight vectors.

    Canonical definition: 0.5 * sum(abs(delta_weights))

    Args:
        weights_t0: Weights at time t (N,)
        weights_t1: Weights at time t+1 (N,)
        method: Turnover method ("half_sum_abs")

    Returns:
        Turnover scalar, or NaN if invalid
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    # Compute pairwise finite mask
    mask = np.isfinite(weights_t0) & np.isfinite(weights_t1)

    if np.sum(mask) == 0:
        return np.nan

    w0 = weights_t0[mask]
    w1 = weights_t1[mask]

    # Canonical turnover: 0.5 * sum(abs(delta))
    turnover = 0.5 * np.sum(np.abs(w1 - w0))

    return float(turnover)


def compute_turnover_series(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute turnover time series from weight matrix (vectorized).

    Optimized implementation using numba JIT compilation when available,
    falling back to numpy broadcasting for compatibility.

    Args:
        weights: Weight matrix (T, N)
        method: Turnover method

    Returns:
        Turnover series (T,), first observation is NaN
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N = weights.shape

    if T < 2:
        return np.full(T, np.nan, dtype=np.float64)

    # Use numba-optimized version if available (10-20x faster)
    if HAS_NUMBA:
        return _compute_turnover_series_numba(weights)

    # Fallback: pure numpy vectorized implementation
    turnover_series = np.empty(T, dtype=np.float64)
    turnover_series[0] = np.nan

    # Process differences in-place using slicing
    w_diff = weights[1:] - weights[:-1]  # (T-1, N)

    # Compute finite mask efficiently
    finite_mask = np.isfinite(w_diff)  # (T-1, N)

    # Fast path: if no NaNs, use simple sum
    if np.all(finite_mask):
        turnover_series[1:] = 0.5 * np.sum(np.abs(w_diff), axis=1)
    else:
        # Need to handle NaNs: set them to zero for summation
        abs_diff = np.abs(w_diff)
        abs_diff[~finite_mask] = 0.0

        # Sum and check validity
        turnover_values = 0.5 * np.sum(abs_diff, axis=1)  # (T-1,)
        n_valid = np.sum(finite_mask, axis=1)  # (T-1,)

        # Set to NaN where no valid observations
        turnover_values[n_valid == 0] = np.nan
        turnover_series[1:] = turnover_values

    return turnover_series


def estimate_turnover_from_ranks(
    factor_batch: FactorBatch,
    window: int = 1,
) -> np.ndarray:
    """
    Estimate turnover from factor rank changes (proxy, not actual portfolio).

    This is a diagnostic approximation when actual weights are unavailable.
    Vectorized implementation for efficiency.

    Args:
        factor_batch: Factor values (T, N, F)
        window: Lag for comparison (default 1)

    Returns:
        Estimated turnover series (T, F)
    """
    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape

    turnover_est = np.full((T, F), np.nan, dtype=np.float64)

    # Import once outside loop
    from scipy.stats import spearmanr

    # Process each time period (vectorize across factors within each period)
    for t in range(window, T):
        v_t0 = values[t - window, :, :]  # (N, F)
        v_t1 = values[t, :, :]           # (N, F)

        # Compute finite mask per factor
        mask = np.isfinite(v_t0) & np.isfinite(v_t1)  # (N, F)
        n_valid = np.sum(mask, axis=0)  # (F,)

        # Process each factor with sufficient observations
        for f in range(F):
            if n_valid[f] < 10:
                continue

            m = mask[:, f]
            x = v_t0[m, f]
            y = v_t1[m, f]

            # Rank correlation as turnover proxy
            corr, _ = spearmanr(x, y)

            if np.isfinite(corr):
                # Turnover proxy: 1 - abs(rank_corr)
                # High correlation -> low turnover
                turnover_est[t, f] = 1.0 - abs(corr)

    return turnover_est


def compute_turnover_matrix_batch(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute turnover for a batch of weight matrices (fully vectorized with einsum).

    Optimized for large batches of portfolios (e.g., multiple factors or strategies).
    Uses einsum for maximum efficiency on large arrays.

    Args:
        weights: Weight tensor (T, N, P) where P is number of portfolios
        method: Turnover method

    Returns:
        Turnover series (T, P), first observation is NaN for each portfolio
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N, P = weights.shape

    if T < 2:
        return np.full((T, P), np.nan, dtype=np.float64)

    # For small batches, loop is faster due to overhead
    # Threshold determined empirically
    if P <= 10:
        # Use loop for small batches
        result = np.empty((T, P), dtype=np.float64)
        for p in range(P):
            result[:, p] = compute_turnover_series(weights[:, :, p])
        return result

    # Allocate output
    turnover_series = np.empty((T, P), dtype=np.float64)
    turnover_series[0, :] = np.nan

    # Compute differences (views, not copies)
    w_diff = weights[1:] - weights[:-1]  # (T-1, N, P)

    # Check for NaNs
    has_nans = not np.all(np.isfinite(w_diff))

    if not has_nans:
        # Fast path: no NaNs, direct einsum on absolute differences
        # einsum is faster than sum for large arrays
        turnover_series[1:, :] = 0.5 * np.einsum('tnp->tp', np.abs(w_diff))
    else:
        # Slow path: handle NaNs
        finite_mask = np.isfinite(w_diff)
        abs_diff = np.abs(w_diff)
        abs_diff[~finite_mask] = 0.0

        # Use einsum for summation
        turnover_values = 0.5 * np.einsum('tnp->tp', abs_diff)  # (T-1, P)

        # Check validity per portfolio per period
        n_valid = np.sum(finite_mask, axis=1)  # (T-1, P)
        turnover_values[n_valid == 0] = np.nan

        turnover_series[1:, :] = turnover_values

    return turnover_series


def compute_cross_sectional_turnover(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> float:
    """
    Compute average turnover across all time periods (single scalar).

    Useful for evaluating overall portfolio stability.

    Args:
        weights: Weight matrix (T, N)
        method: Turnover method

    Returns:
        Mean turnover (scalar), or NaN if insufficient data
    """
    turnover_series = compute_turnover_series(weights, method=method)
    return float(np.nanmean(turnover_series))


def compute_weighted_turnover(
    weights: np.ndarray,
    position_sizes: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute position-size-weighted turnover.

    Standard turnover treats all weight changes equally. Weighted turnover
    accounts for the fact that changes in larger positions have greater impact.

    Args:
        weights: Weight matrix (T, N)
        position_sizes: Position sizes (T, N), e.g., abs(weights)
        method: Turnover method

    Returns:
        Weighted turnover series (T,)
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N = weights.shape

    if T < 2:
        return np.full(T, np.nan, dtype=np.float64)

    # Get consecutive pairs
    w_t0 = weights[:-1, :]  # (T-1, N)
    w_t1 = weights[1:, :]   # (T-1, N)
    ps_t0 = position_sizes[:-1, :]  # (T-1, N)
    ps_t1 = position_sizes[1:, :]   # (T-1, N)

    # Average position size across periods
    avg_size = 0.5 * (ps_t0 + ps_t1)  # (T-1, N)

    # Finite mask
    finite_mask = (
        np.isfinite(w_t0) & np.isfinite(w_t1) &
        np.isfinite(ps_t0) & np.isfinite(ps_t1)
    )  # (T-1, N)

    # Weight changes multiplied by average position size
    delta = np.abs(w_t1 - w_t0)  # (T-1, N)
    weighted_delta = np.where(finite_mask, delta * avg_size, 0.0)

    # Weighted turnover: 0.5 * sum(|delta| * size) / sum(size)
    numerator = np.sum(weighted_delta, axis=1)  # (T-1,)
    denominator = np.sum(np.where(finite_mask, avg_size, 0.0), axis=1)  # (T-1,)

    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_turnover = 0.5 * numerator / denominator

    # Mask periods with no valid observations
    n_valid = np.sum(finite_mask, axis=1)
    weighted_turnover = np.where(n_valid > 0, weighted_turnover, np.nan)

    # Prepend NaN
    return np.concatenate([[np.nan], weighted_turnover])


def compute_turnover_contribution(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute per-asset turnover contribution (vectorized with einsum).

    Decomposes total turnover into contributions from each asset.

    Args:
        weights: Weight matrix (T, N)
        method: Turnover method

    Returns:
        Turnover contribution matrix (T, N), where each row sums to that period's turnover
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N = weights.shape

    # Initialize output
    contribution = np.full((T, N), np.nan, dtype=np.float64)

    if T < 2:
        return contribution

    # Get consecutive pairs
    w_t0 = weights[:-1, :]  # (T-1, N)
    w_t1 = weights[1:, :]   # (T-1, N)

    # Finite mask
    finite_mask = np.isfinite(w_t0) & np.isfinite(w_t1)  # (T-1, N)

    # Per-asset contribution: 0.5 * |delta|
    delta = w_t1 - w_t0  # (T-1, N)
    contribution_values = 0.5 * np.where(finite_mask, np.abs(delta), 0.0)  # (T-1, N)

    # Store in output (first row remains NaN)
    contribution[1:, :] = contribution_values

    return contribution
