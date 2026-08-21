"""
High-performance Numba JIT-compiled kernels for factor evaluation.

Provides 10-50x speedup over NumPy implementations through:
- @njit compilation with parallel=True for multi-threaded execution
- fastmath=True for aggressive optimization
- Cache-friendly memory access patterns
- Elimination of Python overhead in tight loops

Target functions:
- IC computation (Pearson/Spearman)
- Quantile binning
- Rolling statistics
- Correlation matrices
"""

from typing import Tuple
import numpy as np
from numba import njit, prange


# =============================================================================
# IC Computation Kernels
# =============================================================================

@njit(parallel=True, cache=True)  # Remove fastmath for NaN safety in IC computation
def numba_pearson_ic_batch(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    min_obs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated Pearson IC computation.

    Computes IC across all factors and time periods with pairwise-finite
    filtering and parallel execution.

    Args:
        factor_values: (T, N, F) factor batch
        label_values: (T, N) labels
        min_obs: Minimum valid observations per period

    Returns:
        (ic_matrix, valid_counts)
        ic_matrix: (T, F) IC per day per factor
        valid_counts: (T, F) count of valid obs per day per factor
    """
    T, N, F = factor_values.shape

    ic_matrix = np.full((T, F), np.nan, dtype=np.float64)
    valid_counts = np.zeros((T, F), dtype=np.int32)

    # Parallelize over (t, f) pairs
    for t in prange(T):
        for f in range(F):
            # Extract vectors for this (t, f) pair
            x = factor_values[t, :, f]
            y = label_values[t, :]

            # Count pairwise-finite observations
            n_valid = 0
            for i in range(N):
                if np.isfinite(x[i]) and np.isfinite(y[i]):
                    n_valid += 1

            valid_counts[t, f] = n_valid

            if n_valid < min_obs:
                continue

            # Compute correlation using Welford-style algorithm
            sum_x = 0.0
            sum_y = 0.0
            sum_xx = 0.0
            sum_yy = 0.0
            sum_xy = 0.0

            for i in range(N):
                if np.isfinite(x[i]) and np.isfinite(y[i]):
                    xi = x[i]
                    yi = y[i]
                    sum_x += xi
                    sum_y += yi
                    sum_xx += xi * xi
                    sum_yy += yi * yi
                    sum_xy += xi * yi

            # Pearson correlation formula
            n = float(n_valid)
            numerator = n * sum_xy - sum_x * sum_y
            denom_x = n * sum_xx - sum_x * sum_x
            denom_y = n * sum_yy - sum_y * sum_y

            if denom_x > 0.0 and denom_y > 0.0:
                ic_matrix[t, f] = numerator / np.sqrt(denom_x * denom_y)

    return ic_matrix, valid_counts


@njit(cache=True)  # Remove fastmath for NaN safety
def _rank_with_ties(x: np.ndarray) -> np.ndarray:
    """
    Fast ranking with average tie handling.

    Args:
        x: Input array (1D, all finite)

    Returns:
        Average ranks (1D)
    """
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)

    # Assign initial ranks
    for i in range(n):
        ranks[order[i]] = float(i)

    # Handle ties with average
    i = 0
    while i < n:
        val = x[order[i]]
        j = i + 1

        # Find all equal values
        while j < n and x[order[j]] == val:
            j += 1

        # Average the ranks for tied values
        if j > i + 1:
            avg_rank = 0.0
            for k in range(i, j):
                avg_rank += ranks[order[k]]
            avg_rank /= float(j - i)

            for k in range(i, j):
                ranks[order[k]] = avg_rank

        i = j

    return ranks


@njit(parallel=True, cache=True)  # Remove fastmath for NaN safety
def numba_spearman_ic_batch(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    min_obs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated Spearman IC computation.

    Computes rank correlation across all factors and time periods.

    Args:
        factor_values: (T, N, F) factor batch
        label_values: (T, N) labels
        min_obs: Minimum valid observations per period

    Returns:
        (ic_matrix, valid_counts)
        ic_matrix: (T, F) IC per day per factor
        valid_counts: (T, F) count of valid obs per day per factor
    """
    T, N, F = factor_values.shape

    ic_matrix = np.full((T, F), np.nan, dtype=np.float64)
    valid_counts = np.zeros((T, F), dtype=np.int32)

    # Parallelize over (t, f) pairs
    for t in prange(T):
        for f in range(F):
            # Extract vectors
            x_raw = factor_values[t, :, f]
            y_raw = label_values[t, :]

            # Filter to pairwise-finite
            n_valid = 0
            for i in range(N):
                if np.isfinite(x_raw[i]) and np.isfinite(y_raw[i]):
                    n_valid += 1

            valid_counts[t, f] = n_valid

            if n_valid < min_obs:
                continue

            # Extract finite values
            x_finite = np.empty(n_valid, dtype=np.float64)
            y_finite = np.empty(n_valid, dtype=np.float64)
            idx = 0
            for i in range(N):
                if np.isfinite(x_raw[i]) and np.isfinite(y_raw[i]):
                    x_finite[idx] = x_raw[i]
                    y_finite[idx] = y_raw[i]
                    idx += 1

            # Check for constant values
            x_min = x_finite[0]
            x_max = x_finite[0]
            y_min = y_finite[0]
            y_max = y_finite[0]
            for i in range(1, n_valid):
                if x_finite[i] < x_min:
                    x_min = x_finite[i]
                if x_finite[i] > x_max:
                    x_max = x_finite[i]
                if y_finite[i] < y_min:
                    y_min = y_finite[i]
                if y_finite[i] > y_max:
                    y_max = y_finite[i]

            if x_min == x_max or y_min == y_max:
                continue

            # Rank both arrays
            rank_x = _rank_with_ties(x_finite)
            rank_y = _rank_with_ties(y_finite)

            # Compute Pearson on ranks
            sum_rx = 0.0
            sum_ry = 0.0
            for i in range(n_valid):
                sum_rx += rank_x[i]
                sum_ry += rank_y[i]

            mean_rx = sum_rx / float(n_valid)
            mean_ry = sum_ry / float(n_valid)

            cov = 0.0
            var_x = 0.0
            var_y = 0.0
            for i in range(n_valid):
                dx = rank_x[i] - mean_rx
                dy = rank_y[i] - mean_ry
                cov += dx * dy
                var_x += dx * dx
                var_y += dy * dy

            if var_x > 0.0 and var_y > 0.0:
                ic_matrix[t, f] = cov / np.sqrt(var_x * var_y)

    return ic_matrix, valid_counts


# =============================================================================
# Quantile Binning Kernels
# =============================================================================

@njit(parallel=True, cache=True)  # Remove fastmath to preserve NaN handling
def numba_quantile_binning(
    factor_values: np.ndarray,
    n_quantiles: int,
    min_valid: int,
) -> np.ndarray:
    """
    Numba-accelerated quantile binning.

    Assigns quantile IDs (0 to n_quantiles-1) to factor values at each
    time period, with -1 for invalid/NaN values.

    Args:
        factor_values: (T, N, F) factor batch
        n_quantiles: Number of quantiles
        min_valid: Minimum valid assets per period

    Returns:
        Quantile assignments (T, N, F) with dtype int32
    """
    T, N, F = factor_values.shape
    quantiles = np.full((T, N, F), -1, dtype=np.int32)

    # Parallelize over (t, f) pairs
    for t in prange(T):
        for f in range(F):
            v = factor_values[t, :, f]

            # Count valid observations
            n_valid = 0
            for i in range(N):
                if np.isfinite(v[i]):
                    n_valid += 1

            if n_valid < min_valid:
                continue

            # Extract finite values and their indices
            v_finite = np.empty(n_valid, dtype=np.float64)
            indices = np.empty(n_valid, dtype=np.int32)
            idx = 0
            for i in range(N):
                if np.isfinite(v[i]):
                    v_finite[idx] = v[i]
                    indices[idx] = i
                    idx += 1

            # Argsort finite values once; the sorted view feeds boundary
            # interpolation below (side='right' binning ignores tie ranks).
            order = np.argsort(v_finite)
            v_sorted = v_finite[order]

            # QE-Q-P0-002: percentile boundaries + searchsorted side='right'
            # (reference semantics, QuantileTiePolicy.MAX). The previous
            # rank-floor formula ((rank * n_quantiles) // n_valid) disagreed
            # with the reference whenever n_valid was not divisible by
            # n_quantiles.
            n_boundaries = n_quantiles - 1
            boundaries = np.empty(n_boundaries, dtype=np.float64)
            for b in range(n_boundaries):
                # Linear percentile position b+1 of n_quantiles (matches
                # np.percentile default linear interpolation). Positions
                # within 1e-9 of an integer snap to the exact sorted value
                # so every backend agrees at tie-at-boundary positions
                # (float pos can land one ulp above an exact integer).
                pos = (b + 1) / n_quantiles * (n_valid - 1)
                lo = int(pos)
                frac = pos - lo
                if lo >= n_valid - 1:
                    boundaries[b] = v_sorted[lo]
                elif frac < 1e-9:
                    boundaries[b] = v_sorted[lo]
                elif frac > 1.0 - 1e-9:
                    boundaries[b] = v_sorted[lo + 1]
                else:
                    boundaries[b] = v_sorted[lo] + frac * (v_sorted[lo + 1] - v_sorted[lo])

            # Assign quantile bins only to valid indices (side='right')
            for i in range(n_valid):
                value = v_finite[i]
                q_bin = 0
                for b in range(n_boundaries):
                    if value >= boundaries[b]:
                        q_bin = b + 1
                    else:
                        break
                if q_bin >= n_quantiles:
                    q_bin = n_quantiles - 1

                indices_i = indices[i]
                quantiles[t, indices_i, f] = q_bin

    return quantiles


@njit(parallel=True, cache=True)  # Remove fastmath for NaN safety
def numba_quantile_returns(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    n_quantiles: int,
    min_assets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated quantile return computation.

    Args:
        factor_values: (T, N, F) factor batch
        label_values: (T, N) forward returns
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: (T, n_quantiles, F)
        quantile_counts: (T, n_quantiles, F)
    """
    T, N, F = factor_values.shape

    # First get quantile assignments
    q_assignments = numba_quantile_binning(factor_values, n_quantiles, n_quantiles)

    quantile_returns = np.full((T, n_quantiles, F), np.nan, dtype=np.float64)
    quantile_counts = np.zeros((T, n_quantiles, F), dtype=np.int32)

    # Parallelize over (t, f) pairs
    for t in prange(T):
        for f in range(F):
            for q in range(n_quantiles):
                # Count and sum for this quantile
                count = 0
                sum_ret = 0.0

                for n in range(N):
                    if q_assignments[t, n, f] == q and np.isfinite(label_values[t, n]):
                        count += 1
                        sum_ret += label_values[t, n]

                # Report the raw membership count (matching the NumPy fast
                # kernel), not the post-threshold value — a quantile with
                # count < min_assets still has members even though its mean
                # return stays NaN.
                quantile_counts[t, q, f] = count
                if count >= min_assets:
                    quantile_returns[t, q, f] = sum_ret / float(count)

    return quantile_returns, quantile_counts


# =============================================================================
# Rolling Statistics Kernels
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def numba_rolling_mean(
    values: np.ndarray,
    window: int,
    min_periods: int,
) -> np.ndarray:
    """
    Numba-accelerated rolling mean.

    Args:
        values: (T, F) time series
        window: Rolling window size
        min_periods: Minimum observations required

    Returns:
        Rolling means (T, F)
    """
    T, F = values.shape
    result = np.full((T, F), np.nan, dtype=np.float64)

    for f in prange(F):
        for t in range(T):
            start = max(0, t - window + 1)

            # Count valid and sum
            count = 0
            total = 0.0
            for i in range(start, t + 1):
                if np.isfinite(values[i, f]):
                    count += 1
                    total += values[i, f]

            if count >= min_periods:
                result[t, f] = total / float(count)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def numba_rolling_std(
    values: np.ndarray,
    window: int,
    min_periods: int,
) -> np.ndarray:
    """
    Numba-accelerated rolling standard deviation.

    Args:
        values: (T, F) time series
        window: Rolling window size
        min_periods: Minimum observations required

    Returns:
        Rolling standard deviations (T, F)
    """
    T, F = values.shape
    result = np.full((T, F), np.nan, dtype=np.float64)

    for f in prange(F):
        for t in range(T):
            start = max(0, t - window + 1)

            # First pass: mean
            count = 0
            total = 0.0
            for i in range(start, t + 1):
                if np.isfinite(values[i, f]):
                    count += 1
                    total += values[i, f]

            if count < min_periods:
                continue

            mean = total / float(count)

            # Second pass: variance
            var_sum = 0.0
            for i in range(start, t + 1):
                if np.isfinite(values[i, f]):
                    diff = values[i, f] - mean
                    var_sum += diff * diff

            result[t, f] = np.sqrt(var_sum / float(count))

    return result


@njit(parallel=True, fastmath=True, cache=True)
def numba_rolling_corr(
    x_values: np.ndarray,
    y_values: np.ndarray,
    window: int,
    min_periods: int,
) -> np.ndarray:
    """
    Numba-accelerated rolling correlation between two time series.

    Args:
        x_values: (T, F) first time series
        y_values: (T, F) second time series
        window: Rolling window size
        min_periods: Minimum observations required

    Returns:
        Rolling correlations (T, F)
    """
    T, F = x_values.shape
    result = np.full((T, F), np.nan, dtype=np.float64)

    for f in prange(F):
        for t in range(T):
            start = max(0, t - window + 1)

            # Count pairwise-finite
            count = 0
            for i in range(start, t + 1):
                if np.isfinite(x_values[i, f]) and np.isfinite(y_values[i, f]):
                    count += 1

            if count < min_periods:
                continue

            # Compute correlation
            sum_x = 0.0
            sum_y = 0.0
            sum_xx = 0.0
            sum_yy = 0.0
            sum_xy = 0.0

            for i in range(start, t + 1):
                if np.isfinite(x_values[i, f]) and np.isfinite(y_values[i, f]):
                    xi = x_values[i, f]
                    yi = y_values[i, f]
                    sum_x += xi
                    sum_y += yi
                    sum_xx += xi * xi
                    sum_yy += yi * yi
                    sum_xy += xi * yi

            n = float(count)
            numerator = n * sum_xy - sum_x * sum_y
            denom_x = n * sum_xx - sum_x * sum_x
            denom_y = n * sum_yy - sum_y * sum_y

            if denom_x > 0.0 and denom_y > 0.0:
                result[t, f] = numerator / np.sqrt(denom_x * denom_y)

    return result


# =============================================================================
# Correlation Matrix Kernels
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def numba_corrcoef_matrix(
    values: np.ndarray,
    min_obs: int,
) -> np.ndarray:
    """
    Numba-accelerated correlation matrix computation.

    Computes pairwise correlations between all columns (factors).

    Args:
        values: (T, F) time series matrix
        min_obs: Minimum overlapping observations

    Returns:
        Correlation matrix (F, F)
    """
    T, F = values.shape
    corr_matrix = np.full((F, F), np.nan, dtype=np.float64)

    # Diagonal is always 1.0
    for f in range(F):
        corr_matrix[f, f] = 1.0

    # Compute upper triangle in parallel
    for i in prange(F):
        for j in range(i + 1, F):
            x = values[:, i]
            y = values[:, j]

            # Count pairwise-finite
            n_valid = 0
            for t in range(T):
                if np.isfinite(x[t]) and np.isfinite(y[t]):
                    n_valid += 1

            if n_valid < min_obs:
                continue

            # Compute correlation
            sum_x = 0.0
            sum_y = 0.0
            sum_xx = 0.0
            sum_yy = 0.0
            sum_xy = 0.0

            for t in range(T):
                if np.isfinite(x[t]) and np.isfinite(y[t]):
                    xi = x[t]
                    yi = y[t]
                    sum_x += xi
                    sum_y += yi
                    sum_xx += xi * xi
                    sum_yy += yi * yi
                    sum_xy += xi * yi

            n = float(n_valid)
            numerator = n * sum_xy - sum_x * sum_y
            denom_x = n * sum_xx - sum_x * sum_x
            denom_y = n * sum_yy - sum_y * sum_y

            if denom_x > 0.0 and denom_y > 0.0:
                corr = numerator / np.sqrt(denom_x * denom_y)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    return corr_matrix


# =============================================================================
# High-level API
# =============================================================================

def numba_ic_batch(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    method: str = "pearson",
    min_obs: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-performance IC computation using Numba.

    Args:
        factor_values: (T, N, F) factor batch
        label_values: (T, N) labels
        method: "pearson" or "spearman"
        min_obs: Minimum valid observations per period

    Returns:
        (ic_matrix, valid_counts)
    """
    if method == "pearson":
        return numba_pearson_ic_batch(factor_values, label_values, min_obs)
    elif method == "spearman":
        return numba_spearman_ic_batch(factor_values, label_values, min_obs)
    else:
        raise ValueError(f"Unknown method: {method}")
