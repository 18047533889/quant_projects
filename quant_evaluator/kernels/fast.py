"""
Fast vectorized kernels for large-scale factor evaluation.

Optimized implementations that maintain mathematical parity with reference
implementations while maximizing throughput for 10k+ factor batches.
"""

from typing import Tuple, Optional
import numpy as np


def fast_ic_batch(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    method: str = "pearson",
    min_obs: int = 10,
    factor_validity: Optional[np.ndarray] = None,
    label_validity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized IC computation across all factors and time periods.

    Computes correlation between factors and labels with pairwise-finite
    filtering, optimized for large batches.

    Args:
        factor_values: Factor batch (T, N, F)
        label_values: Labels (T, N) or (T,)
        method: "pearson" or "spearman"
        min_obs: Minimum valid observations per period
        factor_validity: Optional validity mask (T, N, F)
        label_validity: Optional validity mask (T, N)

    Returns:
        (ic_matrix, valid_counts)
        ic_matrix: shape (T, F) with IC per day per factor
        valid_counts: shape (T, F) with count of valid obs per day per factor
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}")

    T, N, F = factor_values.shape

    # Broadcast labels if needed
    if label_values.ndim == 1:
        label_values = label_values[:, np.newaxis]  # (T, 1)
        labels_broadcast = np.broadcast_to(label_values, (T, N))
    elif label_values.shape == (T, N):
        labels_broadcast = label_values
    else:
        raise ValueError(f"Invalid label shape: {label_values.shape}, expected (T,) or (T, N)")

    # Apply validity masks
    factors = factor_values.copy()
    if factor_validity is not None:
        factors = np.where(factor_validity, factors, np.nan)

    labels = labels_broadcast.copy()
    if label_validity is not None:
        labels = np.where(label_validity[:, :, np.newaxis], labels[:, :, np.newaxis], np.nan)
        labels = labels[:, :, 0]  # Back to (T, N)

    # Expand labels for broadcasting: (T, N, 1)
    labels_expanded = labels[:, :, np.newaxis]

    # Compute pairwise finite mask: (T, N, F)
    finite_mask = np.isfinite(factors) & np.isfinite(labels_expanded)

    # Count valid observations per (t, f)
    valid_counts = np.sum(finite_mask, axis=1, dtype=np.int32)  # (T, F)

    # Initialize output
    ic_matrix = np.full((T, F), np.nan, dtype=np.float64)

    if method == "pearson":
        # Fully vectorized Pearson correlation across all (T, F) pairs
        # For each (t, f), compute correlation only on pairwise-finite observations

        # Sum values and squared values where finite: (T, F)
        sum_x = np.where(finite_mask, factors, 0.0).sum(axis=1)  # (T, F)
        sum_y = np.where(finite_mask, labels_expanded, 0.0).sum(axis=1)  # (T, F)
        sum_xx = np.where(finite_mask, factors ** 2, 0.0).sum(axis=1)  # (T, F)
        sum_yy = np.where(finite_mask, labels_expanded ** 2, 0.0).sum(axis=1)  # (T, F)
        sum_xy = np.where(finite_mask, factors * labels_expanded, 0.0).sum(axis=1)  # (T, F)

        # Number of valid pairs: (T, F)
        n = valid_counts.astype(np.float64)

        # Compute correlation using the formula: corr = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_xx - sum_x^2) * (n*sum_yy - sum_y^2))
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = n * sum_xy - sum_x * sum_y
            denom_x = n * sum_xx - sum_x ** 2
            denom_y = n * sum_yy - sum_y ** 2
            ic_matrix = numerator / np.sqrt(denom_x * denom_y)

        # Mask insufficient observations or zero variance
        insufficient = valid_counts < min_obs
        zero_var = (denom_x <= 0) | (denom_y <= 0)
        ic_matrix = np.where(insufficient | zero_var, np.nan, ic_matrix)

    else:  # spearman
        # Vectorized Spearman: rank each (t, f) slice then compute Pearson on ranks
        # We need to rank per (t, f) pair, which requires iteration, but we can
        # vectorize the correlation computation after ranking

        ranked_factors = np.full((T, N, F), np.nan, dtype=np.float64)
        ranked_labels = np.full((T, N, F), np.nan, dtype=np.float64)

        # Rank each (t, f) independently
        for t in range(T):
            for f in range(F):
                if valid_counts[t, f] < min_obs:
                    continue

                mask = finite_mask[t, :, f]
                x = factors[t, mask, f]
                y = labels[t, mask]

                # Check for constants
                if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
                    continue

                # Rank with average tie handling
                rank_x = _fast_rank(x)
                rank_y = _fast_rank(y)

                # Store ranks back in full arrays
                ranked_factors[t, mask, f] = rank_x
                ranked_labels[t, mask, f] = rank_y

        # Now compute Pearson correlation on ranked data (vectorized)
        # Recompute finite mask for ranked data
        ranked_finite_mask = np.isfinite(ranked_factors) & np.isfinite(ranked_labels)
        ranked_counts = np.sum(ranked_finite_mask, axis=1, dtype=np.int32)  # (T, F)

        # Mask invalid observations
        factors_masked = np.where(ranked_finite_mask, ranked_factors, np.nan)
        labels_masked = np.where(ranked_finite_mask, ranked_labels, np.nan)

        # Compute means: (T, F)
        means_x = np.nanmean(factors_masked, axis=1)
        means_y = np.nanmean(labels_masked, axis=1)

        # Center the data: (T, N, F)
        factors_centered = factors_masked - means_x[:, np.newaxis, :]
        labels_centered = labels_masked - means_y[:, np.newaxis, :]

        # Compute covariance and std
        factors_c_clean = np.where(ranked_finite_mask, factors_centered, 0.0)
        labels_c_clean = np.where(ranked_finite_mask, labels_centered, 0.0)

        # Covariance: (T, F)
        cov = np.sum(factors_c_clean * labels_c_clean, axis=1)

        # Standard deviations: (T, F)
        std_x = np.sqrt(np.sum(factors_c_clean ** 2, axis=1))
        std_y = np.sqrt(np.sum(labels_c_clean ** 2, axis=1))

        # Pearson correlation on ranks: (T, F)
        with np.errstate(divide='ignore', invalid='ignore'):
            ic_matrix = cov / (std_x * std_y)

        # Mask insufficient observations or zero variance
        insufficient = ranked_counts < min_obs
        zero_var = (std_x == 0) | (std_y == 0)
        ic_matrix = np.where(insufficient | zero_var, np.nan, ic_matrix)

    return ic_matrix, valid_counts


def _fast_rank(x: np.ndarray) -> np.ndarray:
    """
    Fast ranking with average tie handling.

    Args:
        x: Input array (1D)

    Returns:
        Average ranks (1D)
    """
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    # Handle ties with average
    for val in np.unique(x):
        mask = x == val
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])

    return ranks


def fast_quantile_binning(
    factor_values: np.ndarray,
    n_quantiles: int = 5,
    min_valid: Optional[int] = None,
) -> np.ndarray:
    """
    Fast quantile binning with vectorized operations.

    Assigns quantile IDs (0 to n_quantiles-1) to factor values at each time period,
    with -1 for invalid/NaN values.

    Args:
        factor_values: Factor batch (T, N, F)
        n_quantiles: Number of quantiles
        min_valid: Minimum valid assets per period (defaults to n_quantiles)

    Returns:
        Quantile assignments (T, N, F) with dtype int32
    """
    if min_valid is None:
        min_valid = n_quantiles

    T, N, F = factor_values.shape
    quantiles = np.full((T, N, F), -1, dtype=np.int32)

    for t in range(T):
        for f in range(F):
            v = factor_values[t, :, f]
            finite_mask = np.isfinite(v)
            n_valid = np.sum(finite_mask)

            if n_valid < min_valid:
                continue

            v_finite = v[finite_mask]

            # Fast argsort-based ranking
            order = np.argsort(v_finite)
            ranks = np.empty(n_valid, dtype=np.int32)
            ranks[order] = np.arange(n_valid, dtype=np.int32)

            # Convert ranks to quantile bins using vectorized floor division
            q_bins = (ranks * n_quantiles) // n_valid
            q_bins = np.clip(q_bins, 0, n_quantiles - 1)

            quantiles[t, finite_mask, f] = q_bins

    return quantiles


def compute_quantile_returns_fast(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    n_quantiles: int = 5,
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast quantile return computation with vectorized aggregation.

    Args:
        factor_values: Factor batch (T, N, F)
        label_values: Forward returns (T, N)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    T, N, F = factor_values.shape

    # Broadcast labels if needed
    if label_values.ndim == 1:
        labels = np.broadcast_to(label_values[:, np.newaxis], (T, N))
    else:
        labels = label_values

    # Get quantile assignments
    q_assignments = fast_quantile_binning(factor_values, n_quantiles=n_quantiles)

    quantile_returns = np.full((T, n_quantiles, F), np.nan, dtype=np.float64)
    quantile_counts = np.zeros((T, n_quantiles, F), dtype=np.int32)

    # Vectorized aggregation per quantile
    for t in range(T):
        label_t = labels[t, :]

        for f in range(F):
            q_assign_tf = q_assignments[t, :, f]

            for q in range(n_quantiles):
                q_mask = (q_assign_tf == q) & np.isfinite(label_t)
                count = int(np.sum(q_mask))

                if count >= min_assets:
                    quantile_returns[t, q, f] = np.mean(label_t[q_mask])
                    quantile_counts[t, q, f] = count

    return quantile_returns, quantile_counts


def fast_turnover_estimate(
    factor_values: np.ndarray,
    window: int = 1,
    min_obs: int = 10,
) -> np.ndarray:
    """
    Fast turnover estimation from rank correlation changes.

    Turnover proxy: 1 - abs(rank_correlation(t-window, t))
    Vectorized across factors.

    Args:
        factor_values: Factor batch (T, N, F)
        window: Lag periods for comparison
        min_obs: Minimum overlapping observations

    Returns:
        Turnover estimate (T, F), first 'window' periods are NaN
    """
    from scipy.stats import spearmanr

    T, N, F = factor_values.shape
    turnover_est = np.full((T, F), np.nan, dtype=np.float64)

    # Process each time period
    for t in range(window, T):
        v_t0 = factor_values[t - window, :, :]  # (N, F)
        v_t1 = factor_values[t, :, :]  # (N, F)

        # Compute mask per factor
        mask = np.isfinite(v_t0) & np.isfinite(v_t1)  # (N, F)
        n_valid = np.sum(mask, axis=0)  # (F,)

        # Process each factor with enough observations
        for f in range(F):
            if n_valid[f] < min_obs:
                continue

            m = mask[:, f]
            x = v_t0[m, f]
            y = v_t1[m, f]

            # Use scipy's optimized spearmanr
            corr, _ = spearmanr(x, y)

            if np.isfinite(corr):
                turnover_est[t, f] = 1.0 - abs(corr)

    return turnover_est
