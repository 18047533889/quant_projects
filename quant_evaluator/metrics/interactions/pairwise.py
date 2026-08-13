"""
Pairwise factor correlation analysis.

Computes correlations between factor pairs over time to detect redundancy
and track evolving relationships.
"""

from typing import Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch


def compute_pairwise_correlation(
    factor_batch: FactorBatch,
    method: str = "pearson",
    min_obs: int = 30,
) -> np.ndarray:
    """
    Compute pairwise correlation matrix between factors across entire time period.

    Args:
        factor_batch: Input factor batch (T, N, F)
        method: "pearson" or "spearman"
        min_obs: Minimum observations required

    Returns:
        Correlation matrix of shape (F, F)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape

    # Reshape to (T*N, F) for correlation computation
    values_flat = values.reshape(-1, F)

    # Apply validity mask if present
    if factor_batch.validity is not None:
        validity_flat = factor_batch.validity.reshape(-1, F)
        values_flat = np.where(validity_flat, values_flat, np.nan)

    corr_matrix = np.full((F, F), np.nan, dtype=np.float64)

    for i in range(F):
        for j in range(i, F):
            x = values_flat[:, i]
            y = values_flat[:, j]

            # Pairwise finite mask
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            if len(x_valid) < min_obs:
                continue

            # Check for constants
            if np.std(x_valid) == 0 or np.std(y_valid) == 0:
                corr_matrix[i, j] = 1.0 if i == j else np.nan
                corr_matrix[j, i] = corr_matrix[i, j]
                continue

            if method == "pearson":
                corr = np.corrcoef(x_valid, y_valid)[0, 1]
            else:
                # Spearman via ranks
                from scipy import stats
                corr, _ = stats.spearmanr(x_valid, y_valid)

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def compute_rolling_pairwise_correlation(
    factor_batch: FactorBatch,
    window: int,
    method: str = "pearson",
    min_obs: int = 30,
) -> np.ndarray:
    """
    Compute rolling pairwise correlation over time.

    Args:
        factor_batch: Input factor batch (T, N, F)
        window: Rolling window size in time periods
        method: "pearson" or "spearman"
        min_obs: Minimum observations per window

    Returns:
        Rolling correlation array of shape (T, F, F)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape

    rolling_corr = np.full((T, F, F), np.nan, dtype=np.float64)

    for t in range(window - 1, T):
        window_start = max(0, t - window + 1)
        window_values = values[window_start:t+1, :, :]  # (W, N, F)

        # Reshape for correlation
        window_flat = window_values.reshape(-1, F)

        # Apply validity mask if present
        if factor_batch.validity is not None:
            window_validity = factor_batch.validity[window_start:t+1, :, :]
            window_validity_flat = window_validity.reshape(-1, F)
            window_flat = np.where(window_validity_flat, window_flat, np.nan)

        # Compute correlation for this window
        for i in range(F):
            for j in range(i, F):
                x = window_flat[:, i]
                y = window_flat[:, j]

                valid_mask = np.isfinite(x) & np.isfinite(y)
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                if len(x_valid) < min_obs:
                    continue

                if np.std(x_valid) == 0 or np.std(y_valid) == 0:
                    rolling_corr[t, i, j] = 1.0 if i == j else np.nan
                    rolling_corr[t, j, i] = rolling_corr[t, i, j]
                    continue

                if method == "pearson":
                    corr = np.corrcoef(x_valid, y_valid)[0, 1]
                else:
                    from scipy import stats
                    corr, _ = stats.spearmanr(x_valid, y_valid)

                rolling_corr[t, i, j] = corr
                rolling_corr[t, j, i] = corr

    return rolling_corr


def compute_correlation_matrix(
    factor_batch: FactorBatch,
    cross_sectional: bool = False,
    method: str = "pearson",
    min_obs: int = 10,
) -> np.ndarray:
    """
    Compute correlation matrix with configurable aggregation.

    Args:
        factor_batch: Input factor batch (T, N, F)
        cross_sectional: If True, compute average cross-sectional correlation
                        If False, compute time-series correlation (default)
        method: "pearson" or "spearman"
        min_obs: Minimum observations required

    Returns:
        Correlation matrix of shape (F, F)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape

    if cross_sectional:
        # Average correlation across time periods
        corr_matrices = []

        for t in range(T):
            values_t = values[t, :, :]  # (N, F)

            # Apply validity mask
            if factor_batch.validity is not None:
                validity_t = factor_batch.validity[t, :, :]
                values_t = np.where(validity_t, values_t, np.nan)

            # Check if enough valid assets
            valid_assets = np.sum(np.all(np.isfinite(values_t), axis=1))
            if valid_assets < min_obs:
                continue

            # Compute correlation for this time period
            corr_t = np.full((F, F), np.nan, dtype=np.float64)
            for i in range(F):
                for j in range(i, F):
                    x = values_t[:, i]
                    y = values_t[:, j]

                    valid_mask = np.isfinite(x) & np.isfinite(y)
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]

                    if len(x_valid) < min_obs:
                        continue

                    if np.std(x_valid) == 0 or np.std(y_valid) == 0:
                        corr_t[i, j] = 1.0 if i == j else np.nan
                        corr_t[j, i] = corr_t[i, j]
                        continue

                    if method == "pearson":
                        corr = np.corrcoef(x_valid, y_valid)[0, 1]
                    else:
                        from scipy import stats
                        corr, _ = stats.spearmanr(x_valid, y_valid)

                    corr_t[i, j] = corr
                    corr_t[j, i] = corr

            corr_matrices.append(corr_t)

        if not corr_matrices:
            return np.full((F, F), np.nan, dtype=np.float64)

        # Average across time
        corr_stack = np.stack(corr_matrices, axis=0)
        with np.errstate(invalid='ignore'):
            avg_corr = np.nanmean(corr_stack, axis=0)

        return avg_corr

    else:
        # Time-series correlation (default behavior)
        return compute_pairwise_correlation(factor_batch, method=method, min_obs=min_obs)
