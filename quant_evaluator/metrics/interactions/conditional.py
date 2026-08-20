"""
Conditional IC analysis.

Measures factor performance conditional on another factor's value or after
controlling for other factors.
"""

from typing import Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.label_panel import normalize_label_panel


def compute_conditional_ic(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    conditioning_factor_idx: int,
    quantiles: int = 5,
    method: str = "pearson",
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IC of each factor conditional on quantiles of a conditioning factor.

    For each quantile of the conditioning factor, compute IC of all factors
    within that subset of assets.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        conditioning_factor_idx: Index of factor to condition on
        quantiles: Number of quantiles to split conditioning factor
        method: "pearson" or "spearman"
        min_assets: Minimum assets per quantile

    Returns:
        (conditional_ic, sample_counts)
        conditional_ic: shape (T, F, Q) - IC per factor per quantile
        sample_counts: shape (T, Q) - asset count per quantile
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    labels, label_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )

    T, N, F = values.shape

    conditional_ic = np.full((T, F, quantiles), np.nan, dtype=np.float64)
    sample_counts = np.zeros((T, quantiles), dtype=np.int32)

    for t in range(T):
        conditioning_values = values[t, :, conditioning_factor_idx]

        # Apply validity mask
        if factor_batch.validity is not None:
            conditioning_valid = factor_batch.validity[t, :, conditioning_factor_idx]
            conditioning_values = np.where(conditioning_valid, conditioning_values, np.nan)

        # Filter finite conditioning values and labels
        valid_mask = np.isfinite(conditioning_values) & np.isfinite(labels[t])
        if np.sum(valid_mask) < min_assets:
            continue

        # Compute quantile breakpoints
        conditioning_finite = conditioning_values[valid_mask]
        try:
            quantile_edges = np.percentile(
                conditioning_finite,
                np.linspace(0, 100, quantiles + 1)
            )
        except (ValueError, IndexError):
            continue

        # Assign assets to quantiles
        quantile_assignments = np.digitize(conditioning_values, quantile_edges[1:-1], right=False)
        quantile_assignments = np.where(valid_mask, quantile_assignments, -1)

        # Compute IC within each quantile
        for q in range(quantiles):
            quantile_mask = quantile_assignments == q

            if np.sum(quantile_mask) < min_assets:
                continue

            sample_counts[t, q] = int(np.sum(quantile_mask))

            # Compute IC for each factor within this quantile
            for f in range(F):
                factor_t = values[t, :, f]
                label_t = labels[t, :]

                # Apply validity masks
                if factor_batch.validity is not None:
                    factor_valid = factor_batch.validity[t, :, f]
                    factor_t = np.where(factor_valid, factor_t, np.nan)

                if label_validity is not None:
                    label_t = np.where(label_validity[t], label_t, np.nan)

                # Filter to quantile
                factor_q = factor_t[quantile_mask]
                label_q = label_t[quantile_mask]

                # Compute correlation
                valid_obs = np.isfinite(factor_q) & np.isfinite(label_q)
                factor_valid = factor_q[valid_obs]
                label_valid = label_q[valid_obs]

                if len(factor_valid) < min_assets:
                    continue

                if np.std(factor_valid) == 0 or np.std(label_valid) == 0:
                    continue

                if method == "pearson":
                    ic = np.corrcoef(factor_valid, label_valid)[0, 1]
                else:
                    from scipy import stats
                    ic, _ = stats.spearmanr(factor_valid, label_valid)

                conditional_ic[t, f, q] = ic

    return conditional_ic, sample_counts


def compute_incremental_ic(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    base_factor_indices: Tuple[int, ...],
    test_factor_idx: int,
    method: str = "pearson",
    min_assets: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute incremental IC: IC contribution of test factor after controlling for base factors.

    Uses residualization: regress test factor on base factors, then compute IC
    of residuals with labels.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        base_factor_indices: Indices of factors to control for
        test_factor_idx: Index of factor to test
        method: "pearson" or "spearman" for IC computation
        min_assets: Minimum assets per period

    Returns:
        (incremental_ic, base_ic, total_ic)
        incremental_ic: shape (T,) - IC of residualized test factor
        base_ic: shape (T,) - IC of base factors' prediction
        total_ic: shape (T,) - IC of test factor without residualization
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    labels, label_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )

    T, N, F = values.shape

    incremental_ic = np.full(T, np.nan, dtype=np.float64)
    base_ic = np.full(T, np.nan, dtype=np.float64)
    total_ic = np.full(T, np.nan, dtype=np.float64)

    for t in range(T):
        # Extract base factors and test factor
        base_factors = values[t, :, list(base_factor_indices)]  # (K, N) with list indexing
        if base_factors.ndim == 2:
            # List indexing always gives (K, N), need (N, K)
            base_factors = base_factors.T
        else:
            # Single dimension case (shouldn't happen with list indexing, but be safe)
            base_factors = base_factors[:, np.newaxis]

        test_factor = values[t, :, test_factor_idx]  # (N,)
        label_t = labels[t, :]  # (N,)

        # Apply validity masks
        if factor_batch.validity is not None:
            base_valid = factor_batch.validity[t, :, list(base_factor_indices)]
            if base_valid.ndim == 2:
                base_valid = base_valid.T
            else:
                base_valid = base_valid[:, np.newaxis]
            base_factors = np.where(base_valid, base_factors, np.nan)
            test_valid = factor_batch.validity[t, :, test_factor_idx]
            test_factor = np.where(test_valid, test_factor, np.nan)

        if label_validity is not None:
            label_t = np.where(label_validity[t], label_t, np.nan)

        # Filter finite observations
        valid_mask = (
            np.all(np.isfinite(base_factors), axis=1) &
            np.isfinite(test_factor) &
            np.isfinite(label_t)
        )

        if np.sum(valid_mask) < min_assets:
            continue

        base_valid = base_factors[valid_mask, :]
        test_valid = test_factor[valid_mask]
        label_valid = label_t[valid_mask]

        # Compute total IC (test factor without residualization)
        if np.std(test_valid) > 0 and np.std(label_valid) > 0:
            if method == "pearson":
                total_ic[t] = np.corrcoef(test_valid, label_valid)[0, 1]
            else:
                from scipy import stats
                total_ic[t], _ = stats.spearmanr(test_valid, label_valid)

        # Residualize test factor against base factors
        try:
            # Add intercept
            X = np.column_stack([np.ones(len(base_valid)), base_valid])
            XtX = X.T @ X
            Xty_test = X.T @ test_valid
            Xty_label = X.T @ label_valid

            beta_test = np.linalg.solve(XtX, Xty_test)
            beta_label = np.linalg.solve(XtX, Xty_label)

            # Residuals
            test_residual = test_valid - X @ beta_test
            label_residual = label_valid - X @ beta_label

            # Base prediction
            base_pred = X @ beta_label

            # Compute base IC
            if np.std(base_pred) > 0 and np.std(label_valid) > 0:
                if method == "pearson":
                    base_ic[t] = np.corrcoef(base_pred, label_valid)[0, 1]
                else:
                    from scipy import stats
                    base_ic[t], _ = stats.spearmanr(base_pred, label_valid)

            # Compute incremental IC
            if np.std(test_residual) > 0 and np.std(label_residual) > 0:
                if method == "pearson":
                    incremental_ic[t] = np.corrcoef(test_residual, label_residual)[0, 1]
                else:
                    from scipy import stats
                    incremental_ic[t], _ = stats.spearmanr(test_residual, label_residual)

        except np.linalg.LinAlgError:
            continue

    return incremental_ic, base_ic, total_ic


def compute_partial_ic(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    factor_idx: int,
    control_indices: Tuple[int, ...],
    method: str = "pearson",
    min_assets: int = 30,
) -> np.ndarray:
    """
    Compute partial IC: correlation between factor and label after removing linear effects of control factors.

    Residualizes both the factor and label against control factors, then computes IC.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        factor_idx: Index of factor to test
        control_indices: Indices of factors to control for
        method: "pearson" or "spearman"
        min_assets: Minimum assets per period

    Returns:
        partial_ic: shape (T,) - partial IC per period
    """
    incremental_ic, _, _ = compute_incremental_ic(
        factor_batch,
        label_bundle,
        base_factor_indices=control_indices,
        test_factor_idx=factor_idx,
        method=method,
        min_assets=min_assets,
    )

    return incremental_ic
