"""
Conditional IC analysis.

Measures factor performance conditional on another factor's value or after
controlling for other factors.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.label_panel import normalize_label_panel
from quant_evaluator.metrics.exposure import rank_aware_projection


SAME_DATE_DESCRIPTIVE = "SAME_DATE_DESCRIPTIVE"
SPEARMAN_DEFINITION = "raw_value_residuals_then_spearman_correlation"


@dataclass(frozen=True)
class IncrementalICDiagnostics:
    estimation_scope: str
    method_definition: str
    test_projection: Tuple[object, ...]
    label_projection: Tuple[object, ...]


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
        sample_counts: shape (T, F, Q) - actual joint evaluation count
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    labels, label_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )

    T, N, F = values.shape

    conditional_ic = np.full((T, F, quantiles), np.nan, dtype=np.float64)
    sample_counts = np.zeros((T, F, quantiles), dtype=np.int32)

    for t in range(T):
        conditioning_values = values[t, :, conditioning_factor_idx]

        # Apply validity mask
        if factor_batch.validity is not None:
            conditioning_valid = factor_batch.validity[t, :, conditioning_factor_idx]
            conditioning_values = np.where(conditioning_valid, conditioning_values, np.nan)

        # Group membership is decision-time information and must not depend on labels.
        valid_mask = np.isfinite(conditioning_values)
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
                sample_counts[t, f, q] = int(np.sum(valid_obs))
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
    *,
    return_diagnostics: bool = False,
):
    """
    Compute same-date descriptive partial association after controlling for bases.

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
        base_ic: shape (T,) - descriptive in-sample fitted association; not OOS
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
    test_diagnostics = [None] * T
    label_diagnostics = [None] * T

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

        # Authority is rank-aware; duplicate controls span the same subspace.
        try:
            test_fitted, test_residual, test_diag = rank_aware_projection(
                base_valid, test_valid, add_intercept=True,
            )
            base_pred, label_residual, label_diag = rank_aware_projection(
                base_valid, label_valid, add_intercept=True,
            )
            test_diagnostics[t] = test_diag
            label_diagnostics[t] = label_diag

            # Compute base IC
            if label_diag["status"] != "INSUFFICIENT_DF" and np.std(base_pred) > 0 and np.std(label_valid) > 0:
                if method == "pearson":
                    base_ic[t] = np.corrcoef(base_pred, label_valid)[0, 1]
                else:
                    from scipy import stats
                    base_ic[t], _ = stats.spearmanr(base_pred, label_valid)

            # Compute incremental IC
            if (test_diag["status"] != "NO_RESIDUAL_VARIANCE" and
                    label_diag["status"] != "NO_RESIDUAL_VARIANCE" and
                    test_diag["effective_df"] >= 2 and label_diag["effective_df"] >= 2):
                if method == "pearson":
                    incremental_ic[t] = np.corrcoef(test_residual, label_residual)[0, 1]
                else:
                    from scipy import stats
                    incremental_ic[t], _ = stats.spearmanr(test_residual, label_residual)

        except (np.linalg.LinAlgError, ValueError):
            continue

    if return_diagnostics:
        return incremental_ic, base_ic, total_ic, IncrementalICDiagnostics(
            SAME_DATE_DESCRIPTIVE,
            SPEARMAN_DEFINITION if method == "spearman" else "raw_value_residuals_then_pearson_correlation",
            tuple(test_diagnostics), tuple(label_diagnostics),
        )
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
