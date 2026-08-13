"""
Factor substitution analysis.

Detects when factors can substitute for each other and measures the marginal
contribution of each factor in a portfolio context.
"""

from typing import List, Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def compute_substitution_effect(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    factor_a_idx: int,
    factor_b_idx: int,
    method: str = "pearson",
    min_assets: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure substitution effect between two factors.

    Computes:
    1. IC of factor A alone
    2. IC of factor B alone
    3. IC of combined (A, B) factor portfolio

    Strong substitution: IC_combined ≈ max(IC_A, IC_B)
    Complementarity: IC_combined > IC_A + IC_B

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        factor_a_idx: Index of first factor
        factor_b_idx: Index of second factor
        method: "pearson" or "spearman"
        min_assets: Minimum assets per period

    Returns:
        (ic_a, ic_b, ic_combined)
        ic_a: shape (T,) - IC of factor A alone
        ic_b: shape (T,) - IC of factor B alone
        ic_combined: shape (T,) - IC of equal-weighted (A+B)/2
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    labels = label_bundle.values

    if labels.ndim == 1:
        labels = np.broadcast_to(labels[:, np.newaxis], (factor_batch.num_times, factor_batch.num_assets))

    T, N, F = values.shape

    ic_a = np.full(T, np.nan, dtype=np.float64)
    ic_b = np.full(T, np.nan, dtype=np.float64)
    ic_combined = np.full(T, np.nan, dtype=np.float64)

    for t in range(T):
        factor_a = values[t, :, factor_a_idx]
        factor_b = values[t, :, factor_b_idx]
        label_t = labels[t, :]

        # Apply validity masks
        if factor_batch.validity is not None:
            factor_a = np.where(factor_batch.validity[t, :, factor_a_idx], factor_a, np.nan)
            factor_b = np.where(factor_batch.validity[t, :, factor_b_idx], factor_b, np.nan)

        if label_bundle.validity is not None:
            label_t = np.where(label_bundle.validity[t, :], label_t, np.nan)

        # Filter finite observations for each pair
        valid_a = np.isfinite(factor_a) & np.isfinite(label_t)
        valid_b = np.isfinite(factor_b) & np.isfinite(label_t)
        valid_both = np.isfinite(factor_a) & np.isfinite(factor_b) & np.isfinite(label_t)

        # IC of factor A
        if np.sum(valid_a) >= min_assets:
            fa = factor_a[valid_a]
            la = label_t[valid_a]
            if np.std(fa) > 0 and np.std(la) > 0:
                if method == "pearson":
                    ic_a[t] = np.corrcoef(fa, la)[0, 1]
                else:
                    from scipy import stats
                    ic_a[t], _ = stats.spearmanr(fa, la)

        # IC of factor B
        if np.sum(valid_b) >= min_assets:
            fb = factor_b[valid_b]
            lb = label_t[valid_b]
            if np.std(fb) > 0 and np.std(lb) > 0:
                if method == "pearson":
                    ic_b[t] = np.corrcoef(fb, lb)[0, 1]
                else:
                    from scipy import stats
                    ic_b[t], _ = stats.spearmanr(fb, lb)

        # IC of combined factor (equal-weighted average)
        if np.sum(valid_both) >= min_assets:
            fa_both = factor_a[valid_both]
            fb_both = factor_b[valid_both]
            l_both = label_t[valid_both]

            # Standardize factors before combining
            fa_std = (fa_both - np.mean(fa_both)) / (np.std(fa_both) + 1e-8)
            fb_std = (fb_both - np.mean(fb_both)) / (np.std(fb_both) + 1e-8)
            combined = (fa_std + fb_std) / 2.0

            if np.std(combined) > 0 and np.std(l_both) > 0:
                if method == "pearson":
                    ic_combined[t] = np.corrcoef(combined, l_both)[0, 1]
                else:
                    from scipy import stats
                    ic_combined[t], _ = stats.spearmanr(combined, l_both)

    return ic_a, ic_b, ic_combined


def detect_substitutable_factors(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    threshold: float = 0.9,
    method: str = "pearson",
    min_assets: int = 30,
    min_periods: int = 20,
) -> List[Tuple[int, int, float]]:
    """
    Detect pairs of factors that are substitutable.

    A pair (A, B) is substitutable if:
    - They are highly correlated
    - IC_combined ≈ max(IC_A, IC_B) (no incremental benefit)

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        threshold: Substitution threshold (ratio of combined IC to max individual IC)
        method: "pearson" or "spearman"
        min_assets: Minimum assets per period
        min_periods: Minimum periods with valid IC

    Returns:
        List of (factor_a_idx, factor_b_idx, substitution_score)
        substitution_score: mean(IC_combined / max(IC_A, IC_B))
    """
    F = factor_batch.num_factors
    substitutable_pairs = []

    for i in range(F):
        for j in range(i + 1, F):
            ic_a, ic_b, ic_combined = compute_substitution_effect(
                factor_batch, label_bundle, i, j, method=method, min_assets=min_assets
            )

            # Compute substitution score
            valid_mask = np.isfinite(ic_a) & np.isfinite(ic_b) & np.isfinite(ic_combined)

            if np.sum(valid_mask) < min_periods:
                continue

            ic_a_valid = ic_a[valid_mask]
            ic_b_valid = ic_b[valid_mask]
            ic_combined_valid = ic_combined[valid_mask]

            # Substitution score: how close is combined IC to max individual IC
            max_individual = np.maximum(np.abs(ic_a_valid), np.abs(ic_b_valid))
            substitution_ratios = np.abs(ic_combined_valid) / (max_individual + 1e-8)

            mean_substitution = np.mean(substitution_ratios)

            if mean_substitution >= threshold:
                substitutable_pairs.append((i, j, float(mean_substitution)))

    # Sort by substitution score (descending)
    substitutable_pairs.sort(key=lambda x: x[2], reverse=True)

    return substitutable_pairs


def compute_marginal_contribution(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    target_factor_idx: int,
    baseline_indices: Optional[Tuple[int, ...]] = None,
    method: str = "pearson",
    min_assets: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal IC contribution of a factor given a baseline portfolio.

    Measures incremental IC when adding target factor to baseline portfolio.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        target_factor_idx: Index of factor to evaluate
        baseline_indices: Indices of baseline factors (if None, use all others)
        method: "pearson" or "spearman"
        min_assets: Minimum assets per period

    Returns:
        (marginal_ic, baseline_ic)
        marginal_ic: shape (T,) - incremental IC from adding target
        baseline_ic: shape (T,) - IC of baseline portfolio alone
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    labels = label_bundle.values

    if labels.ndim == 1:
        labels = np.broadcast_to(labels[:, np.newaxis], (factor_batch.num_times, factor_batch.num_assets))

    T, N, F = values.shape

    # Default: use all factors except target as baseline
    if baseline_indices is None:
        baseline_indices = tuple(i for i in range(F) if i != target_factor_idx)

    if len(baseline_indices) == 0:
        # No baseline, marginal IC = total IC
        marginal_ic = np.full(T, np.nan, dtype=np.float64)
        baseline_ic = np.zeros(T, dtype=np.float64)

        for t in range(T):
            target_factor = values[t, :, target_factor_idx]
            label_t = labels[t, :]

            if factor_batch.validity is not None:
                target_factor = np.where(factor_batch.validity[t, :, target_factor_idx], target_factor, np.nan)

            if label_bundle.validity is not None:
                label_t = np.where(label_bundle.validity[t, :], label_t, np.nan)

            valid = np.isfinite(target_factor) & np.isfinite(label_t)

            if np.sum(valid) < min_assets:
                continue

            tf = target_factor[valid]
            lt = label_t[valid]

            if np.std(tf) > 0 and np.std(lt) > 0:
                if method == "pearson":
                    marginal_ic[t] = np.corrcoef(tf, lt)[0, 1]
                else:
                    from scipy import stats
                    marginal_ic[t], _ = stats.spearmanr(tf, lt)

        return marginal_ic, baseline_ic

    # Compute baseline portfolio IC
    baseline_ic = np.full(T, np.nan, dtype=np.float64)
    combined_ic = np.full(T, np.nan, dtype=np.float64)

    for t in range(T):
        baseline_factors = values[t, :, list(baseline_indices)]  # (K, N) with list indexing
        if baseline_factors.ndim == 2:
            # List indexing always gives (K, N), need (N, K)
            baseline_factors = baseline_factors.T
        else:
            # Single dimension case (shouldn't happen with list indexing, but be safe)
            baseline_factors = baseline_factors[:, np.newaxis]

        target_factor = values[t, :, target_factor_idx]  # (N,)
        label_t = labels[t, :]  # (N,)

        # Apply validity masks
        if factor_batch.validity is not None:
            baseline_valid = factor_batch.validity[t, :, list(baseline_indices)]
            if baseline_valid.ndim == 2:
                baseline_valid = baseline_valid.T
            else:
                baseline_valid = baseline_valid[:, np.newaxis]
            baseline_factors = np.where(baseline_valid, baseline_factors, np.nan)
            target_factor = np.where(factor_batch.validity[t, :, target_factor_idx], target_factor, np.nan)

        if label_bundle.validity is not None:
            label_t = np.where(label_bundle.validity[t, :], label_t, np.nan)

        # Baseline IC: equal-weighted baseline factors
        valid_baseline = np.all(np.isfinite(baseline_factors), axis=1) & np.isfinite(label_t)

        if np.sum(valid_baseline) >= min_assets:
            bf = baseline_factors[valid_baseline]
            lb = label_t[valid_baseline]

            # Standardize and average
            bf_std = (bf - np.mean(bf, axis=0, keepdims=True)) / (np.std(bf, axis=0, keepdims=True) + 1e-8)
            baseline_composite = np.mean(bf_std, axis=1)

            if np.std(baseline_composite) > 0 and np.std(lb) > 0:
                if method == "pearson":
                    baseline_ic[t] = np.corrcoef(baseline_composite, lb)[0, 1]
                else:
                    from scipy import stats
                    baseline_ic[t], _ = stats.spearmanr(baseline_composite, lb)

        # Combined IC: baseline + target
        valid_combined = (
            np.all(np.isfinite(baseline_factors), axis=1) &
            np.isfinite(target_factor) &
            np.isfinite(label_t)
        )

        if np.sum(valid_combined) >= min_assets:
            bf_comb = baseline_factors[valid_combined]
            tf_comb = target_factor[valid_combined]
            lc = label_t[valid_combined]

            # Standardize all factors
            bf_std = (bf_comb - np.mean(bf_comb, axis=0, keepdims=True)) / (np.std(bf_comb, axis=0, keepdims=True) + 1e-8)
            tf_std = (tf_comb - np.mean(tf_comb)) / (np.std(tf_comb) + 1e-8)

            # Equal-weighted combination
            all_factors = np.column_stack([bf_std, tf_std[:, np.newaxis]])
            combined_composite = np.mean(all_factors, axis=1)

            if np.std(combined_composite) > 0 and np.std(lc) > 0:
                if method == "pearson":
                    combined_ic[t] = np.corrcoef(combined_composite, lc)[0, 1]
                else:
                    from scipy import stats
                    combined_ic[t], _ = stats.spearmanr(combined_composite, lc)

    # Marginal contribution: difference between combined and baseline
    marginal_ic = combined_ic - baseline_ic

    return marginal_ic, baseline_ic
