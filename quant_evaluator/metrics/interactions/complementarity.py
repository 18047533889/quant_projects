"""
Factor complementarity detection.

Identifies factor pairs that work better together than individually, measuring
synergistic relationships and interaction strength.
"""

from typing import List, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def compute_complementarity_score(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    factor_a_idx: int,
    factor_b_idx: int,
    method: str = "pearson",
    min_assets: int = 30,
    min_periods: int = 20,
) -> Tuple[float, float, float]:
    """
    Compute complementarity score for a factor pair.

    Complementarity exists when combined IC > IC_A + IC_B (superadditive).

    Score = mean(IC_combined - max(IC_A, IC_B)) / std(max(IC_A, IC_B))

    Positive score indicates complementarity, negative indicates substitution.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        factor_a_idx: Index of first factor
        factor_b_idx: Index of second factor
        method: "pearson" or "spearman"
        min_assets: Minimum assets per period
        min_periods: Minimum periods required

    Returns:
        (complementarity_score, mean_lift, std_lift)
        complementarity_score: Standardized lift score
        mean_lift: Average IC lift from combination
        std_lift: Std dev of IC lift
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

        # Compute IC for each factor individually
        valid_a = np.isfinite(factor_a) & np.isfinite(label_t)
        if np.sum(valid_a) >= min_assets:
            fa = factor_a[valid_a]
            la = label_t[valid_a]
            if np.std(fa) > 0 and np.std(la) > 0:
                if method == "pearson":
                    ic_a[t] = np.corrcoef(fa, la)[0, 1]
                else:
                    from scipy import stats
                    ic_a[t], _ = stats.spearmanr(fa, la)

        valid_b = np.isfinite(factor_b) & np.isfinite(label_t)
        if np.sum(valid_b) >= min_assets:
            fb = factor_b[valid_b]
            lb = label_t[valid_b]
            if np.std(fb) > 0 and np.std(lb) > 0:
                if method == "pearson":
                    ic_b[t] = np.corrcoef(fb, lb)[0, 1]
                else:
                    from scipy import stats
                    ic_b[t], _ = stats.spearmanr(fb, lb)

        # Compute combined IC (equal-weighted)
        valid_both = np.isfinite(factor_a) & np.isfinite(factor_b) & np.isfinite(label_t)
        if np.sum(valid_both) >= min_assets:
            fa_both = factor_a[valid_both]
            fb_both = factor_b[valid_both]
            l_both = label_t[valid_both]

            # Standardize and combine
            fa_std = (fa_both - np.mean(fa_both)) / (np.std(fa_both) + 1e-8)
            fb_std = (fb_both - np.mean(fb_both)) / (np.std(fb_both) + 1e-8)
            combined = (fa_std + fb_std) / 2.0

            if np.std(combined) > 0 and np.std(l_both) > 0:
                if method == "pearson":
                    ic_combined[t] = np.corrcoef(combined, l_both)[0, 1]
                else:
                    from scipy import stats
                    ic_combined[t], _ = stats.spearmanr(combined, l_both)

    # Compute complementarity metrics
    valid_mask = np.isfinite(ic_a) & np.isfinite(ic_b) & np.isfinite(ic_combined)

    if np.sum(valid_mask) < min_periods:
        return np.nan, np.nan, np.nan

    ic_a_valid = ic_a[valid_mask]
    ic_b_valid = ic_b[valid_mask]
    ic_combined_valid = ic_combined[valid_mask]

    # Lift: how much better is combined vs best individual
    max_individual = np.maximum(np.abs(ic_a_valid), np.abs(ic_b_valid))
    lift = np.abs(ic_combined_valid) - max_individual

    mean_lift = np.mean(lift)
    std_lift = np.std(lift, ddof=1)

    # Standardized score
    if std_lift > 0:
        complementarity_score = mean_lift / std_lift
    else:
        complementarity_score = 0.0

    return float(complementarity_score), float(mean_lift), float(std_lift)


def detect_complementary_pairs(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    threshold: float = 1.0,
    method: str = "pearson",
    min_assets: int = 30,
    min_periods: int = 20,
) -> List[Tuple[int, int, float, float]]:
    """
    Detect pairs of factors that exhibit complementarity.

    Returns factor pairs where combined performance exceeds individual performance.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        threshold: Minimum complementarity score (standardized)
        method: "pearson" or "spearman"
        min_assets: Minimum assets per period
        min_periods: Minimum periods required

    Returns:
        List of (factor_a_idx, factor_b_idx, complementarity_score, mean_lift)
        Sorted by complementarity score (descending)
    """
    F = factor_batch.num_factors
    complementary_pairs = []

    for i in range(F):
        for j in range(i + 1, F):
            comp_score, mean_lift, _ = compute_complementarity_score(
                factor_batch, label_bundle, i, j,
                method=method, min_assets=min_assets, min_periods=min_periods
            )

            if np.isfinite(comp_score) and comp_score >= threshold:
                complementary_pairs.append((i, j, float(comp_score), float(mean_lift)))

    # Sort by complementarity score (descending)
    complementary_pairs.sort(key=lambda x: x[2], reverse=True)

    return complementary_pairs


def compute_interaction_strength(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    factor_a_idx: int,
    factor_b_idx: int,
    method: str = "pearson",
    min_assets: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time-varying interaction strength between two factors.

    Interaction strength measures whether the factors work together
    differently in different market regimes.

    Uses a simple interaction term: (A - mean_A) * (B - mean_B)

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        factor_a_idx: Index of first factor
        factor_b_idx: Index of second factor
        method: "pearson" or "spearman"
        min_assets: Minimum assets per period

    Returns:
        (interaction_ic, additive_ic)
        interaction_ic: shape (T,) - IC of interaction term A*B
        additive_ic: shape (T,) - IC of additive model A+B
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    labels = label_bundle.values

    if labels.ndim == 1:
        labels = np.broadcast_to(labels[:, np.newaxis], (factor_batch.num_times, factor_batch.num_assets))

    T, N, F = values.shape

    interaction_ic = np.full(T, np.nan, dtype=np.float64)
    additive_ic = np.full(T, np.nan, dtype=np.float64)

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

        # Filter valid observations
        valid = np.isfinite(factor_a) & np.isfinite(factor_b) & np.isfinite(label_t)

        if np.sum(valid) < min_assets:
            continue

        fa = factor_a[valid]
        fb = factor_b[valid]
        label_v = label_t[valid]

        # Standardize factors
        fa_std = (fa - np.mean(fa)) / (np.std(fa) + 1e-8)
        fb_std = (fb - np.mean(fb)) / (np.std(fb) + 1e-8)

        # Compute interaction term
        interaction = fa_std * fb_std

        # Compute additive term
        additive = (fa_std + fb_std) / 2.0

        # IC of interaction term
        if np.std(interaction) > 0 and np.std(label_v) > 0:
            if method == "pearson":
                interaction_ic[t] = np.corrcoef(interaction, label_v)[0, 1]
            else:
                from scipy import stats
                interaction_ic[t], _ = stats.spearmanr(interaction, label_v)

        # IC of additive term
        if np.std(additive) > 0 and np.std(label_v) > 0:
            if method == "pearson":
                additive_ic[t] = np.corrcoef(additive, label_v)[0, 1]
            else:
                from scipy import stats
                additive_ic[t], _ = stats.spearmanr(additive, label_v)

    return interaction_ic, additive_ic
