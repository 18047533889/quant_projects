"""
Factor complementarity detection.

Identifies factor pairs that work better together than individually, measuring
synergistic relationships and interaction strength.
"""

from typing import List, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.interactions.substitution import _require_matching_asset_coordinates
from quant_evaluator.metrics.interactions.substitution import compute_substitution_effect


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

    ic_a, ic_b, ic_combined = compute_substitution_effect(
        factor_batch, label_bundle, factor_a_idx, factor_b_idx,
        method=method, min_assets=min_assets,
    )

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
    candidate_pairs=None,
    max_pairs=None,
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

    pairs = list(candidate_pairs) if candidate_pairs is not None else [
        (i, j) for i in range(F) for j in range(i + 1, F)
    ]
    if max_pairs is not None and (max_pairs < 0 or len(pairs) > max_pairs):
        raise ValueError("candidate pair budget exceeded")
    for i, j in pairs:
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
    _require_matching_asset_coordinates(factor_batch, label_bundle)
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")
    if method == "spearman":
        return _compute_interaction_strength_reference(
            factor_batch, label_bundle, factor_a_idx, factor_b_idx,
            method=method, min_assets=min_assets,
        )

    values = factor_batch.values  # (T, N, F)
    labels = label_bundle.values

    if labels.ndim == 1:
        labels = np.broadcast_to(labels[:, np.newaxis], (factor_batch.num_times, factor_batch.num_assets))

    T, N, F = values.shape

    interaction_ic = np.full(T, np.nan, dtype=np.float64)
    additive_ic = np.full(T, np.nan, dtype=np.float64)

    # Hoisted validity-masked panels (identical values to the legacy per-t
    # np.where applications).
    fa = values[:, :, factor_a_idx]
    fb = values[:, :, factor_b_idx]
    if factor_batch.validity is not None:
        fa = np.where(factor_batch.validity[:, :, factor_a_idx], fa, np.nan)
        fb = np.where(factor_batch.validity[:, :, factor_b_idx], fb, np.nan)
    lab = labels
    if label_bundle.validity is not None:
        lab = np.where(label_bundle.validity, lab, np.nan)

    valid = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(lab)  # (T, N)
    n = valid.sum(axis=1).astype(np.float64)

    # Per-column centering shifts keep the raw-moment correlation accurate;
    # suspicious (near-constant) cells fall back to the verbatim legacy path.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        mu_a = np.nanmean(fa, axis=1)
        mu_b = np.nanmean(fb, axis=1)
        mu_l = np.nanmean(lab, axis=1)
        xac = np.where(np.isfinite(fa), fa - mu_a[:, None], 0.0)
        xbc = np.where(np.isfinite(fb), fb - mu_b[:, None], 0.0)
        labc = np.where(np.isfinite(lab), lab - mu_l[:, None], 0.0)
        sprime_a = np.max(np.abs(xac), axis=1)
        sprime_b = np.max(np.abs(xbc), axis=1)
        sprime_l = np.max(np.abs(labc), axis=1)
    thr_a = 1e4 * np.finfo(float).eps * np.maximum(1.0, sprime_a) ** 2
    thr_b = 1e4 * np.finfo(float).eps * np.maximum(1.0, sprime_b) ** 2
    thr_l = 1e4 * np.finfo(float).eps * np.maximum(1.0, sprime_l) ** 2

    xz = np.where(valid, xac, 0.0)
    yz = np.where(valid, xbc, 0.0)
    lz = np.where(valid, labc, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        # Standardize factors: (x - mean) / (std + 1e-8), legacy formula.
        mean_a_c = xz.sum(axis=1) / np.maximum(n, 1.0)
        mean_b_c = yz.sum(axis=1) / np.maximum(n, 1.0)
        var_a = (xz * xz).sum(axis=1) / np.maximum(n, 1.0) - mean_a_c ** 2
        var_b = (yz * yz).sum(axis=1) / np.maximum(n, 1.0) - mean_b_c ** 2
        std_a = np.sqrt(np.maximum(var_a, 0.0))
        std_b = np.sqrt(np.maximum(var_b, 0.0))
        fa_std = (xz - mean_a_c[:, None]) / (std_a + 1e-8)[:, None]
        fb_std = (yz - mean_b_c[:, None]) / (std_b + 1e-8)[:, None]
        inter = np.where(valid, fa_std * fb_std, 0.0)
        add = np.where(valid, (fa_std + fb_std) / 2.0, 0.0)
        si = inter.sum(axis=1); sii = (inter * inter).sum(axis=1); sil = (inter * lz).sum(axis=1)
        sa = add.sum(axis=1); saa = (add * add).sum(axis=1); sal = (add * lz).sum(axis=1)
        sl = lz.sum(axis=1); sll = (lz * lz).sum(axis=1)
        corr_i = np.where((n * sii - si * si) * (n * sll - sl * sl) > 0,
                          (n * sil - si * sl) / np.sqrt((n * sii - si * si) * (n * sll - sl * sl)),
                          np.nan)
        corr_a2 = np.where((n * saa - sa * sa) * (n * sll - sl * sl) > 0,
                           (n * sal - sa * sl) / np.sqrt((n * saa - sa * sa) * (n * sll - sl * sl)),
                           np.nan)
        var_i = sii / np.maximum(n, 1.0) - (si / np.maximum(n, 1.0)) ** 2
        var_add = saa / np.maximum(n, 1.0) - (sa / np.maximum(n, 1.0)) ** 2
        var_l = sll / np.maximum(n, 1.0) - (sl / np.maximum(n, 1.0)) ** 2

    suspicious = (
        (n < min_assets)
        | ~(var_i > thr_a) | ~(var_add > thr_a) | ~(var_a > thr_a) | ~(var_b > thr_b)
        | ~(var_l > thr_l)
        | ~np.isfinite(corr_i) | ~np.isfinite(corr_a2)
    )
    fast = ~suspicious
    interaction_ic[fast] = corr_i[fast]
    additive_ic[fast] = corr_a2[fast]

    for t in np.flatnonzero(suspicious):
        factor_a = fa[t]
        factor_b = fb[t]
        label_t = lab[t]

        # Filter valid observations
        v = np.isfinite(factor_a) & np.isfinite(factor_b) & np.isfinite(label_t)

        if np.sum(v) < min_assets:
            continue

        fa_v = factor_a[v]
        fb_v = factor_b[v]
        label_v = label_t[v]

        # Standardize factors
        fa_std_v = (fa_v - np.mean(fa_v)) / (np.std(fa_v) + 1e-8)
        fb_std_v = (fb_v - np.mean(fb_v)) / (np.std(fb_v) + 1e-8)

        # Compute interaction term
        interaction = fa_std_v * fb_std_v

        # Compute additive term
        additive = (fa_std_v + fb_std_v) / 2.0

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


def _compute_interaction_strength_reference(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    factor_a_idx: int,
    factor_b_idx: int,
    method: str = "pearson",
    min_assets: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Verbatim legacy oracle for equivalence testing."""
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
