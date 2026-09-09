"""
Factor substitution analysis.

Detects when factors can substitute for each other and measures the marginal
contribution of each factor in a portfolio context.
"""

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Iterable, List, Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.interactions.pairwise import iter_bounded_pairs


class FactorRelation(str, Enum):
    NEAR_DUPLICATE = "near_duplicate"
    SUBSTITUTABLE = "substitutable"
    COMPLEMENTARY = "complementary"
    TRADEOFF = "tradeoff"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FactorRelationEvidence:
    factor_id_a: str
    factor_id_b: str
    factor_a_idx: int
    factor_b_idx: int
    relation: FactorRelation
    signed_similarity: float
    combined_lift: float
    n_periods: int
    common_support_ref: str
    method_version: str = "paired_relation.v2"
    substitution_score: Optional[float] = None
    orientation: str = "frozen_as_supplied"
    sample_scope: str = "paired_common_support"


@dataclass(frozen=True)
class MarginalContributionEvidence:
    factor_id: str
    baseline_factor_ids: Tuple[str, ...]
    common_support_baseline_ic: np.ndarray
    common_support_combined_ic: np.ndarray
    native_baseline_ic: np.ndarray
    native_target_ic: np.ndarray
    common_support_counts: np.ndarray
    native_baseline_counts: np.ndarray
    native_target_counts: np.ndarray
    contribution_scope: str = "continuous_value_on_common_support"
    method_version: str = "paired_marginal.v2"

    def __post_init__(self):
        for name in ("common_support_baseline_ic", "common_support_combined_ic",
                     "native_baseline_ic", "native_target_ic", "common_support_counts",
                     "native_baseline_counts", "native_target_counts"):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return np.nan
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    from scipy import stats
    return float(stats.spearmanr(x, y)[0])


def _standardize(x: np.ndarray) -> Optional[np.ndarray]:
    scale = float(np.std(x))
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(x)))) * 16
    if not np.isfinite(scale) or scale <= tolerance:
        return None
    return (x - np.mean(x)) / scale


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

    This primitive returns three paired utilities on identical support. It does
    not classify relations by adding correlations: classification additionally
    requires signed factor similarity, frozen orientation, and a scale-aware
    utility-equivalence test in :func:`detect_substitutable_factors`.

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

        # All three utilities are deliberately measured on identical support.
        valid_both = np.isfinite(factor_a) & np.isfinite(factor_b) & np.isfinite(label_t)
        if np.sum(valid_both) >= min_assets:
            fa_both = factor_a[valid_both]
            fb_both = factor_b[valid_both]
            l_both = label_t[valid_both]

            # Standardize factors before combining
            ic_a[t] = _corr(fa_both, l_both, method)
            ic_b[t] = _corr(fb_both, l_both, method)
            fa_std = _standardize(fa_both)
            fb_std = _standardize(fb_both)
            if fa_std is None or fb_std is None:
                continue
            combined = (fa_std + fb_std) / 2.0
            ic_combined[t] = _corr(combined, l_both, method)

    return ic_a, ic_b, ic_combined


def detect_substitutable_factors(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    threshold: float = 0.9,
    method: str = "pearson",
    min_assets: int = 30,
    min_periods: int = 20,
    candidate_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    max_pairs: int = 10_000,
    similarity_threshold: float = 0.9,
    equivalence_tolerance: Optional[float] = None,
    return_evidence: bool = False,
):
    """
    Detect typed pair relations on common support with frozen orientation.

    A pair (A, B) is substitutable if:
    - They are highly correlated
    - IC_combined ≈ max(IC_A, IC_B) (no incremental benefit)

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N)
        threshold: Backward-compatible equivalence threshold (0.9 -> 10% tolerance)
        method: "pearson" or "spearman"
        min_assets: Minimum assets per period
        min_periods: Minimum periods with valid IC

    Returns:
        List of (factor_a_idx, factor_b_idx, substitution_score)
        substitution_score: similarity discounted by paired absolute utility lift
    """
    F = factor_batch.num_factors
    relation_evidence = []

    tolerance = (1.0 - threshold) if equivalence_tolerance is None else equivalence_tolerance
    for i, j in iter_bounded_pairs(F, candidate_pairs, max_pairs):
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

            # Orientation is frozen: negative correlation is not silently flipped.
            values = np.asarray(factor_batch.values, float)
            labels = np.asarray(label_bundle.values, float)
            if labels.ndim == 1:
                labels = np.broadcast_to(labels[:, None], values.shape[:2])
            masks = []
            daily_similarity = []
            for t in np.flatnonzero(valid_mask):
                mask = (np.isfinite(values[t, :, i]) & np.isfinite(values[t, :, j]) &
                        np.isfinite(labels[t]))
                if factor_batch.validity is not None:
                    mask &= factor_batch.validity[t, :, i] & factor_batch.validity[t, :, j]
                if label_bundle.validity is not None:
                    mask &= label_bundle.validity[t]
                masks.append(np.asarray(mask, dtype=np.uint8))
                if int(mask.sum()) >= min_assets:
                    daily_similarity.append(_corr(values[t, mask, i], values[t, mask, j], method))
            finite_similarity = np.asarray(daily_similarity, float)
            finite_similarity = finite_similarity[np.isfinite(finite_similarity)]
            similarity = float(np.mean(finite_similarity)) if finite_similarity.size else np.nan
            support_ref = "common-support:" + sha256(
                b"".join(mask.tobytes() for mask in masks)
            ).hexdigest()
            best = np.maximum(ic_a_valid, ic_b_valid)
            lift = float(np.mean(ic_combined_valid - best))
            scale = float(np.mean(np.maximum(np.abs(ic_a_valid), np.abs(ic_b_valid))))
            weak_tol = np.sqrt(np.finfo(float).eps)
            if not np.isfinite(similarity) or scale <= weak_tol:
                relation = FactorRelation.UNKNOWN
            elif similarity >= similarity_threshold and abs(lift) <= tolerance * scale:
                relation = FactorRelation.NEAR_DUPLICATE if similarity >= 0.999 else FactorRelation.SUBSTITUTABLE
            elif lift > tolerance * scale:
                relation = FactorRelation.COMPLEMENTARY
            else:
                relation = FactorRelation.TRADEOFF
            relation_evidence.append(FactorRelationEvidence(
                factor_batch.factor_ids[i], factor_batch.factor_ids[j], i, j, relation,
                float(similarity), lift, int(np.sum(valid_mask)), support_ref,
                substitution_score=(float(similarity * max(0.0, 1.0 - abs(lift) / scale))
                                    if relation in (FactorRelation.NEAR_DUPLICATE,
                                                    FactorRelation.SUBSTITUTABLE) else None),
            ))

    # Sort by substitution score (descending)
    if return_evidence:
        return tuple(relation_evidence)
    substitutable_pairs = [
        (e.factor_a_idx, e.factor_b_idx, float(e.substitution_score))
        for e in relation_evidence
        if e.relation in (FactorRelation.NEAR_DUPLICATE, FactorRelation.SUBSTITUTABLE)
    ]
    return sorted(substitutable_pairs, key=lambda x: x[2], reverse=True)


def compute_marginal_contribution(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    target_factor_idx: int,
    baseline_indices: Optional[Tuple[int, ...]] = None,
    method: str = "pearson",
    min_assets: int = 30,
    *,
    return_diagnostics: bool = False,
):
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

        if return_diagnostics:
            counts = np.asarray([np.isfinite(marginal_ic[t]) * np.sum(
                np.isfinite(values[t, :, target_factor_idx]) & np.isfinite(labels[t])
            ) for t in range(T)], dtype=np.int32)
            evidence = MarginalContributionEvidence(
                factor_batch.factor_ids[target_factor_idx], (), baseline_ic,
                marginal_ic, baseline_ic, marginal_ic, counts, np.zeros(T, np.int32), counts,
            )
            return marginal_ic, baseline_ic, evidence
        return marginal_ic, baseline_ic

    # Compute baseline portfolio IC
    baseline_ic = np.full(T, np.nan, dtype=np.float64)
    combined_ic = np.full(T, np.nan, dtype=np.float64)
    native_baseline_ic = np.full(T, np.nan, dtype=np.float64)
    native_target_ic = np.full(T, np.nan, dtype=np.float64)
    common_counts = np.zeros(T, dtype=np.int32)
    native_baseline_counts = np.zeros(T, dtype=np.int32)
    native_target_counts = np.zeros(T, dtype=np.int32)

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

        # Paired comparison: baseline and combined utilities use common support.
        native_baseline_mask = np.all(np.isfinite(baseline_factors), axis=1) & np.isfinite(label_t)
        native_target_mask = np.isfinite(target_factor) & np.isfinite(label_t)
        native_baseline_counts[t] = int(native_baseline_mask.sum())
        native_target_counts[t] = int(native_target_mask.sum())
        if native_baseline_counts[t] >= min_assets:
            bf_native = baseline_factors[native_baseline_mask]
            scales = np.std(bf_native, axis=0, keepdims=True)
            standardized = np.divide(
                bf_native - np.mean(bf_native, axis=0, keepdims=True), scales,
                out=np.zeros_like(bf_native), where=scales > np.finfo(float).eps,
            )
            native_baseline_ic[t] = _corr(
                np.mean(standardized, axis=1), label_t[native_baseline_mask], method,
            )
        if native_target_counts[t] >= min_assets:
            native_target_ic[t] = _corr(
                target_factor[native_target_mask], label_t[native_target_mask], method,
            )
        valid_baseline = (np.all(np.isfinite(baseline_factors), axis=1) &
                          np.isfinite(target_factor) & np.isfinite(label_t))
        common_counts[t] = int(valid_baseline.sum())

        if np.sum(valid_baseline) >= min_assets:
            bf = baseline_factors[valid_baseline]
            lb = label_t[valid_baseline]

            # Standardize and average
            scales = np.std(bf, axis=0, keepdims=True)
            bf_std = np.divide(bf - np.mean(bf, axis=0, keepdims=True), scales,
                               out=np.zeros_like(bf), where=scales > np.finfo(float).eps)
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
            scales = np.std(bf_comb, axis=0, keepdims=True)
            bf_std = np.divide(bf_comb - np.mean(bf_comb, axis=0, keepdims=True), scales,
                               out=np.zeros_like(bf_comb), where=scales > np.finfo(float).eps)
            tf_std = _standardize(tf_comb)
            # A constant on-support candidate carries no continuous value signal.
            if tf_std is None:
                tf_std = np.zeros_like(tf_comb)

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

    if return_diagnostics:
        return marginal_ic, baseline_ic, MarginalContributionEvidence(
            factor_batch.factor_ids[target_factor_idx],
            tuple(factor_batch.factor_ids[i] for i in baseline_indices),
            baseline_ic, combined_ic, native_baseline_ic, native_target_ic,
            common_counts, native_baseline_counts, native_target_counts,
        )
    return marginal_ic, baseline_ic
