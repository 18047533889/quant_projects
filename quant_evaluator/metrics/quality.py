"""
Coverage diagnostics for factor batches.

Pure batch operations: no data fetch, no implicit fills.
"""

from typing import Dict, Tuple
import warnings
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.metrics.label_panel import normalize_label_panel


def _valid_pair_mask(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """Return the pairwise-valid boolean mask of shape (T, N, F).

    A (t, n, f) cell is valid when the factor value, the normalized label
    value, the factor validity mask (if present), and the label validity
    mask (if present) are all finite/true.
    """
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(
            f"Factor time axis ({factor_batch.num_times}) "
            f"does not match label length ({len(label_bundle.values)})"
        )

    values = factor_batch.values  # shape: (T, N, F)
    labels, label_validity = normalize_label_panel(label_bundle, factor_batch.num_assets)

    factor_finite = np.isfinite(values)  # (T, N, F)
    label_finite = np.isfinite(labels)   # (T, N)

    label_finite_expanded = label_finite[:, :, np.newaxis]  # (T, N, 1)
    valid_pairs = factor_finite & label_finite_expanded    # (T, N, F)

    if factor_batch.validity is not None:
        valid_pairs = valid_pairs & factor_batch.validity
    if label_validity is not None:
        valid_pairs = valid_pairs & label_validity[:, :, np.newaxis]

    return valid_pairs


def compute_coverage(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> Tuple[float, int, int]:
    """
    Compute coverage aggregated over ALL factors: fraction of valid
    (factor, label) pairs.

    .. deprecated::
        This aggregates every factor column into a single number, which is
        a meaningless average for multi-factor batches (values are
        (T, N, F)). ``min_assets`` is accepted but ignored here. Use
        :func:`compute_coverage_per_factor` instead, which reports one
        entry per factor and actually enforces ``min_assets``.

    Args:
        factor_batch: Input factor batch
        label_bundle: Input label bundle
        min_assets: Unused (kept for signature compatibility)

    Returns:
        (coverage_fraction, num_valid, num_total)

    Raises:
        InvalidContractError: If shapes are incompatible
    """
    warnings.warn(
        "compute_coverage aggregates coverage across ALL factor columns "
        "into a single number; use compute_coverage_per_factor (one entry "
        "per factor, min_assets enforced) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    valid_pairs = _valid_pair_mask(factor_batch, label_bundle)

    num_valid = int(np.sum(valid_pairs))
    num_total = factor_batch.values.size

    coverage = num_valid / num_total if num_total > 0 else 0.0

    return coverage, num_valid, num_total


def compute_coverage_per_factor(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute a per-factor coverage report.

    Unlike the deprecated :func:`compute_coverage`, this never averages
    across factor columns and actually USES ``min_assets``: days on which
    a factor has fewer than ``min_assets`` jointly valid (factor, label)
    observations are counted in ``days_below_min_assets``.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle
        min_assets: Minimum valid assets per day for a day to count as
            a "valid day" (enforced, not diagnostic)

    Returns:
        Dict keyed by factor_id with entries:
            {"coverage": float, "num_valid": int, "num_total": int,
             "valid_days": int, "days_below_min_assets": int}

    Raises:
        InvalidContractError: If shapes are incompatible
    """
    valid_pairs = _valid_pair_mask(factor_batch, label_bundle)
    per_day_counts = np.sum(valid_pairs, axis=1)  # (T, F)
    num_times = valid_pairs.shape[0]

    report: Dict[str, Dict[str, float]] = {}
    for index, factor_id in enumerate(factor_batch.factor_ids):
        counts = per_day_counts[:, index]
        num_valid = int(counts.sum())
        num_total = int(valid_pairs[:, :, index].size)
        valid_days = int(np.sum(counts >= min_assets))
        report[factor_id] = {
            "coverage": num_valid / num_total if num_total > 0 else 0.0,
            "num_valid": num_valid,
            "num_total": num_total,
            "valid_days": valid_days,
            "days_below_min_assets": int(num_times) - valid_days,
        }
    return report


def compute_valid_pair_counts(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """
    Compute the number of jointly valid (factor, label) observations per
    factor.

    Returns:
        Integer array of shape (F,) — per-factor observation counts over
        the whole batch.
    """
    valid_pairs = _valid_pair_mask(factor_batch, label_bundle)
    return np.sum(valid_pairs, axis=(0, 1)).astype(np.int64)


def compute_per_time_coverage(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """
    Compute coverage per time period.

    Returns:
        Array of shape (T, F) with coverage per time per factor.
    """
    valid_pairs = _valid_pair_mask(factor_batch, label_bundle)

    # Count valid pairs per time: (T, F)
    valid_per_time = np.sum(valid_pairs, axis=1)
    total_per_time = factor_batch.num_assets

    coverage_per_time = valid_per_time / total_per_time

    return coverage_per_time
