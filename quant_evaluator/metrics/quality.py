"""
Coverage diagnostics for factor batches.

Pure batch operations: no data fetch, no implicit fills.
"""

from typing import Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.metrics.label_panel import normalize_label_panel


def compute_coverage(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> Tuple[float, int, int]:
    """
    Compute coverage: fraction of valid (factor, label) pairs.

    Args:
        factor_batch: Input factor batch
        label_bundle: Input label bundle
        min_assets: Minimum valid assets per time period (for diagnostic only)

    Returns:
        (coverage_fraction, num_valid, num_total)

    Raises:
        InvalidContractError: If shapes are incompatible
    """
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(
            f"Factor time axis ({factor_batch.num_times}) "
            f"does not match label length ({len(label_bundle.values)})"
        )

    # Extract first factor for single-factor coverage
    # For multi-factor, this computes per-factor coverage
    values = factor_batch.values  # shape: (T, N, F)
    labels, label_validity = normalize_label_panel(label_bundle, factor_batch.num_assets)

    # Compute pairwise finite mask
    factor_finite = np.isfinite(values)  # (T, N, F)
    label_finite = np.isfinite(labels)   # (T, N)

    # For each factor, compute valid pairs
    label_finite_expanded = label_finite[:, :, np.newaxis]  # (T, N, 1)
    valid_pairs = factor_finite & label_finite_expanded  # (T, N, F)

    # Apply validity masks if present
    if factor_batch.validity is not None:
        valid_pairs = valid_pairs & factor_batch.validity
    if label_validity is not None:
        valid_pairs = valid_pairs & label_validity[:, :, np.newaxis]

    num_valid = int(np.sum(valid_pairs))
    num_total = values.size

    coverage = num_valid / num_total if num_total > 0 else 0.0

    return coverage, num_valid, num_total


def compute_per_time_coverage(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """
    Compute coverage per time period.

    Returns:
        Array of shape (T, F) with coverage per time per factor.
    """
    values = factor_batch.values  # (T, N, F)
    labels, label_validity = normalize_label_panel(label_bundle, factor_batch.num_assets)

    factor_finite = np.isfinite(values)
    label_finite = np.isfinite(labels)[:, :, np.newaxis]

    valid_pairs = factor_finite & label_finite

    if factor_batch.validity is not None:
        valid_pairs = valid_pairs & factor_batch.validity
    if label_validity is not None:
        valid_pairs = valid_pairs & label_validity[:, :, np.newaxis]

    # Count valid pairs per time: (T, F)
    valid_per_time = np.sum(valid_pairs, axis=1)
    total_per_time = factor_batch.num_assets

    coverage_per_time = valid_per_time / total_per_time

    return coverage_per_time
