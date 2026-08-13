"""
Diagnosis utilities for factor quality assessment.
"""

from typing import Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.api.requests import FactorDiagnosis


def diagnose_factor(
    factor_batch: FactorBatch,
    factor_idx: int = 0,
) -> FactorDiagnosis:
    """
    Generate diagnostic information for a single factor.

    Args:
        factor_batch: Input factor batch
        factor_idx: Index of factor to diagnose

    Returns:
        FactorDiagnosis with coverage, validity, and distribution info
    """
    if factor_idx >= factor_batch.num_factors:
        raise ValueError(f"factor_idx {factor_idx} out of range (max {factor_batch.num_factors - 1})")

    factor_id = factor_batch.factor_ids[factor_idx]
    values = factor_batch.values[:, :, factor_idx].flatten()

    # Validity checks
    finite_mask = np.isfinite(values)
    has_nans = np.any(np.isnan(values))
    has_infs = np.any(np.isinf(values))

    # Apply validity mask if present
    if factor_batch.validity is not None:
        validity_mask = factor_batch.validity[:, :, factor_idx].flatten()
        valid_mask = finite_mask & validity_mask
    else:
        valid_mask = finite_mask

    num_valid = int(np.sum(valid_mask))
    num_total = len(values)
    num_missing = num_total - num_valid
    coverage = num_valid / num_total if num_total > 0 else 0.0

    # Distribution statistics
    valid_values = values[valid_mask]

    if num_valid > 0:
        is_constant = (np.std(valid_values) == 0)
        min_value = float(np.min(valid_values))
        max_value = float(np.max(valid_values))
        mean_value = float(np.mean(valid_values))
    else:
        is_constant = True
        min_value = None
        max_value = None
        mean_value = None

    # Warnings
    warnings = []
    if coverage < 0.5:
        warnings.append(f"Low coverage: {coverage:.2%}")
    if is_constant:
        warnings.append("Factor is constant")
    if has_nans:
        warnings.append("Contains NaN values")
    if has_infs:
        warnings.append("Contains Inf values")

    return FactorDiagnosis(
        factor_id=factor_id,
        num_valid_observations=num_valid,
        num_missing=num_missing,
        coverage=coverage,
        is_constant=is_constant,
        has_nans=has_nans,
        has_infs=has_infs,
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        warnings=tuple(warnings),
    )


def diagnose_all_factors(factor_batch: FactorBatch) -> dict:
    """
    Generate diagnostics for all factors in batch.

    Returns:
        Dict mapping factor_id -> FactorDiagnosis
    """
    diagnostics = {}

    for f_idx in range(factor_batch.num_factors):
        factor_id = factor_batch.factor_ids[f_idx]
        diagnostics[factor_id] = diagnose_factor(factor_batch, f_idx)

    return diagnostics
