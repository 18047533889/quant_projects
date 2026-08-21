"""
Reference-Fast parity checker for kernel validation.

Compares fast kernel outputs against reference implementations to ensure
mathematical correctness. Used in testing and validation.
"""

from typing import Tuple, Optional, Dict
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.metrics.label_panel import normalize_label_panel
from quant_evaluator.metrics.quantile import assign_quantiles, compute_quantile_returns
from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks
from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    fast_turnover_estimate,
    compute_quantile_returns_fast,
)


def check_ic_parity(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    method: str = "pearson",
    min_assets: int = 10,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Dict[str, any]:
    """
    Verify parity between reference and fast IC implementations.

    Args:
        factor_batch: Input factor batch
        label_bundle: Input labels
        method: Correlation method
        min_assets: Minimum assets per period
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose

    Returns:
        Dictionary with parity results:
        - passed: bool
        - ic_match: bool
        - counts_match: bool
        - max_ic_diff: float
        - max_count_diff: int
        - reference_ic: np.ndarray
        - fast_ic: np.ndarray
        - reference_counts: np.ndarray
        - fast_counts: np.ndarray
    """
    # Reference implementation
    ic_ref, counts_ref = compute_daily_ic(
        factor_batch, label_bundle, method=method, min_assets=min_assets
    )

    normalized_labels, normalized_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )
    ic_fast, counts_fast = fast_ic_batch(
        factor_values=factor_batch.values,
        label_values=normalized_labels,
        method=method,
        min_obs=min_assets,
        factor_validity=factor_batch.validity,
        label_validity=normalized_validity,
    )

    # Compare IC values (NaN-aware)
    ic_match = _arrays_close_nanaware(ic_ref, ic_fast, rtol=rtol, atol=atol)

    # Compute max difference (excluding NaN pairs)
    mask = np.isfinite(ic_ref) & np.isfinite(ic_fast)
    if np.any(mask):
        max_ic_diff = float(np.max(np.abs(ic_ref[mask] - ic_fast[mask])))
    else:
        max_ic_diff = 0.0

    # Compare counts (exact match required)
    counts_match = np.array_equal(counts_ref, counts_fast)
    max_count_diff = int(np.max(np.abs(counts_ref - counts_fast)))

    passed = ic_match and counts_match

    return {
        "passed": passed,
        "ic_match": ic_match,
        "counts_match": counts_match,
        "max_ic_diff": max_ic_diff,
        "max_count_diff": max_count_diff,
        "reference_ic": ic_ref,
        "fast_ic": ic_fast,
        "reference_counts": counts_ref,
        "fast_counts": counts_fast,
    }


def check_quantile_parity(
    factor_batch: FactorBatch,
    n_quantiles: int = 5,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Dict[str, any]:
    """
    Verify parity between reference and fast quantile binning.

    Args:
        factor_batch: Input factor batch
        n_quantiles: Number of quantiles
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with parity results:
        - passed: bool
        - binning_match: bool
        - max_bin_diff: int
        - mismatch_rate: float (fraction of disagreements)
        - reference_bins: np.ndarray
        - fast_bins: np.ndarray
    """
    T, N, F = factor_batch.values.shape

    # Reference implementation (per time slice)
    bins_ref = np.full((T, N, F), -1, dtype=np.int32)
    for t in range(T):
        for f in range(F):
            v = factor_batch.values[t, :, f]
            # assign_quantiles expects shape (T, N), so reshape to (1, N)
            bins_t = assign_quantiles(v.reshape(1, -1), n_quantiles=n_quantiles)
            bins_ref[t, :, f] = bins_t.flatten()

    # Fast implementation
    bins_fast = fast_quantile_binning(factor_batch.values, n_quantiles=n_quantiles)

    # Compare binning (exact match for valid bins)
    binning_match = np.array_equal(bins_ref, bins_fast)

    # Compute mismatch rate
    valid_mask = (bins_ref >= 0) & (bins_fast >= 0)
    if np.any(valid_mask):
        mismatches = np.sum((bins_ref != bins_fast) & valid_mask)
        mismatch_rate = float(mismatches) / float(np.sum(valid_mask))
    else:
        mismatch_rate = 0.0

    max_bin_diff = int(np.max(np.abs(bins_ref - bins_fast)))

    passed = binning_match

    return {
        "passed": passed,
        "binning_match": binning_match,
        "max_bin_diff": max_bin_diff,
        "mismatch_rate": mismatch_rate,
        "reference_bins": bins_ref,
        "fast_bins": bins_fast,
    }


def check_quantile_returns_parity(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Dict[str, any]:
    """
    Verify parity between reference and fast quantile returns computation.

    Args:
        factor_batch: Input factor batch
        label_bundle: Input labels
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with parity results
    """
    # Reference implementation
    qret_ref, qcounts_ref = compute_quantile_returns(
        factor_batch, label_bundle, n_quantiles=n_quantiles, min_assets=min_assets
    )

    normalized_labels, normalized_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )
    if normalized_validity is not None:
        normalized_labels = np.where(
            normalized_validity, normalized_labels, np.nan
        )
    qret_fast, qcounts_fast = compute_quantile_returns_fast(
        factor_values=factor_batch.values,
        label_values=normalized_labels,
        n_quantiles=n_quantiles,
        min_assets=min_assets,
    )

    # Compare returns (NaN-aware)
    returns_match = _arrays_close_nanaware(qret_ref, qret_fast, rtol=rtol, atol=atol)

    # Compute max difference
    mask = np.isfinite(qret_ref) & np.isfinite(qret_fast)
    if np.any(mask):
        max_return_diff = float(np.max(np.abs(qret_ref[mask] - qret_fast[mask])))
    else:
        max_return_diff = 0.0

    # Compare counts
    counts_match = np.array_equal(qcounts_ref, qcounts_fast)
    max_count_diff = int(np.max(np.abs(qcounts_ref - qcounts_fast)))

    passed = returns_match and counts_match

    return {
        "passed": passed,
        "returns_match": returns_match,
        "counts_match": counts_match,
        "max_return_diff": max_return_diff,
        "max_count_diff": max_count_diff,
        "reference_returns": qret_ref,
        "fast_returns": qret_fast,
        "reference_counts": qcounts_ref,
        "fast_counts": qcounts_fast,
    }


def check_turnover_parity(
    factor_batch: FactorBatch,
    window: int = 1,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Dict[str, any]:
    """
    Verify parity between reference and fast turnover estimation.

    Args:
        factor_batch: Input factor batch
        window: Lag periods for comparison
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with parity results:
        - passed: bool
        - turnover_match: bool
        - max_turnover_diff: float
        - reference_turnover: np.ndarray
        - fast_turnover: np.ndarray
    """
    # Reference implementation
    turnover_ref = estimate_turnover_from_ranks(factor_batch, window=window)

    # Fast implementation
    turnover_fast = fast_turnover_estimate(factor_batch.values, window=window)

    # Compare (NaN-aware)
    turnover_match = _arrays_close_nanaware(turnover_ref, turnover_fast, rtol=rtol, atol=atol)

    # Compute max difference
    mask = np.isfinite(turnover_ref) & np.isfinite(turnover_fast)
    if np.any(mask):
        max_turnover_diff = float(np.max(np.abs(turnover_ref[mask] - turnover_fast[mask])))
    else:
        max_turnover_diff = 0.0

    passed = turnover_match

    return {
        "passed": passed,
        "turnover_match": turnover_match,
        "max_turnover_diff": max_turnover_diff,
        "reference_turnover": turnover_ref,
        "fast_turnover": turnover_fast,
    }


def _arrays_close_nanaware(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> bool:
    """
    NaN-aware array comparison.

    Two arrays are considered close if:
    1. NaN positions match exactly
    2. Non-NaN values are close within tolerance

    Args:
        a: First array
        b: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if arrays match within tolerance
    """
    if a.shape != b.shape:
        return False

    # Check NaN positions match
    nan_mask_a = np.isnan(a)
    nan_mask_b = np.isnan(b)

    if not np.array_equal(nan_mask_a, nan_mask_b):
        return False

    # Check finite values
    finite_mask = np.isfinite(a) & np.isfinite(b)

    if not np.any(finite_mask):
        return True  # All NaN in both

    return np.allclose(a[finite_mask], b[finite_mask], rtol=rtol, atol=atol)
