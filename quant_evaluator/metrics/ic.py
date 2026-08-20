"""
Information Coefficient (IC) metrics.

Reference implementation: Pearson correlation and Spearman RankIC with
pairwise-finite filtering and average-tie handling.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import stats

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InsufficientObservations, InvalidContractError
from quant_evaluator.metrics.label_panel import normalize_label_panel


def _pairwise_finite_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise finite mask for two arrays.

    Args:
        x: First array
        y: Second array (must be broadcastable with x)

    Returns:
        Boolean mask where both x and y are finite
    """
    return np.isfinite(x) & np.isfinite(y)


def _pearson_correlation(x: np.ndarray, y: np.ndarray, min_obs: int = 10) -> float:
    """
    Pearson correlation with pairwise finite filtering.

    Args:
        x: Factor values (1D)
        y: Label values (1D)
        min_obs: Minimum observations required

    Returns:
        Correlation coefficient, or NaN if insufficient data or constant
    """
    mask = _pairwise_finite_mask(x, y)
    x_valid = x[mask]
    y_valid = y[mask]

    n = len(x_valid)
    if n < min_obs:
        return np.nan

    # Check for constants
    if np.std(x_valid) == 0 or np.std(y_valid) == 0:
        return np.nan

    # Compute Pearson correlation
    corr = np.corrcoef(x_valid, y_valid)[0, 1]

    return corr


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray, min_obs: int = 10) -> float:
    """
    Spearman rank correlation with pairwise finite filtering and average ties.

    Args:
        x: Factor values (1D)
        y: Label values (1D)
        min_obs: Minimum observations required

    Returns:
        Rank correlation coefficient, or NaN if insufficient data or constant
    """
    mask = _pairwise_finite_mask(x, y)
    x_valid = x[mask]
    y_valid = y[mask]

    n = len(x_valid)
    if n < min_obs:
        return np.nan

    # Check for constants (all values same after ranking)
    if len(np.unique(x_valid)) == 1 or len(np.unique(y_valid)) == 1:
        return np.nan

    # Use scipy's spearmanr with average tie handling
    corr, _ = stats.spearmanr(x_valid, y_valid)

    return corr


def compute_daily_ic(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    method: str = "pearson",
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute daily IC series for factor batch.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N) or (T,)
        method: "pearson" or "spearman"
        min_assets: Minimum valid assets per day

    Returns:
        (ic_series, valid_count_series)
        ic_series: shape (T, F) with IC per day per factor
        valid_count_series: shape (T, F) with count of valid obs

    Raises:
        InvalidContractError: If shapes incompatible
        InsufficientObservations: If no valid periods found
    """
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(
            f"Factor time axis ({factor_batch.num_times}) "
            f"does not match label length ({len(label_bundle.values)})"
        )

    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    corr_fn = _pearson_correlation if method == "pearson" else _spearman_rank_correlation

    values = factor_batch.values  # (T, N, F)
    labels, label_validity = normalize_label_panel(label_bundle, factor_batch.num_assets)

    T, N, F = values.shape
    ic_series = np.full((T, F), np.nan, dtype=np.float64)
    valid_counts = np.zeros((T, F), dtype=np.int32)

    # Compute IC per day per factor
    for t in range(T):
        for f in range(F):
            factor_t = values[t, :, f]  # (N,)
            label_t = labels[t, :]       # (N,)

            # Apply validity masks if present
            if factor_batch.validity is not None:
                factor_valid = factor_batch.validity[t, :, f]
                factor_t = np.where(factor_valid, factor_t, np.nan)

            if label_validity is not None:
                label_t = np.where(label_validity[t], label_t, np.nan)

            # Compute correlation
            ic = corr_fn(factor_t, label_t, min_obs=min_assets)
            ic_series[t, f] = ic

            # Count valid observations
            mask = _pairwise_finite_mask(factor_t, label_t)
            valid_counts[t, f] = int(np.sum(mask))

    return ic_series, valid_counts


def _reject_boolean_ic_series(ic_series: np.ndarray) -> None:
    """
    Reject boolean IC series.

    True/False silently coerces to 1.0/0.0 (e.g. a validity mask), which
    would yield a plausible-looking mean IC that carries no information.

    Raises:
        ValueError: If ic_series has boolean dtype or contains Python bools.
    """
    if ic_series.dtype == bool:
        raise ValueError(
            "ic_series must be numeric, got boolean dtype "
            "(True/False would silently coerce to 1.0/0.0)"
        )
    if ic_series.dtype == object:
        if any(isinstance(v, (bool, np.bool_)) for v in ic_series.ravel()):
            raise ValueError(
                "ic_series must be numeric, got Python bools "
                "(True/False would silently coerce to 1.0/0.0)"
            )


def compute_mean_ic(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean IC and its standard deviation across time.

    Args:
        ic_series: Daily IC series (T, F)
        valid_counts: Valid observation counts (T, F)
        min_periods: Minimum periods required for mean

    Returns:
        (mean_ic, ic_std) arrays of shape (F,)

    Raises:
        ValueError: If ic_series is boolean (or contains Python bools)
    """
    _reject_boolean_ic_series(np.asarray(ic_series))

    # Count non-NaN periods per factor
    valid_periods = np.sum(~np.isnan(ic_series), axis=0)  # (F,)

    # Compute mean and std with nan-safe operations
    with np.errstate(invalid='ignore'):
        mean_ic = np.nanmean(ic_series, axis=0)  # (F,)
        ic_std = np.nanstd(ic_series, axis=0, ddof=1)  # (F,)

    # Mask insufficient periods
    insufficient = valid_periods < min_periods
    mean_ic = np.where(insufficient, np.nan, mean_ic)
    ic_std = np.where(insufficient, np.nan, ic_std)

    # Mask non-finite results: inf in the series propagates through nanmean
    # into an infinite (invalid) mean/std — report NaN instead.
    mean_ic = np.where(np.isfinite(mean_ic), mean_ic, np.nan)
    ic_std = np.where(np.isfinite(ic_std), ic_std, np.nan)

    return mean_ic, ic_std


def compute_mean_ic_value(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Compute only the mean IC component for registry execution."""
    mean_ic, _ = compute_mean_ic(
        ic_series,
        valid_counts=valid_counts,
        min_periods=min_periods,
    )
    return mean_ic


def compute_ic_std(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Compute only the IC standard-deviation component."""
    _, ic_std = compute_mean_ic(
        ic_series,
        valid_counts=valid_counts,
        min_periods=min_periods,
    )
    return ic_std
