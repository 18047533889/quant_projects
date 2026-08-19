"""
IC summary statistics: ICIR, t-statistics, and decay analysis.

Reference implementation for IC-based performance metrics with proper
statistical inference and temporal decay patterns.
"""

from typing import Tuple
import numpy as np
from scipy import stats

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic


def compute_icir(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Information Coefficient Information Ratio (ICIR).

    ICIR = mean(IC) / std(IC), measures consistency of IC signal.

    Args:
        ic_series: Daily IC series (T, F)
        min_periods: Minimum periods required

    Returns:
        ICIR array of shape (F,), NaN if insufficient periods or zero std
    """
    valid_periods = np.sum(~np.isnan(ic_series), axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_ic = np.nanmean(ic_series, axis=0)
        std_ic = np.nanstd(ic_series, axis=0, ddof=1)
        icir = mean_ic / std_ic

    insufficient = (
        (valid_periods < min_periods)
        | (std_ic < 1e-10)
        | np.isnan(std_ic)
        | ~np.isfinite(icir)
    )
    icir = np.where(insufficient, np.nan, icir)

    return icir


def compute_ic_tstat(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute t-statistic for IC series testing H0: mean(IC) = 0.

    Args:
        ic_series: Daily IC series (T, F)
        min_periods: Minimum periods required

    Returns:
        (t_stat, p_value) arrays of shape (F,)
        Two-tailed p-value for t-test
    """
    T, F = ic_series.shape
    t_stats = np.full(F, np.nan, dtype=np.float64)
    p_values = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = ic_series[:, f]
        valid_ic = ic_f[~np.isnan(ic_f)]

        if len(valid_ic) < min_periods:
            continue

        # One-sample t-test against H0: mean = 0
        t_stat, p_val = stats.ttest_1samp(valid_ic, 0.0)
        t_stats[f] = t_stat
        p_values[f] = p_val

    return t_stats, p_values


def compute_ic_decay(
    factor_batch: FactorBatch,
    label_bundles: Tuple[LabelBundle, ...],
    method: str = "pearson",
    min_assets: int = 10,
) -> np.ndarray:
    """
    Compute IC decay across multiple horizons.

    Measures how factor predictive power decays over time by computing
    IC for each horizon in label_bundles.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundles: Tuple of labels at different horizons (1d, 5d, 20d, etc.)
        method: "pearson" or "spearman"
        min_assets: Minimum valid assets per day

    Returns:
        decay_matrix: shape (num_horizons, F)
        Each row is mean IC for that horizon
    """
    num_horizons = len(label_bundles)
    F = factor_batch.num_factors

    decay_matrix = np.full((num_horizons, F), np.nan, dtype=np.float64)

    for h_idx, label_bundle in enumerate(label_bundles):
        ic_series, _ = compute_daily_ic(
            factor_batch, label_bundle, method=method, min_assets=min_assets
        )

        # Mean IC for this horizon
        with np.errstate(invalid='ignore'):
            mean_ic = np.nanmean(ic_series, axis=0)

        decay_matrix[h_idx, :] = mean_ic

    return decay_matrix


def compute_ic_stability(
    ic_series: np.ndarray,
    window_size: int = 60,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling IC stability metrics.

    Stability = correlation between first-half and second-half IC within rolling windows.
    High stability indicates consistent factor performance.

    Args:
        ic_series: Daily IC series (T, F)
        window_size: Rolling window size (must be even, >= 40)
        min_periods: Minimum valid periods in each half

    Returns:
        (stability_series, valid_windows)
        stability_series: shape (num_windows, F)
        valid_windows: count of valid windows per factor (F,)
    """
    if window_size < 40 or window_size % 2 != 0:
        raise ValueError(f"window_size must be even and >= 40, got {window_size}")

    T, F = ic_series.shape
    half_window = window_size // 2

    num_windows = T - window_size + 1
    if num_windows <= 0:
        return np.full((0, F), np.nan), np.zeros(F, dtype=np.int32)

    stability_series = np.full((num_windows, F), np.nan, dtype=np.float64)

    for f in range(F):
        for w in range(num_windows):
            first_half = ic_series[w:w+half_window, f]
            second_half = ic_series[w+half_window:w+window_size, f]

            # Correlate the same time periods across halves: drop a period
            # only when EITHER half is NaN.  Dropping per half independently
            # misaligns the series (and crashes corrcoef on length mismatch).
            valid = ~np.isnan(first_half) & ~np.isnan(second_half)
            first_valid = first_half[valid]
            second_valid = second_half[valid]

            if len(first_valid) < min_periods:
                continue

            # Pearson correlation between halves
            if np.std(first_valid) == 0 or np.std(second_valid) == 0:
                continue

            corr = np.corrcoef(first_valid, second_valid)[0, 1]
            stability_series[w, f] = corr

    valid_windows = np.sum(~np.isnan(stability_series), axis=0)

    return stability_series, valid_windows
