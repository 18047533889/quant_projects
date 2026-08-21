"""
Robustness metrics: subsample stability, HAC variance estimation.

Reference implementation for measuring factor robustness and inference quality
under various data perturbations and autocorrelation structures.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats


def compute_subsample_ic(
    ic_series: np.ndarray,
    num_subsamples: int = 100,
    subsample_fraction: float = 0.8,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IC stability via bootstrap subsampling.

    Randomly samples subsample_fraction of time periods and computes mean IC.
    Repeated num_subsamples times to build distribution.

    Args:
        ic_series: Daily IC series (T, F)
        num_subsamples: Number of bootstrap samples
        subsample_fraction: Fraction of periods to sample (0, 1)
        random_seed: Random seed for reproducibility

    Returns:
        (subsample_means, subsample_stds)
        subsample_means: shape (num_subsamples, F)
        subsample_stds: shape (num_subsamples, F)
    """
    if subsample_fraction <= 0 or subsample_fraction >= 1:
        raise ValueError(f"subsample_fraction must be in (0, 1), got {subsample_fraction}")

    T, F = ic_series.shape
    subsample_size = max(1, int(T * subsample_fraction))

    # Local Generator: never touch the global numpy RNG state.
    rng = np.random.default_rng(random_seed)

    subsample_means = np.full((num_subsamples, F), np.nan, dtype=np.float64)
    subsample_stds = np.full((num_subsamples, F), np.nan, dtype=np.float64)

    for b in range(num_subsamples):
        # Random sample of time indices (local Generator)
        sampled_indices = rng.choice(T, size=subsample_size, replace=False)
        ic_subsample = ic_series[sampled_indices, :]  # (subsample_size, F)

        # Compute mean and std for this subsample
        with np.errstate(invalid='ignore'):
            subsample_means[b, :] = np.nanmean(ic_subsample, axis=0)
            subsample_stds[b, :] = np.nanstd(ic_subsample, axis=0, ddof=1)

    return subsample_means, subsample_stds


def compute_subsample_ic_std(
    ic_series: np.ndarray,
    num_subsamples: int = 100,
    subsample_fraction: float = 0.8,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Compute standard deviation of mean IC across subsamples.

    Low std indicates robust IC estimate across different time periods.

    Args:
        ic_series: Daily IC series (T, F)
        num_subsamples: Number of bootstrap samples
        subsample_fraction: Fraction of periods to sample
        random_seed: Random seed

    Returns:
        robustness_std: shape (F,) - std of subsample mean ICs
    """
    subsample_means, _ = compute_subsample_ic(
        ic_series, num_subsamples, subsample_fraction, random_seed
    )

    with np.errstate(invalid='ignore'):
        robustness_std = np.nanstd(subsample_means, axis=0, ddof=1)

    return robustness_std


def compute_hac_variance(
    series: np.ndarray,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> np.ndarray:
    """
    Compute Heteroskedasticity and Autocorrelation Consistent (HAC) variance.

    Newey-West HAC variance estimator for time series with autocorrelation.
    Uses weighted sum of autocovariances with kernel weighting.

    Args:
        series: Time series (T,) or (T, F)
        max_lag: Maximum lag for HAC estimation
        kernel: Kernel type - "bartlett" (triangular) or "uniform"

    Returns:
        hac_var: shape (F,) - HAC variance estimate per factor
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    T, F = series.shape

    if kernel not in ("bartlett", "uniform"):
        raise ValueError(f"Unknown kernel: {kernel}, must be 'bartlett' or 'uniform'")

    hac_var = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ts = series[:, f]
        valid_ts = ts[~np.isnan(ts)]

        if len(valid_ts) < max_lag + 10:
            continue

        # Demean
        ts_demean = valid_ts - np.mean(valid_ts)
        n = len(ts_demean)

        # Compute lag-0 autocovariance (variance)
        gamma_0 = np.mean(ts_demean ** 2)

        # HAC variance: gamma_0 + 2 * sum(weight(lag) * gamma(lag))
        hac_est = gamma_0

        for lag in range(1, max_lag + 1):
            if lag >= n:
                break

            # Autocovariance at lag
            gamma_lag = np.mean(ts_demean[:-lag] * ts_demean[lag:])

            # Kernel weight
            if kernel == "bartlett":
                weight = 1.0 - lag / (max_lag + 1)
            else:  # uniform
                weight = 1.0

            hac_est += 2.0 * weight * gamma_lag

        hac_var[f] = hac_est / n

    return hac_var


def compute_hac_tstat(
    ic_series: np.ndarray,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HAC-robust t-statistic for IC series.

    Tests H0: mean(IC) = 0 using HAC standard errors.
    More conservative than standard t-test when IC is autocorrelated.

    Args:
        ic_series: Daily IC series (T, F)
        max_lag: Maximum lag for HAC estimation
        kernel: Kernel type

    Returns:
        (t_stat_hac, se_hac)
        t_stat_hac: shape (F,) - HAC-robust t-statistics
        se_hac: shape (F,) - HAC standard errors
    """
    T, F = ic_series.shape

    t_stat_hac = np.full(F, np.nan, dtype=np.float64)
    se_hac = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = ic_series[:, f]
        valid_ic = ic_f[~np.isnan(ic_f)]

        if len(valid_ic) < max_lag + 10:
            continue

        # Mean IC
        mean_ic = np.mean(valid_ic)
        n = len(valid_ic)

        # HAC variance
        hac_var = compute_hac_variance(
            valid_ic.reshape(-1, 1), max_lag=max_lag, kernel=kernel
        )[0]

        if hac_var <= 0 or np.isnan(hac_var):
            continue

        # HAC standard error
        se = np.sqrt(hac_var)
        se_hac[f] = se

        # HAC t-statistic
        t_stat_hac[f] = mean_ic / se

    return t_stat_hac, se_hac


def compute_block_bootstrap_ci(
    ic_series: np.ndarray,
    block_length: int = 10,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence interval via block bootstrap.

    Block bootstrap preserves temporal dependence structure by resampling
    contiguous blocks of observations.

    Args:
        ic_series: Daily IC series (T, F)
        block_length: Length of each bootstrap block
        num_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed

    Returns:
        (ci_lower, ci_upper)
        ci_lower: shape (F,) - lower bound of CI for mean IC
        ci_upper: shape (F,) - upper bound of CI for mean IC
    """
    T, F = ic_series.shape

    if block_length <= 0 or block_length > T:
        raise ValueError(f"block_length must be in (0, {T}], got {block_length}")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    # Local Generator: never touch the global numpy RNG state.
    rng = np.random.default_rng(random_seed)

    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    ci_lower = np.full(F, np.nan, dtype=np.float64)
    ci_upper = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = ic_series[:, f]
        valid_ic = ic_f[~np.isnan(ic_f)]

        if len(valid_ic) < block_length * 2:
            continue

        n = len(valid_ic)
        num_blocks = (n + block_length - 1) // block_length

        bootstrap_means = np.full(num_bootstrap, np.nan, dtype=np.float64)

        for b in range(num_bootstrap):
            # Sample blocks with replacement (local Generator)
            block_starts = rng.choice(n - block_length + 1, size=num_blocks, replace=True)

            # Reconstruct bootstrap sample
            bootstrap_sample = []
            for start in block_starts:
                bootstrap_sample.extend(valid_ic[start:start + block_length])

            bootstrap_sample = np.array(bootstrap_sample[:n])  # Trim to original length
            bootstrap_means[b] = np.mean(bootstrap_sample)

        # Compute percentiles
        ci_lower[f] = np.percentile(bootstrap_means, lower_percentile)
        ci_upper[f] = np.percentile(bootstrap_means, upper_percentile)

    return ci_lower, ci_upper
