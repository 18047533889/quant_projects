"""Scalar adapters used by the metric registry authority.

These adapters keep legacy tuple/series-returning metric APIs intact while
exposing one well-defined value per registered metric.
"""

import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic_summary import compute_icir
from quant_evaluator.metrics.quantile import compute_quantile_returns_fast
from quant_evaluator.metrics.robustness import (
    compute_block_bootstrap_ci,
    compute_hac_tstat,
    compute_subsample_ic_std,
)
from quant_evaluator.metrics.temporal import (
    compute_factor_turnover_rate,
    compute_half_life,
    compute_ic_autocorrelation,
    compute_mean_rank_stability,
)
from quant_evaluator.metrics.turnover import compute_turnover_series


def compute_ic_ir_value(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """Return the scalar IC information ratio per factor."""
    return compute_icir(ic_series, min_periods=min_periods)


def compute_hac_tstat_value(
    ic_series: np.ndarray,
    min_periods: int = 30,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> np.ndarray:
    """Return only the HAC t-statistic component per factor."""
    t_stat, _ = compute_hac_tstat(ic_series, max_lag=max_lag, kernel=kernel)
    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    return np.where(valid_periods >= min_periods, t_stat, np.nan)


def compute_ic_autocorr_lag1_value(
    ic_series: np.ndarray,
    min_periods: int = 30,
    max_lag: int = 20,
) -> np.ndarray:
    """Return lag-one IC autocorrelation per factor."""
    acf = compute_ic_autocorrelation(
        ic_series, max_lag=max_lag, min_obs=min_periods
    )
    return acf[1] if acf.shape[0] > 1 else np.full(ic_series.shape[1], np.nan)


def compute_rank_stability_value(
    factor_batch: FactorBatch,
    min_periods: int = 20,
    lag: int = 1,
    method: str = "spearman",
) -> np.ndarray:
    """Return time-averaged rank stability per factor."""
    return compute_mean_rank_stability(
        factor_batch.values,
        lag=lag,
        method=method,
        min_periods=min_periods,
    )


def compute_half_life_value(
    ic_series: np.ndarray,
    min_periods: int = 60,
) -> np.ndarray:
    """Return estimated IC half-life per factor."""
    return compute_half_life(ic_series, min_periods=min_periods)


def compute_turnover_value(
    factor_batch: FactorBatch,
    min_periods: int = 2,
) -> np.ndarray:
    """Return mean cross-sectional turnover per factor.

    Ranks are used as proxy weights so the value is well defined for a raw
    factor batch without a portfolio construction step.
    """
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("factor_batch.values must be (T, N, F)")
    n_factors = values.shape[2]
    result = np.full(n_factors, np.nan, dtype=np.float64)
    for f in range(n_factors):
        series = values[:, :, f]
        # Cross-sectional rank weights in [0, 1]; NaN positions stay NaN so
        # turnover is only measured over jointly finite neighbours.
        ranks = np.full_like(series, np.nan)
        for t in range(series.shape[0]):
            row = series[t]
            finite = np.isfinite(row)
            if finite.sum() < 2:
                continue
            ranks[t, finite] = np.argsort(np.argsort(row[finite])) / (
                finite.sum() - 1
            )
        if series.shape[0] < 2:
            continue
        turnover_series = compute_turnover_series(ranks)
        valid = turnover_series[np.isfinite(turnover_series)]
        if valid.size >= min_periods - 1:
            result[f] = float(np.mean(valid))
    return result


def compute_quantile_spread_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 20,
    n_quantiles: int = 5,
) -> np.ndarray:
    """Return time-averaged top-minus-bottom quantile return spread per factor."""
    quantile_returns, _ = compute_quantile_returns_fast(
        factor_batch, label_bundle, n_quantiles=n_quantiles
    )
    # (T, n_quantiles, F) -> mean over time of Q_top - Q_bottom per factor.
    with np.errstate(invalid="ignore"):
        spread_series = quantile_returns[:, -1, :] - quantile_returns[:, 0, :]
    valid_counts = np.sum(np.isfinite(spread_series), axis=0)
    means = np.nanmean(spread_series, axis=0) if spread_series.size else np.array([])
    return np.where(valid_counts >= min_periods, means, np.nan)


def compute_subsample_stability_value(
    ic_series: np.ndarray,
    min_periods: int = 40,
    num_subsamples: int = 100,
    subsample_fraction: float = 0.8,
    random_seed: int = 0,
) -> np.ndarray:
    """Return the std of mean IC across bootstrap subsamples per factor."""
    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    stability = compute_subsample_ic_std(
        ic_series,
        num_subsamples=num_subsamples,
        subsample_fraction=subsample_fraction,
        random_seed=random_seed,
    )
    return np.where(valid_periods >= min_periods, stability, np.nan)


def compute_quantile_returns_full_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 20,
    n_quantiles: int = 5,
) -> np.ndarray:
    """Return per-quantile time-averaged returns, shape (n_quantiles, F)."""
    quantile_returns, _ = compute_quantile_returns_fast(
        factor_batch, label_bundle, n_quantiles=n_quantiles
    )
    with np.errstate(invalid="ignore"):
        means = np.nanmean(quantile_returns, axis=0)  # (n_quantiles, F)
    valid_counts = np.sum(np.isfinite(quantile_returns), axis=0)  # (n_quantiles, F)
    return np.where(valid_counts >= min_periods, means, np.nan)

def compute_block_bootstrap_ci_value(
    ic_series: np.ndarray,
    min_periods: int = 60,
    block_length: int = 10,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> np.ndarray:
    """Return the block-bootstrap CI half-width per factor.

    compute_block_bootstrap_ci returns a (lower, upper) tuple; the registry
    exposes the scalar half-width (upper - lower) / 2, NaN below min_periods
    (matching the other ic_series adapters' contract).
    """
    ci_lower, ci_upper = compute_block_bootstrap_ci(
        ic_series,
        block_length=block_length,
        num_bootstrap=num_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    with np.errstate(invalid="ignore"):
        half_width = (ci_upper - ci_lower) / 2.0
    return np.where(valid_periods >= min_periods, half_width, np.nan)

def compute_factor_turnover_rate_value(
    factor_batch: FactorBatch,
    min_periods: int = 30,
    quantile: float = 0.9,
) -> np.ndarray:
    """Return mean top-quantile membership turnover per factor.

    compute_factor_turnover_rate returns a (T-1, F) series; the registry
    exposes the time-averaged scalar per factor, NaN below min_periods.
    """
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("factor_batch.values must be (T, N, F)")
    turnover_series = compute_factor_turnover_rate(values, quantile=quantile)
    if turnover_series.size == 0:
        return np.full(values.shape[2], np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(turnover_series, axis=0)
    valid_counts = np.sum(np.isfinite(turnover_series), axis=0)
    return np.where(valid_counts >= min_periods, means, np.nan)
