"""Scalar adapters used by the metric registry authority.

These adapters keep legacy tuple/series-returning metric APIs intact while
exposing one well-defined value per registered metric.
"""

import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.metrics.ic_summary import compute_icir
from quant_evaluator.metrics.robustness import compute_hac_tstat
from quant_evaluator.metrics.temporal import (
    compute_ic_autocorrelation,
    compute_mean_rank_stability,
    compute_half_life,
)


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
