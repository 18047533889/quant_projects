"""Scalar adapters used by the metric registry authority.

These adapters keep legacy tuple/series-returning metric APIs intact while
exposing one well-defined value per registered metric.
"""

import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic
from quant_evaluator.metrics.ic_summary import compute_icir
from quant_evaluator.metrics.quality import compute_coverage_per_factor
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
    factor batch without a portfolio construction step. Weights are
    average-tie ranks (``scipy.stats.rankdata(method="average")``)
    normalized to sum 1 per row, so the value is universe-size invariant
    and conforms to the canonical turnover definition in
    ``metrics/turnover.py``.
    """
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("factor_batch.values must be (T, N, F)")
    n_factors = values.shape[2]
    result = np.full(n_factors, np.nan, dtype=np.float64)
    for f in range(n_factors):
        series = values[:, :, f]
        # Cross-sectional average-tie rank weights (scipy rankdata,
        # method="average"), normalized to sum 1 over the finite assets of
        # each row: universe-size invariant and consistent with the canonical
        # turnover definition (see metrics/turnover.py module docstring).
        # NaN positions stay NaN so turnover is only measured over jointly
        # finite neighbours.
        from scipy.stats import rankdata

        ranks = np.full_like(series, np.nan)
        for t in range(series.shape[0]):
            row = series[t]
            finite = np.isfinite(row)
            if finite.sum() < 2:
                continue
            ranks_f = rankdata(row[finite], method="average")
            ranks[t, finite] = ranks_f / np.sum(ranks_f)
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


# ---------------------------------------------------------------------------
# QE-METRIC-P0-01..04 additions (coverage semantics, canonical IC namespace,
# metric artifacts). Purely additive: append-only block.
# ---------------------------------------------------------------------------

def compute_coverage_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> np.ndarray:
    """Return the coverage fraction per factor, shape (F,).

    Uses :func:`quant_evaluator.metrics.quality.compute_coverage_per_factor`
    (the single truth for coverage semantics): no aggregation across factor
    columns, and ``min_assets`` is honoured for day-level diagnostics.
    """
    report = compute_coverage_per_factor(factor_batch, label_bundle, min_assets=min_assets)
    return np.asarray(
        [report[factor_id]["coverage"] for factor_id in factor_batch.factor_ids],
        dtype=np.float64,
    )


def compute_pearson_ic_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 1,
    min_assets: int = 10,
) -> np.ndarray:
    """Return the time-mean daily Pearson IC per factor, shape (F,)."""
    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="pearson", min_assets=min_assets
    )
    mean, _ = compute_mean_ic(ic_series, min_periods=min_periods)
    return mean


def compute_rank_ic_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 1,
    min_assets: int = 10,
) -> np.ndarray:
    """Return the time-mean daily Spearman (rank) IC per factor, shape (F,).

    ``rank_ic`` has exactly ONE meaning in this package: the time-average of
    daily Spearman rank IC between factor values and labels.
    """
    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="spearman", min_assets=min_assets
    )
    mean, _ = compute_mean_ic(ic_series, min_periods=min_periods)
    return mean


def compute_pearson_ic_series_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> np.ndarray:
    """Return the daily Pearson IC series per factor, shape (T, F)."""
    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="pearson", min_assets=min_assets
    )
    return ic_series


def compute_rank_ic_series_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> np.ndarray:
    """Return the daily Spearman (rank) IC series per factor, shape (T, F)."""
    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="spearman", min_assets=min_assets
    )
    return ic_series


def compute_ic_median_value(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """Return the time-median IC per factor, shape (F,)."""
    import warnings

    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            medians = np.nanmedian(ic_series, axis=0)
    medians = np.where(np.isfinite(medians), medians, np.nan)
    return np.where(valid_periods >= min_periods, medians, np.nan)


def compute_hac_pvalue_value(
    ic_series: np.ndarray,
    min_periods: int = 30,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> np.ndarray:
    """Return the two-sided HAC p-value for mean(IC) != 0 per factor.

    Uses the HAC t-statistic with a Gaussian null approximation; NaN below
    min_periods (matching the hac_tstat adapter contract).
    """
    from scipy import stats as _stats

    t_stat, _ = compute_hac_tstat(ic_series, max_lag=max_lag, kernel=kernel)
    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    with np.errstate(invalid="ignore"):
        p_values = 2.0 * _stats.norm.sf(np.abs(t_stat))
    p_values = np.where(np.isfinite(p_values), p_values, np.nan)
    return np.where(valid_periods >= min_periods, p_values, np.nan)


def compute_quantile_returns_full_artifact(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 20,
    n_quantiles: int = 5,
) -> "VectorMetricArtifact":
    """Wrap the per-quantile return vector as a VectorMetricArtifact.

    ``quantile_returns_full`` is a VECTOR per factor — shape
    (n_quantiles, F) — not a scalar; this artifact type says so in the type
    system instead of lying about dimensionality.
    """
    from quant_evaluator.contracts.metric_artifacts import VectorMetricArtifact

    values = compute_quantile_returns_full_value(
        factor_batch,
        label_bundle,
        min_periods=min_periods,
        n_quantiles=n_quantiles,
    )
    return VectorMetricArtifact(
        metric_id="quantile_returns_full",
        domain="quantile",
        artifact_kind="vector",
        values=np.asarray(values),
        provenance={"n_quantiles": int(n_quantiles), "min_periods": int(min_periods)},
        created_from=("factor_batch", "label_bundle"),
    )
