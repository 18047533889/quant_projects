"""
Factor Auto-Treatment Optimizer — dimension metric registry mapping.

Maps the six evaluation dimensions (plus a separate COMPLEXITY penalty
dimension) onto the ACTUAL metric_ids registered in the single MetricRegistry
authority (``quant_evaluator.registry.metrics``).

Fail-closed contract:
  - ``dimension_for`` returns ``None`` for any metric_id the registry does not
    know (never silently buckets an unknown id).
  - ``collect_dimension_metric_ids`` filters ``DIMENSION_METRICS`` through the
    registry's known ids, so we never reference a metric the registry does not
    actually register.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

from quant_evaluator.registry.metrics import MetricRegistry, list_metrics


class DimensionName(Enum):
    """Evaluation dimensions for factor auto-treatment optimization."""

    PREDICTIVE = "predictive"
    STABILITY = "stability"
    ROBUSTNESS = "robustness"
    TRADABILITY = "tradability"
    PURITY_EXPOSURE = "purity_exposure"
    DATA_QUALITY = "data_quality"


# COMPLEXITY is a separate penalty dimension (not part of the six quality
# dimensions above). It is intentionally kept out of DimensionName so callers
# can treat it as a cost/penalty axis rather than a quality axis.
COMPLEXITY = "complexity"


# ---------------------------------------------------------------------------
# Dimension -> registered metric_id mapping.
#
# IMPORTANT: every id below MUST exist in the MetricRegistry. The registry
# currently registers the following ids (see registry/metrics.py):
#   block_bootstrap_ci, coverage, factor_turnover_rate, hac_pvalue, hac_tstat,
#   half_life, ic_autocorr_lag1, ic_ir, ic_median, ic_std, mean_ic,
#   pearson_ic, pearson_ic_ir, pearson_ic_series, pearson_ic_std,
#   quantile_returns_full, quantile_spread, rank_ic, rank_ic_series,
#   rank_stability, subsample_stability, turnover
#
# GAPS (metrics the task spec names but the registry does NOT yet register):
#   - PREDICTIVE: top_bottom_spread (registry has quantile_spread instead)
#   - STABILITY: positive_ic_ratio (not registered)
#   - ROBUSTNESS: worst_slice, rolling_ic_std (not registered)
#   - TRADABILITY: signal_turnover, delay_degradation, cost_adjusted_alpha
#     (not registered; registry has turnover + factor_turnover_rate)
#   - PURITY_EXPOSURE: industry_exposure, size_exposure, beta_exposure,
#     style_concentration (NOT registered — dimension currently has no
#     registered metrics; a sibling owns exposure extension)
#   - DATA_QUALITY: missing_rate (not registered; registry has coverage only)
# ---------------------------------------------------------------------------
DIMENSION_METRICS: Dict[DimensionName, Tuple[str, ...]] = {
    DimensionName.PREDICTIVE: (
        "rank_ic",
        "pearson_ic",
        "mean_ic",
        "ic_median",
        "quantile_spread",
        "rank_ic_series",
        "pearson_ic_series",
        "quantile_returns_full",
    ),
    DimensionName.STABILITY: (
        "ic_ir",
        "pearson_ic_ir",
        "ic_std",
        "pearson_ic_std",
        "ic_autocorr_lag1",
        "rank_stability",
        "subsample_stability",
        "half_life",
        "block_bootstrap_ci",
        "hac_tstat",
        "hac_pvalue",
    ),
    DimensionName.ROBUSTNESS: (
        "subsample_stability",
        "block_bootstrap_ci",
        "ic_std",
        "pearson_ic_std",
        "ic_autocorr_lag1",
        "half_life",
    ),
    DimensionName.TRADABILITY: (
        "turnover",
        "factor_turnover_rate",
    ),
    # PURITY_EXPOSURE: no exposure metrics are registered in the MetricRegistry
    # yet (industry/size/beta exposure, style concentration are not present).
    # A sibling owns further registry extension for exposure; until then this
    # dimension maps to no registered ids.
    DimensionName.PURITY_EXPOSURE: (),
    # DATA_QUALITY: only `coverage` is registered; missing_rate is not.
    DimensionName.DATA_QUALITY: (
        "coverage",
    ),
}


def _registered_ids() -> Tuple[str, ...]:
    """Return the registry's known metric ids (sorted, deduplicated)."""
    return tuple(sorted(set(list_metrics())))


def dimension_for(metric_id: str) -> Optional[DimensionName]:
    """Resolve a registry metric_id to its dimension.

    Fail-closed: an unknown metric_id (one the registry does not register)
    returns ``None`` rather than being silently bucketed.
    """
    if metric_id not in _registered_ids():
        return None
    for dimension, ids in DIMENSION_METRICS.items():
        if metric_id in ids:
            return dimension
    return None


def collect_dimension_metric_ids(
    dimension: DimensionName,
    registry: Optional[MetricRegistry] = None,
) -> Tuple[str, ...]:
    """Return the registered metric ids for ``dimension``.

    Resolves through the registry's known ids (``list_metrics`` by default, or
    ``registry.list_all_metric_ids()`` when a ``MetricRegistry`` instance is
    supplied) so only ids the registry actually knows are returned — never a
    metric the registry does not register.
    """
    if registry is not None:
        known = set(registry.list_all_metric_ids())
    else:
        known = set(_registered_ids())
    return tuple(
        sorted(
            metric_id
            for metric_id in DIMENSION_METRICS.get(dimension, ())
            if metric_id in known
        )
    )
