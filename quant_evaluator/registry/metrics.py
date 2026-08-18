"""
Metric specification catalog.

Central registry of available metrics with status and tier metadata.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

from quant_evaluator.metrics.ic import compute_ic_std, compute_mean_ic_value
from quant_evaluator.metrics.quality import compute_coverage
from quant_evaluator.metrics.registry_adapters import (
    compute_hac_tstat_value,
    compute_half_life_value,
    compute_ic_autocorr_lag1_value,
    compute_ic_ir_value,
    compute_rank_stability_value,
)


class MetricStatus(Enum):
    """Metric implementation and validation status."""
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


class MetricTier(Enum):
    """Metric importance tier for production workflows."""
    CORE = "core"
    EXTENDED = "extended"
    RESEARCH = "research"


@dataclass(frozen=True)
class MetricSpec:
    """
    Specification for a registered metric.

    Attributes:
        name: Unique metric identifier
        display_name: Human-readable name
        description: Brief description of what the metric measures
        status: Implementation status
        tier: Importance tier
        compute_fn: Callable that computes the metric (optional)
        requires: List of required input types (e.g., ["ic_series", "factor_batch"])
        min_periods: Minimum time periods required (None if not applicable)
    """
    name: str
    display_name: str
    description: str
    status: MetricStatus
    tier: MetricTier
    compute_fn: Optional[Callable] = None
    requires: Optional[List[str]] = None
    min_periods: Optional[int] = None

    def __post_init__(self):
        if self.requires is None:
            object.__setattr__(self, 'requires', [])


# Central metric catalog
_METRIC_CATALOG: Dict[str, MetricSpec] = {}


def register_metric(spec: MetricSpec) -> None:
    """
    Register a metric specification.

    Args:
        spec: MetricSpec to register

    Raises:
        ValueError: If metric name already registered
    """
    if spec.name in _METRIC_CATALOG:
        raise ValueError(f"Metric '{spec.name}' already registered")
    _METRIC_CATALOG[spec.name] = spec


def get_metric(name: str) -> MetricSpec:
    """
    Retrieve metric specification by name.

    Args:
        name: Metric identifier

    Returns:
        MetricSpec for the requested metric

    Raises:
        KeyError: If metric not found
    """
    if name not in _METRIC_CATALOG:
        raise KeyError(f"Metric '{name}' not found in registry")
    return _METRIC_CATALOG[name]


def list_metrics() -> List[str]:
    """
    List all registered metric names.

    Returns:
        Sorted list of metric names
    """
    return sorted(_METRIC_CATALOG.keys())


def list_metrics_by_status(status: MetricStatus) -> List[str]:
    """
    List metrics filtered by status.

    Args:
        status: MetricStatus to filter by

    Returns:
        Sorted list of metric names
    """
    return sorted(
        name for name, spec in _METRIC_CATALOG.items()
        if spec.status == status
    )


def list_metrics_by_tier(tier: MetricTier) -> List[str]:
    """
    List metrics filtered by tier.

    Args:
        tier: MetricTier to filter by

    Returns:
        Sorted list of metric names
    """
    return sorted(
        name for name, spec in _METRIC_CATALOG.items()
        if spec.tier == tier
    )


# Register core metrics
register_metric(MetricSpec(
    name="mean_ic",
    display_name="Mean IC",
    description="Time-averaged information coefficient",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_mean_ic_value,
    requires=["ic_series"],
    min_periods=20,
))

register_metric(MetricSpec(
    name="ic_std",
    display_name="IC Standard Deviation",
    description="Standard deviation of IC series",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_ic_std,
    requires=["ic_series"],
    min_periods=20,
))

register_metric(MetricSpec(
    name="ic_ir",
    display_name="IC Information Ratio",
    description="Mean IC divided by IC standard deviation",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_ic_ir_value,
    requires=["ic_series"],
    min_periods=20,
))

register_metric(MetricSpec(
    name="coverage",
    display_name="Coverage Rate",
    description="Fraction of universe with valid factor values",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_coverage,
    requires=["factor_batch", "label_bundle"],
    min_periods=None,
))

register_metric(MetricSpec(
    name="turnover",
    display_name="Portfolio Turnover",
    description="Average turnover rate for factor-based portfolios",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    requires=["factor_batch"],
    min_periods=2,
))

register_metric(MetricSpec(
    name="quantile_spread",
    display_name="Top-Bottom Quantile Spread",
    description="Return spread between top and bottom quantiles",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    requires=["factor_batch", "label_bundle"],
    min_periods=20,
))

# Extended metrics
register_metric(MetricSpec(
    name="hac_tstat",
    display_name="HAC t-statistic",
    description="Heteroskedasticity and autocorrelation consistent t-statistic for IC",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    requires=["ic_series"],
    min_periods=30,
))

register_metric(MetricSpec(
    name="subsample_stability",
    display_name="Subsample IC Stability",
    description="Standard deviation of mean IC across bootstrap subsamples",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    requires=["ic_series"],
    min_periods=40,
))

register_metric(MetricSpec(
    name="ic_autocorr_lag1",
    display_name="IC Autocorrelation (Lag 1)",
    description="First-order autocorrelation of IC series",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    requires=["ic_series"],
    min_periods=30,
))

register_metric(MetricSpec(
    name="rank_stability",
    display_name="Rank Stability",
    description="Spearman correlation of factor ranks across time",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    requires=["factor_batch"],
    min_periods=20,
))

register_metric(MetricSpec(
    name="half_life",
    display_name="IC Half-Life",
    description="Estimated half-life of IC decay via AR(1)",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    requires=["ic_series"],
    min_periods=60,
))

# Research metrics
register_metric(MetricSpec(
    name="block_bootstrap_ci",
    display_name="Block Bootstrap Confidence Interval",
    description="95% confidence interval for mean IC via block bootstrap",
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.RESEARCH,
    requires=["ic_series"],
    min_periods=60,
))

register_metric(MetricSpec(
    name="factor_turnover_rate",
    display_name="Factor Turnover Rate",
    description="Turnover rate of top/bottom quantile membership",
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.RESEARCH,
    requires=["factor_batch"],
    min_periods=30,
))

register_metric(MetricSpec(
    name="quantile_returns_full",
    display_name="Full Quantile Returns",
    description="Return distribution across all quantiles",
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    requires=["factor_batch", "label_bundle"],
    min_periods=20,
))
