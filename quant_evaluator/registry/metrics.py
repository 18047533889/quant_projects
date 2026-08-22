"""
Metric specification catalog.

Central registry of available metrics with status and tier metadata.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional
import copyreg

from quant_evaluator.metrics.ic import compute_ic_std, compute_mean_ic_value
from quant_evaluator.metrics.registry_adapters import (
    compute_block_bootstrap_ci_value,
    compute_coverage_value,
    compute_hac_pvalue_value,
    compute_hac_tstat_value,
    compute_half_life_value,
    compute_factor_turnover_rate_value,
    compute_ic_autocorr_lag1_value,
    compute_ic_ir_value,
    compute_ic_median_value,
    compute_pearson_ic_series_value,
    compute_pearson_ic_value,
    compute_quantile_returns_full_value,
    compute_quantile_spread_value,
    compute_rank_ic_series_value,
    compute_rank_ic_value,
    compute_rank_stability_value,
    compute_subsample_stability_value,
    compute_turnover_value,
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
        requires: List of required input types (e.g.,
            ["ICSeriesArtifact", "factor_batch", "label_bundle"]). Derived
            inputs reference the formal artifact-type class names from
            ``quant_evaluator.contracts.artifact_types``.
        min_periods: Minimum time periods required (None if not applicable)
        ic_method: Correlation method ("pearson" or "spearman") the metric's
            ``ICSeriesArtifact`` input must be computed with. Only meaningful
            for metrics with ``ICSeriesArtifact`` in ``requires``; the public
            facade reads it to build the wrapper IC series. Defaults to
            "pearson".
    """
    name: str
    display_name: str
    description: str
    status: MetricStatus
    tier: MetricTier
    compute_fn: Optional[Callable] = None
    requires: Optional[List[str]] = None
    min_periods: Optional[int] = None
    ic_method: str = "pearson"
    metric_version: str = "1"

    def __post_init__(self):
        if self.requires is None:
            object.__setattr__(self, 'requires', [])
        if self.ic_method not in ("pearson", "spearman"):
            raise ValueError(
                f"Metric '{self.name}' declares invalid ic_method "
                f"{self.ic_method!r}; must be 'pearson' or 'spearman'"
            )
        if not isinstance(self.metric_version, str) or not self.metric_version.strip():
            raise ValueError(
                f"Metric '{self.name}' declares invalid metric_version "
                f"{self.metric_version!r}; must be a non-empty string"
            )


# Central metric catalog (QE-P0-07): sealed registry.
#
# Lifecycle: BUILDING -> SEALED.
#   - BUILDING: ``register_metric`` accepts new MetricSpecs.  The live read
#     surface (``get_metric`` / ``list_metrics`` / ``list_metrics_by_status`` /
#     ``list_metrics_by_tier``) already goes through a read-only
#     ``MappingProxyType`` view of the backing dict, so no caller can mutate
#     the catalog even before the seal.
#   - SEALED: ``seal_metric_registry()`` freezes the backing dict by replacing
#     it with a plain dict copy (a mappingproxy is NOT picklable, and the
#     registry must stay picklable) and flips the state flag.  Any further
#     registration raises ``RuntimeError`` (fail closed) — the catalog is the
#     single source of truth for production metrics and must not grow after
#     seal.
#   - Duplicate ``name`` registration always raises ``ValueError`` (before the
#     seal) or ``RuntimeError`` (after the seal).  There is no silent
#     overwrite path.
_REGISTRY_STATE_BUILDING = "building"
_REGISTRY_STATE_SEALED = "sealed"

_REGISTRY_STATE: str = _REGISTRY_STATE_BUILDING
_METRIC_CATALOG: Dict[str, MetricSpec] = {}
_METRIC_CATALOG_VIEW: Mapping[str, MetricSpec] = MappingProxyType(_METRIC_CATALOG)


def registry_state() -> str:
    """Return the current registry lifecycle state: "building" or "sealed"."""
    return _REGISTRY_STATE


def seal_metric_registry() -> str:
    """Seal the metric registry against further mutation.

    Replaces the backing dict with an immutable plain-dict snapshot (so the
    registry remains picklable — a ``MappingProxyType`` is not) and flips the
    state to ``"sealed"``.  After sealing, ``register_metric`` raises
    ``RuntimeError``.

    Returns:
        The new registry state (``"sealed"``).
    """
    global _REGISTRY_STATE, _METRIC_CATALOG, _METRIC_CATALOG_VIEW
    _METRIC_CATALOG = dict(_METRIC_CATALOG)
    _METRIC_CATALOG_VIEW = MappingProxyType(_METRIC_CATALOG)
    _REGISTRY_STATE = _REGISTRY_STATE_SEALED
    return _REGISTRY_STATE


def register_metric(spec: MetricSpec) -> None:
    """
    Register a metric specification.

    Args:
        spec: MetricSpec to register

    Raises:
        ValueError: If metric name already registered, or if a STABLE metric
            has no ``compute_fn`` (a STABLE claim without a bound
            implementation is a fail-open capability lie).
        RuntimeError: If the registry has been sealed (QE-P0-07).
    """
    if spec.status is MetricStatus.STABLE and spec.compute_fn is None:
        raise ValueError(
            f"Metric '{spec.name}' claims STABLE but has no compute_fn; "
            "register it as EXPERIMENTAL until an implementation is bound"
        )
    if _REGISTRY_STATE == _REGISTRY_STATE_SEALED:
        raise RuntimeError(
            f"Cannot register metric '{spec.name}': the metric registry is "
            "sealed (QE-P0-07). Registry changes must be added before "
            "seal_metric_registry() is called."
        )
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
    if name not in _METRIC_CATALOG_VIEW:
        raise KeyError(f"Metric '{name}' not found in registry")
    return _METRIC_CATALOG_VIEW[name]


def list_metrics() -> List[str]:
    """
    List all registered metric names.

    Returns:
        Sorted list of metric names
    """
    return sorted(_METRIC_CATALOG_VIEW.keys())


def list_metrics_by_status(status: MetricStatus) -> List[str]:
    """
    List metrics filtered by status.

    Args:
        status: MetricStatus to filter by

    Returns:
        Sorted list of metric names
    """
    return sorted(
        name for name, spec in _METRIC_CATALOG_VIEW.items()
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
        name for name, spec in _METRIC_CATALOG_VIEW.items()
        if spec.tier == tier
    )


def catalog_snapshot() -> Mapping[str, MetricSpec]:
    """Return the current catalog as an immutable mapping.

    Always a fresh read-only snapshot: pre-seal it is a live view of the
    backing dict, post-seal it is a copy of the sealed dict.  Either way the
    caller receives a read-only mapping that cannot mutate the registry.
    """
    return _METRIC_CATALOG_VIEW


def resolve_alias(metric_id: str) -> str:
    """
    Resolve a canonical dotted metric name to its registry name.

    Canonical dotted names (e.g. ``"ic.pearson.mean"``) map onto registry
    names (e.g. ``"mean_ic"``). Unknown names are returned unchanged so the
    caller can surface the original identifier in its own error.

    Args:
        metric_id: Registry name or canonical dotted alias

    Returns:
        The registry metric name
    """
    return CANONICAL_METRIC_ALIASES.get(metric_id, metric_id)


# QE-METRIC-P0-03: canonical dotted metric namespace. Frozen single source
# of truth for dotted-name -> registry-name resolution. rank_ic has exactly
# ONE meaning: the time-mean of daily Spearman IC (see its MetricSpec).
#
# QE-P0: mappingproxy is NOT picklable. Any object that embeds this mapping
# (e.g. an evaluation result carrying the canonical aliases) must survive
# pickle. We therefore keep the live read API as a read-only MappingProxyType
# backed by a plain dict, and teach the wrapper's pickle protocol (via
# __reduce__ + a module-level reconstruction factory) to convert to the plain
# dict on the way out and restore the read-only wrapper on the way back, so
# read-only semantics are preserved everywhere.
_CANONICAL_METRIC_ALIASES_DATA: Dict[str, str] = {
    "ic.rank.daily": "rank_ic_series",
    "ic.rank.mean": "rank_ic",
    "ic.rank.median": "ic_median",
    "ic.rank.std": "ic_std",
    "ic.rank.ir": "ic_ir",
    "ic.rank.hac_t": "hac_tstat",
    "ic.rank.hac_p": "hac_pvalue",
    "ic.pearson.daily": "pearson_ic_series",
    "ic.pearson.mean": "pearson_ic",
    "ic.pearson.std": "pearson_ic_series",
    "ic.pearson.ir": "pearson_ic_series",
}
CANONICAL_METRIC_ALIASES: Dict[str, str] = MappingProxyType(_CANONICAL_METRIC_ALIASES_DATA)


def _pickle_aliases(mapping: MappingProxyType) -> Any:
    """Pickle reducer for the canonical alias mappingproxy.

    ``mappingproxy`` itself cannot be subclassed or assigned a reducer, so we
    register ``copyreg.pickle`` to route every mappingproxy through a
    module-level reconstruction factory. This keeps the alias mapping
    picklable wherever it is embedded (registry object, evaluation result,
    report payload) while preserving its read-only semantics after the
    round-trip.
    """
    return _reconstruct_read_only_aliases, (dict(mapping),)


def _reconstruct_read_only_aliases(state: Dict[str, str]) -> MappingProxyType:
    """Reconstruct a read-only alias mapping from plain-dict pickle state."""
    return MappingProxyType(state)


copyreg.pickle(MappingProxyType, _pickle_aliases)


# Register core metrics
register_metric(MetricSpec(
    name="rank_ic",
    display_name="Mean Rank IC",
    description=(
        "rank_ic has exactly ONE meaning: the time-mean of daily Spearman "
        "rank IC between factor values and labels (canonical alias "
        "ic.rank.mean)"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_rank_ic_value,
    requires=["factor_batch", "label_bundle"],
    min_periods=None,
    ic_method="spearman",
))

register_metric(MetricSpec(
    name="ic_std",
    display_name="IC Standard Deviation",
    description=(
        "Standard deviation of the daily IC series per factor (canonical "
        "aliases ic.rank.std and ic.pearson.std). The alias namespace is "
        "Spearman-family, so the spec declares ic_method='spearman'; the "
        "same kernel serves the Pearson alias."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_ic_std,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
))

register_metric(MetricSpec(
    name="ic_ir",
    display_name="IC Information Ratio",
    description=(
        "Mean IC divided by IC standard deviation per factor (canonical "
        "aliases ic.rank.ir and ic.pearson.ir). The alias namespace is "
        "Spearman-family, so the spec declares ic_method='spearman'; the "
        "same kernel serves the Pearson alias."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_ic_ir_value,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
))

register_metric(MetricSpec(
    name="mean_ic",
    display_name="Mean IC",
    description=(
        "Time-averaged Pearson information coefficient: the time-mean of "
        "daily Pearson IC between factor values and labels (canonical alias "
        "ic.pearson.mean). NOTE on observation_count: the public evaluate "
        "facade reports the number of jointly valid (factor, label) panel "
        "cells for this alias, whereas pearson_ic/ic.pearson.mean report "
        "the number of finite daily IC days — same kernel, two documented "
        "observation bases (kept for back-compat with the historical "
        "facade behaviour)"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_mean_ic_value,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="pearson",
))

register_metric(MetricSpec(
    name="coverage",
    display_name="Coverage Rate",
    description=(
        "Per-factor fraction of the (T, N) panel with jointly valid factor "
        "and label values (never averaged across factor columns; canonical "
        "family: coverage)"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_coverage_value,
    requires=["factor_batch", "label_bundle"],
    min_periods=None,
))

register_metric(MetricSpec(
    name="pearson_ic",
    display_name="Mean Pearson IC",
    description=(
        "Time-mean of daily Pearson IC between factor values and labels "
        "(canonical alias ic.pearson.mean; same kernel as mean_ic). "
        "observation_count here is the number of finite daily IC days — "
        "whereas the mean_ic alias reports jointly valid (factor, label) "
        "panel cells; see the mean_ic spec note"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_pearson_ic_value,
    requires=["factor_batch", "label_bundle"],
    min_periods=None,
    ic_method="pearson",
))

register_metric(MetricSpec(
    name="pearson_ic_series",
    display_name="Daily Pearson IC Series",
    description=(
        "Daily Pearson IC per factor over time, shape (T, F) (canonical "
        "alias ic.pearson.daily)"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_pearson_ic_series_value,
    requires=["factor_batch", "label_bundle"],
    min_periods=None,
    ic_method="pearson",
))

register_metric(MetricSpec(
    name="rank_ic_series",
    display_name="Daily Rank IC Series",
    description=(
        "Daily Spearman rank IC per factor over time, shape (T, F) "
        "(canonical alias ic.rank.daily)"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rank_ic_series_value,
    requires=["factor_batch", "label_bundle"],
    min_periods=None,
    ic_method="spearman",
))

register_metric(MetricSpec(
    name="ic_median",
    display_name="Median IC",
    description=(
        "Time-median of the daily IC series per factor (canonical alias "
        "ic.rank.median)"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_median_value,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
))

register_metric(MetricSpec(
    name="hac_pvalue",
    display_name="HAC p-value",
    description=(
        "Two-sided HAC-robust p-value for mean(IC) != 0 per factor "
        "(canonical alias ic.rank.hac_p)"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_hac_pvalue_value,
    requires=["ICSeriesArtifact"],
    min_periods=30,
    ic_method="spearman",
))

register_metric(MetricSpec(
    name="turnover",
    display_name="Portfolio Turnover",
    description="Average turnover rate for factor-based portfolios",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_turnover_value,
    requires=["factor_batch"],
    min_periods=2,
))

register_metric(MetricSpec(
    name="quantile_spread",
    display_name="Top-Bottom Quantile Spread",
    description="Return spread between top and bottom quantiles",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_quantile_spread_value,
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
    compute_fn=compute_hac_tstat_value,
    requires=["ICSeriesArtifact"],
    min_periods=30,
    ic_method="spearman",
))

register_metric(MetricSpec(
    name="subsample_stability",
    display_name="Subsample IC Stability",
    description="Standard deviation of mean IC across bootstrap subsamples",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_subsample_stability_value,
    requires=["ICSeriesArtifact"],
    min_periods=40,
))

register_metric(MetricSpec(
    name="ic_autocorr_lag1",
    display_name="IC Autocorrelation (Lag 1)",
    description="First-order autocorrelation of IC series",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_autocorr_lag1_value,
    requires=["ICSeriesArtifact"],
    min_periods=30,
))

register_metric(MetricSpec(
    name="rank_stability",
    display_name="Rank Stability",
    description="Spearman correlation of factor ranks across time",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rank_stability_value,
    requires=["factor_batch"],
    min_periods=20,
))

register_metric(MetricSpec(
    name="half_life",
    display_name="IC Half-Life",
    description="Estimated half-life of IC decay via AR(1)",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_half_life_value,
    requires=["ICSeriesArtifact"],
    min_periods=60,
))

# Research metrics
register_metric(MetricSpec(
    name="block_bootstrap_ci",
    display_name="Block Bootstrap Confidence Interval",
    description="95% confidence interval half-width for mean IC via block bootstrap",
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=compute_block_bootstrap_ci_value,
    requires=["ICSeriesArtifact"],
    min_periods=60,
))

register_metric(MetricSpec(
    name="factor_turnover_rate",
    display_name="Factor Turnover Rate",
    description="Turnover rate of top/bottom quantile membership",
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=compute_factor_turnover_rate_value,
    requires=["factor_batch"],
    min_periods=30,
))

register_metric(MetricSpec(
    name="quantile_returns_full",
    display_name="Full Quantile Returns",
    description=(
        "Per-quantile time-averaged returns as a VECTOR per factor — shape "
        "(n_quantiles, F), NOT a scalar; wrap with "
        "metrics.registry_adapters.compute_quantile_returns_full_artifact "
        "for the typed VectorMetricArtifact"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=compute_quantile_returns_full_value,
    requires=["factor_batch", "label_bundle"],
    min_periods=20,
))
