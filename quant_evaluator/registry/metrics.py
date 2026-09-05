"""
Metric specification catalog.

Central registry of available metrics with status and tier metadata.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple
import copyreg
import hashlib
import inspect

import numpy as np

from quant_evaluator.metrics.ic import compute_ic_std, compute_mean_ic_value
from quant_evaluator.metrics.predictive import (
    compute_ic_positive_ratio,
    compute_ic_recent_vs_history_delta,
    compute_ic_sign_consistency,
    compute_monthly_rank_ic,
    compute_quarterly_rank_ic,
    compute_rank_ic_decay,
    compute_rank_ic_positive_ratio,
    compute_recent_12m_rank_ic,
    compute_recent_3m_rank_ic,
    compute_recent_6m_rank_ic,
    compute_rolling_rank_ic_ir,
    compute_rolling_rank_ic_mean,
    compute_worst_quarter_rank_ic,
    compute_worst_year_rank_ic,
    compute_yearly_rank_ic,
)
from quant_evaluator.metrics.quantile_shape import (
    compute_bottom_quantile_cliff,
    compute_quantile_adjacent_spread,
    compute_quantile_curvature,
    compute_quantile_extreme_cliff,
    compute_quantile_monotonicity,
    compute_quantile_tail_asymmetry,
    compute_top_quantile_cliff,
)
from quant_evaluator.metrics.shape_evidence import (
    compute_adaptive_quantile_count,
    compute_bottom_quantile_cliff_robust,
    compute_bottom_tail_slope,
    compute_inverted_u_score,
    compute_left_right_asymmetry,
    compute_linear_trend_score,
    compute_shape_bootstrap_confidence,
    compute_shape_regime_stability,
    compute_shape_stability,
    compute_tail_vs_middle_contrast,
    compute_top_quantile_cliff_robust,
    compute_top_tail_slope,
    compute_u_shape_score,
)
from quant_evaluator.metrics.stability_regime import (
    compute_change_point_score,
    compute_cusum_break_score,
    compute_ic_sign_flip_rate,
    compute_month_consistency,
    compute_quarter_consistency,
    compute_recent_degradation_score,
    compute_regime_conditional_ic,
    compute_regime_dispersion,
    compute_regime_sign_consistency,
    compute_regime_worst_ic,
    compute_rolling_ic_drawdown,
    compute_rolling_ic_volatility,
    compute_year_consistency,
)
from quant_evaluator.metrics.data_quality import (
    compute_cross_section_cardinality,
    compute_distinct_level_ratio,
    compute_effective_n,
    compute_label_maturity,
    compute_missing_ratio,
    compute_missing_timeline,
    compute_outlier_ratio,
    compute_staleness,
    compute_tie_ratio,
    compute_tradable_coverage,
    compute_universe_churn,
)
from quant_evaluator.metrics.multiple_testing import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    holm_bonferroni_correction,
    sidak_correction,
)
from quant_evaluator.metrics.portfolio_stats import (
    compute_long_short_returns,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_win_rate,
)
from quant_evaluator.metrics.underwater import (
    compute_max_underwater_duration,
    compute_mean_underwater_duration,
    compute_time_to_recovery,
    compute_worst_period_return,
    compute_rolling_sharpe_tail,
    compute_return_skew,
    compute_downside_deviation,
    compute_cvar_expected_shortfall,
)
from quant_evaluator.metrics.exposure_evidence import (
    compute_style_exposure_evidence,
    compute_max_absolute_style_exposure,
    compute_exposure_drift,
    compute_purity_ratio,
    compute_neutralized_rank_ic,
    compute_residual_rank_ic,
    compute_industry_exposure,
    compute_size_exposure,
    compute_beta_exposure,
    compute_liquidity_exposure,
    compute_volatility_exposure,
    compute_momentum_exposure,
)
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


class Domain(Enum):
    """Metric evaluation domains (QE-P0-01: single authority)."""
    IC = "ic"
    RANK_IC = "rank_ic"
    QUANTILE = "quantile"
    DRAWDOWN = "drawdown"
    TURNOVER = "turnover"
    TAIL_RISK = "tail_risk"
    COVERAGE = "coverage"
    HHI = "hhi"
    STABILITY = "stability"
    TEMPORAL = "temporal"
    PREDICTIVE = "predictive"
    QUANTILE_SHAPE = "quantile_shape"
    REGIME = "regime"
    DATA_QUALITY = "data_quality"
    RESEARCH_INTEGRITY = "research_integrity"
    UNDERWATER = "underwater"
    EXPOSURE = "exposure"


class MetricTier(Enum):
    """Metric importance tier for production workflows."""
    CORE = "core"
    EXTENDED = "extended"
    RESEARCH = "research"


# QE-P0-01: catalog-style helpers merged into the single authority.
_OUTPUT_TYPE_TO_ARTIFACT_KIND: Dict[str, str] = {
    "scalar": "scalar",
    "series": "series",
    "timeseries": "series",
    "vector": "vector",
    "matrix": "matrix",
    "distribution": "distribution",
}


def _content_hash(identity: str) -> str:
    """Return a stable 16-hex content hash of an implementation identity."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _source_implementation_hash(implementation_id: str) -> str:
    """Return a 16-hex hash that reflects the ACTUAL kernel source.

    QE-P0-03: the implementation identity must be a real fingerprint of the
    implementation, not a hash of the id string.  We resolve the dotted
    ``module.attr`` and hash the module source (plus, when the attribute is a
    function, its source closure).  If the module/attr cannot be resolved we
    fall back to the id-string hash so the field is always populated, but any
    resolvable kernel is source-addressed: editing the body flips the hash.
    """
    if not implementation_id or "." not in implementation_id:
        return _content_hash(implementation_id)
    module_path, _, attr = implementation_id.rpartition(".")
    try:
        module = __import__(module_path, fromlist=[attr])
    except Exception:  # noqa: BLE001 - fall back to id-string hash
        return _content_hash(implementation_id)
    try:
        module_src = inspect.getsource(module)
    except (OSError, TypeError):
        module_src = ""
    fn_src = ""
    obj = getattr(module, attr, None)
    if obj is not None:
        try:
            fn_src = inspect.getsource(obj)
        except (OSError, TypeError):
            fn_src = ""
    return hashlib.sha256(
        f"{module_path}.{attr}\x00{module_src}\x00{fn_src}".encode("utf-8")
    ).hexdigest()[:16]


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
    name: str = ""
    display_name: str = ""
    description: str = ""
    status: MetricStatus = MetricStatus.EXPERIMENTAL
    tier: MetricTier = MetricTier.EXTENDED
    compute_fn: Optional[Callable] = None
    requires: Optional[List[str]] = None
    min_periods: Optional[int] = None
    ic_method: str = "pearson"
    metric_version: str = "0.1.0"
    # QE-P0-01: catalog-style fields merged into the single MetricSpec so the
    # metrics.catalog layer is a read-only view over this authority, not a
    # second definition.  Defaults keep every existing registration valid.
    domain: Optional[Any] = None
    metric_id: str = ""
    required_inputs: Optional[Set[str]] = None
    output_type: str = "scalar"
    implementation_id: str = ""
    implementation_hash: str = ""
    artifact_kind: str = "scalar"
    required_axes: Tuple[str, ...] = ()
    units: str = ""
    direction: str = "higher_is_better"
    missing_policy: str = "nan"
    numeric_policy: str = "finite"

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
        # QE-P0-01: catalog-style normalization (mirrors the old catalog
        # MetricSpec.__post_init__ so the merged class keeps both contracts).
        if self.required_inputs is not None:
            object.__setattr__(self, "required_inputs", frozenset(self.required_inputs))
        if not self.metric_id:
            object.__setattr__(self, "metric_id", self.name)
        if not self.name:
            object.__setattr__(self, "name", self.metric_id)
        if not self.display_name:
            object.__setattr__(self, "display_name", self.name)
        if not self.implementation_id:
            object.__setattr__(self, "implementation_id", self.metric_id)
        if not self.implementation_hash:
            object.__setattr__(
                self,
                "implementation_hash",
                _source_implementation_hash(self.implementation_id),
            )
        if self.required_axes is not None and not isinstance(self.required_axes, tuple):
            raise ValueError("MetricSpec.required_axes must be a tuple")
        for axis in (self.required_axes or ()):
            if not isinstance(axis, str) or not axis.strip():
                raise ValueError(
                    "MetricSpec.required_axes entries must be non-empty strings"
                )
        if not isinstance(self.metric_id, str) or not self.metric_id.strip():
            raise ValueError("MetricSpec.metric_id must be a non-empty string")
        if self.direction not in ("higher_is_better", "lower_is_better", "neutral"):
            raise ValueError(
                f"MetricSpec.direction must be one of "
                f"{sorted(('higher_is_better', 'lower_is_better', 'neutral'))}, "
                f"got {self.direction!r}"
            )
        if self.artifact_kind not in (
            "scalar", "series", "vector", "matrix", "distribution"
        ):
            raise ValueError(
                f"MetricSpec.artifact_kind must be one of "
                f"{sorted(('scalar', 'series', 'vector', 'matrix', 'distribution'))}, "
                f"got {self.artifact_kind!r}"
            )
        if self.artifact_kind == "scalar" and self.output_type != "scalar":
            mapped = _OUTPUT_TYPE_TO_ARTIFACT_KIND.get(self.output_type)
            if mapped is not None and mapped != "scalar":
                object.__setattr__(self, "artifact_kind", mapped)


class MetricRegistry:
    """Sealed metric registry with a BUILDING -> SEALED lifecycle.

    QE-P0-01: this is the SINGLE MetricRegistry authority.  The
    ``metrics.catalog`` layer is a read-only view over it (or over a fresh
    instance for standalone use), never a second definition.

    During BUILDING, ``register(spec)`` accepts distinct metric_ids and raises
    ``ValueError`` on duplicates.  ``seal()`` freezes the backing mapping into
    a ``MappingProxyType``; after sealing, ``register`` fails closed and all
    reads are served from the immutable view.
    """

    _BUILDING = "building"
    _SEALED = "sealed"

    def __init__(self) -> None:
        self._catalog: Dict[str, MetricSpec] = {}
        self._sealed: bool = False

    @property
    def sealed(self) -> bool:
        """True once the registry has been sealed."""
        return self._sealed

    def _check_sealed(self) -> None:
        if self._sealed:
            raise RuntimeError(
                "MetricRegistry is SEALED: no further registrations are allowed."
            )

    def register(self, spec: MetricSpec) -> None:
        """Register a MetricSpec during the BUILDING phase."""
        self._check_sealed()
        if not isinstance(spec, MetricSpec):
            raise TypeError(f"register expects MetricSpec, got {type(spec).__name__}")
        if spec.metric_id in self._catalog:
            raise ValueError(
                f"Duplicate metric_id {spec.metric_id!r}: "
                "MetricRegistry does not allow silent overwrites."
            )
        self._catalog[spec.metric_id] = spec

    def seal(self) -> None:
        """Freeze the registry into an immutable MappingProxyType view."""
        self._catalog = dict(self._catalog)
        self._catalog = MappingProxyType(self._catalog)  # type: ignore[assignment]
        self._sealed = True

    def __len__(self) -> int:
        return len(self._catalog)

    def __contains__(self, metric_id: object) -> bool:
        return metric_id in self._catalog

    def get(self, metric_id: str) -> MetricSpec:
        """Return the spec for ``metric_id``; unknown IDs raise UnsupportedMetricError."""
        from quant_evaluator.contracts.errors import UnsupportedMetricError

        try:
            return self._catalog[metric_id]
        except KeyError:
            raise UnsupportedMetricError(
                f"Unknown metric_id '{metric_id}'. "
                f"Valid IDs: {sorted(self._catalog.keys())}"
            )

    def get_metric_spec(self, metric_id: str) -> MetricSpec:
        """Alias for ``get`` (read-only after sealing)."""
        return self.get(metric_id)

    def list_all_metric_ids(self) -> List[str]:
        """Return a sorted list of all registered metric IDs."""
        return sorted(self._catalog.keys())

    def get_metric_specs_by_domain(self, domain: Domain) -> List[MetricSpec]:
        """Return all MetricSpec entries for a Domain, ordered by metric_id."""
        return sorted(
            (spec for spec in self._catalog.values() if spec.domain == domain),
            key=lambda s: s.metric_id,
        )

    def list_all_domains(self) -> List[Domain]:
        """Return a sorted list of all Domain enum values."""
        return sorted(Domain, key=lambda d: d.value)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return a plain-dict snapshot of the registry contents."""
        return {metric_id: spec for metric_id, spec in self._catalog.items()}


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

# QE-P0-R-C2: SINGLE populated metric authority.
#
# ``_REGISTRY`` is the ONE populated ``MetricRegistry`` instance in the whole
# package.  It is the canonical backing store for every metric spec.  The
# module-level functional API below (``register_metric`` / ``get_metric`` /
# ``list_metrics`` / ``list_metrics_by_status`` / ``list_metrics_by_tier`` /
# ``catalog_snapshot``) is a thin read-only facade over this single instance,
# and ``metrics.catalog`` is a read-only view over the SAME instance.  There
# is exactly ONE populated metric catalog at runtime — no dual authority.
#
# The 10-domain catalog specs (previously populated into a separate
# ``metrics.catalog._REGISTRY``) are merged here so the registry is the single
# source of truth for BOTH the runtime metric set and the 10-domain catalog.
_REGISTRY: MetricRegistry = MetricRegistry()


def registry_state() -> str:
    """Return the current registry lifecycle state: "building" or "sealed"."""
    return _REGISTRY_STATE


def seal_metric_registry() -> str:
    """Seal the metric registry against further mutation.

    Seals the SINGLE ``_REGISTRY`` instance (QE-P0-R-C2).  After sealing,
    ``register_metric`` raises ``RuntimeError``.

    Returns:
        The new registry state (``"sealed"``).
    """
    global _REGISTRY_STATE
    _REGISTRY.seal()
    _REGISTRY_STATE = _REGISTRY_STATE_SEALED
    return _REGISTRY_STATE


def register_metric(spec: MetricSpec) -> None:
    """
    Register a metric specification into the SINGLE registry authority.

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
    _REGISTRY.register(spec)


def get_metric(name: str) -> MetricSpec:
    """
    Retrieve metric specification by name from the SINGLE registry authority.

    Args:
        name: Metric identifier

    Returns:
        MetricSpec for the requested metric

    Raises:
        KeyError: If metric not found
    """
    try:
        return _REGISTRY.get(name)
    except Exception as exc:
        # Preserve the historical KeyError contract for unknown metrics.
        if isinstance(exc, KeyError):
            raise
        raise KeyError(f"Metric '{name}' not found in registry") from exc


def list_metrics() -> List[str]:
    """
    List all registered metric names from the SINGLE registry authority.

    Returns:
        Sorted list of metric names
    """
    return _REGISTRY.list_all_metric_ids()


def list_metrics_by_status(status: MetricStatus) -> List[str]:
    """
    List metrics filtered by status.

    Args:
        status: MetricStatus to filter by

    Returns:
        Sorted list of metric names
    """
    return sorted(
        name for name, spec in _REGISTRY.to_dict().items()
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
        name for name, spec in _REGISTRY.to_dict().items()
        if spec.tier == tier
    )


def catalog_snapshot() -> Mapping[str, MetricSpec]:
    """Return the current catalog as an immutable mapping.

    A read-only snapshot of the SINGLE registry authority.  Pre-seal it is a
    live ``MappingProxyType`` view of the backing dict; post-seal it is the
    sealed immutable view.  Either way the caller receives a read-only mapping
    that cannot mutate the registry.
    """
    if _REGISTRY.sealed:
        return _REGISTRY._catalog  # type: ignore[return-value]
    return MappingProxyType(_REGISTRY._catalog)  # type: ignore[arg-type]


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
    "ic.pearson.std": "pearson_ic_std",
    "ic.pearson.ir": "pearson_ic_ir",
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


# Register core metrics into the SINGLE registry authority (QE-P0-R-C2).
_REGISTRY.register(MetricSpec(
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
    # QE-P0-R-C2: rank_ic is also a 10-domain catalog metric (Domain.IC).
    domain=Domain.IC,
    metric_id="rank_ic",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
))

_REGISTRY.register(MetricSpec(
    name="ic_std",
    display_name="IC Standard Deviation",
    description=(
        "Standard deviation of the daily IC series per factor (canonical "
        "alias ic.rank.std). Spearman-family: the pearson.std alias is its "
        "own spec (pearson_ic_std)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_ic_std,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
))

_REGISTRY.register(MetricSpec(
    name="ic_ir",
    display_name="IC Information Ratio",
    description=(
        "Mean IC divided by IC standard deviation per factor (canonical "
        "alias ic.rank.ir). Spearman-family: the pearson.ir alias is its "
        "own spec (pearson_ic_ir)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_ic_ir_value,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
))

_REGISTRY.register(MetricSpec(
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
    metric_version="1",
))

_REGISTRY.register(MetricSpec(
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

_REGISTRY.register(MetricSpec(
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
    # QE-P0-R-C2: pearson_ic is also a 10-domain catalog metric (Domain.IC).
    domain=Domain.IC,
    metric_id="pearson_ic",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
))

_REGISTRY.register(MetricSpec(
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

_REGISTRY.register(MetricSpec(
    name="pearson_ic_std",
    display_name="Pearson IC Standard Deviation",
    description=(
        "Standard deviation of the daily Pearson IC series per factor "
        "(canonical alias ic.pearson.std). Fully separated from the "
        "Spearman-family ic_std (ic.rank.std)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_std,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="pearson",
))

_REGISTRY.register(MetricSpec(
    name="pearson_ic_ir",
    display_name="Pearson IC Information Ratio",
    description=(
        "Mean Pearson IC divided by Pearson IC standard deviation per factor "
        "(canonical alias ic.pearson.ir). Fully separated from the "
        "Spearman-family ic_ir (ic.rank.ir)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_ir_value,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="pearson",
))

_REGISTRY.register(MetricSpec(
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

_REGISTRY.register(MetricSpec(
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

_REGISTRY.register(MetricSpec(
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

_REGISTRY.register(MetricSpec(
    name="turnover",
    display_name="Portfolio Turnover",
    description="Average turnover rate for factor-based portfolios",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_turnover_value,
    requires=["factor_batch"],
    min_periods=2,
))

_REGISTRY.register(MetricSpec(
    name="quantile_spread",
    display_name="Top-Bottom Quantile Spread",
    description="Return spread between top and bottom quantiles",
    status=MetricStatus.STABLE,
    tier=MetricTier.CORE,
    compute_fn=compute_quantile_spread_value,
    requires=["factor_batch", "label_bundle"],
    min_periods=20,
    # QE-P0-R-C2: quantile_spread is also a 10-domain catalog metric
    # (Domain.QUANTILE).
    domain=Domain.QUANTILE,
    metric_id="quantile_spread",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.quantile.compute_top_bottom_spread",
    metric_version="1.0.0",
    units="return",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# Extended metrics
_REGISTRY.register(MetricSpec(
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

_REGISTRY.register(MetricSpec(
    name="subsample_stability",
    display_name="Subsample IC Stability",
    description="Standard deviation of mean IC across bootstrap subsamples",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_subsample_stability_value,
    requires=["ICSeriesArtifact"],
    min_periods=40,
))

_REGISTRY.register(MetricSpec(
    name="ic_autocorr_lag1",
    display_name="IC Autocorrelation (Lag 1)",
    description="First-order autocorrelation of IC series",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_autocorr_lag1_value,
    requires=["ICSeriesArtifact"],
    min_periods=30,
))

_REGISTRY.register(MetricSpec(
    name="rank_stability",
    display_name="Rank Stability",
    description="Spearman correlation of factor ranks across time",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rank_stability_value,
    requires=["factor_batch"],
    min_periods=20,
))

_REGISTRY.register(MetricSpec(
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
_REGISTRY.register(MetricSpec(
    name="block_bootstrap_ci",
    display_name="Block Bootstrap Confidence Interval",
    description="95% confidence interval half-width for mean IC via block bootstrap",
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=compute_block_bootstrap_ci_value,
    requires=["ICSeriesArtifact"],
    min_periods=60,
))

_REGISTRY.register(MetricSpec(
    name="factor_turnover_rate",
    display_name="Factor Turnover Rate",
    description="Turnover rate of top/bottom quantile membership",
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=compute_factor_turnover_rate_value,
    requires=["factor_batch"],
    min_periods=30,
))

_REGISTRY.register(MetricSpec(
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


# ---------------------------------------------------------------------------
# QE-P0-R-C2: 10-domain catalog specs merged into the SINGLE registry.
#
# These were previously populated into a SEPARATE ``metrics.catalog._REGISTRY``
# (a second populated catalog — the dual-authority smell).  They are now
# registered into the one ``_REGISTRY`` here so the registry is the single
# source of truth for BOTH the runtime metric set and the 10-domain catalog.
# ``metrics.catalog`` is a read-only view over this same instance.
#
# The three ids that already exist above (pearson_ic, rank_ic, quantile_spread)
# are intentionally NOT re-registered here — the registry forbids duplicates.
# ---------------------------------------------------------------------------

# ---- Domain IC ----
_REGISTRY.register(MetricSpec(
    name="spearman_ic",
    display_name="spearman_ic",
    description="Spearman rank correlation between factor values and forward returns",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.IC,
    metric_id="spearman_ic",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="ic_summary",
    display_name="ic_summary",
    description="Summary statistics (mean, std, skew, kurtosis) of IC time series",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.IC,
    metric_id="ic_summary",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic_summary.compute_rolling_ic_stats",
    metric_version="1.0.0",
    artifact_kind="distribution",
    required_axes=("time",),
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain RANK_IC ----
_REGISTRY.register(MetricSpec(
    name="rank_ic_time_series",
    display_name="rank_ic_time_series",
    description="Rank IC computed per time slice, returned as a time series",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.RANK_IC,
    metric_id="rank_ic_time_series",
    required_inputs={"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    required_axes=("time",),
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rank_ic_cross_section",
    display_name="rank_ic_cross_section",
    description="Rank IC computed cross-sectionally for each date",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.RANK_IC,
    metric_id="rank_ic_cross_section",
    required_inputs={"factor", "forward_returns"},
    output_type="series",
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    required_axes=("time",),
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
))

# ---- Domain QUANTILE ----
_REGISTRY.register(MetricSpec(
    name="quantile_returns",
    display_name="quantile_returns",
    description="Average forward return per quantile bucket",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.QUANTILE,
    metric_id="quantile_returns",
    required_inputs={"factor", "forward_returns"},
    output_type="series",
    implementation_id="quant_evaluator.metrics.quantile.compute_quantile_returns",
    metric_version="1.0.0",
    required_axes=("quantile",),
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="quantile_stability",
    display_name="quantile_stability",
    description="Stability of quantile return rankings across time",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.QUANTILE,
    metric_id="quantile_stability",
    required_inputs={"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.ic_summary.compute_ic_stability",
    metric_version="1.0.0",
    required_axes=("time",),
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain DRAWDOWN ----
_REGISTRY.register(MetricSpec(
    name="max_drawdown",
    display_name="max_drawdown",
    description="Maximum drawdown of the cumulative IC series",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.DRAWDOWN,
    metric_id="max_drawdown",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.portfolio_stats.compute_maximum_drawdown",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="drawdown_duration",
    display_name="drawdown_duration",
    description="Duration (in periods) of the longest drawdown",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.DRAWDOWN,
    metric_id="drawdown_duration",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.drawdown_analysis.compute_drawdown_duration",
    metric_version="1.0.0",
    units="periods",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="calmar_ratio",
    display_name="calmar_ratio",
    description="Calmar ratio: annualized return / max drawdown",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.DRAWDOWN,
    metric_id="calmar_ratio",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.portfolio_stats.compute_calmar_ratio",
    metric_version="1.0.0",
    units="ratio",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain TURNOVER ----
_REGISTRY.register(MetricSpec(
    name="turnover_rate",
    display_name="turnover_rate",
    description="Average rate of change in factor ranking between periods",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TURNOVER,
    metric_id="turnover_rate",
    required_inputs={"factor"},
    implementation_id="quant_evaluator.metrics.turnover.compute_turnover",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="turnover_cost",
    display_name="turnover_cost",
    description="Estimated transaction cost from factor rebalancing",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TURNOVER,
    metric_id="turnover_cost",
    required_inputs={"factor", "transaction_costs"},
    implementation_id="quant_evaluator.metrics.turnover.compute_weighted_turnover",
    metric_version="1.0.0",
    units="bps",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="turnover_adjusted_ic",
    display_name="turnover_adjusted_ic",
    description="IC adjusted for turnover-induced transaction costs",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TURNOVER,
    metric_id="turnover_adjusted_ic",
    required_inputs={"factor", "forward_returns", "transaction_costs"},
    implementation_id="quant_evaluator.metrics.turnover.compute_turnover_contribution",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain TAIL_RISK ----
_REGISTRY.register(MetricSpec(
    name="var_95",
    display_name="var_95",
    description="Value at Risk at 95% confidence level",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TAIL_RISK,
    metric_id="var_95",
    required_inputs={"forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.var_cvar.compute_var",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="var_99",
    display_name="var_99",
    description="Value at Risk at 99% confidence level",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TAIL_RISK,
    metric_id="var_99",
    required_inputs={"forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.var_cvar.compute_var",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="cvar_95",
    display_name="cvar_95",
    description="Conditional Value at Risk (Expected Shortfall) at 95%",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TAIL_RISK,
    metric_id="cvar_95",
    required_inputs={"forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.var_cvar.compute_cvar",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="cvar_99",
    display_name="cvar_99",
    description="Conditional Value at Risk (Expected Shortfall) at 99%",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TAIL_RISK,
    metric_id="cvar_99",
    required_inputs={"forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.var_cvar.compute_cvar",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="skewness",
    display_name="skewness",
    description="Skewness of the return distribution",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TAIL_RISK,
    metric_id="skewness",
    required_inputs={"forward_returns"},
    implementation_id="quant_evaluator.metrics.distribution.compute_skewness",
    metric_version="1.0.0",
    units="dimensionless",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="kurtosis",
    display_name="kurtosis",
    description="Excess kurtosis of the return distribution",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TAIL_RISK,
    metric_id="kurtosis",
    required_inputs={"forward_returns"},
    implementation_id="quant_evaluator.metrics.distribution.compute_kurtosis",
    metric_version="1.0.0",
    units="dimensionless",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain COVERAGE ----
_REGISTRY.register(MetricSpec(
    name="factor_coverage",
    display_name="factor_coverage",
    description="Fraction of universe with non-null factor values",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.COVERAGE,
    metric_id="factor_coverage",
    required_inputs={"factor"},
    implementation_id="quant_evaluator.metrics.quality.compute_coverage",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="return_coverage",
    display_name="return_coverage",
    description="Fraction of universe with non-null forward returns",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.COVERAGE,
    metric_id="return_coverage",
    required_inputs={"forward_returns"},
    implementation_id="quant_evaluator.metrics.quality.compute_coverage",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="joint_coverage",
    display_name="joint_coverage",
    description="Fraction of universe with both factor and return available",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.COVERAGE,
    metric_id="joint_coverage",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.quality.compute_coverage_per_factor",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain HHI ----
_REGISTRY.register(MetricSpec(
    name="hhi_concentration",
    display_name="hhi_concentration",
    description="Herfindahl-Hirschman Index of factor value concentration",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.HHI,
    metric_id="hhi_concentration",
    required_inputs={"factor"},
    implementation_id="quant_evaluator.metrics.exposure.compute_concentration_hhi",
    metric_version="1.0.0",
    units="dimensionless",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="hhi_effective_n",
    display_name="hhi_effective_n",
    description="Effective number of groups (1/HHI) for factor concentration",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.HHI,
    metric_id="hhi_effective_n",
    required_inputs={"factor"},
    implementation_id="quant_evaluator.metrics.exposure.compute_concentration_hhi",
    metric_version="1.0.0",
    units="count",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain STABILITY ----
_REGISTRY.register(MetricSpec(
    name="ic_stability",
    display_name="ic_stability",
    description="Rolling correlation of IC values across sub-periods",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.STABILITY,
    metric_id="ic_stability",
    required_inputs={"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic_summary.compute_ic_stability",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="turnover_stability",
    display_name="turnover_stability",
    description="Variance of turnover rate across periods",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.STABILITY,
    metric_id="turnover_stability",
    required_inputs={"factor"},
    implementation_id="quant_evaluator.metrics.turnover.compute_turnover",
    metric_version="1.0.0",
    units="variance",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="coverage_stability",
    display_name="coverage_stability",
    description="Variance of factor coverage across periods",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.STABILITY,
    metric_id="coverage_stability",
    required_inputs={"factor"},
    implementation_id="quant_evaluator.metrics.quality.compute_per_time_coverage",
    metric_version="1.0.0",
    units="variance",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain TEMPORAL ----
_REGISTRY.register(MetricSpec(
    name="rolling_ic",
    display_name="rolling_ic",
    description="Rolling window IC values over time",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TEMPORAL,
    metric_id="rolling_ic",
    required_inputs={"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.ic_summary.compute_rolling_ic_stats",
    metric_version="1.0.0",
    required_axes=("time",),
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="ic_decay",
    display_name="ic_decay",
    description="IC decay: correlation at increasing forward horizons",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TEMPORAL,
    metric_id="ic_decay",
    required_inputs={"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.ic_summary.compute_ic_decay",
    metric_version="1.0.0",
    required_axes=("horizon",),
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="autocorrelation_ic",
    display_name="autocorrelation_ic",
    description="Autocorrelation of IC values at specified lags",
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    domain=Domain.TEMPORAL,
    metric_id="autocorrelation_ic",
    required_inputs={"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.temporal.compute_ic_autocorrelation",
    metric_version="1.0.0",
    required_axes=("lag",),
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---------------------------------------------------------------------------
# QE-METRIC portfolio-statistics family (R2 AlphaPROBE integration).
#
# Long/short backtest and risk-adjusted-return metrics used by AlphaPROBE's
# research-probe profile.  All four delegate to the implementations in
# ``metrics/portfolio_stats.py`` (the single authority for these kernels) and
# consume a (T,) or (T, F) return series.  Registered at module import time,
# i.e. during the BUILDING phase, so the sealed-registry lifecycle is
# respected: they are visible to ``list_metrics()`` and ``get_metric()``, and
# any later seal freezes them along with the rest of the catalog.
# ---------------------------------------------------------------------------

# ---- Domain DRAWDOWN (returns-based) ----
_REGISTRY.register(MetricSpec(
    name="long_short_returns",
    display_name="long_short_returns",
    description=(
        "Time series of long-minus-short portfolio returns, shape (T,) or "
        "(T, F). Portfolio construction: long bucket = factor values above "
        "the long_threshold quantile (default 0.8), short bucket = values "
        "below short_threshold (default 0.2); equal-weight mean of forward "
        "returns per bucket per day, long minus short. Missing-return policy "
        "defaults to 'zero_fill' (NaN returns contribute 0)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_long_short_returns,
    domain=Domain.DRAWDOWN,
    metric_id="long_short_returns",
    required_inputs={"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.portfolio_stats.compute_long_short_returns",
    metric_version="1.0.0",
    required_axes=("time",),
    units="return",
    direction="higher_is_better",
    missing_policy="zero_fill",
    numeric_policy="finite",
))

_REGISTRY.register(MetricSpec(
    name="sharpe_ratio",
    display_name="sharpe_ratio",
    description=(
        "Annualized Sharpe ratio of a return series per factor: "
        "mean(excess return) / std(excess return, ddof=1) * sqrt(periods_per_year), "
        "with excess return = return - risk_free_rate / periods_per_year. "
        "NaN when fewer than min_periods finite returns or when the return "
        "standard deviation is not positive-finite. period: daily (252)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_sharpe_ratio,
    domain=Domain.DRAWDOWN,
    metric_id="sharpe_ratio",
    required_inputs={"forward_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.portfolio_stats.compute_sharpe_ratio",
    metric_version="1.0.0",
    units="ratio",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

_REGISTRY.register(MetricSpec(
    name="sortino_ratio",
    display_name="sortino_ratio",
    description=(
        "Annualized Sortino ratio of a return series per factor: "
        "mean(excess return) / downside deviation * sqrt(periods_per_year), "
        "where downside deviation is the root mean square of negative excess "
        "returns only (no ddof). NaN when fewer than min_periods finite "
        "returns, when there are no negative excess returns, or when the "
        "downside deviation is not positive-finite."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_sortino_ratio,
    domain=Domain.DRAWDOWN,
    metric_id="sortino_ratio",
    required_inputs={"forward_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.portfolio_stats.compute_sortino_ratio",
    metric_version="1.0.0",
    units="ratio",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

_REGISTRY.register(MetricSpec(
    name="win_rate",
    display_name="win_rate",
    description=(
        "Win rate of a return series per factor: fraction of finite returns "
        "that are strictly positive, in [0, 1]. NaN when there are no finite "
        "returns. (Returns equal to exactly 0 count as neither wins nor "
        "losses.)"
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_win_rate,
    domain=Domain.DRAWDOWN,
    metric_id="win_rate",
    required_inputs={"forward_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.portfolio_stats.compute_win_rate",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))


# ---------------------------------------------------------------------------
# Metric Expansion (spec §28-33, §35, §37, Phase 7).
#
# New metric families added on top of the sealed 55-metric core.  Per Phase 7
# every new metric has a CPU reference compute_fn bound here; the cheap ones
# (predictive + quantile-shape) also have GPU parity kernels under
# ``kernels/gpu/``.  All are registered during the BUILDING phase so the
# sealed-registry lifecycle is respected.
# ---------------------------------------------------------------------------

# ---- Domain PREDICTIVE (spec §28) — consume an ICSeriesArtifact ----
_REGISTRY.register(MetricSpec(
    name="ic_positive_ratio",
    display_name="IC Positive Ratio",
    description=(
        "Fraction of finite daily IC values that are strictly positive, per "
        "factor. A value near 1.0 indicates the factor's IC is consistently "
        "positive (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_positive_ratio,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="ic_positive_ratio",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_ic_positive_ratio",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rank_ic_positive_ratio",
    display_name="Rank IC Positive Ratio",
    description=(
        "Fraction of finite daily rank-IC values that are strictly positive, "
        "per factor (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rank_ic_positive_ratio,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="rank_ic_positive_ratio",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_rank_ic_positive_ratio",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="yearly_rank_ic",
    display_name="Yearly Rank IC",
    description=(
        "Mean of the per-year mean rank IC, per factor. A robust annual "
        "average that down-weights any single strong year (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_yearly_rank_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="yearly_rank_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_yearly_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="monthly_rank_ic",
    display_name="Monthly Rank IC",
    description=(
        "Mean of the per-month mean rank IC, per factor (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_monthly_rank_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="monthly_rank_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_monthly_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="quarterly_rank_ic",
    display_name="Quarterly Rank IC",
    description=(
        "Mean of the per-quarter mean rank IC, per factor (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_quarterly_rank_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="quarterly_rank_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_quarterly_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rolling_rank_ic_mean",
    display_name="Rolling Rank IC Mean",
    description=(
        "Time-mean of the rolling-window mean rank IC, per factor. A "
        "smoother estimate of average predictive power (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rolling_rank_ic_mean,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="rolling_rank_ic_mean",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_rolling_rank_ic_mean",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rolling_rank_ic_ir",
    display_name="Rolling Rank IC IR",
    description=(
        "Time-mean of the rolling-window IC information ratio, per factor "
        "(spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rolling_rank_ic_ir,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="rolling_rank_ic_ir",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_rolling_rank_ic_ir",
    metric_version="1.0.0",
    units="ratio",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="recent_3m_rank_ic",
    display_name="Recent 3-Month Rank IC",
    description=(
        "Mean rank IC over the most recent ~3 months (63 trading days), per "
        "factor (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_recent_3m_rank_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="recent_3m_rank_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_recent_3m_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="recent_6m_rank_ic",
    display_name="Recent 6-Month Rank IC",
    description=(
        "Mean rank IC over the most recent ~6 months (126 trading days), per "
        "factor (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_recent_6m_rank_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="recent_6m_rank_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_recent_6m_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="recent_12m_rank_ic",
    display_name="Recent 12-Month Rank IC",
    description=(
        "Mean rank IC over the most recent ~12 months (252 trading days), per "
        "factor (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_recent_12m_rank_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="recent_12m_rank_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_recent_12m_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="worst_year_rank_ic",
    display_name="Worst-Year Rank IC",
    description=(
        "Minimum per-year mean rank IC (the factor's worst year), per factor. "
        "A robustness check on how bad the factor's worst annual performance "
        "is (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_worst_year_rank_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="worst_year_rank_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_worst_year_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="worst_quarter_rank_ic",
    display_name="Worst-Quarter Rank IC",
    description=(
        "Minimum per-quarter mean rank IC (the factor's worst quarter), per "
        "factor (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_worst_quarter_rank_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="worst_quarter_rank_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_worst_quarter_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rank_ic_decay_h01_h05_h10_h20",
    display_name="Rank IC Decay (H01/H05/H10/H20)",
    description=(
        "Mean IC autocorrelation across the decay horizons {1, 5, 10, 20} "
        "trading days, per factor. A single scalar summarising how quickly "
        "the IC series loses autocorrelation (decays) at increasing lags "
        "(spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rank_ic_decay,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="rank_ic_decay_h01_h05_h10_h20",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_rank_ic_decay",
    metric_version="1.0.0",
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="ic_sign_consistency",
    display_name="IC Sign Consistency",
    description=(
        "Fraction of finite daily IC values sharing the sign of the mean IC, "
        "per factor. A value near 1.0 indicates the factor's IC sign is "
        "highly consistent (spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_sign_consistency,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="ic_sign_consistency",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_ic_sign_consistency",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="ic_recent_vs_history_delta",
    display_name="IC Recent vs History Delta",
    description=(
        "(recent mean IC - full-history mean IC) / full-history std, per "
        "factor. Positive means recent IC is stronger than the historical "
        "average; a strongly negative value flags recent degradation "
        "(spec §28)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_recent_vs_history_delta,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.PREDICTIVE,
    metric_id="ic_recent_vs_history_delta",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.predictive.compute_ic_recent_vs_history_delta",
    metric_version="1.0.0",
    units="zscore",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain QUANTILE_SHAPE (spec §29) — consume a QuantileReturnArtifact ----
_REGISTRY.register(MetricSpec(
    name="quantile_monotonicity",
    display_name="Quantile Monotonicity",
    description=(
        "Fraction of adjacent quantile steps that are monotone increasing, "
        "per factor. 1.0 = perfectly monotone (higher factor value -> higher "
        "return); 0.0 = fully inverted (spec §29)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_quantile_monotonicity,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="quantile_monotonicity",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.quantile_shape.compute_quantile_monotonicity",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="quantile_curvature",
    display_name="Quantile Curvature",
    description=(
        "Signed curvature of the quantile-return profile (mean second "
        "difference), per factor. Positive = convex (accelerating) profile; "
        "negative = concave (decelerating) (spec §29)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_quantile_curvature,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="quantile_curvature",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.quantile_shape.compute_quantile_curvature",
    metric_version="1.0.0",
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="quantile_tail_asymmetry",
    display_name="Quantile Tail Asymmetry",
    description=(
        "Asymmetry between the top and bottom quantile tails, per factor. "
        "Positive = top tail is stronger than the bottom tail (spec §29)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_quantile_tail_asymmetry,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="quantile_tail_asymmetry",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.quantile_shape.compute_quantile_tail_asymmetry",
    metric_version="1.0.0",
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="quantile_adjacent_spread",
    display_name="Quantile Adjacent Spread",
    description=(
        "Mean absolute return difference between adjacent quantiles, per "
        "factor. A measure of how smooth (vs step-like) the quantile profile "
        "is (spec §29)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_quantile_adjacent_spread,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="quantile_adjacent_spread",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.quantile_shape.compute_quantile_adjacent_spread",
    metric_version="1.0.0",
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="quantile_extreme_cliff",
    display_name="Quantile Extreme Cliff",
    description=(
        "Mean of the top and bottom quantile cliffs, per factor. A large "
        "value means the extreme quantiles carry most of the spread (a cliff "
        "profile) (spec §29)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_quantile_extreme_cliff,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="quantile_extreme_cliff",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.quantile_shape.compute_quantile_extreme_cliff",
    metric_version="1.0.0",
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="top_quantile_cliff",
    display_name="Top Quantile Cliff",
    description=(
        "Top-quantile cliff: ret[top] - ret[top-1], per factor. The jump in "
        "return from the second-highest to the highest quantile (spec §29)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_top_quantile_cliff,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="top_quantile_cliff",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.quantile_shape.compute_top_quantile_cliff",
    metric_version="1.0.0",
    units="return",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="bottom_quantile_cliff",
    display_name="Bottom Quantile Cliff",
    description=(
        "Bottom-quantile cliff: ret[1] - ret[0], per factor. The jump in "
        "return from the lowest to the second-lowest quantile (spec §29)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_bottom_quantile_cliff,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="bottom_quantile_cliff",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.quantile_shape.compute_bottom_quantile_cliff",
    metric_version="1.0.0",
    units="return",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain REGIME (spec §30) — consume an ICSeriesArtifact ----
_REGISTRY.register(MetricSpec(
    name="year_consistency",
    display_name="Year Consistency",
    description=(
        "Fraction of years whose mean IC matches the overall IC sign, per "
        "factor. A robustness check on how consistently the factor works "
        "across years (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_year_consistency,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="year_consistency",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_year_consistency",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="quarter_consistency",
    display_name="Quarter Consistency",
    description=(
        "Fraction of quarters whose mean IC matches the overall IC sign, per "
        "factor (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_quarter_consistency,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="quarter_consistency",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_quarter_consistency",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="month_consistency",
    display_name="Month Consistency",
    description=(
        "Fraction of months whose mean IC matches the overall IC sign, per "
        "factor (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_month_consistency,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="month_consistency",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_month_consistency",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rolling_ic_volatility",
    display_name="Rolling IC Volatility",
    description=(
        "Time-mean of the rolling-window IC standard deviation, per factor. "
        "Lower values indicate a more stable IC series (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rolling_ic_volatility,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="rolling_ic_volatility",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_rolling_ic_volatility",
    metric_version="1.0.0",
    units="correlation",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rolling_ic_drawdown",
    display_name="Rolling IC Drawdown",
    description=(
        "Mean of the rolling-window IC drawdown (peak-to-trough), per factor. "
        "A more negative value indicates deeper IC drawdowns (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rolling_ic_drawdown,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="rolling_ic_drawdown",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_rolling_ic_drawdown",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="ic_sign_flip_rate",
    display_name="IC Sign Flip Rate",
    description=(
        "Fraction of adjacent finite IC pairs whose sign flips, per factor. "
        "Lower values indicate a more persistent (stable) IC sign (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_ic_sign_flip_rate,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="ic_sign_flip_rate",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_ic_sign_flip_rate",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="change_point_score",
    display_name="Change-Point Score",
    description=(
        "CUSUM-based change-point score: max |cumulative deviation from the "
        "mean IC|, per factor. A large score indicates a structural break in "
        "the IC level (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_change_point_score,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="change_point_score",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_change_point_score",
    metric_version="1.0.0",
    units="correlation",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="cusum_break_score",
    display_name="CUSUM Break Score",
    description=(
        "CUSUM break score: max |cumulative deviation| normalised by "
        "std*sqrt(T), per factor. A standardised change-point statistic; "
        "larger values flag a stronger break (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_cusum_break_score,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="cusum_break_score",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_cusum_break_score",
    metric_version="1.0.0",
    units="zscore",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="recent_degradation_score",
    display_name="Recent Degradation Score",
    description=(
        "Recent degradation: (full mean IC - recent mean IC) / full std, per "
        "factor. Positive values indicate the factor's recent IC is weaker "
        "than its historical average (degradation) (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_recent_degradation_score,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="recent_degradation_score",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_recent_degradation_score",
    metric_version="1.0.0",
    units="zscore",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="regime_conditional_ic",
    display_name="Regime Conditional IC",
    description=(
        "Mean IC in the late (recent) regime, per factor. Measures the "
        "factor's current predictive power (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_regime_conditional_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="regime_conditional_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_regime_conditional_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="regime_worst_ic",
    display_name="Regime Worst IC",
    description=(
        "Minimum of the early/late regime mean IC, per factor. The weaker of "
        "the two regime averages (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_regime_worst_ic,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="regime_worst_ic",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_regime_worst_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="regime_dispersion",
    display_name="Regime Dispersion",
    description=(
        "Absolute difference between early and late regime mean IC, per "
        "factor. Larger values indicate the factor's predictive power changed "
        "across regimes (instability) (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_regime_dispersion,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="regime_dispersion",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_regime_dispersion",
    metric_version="1.0.0",
    units="correlation",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="regime_sign_consistency",
    display_name="Regime Sign Consistency",
    description=(
        "1.0 if early and late regime mean IC share the same sign, else 0.0, "
        "per factor (spec §30)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_regime_sign_consistency,
    requires=["ICSeriesArtifact"],
    min_periods=20,
    ic_method="spearman",
    domain=Domain.REGIME,
    metric_id="regime_sign_consistency",
    required_inputs={"ic_series"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.stability_regime.compute_regime_sign_consistency",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain DATA_QUALITY (spec §35) — consume factor_batch / label_bundle ----
_REGISTRY.register(MetricSpec(
    name="missing_ratio",
    display_name="Missing Ratio",
    description=(
        "Fraction of (T, N) cells with non-finite factor values, per factor "
        "(spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_missing_ratio,
    requires=["factor_batch"],
    min_periods=None,
    domain=Domain.DATA_QUALITY,
    metric_id="missing_ratio",
    required_inputs={"factor"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_missing_ratio",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="missing_timeline",
    display_name="Missing Timeline",
    description=(
        "Fraction of time periods with any missing factor value, per factor. "
        "1.0 means every day has at least one missing asset (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_missing_timeline,
    requires=["factor_batch"],
    min_periods=None,
    domain=Domain.DATA_QUALITY,
    metric_id="missing_timeline",
    required_inputs={"factor"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_missing_timeline",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="staleness",
    display_name="Staleness",
    description=(
        "Mean fraction of assets whose value is unchanged from the prior day, "
        "per factor. A high staleness ratio indicates a slow-moving / sticky "
        "factor (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_staleness,
    requires=["factor_batch"],
    min_periods=2,
    domain=Domain.DATA_QUALITY,
    metric_id="staleness",
    required_inputs={"factor"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_staleness",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="outlier_ratio",
    display_name="Outlier Ratio",
    description=(
        "Fraction of finite factor values that are z-score outliers (|z|>3), "
        "per factor (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_outlier_ratio,
    requires=["factor_batch"],
    min_periods=10,
    domain=Domain.DATA_QUALITY,
    metric_id="outlier_ratio",
    required_inputs={"factor"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_outlier_ratio",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="effective_n",
    display_name="Effective N",
    description=(
        "Mean number of finite factor values per day, per factor. A measure "
        "of the effective universe size (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_effective_n,
    requires=["factor_batch"],
    min_periods=None,
    domain=Domain.DATA_QUALITY,
    metric_id="effective_n",
    required_inputs={"factor"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_effective_n",
    metric_version="1.0.0",
    units="count",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="distinct_level_ratio",
    display_name="Distinct Level Ratio",
    description=(
        "Mean fraction of distinct values among finite factor values per day, "
        "per factor. A low ratio indicates heavy duplication / coarse factor "
        "values (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_distinct_level_ratio,
    requires=["factor_batch"],
    min_periods=None,
    domain=Domain.DATA_QUALITY,
    metric_id="distinct_level_ratio",
    required_inputs={"factor"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_distinct_level_ratio",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="tie_ratio",
    display_name="Tie Ratio",
    description=(
        "Mean fraction of finite factor values that are tied with another "
        "value, per factor. The complement of the distinct level ratio "
        "(spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_tie_ratio,
    requires=["factor_batch"],
    min_periods=None,
    domain=Domain.DATA_QUALITY,
    metric_id="tie_ratio",
    required_inputs={"factor"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_tie_ratio",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="cross_section_cardinality",
    display_name="Cross-Section Cardinality",
    description=(
        "Mean number of distinct values per day (cross-section cardinality), "
        "per factor (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_cross_section_cardinality,
    requires=["factor_batch"],
    min_periods=None,
    domain=Domain.DATA_QUALITY,
    metric_id="cross_section_cardinality",
    required_inputs={"factor"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_cross_section_cardinality",
    metric_version="1.0.0",
    units="count",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="label_maturity",
    display_name="Label Maturity",
    description=(
        "Fraction of (T, N) cells with a finite forward-return label, per "
        "factor. Measures how much of the factor's universe has a mature "
        "(available) label for evaluation (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_label_maturity,
    requires=["factor_batch", "label_bundle"],
    min_periods=None,
    domain=Domain.DATA_QUALITY,
    metric_id="label_maturity",
    required_inputs={"factor", "forward_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_label_maturity",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="tradable_coverage",
    display_name="Tradable Coverage",
    description=(
        "Fraction of days with at least ``min_assets`` jointly valid cells, "
        "per factor. A tradable day is one where the factor and label are "
        "both available for enough assets (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_tradable_coverage,
    requires=["factor_batch", "label_bundle"],
    min_periods=None,
    domain=Domain.DATA_QUALITY,
    metric_id="tradable_coverage",
    required_inputs={"factor", "forward_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_tradable_coverage",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="universe_churn",
    display_name="Universe Churn",
    description=(
        "Mean fraction of the tradable universe that changes membership per "
        "day, per factor. Churn = fraction of assets tradable on exactly one "
        "of two adjacent days (spec §35)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_universe_churn,
    requires=["factor_batch", "label_bundle"],
    min_periods=2,
    domain=Domain.DATA_QUALITY,
    metric_id="universe_churn",
    required_inputs={"factor", "forward_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.data_quality.compute_universe_churn",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---- Domain RESEARCH_INTEGRITY (spec §34) — multiple-testing corrections ----
_REGISTRY.register(MetricSpec(
    name="bonferroni_correction",
    display_name="Bonferroni Correction",
    description=(
        "Bonferroni multiple-testing correction: adjusted p = min(p * n, 1). "
        "Controls the family-wise error rate (spec §34)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=bonferroni_correction,
    requires=["p_values"],
    min_periods=None,
    domain=Domain.RESEARCH_INTEGRITY,
    metric_id="bonferroni_correction",
    required_inputs={"p_values"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.multiple_testing.bonferroni_correction",
    metric_version="1.0.0",
    units="pvalue",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="benjamini_hochberg_correction",
    display_name="Benjamini-Hochberg Correction",
    description=(
        "Benjamini-Hochberg FDR correction: controls the false discovery "
        "rate (spec §34)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=benjamini_hochberg_correction,
    requires=["p_values"],
    min_periods=None,
    domain=Domain.RESEARCH_INTEGRITY,
    metric_id="benjamini_hochberg_correction",
    required_inputs={"p_values"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.multiple_testing.benjamini_hochberg_correction",
    metric_version="1.0.0",
    units="pvalue",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="holm_bonferroni_correction",
    display_name="Holm-Bonferroni Correction",
    description=(
        "Holm-Bonferroni step-down correction: more powerful than Bonferroni, "
        "controls the family-wise error rate (spec §34)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=holm_bonferroni_correction,
    requires=["p_values"],
    min_periods=None,
    domain=Domain.RESEARCH_INTEGRITY,
    metric_id="holm_bonferroni_correction",
    required_inputs={"p_values"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.multiple_testing.holm_bonferroni_correction",
    metric_version="1.0.0",
    units="pvalue",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="sidak_correction",
    display_name="Sidak Correction",
    description=(
        "Sidak multiple-testing correction: adjusted p = 1 - (1 - p)^n. "
        "Assumes independence, slightly less conservative than Bonferroni "
        "(spec §34)."
    ),
    status=MetricStatus.STABLE,
    tier=MetricTier.RESEARCH,
    compute_fn=sidak_correction,
    requires=["p_values"],
    min_periods=None,
    domain=Domain.RESEARCH_INTEGRITY,
    metric_id="sidak_correction",
    required_inputs={"p_values"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.multiple_testing.sidak_correction",
    metric_version="1.0.0",
    units="pvalue",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---------------------------------------------------------------------------
# R61-FI-023: probe-PnL underwater / drawdown-extent evidence (plan §13.9).
#
# All kernels consume a dot-frequency (T,) probe daily PnL series (output of
# metrics.probe_portfolio.compute_cohort_pnl).  max_drawdown already exists
# (portfolio_stats.compute_maximum_drawdown) and is REUSED/MAPPED — never
# re-registered.  Missing data -> NaN evidence (never fabricated 0.0).
# ---------------------------------------------------------------------------
_REGISTRY.register(MetricSpec(
    name="max_underwater_duration",
    display_name="Max Underwater Duration",
    description=(
        "Longest continuous stretch (periods) of the probe daily PnL series "
        "staying below its running-max wealth (waterline). Computed by "
        "metrics.underwater.compute_max_underwater_duration on a (T,) "
        "dot-frequency series (probe cohort pnl). NaN when insufficient data — "
        "never a fabricated 0."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_max_underwater_duration,
    requires=["probe_pnl"],
    min_periods=10,
    domain=Domain.UNDERWATER,
    metric_id="max_underwater_duration",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_max_underwater_duration",
    metric_version="1.0.0",
    units="periods",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="mean_underwater_duration",
    display_name="Mean Underwater Duration",
    description=(
        "Mean length (periods) of underwater episodes of the probe daily PnL "
        "series. NaN when insufficient data; 0.0 when never underwater."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_mean_underwater_duration,
    requires=["probe_pnl"],
    min_periods=10,
    domain=Domain.UNDERWATER,
    metric_id="mean_underwater_duration",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_mean_underwater_duration",
    metric_version="1.0.0",
    units="periods",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="time_to_recovery",
    display_name="Time to Recovery",
    description=(
        "Mean time (periods) from an underwater episode's trough back to a new "
        "wealth high. Only completed recoveries are averaged; an ongoing "
        "all-episode drawdown reports the observed lookback instead (not 0). "
        "NaN when insufficient data."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_time_to_recovery,
    requires=["probe_pnl"],
    min_periods=10,
    domain=Domain.UNDERWATER,
    metric_id="time_to_recovery",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_time_to_recovery",
    metric_version="1.0.0",
    units="periods",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="worst_month",
    display_name="Worst Month",
    description=(
        "Worst fixed 21-period (trading) block compounded return of the probe "
        "daily PnL series. Uses the codebase's fixed trading-period calendar "
        "(21/period month); NaN when the series is shorter than one block."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_worst_period_return,
    requires=["probe_pnl"],
    min_periods=21,
    domain=Domain.UNDERWATER,
    metric_id="worst_month",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_worst_period_return",
    metric_version="1.0.0",
    units="return",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="worst_quarter",
    display_name="Worst Quarter",
    description=(
        "Worst fixed 63-period (trading) block compounded return of the probe "
        "daily PnL series. NaN when the series is shorter than one block."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_worst_period_return,
    requires=["probe_pnl"],
    min_periods=63,
    domain=Domain.UNDERWATER,
    metric_id="worst_quarter",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_worst_period_return",
    metric_version="1.0.0",
    units="return",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="worst_12m",
    display_name="Worst 12-Month",
    description=(
        "Worst fixed 252-period (trading) block compounded return of the probe "
        "daily PnL series. NaN when the series is shorter than one 12m block."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_worst_period_return,
    requires=["probe_pnl"],
    min_periods=252,
    domain=Domain.UNDERWATER,
    metric_id="worst_12m",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_worst_period_return",
    metric_version="1.0.0",
    units="return",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rolling_1y_sharpe_min",
    display_name="Rolling 1Y Sharpe Min",
    description=(
        "Minimum rolling-252-period annualized Sharpe of the probe daily PnL "
        "series (worst observed 1y window). NaN when the series is shorter "
        "than one window."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rolling_sharpe_tail,
    requires=["probe_pnl"],
    min_periods=60,
    domain=Domain.UNDERWATER,
    metric_id="rolling_1y_sharpe_min",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail",
    metric_version="1.0.0",
    units="ratio",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="rolling_1y_sharpe_q10",
    display_name="Rolling 1Y Sharpe Q10",
    description=(
        "10th-percentile rolling-252-period annualized Sharpe of the probe "
        "daily PnL series (tail floor of the 1y rolling Sharpe distribution). "
        "NaN when the series is shorter than one window."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_rolling_sharpe_tail,
    requires=["probe_pnl"],
    min_periods=60,
    domain=Domain.UNDERWATER,
    metric_id="rolling_1y_sharpe_q10",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail",
    metric_version="1.0.0",
    units="ratio",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="return_skew",
    display_name="Return Skew",
    description=(
        "Sample skewness of the probe daily PnL return series "
        "(population convention, same statistic as the tail-risk "
        "distribution.compute_skewness family). NaN when insufficient data "
        "or zero std."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_return_skew,
    requires=["probe_pnl"],
    min_periods=20,
    domain=Domain.UNDERWATER,
    metric_id="return_skew",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_return_skew",
    metric_version="1.0.0",
    units="dimensionless",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="downside_deviation",
    display_name="Downside Deviation",
    description=(
        "Annualized downside deviation (RMS of negative excess returns) of "
        "the probe daily PnL series, matching the portfolio_stats.sortino "
        "semantics. NaN when insufficient data or no negative excess returns."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_downside_deviation,
    requires=["probe_pnl"],
    min_periods=20,
    domain=Domain.UNDERWATER,
    metric_id="downside_deviation",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_downside_deviation",
    metric_version="1.0.0",
    units="return",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="cvar_expected_shortfall",
    display_name="CVaR Expected Shortfall",
    description=(
        "Historical CVaR / expected shortfall at 95% of the probe daily PnL "
        "series (delegates to risk.var_cvar.compute_cvar — single numeric "
        "authority). Positive loss magnitude; NaN when insufficient data."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_cvar_expected_shortfall,
    requires=["probe_pnl"],
    min_periods=20,
    domain=Domain.UNDERWATER,
    metric_id="cvar_expected_shortfall",
    required_inputs={"probe_pnl"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.underwater.compute_cvar_expected_shortfall",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---------------------------------------------------------------------------
# R61-FI-024: per-style exposure / purity evidence (plan §13.11 + §16.2).
#
# Every metric consumes an injected ExposurePanel (DataAccess-authoritative
# ref; QE never fabricates exposure data).  Per-style typed fields only —
# there is no single scalar "exposure" metric.  GPU parity path reuses the
# R60 kernels/gpu/exposure.py batch kernels (CPU reference authoritative).
# ---------------------------------------------------------------------------
_REGISTRY.register(MetricSpec(
    name="industry_exposure",
    display_name="Industry Exposure",
    description=(
        "Signed mean industry-style exposure of the factor (typed per-style "
        "field, from the injected ExposurePanel). NaN when style absent or no "
        "finite cells."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_industry_exposure,
    requires=["exposure_panel"],
    min_periods=None,
    domain=Domain.EXPOSURE,
    metric_id="industry_exposure",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_industry_exposure",
    metric_version="1.0.0",
    units="exposure",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="size_exposure",
    display_name="Size Exposure",
    description=(
        "Signed mean size-style exposure of the factor (typed per-style "
        "field). NaN when style absent or no finite cells."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_size_exposure,
    requires=["exposure_panel"],
    min_periods=None,
    domain=Domain.EXPOSURE,
    metric_id="size_exposure",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_size_exposure",
    metric_version="1.0.0",
    units="exposure",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="beta_exposure",
    display_name="Beta Exposure",
    description=(
        "Signed mean beta-style exposure of the factor (typed per-style "
        "field). NaN when style absent or no finite cells."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_beta_exposure,
    requires=["exposure_panel"],
    min_periods=None,
    domain=Domain.EXPOSURE,
    metric_id="beta_exposure",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_beta_exposure",
    metric_version="1.0.0",
    units="exposure",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="liquidity_exposure",
    display_name="Liquidity Exposure",
    description=(
        "Signed mean liquidity-style exposure of the factor (typed per-style "
        "field). NaN when style absent or no finite cells."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_liquidity_exposure,
    requires=["exposure_panel"],
    min_periods=None,
    domain=Domain.EXPOSURE,
    metric_id="liquidity_exposure",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_liquidity_exposure",
    metric_version="1.0.0",
    units="exposure",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="volatility_exposure",
    display_name="Volatility Exposure",
    description=(
        "Signed mean volatility-style exposure of the factor (typed per-style "
        "field). NaN when style absent or no finite cells."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_volatility_exposure,
    requires=["exposure_panel"],
    min_periods=None,
    domain=Domain.EXPOSURE,
    metric_id="volatility_exposure",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_volatility_exposure",
    metric_version="1.0.0",
    units="exposure",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="momentum_exposure",
    display_name="Momentum Exposure",
    description=(
        "Signed mean momentum-style exposure of the factor (typed per-style "
        "field). NaN when style absent or no finite cells."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_momentum_exposure,
    requires=["exposure_panel"],
    min_periods=None,
    domain=Domain.EXPOSURE,
    metric_id="momentum_exposure",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_momentum_exposure",
    metric_version="1.0.0",
    units="exposure",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="max_absolute_style_exposure",
    display_name="Max Absolute Style Exposure",
    description=(
        "The style dimension with the largest mean absolute exposure (dict: "
        "style / value / absolute_mean / counts). NaN when no style has "
        "enough finite cells."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_max_absolute_style_exposure,
    requires=["exposure_panel"],
    min_periods=None,
    domain=Domain.EXPOSURE,
    metric_id="max_absolute_style_exposure",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_max_absolute_style_exposure",
    metric_version="1.0.0",
    units="exposure",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="exposure_drift",
    display_name="Exposure Drift",
    description=(
        "Mean absolute change of the per-style exposure panel between "
        "adjacent periods (persistence / stability measure). NaN when fewer "
        "than 2 periods or no finite adjacent pair."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_exposure_drift,
    requires=["exposure_panel"],
    min_periods=2,
    domain=Domain.EXPOSURE,
    metric_id="exposure_drift",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_exposure_drift",
    metric_version="1.0.0",
    units="exposure",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="neutralized_rank_ic",
    display_name="Neutralized Rank IC",
    description=(
        "Cross-sectional residual (neutralized) rank IC: per date regress the "
        "factor on the exposure panel (OLS intercept + styles), Spearman-"
        "correlate residuals with forward returns, time-mean. A factor whose "
        "IC survives neutralization has alpha orthogonal to style exposures. "
        "CPU reference; GPU optional (kernels/gpu/exposure_evidence.py)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_neutralized_rank_ic,
    requires=["factor_values", "forward_returns", "exposure_panel"],
    min_periods=10,
    domain=Domain.EXPOSURE,
    metric_id="neutralized_rank_ic",
    required_inputs={"factor_values", "forward_returns", "exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_neutralized_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="residual_rank_ic",
    display_name="Residual Rank IC",
    description=(
        "Named alias of neutralized_rank_ic (residual version of the rank "
        "IC, residualized against style exposures). Same kernel, registered "
        "under its own id for downstream reporting."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_residual_rank_ic,
    requires=["factor_values", "forward_returns", "exposure_panel"],
    min_periods=10,
    domain=Domain.EXPOSURE,
    metric_id="residual_rank_ic",
    required_inputs={"factor_values", "forward_returns", "exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_residual_rank_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="purity_ratio",
    display_name="Purity Ratio",
    description=(
        "Purity ratio = 1 - style-explained share of total exposure "
        "dispersion. Higher = cleaner (the factor's own signal dominates its "
        "style footprint). NaN when the panel has no valid style."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_purity_ratio,
    requires=["exposure_panel"],
    min_periods=None,
    domain=Domain.EXPOSURE,
    metric_id="purity_ratio",
    required_inputs={"exposure_panel"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.exposure_evidence.compute_purity_ratio",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---------------------------------------------------------------------------
# R61-FI-021 (plan §13.4/§14.1/§14.3/§14.4): shape-evidence family additions.
#
# Audit result (2026-09-05): the pre-existing SHAPE family (monotonicity /
# curvature / tail_asymmetry / adjacent_spread / extreme_cliff / top+bottom
# cliff above) is REUSED unchanged.  This block registers the plan ids that
# were MISSING: u_shape_score / inverted_u_score / adaptive_quantile_count /
# top_tail_slope / bottom_tail_slope / tail_vs_middle_contrast /
# left_right_asymmetry / linear_trend_score / shape_stability /
# shape_regime_stability / shape_bootstrap_confidence plus the robust tail
# cliff variants (plan §14.4: Q_K - mean(Q_(K-3..K-1))) which are a DIFFERENT
# semantic from the one-bin top_quantile_cliff / bottom_quantile_cliff, so
# they are separate ids (top_quantile_cliff_robust / bottom_quantile_cliff_robust).
#
# All are CPU reference implementations in metrics/shape_evidence.py consuming
# a QuantileReturnArtifact (n_quantiles, F) profile (or (W, nq, F) windowed
# profiles for the stability family).  Orientation metadata (plan §14.4:
# never assume all factors positive) is carried on each MetricSpec.direction.
# ---------------------------------------------------------------------------
_REGISTRY.register(MetricSpec(
    name="u_shape_score",
    display_name="U-Shape Score",
    description=(
        "U-shape score in [0, 1] per factor (plan §14.3). NOT RankIC-about-0 "
        "detection: combines U-template R^2 (middle underperforms both tails, "
        "convex), the fraction of positive (convex) interior second "
        "differences, and the requirement that the U-template fit EXCEEDS the "
        "monotone-linear template fit. 1 = textbook U, 0 = not U."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_u_shape_score,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="u_shape_score",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_u_shape_score",
    metric_version="1.0.0",
    units="score",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="inverted_u_score",
    display_name="Inverted-U Shape Score",
    description=(
        "Inverted-U (hill) score in [0, 1] per factor. Mirror of "
        "u_shape_score with concave curvature and an inverted-U template; "
        "template fit must exceed the monotone fit."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_inverted_u_score,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="inverted_u_score",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_inverted_u_score",
    metric_version="1.0.0",
    units="score",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="adaptive_quantile_count",
    display_name="Adaptive Quantile Count",
    description=(
        "ACTUAL quantile-bin count used to build the profile per factor "
        "(plan §14.1 adaptive 20 -> (10, 5) fallback). Reads the artifact's "
        "n_quantiles; consumers never assume 20. Integer-valued float. "
        "NaN when no profile exists (never 0 bins)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_adaptive_quantile_count,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="adaptive_quantile_count",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_adaptive_quantile_count",
    metric_version="1.0.0",
    units="count",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="top_tail_slope",
    display_name="Top Tail Slope",
    description=(
        "Mean adjacent return difference over the TOP segment of the "
        "quantile profile as drawn (quantiles K-3..K-1), per factor. "
        "Positive for a positively inclined factor; large = top-tail cliff. "
        "Direction NEUTRAL: the metric measures the drawn profile's top "
        "segment; orientation is informative, never assumed (plan §14.4)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_top_tail_slope,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="top_tail_slope",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_top_tail_slope",
    metric_version="1.0.0",
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="bottom_tail_slope",
    display_name="Bottom Tail Slope",
    description=(
        "Mean adjacent return difference over the BOTTOM segment of the "
        "quantile profile as drawn (quantiles 0..2), per factor. Direction "
        "NEUTRAL (orientation informative, never assumed, plan §14.4)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_bottom_tail_slope,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="bottom_tail_slope",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_bottom_tail_slope",
    metric_version="1.0.0",
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="tail_vs_middle_contrast",
    display_name="Tail vs Middle Contrast",
    description=(
        "Mean |tail returns - middle return| per factor (tails = outer "
        "quartiles of the quantile range). High for U/inverted-U profiles, "
        "low for flat. Always >= 0. NaN when profile too small or tail/middle "
        "returns not finite."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_tail_vs_middle_contrast,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="tail_vs_middle_contrast",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_tail_vs_middle_contrast",
    metric_version="1.0.0",
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="left_right_asymmetry",
    display_name="Left-Right Asymmetry",
    description=(
        "Signed asymmetry of the quantile profile about the median quantile "
        "per factor: (top-half rise) - (bottom-half rise). Positive = "
        "upside tail stronger. Direction NEUTRAL (informative sign). NOTE: "
        "on an even quantile count the drawn quantile grid itself is "
        "asymmetric, so a perfectly symmetric latent U reports a small "
        "positive value - compare across factors, never read absolute."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_left_right_asymmetry,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="left_right_asymmetry",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_left_right_asymmetry",
    metric_version="1.0.0",
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="linear_trend_score",
    display_name="Linear Trend Score",
    description=(
        "Pearson correlation of the quantile profile with the linear "
        "quantile coordinate per factor in [-1, 1]. +1 = perfectly monotone "
        "increasing, -1 = decreasing, ~0 = flat or U-shaped (separate the U "
        "with u_shape_score)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_linear_trend_score,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="linear_trend_score",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_linear_trend_score",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="shape_stability",
    display_name="Shape Stability",
    description=(
        "Mean Fisher-z window-vs-overall correlation of the quantile "
        "profile across W windows per factor. Consumes a (W, n_quantiles, F) "
        "windowed profile panel. 1 = identical shape in every window. NaN "
        "for a single-window profile (stability is undefined - missing "
        "evidence, never 0/1)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_shape_stability,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="shape_stability",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_shape_stability",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="shape_regime_stability",
    display_name="Shape Regime Stability",
    description=(
        "Regime version of shape stability: mean Fisher-z correlation of "
        "CONSECUTIVE window profiles per factor (W >= 3 windows). Drops a "
        "single common shape that is stable overall but regime-uncorrelated. "
        "NaN when fewer than 3 windows."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_shape_regime_stability,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="shape_regime_stability",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_shape_regime_stability",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="shape_bootstrap_confidence",
    display_name="Shape Bootstrap Confidence",
    description=(
        "Bootstrap confidence of the profile's quantile RANK ORDER per "
        "factor: fraction of window-resamples whose order reproduces the "
        "overall mean profile order (W >= 3). 1 = shape ordering reproduced "
        "in every resample. Deterministic (seeded). NaN for single-window."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_shape_bootstrap_confidence,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="shape_bootstrap_confidence",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="top_quantile_cliff_robust",
    display_name="Top Quantile Cliff (Robust)",
    description=(
        "ROBUST top cliff per factor: Q_K - mean(Q_(K-3)..Q_(K-1)) (plan "
        "§14.4 - contrast the top bucket against the mean of the three "
        "PRIOR buckets, not the noisy one-bin Q_K - Q_(K-1)). Direction "
        "higher_is_better (big positive jump into the top bucket). NaN when "
        "the top 4 quantile returns are not all finite."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_top_quantile_cliff_robust,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="top_quantile_cliff_robust",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_top_quantile_cliff_robust",
    metric_version="1.0.0",
    units="return",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="bottom_quantile_cliff_robust",
    display_name="Bottom Quantile Cliff (Robust)",
    description=(
        "ROBUST bottom cliff per factor: Q_1 - mean(Q_2..Q_4) (plan §14.4 "
        "mirror). Direction higher_is_better (big positive step out of the "
        "bottom bucket). NaN when the bottom 4 quantile returns are not all "
        "finite."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=compute_bottom_quantile_cliff_robust,
    requires=["QuantileReturnArtifact"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="bottom_quantile_cliff_robust",
    required_inputs={"quantile_returns"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.shape_evidence.compute_bottom_quantile_cliff_robust",
    metric_version="1.0.0",
    units="return",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))

# ---------------------------------------------------------------------------
# R61-FI-022 (plan §13.6): train-vs-validation generalization evidence.
#
# Registered ids (8) expose typed comparison fields of the frozen
# TrainVsValidationArtifact (metrics/generalization_evidence.py).  These are
# per-factor SCALAR reporting metrics for the registry/facade; the full typed
# artifact (retention + reasons + confidence interval + sign consistency +
# versioned policy) is the authoritative container consumed by callers that
# need the whole comparison.  NO metric here exposes or references a
# sealed/Test split - the sealed boundary is contracts/sealed_split.py.
# ---------------------------------------------------------------------------
_REGISTRY.register(MetricSpec(
    name="train_predictive_dimension",
    display_name="Train Predictive Dimension",
    description=(
        "Train-side predictive dimension of the authorized train-vs-validation "
        "comparison (plan §13.6), per factor (e.g. mean daily rank IC on the "
        "train evaluation)."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
    requires=["factor_batch"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="train_predictive_dimension",
    required_inputs={"factor_batch"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.generalization_evidence.train_predictive_dimension",
    metric_version="1.0.0",
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="validation_predictive_dimension",
    display_name="Validation Predictive Dimension",
    description=(
        "Validation-side predictive dimension of the authorized "
        "train-vs-validation comparison (plan §13.6), per factor."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
    requires=["factor_batch"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="validation_predictive_dimension",
    required_inputs={"factor_batch"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.generalization_evidence.validation_predictive_dimension",
    metric_version="1.0.0",
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="validation_retention",
    display_name="Validation Retention",
    description=(
        "Robust validation/train retention per factor (plan §13.6): "
        "validation/train where the train denominator is stable, NaN with an "
        "explicit reason (train_near_zero / sign_flip_guard / "
        "insufficient_data) otherwise - never a blind division by a tiny "
        "train value. Grade anchors are versioned policy constants in "
        "RetentionPolicy."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
    requires=["factor_batch"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="validation_retention",
    required_inputs={"factor_batch"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.generalization_evidence.compute_validation_retention_array",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="train_validation_rankic_delta",
    display_name="Train-Validation RankIC Delta",
    description=(
        "Absolute rank-IC delta (validation - train) per factor (plan "
        "§13.6). A DELTA, never a ratio: well-defined even when train is "
        "near zero. NaN when either side is not finite."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
    requires=["factor_batch"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="train_validation_rankic_delta",
    required_inputs={"factor_batch"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas",
    metric_version="1.0.0",
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="train_validation_icir_delta",
    display_name="Train-Validation ICIR Delta",
    description=(
        "Absolute ICIR delta (validation - train) per factor (plan §13.6). "
        "Delta, never a ratio."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
    requires=["factor_batch"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="train_validation_icir_delta",
    required_inputs={"factor_batch"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas",
    metric_version="1.0.0",
    units="ratio",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="train_validation_sharpe_delta",
    display_name="Train-Validation Sharpe Delta",
    description=(
        "Absolute long/short Sharpe delta (validation - train) per factor "
        "(plan §13.6). Delta, never a ratio."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
    requires=["factor_batch"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="train_validation_sharpe_delta",
    required_inputs={"factor_batch"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas",
    metric_version="1.0.0",
    units="ratio",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="train_validation_shape_delta",
    display_name="Train-Validation Shape Delta",
    description=(
        "Absolute shape-evidence delta (validation - train) per factor "
        "(plan §13.6). Delta, never a ratio."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
    requires=["factor_batch"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="train_validation_shape_delta",
    required_inputs={"factor_batch"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas",
    metric_version="1.0.0",
    units="score",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
))
_REGISTRY.register(MetricSpec(
    name="parameter_generalization",
    display_name="Parameter Generalization",
    description=(
        "Parameter-generalization summary per factor (plan §13.6): the "
        "robust retention mean where the train denominator is stable, NaN "
        "when no factor has a stable denominator (missing evidence - never "
        "0). Versioned anchors in RetentionPolicy."
    ),
    status=MetricStatus.EXPERIMENTAL,
    tier=MetricTier.EXTENDED,
    compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
    requires=["factor_batch"],
    min_periods=20,
    domain=Domain.QUANTILE_SHAPE,
    metric_id="parameter_generalization",
    required_inputs={"factor_batch"},
    output_type="scalar",
    implementation_id="quant_evaluator.metrics.generalization_evidence.compute_validation_retention",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
))
