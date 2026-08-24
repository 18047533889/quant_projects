"""
10-domain metric catalog for quant_evaluator (QE-METRIC overhaul, section B).

Defines MetricSpec, Domain enum, and a sealed MetricRegistry mapping metric IDs
to their specifications across 10 evaluation domains.

The default catalog is registered at import time and then sealed: the module
level ``CATALOG`` is an immutable ``MappingProxyType`` view over the backing
registry, and any further ``register`` attempt fails closed.  Public query
helpers (``get_metric_spec``, ``get_metric_specs_by_domain``,
``list_all_metric_ids``, ``list_all_domains``) keep their exact historical
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Set, Tuple


class Domain(Enum):
    """Metric evaluation domains."""

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


# Mapping of output_type -> canonical artifact kind.
_OUTPUT_TYPE_TO_ARTIFACT_KIND: Dict[str, str] = {
    "scalar": "scalar",
    "series": "series",
    "timeseries": "series",
    "vector": "vector",
    "matrix": "matrix",
    "distribution": "distribution",
}

_VALID_DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})

_VALID_ARTIFACT_KINDS = frozenset(
    {"scalar", "series", "vector", "matrix", "distribution"}
)


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single metric."""

    domain: Domain
    metric_id: str
    description: str
    required_inputs: Set[str]
    output_type: str = "scalar"
    metric_version: str = "0.1.0"
    implementation_id: str = ""
    artifact_kind: str = "scalar"
    required_axes: Tuple[str, ...] = ()
    units: str = ""
    direction: str = "higher_is_better"
    missing_policy: str = "nan"
    numeric_policy: str = "finite"

    def __post_init__(self) -> None:
        """Validate inputs after construction."""
        # Normalize required_inputs to a frozenset for hashability.
        object.__setattr__(self, "required_inputs", frozenset(self.required_inputs))

        if not isinstance(self.metric_id, str) or not self.metric_id.strip():
            raise ValueError("MetricSpec.metric_id must be a non-empty string")

        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"MetricSpec.direction must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )

        if not isinstance(self.required_axes, tuple):
            raise ValueError("MetricSpec.required_axes must be a tuple")
        for axis in self.required_axes:
            if not isinstance(axis, str) or not axis.strip():
                raise ValueError(
                    "MetricSpec.required_axes entries must be non-empty strings"
                )

        # Map output_type -> artifact_kind when the caller left the default.
        if self.artifact_kind == "scalar" and self.output_type != "scalar":
            mapped = _OUTPUT_TYPE_TO_ARTIFACT_KIND.get(self.output_type)
            if mapped is not None and mapped != "scalar":
                object.__setattr__(self, "artifact_kind", mapped)
        if self.artifact_kind not in _VALID_ARTIFACT_KINDS:
            raise ValueError(
                f"MetricSpec.artifact_kind must be one of "
                f"{sorted(_VALID_ARTIFACT_KINDS)}, got {self.artifact_kind!r}"
            )

        # implementation_id auto-defaults to metric_id when left empty.
        if not isinstance(self.implementation_id, str):
            raise ValueError("MetricSpec.implementation_id must be a string")
        if not self.implementation_id.strip():
            object.__setattr__(self, "implementation_id", self.metric_id)

        if not isinstance(self.missing_policy, str) or not self.missing_policy.strip():
            raise ValueError("MetricSpec.missing_policy must be a non-empty string")
        if not isinstance(self.numeric_policy, str) or not self.numeric_policy.strip():
            raise ValueError("MetricSpec.numeric_policy must be a non-empty string")
        if not isinstance(self.metric_version, str) or not self.metric_version.strip():
            raise ValueError("MetricSpec.metric_version must be a non-empty string")


class MetricRegistry:
    """Sealed metric registry with a BUILDING -> SEALED lifecycle.

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


# ---------------------------------------------------------------------------
# Module-level registry + CATALOG view.
#
# ``_REGISTRY`` is the authority; the module-level ``CATALOG`` is the sealed
# MappingProxyType (or the backing dict before sealing) so existing callers
# doing ``CATALOG[metric_id]`` lookups and iteration keep working, while any
# assignment raises ``TypeError`` once the default catalog has been sealed at
# import time.
# ---------------------------------------------------------------------------
_REGISTRY: MetricRegistry = MetricRegistry()
CATALOG: Mapping[str, MetricSpec] = _REGISTRY._catalog  # type: ignore[assignment]


def _register(
    domain: Domain,
    metric_id: str,
    description: str,
    required_inputs: Set[str],
    output_type: str = "scalar",
) -> None:
    """Register a metric spec into the global catalog via the registry."""
    spec = MetricSpec(
        domain=domain,
        metric_id=metric_id,
        description=description,
        required_inputs=required_inputs,
        output_type=output_type,
    )
    _REGISTRY.register(spec)


def seal_metric_registry() -> None:
    """Seal the module-level registry so the catalog becomes immutable.

    Idempotent: calling it again after the registry is already sealed is a
    no-op.  Re-binds the module-level ``CATALOG`` to the registry's immutable
    view so ``CATALOG[...]`` reads keep working while assignment raises.
    """
    global CATALOG
    _REGISTRY.seal()
    CATALOG = _REGISTRY._catalog  # type: ignore[assignment]


# ---- Domain IC ----
_register(
    Domain.IC,
    "pearson_ic",
    "Pearson correlation between factor values and forward returns",
    {"factor", "forward_returns"},
)
_register(
    Domain.IC,
    "spearman_ic",
    "Spearman rank correlation between factor values and forward returns",
    {"factor", "forward_returns"},
)
_register(
    Domain.IC,
    "rank_ic",
    "Rank IC (alias for spearman_ic) averaged cross-sectionally",
    {"factor", "forward_returns"},
)
_register(
    Domain.IC,
    "ic_summary",
    "Summary statistics (mean, std, skew, kurtosis) of IC time series",
    {"factor", "forward_returns"},
)

# ---- Domain RANK_IC ----
_register(
    Domain.RANK_IC,
    "rank_ic_time_series",
    "Rank IC computed per time slice, returned as a time series",
    {"factor", "forward_returns"},
    output_type="timeseries",
)
_register(
    Domain.RANK_IC,
    "rank_ic_cross_section",
    "Rank IC computed cross-sectionally for each date",
    {"factor", "forward_returns"},
    output_type="series",
)

# ---- Domain QUANTILE ----
_register(
    Domain.QUANTILE,
    "quantile_returns",
    "Average forward return per quantile bucket",
    {"factor", "forward_returns"},
    output_type="series",
)
_register(
    Domain.QUANTILE,
    "quantile_spread",
    "Spread between top and bottom quantile returns",
    {"factor", "forward_returns"},
)
_register(
    Domain.QUANTILE,
    "quantile_stability",
    "Stability of quantile return rankings across time",
    {"factor", "forward_returns"},
    output_type="timeseries",
)

# ---- Domain DRAWDOWN ----
_register(
    Domain.DRAWDOWN,
    "max_drawdown",
    "Maximum drawdown of the cumulative IC series",
    {"factor", "forward_returns"},
)
_register(
    Domain.DRAWDOWN,
    "drawdown_duration",
    "Duration (in periods) of the longest drawdown",
    {"factor", "forward_returns"},
)
_register(
    Domain.DRAWDOWN,
    "calmar_ratio",
    "Calmar ratio: annualized return / max drawdown",
    {"factor", "forward_returns"},
)

# ---- Domain TURNOVER ----
_register(
    Domain.TURNOVER,
    "turnover_rate",
    "Average rate of change in factor ranking between periods",
    {"factor"},
)
_register(
    Domain.TURNOVER,
    "turnover_cost",
    "Estimated transaction cost from factor rebalancing",
    {"factor", "transaction_costs"},
)
_register(
    Domain.TURNOVER,
    "turnover_adjusted_ic",
    "IC adjusted for turnover-induced transaction costs",
    {"factor", "forward_returns", "transaction_costs"},
)

# ---- Domain TAIL_RISK ----
_register(
    Domain.TAIL_RISK,
    "var_95",
    "Value at Risk at 95% confidence level",
    {"forward_returns"},
)
_register(
    Domain.TAIL_RISK,
    "var_99",
    "Value at Risk at 99% confidence level",
    {"forward_returns"},
)
_register(
    Domain.TAIL_RISK,
    "cvar_95",
    "Conditional Value at Risk (Expected Shortfall) at 95%",
    {"forward_returns"},
)
_register(
    Domain.TAIL_RISK,
    "cvar_99",
    "Conditional Value at Risk (Expected Shortfall) at 99%",
    {"forward_returns"},
)
_register(
    Domain.TAIL_RISK,
    "skewness",
    "Skewness of the return distribution",
    {"forward_returns"},
)
_register(
    Domain.TAIL_RISK,
    "kurtosis",
    "Excess kurtosis of the return distribution",
    {"forward_returns"},
)

# ---- Domain COVERAGE ----
_register(
    Domain.COVERAGE,
    "factor_coverage",
    "Fraction of universe with non-null factor values",
    {"factor"},
)
_register(
    Domain.COVERAGE,
    "return_coverage",
    "Fraction of universe with non-null forward returns",
    {"forward_returns"},
)
_register(
    Domain.COVERAGE,
    "joint_coverage",
    "Fraction of universe with both factor and return available",
    {"factor", "forward_returns"},
)

# ---- Domain HHI ----
_register(
    Domain.HHI,
    "hhi_concentration",
    "Herfindahl-Hirschman Index of factor value concentration",
    {"factor"},
)
_register(
    Domain.HHI,
    "hhi_effective_n",
    "Effective number of groups (1/HHI) for factor concentration",
    {"factor"},
)

# ---- Domain STABILITY ----
_register(
    Domain.STABILITY,
    "ic_stability",
    "Rolling correlation of IC values across sub-periods",
    {"factor", "forward_returns"},
)
_register(
    Domain.STABILITY,
    "turnover_stability",
    "Variance of turnover rate across periods",
    {"factor"},
)
_register(
    Domain.STABILITY,
    "coverage_stability",
    "Variance of factor coverage across periods",
    {"factor"},
)

# ---- Domain TEMPORAL ----
_register(
    Domain.TEMPORAL,
    "rolling_ic",
    "Rolling window IC values over time",
    {"factor", "forward_returns"},
    output_type="timeseries",
)
_register(
    Domain.TEMPORAL,
    "ic_decay",
    "IC decay: correlation at increasing forward horizons",
    {"factor", "forward_returns"},
    output_type="timeseries",
)
_register(
    Domain.TEMPORAL,
    "autocorrelation_ic",
    "Autocorrelation of IC values at specified lags",
    {"factor", "forward_returns"},
    output_type="timeseries",
)

# Seal the default catalog: from this point on the module-level CATALOG is
# immutable and any further registration fails closed.
seal_metric_registry()


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------

def get_metric_spec(metric_id: str) -> MetricSpec:
    """
    Retrieve the MetricSpec for a given metric_id.

    Raises
    ------
    UnsupportedMetricError
        If metric_id is not in the catalog.
    """
    return _REGISTRY.get_metric_spec(metric_id)


def get_metric_specs_by_domain(domain: Domain) -> List[MetricSpec]:
    """
    Return all MetricSpec entries for a given Domain, ordered by metric_id.
    """
    return _REGISTRY.get_metric_specs_by_domain(domain)


def list_all_metric_ids() -> List[str]:
    """Return a sorted list of all registered metric IDs."""
    return _REGISTRY.list_all_metric_ids()


def list_all_domains() -> List[Domain]:
    """Return a sorted list of all Domain enum values."""
    return _REGISTRY.list_all_domains()
