"""
10-domain metric catalog for quant_evaluator (QE-METRIC overhaul, section B).

Defines MetricSpec, Domain enum, and CATALOG registry mapping metric IDs
to their specifications across 10 evaluation domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


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


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single metric."""

    domain: Domain
    metric_id: str
    description: str
    required_inputs: Set[str]
    output_type: str = "scalar"

    def __post_init__(self) -> None:
        """Validate inputs after construction."""
        # Ensure required_inputs is a frozenset for hashability
        object.__setattr__(self, "required_inputs", frozenset(self.required_inputs))


# ---------------------------------------------------------------------------
# CATALOG: metric_id -> MetricSpec
# ---------------------------------------------------------------------------
CATALOG: Dict[str, MetricSpec] = {}


def _register(
    domain: Domain,
    metric_id: str,
    description: str,
    required_inputs: Set[str],
    output_type: str = "scalar",
) -> None:
    """Register a metric spec into the global CATALOG."""
    spec = MetricSpec(
        domain=domain,
        metric_id=metric_id,
        description=description,
        required_inputs=required_inputs,
        output_type=output_type,
    )
    CATALOG[metric_id] = spec


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
    from quant_evaluator.contracts.errors import UnsupportedMetricError

    try:
        return CATALOG[metric_id]
    except KeyError:
        raise UnsupportedMetricError(
            f"Unknown metric_id '{metric_id}'. "
            f"Valid IDs: {sorted(CATALOG.keys())}"
        )


def get_metric_specs_by_domain(domain: Domain) -> List[MetricSpec]:
    """
    Return all MetricSpec entries for a given Domain, ordered by metric_id.
    """
    return sorted(
        (spec for spec in CATALOG.values() if spec.domain == domain),
        key=lambda s: s.metric_id,
    )


def list_all_metric_ids() -> List[str]:
    """Return a sorted list of all registered metric IDs."""
    return sorted(CATALOG.keys())


def list_all_domains() -> List[Domain]:
    """Return a sorted list of all Domain enum values."""
    return sorted(Domain, key=lambda d: d.value)
