"""
10-domain metric catalog for quant_evaluator (QE-METRIC overhaul, section B).

QE-P0-01: this module is a **compatibility adapter / generated read-only view**
over the SINGLE metric authority in ``quant_evaluator.registry.metrics``.  It
does NOT define its own ``MetricSpec`` / ``MetricRegistry`` / ``Domain`` — those
are re-exported from the registry so there is exactly ONE class of each in the
package.  The catalog keeps its historical 10-domain metric_ids and query
helpers (``get_metric_spec``, ``get_metric_specs_by_domain``,
``list_all_metric_ids``, ``list_all_domains``) as a read-only view over a
catalog-scoped ``MetricRegistry`` instance built from the single ``MetricSpec``.

The default catalog is registered at import time and then sealed: the module
level ``CATALOG`` is an immutable ``MappingProxyType`` view over the backing
registry, and any further ``register`` attempt fails closed.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Set, Tuple

# QE-P0-01: single authority — re-export the one MetricSpec / MetricRegistry /
# Domain from the registry.  No second definition lives here.
from quant_evaluator.registry.metrics import (
    Domain,
    MetricRegistry,
    MetricSpec,
    MetricStatus,
    MetricTier,
)

__all__ = [
    "Domain",
    "MetricSpec",
    "MetricRegistry",
    "CATALOG",
    "get_metric_spec",
    "get_metric_specs_by_domain",
    "list_all_metric_ids",
    "list_all_domains",
    "seal_metric_registry",
]


# ---------------------------------------------------------------------------
# Module-level registry + CATALOG view.
#
# ``_REGISTRY`` is a catalog-scoped MetricRegistry (the single class) holding
# the 10-domain specs; the module-level ``CATALOG`` is the sealed
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
    *,
    implementation_id: str = "",
    metric_version: str = "0.1.0",
    artifact_kind: str = "scalar",
    required_axes: Tuple[str, ...] = (),
    units: str = "",
    direction: str = "higher_is_better",
    missing_policy: str = "nan",
    numeric_policy: str = "finite",
) -> None:
    """Register a metric spec into the catalog-scoped registry."""
    spec = MetricSpec(
        name=metric_id,
        display_name=metric_id,
        description=description,
        status=MetricStatus.STABLE,
        tier=MetricTier.EXTENDED,
        domain=domain,
        metric_id=metric_id,
        required_inputs=required_inputs,
        output_type=output_type,
        implementation_id=implementation_id,
        metric_version=metric_version,
        artifact_kind=artifact_kind,
        required_axes=required_axes,
        units=units,
        direction=direction,
        missing_policy=missing_policy,
        numeric_policy=numeric_policy,
    )
    _REGISTRY.register(spec)


def seal_metric_registry() -> None:
    """Seal the module-level catalog registry so it becomes immutable.

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
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
)
_register(
    Domain.IC,
    "spearman_ic",
    "Spearman rank correlation between factor values and forward returns",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
)
_register(
    Domain.IC,
    "rank_ic",
    "Rank IC (alias for spearman_ic) averaged cross-sectionally",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
)
_register(
    Domain.IC,
    "ic_summary",
    "Summary statistics (mean, std, skew, kurtosis) of IC time series",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic_summary.compute_rolling_ic_stats",
    metric_version="1.0.0",
    artifact_kind="distribution",
    required_axes=("time",),
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
)

# ---- Domain RANK_IC ----
_register(
    Domain.RANK_IC,
    "rank_ic_time_series",
    "Rank IC computed per time slice, returned as a time series",
    {"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    required_axes=("time",),
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
)
_register(
    Domain.RANK_IC,
    "rank_ic_cross_section",
    "Rank IC computed cross-sectionally for each date",
    {"factor", "forward_returns"},
    output_type="series",
    implementation_id="quant_evaluator.metrics.ic.compute_daily_ic",
    metric_version="1.0.0",
    required_axes=("time",),
    units="correlation",
    direction="higher_is_better",
    missing_policy="drop_pair",
    numeric_policy="finite",
)

# ---- Domain QUANTILE ----
_register(
    Domain.QUANTILE,
    "quantile_returns",
    "Average forward return per quantile bucket",
    {"factor", "forward_returns"},
    output_type="series",
    implementation_id="quant_evaluator.metrics.quantile.compute_quantile_returns",
    metric_version="1.0.0",
    required_axes=("quantile",),
    units="return",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.QUANTILE,
    "quantile_spread",
    "Spread between top and bottom quantile returns",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.quantile.compute_top_bottom_spread",
    metric_version="1.0.0",
    units="return",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.QUANTILE,
    "quantile_stability",
    "Stability of quantile return rankings across time",
    {"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.ic_summary.compute_ic_stability",
    metric_version="1.0.0",
    required_axes=("time",),
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)

# ---- Domain DRAWDOWN ----
_register(
    Domain.DRAWDOWN,
    "max_drawdown",
    "Maximum drawdown of the cumulative IC series",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.portfolio_stats.compute_maximum_drawdown",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.DRAWDOWN,
    "drawdown_duration",
    "Duration (in periods) of the longest drawdown",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.drawdown_analysis.compute_drawdown_duration",
    metric_version="1.0.0",
    units="periods",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.DRAWDOWN,
    "calmar_ratio",
    "Calmar ratio: annualized return / max drawdown",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.portfolio_stats.compute_calmar_ratio",
    metric_version="1.0.0",
    units="ratio",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)

# ---- Domain TURNOVER ----
_register(
    Domain.TURNOVER,
    "turnover_rate",
    "Average rate of change in factor ranking between periods",
    {"factor"},
    implementation_id="quant_evaluator.metrics.turnover.compute_turnover",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TURNOVER,
    "turnover_cost",
    "Estimated transaction cost from factor rebalancing",
    {"factor", "transaction_costs"},
    implementation_id="quant_evaluator.metrics.turnover.compute_weighted_turnover",
    metric_version="1.0.0",
    units="bps",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TURNOVER,
    "turnover_adjusted_ic",
    "IC adjusted for turnover-induced transaction costs",
    {"factor", "forward_returns", "transaction_costs"},
    implementation_id="quant_evaluator.metrics.turnover.compute_turnover_contribution",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)

# ---- Domain TAIL_RISK ----
_register(
    Domain.TAIL_RISK,
    "var_95",
    "Value at Risk at 95% confidence level",
    {"forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.var_cvar.compute_var",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TAIL_RISK,
    "var_99",
    "Value at Risk at 99% confidence level",
    {"forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.var_cvar.compute_var",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TAIL_RISK,
    "cvar_95",
    "Conditional Value at Risk (Expected Shortfall) at 95%",
    {"forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.var_cvar.compute_cvar",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TAIL_RISK,
    "cvar_99",
    "Conditional Value at Risk (Expected Shortfall) at 99%",
    {"forward_returns"},
    implementation_id="quant_evaluator.metrics.risk.var_cvar.compute_cvar",
    metric_version="1.0.0",
    units="fraction",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TAIL_RISK,
    "skewness",
    "Skewness of the return distribution",
    {"forward_returns"},
    implementation_id="quant_evaluator.metrics.distribution.compute_skewness",
    metric_version="1.0.0",
    units="dimensionless",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TAIL_RISK,
    "kurtosis",
    "Excess kurtosis of the return distribution",
    {"forward_returns"},
    implementation_id="quant_evaluator.metrics.distribution.compute_kurtosis",
    metric_version="1.0.0",
    units="dimensionless",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
)

# ---- Domain COVERAGE ----
_register(
    Domain.COVERAGE,
    "factor_coverage",
    "Fraction of universe with non-null factor values",
    {"factor"},
    implementation_id="quant_evaluator.metrics.quality.compute_coverage",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.COVERAGE,
    "return_coverage",
    "Fraction of universe with non-null forward returns",
    {"forward_returns"},
    implementation_id="quant_evaluator.metrics.quality.compute_coverage",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.COVERAGE,
    "joint_coverage",
    "Fraction of universe with both factor and return available",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.quality.compute_coverage_per_factor",
    metric_version="1.0.0",
    units="fraction",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)

# ---- Domain HHI ----
_register(
    Domain.HHI,
    "hhi_concentration",
    "Herfindahl-Hirschman Index of factor value concentration",
    {"factor"},
    implementation_id="quant_evaluator.metrics.exposure.compute_concentration_hhi",
    metric_version="1.0.0",
    units="dimensionless",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.HHI,
    "hhi_effective_n",
    "Effective number of groups (1/HHI) for factor concentration",
    {"factor"},
    implementation_id="quant_evaluator.metrics.exposure.compute_concentration_hhi",
    metric_version="1.0.0",
    units="count",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)

# ---- Domain STABILITY ----
_register(
    Domain.STABILITY,
    "ic_stability",
    "Rolling correlation of IC values across sub-periods",
    {"factor", "forward_returns"},
    implementation_id="quant_evaluator.metrics.ic_summary.compute_ic_stability",
    metric_version="1.0.0",
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.STABILITY,
    "turnover_stability",
    "Variance of turnover rate across periods",
    {"factor"},
    implementation_id="quant_evaluator.metrics.turnover.compute_turnover",
    metric_version="1.0.0",
    units="variance",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.STABILITY,
    "coverage_stability",
    "Variance of factor coverage across periods",
    {"factor"},
    implementation_id="quant_evaluator.metrics.quality.compute_per_time_coverage",
    metric_version="1.0.0",
    units="variance",
    direction="lower_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)

# ---- Domain TEMPORAL ----
_register(
    Domain.TEMPORAL,
    "rolling_ic",
    "Rolling window IC values over time",
    {"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.ic_summary.compute_rolling_ic_stats",
    metric_version="1.0.0",
    required_axes=("time",),
    units="correlation",
    direction="higher_is_better",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TEMPORAL,
    "ic_decay",
    "IC decay: correlation at increasing forward horizons",
    {"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.ic_summary.compute_ic_decay",
    metric_version="1.0.0",
    required_axes=("horizon",),
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
)
_register(
    Domain.TEMPORAL,
    "autocorrelation_ic",
    "Autocorrelation of IC values at specified lags",
    {"factor", "forward_returns"},
    output_type="timeseries",
    implementation_id="quant_evaluator.metrics.temporal.compute_ic_autocorrelation",
    metric_version="1.0.0",
    required_axes=("lag",),
    units="correlation",
    direction="neutral",
    missing_policy="nan",
    numeric_policy="finite",
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
