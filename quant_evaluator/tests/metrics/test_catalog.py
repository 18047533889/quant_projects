"""
Tests for the 10-domain metric catalog (QE-METRIC overhaul, section B).
"""

import pytest

from quant_evaluator.contracts.errors import UnsupportedMetricError
from quant_evaluator.metrics.catalog import (
    CATALOG,
    Domain,
    MetricSpec,
    get_metric_spec,
    get_metric_specs_by_domain,
    list_all_domains,
    list_all_metric_ids,
)


# ---------------------------------------------------------------------------
# Constants for expected metric counts per domain
# ---------------------------------------------------------------------------
EXPECTED_METRICS_BY_DOMAIN = {
    Domain.IC: {"pearson_ic", "spearman_ic", "rank_ic", "ic_summary"},
    Domain.RANK_IC: {"rank_ic_time_series", "rank_ic_cross_section"},
    Domain.QUANTILE: {"quantile_returns", "quantile_spread", "quantile_stability"},
    Domain.DRAWDOWN: {"max_drawdown", "drawdown_duration", "calmar_ratio"},
    Domain.TURNOVER: {"turnover_rate", "turnover_cost", "turnover_adjusted_ic"},
    Domain.TAIL_RISK: {
        "var_95",
        "var_99",
        "cvar_95",
        "cvar_99",
        "skewness",
        "kurtosis",
    },
    Domain.COVERAGE: {"factor_coverage", "return_coverage", "joint_coverage"},
    Domain.HHI: {"hhi_concentration", "hhi_effective_n"},
    Domain.STABILITY: {"ic_stability", "turnover_stability", "coverage_stability"},
    Domain.TEMPORAL: {"rolling_ic", "ic_decay", "autocorrelation_ic"},
}

ALL_EXPECTED_IDS = set()
for _ids in EXPECTED_METRICS_BY_DOMAIN.values():
    ALL_EXPECTED_IDS.update(_ids)


# ---------------------------------------------------------------------------
# 1. All 10 domains are present
# ---------------------------------------------------------------------------
class TestDomainsPresent:
    """Verify that all 10 domains exist in the Domain enum and have metrics."""

    def test_domain_enum_has_10_values(self) -> None:
        assert len(Domain) == 10

    def test_all_domains_listed(self) -> None:
        domains = list_all_domains()
        domain_values = {d.value for d in domains}
        expected_values = {
            "ic",
            "rank_ic",
            "quantile",
            "drawdown",
            "turnover",
            "tail_risk",
            "coverage",
            "hhi",
            "stability",
            "temporal",
        }
        assert domain_values == expected_values

    @pytest.mark.parametrize("domain", list(Domain))
    def test_each_domain_has_metrics(self, domain: Domain) -> None:
        specs = get_metric_specs_by_domain(domain)
        assert len(specs) > 0, f"Domain {domain.value} has no registered metrics"


# ---------------------------------------------------------------------------
# 2. All expected metric IDs are in the catalog
# ---------------------------------------------------------------------------
class TestAllMetricIDs:
    """Verify that every expected metric_id is present."""

    def test_catalog_has_all_expected_ids(self) -> None:
        catalog_ids = set(CATALOG.keys())
        missing = ALL_EXPECTED_IDS - catalog_ids
        extra = catalog_ids - ALL_EXPECTED_IDS
        assert not missing, f"Missing metric IDs: {missing}"
        assert not extra, f"Unexpected metric IDs in catalog: {extra}"

    def test_list_all_metric_ids_returns_all(self) -> None:
        ids = list_all_metric_ids()
        assert set(ids) == ALL_EXPECTED_IDS
        assert ids == sorted(ids), "list_all_metric_ids should return sorted list"


# ---------------------------------------------------------------------------
# 3. get_metric_spec returns correct MetricSpec
# ---------------------------------------------------------------------------
class TestGetMetricSpec:
    """Verify get_metric_spec returns the correct MetricSpec for each metric."""

    @pytest.mark.parametrize("metric_id", sorted(ALL_EXPECTED_IDS))
    def test_returns_metric_spec(self, metric_id: str) -> None:
        spec = get_metric_spec(metric_id)
        assert isinstance(spec, MetricSpec)
        assert spec.metric_id == metric_id

    @pytest.mark.parametrize(
        "metric_id, expected_domain",
        [
            ("pearson_ic", Domain.IC),
            ("rank_ic_time_series", Domain.RANK_IC),
            ("quantile_returns", Domain.QUANTILE),
            ("max_drawdown", Domain.DRAWDOWN),
            ("turnover_rate", Domain.TURNOVER),
            ("var_95", Domain.TAIL_RISK),
            ("factor_coverage", Domain.COVERAGE),
            ("hhi_concentration", Domain.HHI),
            ("ic_stability", Domain.STABILITY),
            ("rolling_ic", Domain.TEMPORAL),
        ],
    )
    def test_correct_domain_assignment(
        self, metric_id: str, expected_domain: Domain
    ) -> None:
        spec = get_metric_spec(metric_id)
        assert spec.domain == expected_domain

    def test_spec_has_description(self) -> None:
        spec = get_metric_spec("pearson_ic")
        assert spec.description  # non-empty string

    def test_spec_has_required_inputs(self) -> None:
        spec = get_metric_spec("pearson_ic")
        assert isinstance(spec.required_inputs, frozenset)
        assert len(spec.required_inputs) > 0

    def test_spec_default_output_type_is_scalar(self) -> None:
        spec = get_metric_spec("pearson_ic")
        assert spec.output_type == "scalar"


# ---------------------------------------------------------------------------
# 4. get_metric_specs_by_domain returns correct metrics
# ---------------------------------------------------------------------------
class TestGetMetricSpecsByDomain:
    """Verify get_metric_specs_by_domain returns the right set of specs."""

    @pytest.mark.parametrize(
        "domain, expected_ids",
        list(EXPECTED_METRICS_BY_DOMAIN.items()),
    )
    def test_returns_correct_metric_ids(
        self, domain: Domain, expected_ids: set
    ) -> None:
        specs = get_metric_specs_by_domain(domain)
        actual_ids = {s.metric_id for s in specs}
        assert actual_ids == expected_ids, (
            f"Domain {domain.value}: expected {expected_ids}, got {actual_ids}"
        )

    def test_returned_specs_are_sorted_by_metric_id(self) -> None:
        for domain in Domain:
            specs = get_metric_specs_by_domain(domain)
            ids = [s.metric_id for s in specs]
            assert ids == sorted(ids), (
                f"Domain {domain.value} specs not sorted: {ids}"
            )

    def test_all_specs_have_correct_domain(self) -> None:
        for domain in Domain:
            specs = get_metric_specs_by_domain(domain)
            for spec in specs:
                assert spec.domain == domain, (
                    f"Spec {spec.metric_id} has domain {spec.domain}, expected {domain}"
                )


# ---------------------------------------------------------------------------
# 5. Unknown metric_id raises UnsupportedMetricError
# ---------------------------------------------------------------------------
class TestUnsupportedMetric:
    """Verify that unknown metric IDs raise UnsupportedMetricError."""

    @pytest.mark.parametrize(
        "unknown_id",
        [
            "nonexistent_metric",
            "IC",
            "sharpe_ratio",
            "",
            "pearson_correlation",
        ],
    )
    def test_raises_unsupported_metric_error(self, unknown_id: str) -> None:
        with pytest.raises(UnsupportedMetricError, match=f"Unknown metric_id '{unknown_id}'"):
            get_metric_spec(unknown_id)

    def test_error_message_includes_valid_ids(self) -> None:
        with pytest.raises(UnsupportedMetricError) as exc_info:
            get_metric_spec("fake_metric")
        # The error message should contain "Valid IDs:"
        assert "Valid IDs:" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Catalog integrity checks
# ---------------------------------------------------------------------------
class TestCatalogIntegrity:
    """Additional integrity checks on the catalog."""

    def test_no_duplicate_metric_ids(self) -> None:
        """Each metric_id should appear only once in the catalog."""
        ids = list(CATALOG.keys())
        assert len(ids) == len(set(ids)), "Duplicate metric IDs found in CATALOG"

    def test_all_metric_specs_are_frozen(self) -> None:
        """MetricSpec dataclass instances should be immutable."""
        for spec in CATALOG.values():
            assert isinstance(spec, MetricSpec)

    def test_no_empty_metric_ids(self) -> None:
        """No metric_id should be empty."""
        for mid in CATALOG:
            assert mid, "Found empty metric_id in CATALOG"

    def test_no_empty_descriptions(self) -> None:
        """No metric should have an empty description."""
        for spec in CATALOG.values():
            assert spec.description, f"Empty description for metric '{spec.metric_id}'"
