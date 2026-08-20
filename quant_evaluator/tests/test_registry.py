"""
Tests for metric registry and presets.
"""

import pytest
from quant_evaluator.registry import (
    MetricSpec,
    MetricStatus,
    MetricTier,
    get_metric,
    list_metrics,
    list_metrics_by_status,
    list_metrics_by_tier,
    get_preset,
    list_presets,
    FACTOR_CORE,
    FACTOR_EXTENDED,
    PRODUCTION_DAILY,
)
from quant_evaluator.registry.metrics import register_metric
from quant_evaluator.metrics import (
    compute_coverage,
    compute_ic_std,
    compute_mean_ic_value,
)


class TestMetricSpec:
    """Tests for MetricSpec dataclass."""

    def test_metric_spec_creation(self):
        spec = MetricSpec(
            name="test_metric",
            display_name="Test Metric",
            description="A test metric",
            status=MetricStatus.STABLE,
            tier=MetricTier.CORE,
            requires=["ic_series"],
            min_periods=20,
        )
        assert spec.name == "test_metric"
        assert spec.display_name == "Test Metric"
        assert spec.status == MetricStatus.STABLE
        assert spec.tier == MetricTier.CORE
        assert spec.requires == ["ic_series"]
        assert spec.min_periods == 20

    def test_metric_spec_defaults(self):
        spec = MetricSpec(
            name="minimal",
            display_name="Minimal",
            description="Minimal spec",
            status=MetricStatus.STABLE,
            tier=MetricTier.CORE,
        )
        assert spec.requires == []
        assert spec.min_periods is None
        assert spec.compute_fn is None

    def test_metric_spec_frozen(self):
        spec = MetricSpec(
            name="frozen_test",
            display_name="Frozen Test",
            description="Test immutability",
            status=MetricStatus.STABLE,
            tier=MetricTier.CORE,
        )
        with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError or AttributeError
            spec.name = "new_name"


class TestMetricRegistry:
    """Tests for metric registry functions."""

    def test_list_metrics(self):
        metrics = list_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        assert "mean_ic" in metrics
        assert "ic_std" in metrics
        assert "coverage" in metrics

    def test_get_metric(self):
        spec = get_metric("mean_ic")
        assert spec.name == "mean_ic"
        assert spec.display_name == "Mean IC"
        assert spec.status == MetricStatus.STABLE
        assert spec.tier == MetricTier.CORE
        assert "ic_series" in spec.requires
        assert spec.min_periods == 20

    def test_get_metric_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            get_metric("nonexistent_metric")

    def test_list_metrics_by_status(self):
        stable = list_metrics_by_status(MetricStatus.STABLE)
        experimental = list_metrics_by_status(MetricStatus.EXPERIMENTAL)
        deprecated = list_metrics_by_status(MetricStatus.DEPRECATED)

        assert len(stable) > 0
        assert len(experimental) >= 0
        assert len(deprecated) == 0

        assert "mean_ic" in stable
        assert "ic_ir" in stable

    def test_list_metrics_by_tier(self):
        core = list_metrics_by_tier(MetricTier.CORE)
        extended = list_metrics_by_tier(MetricTier.EXTENDED)
        research = list_metrics_by_tier(MetricTier.RESEARCH)

        assert len(core) > 0
        assert len(extended) > 0
        assert len(research) > 0

        assert "mean_ic" in core
        assert "ic_std" in core
        assert "hac_tstat" in extended
        assert "ic_autocorr_lag1" in extended


class TestCoreMetrics:
    """Tests for registered core metrics."""

    def test_mean_ic_spec(self):
        spec = get_metric("mean_ic")
        assert spec.tier == MetricTier.CORE
        assert spec.status == MetricStatus.STABLE
        assert spec.min_periods == 20
        assert spec.compute_fn is compute_mean_ic_value

    def test_ic_std_spec(self):
        spec = get_metric("ic_std")
        assert spec.tier == MetricTier.CORE
        assert spec.status == MetricStatus.STABLE
        assert spec.compute_fn is compute_ic_std

    def test_ic_ir_spec(self):
        spec = get_metric("ic_ir")
        assert spec.tier == MetricTier.CORE
        assert spec.display_name == "IC Information Ratio"

    def test_coverage_spec(self):
        spec = get_metric("coverage")
        assert spec.tier == MetricTier.CORE
        assert "factor_batch" in spec.requires
        assert spec.min_periods is None
        assert spec.compute_fn is not None
        assert callable(spec.compute_fn)

    def test_turnover_spec(self):
        spec = get_metric("turnover")
        assert spec.tier == MetricTier.CORE
        assert spec.min_periods == 2

    def test_quantile_spread_spec(self):
        spec = get_metric("quantile_spread")
        assert spec.tier == MetricTier.CORE
        assert "factor_batch" in spec.requires
        assert "label_bundle" in spec.requires


class TestExtendedMetrics:
    """Tests for registered extended metrics."""

    def test_hac_tstat_spec(self):
        spec = get_metric("hac_tstat")
        assert spec.tier == MetricTier.EXTENDED
        assert spec.status == MetricStatus.STABLE
        assert spec.min_periods == 30

    def test_subsample_stability_spec(self):
        spec = get_metric("subsample_stability")
        assert spec.tier == MetricTier.EXTENDED
        assert spec.min_periods == 40

    def test_ic_autocorr_lag1_spec(self):
        spec = get_metric("ic_autocorr_lag1")
        assert spec.tier == MetricTier.EXTENDED
        assert "ic_series" in spec.requires

    def test_rank_stability_spec(self):
        spec = get_metric("rank_stability")
        assert spec.tier == MetricTier.EXTENDED
        assert "factor_batch" in spec.requires

    def test_half_life_spec(self):
        spec = get_metric("half_life")
        assert spec.tier == MetricTier.EXTENDED
        assert spec.min_periods == 60


class TestResearchMetrics:
    """Tests for registered research metrics."""

    def test_block_bootstrap_ci_spec(self):
        spec = get_metric("block_bootstrap_ci")
        assert spec.tier == MetricTier.RESEARCH
        # QE-P0-7: bound to a compute_fn adapter, so STABLE is truthful
        assert spec.status == MetricStatus.STABLE
        assert spec.compute_fn is not None

    def test_factor_turnover_rate_spec(self):
        spec = get_metric("factor_turnover_rate")
        assert spec.tier == MetricTier.RESEARCH
        # QE-P0-7: bound to a compute_fn adapter, so STABLE is truthful
        assert spec.status == MetricStatus.STABLE
        assert spec.compute_fn is not None


class TestMetricPresets:
    """Tests for metric presets."""

    def test_list_presets(self):
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) == 3
        assert "factor_core" in presets
        assert "factor_extended" in presets
        assert "production_daily" in presets

    def test_get_preset(self):
        preset = get_preset("factor_core")
        assert preset.name == "factor_core"
        assert preset.display_name == "Factor Core Metrics"
        assert len(preset.metric_names) == 6

    def test_get_preset_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            get_preset("nonexistent_preset")

    def test_factor_core_preset(self):
        assert FACTOR_CORE.name == "factor_core"
        assert len(FACTOR_CORE.metric_names) == 6
        assert "mean_ic" in FACTOR_CORE.metric_names
        assert "ic_std" in FACTOR_CORE.metric_names
        assert "ic_ir" in FACTOR_CORE.metric_names
        assert "coverage" in FACTOR_CORE.metric_names
        assert "turnover" in FACTOR_CORE.metric_names
        assert "quantile_spread" in FACTOR_CORE.metric_names

    def test_factor_extended_preset(self):
        assert FACTOR_EXTENDED.name == "factor_extended"
        assert len(FACTOR_EXTENDED.metric_names) == 11
        # Core metrics included
        assert "mean_ic" in FACTOR_EXTENDED.metric_names
        # Extended metrics included
        assert "hac_tstat" in FACTOR_EXTENDED.metric_names
        assert "subsample_stability" in FACTOR_EXTENDED.metric_names
        assert "ic_autocorr_lag1" in FACTOR_EXTENDED.metric_names
        assert "rank_stability" in FACTOR_EXTENDED.metric_names
        assert "half_life" in FACTOR_EXTENDED.metric_names

    def test_production_daily_preset(self):
        assert PRODUCTION_DAILY.name == "production_daily"
        assert len(PRODUCTION_DAILY.metric_names) == 4
        assert "mean_ic" in PRODUCTION_DAILY.metric_names
        assert "ic_ir" in PRODUCTION_DAILY.metric_names
        assert "coverage" in PRODUCTION_DAILY.metric_names
        assert "turnover" in PRODUCTION_DAILY.metric_names

    def test_preset_get_metrics(self):
        specs = FACTOR_CORE.get_metrics()
        assert len(specs) == 6
        assert all(isinstance(spec, MetricSpec) for spec in specs)
        assert specs[0].name == "mean_ic"

    def test_preset_repr(self):
        repr_str = repr(FACTOR_CORE)
        assert "factor_core" in repr_str
        assert "Factor Core Metrics" in repr_str
        assert "metrics=6" in repr_str


class TestPresetMetricValidity:
    """Tests that all preset metrics are registered."""

    def test_factor_core_all_metrics_exist(self):
        for metric_name in FACTOR_CORE.metric_names:
            spec = get_metric(metric_name)
            assert spec is not None

    def test_factor_extended_all_metrics_exist(self):
        for metric_name in FACTOR_EXTENDED.metric_names:
            spec = get_metric(metric_name)
            assert spec is not None

    def test_production_daily_all_metrics_exist(self):
        for metric_name in PRODUCTION_DAILY.metric_names:
            spec = get_metric(metric_name)
            assert spec is not None


class TestRegistryCoverage:
    """Tests for registry completeness."""

    def test_all_tiers_represented(self):
        core = list_metrics_by_tier(MetricTier.CORE)
        extended = list_metrics_by_tier(MetricTier.EXTENDED)
        research = list_metrics_by_tier(MetricTier.RESEARCH)

        assert len(core) >= 6
        assert len(extended) >= 5
        assert len(research) >= 3

    def test_stable_metrics_count(self):
        stable = list_metrics_by_status(MetricStatus.STABLE)
        assert len(stable) >= 11

    def test_no_duplicate_metrics(self):
        metrics = list_metrics()
        assert len(metrics) == len(set(metrics))
