"""
Tests for the institutional tear-sheet panel set.

Verifies that the 22 institutional panels render ONLY from pre-computed
MetricArtifacts and that an absent source artifact produces a NOT_COMPUTED
placeholder - never a 0.0.
"""

import os
import sys

import pytest

# Add repo root to path for imports (matches test_chart_spec.py convention)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quant_evaluator.reporting.tear_sheet import (
    EvaluationResult,
    generate_tear_sheet,
    MetricArtifact,
    metric_artifact,
    NOT_COMPUTED,
    INSTITUTIONAL_PANELS,
)

from quant_evaluator.reporting.chart_spec import ChartSpec


# The institutional panels required by the spec.
REQUIRED_PANELS = [
    "year_month_ic_heatmap",
    "rolling_rank_ic",
    "ic_recent_vs_full",
    "ic_positive_ratio_sign_survival",
    "ic_hac_confidence_band",
    "ic_horizon_surface",
    "quantile_monotonicity",
    "quantile_curvature",
    "tail_asymmetry",
    "cumulative_quantile_return",
    "cumulative_top_bottom_spread",
    "top_bottom_drawdown",
    "size_liquidity_heatmap",
    "regime_heatmap",
    "cost_sensitivity_curve",
    "delay_sensitivity_curve",
    "industry_style_exposure",
    "exposure_drift",
    "raw_vs_neutralized_ic",
    "missing_staleness_timeline",
    "spec_robustness_cube",
    "data_quality_missingness",
]


class TestInstitutionalPanels:
    """Tests for the institutional 22-panel tear-sheet set."""

    def test_registry_contains_all_22_panels(self):
        """The INSTITUTIONAL_PANELS registry must list all 22 panels."""
        keys = {p[0] for p in INSTITUTIONAL_PANELS}
        for name in REQUIRED_PANELS:
            assert name in keys, f"missing panel in registry: {name}"
        assert len(keys) == 22

    def test_all_panels_render_not_computed_when_artifacts_absent(self):
        """Every institutional panel must render NOT_COMPUTED (never 0.0)
        when its source artifact is absent."""
        result = EvaluationResult()
        panels = generate_tear_sheet(result)
        for name in REQUIRED_PANELS:
            assert name in panels, f"panel not generated: {name}"
            spec = panels[name]
            assert isinstance(spec, ChartSpec)
            # Never a 0.0 -- must be an explicit placeholder status.
            assert spec.data.get("status") == NOT_COMPUTED, (
                f"panel {name} should be NOT_COMPUTED, got data={spec.data}"
            )
            assert spec.data.get("panel") == name

    def test_more_than_ten_specific_panels_are_not_computed(self):
        """At least 10 specific panels render NOT_COMPUTED when artifacts are
        missing (spec requires >=10)."""
        result = EvaluationResult()
        panels = generate_tear_sheet(result)
        not_computed = [
            name for name in REQUIRED_PANELS
            if panels[name].data.get("status") == NOT_COMPUTED
        ]
        assert len(not_computed) >= 10
        assert len(not_computed) == len(REQUIRED_PANELS)

    def test_panels_render_from_artifact_when_present(self):
        """Panels render real chart data when the MetricArtifact is present."""
        result = EvaluationResult()
        result.artifacts = {
            "year_month_ic_heatmap": metric_artifact(
                "year_month_ic_heatmap",
                metric_type="heatmap",
                matrix=[[0.05, 0.06], [0.07, 0.08]],
                years=[2024, 2025],
            ),
            "rolling_rank_ic": metric_artifact(
                "rolling_rank_ic",
                windows={"RankIC60": [0.04, 0.05, 0.06]},
            ),
            "quantile_monotonicity": metric_artifact(
                "quantile_monotonicity",
                metric_type="bar",
                categories=["Q1", "Q2", "Q3", "Q4", "Q5"],
                values=[-0.02, -0.005, 0.004, 0.02, 0.035],
            ),
            "ic_horizon_surface": metric_artifact(
                "ic_horizon_surface",
                horizons=[1, 2, 3, 5, 10, 20, 60],
                ics=[0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02],
            ),
            "spec_robustness_cube": metric_artifact(
                "spec_robustness_cube",
                metric_type="scatter",
                points=[[1, 2, 0.1], [2, 3, 0.15]],
            ),
        }
        panels = generate_tear_sheet(result)
        assert panels["year_month_ic_heatmap"].chart_type == "heatmap"
        assert panels["year_month_ic_heatmap"].data["matrix"] == [
            [0.05, 0.06], [0.07, 0.08]
        ]
        assert panels["year_month_ic_heatmap"].source_artifact_refs == (
            "year_month_ic_heatmap",
        )
        assert panels["rolling_rank_ic"].data.get("timestamps") == [0, 1, 2]
        assert panels["quantile_monotonicity"].chart_type == "bar"
        assert panels["quantile_monotonicity"].data["values"] == [
            -0.02, -0.005, 0.004, 0.02, 0.035
        ]
        assert panels["ic_horizon_surface"].data["horizons"] == [
            1, 2, 3, 5, 10, 20, 60
        ]
        assert panels["spec_robustness_cube"].chart_type == "scatter"

    def test_mixed_artifacts_present_and_absent(self):
        """Present artifacts render real panels; absent ones stay NOT_COMPUTED."""
        result = EvaluationResult()
        result.artifacts = {
            "top_bottom_drawdown": metric_artifact(
                "top_bottom_drawdown",
                timestamps=[0, 1, 2],
                series={"Drawdown": [-0.0, -0.05, -0.12]},
            ),
        }
        panels = generate_tear_sheet(result)
        assert panels["top_bottom_drawdown"].data.get("status") is None
        assert panels["top_bottom_drawdown"].data["series"] == {
            "Drawdown": [-0.0, -0.05, -0.12]
        }
        # A genuinely absent panel stays NOT_COMPUTED (never 0.0).
        assert panels["regime_heatmap"].data.get("status") == NOT_COMPUTED
        assert panels["regime_heatmap"].data.get("matrix", "no matrix") != 0.0

    def test_metric_artifact_dataclass(self):
        """MetricArtifact constructor and convenience builder work."""
        art = metric_artifact("ic_recent_vs_full", metric_type="bar",
                              windows=["60d", "120d", "252d"],
                              ic=[0.06, 0.055, 0.05])
        assert isinstance(art, MetricArtifact)
        assert art.name == "ic_recent_vs_full"
        assert art.metric_type == "bar"
        assert art.data["ic"] == [0.06, 0.055, 0.05]
