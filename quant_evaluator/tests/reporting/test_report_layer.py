"""
Tests for the reporting layer.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from quant_evaluator.reporting.chart_spec import ChartSpec
from quant_evaluator.reporting.artifacts import ArtifactStore
from quant_evaluator.reporting.tear_sheet import (
    generate_tear_sheet,
    EvaluationResult,
)
from quant_evaluator.reporting.library_reports import (
    generate_library_report,
    compare_libraries,
    generate_adversarial_report,
)


class TestChartSpec:
    """Tests for ChartSpec dataclass."""

    def test_creation(self):
        """Test ChartSpec creation."""
        spec = ChartSpec(
            title="Test Chart",
            x_label="X",
            y_label="Y",
            chart_type="line",
            data={"series": [1, 2, 3]}
        )
        assert spec.title == "Test Chart"
        assert spec.x_label == "X"
        assert spec.y_label == "Y"
        assert spec.chart_type == "line"
        assert spec.data == {"series": [1, 2, 3]}
        assert spec.content_hash != ""

    def test_content_hash_computation(self):
        """Test that content hash is computed from data."""
        spec1 = ChartSpec(
            title="Chart 1",
            x_label="X",
            y_label="Y",
            chart_type="line",
            data={"series": [1, 2, 3]}
        )
        spec2 = ChartSpec(
            title="Chart 2",  # Different title
            x_label="X",
            y_label="Y",
            chart_type="line",
            data={"series": [1, 2, 3]}  # Same data
        )
        # Same data should produce same hash
        assert spec1.content_hash == spec2.content_hash

    def test_content_hash_different_data(self):
        """Test that different data produces different hashes."""
        spec1 = ChartSpec(
            title="Chart",
            x_label="X",
            y_label="Y",
            chart_type="line",
            data={"series": [1, 2, 3]}
        )
        spec2 = ChartSpec(
            title="Chart",
            x_label="X",
            y_label="Y",
            chart_type="line",
            data={"series": [1, 2, 4]}  # Different data
        )
        assert spec1.content_hash != spec2.content_hash

    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        spec = ChartSpec(
            title="Test Chart",
            x_label="X",
            y_label="Y",
            chart_type="bar",
            data={"categories": ["A", "B"], "values": [10, 20]}
        )
        d = spec.to_dict()
        spec2 = ChartSpec.from_dict(d)
        assert spec.title == spec2.title
        assert spec.x_label == spec2.x_label
        assert spec.y_label == spec2.y_label
        assert spec.chart_type == spec2.chart_type
        assert spec.data == spec2.data
        assert spec.content_hash == spec2.content_hash

    def test_json_roundtrip(self):
        """Test to_json/from_json roundtrip."""
        spec = ChartSpec(
            title="JSON Chart",
            x_label="X",
            y_label="Y",
            chart_type="scatter",
            data={"x": [1, 2], "y": [3, 4]}
        )
        json_str = spec.to_json()
        spec2 = ChartSpec.from_json(json_str)
        assert spec.title == spec2.title
        assert spec.data == spec2.data


class TestArtifactStore:
    """Tests for ArtifactStore."""

    def test_save_and_load(self):
        """Test saving and loading artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            spec = ChartSpec(
                title="Test",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1, 2, 3]}
            )
            artifact_id = store.save(spec)
            loaded = store.load(artifact_id)
            assert loaded is not None
            assert loaded.title == "Test"
            assert loaded.data == {"series": [1, 2, 3]}

    def test_deduplication(self):
        """Test that same content is deduplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            spec1 = ChartSpec(
                title="Chart 1",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1, 2, 3]}
            )
            spec2 = ChartSpec(
                title="Chart 2",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1, 2, 3]}  # Same data
            )
            id1 = store.save(spec1)
            id2 = store.save(spec2)
            # Should return same ID due to deduplication
            assert id1 == id2

    def test_list_artifacts(self):
        """Test listing artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            for i in range(3):
                spec = ChartSpec(
                    title=f"Chart {i}",
                    x_label="X",
                    y_label="Y",
                    chart_type="line",
                    data={"series": [i]}
                )
                store.save(spec)
            artifacts = store.list_artifacts()
            assert len(artifacts) == 3

    def test_get_artifact(self):
        """Test getting artifact metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            spec = ChartSpec(
                title="Metadata Test",
                x_label="X",
                y_label="Y",
                chart_type="bar",
                data={"values": [1, 2]}
            )
            artifact_id = store.save(spec)
            info = store.get_artifact(artifact_id)
            assert info is not None
            assert info.title == "Metadata Test"
            assert info.chart_type == "bar"

    def test_delete_artifact(self):
        """Test deleting artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            spec = ChartSpec(
                title="To Delete",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1]}
            )
            artifact_id = store.save(spec)
            assert store.exists(artifact_id)
            result = store.delete_artifact(artifact_id)
            assert result is True
            assert not store.exists(artifact_id)
            assert store.load(artifact_id) is None

    def test_nonexistent_artifact(self):
        """Test loading nonexistent artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            assert store.load("nonexistent") is None
            assert store.get_artifact("nonexistent") is None
            assert store.delete_artifact("nonexistent") is False


class TestTearSheet:
    """Tests for tear sheet generation."""

    def test_generates_22_panels(self):
        """Test that tear sheet generates 22 panels."""
        result = EvaluationResult(
            ic_series=np.random.randn(100),
            rank_ic_series=np.random.randn(100),
            ic_mean=0.05,
            ic_std=0.1,
            icir=0.5,
            quantile_returns=np.random.randn(100, 5),
            quantile_spread=np.random.randn(100),
            long_short_returns=np.random.randn(100),
            cumulative_returns=np.cumsum(np.random.randn(100)),
            drawdown_series=-np.abs(np.random.randn(100)),
            max_drawdown=-0.15,
            drawdown_durations=np.array([5, 10, 15, 20]),
            turnover_series=np.random.rand(100) * 0.3,
            avg_turnover=0.15,
            coverage_series=np.ones(100) * 0.95,
            avg_coverage=0.95,
            factor_correlation=np.eye(5),
            factor_names=[f"F{i}" for i in range(5)],
            skewness=0.2,
            kurtosis=3.1,
            variance=0.01,
            cvar=-0.05,
            sharpe_ratio=1.2,
            annual_return=0.15,
            annual_volatility=0.12,
            sortino_ratio=1.5,
            calmar_ratio=1.0,
            hhi=np.array([0.2, 0.3, 0.1, 0.25, 0.15]),
            ic_stability=np.random.randn(50),
            rolling_ic_20=np.random.randn(80),
            rolling_ic_60=np.random.randn(40),
            rolling_ic_120=np.random.randn(10),
            ic_decay=np.exp(-np.linspace(0, 2, 20)),
            ic_autocorrelation=np.exp(-np.linspace(0, 3, 10)),
            coverage_heatmap=np.random.rand(50, 20),
            attribution={"Alpha": 0.6, "Beta": 0.3, "Residual": 0.1},
        )

        panels = generate_tear_sheet(result)

        # Should have 22 panels
        assert len(panels) == 22

        # All values should be ChartSpec
        for name, spec in panels.items():
            assert isinstance(spec, ChartSpec)
            assert spec.title != ""
            assert spec.chart_type in ["line", "bar", "scatter", "heatmap"]

    def test_panel_names(self):
        """Test that all expected panel names are present."""
        result = EvaluationResult(
            ic_series=np.random.randn(10),
            rank_ic_series=np.random.randn(10),
            quantile_returns=np.random.randn(10, 5),
            quantile_spread=np.random.randn(10),
            drawdown_series=-np.abs(np.random.randn(10)),
            drawdown_durations=np.array([5, 10]),
            turnover_series=np.random.rand(10),
            coverage_series=np.ones(10),
            factor_correlation=np.eye(3),
            hhi=np.array([0.3, 0.4, 0.3]),
            ic_stability=np.random.randn(10),
            rolling_ic_20=np.random.randn(10),
            ic_decay=np.array([1.0, 0.5, 0.25]),
            ic_autocorrelation=np.array([1.0, 0.5]),
            coverage_heatmap=np.random.rand(10, 5),
            attribution={"A": 0.5, "B": 0.5},
        )

        panels = generate_tear_sheet(result)

        expected_panels = [
            "ic_time_series",
            "ic_distribution",
            "rank_ic_time_series",
            "quantile_returns_bar",
            "quantile_spread_time_series",
            "drawdown_curve",
            "drawdown_duration_histogram",
            "turnover_time_series",
            "coverage_time_series",
            "factor_correlation_heatmap",
            "hhi_bar_chart",
            "ic_stability_scatter",
            "rolling_ic",
            "ic_decay_curve",
            "variance_cvar_bar",
            "skewness_kurtosis_bar",
            "summary_statistics_table",
            "risk_return_scatter",
            "coverage_heatmap",
            "turnover_histogram",
            "ic_autocorrelation",
            "performance_attribution_pie",
        ]

        for panel_name in expected_panels:
            assert panel_name in panels, f"Missing panel: {panel_name}"


class TestLibraryReports:
    """Tests for library-level reports."""

    def test_library_report(self):
        """Test library report generation."""
        results = [
            {"ic_mean": 0.05, "ic_std": 0.1, "icir": 0.5, "sharpe_ratio": 1.0,
             "max_drawdown": -0.1, "avg_turnover": 0.2, "annual_return": 0.1,
             "annual_volatility": 0.15},
            {"ic_mean": 0.03, "ic_std": 0.12, "icir": 0.25, "sharpe_ratio": 0.8,
             "max_drawdown": -0.15, "avg_turnover": 0.25, "annual_return": 0.08,
             "annual_volatility": 0.12},
            {"ic_mean": 0.07, "ic_std": 0.08, "icir": 0.875, "sharpe_ratio": 1.5,
             "max_drawdown": -0.08, "avg_turnover": 0.18, "annual_return": 0.12,
             "annual_volatility": 0.1},
        ]

        report = generate_library_report(results, "Test Library")

        assert report.library_name == "Test Library"
        assert report.n_factors == 3
        assert "mean_ic_mean" in report.summary_stats
        assert "mean_icir" in report.summary_stats
        assert len(report.charts) == 3  # 3 charts generated

    def test_empty_library_report(self):
        """Test library report with empty results."""
        report = generate_library_report([], "Empty Library")
        assert report.n_factors == 0
        assert report.summary_stats == {}

    def test_comparison_report(self):
        """Test comparison report generation."""
        results_a = [
            {"ic_mean": 0.05, "sharpe_ratio": 1.0, "max_drawdown": -0.1},
            {"ic_mean": 0.03, "sharpe_ratio": 0.8, "max_drawdown": -0.15},
        ]
        results_b = [
            {"ic_mean": 0.07, "sharpe_ratio": 1.2, "max_drawdown": -0.08},
            {"ic_mean": 0.06, "sharpe_ratio": 1.1, "max_drawdown": -0.09},
        ]

        report = compare_libraries(results_a, results_b, "Library A", "Library B")

        assert report.library_a_name == "Library A"
        assert report.library_b_name == "Library B"
        assert "library_a" in report.comparison_stats
        assert "library_b" in report.comparison_stats
        assert len(report.charts) == 3  # 3 comparison charts

    def test_adversarial_report(self):
        """Test adversarial report generation."""
        adversarial_results = {
            "stability_scores": {"perturbation_1": 0.9, "perturbation_2": 0.85},
            "robustness_metrics": {"noise_tolerance": 0.8, "outlier_resistance": 0.7},
        }

        report = generate_adversarial_report(adversarial_results, "Noise Test")

        assert report.test_name == "Noise Test"
        assert report.results == adversarial_results
        assert len(report.charts) == 2  # 2 charts generated
