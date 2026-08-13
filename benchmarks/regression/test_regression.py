#!/usr/bin/env python3
"""
Tests for performance regression tracking system.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from tracker import BenchmarkTracker, BenchmarkResult
from compare import BenchmarkComparator, Comparison
from report import RegressionReporter


class TestBenchmarkTracker:
    """Tests for BenchmarkTracker."""

    def test_store_and_load_result(self, tmp_path: Path):
        """Test storing and loading a single result."""
        tracker = BenchmarkTracker(results_dir=tmp_path)

        result = tracker.store_result(
            benchmark_name="test_bench",
            metric_name="elapsed_time",
            value=1.234,
            unit="seconds",
            metadata={"foo": "bar"},
        )

        assert result.benchmark_name == "test_bench"
        assert result.metric_name == "elapsed_time"
        assert result.value == 1.234
        assert result.unit == "seconds"
        assert result.metadata["foo"] == "bar"

        # Load results
        loaded = tracker.load_results("test_bench")
        assert len(loaded) == 1
        assert loaded[0].value == 1.234

    def test_multiple_results_same_benchmark(self, tmp_path: Path):
        """Test storing multiple results for same benchmark."""
        tracker = BenchmarkTracker(results_dir=tmp_path)

        for i in range(5):
            tracker.store_result(
                benchmark_name="test_bench",
                metric_name="elapsed_time",
                value=float(i),
                unit="seconds",
            )

        loaded = tracker.load_results("test_bench")
        assert len(loaded) == 5
        assert [r.value for r in loaded] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_get_baseline_most_recent(self, tmp_path: Path):
        """Test getting most recent baseline."""
        tracker = BenchmarkTracker(results_dir=tmp_path)

        for i in range(3):
            tracker.store_result(
                benchmark_name="test_bench",
                metric_name="elapsed_time",
                value=float(i),
                unit="seconds",
            )

        baseline = tracker.get_baseline("test_bench")
        assert baseline is not None
        assert baseline.value == 2.0

    def test_get_baseline_no_results(self, tmp_path: Path):
        """Test getting baseline when no results exist."""
        tracker = BenchmarkTracker(results_dir=tmp_path)
        baseline = tracker.get_baseline("nonexistent")
        assert baseline is None

    @patch("tracker.subprocess.check_output")
    def test_git_info_extraction(self, mock_check_output, tmp_path: Path):
        """Test git metadata extraction."""
        mock_check_output.side_effect = [
            "abc123\n",  # commit SHA
            "2026-08-14 10:00:00 +0000\n",  # commit date
            "main\n",  # branch
            "Test Author\n",  # author
        ]

        tracker = BenchmarkTracker(results_dir=tmp_path)
        git_info = tracker.get_git_info()

        assert git_info["commit_sha"] == "abc123"
        assert git_info["branch"] == "main"
        assert git_info["author"] == "Test Author"

    def test_store_benchmark_suite(self, tmp_path: Path):
        """Test storing complete benchmark suite."""
        tracker = BenchmarkTracker(results_dir=tmp_path)

        suite_data = {
            "benchmarks": {
                "qe": {
                    "total_time_s": 10.5,
                    "results": [
                        {
                            "n_factors": 100,
                            "mean_time_s": 2.5,
                            "throughput_factors_per_s": 40.0,
                            "per_factor_ms": 25.0,
                        }
                    ],
                },
                "fp": {
                    "total_time_s": 5.2,
                    "results": {
                        "small": {
                            "total_cells": 10000,
                            "transforms": {
                                "zscore": {"elapsed_s": 0.5},
                                "rank": {"elapsed_s": 0.3},
                            },
                        }
                    },
                },
            }
        }

        results = tracker.store_benchmark_suite(suite_data)

        # Should store: qe total_time, qe_100, fp total_time, fp_small_zscore, fp_small_rank
        assert len(results) == 5

        # Verify individual results are stored
        qe_results = tracker.load_results("qe")
        assert len(qe_results) == 1
        assert qe_results[0].value == 10.5

        fp_results = tracker.load_results("fp_small_zscore")
        assert len(fp_results) == 1
        assert fp_results[0].value == 0.5


class TestBenchmarkComparator:
    """Tests for BenchmarkComparator."""

    def test_compare_values_regression(self, tmp_path: Path):
        """Test detecting performance regression."""
        tracker = BenchmarkTracker(results_dir=tmp_path)
        comparator = BenchmarkComparator(tracker, regression_threshold_pct=10.0)

        comp = comparator._compare_values(
            benchmark_name="test",
            metric_name="elapsed_time",
            current_value=11.0,
            baseline_value=10.0,
            unit="seconds",
            current_commit="abc123",
            baseline_commit="def456",
        )

        assert comp.is_regression is False  # 10% exactly, not > 10%
        assert comp.percent_change == 10.0

        comp2 = comparator._compare_values(
            benchmark_name="test",
            metric_name="elapsed_time",
            current_value=11.1,
            baseline_value=10.0,
            unit="seconds",
            current_commit="abc123",
            baseline_commit="def456",
        )

        assert comp2.is_regression is True  # > 10%

    def test_compare_values_improvement(self, tmp_path: Path):
        """Test detecting performance improvement."""
        tracker = BenchmarkTracker(results_dir=tmp_path)
        comparator = BenchmarkComparator(tracker, regression_threshold_pct=10.0)

        comp = comparator._compare_values(
            benchmark_name="test",
            metric_name="elapsed_time",
            current_value=9.0,
            baseline_value=10.0,
            unit="seconds",
            current_commit="abc123",
            baseline_commit="def456",
        )

        assert comp.is_improvement is True
        assert comp.percent_change == -10.0

    def test_compare_values_throughput(self, tmp_path: Path):
        """Test throughput metric (higher is better)."""
        tracker = BenchmarkTracker(results_dir=tmp_path)
        comparator = BenchmarkComparator(tracker, regression_threshold_pct=10.0)

        # Lower throughput = regression
        comp = comparator._compare_values(
            benchmark_name="test",
            metric_name="throughput",
            current_value=90.0,
            baseline_value=100.0,
            unit="items_per_second",
            current_commit="abc123",
            baseline_commit="def456",
        )

        assert comp.is_regression is False  # Exactly -10%, not < -10%
        assert comp.percent_change == -10.0

        # Higher throughput = improvement
        comp2 = comparator._compare_values(
            benchmark_name="test",
            metric_name="throughput",
            current_value=110.0,
            baseline_value=100.0,
            unit="items_per_second",
            current_commit="abc123",
            baseline_commit="def456",
        )

        assert comp2.is_improvement is True

    def test_compare_suite_no_baseline(self, tmp_path: Path):
        """Test comparison when no baseline exists."""
        tracker = BenchmarkTracker(results_dir=tmp_path)
        comparator = BenchmarkComparator(tracker)

        current_data = {
            "benchmarks": {
                "qe": {
                    "total_time_s": 10.5,
                },
            }
        }

        comparisons = comparator.compare_suite(current_data)
        assert len(comparisons) == 0  # No baseline to compare

    def test_compare_suite_with_baseline(self, tmp_path: Path):
        """Test comparison with existing baseline."""
        tracker = BenchmarkTracker(results_dir=tmp_path)

        # Store baseline
        tracker.store_result(
            benchmark_name="qe",
            metric_name="total_time",
            value=10.0,
            unit="seconds",
        )

        # Compare current
        comparator = BenchmarkComparator(tracker, regression_threshold_pct=10.0)
        current_data = {
            "benchmarks": {
                "qe": {
                    "total_time_s": 12.0,  # 20% slower
                },
            }
        }

        comparisons = comparator.compare_suite(current_data)
        assert len(comparisons) == 1
        assert comparisons[0].is_regression is True
        assert comparisons[0].percent_change == 20.0

    def test_get_regressions_and_improvements(self, tmp_path: Path):
        """Test filtering regressions and improvements."""
        tracker = BenchmarkTracker(results_dir=tmp_path)
        comparator = BenchmarkComparator(tracker)

        comparisons = [
            Comparison(
                benchmark_name="test1",
                metric_name="elapsed_time",
                current_value=12.0,
                baseline_value=10.0,
                delta=2.0,
                percent_change=20.0,
                unit="seconds",
                is_regression=True,
                current_commit="abc",
                baseline_commit="def",
            ),
            Comparison(
                benchmark_name="test2",
                metric_name="elapsed_time",
                current_value=8.0,
                baseline_value=10.0,
                delta=-2.0,
                percent_change=-20.0,
                unit="seconds",
                is_regression=False,
                current_commit="abc",
                baseline_commit="def",
            ),
        ]

        regressions = comparator.get_regressions(comparisons)
        assert len(regressions) == 1
        assert regressions[0].benchmark_name == "test1"

        improvements = comparator.get_improvements(comparisons)
        assert len(improvements) == 1
        assert improvements[0].benchmark_name == "test2"


class TestRegressionReporter:
    """Tests for RegressionReporter."""

    def test_build_time_series(self, tmp_path: Path):
        """Test building time series data."""
        tracker = BenchmarkTracker(results_dir=tmp_path)

        # Store multiple results
        for i in range(3):
            tracker.store_result(
                benchmark_name="test_bench",
                metric_name="elapsed_time",
                value=float(i),
                unit="seconds",
            )

        reporter = RegressionReporter(tracker)
        series = reporter.build_time_series("test_bench")

        assert len(series) == 3
        assert series[0].value == 0.0
        assert series[1].value == 1.0
        assert series[2].value == 2.0

    def test_generate_markdown_report(self, tmp_path: Path):
        """Test markdown report generation."""
        tracker = BenchmarkTracker(results_dir=tmp_path)
        reporter = RegressionReporter(tracker)

        comparisons = [
            {
                "benchmark_name": "test1",
                "metric_name": "elapsed_time",
                "current_value": 12.0,
                "baseline_value": 10.0,
                "percent_change": 20.0,
                "is_regression": True,
                "is_improvement": False,
            },
            {
                "benchmark_name": "test2",
                "metric_name": "elapsed_time",
                "current_value": 8.0,
                "baseline_value": 10.0,
                "percent_change": -20.0,
                "is_regression": False,
                "is_improvement": True,
            },
        ]

        output_path = tmp_path / "report.md"
        reporter.generate_markdown_report(comparisons, output_path)

        content = output_path.read_text()
        assert "# Performance Regression Report" in content
        assert "🔴 Regressions Detected" in content
        assert "🟢 Performance Improvements" in content
        assert "test1" in content
        assert "test2" in content

    def test_generate_html_report(self, tmp_path: Path):
        """Test HTML report generation."""
        tracker = BenchmarkTracker(results_dir=tmp_path)

        # Store some time series data
        for i in range(3):
            tracker.store_result(
                benchmark_name="test_bench",
                metric_name="elapsed_time",
                value=float(i),
                unit="seconds",
            )

        reporter = RegressionReporter(tracker)

        comparisons = [
            {
                "benchmark_name": "test_bench",
                "metric_name": "elapsed_time",
                "current_value": 3.0,
                "baseline_value": 2.0,
                "percent_change": 50.0,
                "is_regression": True,
                "is_improvement": False,
            },
        ]

        output_path = tmp_path / "report.html"
        reporter.generate_html_report(comparisons, output_path)

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Performance Regression Report" in content
        assert "test_bench" in content
        assert "Chart.js" in content or "chart" in content.lower()


@pytest.fixture
def tmp_path():
    """Provide temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
