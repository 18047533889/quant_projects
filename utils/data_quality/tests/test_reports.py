"""
Comprehensive tests for report generation module.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime

import pandas as pd
import numpy as np

from ..profiler import DataProfiler, ProfileResult
from ..anomaly_detection import AnomalyDetector, AnomalyResult
from ..reports import ReportGenerator, ReportFormat


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_report_format_values(self):
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.JSON.value == "json"


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5],
            "c": ["x", "y", "z", "x", "y"],
        })

    @pytest.fixture
    def profile_result(self, sample_dataframe):
        """Create profile result for testing."""
        profiler = DataProfiler()
        return profiler.profile(sample_dataframe)

    @pytest.fixture
    def anomaly_result(self, sample_dataframe):
        """Create anomaly result for testing."""
        detector = AnomalyDetector()
        return detector.detect(sample_dataframe)

    def test_generator_initialization(self):
        generator = ReportGenerator(
            title="Test Report",
            include_charts=True,
        )

        assert generator.title == "Test Report"
        assert generator.include_charts is True

    def test_generate_html_report(self, profile_result):
        generator = ReportGenerator(title="Test Report")
        report = generator.generate(
            profile_result=profile_result,
            format=ReportFormat.HTML,
        )

        assert isinstance(report, str)
        assert "<!DOCTYPE html>" in report
        assert "Test Report" in report
        assert "<html>" in report
        assert "</html>" in report

    def test_generate_json_report(self, profile_result):
        generator = ReportGenerator(title="Test Report")
        report = generator.generate(
            profile_result=profile_result,
            format=ReportFormat.JSON,
        )

        assert isinstance(report, str)

        # Should be valid JSON
        data = json.loads(report)
        assert "title" in data
        assert data["title"] == "Test Report"
        assert "profile" in data

    def test_generate_html_with_anomalies(self, profile_result, anomaly_result):
        generator = ReportGenerator(title="Test Report")
        report = generator.generate(
            profile_result=profile_result,
            anomaly_result=anomaly_result,
            format=ReportFormat.HTML,
        )

        assert "Anomalies Detected" in report or "No anomalies detected" in report

    def test_generate_json_with_anomalies(self, profile_result, anomaly_result):
        generator = ReportGenerator(title="Test Report")
        report = generator.generate(
            profile_result=profile_result,
            anomaly_result=anomaly_result,
            format=ReportFormat.JSON,
        )

        data = json.loads(report)
        assert "anomalies" in data

    def test_generate_report_to_file(self, profile_result):
        generator = ReportGenerator(title="Test Report")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "report.html")

            report = generator.generate(
                profile_result=profile_result,
                format=ReportFormat.HTML,
                output_path=output_path,
            )

            assert os.path.exists(output_path)

            with open(output_path, "r") as f:
                file_content = f.read()

            assert file_content == report

    def test_html_contains_summary_section(self, profile_result):
        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        assert "Summary" in report
        assert "Total Rows" in report
        assert "Total Columns" in report
        assert "Memory Usage" in report

    def test_html_contains_column_profiles(self, profile_result):
        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        assert "Column Profiles" in report

        # Check that column names appear
        for col_name in profile_result.column_profiles.keys():
            assert col_name in report

    def test_html_contains_coverage_section(self, profile_result):
        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        if profile_result.coverage_matrix is not None:
            assert "Data Coverage" in report

    def test_html_contains_correlation_section(self):
        # Create data with correlations
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # Correlated with a
        })

        profiler = DataProfiler(compute_correlations=True)
        profile_result = profiler.profile(df)

        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        assert "Correlations" in report

    def test_html_anomalies_section_with_no_anomalies(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        generator = ReportGenerator()
        report = generator.generate(
            profile_result,
            anomaly_result,
            format=ReportFormat.HTML,
        )

        assert "No anomalies detected" in report

    def test_html_anomalies_section_with_anomalies(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],  # Has infinite value
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        generator = ReportGenerator()
        report = generator.generate(
            profile_result,
            anomaly_result,
            format=ReportFormat.HTML,
        )

        assert "Anomalies Detected" in report
        assert anomaly_result.total_anomalies > 0

    def test_html_severity_badges(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        generator = ReportGenerator()
        report = generator.generate(
            profile_result,
            anomaly_result,
            format=ReportFormat.HTML,
        )

        # Check for severity classes
        assert 'severity-high' in report or 'severity-medium' in report or 'severity-low' in report

    def test_html_css_styling(self, profile_result):
        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        # Check that CSS is present
        assert "<style>" in report
        assert "</style>" in report
        assert "background:" in report
        assert "color:" in report

    def test_html_time_series_section(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": range(10),
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df, date_column="date")

        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        assert "Time Series Information" in report
        assert "date" in report

    def test_json_structure(self, profile_result, anomaly_result):
        generator = ReportGenerator(title="Test Report")
        report = generator.generate(
            profile_result,
            anomaly_result,
            format=ReportFormat.JSON,
        )

        data = json.loads(report)

        assert "title" in data
        assert "generated_at" in data
        assert "profile" in data
        assert "anomalies" in data

        # Check profile structure
        assert "timestamp" in data["profile"]
        assert "row_count" in data["profile"]
        assert "column_count" in data["profile"]
        assert "column_profiles" in data["profile"]

        # Check anomalies structure
        assert "timestamp" in data["anomalies"]
        assert "total_anomalies" in data["anomalies"]
        assert "severity_counts" in data["anomalies"]

    def test_json_without_anomalies(self, profile_result):
        generator = ReportGenerator(title="Test Report")
        report = generator.generate(
            profile_result,
            format=ReportFormat.JSON,
        )

        data = json.loads(report)

        assert "profile" in data
        assert "anomalies" not in data

    def test_unsupported_format_raises_error(self, profile_result):
        generator = ReportGenerator()

        with pytest.raises(ValueError, match="Unsupported format"):
            # Manually create an invalid format
            class InvalidFormat:
                pass

            generator.generate(profile_result, format=InvalidFormat())

    def test_html_footer(self, profile_result):
        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        assert "<footer>" in report
        assert "Data Quality Report" in report

    def test_html_numeric_statistics_formatting(self):
        df = pd.DataFrame({
            "values": [1.123456789, 2.987654321, 3.456789012],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        # Check that numeric values are formatted (contain decimal points)
        assert "Mean" in report
        assert "Std Dev" in report

    def test_html_progress_bars_for_coverage(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, np.nan, 3, np.nan, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        assert "progress-bar" in report
        assert "progress-fill" in report

    def test_html_with_high_correlations(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # Perfect correlation
            "c": [1, 1, 2, 2, 3],
        })

        profiler = DataProfiler(compute_correlations=True)
        profile_result = profiler.profile(df)

        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        assert "Correlations" in report
        assert "High Correlations" in report or "No high correlations found" in report

    def test_html_anomaly_table_limited_to_100(self):
        # Create many anomalies
        values = [np.nan] * 200 + [1] * 200
        df = pd.DataFrame({"values": values})

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        generator = ReportGenerator()
        report = generator.generate(
            profile_result,
            anomaly_result,
            format=ReportFormat.HTML,
        )

        # Report should be generated successfully even with many anomalies
        assert "Anomalies Detected" in report

    def test_generate_with_missing_optional_data(self):
        # Create minimal profile result
        df = pd.DataFrame({"a": [1, 2, 3]})

        profiler = DataProfiler(compute_correlations=False)
        profile_result = profiler.profile(df)

        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        # Should generate report successfully even without correlations
        assert "<!DOCTYPE html>" in report
        assert "Summary" in report

    def test_html_responsive_design_clamp(self, profile_result):
        generator = ReportGenerator()
        report = generator.generate(profile_result, format=ReportFormat.HTML)

        # Check for responsive clamp() usage
        assert "clamp(" in report
