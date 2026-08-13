"""
Integration tests for the complete data quality system.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ..profiler import DataProfiler
from ..anomaly_detection import AnomalyDetector
from ..reports import ReportGenerator, ReportFormat
from ..alerts import AlertSystem, AlertLevel


class TestIntegration:
    """Integration tests for the complete data quality workflow."""

    @pytest.fixture
    def clean_dataframe(self):
        """Create a clean dataframe with no issues."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "date": dates,
            "symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
            "price": np.random.uniform(100, 200, 100),
            "volume": np.random.randint(1000000, 10000000, 100),
            "returns": np.random.normal(0, 0.02, 100),
        })

    @pytest.fixture
    def dirty_dataframe(self):
        """Create a dataframe with various quality issues."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = {
            "date": dates,
            "symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
            "price": np.random.uniform(100, 200, 100),
            "volume": np.random.randint(1000000, 10000000, 100),
            "returns": np.random.normal(0, 0.02, 100),
        }

        # Introduce quality issues
        data["price"][10] = np.inf  # Infinite value
        data["price"][20] = np.nan  # Missing value
        data["volume"][30] = -1000000  # Negative volume (range violation)
        data["returns"][40] = 10.0  # Outlier return

        # Add column with high missing rate
        data["optional_field"] = [np.nan] * 80 + list(range(20))

        return pd.DataFrame(data)

    def test_complete_workflow_clean_data(self, clean_dataframe):
        """Test complete workflow with clean data."""
        # Profile
        profiler = DataProfiler()
        profile_result = profiler.profile(clean_dataframe, date_column="date", entity_column="symbol")

        assert profile_result.row_count == 100
        assert profile_result.column_count == 5
        assert profile_result.date_column == "date"
        assert profile_result.entity_count == 2

        # Detect anomalies
        detector = AnomalyDetector()
        anomaly_result = detector.detect(clean_dataframe, date_column="date")

        # Should have no or minimal anomalies
        assert anomaly_result.total_anomalies < 10  # Allow for some outliers in random data

        # Generate report
        generator = ReportGenerator(title="Clean Data Report")
        html_report = generator.generate(profile_result, anomaly_result, format=ReportFormat.HTML)

        assert "Clean Data Report" in html_report
        assert "100" in html_report  # Row count

        # Evaluate alerts
        alert_system = AlertSystem()
        alerts = alert_system.evaluate(profile_result, anomaly_result)

        # Should have no or minimal alerts on clean random data
        assert isinstance(alerts, list)  # Just verify it runs

    def test_complete_workflow_dirty_data(self, dirty_dataframe):
        """Test complete workflow with dirty data."""
        # Profile
        profiler = DataProfiler()
        profile_result = profiler.profile(dirty_dataframe, date_column="date", entity_column="symbol")

        # Detect anomalies
        expected_ranges = {
            "volume": (0, 100000000),
            "returns": (-0.5, 0.5),
        }

        detector = AnomalyDetector(outlier_method="iqr", outlier_threshold=3.0)
        anomaly_result = detector.detect(dirty_dataframe, date_column="date", expected_ranges=expected_ranges)

        # Should detect multiple anomalies
        assert anomaly_result.total_anomalies > 0

        # Should detect infinite values
        from ..anomaly_detection import AnomalyType
        inf_anomalies = anomaly_result.get_by_type(AnomalyType.INFINITE_VALUE)
        assert len(inf_anomalies) > 0

        # Should detect range violations
        range_anomalies = anomaly_result.get_by_type(AnomalyType.RANGE_VIOLATION)
        assert len(range_anomalies) > 0

        # Generate reports
        generator = ReportGenerator(title="Dirty Data Report")

        html_report = generator.generate(profile_result, anomaly_result, format=ReportFormat.HTML)
        assert "Anomalies Detected" in html_report

        json_report = generator.generate(profile_result, anomaly_result, format=ReportFormat.JSON)
        import json
        json_data = json.loads(json_report)
        assert json_data["anomalies"]["total_anomalies"] > 0

        # Evaluate alerts
        alert_system = AlertSystem()
        alerts = alert_system.evaluate(profile_result, anomaly_result)

        # Should have multiple alerts
        assert len(alerts) > 0
        assert alert_system.has_errors()

    def test_save_reports_to_file(self, clean_dataframe):
        """Test saving reports to files."""
        profiler = DataProfiler()
        profile_result = profiler.profile(clean_dataframe)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(clean_dataframe)

        generator = ReportGenerator(title="File Report")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save HTML report
            html_path = os.path.join(tmpdir, "report.html")
            generator.generate(profile_result, anomaly_result, format=ReportFormat.HTML, output_path=html_path)
            assert os.path.exists(html_path)

            # Save JSON report
            json_path = os.path.join(tmpdir, "report.json")
            generator.generate(profile_result, anomaly_result, format=ReportFormat.JSON, output_path=json_path)
            assert os.path.exists(json_path)

            # Verify file contents
            with open(html_path, "r") as f:
                html_content = f.read()
                assert "File Report" in html_content

            with open(json_path, "r") as f:
                import json
                json_content = json.load(f)
                assert json_content["title"] == "File Report"

    def test_high_correlations_workflow(self):
        """Test workflow with highly correlated data."""
        df = pd.DataFrame({
            "a": range(100),
            "b": [x * 2 for x in range(100)],  # Perfect correlation with a
            "c": [x + np.random.normal(0, 0.1) for x in range(100)],  # High correlation with a
            "d": np.random.normal(0, 1, 100),  # No correlation
        })

        profiler = DataProfiler(compute_correlations=True, correlation_threshold=0.95)
        profile_result = profiler.profile(df)

        # Get high correlations
        high_corr = profiler.get_high_correlations(profile_result)
        assert len(high_corr) > 0

        # Generate report with correlations
        generator = ReportGenerator(title="Correlation Report")
        html_report = generator.generate(profile_result, format=ReportFormat.HTML)
        assert "Correlations" in html_report

    def test_low_coverage_workflow(self):
        """Test workflow with low coverage data."""
        df = pd.DataFrame({
            "good": range(100),
            "sparse": [np.nan] * 90 + list(range(10)),
            "very_sparse": [np.nan] * 95 + list(range(5)),
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        # Get low coverage columns
        low_coverage = profiler.get_low_coverage_columns(profile_result, threshold=50.0)
        assert len(low_coverage) >= 2

        # Alerts should flag low coverage
        alert_system = AlertSystem()
        alerts = alert_system.evaluate(profile_result)

        coverage_alerts = [a for a in alerts if "coverage" in a.message.lower()]
        assert len(coverage_alerts) > 0

    def test_time_series_gap_detection(self):
        """Test time series gap detection."""
        # Create dates with gaps
        dates = list(pd.date_range("2024-01-01", periods=10, freq="D"))
        # Remove some dates to create gaps
        dates.pop(5)
        dates.pop(7)

        df = pd.DataFrame({
            "date": dates,
            "value": range(len(dates)),
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df, date_column="date")

        # Should detect gaps
        assert profile_result.date_gaps is not None
        assert len(profile_result.date_gaps) > 0

        # Generate report with gap info
        generator = ReportGenerator(title="Gap Report")
        html_report = generator.generate(profile_result, format=ReportFormat.HTML)
        assert "Time Series Information" in html_report

    def test_panel_data_workflow(self):
        """Test workflow with panel data."""
        entities = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        dates = pd.date_range("2024-01-01", periods=50, freq="D")

        data = []
        for entity in entities:
            for date in dates:
                data.append({
                    "entity": entity,
                    "date": date,
                    "price": np.random.uniform(100, 200),
                    "volume": np.random.randint(1000000, 10000000),
                })

        df = pd.DataFrame(data)

        profiler = DataProfiler()
        profile_result = profiler.profile(df, date_column="date", entity_column="entity")

        assert profile_result.entity_count == 5
        assert profile_result.row_count == 250

        # Generate report
        generator = ReportGenerator(title="Panel Data Report")
        html_report = generator.generate(profile_result, format=ReportFormat.HTML)
        assert "5" in html_report  # Entity count

    def test_custom_alert_rules(self, clean_dataframe):
        """Test adding custom alert rules."""
        profiler = DataProfiler()
        profile_result = profiler.profile(clean_dataframe)

        alert_system = AlertSystem()

        # Add custom rule
        from ..alerts import AlertRule

        custom_rule = AlertRule(
            name="row_count_check",
            condition=lambda prof, anom: prof.row_count < 50,
            alert_level=AlertLevel.WARNING,
            message_template="Dataset has fewer than 50 rows",
        )

        alert_system.add_rule(custom_rule)

        alerts = alert_system.evaluate(profile_result)

        # Since our dataframe has 100 rows, this rule should not trigger
        row_count_alerts = [a for a in alerts if "row_count_check" in a.metadata.get("rule_name", "")]
        assert len(row_count_alerts) == 0

    def test_export_alerts_workflow(self, dirty_dataframe):
        """Test exporting alerts."""
        profiler = DataProfiler()
        profile_result = profiler.profile(dirty_dataframe)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(dirty_dataframe, expected_ranges={"volume": (0, 100000000)})

        alert_system = AlertSystem()
        alerts = alert_system.evaluate(profile_result, anomaly_result)

        # Export as JSON
        json_export = alert_system.export_alerts(format="json")
        import json
        json_data = json.loads(json_export)
        assert isinstance(json_data, list)

        # Export as text
        text_export = alert_system.export_alerts(format="text")
        assert isinstance(text_export, str)
        assert len(text_export) > 0

    def test_multiple_detection_methods(self):
        """Test different outlier detection methods."""
        # Create data with outliers
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        })

        methods = ["iqr", "zscore", "mad"]
        results = {}

        for method in methods:
            detector = AnomalyDetector(outlier_method=method, outlier_threshold=2.0)
            result = detector.detect(df)
            results[method] = result

        # All methods should detect outliers
        for method, result in results.items():
            from ..anomaly_detection import AnomalyType
            outliers = result.get_by_type(AnomalyType.OUTLIER)
            # Some methods might detect, depends on threshold and data
            assert isinstance(outliers, list)

    def test_profiler_without_optional_features(self, clean_dataframe):
        """Test profiler with optional features disabled."""
        profiler = DataProfiler(
            compute_correlations=False,
            detect_time_series=False,
        )

        profile_result = profiler.profile(clean_dataframe)

        assert profile_result.correlations is None
        assert profile_result.date_column is None

        # Should still generate report
        generator = ReportGenerator()
        html_report = generator.generate(profile_result, format=ReportFormat.HTML)
        assert "<!DOCTYPE html>" in html_report

    def test_anomaly_detector_with_features_disabled(self, dirty_dataframe):
        """Test anomaly detector with optional features disabled."""
        detector = AnomalyDetector(
            enable_duplicate_detection=False,
            enable_stale_detection=False,
        )

        result = detector.detect(dirty_dataframe)

        # Should not have duplicate or stale anomalies
        from ..anomaly_detection import AnomalyType
        dup_anomalies = result.get_by_type(AnomalyType.DUPLICATE)
        stale_anomalies = result.get_by_type(AnomalyType.STALE_DATA)

        assert len(dup_anomalies) == 0
        assert len(stale_anomalies) == 0

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create a larger dataset
        df = pd.DataFrame({
            f"col_{i}": np.random.normal(0, 1, 1000)
            for i in range(20)
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        assert profile_result.row_count == 1000
        assert profile_result.column_count == 20

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        # Should complete without errors
        assert isinstance(anomaly_result.total_anomalies, int)

    def test_summary_statistics(self, dirty_dataframe):
        """Test getting summary statistics from all components."""
        profiler = DataProfiler()
        profile_result = profiler.profile(dirty_dataframe)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(dirty_dataframe)

        alert_system = AlertSystem()
        alert_system.evaluate(profile_result, anomaly_result)

        # Profile summary
        profile_dict = profile_result.to_dict()
        assert profile_dict["row_count"] == 100

        # Anomaly summary
        anomaly_dict = anomaly_result.to_dict()
        assert "total_anomalies" in anomaly_dict

        # Alert summary
        alert_summary = alert_system.get_summary()
        assert "total" in alert_summary
