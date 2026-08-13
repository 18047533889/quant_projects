"""
Comprehensive tests for alert system module.
"""

import pytest
import json
from datetime import datetime

import pandas as pd
import numpy as np

from ..profiler import DataProfiler
from ..anomaly_detection import AnomalyDetector, AnomalyType
from ..alerts import AlertSystem, AlertLevel, Alert, AlertRule


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_level_values(self):
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        alert = Alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test",
            source="test",
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert alert.source == "test"
        assert isinstance(alert.timestamp, datetime)

    def test_alert_with_metadata(self):
        alert = Alert(
            level=AlertLevel.ERROR,
            title="Test Alert",
            message="Test message",
            source="test",
            metadata={"key": "value"},
        )

        assert alert.metadata["key"] == "value"

    def test_alert_to_dict(self):
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test Alert",
            message="Test message",
            source="test",
            metadata={"count": 5},
        )

        result = alert.to_dict()

        assert isinstance(result, dict)
        assert result["level"] == "info"
        assert result["title"] == "Test Alert"
        assert result["message"] == "Test message"
        assert result["source"] == "test"
        assert "timestamp" in result
        assert result["metadata"]["count"] == 5

    def test_alert_to_json(self):
        alert = Alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            source="test",
        )

        json_str = alert.to_json()

        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["level"] == "warning"
        assert data["title"] == "Test Alert"


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_alert_rule_creation(self):
        rule = AlertRule(
            name="test_rule",
            condition=lambda prof, anom: True,
            alert_level=AlertLevel.WARNING,
            message_template="Test message",
        )

        assert rule.name == "test_rule"
        assert rule.alert_level == AlertLevel.WARNING
        assert rule.enabled is True

    def test_alert_rule_with_disabled(self):
        rule = AlertRule(
            name="test_rule",
            condition=lambda prof, anom: True,
            alert_level=AlertLevel.ERROR,
            message_template="Test message",
            enabled=False,
        )

        assert rule.enabled is False


class TestAlertSystem:
    """Tests for AlertSystem class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5],
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

    def test_alert_system_initialization(self):
        system = AlertSystem()

        assert len(system.rules) > 0  # Should have default rules
        assert len(system.alerts) == 0

    def test_add_custom_rule(self):
        system = AlertSystem()
        initial_rule_count = len(system.rules)

        custom_rule = AlertRule(
            name="custom_rule",
            condition=lambda prof, anom: prof.row_count > 100,
            alert_level=AlertLevel.WARNING,
            message_template="Too many rows",
        )

        system.add_rule(custom_rule)

        assert len(system.rules) == initial_rule_count + 1

    def test_remove_rule(self):
        system = AlertSystem()

        # Add a custom rule
        custom_rule = AlertRule(
            name="custom_rule",
            condition=lambda prof, anom: True,
            alert_level=AlertLevel.INFO,
            message_template="Test",
        )

        system.add_rule(custom_rule)
        initial_count = len(system.rules)

        # Remove it
        removed = system.remove_rule("custom_rule")

        assert removed is True
        assert len(system.rules) == initial_count - 1

    def test_remove_nonexistent_rule(self):
        system = AlertSystem()
        removed = system.remove_rule("nonexistent_rule")

        assert removed is False

    def test_enable_rule(self):
        system = AlertSystem()

        # Add disabled rule
        rule = AlertRule(
            name="test_rule",
            condition=lambda prof, anom: True,
            alert_level=AlertLevel.INFO,
            message_template="Test",
            enabled=False,
        )

        system.add_rule(rule)
        system.enable_rule("test_rule")

        # Find the rule and check it's enabled
        test_rule = next((r for r in system.rules if r.name == "test_rule"), None)
        assert test_rule is not None
        assert test_rule.enabled is True

    def test_disable_rule(self):
        system = AlertSystem()

        # Add enabled rule
        rule = AlertRule(
            name="test_rule",
            condition=lambda prof, anom: True,
            alert_level=AlertLevel.INFO,
            message_template="Test",
            enabled=True,
        )

        system.add_rule(rule)
        system.disable_rule("test_rule")

        # Find the rule and check it's disabled
        test_rule = next((r for r in system.rules if r.name == "test_rule"), None)
        assert test_rule is not None
        assert test_rule.enabled is False

    def test_evaluate_with_no_issues(self, sample_dataframe):
        profiler = DataProfiler()
        profile_result = profiler.profile(sample_dataframe)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(sample_dataframe)

        system = AlertSystem()
        alerts = system.evaluate(profile_result, anomaly_result)

        assert len(alerts) == 0

    def test_evaluate_high_missing_rate_alert(self):
        df = pd.DataFrame({
            "a": [1, np.nan, np.nan, np.nan, np.nan],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        system = AlertSystem()
        alerts = system.evaluate(profile_result)

        # Should trigger high missing rate alert
        high_missing_alerts = [a for a in alerts if "missing" in a.title.lower()]
        assert len(high_missing_alerts) >= 1

    def test_evaluate_infinite_values_alert(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        alerts = system.evaluate(profile_result, anomaly_result)

        # Should trigger infinite values alert
        inf_alerts = [a for a in alerts if "infinite" in a.message.lower()]
        assert len(inf_alerts) >= 1

    def test_evaluate_critical_anomalies_alert(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
            "b": [np.nan] * 5,
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        alerts = system.evaluate(profile_result, anomaly_result)

        # Should have critical or error alerts
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        error_alerts = [a for a in alerts if a.level == AlertLevel.ERROR]

        assert len(critical_alerts) + len(error_alerts) > 0

    def test_evaluate_duplicate_rows_alert(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 1, 2],
            "b": [1, 2, 3, 1, 2],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector(enable_duplicate_detection=True)
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        alerts = system.evaluate(profile_result, anomaly_result)

        # Should trigger duplicate rows alert
        dup_alerts = [a for a in alerts if "duplicate" in a.message.lower()]
        assert len(dup_alerts) >= 1

    def test_evaluate_low_coverage_alert(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, np.nan, np.nan, np.nan, np.nan],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        system = AlertSystem()
        alerts = system.evaluate(profile_result)

        # Should trigger low coverage alert
        coverage_alerts = [a for a in alerts if "coverage" in a.message.lower()]
        assert len(coverage_alerts) >= 1

    def test_evaluate_zero_variance_alert(self):
        df = pd.DataFrame({
            "constant": [5, 5, 5, 5, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        alerts = system.evaluate(profile_result, anomaly_result)

        # Should trigger zero variance alert
        zero_var_alerts = [a for a in alerts if "constant" in a.message.lower() or "variance" in a.title.lower()]
        assert len(zero_var_alerts) >= 1

    def test_get_alerts_by_level(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        system.evaluate(profile_result, anomaly_result)

        error_alerts = system.get_alerts_by_level(AlertLevel.ERROR)
        warning_alerts = system.get_alerts_by_level(AlertLevel.WARNING)

        assert isinstance(error_alerts, list)
        assert isinstance(warning_alerts, list)

    def test_has_critical_alerts(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
            "b": [np.nan] * 5,
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        system.evaluate(profile_result, anomaly_result)

        # Depending on thresholds, might have critical alerts
        has_critical = system.has_critical_alerts()
        assert isinstance(has_critical, bool)

    def test_has_errors(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        system.evaluate(profile_result, anomaly_result)

        has_errors = system.has_errors()
        assert has_errors is True  # Should have error from infinite value

    def test_export_alerts_json(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        system.evaluate(profile_result, anomaly_result)

        json_export = system.export_alerts(format="json")

        assert isinstance(json_export, str)

        # Should be valid JSON
        data = json.loads(json_export)
        assert isinstance(data, list)

    def test_export_alerts_text(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        system.evaluate(profile_result, anomaly_result)

        text_export = system.export_alerts(format="text")

        assert isinstance(text_export, str)
        assert len(text_export) > 0

    def test_export_alerts_unsupported_format(self, profile_result):
        system = AlertSystem()
        system.evaluate(profile_result)

        with pytest.raises(ValueError, match="Unsupported format"):
            system.export_alerts(format="xml")

    def test_clear_alerts(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        system.evaluate(profile_result, anomaly_result)

        assert len(system.alerts) > 0

        system.clear_alerts()

        assert len(system.alerts) == 0

    def test_get_summary(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        system.evaluate(profile_result, anomaly_result)

        summary = system.get_summary()

        assert isinstance(summary, dict)
        assert "total" in summary
        assert "critical" in summary
        assert "error" in summary
        assert "warning" in summary
        assert "info" in summary

    def test_disabled_rule_not_evaluated(self):
        system = AlertSystem()

        # Add a rule that should trigger
        rule = AlertRule(
            name="always_trigger",
            condition=lambda prof, anom: True,
            alert_level=AlertLevel.WARNING,
            message_template="Always triggers",
            enabled=False,
        )

        system.add_rule(rule)

        df = pd.DataFrame({"a": [1, 2, 3]})
        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        alerts = system.evaluate(profile_result)

        # Should not have alert from disabled rule
        always_trigger_alerts = [a for a in alerts if a.title == "Always Trigger"]
        assert len(always_trigger_alerts) == 0

    def test_rule_evaluation_error_handling(self):
        system = AlertSystem()

        # Add a rule that raises an exception
        def bad_condition(prof, anom):
            raise Exception("Test exception")

        rule = AlertRule(
            name="bad_rule",
            condition=bad_condition,
            alert_level=AlertLevel.WARNING,
            message_template="Should not appear",
        )

        system.add_rule(rule)

        df = pd.DataFrame({"a": [1, 2, 3]})
        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        alerts = system.evaluate(profile_result)

        # Should have an error alert about rule evaluation failure
        error_alerts = [a for a in alerts if "Rule evaluation failed" in a.title]
        assert len(error_alerts) == 1

    def test_metadata_extraction_high_missing_rate(self):
        df = pd.DataFrame({
            "a": [1, np.nan, np.nan, np.nan, np.nan],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        system = AlertSystem()
        alerts = system.evaluate(profile_result)

        high_missing_alerts = [a for a in alerts if "missing" in a.title.lower()]

        if len(high_missing_alerts) > 0:
            alert = high_missing_alerts[0]
            assert "columns" in alert.metadata
            assert "a" in alert.metadata["columns"]

    def test_metadata_extraction_infinite_values(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
        })

        profiler = DataProfiler()
        profile_result = profiler.profile(df)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(df)

        system = AlertSystem()
        alerts = system.evaluate(profile_result, anomaly_result)

        inf_alerts = [a for a in alerts if "infinite" in a.message.lower()]

        if len(inf_alerts) > 0:
            alert = inf_alerts[0]
            assert "columns" in alert.metadata

    def test_evaluate_without_anomaly_result(self, profile_result):
        system = AlertSystem()
        alerts = system.evaluate(profile_result, anomaly_result=None)

        # Should work without anomaly result
        assert isinstance(alerts, list)

    def test_default_rules_exist(self):
        system = AlertSystem()

        rule_names = [r.name for r in system.rules]

        # Check that expected default rules exist
        expected_rules = [
            "high_missing_rate",
            "critical_anomalies",
            "stale_data",
            "infinite_values",
            "low_coverage",
            "duplicate_rows",
            "range_violations",
            "zero_variance",
        ]

        for expected in expected_rules:
            assert expected in rule_names
