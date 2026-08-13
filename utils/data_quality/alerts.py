"""
Alert system module for quality monitoring and notifications.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import json

from .profiler import ProfileResult
from .anomaly_detection import AnomalyResult, AnomalyType


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Single quality alert."""

    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AlertRule:
    """Rule for generating alerts based on data quality metrics."""

    name: str
    condition: Callable[[ProfileResult, Optional[AnomalyResult]], bool]
    alert_level: AlertLevel
    message_template: str
    enabled: bool = True


class AlertSystem:
    """
    Comprehensive alert system for data quality monitoring.

    Evaluates quality metrics against configurable rules and generates alerts.
    """

    def __init__(self):
        """Initialize alert system with default rules."""
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up default alert rules."""

        # High missing value rate
        self.add_rule(
            AlertRule(
                name="high_missing_rate",
                condition=lambda prof, anom: any(
                    p.null_percentage > 50 for p in prof.column_profiles.values()
                ),
                alert_level=AlertLevel.ERROR,
                message_template="Columns with >50% missing values detected",
            )
        )

        # Critical anomalies
        self.add_rule(
            AlertRule(
                name="critical_anomalies",
                condition=lambda prof, anom: (
                    anom is not None and anom.severity_counts.get("high", 0) > 0
                ),
                alert_level=AlertLevel.CRITICAL,
                message_template="Critical anomalies detected: {high_count} high severity issues",
            )
        )

        # Stale data
        self.add_rule(
            AlertRule(
                name="stale_data",
                condition=lambda prof, anom: (
                    anom is not None
                    and any(a.anomaly_type == AnomalyType.STALE_DATA for a in anom.anomalies)
                ),
                alert_level=AlertLevel.WARNING,
                message_template="Data appears to be stale",
            )
        )

        # Infinite values
        self.add_rule(
            AlertRule(
                name="infinite_values",
                condition=lambda prof, anom: any(
                    p.inf_count > 0 for p in prof.column_profiles.values()
                ),
                alert_level=AlertLevel.ERROR,
                message_template="Infinite values detected in numeric columns",
            )
        )

        # Low data coverage
        self.add_rule(
            AlertRule(
                name="low_coverage",
                condition=lambda prof, anom: (
                    prof.coverage_matrix is not None
                    and (prof.coverage_matrix["coverage_pct"] < 70).any()
                ),
                alert_level=AlertLevel.WARNING,
                message_template="Columns with <70% coverage detected",
            )
        )

        # Duplicate rows
        self.add_rule(
            AlertRule(
                name="duplicate_rows",
                condition=lambda prof, anom: (
                    anom is not None
                    and any(a.anomaly_type == AnomalyType.DUPLICATE for a in anom.anomalies)
                ),
                alert_level=AlertLevel.WARNING,
                message_template="Duplicate rows detected",
            )
        )

        # Range violations
        self.add_rule(
            AlertRule(
                name="range_violations",
                condition=lambda prof, anom: (
                    anom is not None
                    and any(
                        a.anomaly_type == AnomalyType.RANGE_VIOLATION for a in anom.anomalies
                    )
                ),
                alert_level=AlertLevel.ERROR,
                message_template="Values outside expected ranges detected",
            )
        )

        # Zero variance columns
        self.add_rule(
            AlertRule(
                name="zero_variance",
                condition=lambda prof, anom: (
                    anom is not None
                    and any(
                        a.anomaly_type == AnomalyType.ZERO_VARIANCE for a in anom.anomalies
                    )
                ),
                alert_level=AlertLevel.INFO,
                message_template="Constant value columns detected",
            )
        )

    def add_rule(self, rule: AlertRule):
        """
        Add a custom alert rule.

        Args:
            rule: AlertRule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule by name.

        Args:
            rule_name: Name of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        original_len = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        return len(self.rules) < original_len

    def enable_rule(self, rule_name: str):
        """Enable a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True

    def disable_rule(self, rule_name: str):
        """Disable a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False

    def evaluate(
        self,
        profile_result: ProfileResult,
        anomaly_result: Optional[AnomalyResult] = None,
    ) -> List[Alert]:
        """
        Evaluate all rules and generate alerts.

        Args:
            profile_result: Data profiling results
            anomaly_result: Optional anomaly detection results

        Returns:
            List of generated alerts
        """
        self.alerts = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                if rule.condition(profile_result, anomaly_result):
                    alert = self._create_alert(rule, profile_result, anomaly_result)
                    self.alerts.append(alert)
            except Exception as e:
                # If rule evaluation fails, create an error alert
                self.alerts.append(
                    Alert(
                        level=AlertLevel.ERROR,
                        title=f"Rule evaluation failed: {rule.name}",
                        message=f"Error evaluating rule: {str(e)}",
                        source="alert_system",
                        metadata={"rule_name": rule.name},
                    )
                )

        return self.alerts

    def _create_alert(
        self,
        rule: AlertRule,
        profile_result: ProfileResult,
        anomaly_result: Optional[AnomalyResult],
    ) -> Alert:
        """Create an alert from a triggered rule."""
        metadata = self._extract_metadata(rule, profile_result, anomaly_result)

        # Format message template
        message = rule.message_template.format(**metadata)

        return Alert(
            level=rule.alert_level,
            title=rule.name.replace("_", " ").title(),
            message=message,
            source="alert_rule",
            metadata=metadata,
        )

    def _extract_metadata(
        self,
        rule: AlertRule,
        profile_result: ProfileResult,
        anomaly_result: Optional[AnomalyResult],
    ) -> Dict[str, Any]:
        """Extract relevant metadata for alert."""
        metadata: Dict[str, Any] = {"rule_name": rule.name}

        if rule.name == "high_missing_rate":
            high_missing = [
                (name, prof.null_percentage)
                for name, prof in profile_result.column_profiles.items()
                if prof.null_percentage > 50
            ]
            metadata["columns"] = [col for col, _ in high_missing]
            metadata["missing_percentages"] = {col: pct for col, pct in high_missing}

        elif rule.name == "critical_anomalies" and anomaly_result:
            metadata["high_count"] = anomaly_result.severity_counts.get("high", 0)
            metadata["total_anomalies"] = anomaly_result.total_anomalies

        elif rule.name == "infinite_values":
            inf_columns = [
                (name, prof.inf_count)
                for name, prof in profile_result.column_profiles.items()
                if prof.inf_count > 0
            ]
            metadata["columns"] = [col for col, _ in inf_columns]
            metadata["inf_counts"] = {col: cnt for col, cnt in inf_columns}

        elif rule.name == "low_coverage" and profile_result.coverage_matrix is not None:
            low_cov = profile_result.coverage_matrix[
                profile_result.coverage_matrix["coverage_pct"] < 70
            ]
            metadata["columns"] = low_cov["column"].tolist()
            metadata["coverage"] = dict(
                zip(low_cov["column"], low_cov["coverage_pct"])
            )

        elif rule.name == "duplicate_rows" and anomaly_result:
            dup_anomalies = [
                a for a in anomaly_result.anomalies
                if a.anomaly_type == AnomalyType.DUPLICATE
            ]
            if dup_anomalies:
                metadata["duplicate_count"] = len(
                    dup_anomalies[0].affected_rows
                ) if dup_anomalies[0].affected_rows else 0

        elif rule.name == "range_violations" and anomaly_result:
            range_anomalies = [
                a for a in anomaly_result.anomalies
                if a.anomaly_type == AnomalyType.RANGE_VIOLATION
            ]
            metadata["columns"] = [a.column for a in range_anomalies]
            metadata["violation_count"] = sum(
                len(a.affected_rows) if a.affected_rows else 0
                for a in range_anomalies
            )

        return metadata

    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """
        Get alerts filtered by level.

        Args:
            level: AlertLevel to filter by

        Returns:
            List of alerts with specified level
        """
        return [a for a in self.alerts if a.level == level]

    def has_critical_alerts(self) -> bool:
        """Check if any critical alerts were generated."""
        return any(a.level == AlertLevel.CRITICAL for a in self.alerts)

    def has_errors(self) -> bool:
        """Check if any error-level alerts were generated."""
        return any(a.level == AlertLevel.ERROR for a in self.alerts)

    def export_alerts(self, format: str = "json") -> str:
        """
        Export alerts in specified format.

        Args:
            format: Export format ("json" or "text")

        Returns:
            Formatted alert string
        """
        if format == "json":
            return json.dumps(
                [a.to_dict() for a in self.alerts],
                indent=2,
                default=str,
            )
        elif format == "text":
            lines = []
            for alert in self.alerts:
                lines.append(
                    f"[{alert.level.value.upper()}] {alert.title}: {alert.message}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_alerts(self):
        """Clear all stored alerts."""
        self.alerts = []

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current alerts.

        Returns:
            Dictionary with alert counts by level
        """
        return {
            "total": len(self.alerts),
            "critical": len(self.get_alerts_by_level(AlertLevel.CRITICAL)),
            "error": len(self.get_alerts_by_level(AlertLevel.ERROR)),
            "warning": len(self.get_alerts_by_level(AlertLevel.WARNING)),
            "info": len(self.get_alerts_by_level(AlertLevel.INFO)),
        }
