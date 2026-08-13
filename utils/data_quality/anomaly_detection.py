"""
Anomaly detection module for identifying unusual values and patterns in financial data.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    OUTLIER = "outlier"
    MISSING_VALUE = "missing_value"
    INFINITE_VALUE = "infinite_value"
    DUPLICATE = "duplicate"
    RANGE_VIOLATION = "range_violation"
    SUDDEN_CHANGE = "sudden_change"
    STALE_DATA = "stale_data"
    ZERO_VARIANCE = "zero_variance"
    INCONSISTENT_TYPE = "inconsistent_type"


@dataclass
class Anomaly:
    """Single anomaly detection result."""

    anomaly_type: AnomalyType
    column: str
    severity: str  # "low", "medium", "high"
    description: str
    affected_rows: Optional[List[int]] = None
    value: Optional[Any] = None
    expected_range: Optional[tuple] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly to dictionary."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "column": self.column,
            "severity": self.severity,
            "description": self.description,
            "affected_count": len(self.affected_rows) if self.affected_rows else None,
            "value": str(self.value) if self.value is not None else None,
            "expected_range": self.expected_range,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AnomalyResult:
    """Complete anomaly detection result."""

    timestamp: datetime
    total_anomalies: int
    anomalies: List[Anomaly]
    severity_counts: Dict[str, int]
    type_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_anomalies": self.total_anomalies,
            "severity_counts": self.severity_counts,
            "type_counts": self.type_counts,
            "anomalies": [a.to_dict() for a in self.anomalies],
        }

    def get_by_severity(self, severity: str) -> List[Anomaly]:
        """Filter anomalies by severity."""
        return [a for a in self.anomalies if a.severity == severity]

    def get_by_type(self, anomaly_type: AnomalyType) -> List[Anomaly]:
        """Filter anomalies by type."""
        return [a for a in self.anomalies if a.anomaly_type == anomaly_type]

    def get_by_column(self, column: str) -> List[Anomaly]:
        """Filter anomalies by column."""
        return [a for a in self.anomalies if a.column == column]


class AnomalyDetector:
    """
    Comprehensive anomaly detector for financial time series and panel data.

    Detects outliers, missing values, range violations, sudden changes, and other anomalies.
    """

    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        missing_threshold: float = 10.0,
        sudden_change_threshold: float = 5.0,
        enable_duplicate_detection: bool = True,
        enable_stale_detection: bool = True,
        stale_threshold_days: int = 7,
    ):
        """
        Initialize anomaly detector.

        Args:
            outlier_method: Method for outlier detection ("iqr", "zscore", or "mad")
            outlier_threshold: Threshold for outlier detection (IQR multiplier or z-score)
            missing_threshold: Percentage of missing values to flag as anomaly
            sudden_change_threshold: Factor for detecting sudden changes (std multiples)
            enable_duplicate_detection: Whether to detect duplicate rows
            enable_stale_detection: Whether to detect stale data in time series
            stale_threshold_days: Days without update to consider data stale
        """
        if outlier_method not in ["iqr", "zscore", "mad"]:
            raise ValueError("outlier_method must be 'iqr', 'zscore', or 'mad'")

        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
        self.sudden_change_threshold = sudden_change_threshold
        self.enable_duplicate_detection = enable_duplicate_detection
        self.enable_stale_detection = enable_stale_detection
        self.stale_threshold_days = stale_threshold_days

    def detect(
        self,
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        expected_ranges: Optional[Dict[str, tuple]] = None,
    ) -> AnomalyResult:
        """
        Detect anomalies in DataFrame.

        Args:
            df: DataFrame to analyze
            date_column: Optional date column for time series anomalies
            expected_ranges: Optional dict of {column: (min, max)} for range checks

        Returns:
            AnomalyResult with detected anomalies
        """
        if df.empty:
            raise ValueError("Cannot detect anomalies in empty DataFrame")

        anomalies: List[Anomaly] = []

        # Detect missing values
        anomalies.extend(self._detect_missing_values(df))

        # Detect infinite values
        anomalies.extend(self._detect_infinite_values(df))

        # Detect outliers in numeric columns
        anomalies.extend(self._detect_outliers(df))

        # Detect range violations
        if expected_ranges:
            anomalies.extend(self._detect_range_violations(df, expected_ranges))

        # Detect sudden changes in time series
        if date_column and date_column in df.columns:
            anomalies.extend(self._detect_sudden_changes(df, date_column))

            # Detect stale data
            if self.enable_stale_detection:
                anomalies.extend(self._detect_stale_data(df, date_column))

        # Detect duplicates
        if self.enable_duplicate_detection:
            anomalies.extend(self._detect_duplicates(df))

        # Detect zero variance columns
        anomalies.extend(self._detect_zero_variance(df))

        # Compute summary statistics
        severity_counts = {
            "low": sum(1 for a in anomalies if a.severity == "low"),
            "medium": sum(1 for a in anomalies if a.severity == "medium"),
            "high": sum(1 for a in anomalies if a.severity == "high"),
        }

        type_counts = {}
        for a in anomalies:
            type_name = a.anomaly_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return AnomalyResult(
            timestamp=datetime.now(),
            total_anomalies=len(anomalies),
            anomalies=anomalies,
            severity_counts=severity_counts,
            type_counts=type_counts,
        )

    def _detect_missing_values(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect excessive missing values."""
        anomalies = []

        for col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100

            if null_pct > self.missing_threshold:
                severity = "high" if null_pct > 50 else "medium" if null_pct > 25 else "low"

                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.MISSING_VALUE,
                        column=col,
                        severity=severity,
                        description=f"{null_pct:.1f}% missing values ({null_count}/{len(df)})",
                        affected_rows=df[df[col].isna()].index.tolist()[:100],  # Limit to first 100
                    )
                )

        return anomalies

    def _detect_infinite_values(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect infinite values in numeric columns."""
        anomalies = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            inf_mask = np.isinf(df[col])
            inf_count = inf_mask.sum()

            if inf_count > 0:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.INFINITE_VALUE,
                        column=col,
                        severity="high",
                        description=f"{inf_count} infinite values detected",
                        affected_rows=df[inf_mask].index.tolist()[:100],
                    )
                )

        return anomalies

    def _detect_outliers(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect statistical outliers in numeric columns."""
        anomalies = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) < 10:  # Need enough data for outlier detection
                continue

            if self.outlier_method == "iqr":
                outliers = self._detect_outliers_iqr(series)
            elif self.outlier_method == "zscore":
                outliers = self._detect_outliers_zscore(series)
            else:  # mad
                outliers = self._detect_outliers_mad(series)

            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(series)) * 100

                # Only flag if outliers are not too common (likely not outliers then)
                if outlier_pct < 10:
                    severity = "high" if outlier_pct > 5 else "medium" if outlier_pct > 1 else "low"

                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.OUTLIER,
                            column=col,
                            severity=severity,
                            description=f"{len(outliers)} outliers detected ({outlier_pct:.1f}%)",
                            affected_rows=outliers.tolist()[:100],
                        )
                    )

        return anomalies

    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Index:
        """Detect outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr

        return series[(series < lower_bound) | (series > upper_bound)].index

    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Index:
        """Detect outliers using z-score method."""
        mean = series.mean()
        std = series.std()

        if std == 0:
            return pd.Index([])

        z_scores = np.abs((series - mean) / std)
        return series[z_scores > self.outlier_threshold].index

    def _detect_outliers_mad(self, series: pd.Series) -> pd.Index:
        """Detect outliers using median absolute deviation (MAD) method."""
        median = series.median()
        mad = np.median(np.abs(series - median))

        if mad == 0:
            return pd.Index([])

        modified_z_scores = 0.6745 * (series - median) / mad
        return series[np.abs(modified_z_scores) > self.outlier_threshold].index

    def _detect_range_violations(
        self,
        df: pd.DataFrame,
        expected_ranges: Dict[str, tuple],
    ) -> List[Anomaly]:
        """Detect values outside expected ranges."""
        anomalies = []

        for col, (min_val, max_val) in expected_ranges.items():
            if col not in df.columns:
                continue

            violations = df[(df[col] < min_val) | (df[col] > max_val)]

            if len(violations) > 0:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.RANGE_VIOLATION,
                        column=col,
                        severity="high",
                        description=f"{len(violations)} values outside range [{min_val}, {max_val}]",
                        affected_rows=violations.index.tolist()[:100],
                        expected_range=(min_val, max_val),
                    )
                )

        return anomalies

    def _detect_sudden_changes(self, df: pd.DataFrame, date_column: str) -> List[Anomaly]:
        """Detect sudden changes in time series data."""
        anomalies = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Sort by date
        df_sorted = df.sort_values(date_column)

        for col in numeric_cols:
            if col == date_column:
                continue

            # Compute differences
            series = df_sorted[col].dropna()

            if len(series) < 5:
                continue

            diffs = series.diff().dropna()

            if len(diffs) < 2:
                continue

            std_diff = diffs.std()

            if std_diff == 0:
                continue

            # Find sudden changes
            sudden_changes = diffs[np.abs(diffs) > self.sudden_change_threshold * std_diff]

            if len(sudden_changes) > 0:
                severity = "medium" if len(sudden_changes) < 5 else "low"

                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.SUDDEN_CHANGE,
                        column=col,
                        severity=severity,
                        description=f"{len(sudden_changes)} sudden changes detected (>{self.sudden_change_threshold} std)",
                        affected_rows=sudden_changes.index.tolist()[:100],
                    )
                )

        return anomalies

    def _detect_stale_data(self, df: pd.DataFrame, date_column: str) -> List[Anomaly]:
        """Detect stale data (no recent updates)."""
        anomalies = []

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                dates = pd.to_datetime(df[date_column])
            except Exception:
                return anomalies
        else:
            dates = df[date_column]

        max_date = dates.max()
        now = pd.Timestamp.now()

        days_since_update = (now - max_date).days

        if days_since_update > self.stale_threshold_days:
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.STALE_DATA,
                    column=date_column,
                    severity="high" if days_since_update > 30 else "medium",
                    description=f"Data is {days_since_update} days old (last: {max_date.date()})",
                )
            )

        return anomalies

    def _detect_duplicates(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect duplicate rows."""
        anomalies = []

        duplicates = df[df.duplicated(keep=False)]

        if len(duplicates) > 0:
            unique_duplicates = df[df.duplicated(keep="first")]
            severity = "high" if len(unique_duplicates) > len(df) * 0.1 else "medium"

            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.DUPLICATE,
                    column="__all__",
                    severity=severity,
                    description=f"{len(unique_duplicates)} duplicate rows detected",
                    affected_rows=unique_duplicates.index.tolist()[:100],
                )
            )

        return anomalies

    def _detect_zero_variance(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect columns with zero variance (constant values)."""
        anomalies = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) < 2:
                continue

            if series.std() == 0:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.ZERO_VARIANCE,
                        column=col,
                        severity="low",
                        description=f"Column has constant value: {series.iloc[0]}",
                        value=series.iloc[0],
                    )
                )

        return anomalies
