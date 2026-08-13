"""
Warning system for detecting factor quality issues.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.api.requests import FactorDiagnosis


class WarningSeverity(Enum):
    """Severity levels for diagnostic warnings."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DiagnosticWarning:
    """Structured warning with severity and context."""
    factor_id: str
    severity: WarningSeverity
    category: str
    message: str
    details: Optional[dict] = None


class WarningSystem:
    """
    Detect and emit structured warnings for factor quality issues.

    Thresholds:
        - High missing rate: coverage < 50%
        - Low coverage: coverage < 80%
        - Constant values: std == 0
        - Extreme outliers: |z-score| > 10
    """

    def __init__(
        self,
        missing_rate_high: float = 0.5,
        missing_rate_medium: float = 0.2,
        coverage_low: float = 0.8,
        outlier_z_threshold: float = 10.0,
        constant_tolerance: float = 1e-12,
    ):
        self.missing_rate_high = missing_rate_high
        self.missing_rate_medium = missing_rate_medium
        self.coverage_low = coverage_low
        self.outlier_z_threshold = outlier_z_threshold
        self.constant_tolerance = constant_tolerance

    def check_coverage(self, diagnosis: FactorDiagnosis) -> Optional[DiagnosticWarning]:
        """Check for low coverage / high missing rate."""
        if diagnosis.coverage < (1.0 - self.missing_rate_high):
            return DiagnosticWarning(
                factor_id=diagnosis.factor_id,
                severity=WarningSeverity.CRITICAL,
                category="coverage",
                message=f"Critical: coverage only {diagnosis.coverage:.2%}",
                details={
                    "coverage": diagnosis.coverage,
                    "num_missing": diagnosis.num_missing,
                    "num_valid": diagnosis.num_valid_observations,
                },
            )
        elif diagnosis.coverage < (1.0 - self.missing_rate_medium):
            return DiagnosticWarning(
                factor_id=diagnosis.factor_id,
                severity=WarningSeverity.MEDIUM,
                category="coverage",
                message=f"Low coverage: {diagnosis.coverage:.2%}",
                details={"coverage": diagnosis.coverage},
            )
        elif diagnosis.coverage < self.coverage_low:
            return DiagnosticWarning(
                factor_id=diagnosis.factor_id,
                severity=WarningSeverity.LOW,
                category="coverage",
                message=f"Coverage below target: {diagnosis.coverage:.2%}",
                details={"coverage": diagnosis.coverage},
            )
        return None

    def check_constant(self, diagnosis: FactorDiagnosis) -> Optional[DiagnosticWarning]:
        """Check if factor has constant values."""
        if diagnosis.is_constant:
            return DiagnosticWarning(
                factor_id=diagnosis.factor_id,
                severity=WarningSeverity.CRITICAL,
                category="constant",
                message="Factor is constant (zero variance)",
                details={
                    "min_value": diagnosis.min_value,
                    "max_value": diagnosis.max_value,
                    "mean_value": diagnosis.mean_value,
                },
            )
        return None

    def check_validity(self, diagnosis: FactorDiagnosis) -> List[DiagnosticWarning]:
        """Check for NaN and Inf values."""
        warnings = []

        if diagnosis.has_nans:
            warnings.append(
                DiagnosticWarning(
                    factor_id=diagnosis.factor_id,
                    severity=WarningSeverity.HIGH,
                    category="validity",
                    message="Contains NaN values",
                    details={"has_nans": True},
                )
            )

        if diagnosis.has_infs:
            warnings.append(
                DiagnosticWarning(
                    factor_id=diagnosis.factor_id,
                    severity=WarningSeverity.HIGH,
                    category="validity",
                    message="Contains Inf values",
                    details={"has_infs": True},
                )
            )

        return warnings

    def check_outliers(
        self,
        factor_batch: FactorBatch,
        factor_idx: int,
    ) -> Optional[DiagnosticWarning]:
        """Check for extreme outliers using z-score."""
        values = factor_batch.values[:, :, factor_idx].flatten()

        if factor_batch.validity is not None:
            validity_mask = factor_batch.validity[:, :, factor_idx].flatten()
            finite_mask = np.isfinite(values)
            valid_mask = finite_mask & validity_mask
        else:
            valid_mask = np.isfinite(values)

        valid_values = values[valid_mask]

        if len(valid_values) == 0:
            return None

        std = np.std(valid_values)
        if std < self.constant_tolerance:
            return None

        mean = np.mean(valid_values)
        z_scores = np.abs((valid_values - mean) / std)

        num_extreme = int(np.sum(z_scores > self.outlier_z_threshold))

        if num_extreme > 0:
            max_z = float(np.max(z_scores))
            factor_id = factor_batch.factor_ids[factor_idx]

            return DiagnosticWarning(
                factor_id=factor_id,
                severity=WarningSeverity.MEDIUM,
                category="outliers",
                message=f"Found {num_extreme} extreme outliers (|z| > {self.outlier_z_threshold})",
                details={
                    "num_outliers": num_extreme,
                    "max_z_score": max_z,
                    "threshold": self.outlier_z_threshold,
                },
            )

        return None

    def check_range(self, diagnosis: FactorDiagnosis) -> Optional[DiagnosticWarning]:
        """Check for suspiciously narrow value range."""
        if diagnosis.min_value is None or diagnosis.max_value is None:
            return None

        value_range = diagnosis.max_value - diagnosis.min_value

        if value_range < self.constant_tolerance:
            return DiagnosticWarning(
                factor_id=diagnosis.factor_id,
                severity=WarningSeverity.HIGH,
                category="range",
                message=f"Suspiciously narrow range: {value_range:.2e}",
                details={
                    "min_value": diagnosis.min_value,
                    "max_value": diagnosis.max_value,
                    "range": value_range,
                },
            )

        return None

    def diagnose_factor(
        self,
        diagnosis: FactorDiagnosis,
        factor_batch: Optional[FactorBatch] = None,
        factor_idx: Optional[int] = None,
    ) -> List[DiagnosticWarning]:
        """
        Run all checks on a factor and return structured warnings.

        Args:
            diagnosis: Pre-computed factor diagnosis
            factor_batch: Optional batch for outlier detection
            factor_idx: Factor index in batch (required if factor_batch provided)

        Returns:
            List of diagnostic warnings sorted by severity
        """
        warnings = []

        # Coverage check
        cov_warning = self.check_coverage(diagnosis)
        if cov_warning:
            warnings.append(cov_warning)

        # Constant check
        const_warning = self.check_constant(diagnosis)
        if const_warning:
            warnings.append(const_warning)

        # Validity checks
        warnings.extend(self.check_validity(diagnosis))

        # Range check
        range_warning = self.check_range(diagnosis)
        if range_warning:
            warnings.append(range_warning)

        # Outlier check (requires raw batch)
        if factor_batch is not None and factor_idx is not None:
            outlier_warning = self.check_outliers(factor_batch, factor_idx)
            if outlier_warning:
                warnings.append(outlier_warning)

        # Sort by severity (critical first)
        severity_order = {
            WarningSeverity.CRITICAL: 0,
            WarningSeverity.HIGH: 1,
            WarningSeverity.MEDIUM: 2,
            WarningSeverity.LOW: 3,
            WarningSeverity.INFO: 4,
        }
        warnings.sort(key=lambda w: severity_order[w.severity])

        return warnings

    def diagnose_all_factors(
        self,
        diagnostics: dict,
        factor_batch: Optional[FactorBatch] = None,
    ) -> dict:
        """
        Diagnose all factors in a batch.

        Args:
            diagnostics: Dict mapping factor_id -> FactorDiagnosis
            factor_batch: Optional batch for outlier detection

        Returns:
            Dict mapping factor_id -> List[DiagnosticWarning]
        """
        warnings_by_factor = {}

        for factor_id, diagnosis in diagnostics.items():
            factor_idx = None
            if factor_batch is not None:
                try:
                    factor_idx = factor_batch.factor_ids.index(factor_id)
                except ValueError:
                    pass

            warnings = self.diagnose_factor(
                diagnosis,
                factor_batch=factor_batch,
                factor_idx=factor_idx,
            )

            warnings_by_factor[factor_id] = warnings

        return warnings_by_factor

    def format_warnings(
        self,
        warnings: List[DiagnosticWarning],
        include_details: bool = False,
    ) -> str:
        """Format warnings as human-readable text."""
        if not warnings:
            return "No warnings detected."

        lines = []
        for warning in warnings:
            severity_marker = {
                WarningSeverity.CRITICAL: "[CRITICAL]",
                WarningSeverity.HIGH: "[HIGH]",
                WarningSeverity.MEDIUM: "[MEDIUM]",
                WarningSeverity.LOW: "[LOW]",
                WarningSeverity.INFO: "[INFO]",
            }[warning.severity]

            line = f"{severity_marker} {warning.category}: {warning.message}"
            lines.append(line)

            if include_details and warning.details:
                for key, value in warning.details.items():
                    lines.append(f"  - {key}: {value}")

        return "\n".join(lines)
