"""
Tests for diagnosis module.
"""

import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.api.requests import FactorDiagnosis

from quant_evaluator.diagnosis.factor import diagnose_factor, diagnose_all_factors
from quant_evaluator.diagnosis.warnings import (
    WarningSystem,
    DiagnosticWarning,
    WarningSeverity,
)


class TestDiagnoseFactor:
    def create_test_batch(self, values, factor_ids=None, validity=None):
        """Helper to create test factor batch."""
        if values.ndim == 2:
            values = values[:, :, np.newaxis]

        T, N, F = values.shape

        if factor_ids is None:
            factor_ids = tuple(f"factor_{i}" for i in range(F))

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        return FactorBatch(
            factor_ids=factor_ids,
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
            validity=validity,
        )

    def test_diagnose_normal_factor(self):
        values = np.random.randn(100, 50, 1)
        batch = self.create_test_batch(values, factor_ids=("test_factor",))

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.factor_id == "test_factor"
        assert diagnosis.num_valid_observations == 5000
        assert diagnosis.num_missing == 0
        assert diagnosis.coverage == 1.0
        assert not diagnosis.is_constant
        assert not diagnosis.has_nans
        assert not diagnosis.has_infs
        assert diagnosis.min_value is not None
        assert diagnosis.max_value is not None
        assert diagnosis.mean_value is not None

    def test_diagnose_with_nans(self):
        values = np.random.randn(100, 50, 1)
        values[0:10, :, 0] = np.nan
        batch = self.create_test_batch(values)

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.has_nans
        assert diagnosis.coverage < 1.0
        assert diagnosis.num_missing == 500

    def test_diagnose_with_infs(self):
        values = np.random.randn(100, 50, 1)
        values[0, 0, 0] = np.inf
        values[1, 0, 0] = -np.inf
        batch = self.create_test_batch(values)

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.has_infs
        assert diagnosis.num_missing == 2

    def test_diagnose_constant_factor(self):
        values = np.ones((100, 50, 1)) * 5.0
        batch = self.create_test_batch(values)

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.is_constant
        assert diagnosis.min_value == 5.0
        assert diagnosis.max_value == 5.0
        assert diagnosis.mean_value == 5.0

    def test_diagnose_with_validity_mask(self):
        values = np.random.randn(100, 50, 1)
        validity = np.ones((100, 50, 1), dtype=bool)
        validity[0:20, :, 0] = False

        batch = self.create_test_batch(values, validity=validity)

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.num_valid_observations == 4000
        assert diagnosis.num_missing == 1000
        assert diagnosis.coverage == 0.8

    def test_diagnose_all_zeros(self):
        values = np.zeros((100, 50, 1))
        batch = self.create_test_batch(values)

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.is_constant
        assert diagnosis.min_value == 0.0
        assert diagnosis.max_value == 0.0

    def test_diagnose_all_missing(self):
        values = np.full((100, 50, 1), np.nan)
        batch = self.create_test_batch(values)

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.num_valid_observations == 0
        assert diagnosis.coverage == 0.0
        assert diagnosis.is_constant
        assert diagnosis.min_value is None
        assert diagnosis.max_value is None
        assert diagnosis.mean_value is None

    def test_diagnose_high_coverage(self):
        values = np.random.randn(100, 50, 1)
        values[0:5, 0:10, 0] = np.nan  # 50 missing out of 5000
        batch = self.create_test_batch(values)

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.coverage == 0.99

    def test_diagnose_multiple_factors(self):
        values = np.random.randn(100, 50, 3)
        batch = self.create_test_batch(
            values,
            factor_ids=("factor_a", "factor_b", "factor_c"),
        )

        diagnostics = diagnose_all_factors(batch)

        assert len(diagnostics) == 3
        assert "factor_a" in diagnostics
        assert "factor_b" in diagnostics
        assert "factor_c" in diagnostics

        for diag in diagnostics.values():
            assert isinstance(diag, FactorDiagnosis)

    def test_diagnose_factor_idx_out_of_range(self):
        values = np.random.randn(100, 50, 2)
        batch = self.create_test_batch(values)

        try:
            diagnose_factor(batch, factor_idx=5)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "out of range" in str(e)

    def test_diagnose_warnings_included(self):
        values = np.random.randn(100, 50, 1)
        values[0:60, :, 0] = np.nan  # Low coverage
        batch = self.create_test_batch(values)

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert len(diagnosis.warnings) > 0
        assert any("coverage" in w.lower() for w in diagnosis.warnings)


class TestWarningSystem:
    def create_diagnosis(
        self,
        factor_id="test_factor",
        coverage=1.0,
        is_constant=False,
        has_nans=False,
        has_infs=False,
        min_value=0.0,
        max_value=1.0,
        mean_value=0.5,
        num_valid=5000,
        num_missing=0,
    ):
        """Helper to create FactorDiagnosis."""
        return FactorDiagnosis(
            factor_id=factor_id,
            num_valid_observations=num_valid,
            num_missing=num_missing,
            coverage=coverage,
            is_constant=is_constant,
            has_nans=has_nans,
            has_infs=has_infs,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
        )

    def test_warning_system_creation(self):
        ws = WarningSystem()

        assert ws.missing_rate_high == 0.5
        assert ws.coverage_low == 0.8
        assert ws.outlier_z_threshold == 10.0

    def test_check_coverage_critical(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(coverage=0.3)

        warning = ws.check_coverage(diagnosis)

        assert warning is not None
        assert warning.severity == WarningSeverity.CRITICAL
        assert warning.category == "coverage"

    def test_check_coverage_medium(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(coverage=0.7)

        warning = ws.check_coverage(diagnosis)

        assert warning is not None
        assert warning.severity == WarningSeverity.MEDIUM

    def test_check_coverage_low(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(coverage=0.75)

        warning = ws.check_coverage(diagnosis)

        assert warning is not None
        assert warning.severity == WarningSeverity.MEDIUM

    def test_check_coverage_ok(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(coverage=0.95)

        warning = ws.check_coverage(diagnosis)

        assert warning is None

    def test_check_constant(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(is_constant=True)

        warning = ws.check_constant(diagnosis)

        assert warning is not None
        assert warning.severity == WarningSeverity.CRITICAL
        assert warning.category == "constant"

    def test_check_validity_nans(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(has_nans=True)

        warnings = ws.check_validity(diagnosis)

        assert len(warnings) == 1
        assert warnings[0].severity == WarningSeverity.HIGH
        assert warnings[0].category == "validity"

    def test_check_validity_infs(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(has_infs=True)

        warnings = ws.check_validity(diagnosis)

        assert len(warnings) == 1
        assert warnings[0].severity == WarningSeverity.HIGH

    def test_check_validity_both(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(has_nans=True, has_infs=True)

        warnings = ws.check_validity(diagnosis)

        assert len(warnings) == 2

    def test_check_range_narrow(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(min_value=1.0, max_value=1.0)

        warning = ws.check_range(diagnosis)

        assert warning is not None
        assert warning.severity == WarningSeverity.HIGH
        assert warning.category == "range"

    def test_check_range_ok(self):
        ws = WarningSystem()
        diagnosis = self.create_diagnosis(min_value=-3.0, max_value=3.0)

        warning = ws.check_range(diagnosis)

        assert warning is None

    def test_check_outliers_with_extremes(self):
        ws = WarningSystem(outlier_z_threshold=3.0)

        values = np.random.randn(100, 50, 1)
        values[0, 0, 0] = 100.0  # Extreme outlier

        time_axis = AxisRef(name="time", dtype="datetime64", size=100)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)
        batch = FactorBatch(
            factor_ids=("test_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        warning = ws.check_outliers(batch, factor_idx=0)

        assert warning is not None
        assert warning.severity == WarningSeverity.MEDIUM
        assert warning.category == "outliers"
        assert "extreme outliers" in warning.message

    def test_check_outliers_none(self):
        ws = WarningSystem()

        values = np.random.randn(100, 50, 1) * 0.1

        time_axis = AxisRef(name="time", dtype="datetime64", size=100)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)
        batch = FactorBatch(
            factor_ids=("test_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        warning = ws.check_outliers(batch, factor_idx=0)

        assert warning is None

    def test_diagnose_factor_full(self):
        ws = WarningSystem()

        diagnosis = self.create_diagnosis(
            coverage=0.4,
            has_nans=True,
            is_constant=False,
        )

        warnings = ws.diagnose_factor(diagnosis)

        assert len(warnings) >= 2
        assert any(w.category == "coverage" for w in warnings)
        assert any(w.category == "validity" for w in warnings)

    def test_diagnose_factor_with_batch(self):
        ws = WarningSystem(outlier_z_threshold=3.0)

        values = np.random.randn(100, 50, 1)
        values[0, 0, 0] = 50.0

        time_axis = AxisRef(name="time", dtype="datetime64", size=100)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)
        batch = FactorBatch(
            factor_ids=("test_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        diagnosis = self.create_diagnosis(coverage=0.95)

        warnings = ws.diagnose_factor(diagnosis, factor_batch=batch, factor_idx=0)

        assert any(w.category == "outliers" for w in warnings)

    def test_diagnose_all_factors(self):
        ws = WarningSystem()

        diagnostics = {
            "factor_a": self.create_diagnosis("factor_a", coverage=0.3),
            "factor_b": self.create_diagnosis("factor_b", is_constant=True),
            "factor_c": self.create_diagnosis("factor_c", coverage=0.99),
        }

        warnings_by_factor = ws.diagnose_all_factors(diagnostics)

        assert len(warnings_by_factor) == 3
        assert len(warnings_by_factor["factor_a"]) > 0
        assert len(warnings_by_factor["factor_b"]) > 0
        assert len(warnings_by_factor["factor_c"]) == 0

    def test_format_warnings_empty(self):
        ws = WarningSystem()

        text = ws.format_warnings([])

        assert "No warnings" in text

    def test_format_warnings_basic(self):
        ws = WarningSystem()

        warnings = [
            DiagnosticWarning(
                factor_id="test",
                severity=WarningSeverity.HIGH,
                category="validity",
                message="Contains NaN values",
            ),
            DiagnosticWarning(
                factor_id="test",
                severity=WarningSeverity.CRITICAL,
                category="coverage",
                message="Coverage only 30%",
            ),
        ]

        text = ws.format_warnings(warnings)

        assert "[CRITICAL]" in text
        assert "[HIGH]" in text
        assert "validity" in text
        assert "coverage" in text

    def test_format_warnings_with_details(self):
        ws = WarningSystem()

        warnings = [
            DiagnosticWarning(
                factor_id="test",
                severity=WarningSeverity.MEDIUM,
                category="outliers",
                message="Found 5 outliers",
                details={"num_outliers": 5, "max_z_score": 12.3},
            ),
        ]

        text = ws.format_warnings(warnings, include_details=True)

        assert "num_outliers: 5" in text
        assert "max_z_score: 12.3" in text

    def test_warning_sorting_by_severity(self):
        ws = WarningSystem()

        diagnosis = self.create_diagnosis(
            coverage=0.3,
            has_nans=True,
            is_constant=True,
        )

        warnings = ws.diagnose_factor(diagnosis)

        # Should be sorted: CRITICAL first, then HIGH
        severities = [w.severity for w in warnings]
        assert severities[0] in (WarningSeverity.CRITICAL,)

        # Verify sorted order
        severity_order = {
            WarningSeverity.CRITICAL: 0,
            WarningSeverity.HIGH: 1,
            WarningSeverity.MEDIUM: 2,
            WarningSeverity.LOW: 3,
        }
        severity_values = [severity_order[s] for s in severities]
        assert severity_values == sorted(severity_values)

    def test_custom_thresholds(self):
        ws = WarningSystem(
            missing_rate_high=0.7,
            coverage_low=0.95,
            outlier_z_threshold=5.0,
        )

        assert ws.missing_rate_high == 0.7
        assert ws.coverage_low == 0.95
        assert ws.outlier_z_threshold == 5.0

        diagnosis = self.create_diagnosis(coverage=0.5)
        warning = ws.check_coverage(diagnosis)

        assert warning is None or warning.severity != WarningSeverity.CRITICAL
