"""
Tests for diagnosis utilities.
"""

import pytest
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.diagnosis.factor import diagnose_factor, diagnose_all_factors


class TestFactorDiagnosis:
    """Test factor diagnosis generation."""

    def test_diagnose_healthy_factor(self):
        """Diagnose a healthy factor with full coverage."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)

        values = np.random.randn(10, 50, 1)

        batch = FactorBatch(
            factor_ids=("healthy_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.factor_id == "healthy_factor"
        assert diagnosis.num_valid_observations == 500
        assert diagnosis.num_missing == 0
        assert diagnosis.coverage == 1.0
        assert not diagnosis.is_constant
        assert not diagnosis.has_nans
        assert not diagnosis.has_infs
        assert diagnosis.min_value is not None
        assert diagnosis.max_value is not None
        assert diagnosis.mean_value is not None
        assert len(diagnosis.warnings) == 0

    def test_diagnose_factor_with_nans(self):
        """Diagnose factor with NaN values."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=5)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        values = np.random.randn(5, 10, 1)
        values[0, :3, 0] = np.nan
        values[2, 5:, 0] = np.nan

        batch = FactorBatch(
            factor_ids=("nan_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.has_nans
        assert diagnosis.num_missing == 8  # 3 + 5
        assert diagnosis.coverage < 1.0
        assert "Contains NaN values" in diagnosis.warnings

    def test_diagnose_factor_with_infs(self):
        """Diagnose factor with Inf values."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=3)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        values = np.random.randn(3, 10, 1)
        values[1, 0, 0] = np.inf
        values[2, 5, 0] = -np.inf

        batch = FactorBatch(
            factor_ids=("inf_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.has_infs
        assert diagnosis.num_missing == 2
        assert "Contains Inf values" in diagnosis.warnings

    def test_diagnose_constant_factor(self):
        """Diagnose constant factor."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=5)
        asset_axis = AxisRef(name="asset", dtype="int64", size=20)

        values = np.full((5, 20, 1), 42.0)

        batch = FactorBatch(
            factor_ids=("constant_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.is_constant
        assert diagnosis.min_value == 42.0
        assert diagnosis.max_value == 42.0
        assert diagnosis.mean_value == 42.0
        assert "Factor is constant" in diagnosis.warnings

    def test_diagnose_low_coverage_warning(self):
        """Low coverage triggers warning."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        values = np.random.randn(10, 10, 1)
        values[:, :6, 0] = np.nan  # 60% missing

        batch = FactorBatch(
            factor_ids=("sparse_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.coverage < 0.5
        assert any("Low coverage" in w for w in diagnosis.warnings)

    def test_diagnose_all_factors(self):
        """Diagnose multiple factors."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=5)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        values = np.random.randn(5, 10, 3)
        values[:, :, 1] = 5.0  # Second factor is constant

        batch = FactorBatch(
            factor_ids=("f1", "f2", "f3"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        diagnostics = diagnose_all_factors(batch)

        assert len(diagnostics) == 3
        assert "f1" in diagnostics
        assert "f2" in diagnostics
        assert "f3" in diagnostics
        assert diagnostics["f2"].is_constant
        assert not diagnostics["f1"].is_constant
        assert not diagnostics["f3"].is_constant

    def test_diagnose_out_of_range_idx(self):
        """Out of range factor index raises error."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=5)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.random.randn(5, 10, 1),
        )

        with pytest.raises(ValueError, match="factor_idx.*out of range"):
            diagnose_factor(batch, factor_idx=5)

    def test_diagnose_with_validity_mask(self):
        """Validity mask affects diagnosis."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=3)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        values = np.ones((3, 10, 1))
        validity = np.ones((3, 10, 1), dtype=bool)
        validity[0, :5, 0] = False

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
            validity=validity,
        )

        diagnosis = diagnose_factor(batch, factor_idx=0)

        assert diagnosis.num_valid_observations == 25  # 30 - 5
        assert diagnosis.num_missing == 5
        assert np.isclose(diagnosis.coverage, 25 / 30, atol=1e-10)
