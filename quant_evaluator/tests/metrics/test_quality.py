"""
Tests for coverage diagnostics with golden values.
"""

import pytest
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.quality import (
    compute_coverage,
    compute_per_time_coverage,
)
from quant_evaluator.contracts.errors import InvalidContractError


class TestCoverage:
    """Test coverage diagnostics."""

    def test_full_coverage(self):
        """All values valid."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=20)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.random.randn(10, 20, 1),
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.random.randn(10, 20),
            horizon=1,
            decision_time=tuple(range(10)),
            label_start_time=tuple(range(10)),
            label_end_time=tuple(range(1, 11)),
        )

        coverage, num_valid, num_total = compute_coverage(batch, bundle, min_assets=10)

        assert coverage == 1.0
        assert num_valid == 200  # 10 * 20
        assert num_total == 200

    def test_partial_coverage_with_nans(self):
        """Some NaN values."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=5)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        values = np.random.randn(5, 10, 1)
        values[0, :3, 0] = np.nan  # 3 NaN in first period
        values[2, 5:, 0] = np.nan  # 5 NaN in third period

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.random.randn(5, 10),
            horizon=1,
            decision_time=tuple(range(5)),
            label_start_time=tuple(range(5)),
            label_end_time=tuple(range(1, 6)),
        )

        coverage, num_valid, num_total = compute_coverage(batch, bundle, min_assets=5)

        assert num_total == 50
        assert num_valid == 50 - 3 - 5  # 42
        assert np.isclose(coverage, 42 / 50, atol=1e-10)

    def test_label_nans_excluded(self):
        """Label NaNs also excluded from coverage."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=3)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.ones((3, 10, 1)),
        )

        labels = np.ones((3, 10))
        labels[0, :5] = np.nan
        labels[1, :3] = np.nan

        bundle = LabelBundle(
            target_id="ret",
            values=labels,
            horizon=1,
            decision_time=(0, 1, 2),
            label_start_time=(0, 1, 2),
            label_end_time=(1, 2, 3),
        )

        coverage, num_valid, num_total = compute_coverage(batch, bundle, min_assets=5)

        # Valid pairs: period 0 has 5 NaN, period 1 has 3 NaN, period 2 has 0 NaN
        expected_valid = 30 - 5 - 3  # 22
        assert num_valid == expected_valid
        assert np.isclose(coverage, expected_valid / 30, atol=1e-10)

    def test_validity_mask_applied(self):
        """Validity masks filter coverage."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=2)
        asset_axis = AxisRef(name="asset", dtype="int64", size=5)

        values = np.ones((2, 5, 1))
        validity = np.ones((2, 5, 1), dtype=bool)
        validity[0, :2, 0] = False
        validity[1, 3:, 0] = False

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
            validity=validity,
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.ones((2, 5)),
            horizon=1,
            decision_time=(0, 1),
            label_start_time=(0, 1),
            label_end_time=(1, 2),
        )

        coverage, num_valid, num_total = compute_coverage(batch, bundle)

        # Invalid: period 0 has 2 invalid, period 1 has 2 invalid
        expected_valid = 10 - 2 - 2  # 6
        assert num_valid == expected_valid
        assert np.isclose(coverage, 0.6, atol=1e-10)

    def test_shape_mismatch_raises(self):
        """Time dimension mismatch raises error."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=20)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.random.randn(10, 20, 1),
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.random.randn(5, 20),  # Wrong time length
            horizon=1,
            decision_time=tuple(range(5)),
            label_start_time=tuple(range(5)),
            label_end_time=tuple(range(1, 6)),
        )

        with pytest.raises(InvalidContractError, match="does not match"):
            compute_coverage(batch, bundle)

    def test_1d_label_broadcast(self):
        """1D labels are broadcast to 2D."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=3)
        asset_axis = AxisRef(name="asset", dtype="int64", size=5)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.ones((3, 5, 1)),
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.array([1.0, 2.0, 3.0]),  # 1D
            horizon=1,
            decision_time=(0, 1, 2),
            label_start_time=(0, 1, 2),
            label_end_time=(1, 2, 3),
        )

        coverage, num_valid, num_total = compute_coverage(batch, bundle)

        assert num_valid == 15  # All valid
        assert coverage == 1.0


class TestPerTimeCoverage:
    """Test per-time coverage computation."""

    def test_per_time_coverage_uniform(self):
        """Uniform coverage across time."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=5)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.ones((5, 10, 1)),
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.ones((5, 10)),
            horizon=1,
            decision_time=tuple(range(5)),
            label_start_time=tuple(range(5)),
            label_end_time=tuple(range(1, 6)),
        )

        coverage_per_time = compute_per_time_coverage(batch, bundle)

        assert coverage_per_time.shape == (5, 1)
        assert np.all(coverage_per_time == 1.0)

    def test_per_time_coverage_varying(self):
        """Coverage varies by time."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=3)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        values = np.ones((3, 10, 1))
        values[0, :5, 0] = np.nan  # 50% coverage in t=0
        values[1, :2, 0] = np.nan  # 80% coverage in t=1
        # t=2 has 100% coverage

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.ones((3, 10)),
            horizon=1,
            decision_time=(0, 1, 2),
            label_start_time=(0, 1, 2),
            label_end_time=(1, 2, 3),
        )

        coverage_per_time = compute_per_time_coverage(batch, bundle)

        assert coverage_per_time.shape == (3, 1)
        assert np.isclose(coverage_per_time[0, 0], 0.5, atol=1e-10)
        assert np.isclose(coverage_per_time[1, 0], 0.8, atol=1e-10)
        assert np.isclose(coverage_per_time[2, 0], 1.0, atol=1e-10)
