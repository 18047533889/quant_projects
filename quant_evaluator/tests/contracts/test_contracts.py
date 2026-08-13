"""
Tests for contract validation and error handling.
"""

import pytest
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import (
    ContractError,
    InvalidContractError,
    TimingContractError,
    MissingInputError,
)


class TestAxisRef:
    """Test AxisRef contract validation."""

    def test_valid_axis(self):
        """Valid axis ref creation."""
        axis = AxisRef(name="time", dtype="datetime64", size=100)
        assert axis.name == "time"
        assert axis.size == 100

    def test_axis_with_values(self):
        """Axis with explicit values."""
        values = np.arange(50)
        axis = AxisRef(name="asset", dtype="int64", size=50, values=values)
        assert axis.size == 50
        assert len(axis.values) == 50

    def test_axis_size_mismatch(self):
        """Axis size must match values length."""
        values = np.arange(50)
        with pytest.raises(ValueError, match="size mismatch"):
            AxisRef(name="asset", dtype="int64", size=100, values=values)


class TestFactorBatch:
    """Test FactorBatch contract validation."""

    def test_valid_single_factor(self):
        """Valid single-factor batch."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)
        values = np.random.randn(10, 50, 1)

        batch = FactorBatch(
            factor_ids=("factor_1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        assert batch.num_factors == 1
        assert batch.num_times == 10
        assert batch.num_assets == 50

    def test_valid_multi_factor(self):
        """Valid multi-factor batch."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=20)
        asset_axis = AxisRef(name="asset", dtype="int64", size=100)
        values = np.random.randn(20, 100, 5)

        batch = FactorBatch(
            factor_ids=("f1", "f2", "f3", "f4", "f5"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        assert batch.num_factors == 5

    def test_empty_factor_ids(self):
        """Empty factor_ids must raise."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)
        values = np.random.randn(10, 50, 1)

        with pytest.raises(ValueError, match="factor_ids cannot be empty"):
            FactorBatch(
                factor_ids=(),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=values,
            )

    def test_missing_axes(self):
        """time_axis and asset_axis are required."""
        with pytest.raises(ValueError, match="time_axis and asset_axis are required"):
            FactorBatch(
                factor_ids=("f1",),
                time_axis=None,
                asset_axis=None,
                values=np.array([1, 2, 3]),
            )

    def test_shape_mismatch(self):
        """Values shape must match axes."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)
        values = np.random.randn(10, 100, 1)  # Wrong asset count

        with pytest.raises(ValueError, match="Shape mismatch"):
            FactorBatch(
                factor_ids=("f1",),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=values,
            )

    def test_validity_shape_mismatch(self):
        """Validity must match values shape."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)
        values = np.random.randn(10, 50, 1)
        validity = np.ones((10, 50), dtype=bool)  # Wrong shape

        with pytest.raises(ValueError, match="Validity shape"):
            FactorBatch(
                factor_ids=("f1",),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=values,
                validity=validity,
            )


class TestLabelBundle:
    """Test LabelBundle contract validation."""

    def test_valid_label_bundle(self):
        """Valid label bundle with explicit timing."""
        values = np.random.randn(100)
        decision_time = tuple(range(100))
        label_start = tuple(range(100))
        label_end = tuple(range(100, 200))

        bundle = LabelBundle(
            target_id="forward_return_1d",
            values=values,
            horizon=1,
            decision_time=decision_time,
            label_start_time=label_start,
            label_end_time=label_end,
        )

        assert bundle.target_id == "forward_return_1d"
        assert bundle.horizon == 1

    def test_missing_target_id(self):
        """target_id is required."""
        with pytest.raises(ValueError, match="target_id cannot be empty"):
            LabelBundle(
                target_id="",
                values=np.array([1, 2, 3]),
                horizon=1,
                decision_time=(0, 1, 2),
                label_start_time=(0, 1, 2),
                label_end_time=(1, 2, 3),
            )

    def test_missing_decision_time(self):
        """decision_time (explicit timing) is required."""
        with pytest.raises(ValueError, match="decision_time.*required"):
            LabelBundle(
                target_id="ret_1d",
                values=np.array([1, 2, 3]),
                horizon=1,
                decision_time=(),  # Empty
                label_start_time=(0, 1, 2),
                label_end_time=(1, 2, 3),
            )

    def test_invalid_horizon(self):
        """Horizon must be positive."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            LabelBundle(
                target_id="ret",
                values=np.array([1, 2, 3]),
                horizon=0,
                decision_time=(0, 1, 2),
                label_start_time=(0, 1, 2),
                label_end_time=(1, 2, 3),
            )

    def test_negative_execution_delay(self):
        """execution_delay cannot be negative."""
        with pytest.raises(ValueError, match="execution_delay cannot be negative"):
            LabelBundle(
                target_id="ret",
                values=np.array([1, 2, 3]),
                horizon=1,
                execution_delay=-1,
                decision_time=(0, 1, 2),
                label_start_time=(0, 1, 2),
                label_end_time=(1, 2, 3),
            )
