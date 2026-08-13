"""
Fold-local tests to ensure train/test split safety.

These tests verify that fitted transforms respect fold boundaries.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from factor_preprocess.contracts import FittedState
from factor_preprocess.errors import TimingContractError, InvalidContractError


@pytest.mark.fold_local
class TestFoldBoundaries:
    """Test suite for fold-local fitting."""

    def test_fitted_state_requires_fit_window(self):
        """Test that FittedState enforces fit window."""
        # Valid state
        state = FittedState(
            state_id="test",
            transform_name="rolling_mean",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
        )
        assert state.fit_start_time < state.fit_end_time

        # Invalid: fit_end before fit_start
        with pytest.raises(TimingContractError, match="fit_start_time must be before fit_end_time"):
            FittedState(
                state_id="test",
                transform_name="rolling_mean",
                transform_version="1.0.0",
                fit_start_time=datetime(2020, 12, 31),
                fit_end_time=datetime(2020, 1, 1),
            )

    def test_fit_window_metadata_preserved(self):
        """Test that fit window is preserved in state."""
        fit_start = datetime(2020, 1, 1)
        fit_end = datetime(2020, 12, 31)

        state = FittedState(
            state_id="test",
            transform_name="rolling_mean",
            transform_version="1.0.0",
            fit_start_time=fit_start,
            fit_end_time=fit_end,
        )

        # Fit window should be exactly preserved
        assert state.fit_start_time == fit_start
        assert state.fit_end_time == fit_end

    def test_feature_contract_validation(self):
        """Test that feature IDs and order are validated."""
        # Valid: matching feature_ids and feature_order
        state = FittedState(
            state_id="test",
            transform_name="transform",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
            feature_ids=["f1", "f2"],
            feature_order=["f1", "f2"],
        )
        assert len(state.feature_ids) == 2

        # Invalid: feature_ids without feature_order
        with pytest.raises(InvalidContractError, match="feature_order required"):
            FittedState(
                state_id="test",
                transform_name="transform",
                transform_version="1.0.0",
                fit_start_time=datetime(2020, 1, 1),
                fit_end_time=datetime(2020, 12, 31),
                feature_ids=["f1", "f2"],
                feature_order=[],
            )

    def test_feature_compatibility_check(self):
        """Test that feature compatibility is checked."""
        state = FittedState(
            state_id="test",
            transform_name="transform",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
            feature_ids=["f1", "f2"],
            feature_order=["f1", "f2"],
        )

        # Compatible
        assert state.is_compatible_with(["f1", "f2"])
        assert state.is_compatible_with(["f2", "f1"])  # Order doesn't matter for set

        # Incompatible
        assert not state.is_compatible_with(["f1"])
        assert not state.is_compatible_with(["f1", "f2", "f3"])
        assert not state.is_compatible_with(["f3", "f4"])

    def test_state_immutability(self):
        """Test that FittedState is immutable after creation."""
        state = FittedState(
            state_id="test",
            transform_name="transform",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
        )

        # Cannot modify after creation
        with pytest.raises(Exception):  # FrozenInstanceError
            state.fit_start_time = datetime(2021, 1, 1)

        with pytest.raises(Exception):
            state.learned_params = {"new": "params"}

    def test_learned_params_metadata(self):
        """Test that learned parameters are stored with metadata."""
        learned_params = {
            "mean": 5.0,
            "std": 2.0,
        }

        state = FittedState(
            state_id="test",
            transform_name="zscore",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
            learned_params=learned_params,
            learned_params_hash="abc123",
        )

        assert state.learned_params["mean"] == 5.0
        assert state.learned_params["std"] == 2.0
        assert state.learned_params_hash == "abc123"

    def test_fit_universe_reference(self):
        """Test that fit universe can be referenced."""
        state = FittedState(
            state_id="test",
            transform_name="transform",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
            fit_universe_ref="universe_snapshot_2020",
        )

        assert state.fit_universe_ref == "universe_snapshot_2020"

    def test_no_full_sample_fit_contract(self):
        """
        Test that fit window must be explicit (no default to full sample).

        This is a contract test: FittedState requires explicit fit_start and fit_end.
        There is no way to create a FittedState without these parameters.
        """
        # This should require explicit parameters (no defaults)
        with pytest.raises(TypeError):
            # Missing required fit_start_time and fit_end_time
            FittedState(
                state_id="test",
                transform_name="transform",
                transform_version="1.0.0",
            )

    def test_fold_local_fit_scenario(self):
        """
        Test a realistic fold-local fitting scenario.

        Train on 2020 data, test on 2021 data.
        """
        # Fit state from training period
        train_state = FittedState(
            state_id="train_state_001",
            transform_name="zscore",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
            feature_ids=["factor_1", "factor_2"],
            feature_order=["factor_1", "factor_2"],
            learned_params={"mean": 5.0, "std": 2.0},
        )

        # Test period is after training period
        test_start = datetime(2021, 1, 1)
        test_end = datetime(2021, 12, 31)

        # Verify no overlap
        assert test_start > train_state.fit_end_time

        # Test data should be compatible with trained state
        assert train_state.is_compatible_with(["factor_1", "factor_2"])

    def test_multiple_folds_no_leakage(self):
        """Test that multiple folds maintain separate fit windows."""
        # Fold 1: Train on 2020, test on 2021 Q1
        fold1_state = FittedState(
            state_id="fold1",
            transform_name="zscore",
            transform_version="1.0.0",
            fit_start_time=datetime(2020, 1, 1),
            fit_end_time=datetime(2020, 12, 31),
            feature_ids=["f1"],
            feature_order=["f1"],
        )

        # Fold 2: Train on 2021 Q1, test on 2021 Q2
        fold2_state = FittedState(
            state_id="fold2",
            transform_name="zscore",
            transform_version="1.0.0",
            fit_start_time=datetime(2021, 1, 1),
            fit_end_time=datetime(2021, 3, 31),
            feature_ids=["f1"],
            feature_order=["f1"],
        )

        # Each fold has distinct fit window
        assert fold1_state.fit_end_time < fold2_state.fit_start_time

        # States are independent (different state_ids)
        assert fold1_state.state_id != fold2_state.state_id
