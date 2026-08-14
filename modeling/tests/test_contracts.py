"""
Test suite for modeling contracts.
"""
import pytest
from datetime import datetime, timedelta
import numpy as np

from modeling.contracts import (
    FitWindow,
    SplitSpec,
    OutOfFoldSpec,
    PreprocessContract,
    ModelReadyData,
    TransformMode,
)
from modeling.errors import (
    FitWindowError,
    SplitError,
    ContractViolation,
)


class TestFitWindow:
    """Tests for FitWindow contract."""

    def test_valid_fit_window(self):
        """Test valid fit window creation."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)
        window = FitWindow(fit_start=start, fit_end=end)

        assert window.fit_start == start
        assert window.fit_end == end
        assert window.duration_days() == 365

    def test_invalid_fit_window(self):
        """Test that fit_start >= fit_end raises error."""
        start = datetime(2020, 12, 31)
        end = datetime(2020, 1, 1)

        with pytest.raises(FitWindowError):
            FitWindow(fit_start=start, fit_end=end)

    def test_same_start_end(self):
        """Test that same start and end raises error."""
        time = datetime(2020, 1, 1)

        with pytest.raises(FitWindowError):
            FitWindow(fit_start=time, fit_end=time)

    def test_is_valid_for_application_causal(self):
        """Test that fit window must end before application."""
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )

        # Valid: apply after fit window
        assert fit_window.is_valid_for_application(
            datetime(2021, 1, 1),
            datetime(2021, 12, 31),
        )

        # Valid: apply exactly at fit_end
        assert fit_window.is_valid_for_application(
            datetime(2020, 12, 31),
            datetime(2021, 12, 31),
        )

        # Invalid: apply before fit_end
        assert not fit_window.is_valid_for_application(
            datetime(2020, 6, 1),
            datetime(2020, 12, 31),
        )


class TestSplitSpec:
    """Tests for SplitSpec contract."""

    def test_valid_train_only_split(self):
        """Test valid train-only split."""
        split = SplitSpec(
            split_id="train_only",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
        )

        assert split.split_id == "train_only"
        assert split.val_start is None

    def test_valid_train_val_split(self):
        """Test valid train-validation split."""
        split = SplitSpec(
            split_id="fold_1",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 1),
            val_end=datetime(2021, 6, 30),
        )

        assert split.train_end <= split.val_start

    def test_invalid_train_period(self):
        """Test that train_start >= train_end raises error."""
        with pytest.raises(SplitError):
            SplitSpec(
                split_id="bad",
                train_start=datetime(2020, 12, 31),
                train_end=datetime(2020, 1, 1),
            )

    def test_overlapping_train_val(self):
        """Test that overlapping train/val raises error."""
        with pytest.raises(SplitError):
            SplitSpec(
                split_id="overlap",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 12, 31),
                val_start=datetime(2020, 6, 1),  # Overlaps with train
                val_end=datetime(2021, 6, 30),
            )

    def test_incomplete_val_period(self):
        """Test that partial val specification raises error."""
        with pytest.raises(SplitError):
            SplitSpec(
                split_id="incomplete",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 12, 31),
                val_start=datetime(2021, 1, 1),
                # Missing val_end
            )

    def test_get_fit_window_for_train(self):
        """Test getting fit window for train split."""
        split = SplitSpec(
            split_id="fold_1",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 1),
            val_end=datetime(2021, 6, 30),
        )

        fit_window = split.get_fit_window("train")
        assert fit_window.fit_start == split.train_start
        assert fit_window.fit_end == split.train_end

    def test_get_fit_window_for_val_no_leakage(self):
        """Test that validation uses train period for fitting (no leakage)."""
        split = SplitSpec(
            split_id="fold_1",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 1),
            val_end=datetime(2021, 6, 30),
        )

        fit_window = split.get_fit_window("val")
        # Should use train period, NOT val period
        assert fit_window.fit_start == split.train_start
        assert fit_window.fit_end == split.train_end


class TestOutOfFoldSpec:
    """Tests for OutOfFoldSpec contract."""

    def test_valid_oof_spec(self):
        """Test valid OOF specification."""
        folds = [
            SplitSpec(
                split_id="fold_1",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 6, 30),
                val_start=datetime(2020, 7, 1),
                val_end=datetime(2020, 12, 31),
            ),
            SplitSpec(
                split_id="fold_2",
                train_start=datetime(2020, 7, 1),
                train_end=datetime(2020, 12, 31),
                val_start=datetime(2021, 1, 1),
                val_end=datetime(2021, 6, 30),
            ),
        ]

        oof = OutOfFoldSpec(oof_id="rolling_2fold", folds=folds, strategy="rolling")
        assert oof.num_folds() == 2

    def test_empty_folds(self):
        """Test that empty folds raises error."""
        with pytest.raises(SplitError):
            OutOfFoldSpec(oof_id="empty", folds=[])

    def test_non_increasing_fold_times(self):
        """Test that non-increasing fold times raises error."""
        folds = [
            SplitSpec(
                split_id="fold_1",
                train_start=datetime(2020, 7, 1),  # Later
                train_end=datetime(2020, 12, 31),
            ),
            SplitSpec(
                split_id="fold_2",
                train_start=datetime(2020, 1, 1),  # Earlier - invalid
                train_end=datetime(2020, 6, 30),
            ),
        ]

        with pytest.raises(SplitError):
            OutOfFoldSpec(oof_id="bad_order", folds=folds)


class TestPreprocessContract:
    """Tests for PreprocessContract."""

    def test_stateless_contract(self):
        """Test stateless preprocessing contract."""
        contract = PreprocessContract(
            contract_id="stateless_rank",
            transforms=[{"name": "rank", "kind": "cross_sectional", "mode": "stateless"}],
            mode=TransformMode.STATELESS,
        )

        assert contract.mode == TransformMode.STATELESS
        assert contract.fit_window is None

    def test_fitted_contract_requires_fit_window(self):
        """Test that fitted mode requires fit_window."""
        with pytest.raises(ContractViolation):
            PreprocessContract(
                contract_id="fitted_no_window",
                transforms=[{"name": "scaler", "mode": "fitted"}],
                mode=TransformMode.FITTED,
                # Missing fit_window
            )

    def test_stateless_contract_should_not_have_fit_window(self):
        """Test that stateless mode should not have fit_window."""
        with pytest.raises(ContractViolation):
            PreprocessContract(
                contract_id="stateless_with_window",
                transforms=[{"name": "rank", "mode": "stateless"}],
                mode=TransformMode.STATELESS,
                fit_window=FitWindow(
                    fit_start=datetime(2020, 1, 1),
                    fit_end=datetime(2020, 12, 31),
                ),
            )

    def test_valid_fitted_contract(self):
        """Test valid fitted contract."""
        contract = PreprocessContract(
            contract_id="fitted_scaler",
            transforms=[{"name": "scaler", "mode": "fitted"}],
            mode=TransformMode.FITTED,
            fit_window=FitWindow(
                fit_start=datetime(2020, 1, 1),
                fit_end=datetime(2020, 12, 31),
            ),
        )

        assert contract.mode == TransformMode.FITTED
        assert contract.fit_window is not None


class TestModelReadyData:
    """Tests for ModelReadyData output format."""

    def test_valid_model_ready_data(self):
        """Test valid ModelReadyData creation."""
        data = ModelReadyData(
            features=np.random.randn(100, 10),
            feature_names=[f"f_{i}" for i in range(10)],
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2020, 12, 31),
            preprocess_contract_id="test_contract",
        )

        assert data.features.shape == (100, 10)
        assert len(data.feature_names) == 10
        assert data.producer == "modeling"

    def test_invalid_data_period(self):
        """Test that data_start >= data_end raises error."""
        with pytest.raises(ContractViolation):
            ModelReadyData(
                features=np.random.randn(100, 10),
                feature_names=[],
                data_start=datetime(2020, 12, 31),
                data_end=datetime(2020, 1, 1),  # Invalid
                preprocess_contract_id="test",
            )
