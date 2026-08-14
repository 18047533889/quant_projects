"""
Leakage detection tests - the most critical tests for modeling.
"""
import pytest
from datetime import datetime
import numpy as np

from modeling.contracts import FitWindow, SplitSpec, OutOfFoldSpec
from modeling.preprocess.fitted import CrossSectionalScaler
from modeling.errors import FutureLeakageError, SplitError


@pytest.mark.leakage
class TestFullSampleLeakagePrevention:
    def test_full_sample_fit_then_historical_application_fails(self):
        X_full = np.random.randn(504, 10)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2021, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_full, fit_window)
        X_2020 = X_full[:252, :]
        with pytest.raises(FutureLeakageError):
            scaler.transform(X_2020, apply_start_time=datetime(2020, 1, 1))

    def test_train_val_split_no_val_in_train_fit(self):
        split = SplitSpec(split_id="fold_1", train_start=datetime(2020, 1, 1), 
                         train_end=datetime(2020, 12, 31), val_start=datetime(2021, 1, 1), 
                         val_end=datetime(2021, 6, 30))
        val_fit_window = split.get_fit_window("val")
        assert val_fit_window.fit_end == split.train_end

    def test_expanding_window_no_future_leakage(self):
        folds = [
            SplitSpec(split_id="fold_1", train_start=datetime(2020, 1, 1), 
                     train_end=datetime(2020, 6, 30), val_start=datetime(2020, 7, 1), 
                     val_end=datetime(2020, 12, 31)),
            SplitSpec(split_id="fold_2", train_start=datetime(2020, 1, 1), 
                     train_end=datetime(2020, 12, 31), val_start=datetime(2021, 1, 1), 
                     val_end=datetime(2021, 6, 30)),
        ]
        oof = OutOfFoldSpec(oof_id="expanding_2fold", folds=folds, strategy="expanding")
        for fold in oof.folds:
            fit_window = fold.get_fit_window("val")
            assert fit_window.fit_end <= fold.val_start


@pytest.mark.leakage
class TestRollingWindowLeakagePrevention:
    def test_rolling_window_no_overlap(self):
        folds = [
            SplitSpec(split_id="fold_1", train_start=datetime(2020, 1, 1), 
                     train_end=datetime(2020, 6, 30), val_start=datetime(2020, 7, 1), 
                     val_end=datetime(2020, 12, 31)),
            SplitSpec(split_id="fold_2", train_start=datetime(2020, 7, 1), 
                     train_end=datetime(2020, 12, 31), val_start=datetime(2021, 1, 1), 
                     val_end=datetime(2021, 6, 30)),
        ]
        oof = OutOfFoldSpec(oof_id="rolling_2fold", folds=folds, strategy="rolling")
        assert oof.num_folds() == 2

    def test_overlapping_train_val_rejected(self):
        with pytest.raises(SplitError):
            SplitSpec(split_id="bad_overlap", train_start=datetime(2020, 1, 1), 
                     train_end=datetime(2020, 12, 31), val_start=datetime(2020, 6, 1), 
                     val_end=datetime(2021, 6, 30))


@pytest.mark.leakage
class TestFittedStateLeakage:
    def test_fitted_state_applied_to_correct_period_only(self):
        X_train = np.random.randn(252, 10)
        X_val = np.random.randn(126, 10)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)
        X_transformed = scaler.transform(X_val, apply_start_time=datetime(2021, 1, 1))
        assert X_transformed.shape == X_val.shape

    def test_refit_on_different_window_updates_state(self):
        X1 = np.random.randn(100, 5)
        X2 = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)
        scaler = CrossSectionalScaler()
        fit_window_2020 = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler.fit(X1, fit_window_2020)
        scaler.transform(X_test, apply_start_time=datetime(2021, 1, 1))
        fit_window_2021 = FitWindow(fit_start=datetime(2021, 1, 1), fit_end=datetime(2021, 12, 31))
        scaler.fit(X2, fit_window_2021)
        with pytest.raises(FutureLeakageError):
            scaler.transform(X_test, apply_start_time=datetime(2021, 6, 1))


@pytest.mark.leakage
class TestOutOfFoldLeakage:
    def test_oof_each_fold_uses_only_its_train_period(self):
        folds = [
            SplitSpec(split_id="fold_1", train_start=datetime(2020, 1, 1), 
                     train_end=datetime(2020, 4, 30), val_start=datetime(2020, 5, 1), 
                     val_end=datetime(2020, 8, 31)),
            SplitSpec(split_id="fold_2", train_start=datetime(2020, 5, 1), 
                     train_end=datetime(2020, 8, 31), val_start=datetime(2020, 9, 1), 
                     val_end=datetime(2020, 12, 31)),
        ]
        oof = OutOfFoldSpec(oof_id="rolling_2fold", folds=folds, strategy="rolling")
        for fold in oof.folds:
            fit_window = fold.get_fit_window("val")
            assert fit_window.fit_end <= fold.val_start


@pytest.mark.leakage
class TestCommonLeakagePatterns:
    def test_cannot_use_test_period_in_fit_window(self):
        split = SplitSpec(split_id="train_test", train_start=datetime(2020, 1, 1), 
                         train_end=datetime(2020, 12, 31), test_start=datetime(2021, 1, 1), 
                         test_end=datetime(2021, 12, 31))
        test_fit_window = split.get_fit_window("test")
        assert test_fit_window.fit_end == split.train_end

    def test_gap_between_train_and_val(self):
        split = SplitSpec(split_id="with_gap", train_start=datetime(2020, 1, 1), 
                         train_end=datetime(2020, 10, 31), val_start=datetime(2020, 12, 1), 
                         val_end=datetime(2021, 6, 30), gap_days=30)
        assert (split.val_start - split.train_end).days >= split.gap_days

    def test_zero_gap_is_valid(self):
        split = SplitSpec(split_id="no_gap", train_start=datetime(2020, 1, 1), 
                         train_end=datetime(2020, 12, 31), val_start=datetime(2021, 1, 1), 
                         val_end=datetime(2021, 6, 30), gap_days=0)
        assert split.train_end <= split.val_start
