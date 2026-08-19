"""
Tests for fitted transforms - especially temporal contract validation.
"""
import pytest
from datetime import datetime
import numpy as np

from modeling_adapters.preprocess.fitted import FittedTransform, CrossSectionalScaler, create_fitted_scaler
from modeling_adapters.contracts import FitWindow
from modeling_adapters.errors import FitWindowError, FutureLeakageError, InsufficientDataError


class TestCrossSectionalScaler:
    def test_fit_and_transform(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 2 + 5
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler(method="zscore")
        X_transformed = scaler.fit_transform(X_train, fit_window)
        assert np.allclose(np.nanmean(X_transformed, axis=0), 0, atol=1e-10)

    def test_fit_records_fit_window(self):
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)
        assert scaler.fit_window == fit_window

    def test_transform_before_fit_raises_error(self):
        scaler = CrossSectionalScaler()
        X = np.random.randn(50, 5)
        with pytest.raises(FitWindowError):
            scaler.transform(X, apply_start_time=datetime(2021, 1, 1))

    def test_no_future_leakage_validation(self):
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)
        with pytest.raises(FutureLeakageError):
            scaler.transform(X_test, apply_start_time=datetime(2020, 6, 1))

    def test_valid_application_after_fit_end(self):
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)
        X_transformed = scaler.transform(X_test, apply_start_time=datetime(2021, 1, 1))
        assert X_transformed.shape == X_test.shape

    def test_application_at_fit_end_is_valid(self):
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)
        X_transformed = scaler.transform(X_test, apply_start_time=datetime(2020, 12, 31))
        assert X_transformed.shape == X_test.shape

    def test_fit_transform(self):
        X = np.random.randn(100, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        X_transformed = scaler.fit_transform(X, fit_window)
        assert X_transformed.shape == X.shape

    def test_empty_data_raises_error(self):
        X_empty = np.array([]).reshape(0, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        with pytest.raises(InsufficientDataError):
            scaler.fit(X_empty, fit_window)

    def test_get_state(self):
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)
        state = scaler.get_state()
        assert state["method"] == "zscore"

    def test_transform_width_mismatch_raises(self):
        """A (T,1) input against a 5-feature fit previously broadcast into
        five fabricated columns instead of failing."""
        from modeling_adapters.errors import ContractViolation

        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)

        with pytest.raises(ContractViolation, match="feature column"):
            scaler.transform(np.random.randn(50, 1), apply_start_time=datetime(2021, 1, 1))
        with pytest.raises(ContractViolation, match="feature column"):
            scaler.transform(np.random.randn(50, 3), apply_start_time=datetime(2021, 1, 1))

    def test_get_state_surfaces_degenerate_features(self):
        """std_ is overwritten to 1.0 for degenerate features; get_state
        must expose the degeneracy explicitly."""
        X_train = np.random.randn(100, 3)
        X_train[:, 2] = 7.0  # constant column
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)

        state = scaler.get_state()
        assert state["degenerate_features"] is not None
        assert state["degenerate_features"].tolist() == [False, False, True]

    def test_failed_refit_keeps_consistent_state(self):
        """A hostile refit must not commit the new fit_window while keeping
        old parameters under old _fitted authority."""
        X_train = np.random.randn(100, 2)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)

        class Hostile(np.ndarray):
            pass

        with pytest.raises(Exception):
            scaler.fit(np.array([["a", "b"]], dtype=object), fit_window)

        # fit_window must still be the old one (no partial commit)
        assert scaler.fit_window == fit_window


class TestFittedTransformFactory:
    def test_create_fitted_scaler_zscore(self):
        scaler = create_fitted_scaler(method="zscore")
        assert isinstance(scaler, CrossSectionalScaler)

    def test_create_fitted_scaler_unknown_method(self):
        with pytest.raises(ValueError):
            create_fitted_scaler(method="unknown")


class TestTemporalContractEnforcement:
    def test_multiple_applications_with_increasing_times(self):
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2020, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_train, fit_window)
        X_q1 = np.random.randn(50, 5)
        scaler.transform(X_q1, apply_start_time=datetime(2021, 1, 1))
        X_q2 = np.random.randn(50, 5)
        scaler.transform(X_q2, apply_start_time=datetime(2021, 4, 1))

    def test_fit_on_full_sample_then_apply_to_past_fails(self):
        X_full = np.random.randn(1000, 5)
        fit_window = FitWindow(fit_start=datetime(2020, 1, 1), fit_end=datetime(2021, 12, 31))
        scaler = CrossSectionalScaler()
        scaler.fit(X_full, fit_window)
        X_2020 = np.random.randn(100, 5)
        with pytest.raises(FutureLeakageError):
            scaler.transform(X_2020, apply_start_time=datetime(2020, 6, 1))
