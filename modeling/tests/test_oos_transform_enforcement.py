"""
FM2-P0-012~015: OOS transform enforcement tests for standalone modeling package.

Tests that FittedTransform.transform() REQUIRES apply_start_time (no silent bypass).
"""
import pytest
from datetime import datetime
import numpy as np

from modeling.contracts import FitWindow
from modeling.preprocess.fitted import CrossSectionalScaler
from modeling.errors import FitWindowError, FutureLeakageError


class TestOOSTransformEnforcement:
    """Test that transform() requires apply_start_time (no silent None bypass)."""

    def test_transform_requires_apply_start_time(self):
        """transform() must require apply_start_time (cannot be None)."""
        scaler = CrossSectionalScaler()
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )
        scaler.fit(X_train, fit_window)

        X_val = np.random.randn(50, 5)

        # FM2-P0-012~015: transform() with None should raise TypeError
        with pytest.raises(TypeError, match="requires apply_start_time"):
            scaler.transform(X_val, apply_start_time=None)

    def test_fit_transform_for_in_sample_use(self):
        """fit_transform() is the correct API for in-sample (training) use."""
        scaler = CrossSectionalScaler()
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )

        # Correct: fit_transform for in-sample
        X_transformed = scaler.fit_transform(X_train, fit_window)
        assert X_transformed.shape == X_train.shape

    def test_transform_with_valid_apply_start_time(self):
        """transform() succeeds when apply_start_time is after fit_window.fit_end."""
        scaler = CrossSectionalScaler()
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )
        scaler.fit(X_train, fit_window)

        X_val = np.random.randn(50, 5)
        # Valid: apply_start_time is after fit_window.fit_end
        X_transformed = scaler.transform(X_val, apply_start_time=datetime(2021, 1, 1))
        assert X_transformed.shape == X_val.shape

    def test_transform_rejects_apply_start_time_before_fit_end(self):
        """transform() rejects apply_start_time before fit_window.fit_end (future leakage)."""
        scaler = CrossSectionalScaler()
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )
        scaler.fit(X_train, fit_window)

        X_val = np.random.randn(50, 5)
        # Invalid: apply_start_time is before fit_window.fit_end
        with pytest.raises(FutureLeakageError):
            scaler.transform(X_val, apply_start_time=datetime(2020, 6, 1))

    def test_transform_allows_apply_start_time_equal_to_fit_end(self):
        """transform() allows apply_start_time exactly at fit_window.fit_end."""
        scaler = CrossSectionalScaler()
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )
        scaler.fit(X_train, fit_window)

        X_val = np.random.randn(50, 5)
        # Valid: apply_start_time exactly at fit_window.fit_end (boundary case)
        X_transformed = scaler.transform(X_val, apply_start_time=datetime(2020, 12, 31))
        assert X_transformed.shape == X_val.shape

    def test_transform_before_fit_raises_error(self):
        """transform() before fit() raises FitWindowError."""
        scaler = CrossSectionalScaler()
        X = np.random.randn(50, 5)

        with pytest.raises(FitWindowError, match="must be fitted"):
            scaler.transform(X, apply_start_time=datetime(2021, 1, 1))

    def test_fit_transform_internally_uses_safe_apply_start_time(self):
        """fit_transform() internally uses fit_window.fit_end as apply_start_time."""
        scaler = CrossSectionalScaler()
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )

        # fit_transform should succeed (uses fit_window.fit_end internally)
        X_transformed = scaler.fit_transform(X_train, fit_window)
        assert X_transformed.shape == X_train.shape
        assert scaler._fitted is True
        assert scaler.fit_window == fit_window

    def test_multiple_transforms_with_different_apply_times(self):
        """Same fitted scaler can transform multiple OOS periods."""
        scaler = CrossSectionalScaler()
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )
        scaler.fit(X_train, fit_window)

        # First OOS period
        X_val = np.random.randn(50, 5)
        X_val_transformed = scaler.transform(X_val, apply_start_time=datetime(2021, 1, 1))
        assert X_val_transformed.shape == X_val.shape

        # Second OOS period (later)
        X_test = np.random.randn(30, 5)
        X_test_transformed = scaler.transform(X_test, apply_start_time=datetime(2021, 7, 1))
        assert X_test_transformed.shape == X_test.shape

    def test_refit_updates_fit_window_validation(self):
        """Refitting updates fit_window, affecting subsequent transform() validation."""
        scaler = CrossSectionalScaler()

        # First fit
        X_train1 = np.random.randn(100, 5)
        fit_window1 = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )
        scaler.fit(X_train1, fit_window1)

        # Can transform data from 2021
        X_2021 = np.random.randn(50, 5)
        scaler.transform(X_2021, apply_start_time=datetime(2021, 1, 1))

        # Refit on later data
        X_train2 = np.random.randn(100, 5)
        fit_window2 = FitWindow(
            fit_start=datetime(2021, 1, 1),
            fit_end=datetime(2021, 12, 31),
        )
        scaler.fit(X_train2, fit_window2)

        # Now cannot transform data from early 2021 (before new fit_end)
        with pytest.raises(FutureLeakageError):
            scaler.transform(X_2021, apply_start_time=datetime(2021, 6, 1))

    def test_error_message_includes_diagnostic_info(self):
        """Error message for None apply_start_time is clear and actionable."""
        scaler = CrossSectionalScaler()
        X_train = np.random.randn(100, 5)
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )
        scaler.fit(X_train, fit_window)

        X_val = np.random.randn(50, 5)

        with pytest.raises(TypeError) as exc_info:
            scaler.transform(X_val, apply_start_time=None)

        error_msg = str(exc_info.value)
        assert "requires apply_start_time" in error_msg
        assert "fit_transform" in error_msg  # Suggests correct alternative

    def test_zscore_transform_correctness(self):
        """Verify zscore transform produces correct output."""
        scaler = CrossSectionalScaler(method="zscore")
        X_train = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )

        # Fit and get parameters
        scaler.fit(X_train, fit_window)
        expected_mean = np.array([4.0, 5.0, 6.0])
        expected_std = np.array([3.0, 3.0, 3.0])
        np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(scaler.std_, expected_std, rtol=1e-10)

        # Transform OOS data
        X_val = np.array([[10.0, 11.0, 12.0]])
        X_transformed = scaler.transform(X_val, apply_start_time=datetime(2021, 1, 1))

        # Expected: (X - mean) / std = ([10, 11, 12] - [4, 5, 6]) / [3, 3, 3] = [2, 2, 2]
        expected = np.array([[2.0, 2.0, 2.0]])
        np.testing.assert_allclose(X_transformed, expected, rtol=1e-10)
