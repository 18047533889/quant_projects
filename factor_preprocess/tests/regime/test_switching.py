"""
Tests for regime-based transform switching.

Verifies fitting, application, causality, and fold-safety.
"""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.regime.switching import (
    fit_regime_switching,
    regime_switching_transform,
    RegimeSwitchingState,
)
from factor_preprocess.errors import (
    InsufficientObservations,
    MissingFittedStateError,
    StaleFittedStateError,
)


class TestFitRegimeSwitching:
    """Test regime-specific transform fitting."""

    def test_fit_zscore_transform(self):
        """Fit z-score parameters per regime."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Regime 0: mean=0, std=1
        # Regime 1: mean=5, std=2
        values = np.concatenate([
            np.random.randn(50) * 1.0 + 0.0,
            np.random.randn(50) * 2.0 + 5.0,
        ])

        df = pd.DataFrame({
            "date": dates,
            "value": values,
        })

        regime_labels = pd.Series([0] * 50 + [1] * 50, index=df.index)

        state = fit_regime_switching(
            df,
            regime_labels,
            transform_type="zscore",
            time_col="date",
            value_col="value",
        )

        # Check regime 0 parameters
        assert state.regime_params[0]["mean"] == pytest.approx(0.0, abs=0.5)
        assert state.regime_params[0]["std"] == pytest.approx(1.0, abs=0.3)

        # Check regime 1 parameters
        assert state.regime_params[1]["mean"] == pytest.approx(5.0, abs=0.5)
        assert state.regime_params[1]["std"] == pytest.approx(2.0, abs=0.3)

    def test_fit_winsor_transform(self):
        """Fit winsorization boundaries per regime."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Two regimes with different distributions
        values = np.concatenate([
            np.random.randn(50) * 1.0,
            np.random.randn(50) * 5.0,
        ])

        df = pd.DataFrame({
            "date": dates,
            "value": values,
        })

        regime_labels = pd.Series([0] * 50 + [1] * 50, index=df.index)

        state = fit_regime_switching(
            df,
            regime_labels,
            transform_type="winsor",
            time_col="date",
            value_col="value",
        )

        # Each regime should have different boundaries
        lower0 = state.regime_params[0]["lower_bound"]
        upper0 = state.regime_params[0]["upper_bound"]
        lower1 = state.regime_params[1]["lower_bound"]
        upper1 = state.regime_params[1]["upper_bound"]

        # Regime 1 should have wider bounds (larger std)
        assert (upper1 - lower1) > (upper0 - lower0)

    def test_fit_scale_transform(self):
        """Fit scaling std per regime."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        values = np.concatenate([
            np.random.randn(50) * 1.0,
            np.random.randn(50) * 3.0,
        ])

        df = pd.DataFrame({
            "date": dates,
            "value": values,
        })

        regime_labels = pd.Series([0] * 50 + [1] * 50, index=df.index)

        state = fit_regime_switching(
            df,
            regime_labels,
            transform_type="scale",
            time_col="date",
            value_col="value",
        )

        # Regime 1 should have larger std
        assert state.regime_params[1]["std"] > state.regime_params[0]["std"]

    def test_fit_rank_transform(self):
        """Rank transform should not store parameters."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(100),
        })

        regime_labels = pd.Series([0] * 100, index=df.index)

        state = fit_regime_switching(
            df,
            regime_labels,
            transform_type="rank",
            time_col="date",
        )

        # Rank should have empty params
        assert state.regime_params[0] == {}

    def test_fit_none_transform(self):
        """None transform should not store parameters."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(100),
        })

        regime_labels = pd.Series([0] * 100, index=df.index)

        state = fit_regime_switching(
            df,
            regime_labels,
            transform_type="none",
            time_col="date",
        )

        assert state.regime_params[0] == {}

    def test_insufficient_observations(self):
        """Should raise if regime has too few observations."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(20),
        })

        regime_labels = pd.Series([0] * 20, index=df.index)

        with pytest.raises(InsufficientObservations):
            fit_regime_switching(
                df,
                regime_labels,
                transform_type="zscore",
                time_col="date",
                min_obs_per_regime=50,
            )

    def test_fit_window_metadata(self):
        """Should record fit window and metadata."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(100),
        })

        regime_labels = pd.Series([0] * 100, index=df.index)

        state = fit_regime_switching(
            df,
            regime_labels,
            transform_type="zscore",
            time_col="date",
        )

        assert state.fit_window_start == dates[0]
        assert state.fit_window_end == dates[-1]
        assert state.transform_type == "zscore"
        assert state.n_regimes == 1


class TestRegimeSwitchingTransform:
    """Test regime-specific transform application."""

    def test_apply_zscore_transform(self):
        """Apply regime-specific z-score normalization."""
        np.random.seed(42)

        # Fit on training data
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_values = np.concatenate([
            np.random.randn(50) * 1.0 + 0.0,  # Regime 0: mean=0, std=1
            np.random.randn(50) * 2.0 + 5.0,  # Regime 1: mean=5, std=2
        ])

        train_df = pd.DataFrame({
            "date": train_dates,
            "value": train_values,
        })

        train_regime = pd.Series([0] * 50 + [1] * 50, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="zscore",
            time_col="date",
        )

        # Apply to test data
        test_dates = pd.date_range("2020-04-10", periods=4, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": [0.0, 5.0, 1.0, 7.0],  # Known values
        })

        # Regimes: [0, 1, 0, 1]
        test_regime = pd.Series([0, 1, 0, 1], index=test_df.index)

        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Regime 0: (x - 0) / 1 = x
        # Regime 1: (x - 5) / 2
        expected = [
            (0.0 - fitted_state.regime_params[0]["mean"]) / fitted_state.regime_params[0]["std"],
            (5.0 - fitted_state.regime_params[1]["mean"]) / fitted_state.regime_params[1]["std"],
            (1.0 - fitted_state.regime_params[0]["mean"]) / fitted_state.regime_params[0]["std"],
            (7.0 - fitted_state.regime_params[1]["mean"]) / fitted_state.regime_params[1]["std"],
        ]

        np.testing.assert_allclose(result.values, expected, rtol=0.2)

    def test_apply_winsor_transform(self):
        """Apply regime-specific winsorization."""
        np.random.seed(42)

        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "value": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="winsor",
            time_col="date",
        )

        # Apply to test data with extreme values
        test_dates = pd.date_range("2020-04-10", periods=3, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": [-100.0, 0.0, 100.0],  # Extremes + normal
        })

        test_regime = pd.Series([0, 0, 0], index=test_df.index)

        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        lower = fitted_state.regime_params[0]["lower_bound"]
        upper = fitted_state.regime_params[0]["upper_bound"]

        # Extremes should be clipped
        assert result.iloc[0] == lower
        assert result.iloc[2] == upper
        assert lower < result.iloc[1] < upper

    def test_apply_scale_transform(self):
        """Apply regime-specific scaling."""
        np.random.seed(42)

        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "value": np.random.randn(100) * 2.0,  # std ~ 2
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="scale",
            time_col="date",
        )

        test_dates = pd.date_range("2020-04-10", periods=2, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": [4.0, 6.0],
        })

        test_regime = pd.Series([0, 0], index=test_df.index)

        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Should divide by fitted std
        expected_std = fitted_state.regime_params[0]["std"]
        np.testing.assert_allclose(
            result.values,
            [4.0 / expected_std, 6.0 / expected_std],
            rtol=0.2
        )

    def test_apply_rank_transform(self):
        """Apply cross-sectional rank."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "value": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="rank",
            time_col="date",
        )

        # Apply to test data (cross-sectional)
        test_dates = pd.date_range("2020-04-10", periods=1, freq="D")
        # Repeat same date for cross-section
        test_df = pd.DataFrame({
            "date": [test_dates[0]] * 3,
            "value": [1.0, 3.0, 2.0],
        })

        test_regime = pd.Series([0, 0, 0], index=test_df.index)

        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Should be percentile ranks [0.0, 1.0, 0.5]
        np.testing.assert_allclose(result.values, [0.0, 1.0, 0.5])

    def test_apply_none_transform(self):
        """None transform should be identity."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "value": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="none",
            time_col="date",
        )

        test_dates = pd.date_range("2020-04-10", periods=3, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": [1.0, 2.0, 3.0],
        })

        test_regime = pd.Series([0, 0, 0], index=test_df.index)

        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Should be identity
        np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0])

    def test_nan_regime_produces_nan(self):
        """NaN regime labels should produce NaN output."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "value": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="zscore",
            time_col="date",
        )

        test_dates = pd.date_range("2020-04-10", periods=3, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": [1.0, 2.0, 3.0],
        })

        test_regime = pd.Series([0, np.nan, 0], index=test_df.index)

        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Row with NaN regime should have NaN output
        assert np.isfinite(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isfinite(result.iloc[2])

    def test_unknown_regime_produces_nan(self):
        """Unknown regime should produce NaN output (fail-closed)."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "value": np.random.randn(100),
        })

        # Only fit regime 0
        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="zscore",
            time_col="date",
        )

        test_dates = pd.date_range("2020-04-10", periods=3, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": [1.0, 2.0, 3.0],
        })

        # Regime 99 is unknown
        test_regime = pd.Series([0, 99, 0], index=test_df.index)

        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Unknown regime should be NaN
        assert np.isfinite(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isfinite(result.iloc[2])

    def test_staleness_check(self):
        """Should raise if test data overlaps with fit window."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "value": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="zscore",
            time_col="date",
        )

        # Test data overlaps with training
        test_dates = pd.date_range("2020-03-01", periods=10, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": np.random.randn(10),
        })

        test_regime = pd.Series([0] * 10, index=test_df.index)

        with pytest.raises(StaleFittedStateError):
            regime_switching_transform(
                test_df,
                test_regime,
                fitted_state,
                time_col="date",
                check_staleness=True,
            )

    def test_staleness_check_disabled(self):
        """Staleness check can be disabled."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "value": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="zscore",
            time_col="date",
        )

        # Test data overlaps, but check is disabled
        test_dates = pd.date_range("2020-03-01", periods=10, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": np.random.randn(10),
        })

        test_regime = pd.Series([0] * 10, index=test_df.index)

        # Should not raise
        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
            check_staleness=False,
        )

        assert len(result) == 10

    def test_missing_fitted_state(self):
        """Should raise if fitted_state is None."""
        test_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "value": np.random.randn(10),
        })

        test_regime = pd.Series([0] * 10, index=test_df.index)

        with pytest.raises(MissingFittedStateError):
            regime_switching_transform(
                test_df,
                test_regime,
                fitted_state=None,
                time_col="date",
            )

    def test_different_regimes_different_transforms(self):
        """Different regimes should apply different transform parameters."""
        np.random.seed(42)

        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_values = np.concatenate([
            np.random.randn(50) * 1.0,
            np.random.randn(50) * 3.0,
        ])

        train_df = pd.DataFrame({
            "date": train_dates,
            "value": train_values,
        })

        train_regime = pd.Series([0] * 50 + [1] * 50, index=train_df.index)

        fitted_state = fit_regime_switching(
            train_df,
            train_regime,
            transform_type="scale",
            time_col="date",
        )

        # Apply same value to both regimes
        test_dates = pd.date_range("2020-04-10", periods=2, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "value": [2.0, 2.0],
        })

        test_regime = pd.Series([0, 1], index=test_df.index)

        result = regime_switching_transform(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Same input, different regimes -> different output
        assert result.iloc[0] != result.iloc[1]
