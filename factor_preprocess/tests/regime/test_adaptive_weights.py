"""
Tests for regime-adaptive factor weighting.

Verifies fitting, application, causality, and fold-safety.
"""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.regime.adaptive_weights import (
    fit_regime_weights,
    regime_adaptive_weights,
    RegimeWeightState,
)
from factor_preprocess.errors import (
    InsufficientObservations,
    MissingFittedStateError,
    StaleFittedStateError,
)


class TestFitRegimeWeights:
    """Test regime weight fitting."""

    def test_equal_weights(self):
        """Equal weighting should give 1/n per factor."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "factor1": np.random.randn(100),
            "factor2": np.random.randn(100),
            "factor3": np.random.randn(100),
        })

        # Simple two-regime labels
        regime_labels = pd.Series([0] * 50 + [1] * 50, index=df.index)

        state = fit_regime_weights(
            df,
            regime_labels,
            method="equal",
            time_col="date",
            factor_cols=["factor1", "factor2", "factor3"],
        )

        # Each regime should have equal weights
        assert len(state.regime_weights) == 2
        np.testing.assert_allclose(state.regime_weights[0], [1/3, 1/3, 1/3])
        np.testing.assert_allclose(state.regime_weights[1], [1/3, 1/3, 1/3])

    def test_volatility_inverse_weights(self):
        """Inverse volatility weighting should favor low-vol factors."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Factor 1: low vol
        # Factor 2: high vol
        df = pd.DataFrame({
            "date": dates,
            "factor1": np.random.randn(100) * 0.01,
            "factor2": np.random.randn(100) * 0.10,
        })

        regime_labels = pd.Series([0] * 100, index=df.index)

        state = fit_regime_weights(
            df,
            regime_labels,
            method="volatility_inverse",
            time_col="date",
            factor_cols=["factor1", "factor2"],
        )

        weights = state.regime_weights[0]

        # Factor 1 (low vol) should have higher weight
        assert weights[0] > weights[1]

        # Weights should sum to 1
        np.testing.assert_allclose(np.sum(weights), 1.0)

    def test_sharpe_weights(self):
        """Sharpe weighting should favor factors correlated with target."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Factor 1: correlated with target
        # Factor 2: uncorrelated with target
        target = np.random.randn(100)
        factor1 = target + np.random.randn(100) * 0.1  # High correlation
        factor2 = np.random.randn(100)  # No correlation

        df = pd.DataFrame({
            "date": dates,
            "factor1": factor1,
            "factor2": factor2,
        })

        regime_labels = pd.Series([0] * 100, index=df.index)
        target_series = pd.Series(target, index=df.index)

        state = fit_regime_weights(
            df,
            regime_labels,
            target=target_series,
            method="sharpe",
            time_col="date",
            factor_cols=["factor1", "factor2"],
        )

        weights = state.regime_weights[0]

        # Factor 1 should have higher weight
        assert weights[0] > weights[1]

        # Weights should sum to 1
        np.testing.assert_allclose(np.sum(weights), 1.0)

    def test_different_weights_per_regime(self):
        """Different regimes should produce different weights."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Regime 0: factor1 low vol
        # Regime 1: factor1 high vol
        factor1 = np.concatenate([
            np.random.randn(50) * 0.01,  # Low vol
            np.random.randn(50) * 0.10,  # High vol
        ])
        factor2 = np.random.randn(100) * 0.05  # Constant vol

        df = pd.DataFrame({
            "date": dates,
            "factor1": factor1,
            "factor2": factor2,
        })

        regime_labels = pd.Series([0] * 50 + [1] * 50, index=df.index)

        state = fit_regime_weights(
            df,
            regime_labels,
            method="volatility_inverse",
            time_col="date",
            factor_cols=["factor1", "factor2"],
        )

        weights_regime0 = state.regime_weights[0]
        weights_regime1 = state.regime_weights[1]

        # Weights should be different across regimes
        assert not np.allclose(weights_regime0, weights_regime1)

        # Regime 0: factor1 has low vol -> higher weight
        assert weights_regime0[0] > weights_regime0[1]

        # Regime 1: factor1 has high vol -> lower weight
        assert weights_regime1[0] < weights_regime1[1]

    def test_insufficient_observations(self):
        """Should raise if regime has too few observations."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "factor1": np.random.randn(20),
        })

        regime_labels = pd.Series([0] * 20, index=df.index)

        with pytest.raises(InsufficientObservations):
            fit_regime_weights(
                df,
                regime_labels,
                method="equal",
                time_col="date",
                min_obs_per_regime=50,
            )

    def test_fit_window_metadata(self):
        """Should record fit window start/end."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "factor1": np.random.randn(100),
        })

        regime_labels = pd.Series([0] * 100, index=df.index)

        state = fit_regime_weights(
            df,
            regime_labels,
            method="equal",
            time_col="date",
        )

        assert state.fit_window_start == dates[0]
        assert state.fit_window_end == dates[-1]

    def test_nan_regime_labels_ignored(self):
        """NaN regime labels should be ignored during fitting."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "factor1": np.random.randn(100),
        })

        # First 20 are NaN (warmup), then regime 0
        regime_labels = pd.Series([np.nan] * 20 + [0] * 80, index=df.index)

        state = fit_regime_weights(
            df,
            regime_labels,
            method="equal",
            time_col="date",
            min_obs_per_regime=30,
        )

        # Should fit successfully with 80 valid observations
        assert 0 in state.regime_weights


class TestRegimeAdaptiveWeights:
    """Test regime-adaptive weight application."""

    def test_apply_weights(self):
        """Weights should be applied row-wise based on regime."""
        np.random.seed(42)

        # Fit on training data
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "factor1": np.random.randn(100),
            "factor2": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 50 + [1] * 50, index=train_df.index)

        fitted_state = fit_regime_weights(
            train_df,
            train_regime,
            method="equal",
            time_col="date",
            factor_cols=["factor1", "factor2"],
        )

        # Apply to test data
        test_dates = pd.date_range("2020-04-10", periods=50, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "factor1": np.ones(50) * 2.0,
            "factor2": np.ones(50) * 4.0,
        })

        test_regime = pd.Series([0] * 50, index=test_df.index)

        result = regime_adaptive_weights(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
            factor_cols=["factor1", "factor2"],
        )

        # Equal weights: factor1 * 0.5, factor2 * 0.5
        np.testing.assert_allclose(result["factor1"].values, 2.0 * 0.5)
        np.testing.assert_allclose(result["factor2"].values, 4.0 * 0.5)

    def test_different_weights_per_regime(self):
        """Different regimes should apply different weights."""
        np.random.seed(42)

        # Fit with regime-specific weights
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Regime 0: factor1 low vol -> high weight
        # Regime 1: factor1 high vol -> low weight
        factor1 = np.concatenate([
            np.random.randn(50) * 0.01,
            np.random.randn(50) * 0.10,
        ])
        factor2 = np.random.randn(100) * 0.05

        train_df = pd.DataFrame({
            "date": train_dates,
            "factor1": factor1,
            "factor2": factor2,
        })

        train_regime = pd.Series([0] * 50 + [1] * 50, index=train_df.index)

        fitted_state = fit_regime_weights(
            train_df,
            train_regime,
            method="volatility_inverse",
            time_col="date",
            factor_cols=["factor1", "factor2"],
        )

        # Apply to test data with known regimes
        test_dates = pd.date_range("2020-04-10", periods=4, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "factor1": [1.0, 1.0, 1.0, 1.0],
            "factor2": [1.0, 1.0, 1.0, 1.0],
        })

        # Alternating regimes
        test_regime = pd.Series([0, 1, 0, 1], index=test_df.index)

        result = regime_adaptive_weights(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
            factor_cols=["factor1", "factor2"],
        )

        # Regime 0 and 1 should produce different weighted values
        assert result.loc[0, "factor1"] != result.loc[1, "factor1"]

    def test_nan_regime_produces_nan(self):
        """NaN regime labels should produce NaN weighted factors."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "factor1": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_weights(
            train_df,
            train_regime,
            method="equal",
            time_col="date",
        )

        # Test with NaN regime
        test_dates = pd.date_range("2020-04-10", periods=3, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "factor1": [1.0, 2.0, 3.0],
        })

        test_regime = pd.Series([0, np.nan, 0], index=test_df.index)

        result = regime_adaptive_weights(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Row with NaN regime should have NaN output
        assert np.isnan(result.loc[1, "factor1"])
        assert np.isfinite(result.loc[0, "factor1"])
        assert np.isfinite(result.loc[2, "factor1"])

    def test_staleness_check(self):
        """Should raise if test data overlaps with fit window."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "factor1": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_weights(
            train_df,
            train_regime,
            method="equal",
            time_col="date",
        )

        # Test data overlaps with training window
        test_dates = pd.date_range("2020-03-01", periods=10, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "factor1": np.random.randn(10),
        })

        test_regime = pd.Series([0] * 10, index=test_df.index)

        with pytest.raises(StaleFittedStateError):
            regime_adaptive_weights(
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
            "factor1": np.random.randn(100),
        })

        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_weights(
            train_df,
            train_regime,
            method="equal",
            time_col="date",
        )

        # Test data overlaps, but check is disabled
        test_dates = pd.date_range("2020-03-01", periods=10, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "factor1": np.random.randn(10),
        })

        test_regime = pd.Series([0] * 10, index=test_df.index)

        # Should not raise
        result = regime_adaptive_weights(
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
            "factor1": np.random.randn(10),
        })

        test_regime = pd.Series([0] * 10, index=test_df.index)

        with pytest.raises(MissingFittedStateError):
            regime_adaptive_weights(
                test_df,
                test_regime,
                fitted_state=None,
                time_col="date",
            )

    def test_unknown_regime_preserves_original(self):
        """Unknown regime fails closed to NaN (no silent raw pass-through)."""
        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        train_df = pd.DataFrame({
            "date": train_dates,
            "factor1": np.random.randn(100),
        })

        # Only fit regime 0
        train_regime = pd.Series([0] * 100, index=train_df.index)

        fitted_state = fit_regime_weights(
            train_df,
            train_regime,
            method="equal",
            time_col="date",
        )

        # Test with regime 99 (unknown)
        test_dates = pd.date_range("2020-04-10", periods=3, freq="D")
        test_df = pd.DataFrame({
            "date": test_dates,
            "factor1": [1.0, 2.0, 3.0],
        })

        test_regime = pd.Series([99, 99, 99], index=test_df.index)

        result = regime_adaptive_weights(
            test_df,
            test_regime,
            fitted_state,
            time_col="date",
        )

        # Unknown regime must not silently pass raw values through
        assert np.all(np.isnan(result["factor1"].values))
