"""
Tests for Granger causality functionality.
"""

import numpy as np
import pytest

from quant_evaluator.metrics.stats.granger_causality import (
    granger_causality_test,
    pairwise_granger_causality,
    _fit_ar_model,
    _fit_var_model,
)


class TestGrangerCausality:
    """Test suite for Granger causality tests."""

    def test_granger_no_causality(self):
        """Test that independent series show no Granger causality."""
        np.random.seed(42)
        T = 200

        # Independent AR(1) processes
        y = np.zeros(T)
        x = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
            x[t] = 0.5 * x[t - 1] + np.random.randn()

        f_stat, p_value, df, best_lag = granger_causality_test(y, x, max_lags=5)

        # Should NOT reject null (high p-value)
        assert p_value > 0.05, f"False positive: p={p_value:.4f}"
        assert np.isfinite(f_stat)
        assert df >= 1
        assert 1 <= best_lag <= 5

    def test_granger_with_causality(self):
        """Test that x -> y causality is detected."""
        np.random.seed(42)
        T = 300

        # y depends on lagged x
        y = np.zeros(T)
        x = np.random.randn(T)

        for t in range(2, T):
            y[t] = 0.3 * y[t - 1] + 0.5 * x[t - 1] + 0.3 * x[t - 2] + np.random.randn() * 0.5

        f_stat, p_value, df, best_lag = granger_causality_test(y, x, max_lags=5)

        # Should reject null (low p-value)
        assert p_value < 0.05, f"False negative: p={p_value:.4f}"
        assert f_stat > 3.0  # Reasonable F-stat for strong relationship

    def test_granger_insufficient_data(self):
        """Test error handling for insufficient data."""
        y = np.random.randn(20)
        x = np.random.randn(20)

        with pytest.raises(ValueError, match="Insufficient observations"):
            granger_causality_test(y, x, max_lags=5, min_obs=30)

    def test_granger_unequal_length(self):
        """Test error for unequal series lengths."""
        y = np.random.randn(100)
        x = np.random.randn(80)

        with pytest.raises(ValueError, match="equal length"):
            granger_causality_test(y, x)

    def test_granger_with_nan(self):
        """Test that NaN values are handled."""
        np.random.seed(42)
        y = np.random.randn(150)
        x = np.random.randn(150)

        # Inject NaN
        y[10:15] = np.nan
        x[50:55] = np.nan

        f_stat, p_value, df, best_lag = granger_causality_test(y, x, max_lags=3)

        # Should still compute (with reduced sample)
        assert np.isfinite(f_stat)
        assert np.isfinite(p_value)

    def test_pairwise_granger_matrix(self):
        """Test pairwise Granger causality matrix."""
        np.random.seed(42)
        T, N = 200, 3

        # Create data with known structure:
        # x0 -> x1 -> x2
        data = np.zeros((T, N))
        data[:, 0] = np.random.randn(T)

        for t in range(1, T):
            data[t, 1] = 0.4 * data[t - 1, 1] + 0.5 * data[t - 1, 0] + np.random.randn() * 0.5
            data[t, 2] = 0.3 * data[t - 1, 2] + 0.6 * data[t - 1, 1] + np.random.randn() * 0.5

        f_matrix, p_matrix, sig_matrix = pairwise_granger_causality(
            data, max_lags=3, alpha=0.05
        )

        # Check shapes
        assert f_matrix.shape == (N, N)
        assert p_matrix.shape == (N, N)
        assert sig_matrix.shape == (N, N)

        # Diagonal should be NaN
        assert np.all(np.isnan(np.diag(f_matrix)))

        # Should detect x0 -> x1 (element [1, 0])
        assert sig_matrix[1, 0] == 1, "Should detect x0 -> x1"

        # Should detect x1 -> x2 (element [2, 1])
        assert sig_matrix[2, 1] == 1, "Should detect x1 -> x2"

    def test_fit_ar_model(self):
        """Test AR model fitting."""
        np.random.seed(42)
        T = 100

        # Generate AR(2): y_t = 0.5*y_{t-1} + 0.3*y_{t-2} + e_t
        y = np.zeros(T)
        for t in range(2, T):
            y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 2] + np.random.randn() * 0.5

        coef, rss, nobs = _fit_ar_model(y, lags=2)

        assert coef.shape == (2,)
        assert rss > 0
        assert nobs == T - 2

        # Coefficients should be close to true values
        assert np.abs(coef[0] - 0.5) < 0.2
        assert np.abs(coef[1] - 0.3) < 0.2

    def test_fit_var_model(self):
        """Test VAR model fitting."""
        np.random.seed(42)
        T = 150

        y = np.zeros(T)
        x = np.random.randn(T)

        # y depends on own lag and x lag
        for t in range(2, T):
            y[t] = 0.4 * y[t - 1] + 0.5 * x[t - 1] + np.random.randn() * 0.5

        coef, rss, nobs = _fit_var_model(y, x, lags=1)

        assert coef.shape == (2,)  # [y_lag, x_lag]
        assert rss > 0
        assert nobs == T - 1

        # x coefficient should be significant
        assert np.abs(coef[1]) > 0.2

    def test_granger_lag_selection(self):
        """Test that BIC selects reasonable lag order."""
        np.random.seed(42)
        T = 250

        # Generate with known lag structure (lag=2)
        y = np.zeros(T)
        x = np.random.randn(T)

        for t in range(3, T):
            y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 2] + np.random.randn() * 0.5

        _, _, _, best_lag = granger_causality_test(y, x, max_lags=5)

        # Should select lag around 2-3
        assert 1 <= best_lag <= 4

    def test_granger_zero_variance(self):
        """Test handling of constant series."""
        np.random.seed(42)
        y = np.ones(100)
        x = np.random.randn(100)

        # Should handle gracefully - constant y means x cannot predict it
        # F-stat should be very small (no predictive power)
        f_stat, p_value, df, best_lag = granger_causality_test(y, x, max_lags=3)

        # Should produce valid result with high p-value (x doesn't predict constant)
        assert np.isfinite(f_stat)
        assert p_value > 0.5  # High p-value for no relationship

    def test_pairwise_with_invalid_series(self):
        """Test pairwise with some invalid series."""
        np.random.seed(42)
        T, N = 150, 4

        data = np.random.randn(T, N)
        data[:, 2] = np.nan  # Entire series NaN

        f_matrix, p_matrix, sig_matrix = pairwise_granger_causality(data, max_lags=3)

        # Row 2 and column 2 should be NaN (invalid series)
        assert np.all(np.isnan(f_matrix[2, :]))
        assert np.all(np.isnan(f_matrix[:, 2]))


class TestGrangerEdgeCases:
    """Test edge cases and error handling."""

    def test_max_lags_too_large(self):
        """Test error when max_lags is too large."""
        y = np.random.randn(50)
        x = np.random.randn(50)

        with pytest.raises(ValueError, match="too large"):
            granger_causality_test(y, x, max_lags=30)

    def test_multidimensional_input(self):
        """Test error for multidimensional input."""
        y = np.random.randn(100, 2)
        x = np.random.randn(100)

        with pytest.raises(ValueError, match="1-dimensional"):
            granger_causality_test(y, x)

    def test_pairwise_wrong_shape(self):
        """Test error for wrong data shape."""
        data = np.random.randn(100)  # Should be 2D

        with pytest.raises(ValueError, match="2D"):
            pairwise_granger_causality(data)

    def test_granger_deterministic_relationship(self):
        """Test with perfect deterministic relationship."""
        np.random.seed(42)
        T = 200
        x = np.random.randn(T)
        y = np.zeros(T)

        # Perfect relationship: y_t = x_{t-1}
        for t in range(1, T):
            y[t] = x[t - 1]

        f_stat, p_value, df, best_lag = granger_causality_test(y, x, max_lags=5)

        # Should strongly reject null
        assert p_value < 0.001
        assert f_stat > 10.0
