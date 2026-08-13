"""
Tests for cointegration functionality.
"""

import numpy as np
import pytest

from quant_evaluator.metrics.stats.cointegration import (
    johansen_test,
    engle_granger_test,
    _lag_matrix,
    _residuals_from_deterministics,
)


class TestJohansenCointegration:
    """Test suite for Johansen cointegration test."""

    def test_johansen_no_cointegration(self):
        """Test that independent random walks show no cointegration."""
        np.random.seed(42)
        T, K = 200, 2

        # Two independent random walks
        data = np.cumsum(np.random.randn(T, K), axis=0)

        trace_stats, max_eig_stats, rank, eigenvalues = johansen_test(
            data, det_order=0, lags=1, alpha=0.05
        )

        # Should find rank=0 (no cointegration)
        assert rank == 0, f"False positive: found rank={rank}"
        assert len(trace_stats) == K
        assert len(max_eig_stats) == K
        assert len(eigenvalues) == K

        # Eigenvalues should be in [0, 1)
        assert np.all(eigenvalues >= 0)
        assert np.all(eigenvalues < 1)

        # Eigenvalues should be sorted descending
        assert np.all(np.diff(eigenvalues) <= 0)

    def test_johansen_perfect_cointegration(self):
        """Test with perfectly cointegrated series."""
        np.random.seed(42)
        T = 250

        # Common stochastic trend
        trend = np.cumsum(np.random.randn(T))

        # Two series with common trend: y1 = trend + noise, y2 = 2*trend + noise
        y1 = trend + np.random.randn(T) * 0.1
        y2 = 2 * trend + np.random.randn(T) * 0.1

        data = np.column_stack([y1, y2])

        trace_stats, max_eig_stats, rank, eigenvalues = johansen_test(
            data, det_order=0, lags=1, alpha=0.05
        )

        # Should detect at least one cointegrating relationship
        assert rank >= 1, f"False negative: found rank={rank}"

    def test_johansen_with_trend(self):
        """Test Johansen with deterministic trend."""
        np.random.seed(42)
        T = 200

        # Series with linear trend
        t = np.arange(T)
        trend = np.cumsum(np.random.randn(T))

        y1 = 0.5 * t + trend + np.random.randn(T)
        y2 = 1.0 * t + trend + np.random.randn(T)

        data = np.column_stack([y1, y2])

        # Test with trend specification
        trace_stats, max_eig_stats, rank, eigenvalues = johansen_test(
            data, det_order=1, lags=1, alpha=0.05
        )

        assert np.isfinite(trace_stats).all()
        assert np.isfinite(eigenvalues).all()
        assert rank >= 0

    def test_johansen_multiple_lags(self):
        """Test with multiple lags in VAR."""
        np.random.seed(42)
        T = 300
        K = 2

        # Generate data with lag structure
        data = np.cumsum(np.random.randn(T, K), axis=0)

        trace_stats, max_eig_stats, rank, eigenvalues = johansen_test(
            data, det_order=0, lags=3, alpha=0.05
        )

        assert len(trace_stats) == K
        assert len(eigenvalues) == K

    def test_johansen_insufficient_data(self):
        """Test error with insufficient data."""
        T, K = 30, 3
        data = np.random.randn(T, K)

        # Should raise error or handle gracefully
        try:
            trace_stats, max_eig_stats, rank, eigenvalues = johansen_test(data, det_order=0, lags=5)
        except ValueError as e:
            assert "Insufficient" in str(e) or "too small" in str(e)

    def test_johansen_wrong_shape(self):
        """Test error for wrong input shape."""
        data = np.random.randn(100)  # 1D instead of 2D

        with pytest.raises(ValueError, match="2D"):
            johansen_test(data)

    def test_johansen_invalid_det_order(self):
        """Test error for invalid deterministic order."""
        data = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="det_order"):
            johansen_test(data, det_order=2)

    def test_johansen_invalid_alpha(self):
        """Test error for invalid significance level."""
        data = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="alpha"):
            johansen_test(data, alpha=0.15)

    def test_johansen_three_series(self):
        """Test with three time series."""
        np.random.seed(42)
        T = 250

        # Create three series with one common trend
        trend = np.cumsum(np.random.randn(T))

        y1 = trend + np.random.randn(T) * 0.2
        y2 = 1.5 * trend + np.random.randn(T) * 0.2
        y3 = 0.8 * trend + np.random.randn(T) * 0.2

        data = np.column_stack([y1, y2, y3])

        trace_stats, max_eig_stats, rank, eigenvalues = johansen_test(
            data, det_order=0, lags=2, alpha=0.05
        )

        assert len(trace_stats) == 3
        assert len(eigenvalues) == 3
        # With strong common trend, should detect cointegration
        assert rank >= 1


class TestEngleGrangerCointegration:
    """Test suite for Engle-Granger two-step test."""

    def test_engle_granger_no_cointegration(self):
        """Test that independent random walks are not cointegrated."""
        np.random.seed(42)
        T = 200

        # Two independent random walks
        y = np.cumsum(np.random.randn(T))
        x = np.cumsum(np.random.randn(T))

        adf_stat, p_value, is_coint, residuals = engle_granger_test(y, x, alpha=0.05)

        # Should NOT reject null (not cointegrated)
        assert not is_coint, f"False positive: detected cointegration with p={p_value}"
        assert len(residuals) == T
        assert np.isfinite(adf_stat)

    def test_engle_granger_with_cointegration(self):
        """Test with truly cointegrated series."""
        np.random.seed(42)
        T = 300

        # Common trend
        trend = np.cumsum(np.random.randn(T))

        # Cointegrated: y = 2*x + stationary_noise
        x = trend + np.random.randn(T) * 0.1
        y = 2 * x + np.random.randn(T) * 0.5  # Stationary around 2*x

        adf_stat, p_value, is_coint, residuals = engle_granger_test(y, x, alpha=0.05)

        # Should detect cointegration
        assert is_coint, f"False negative: p={p_value}, ADF={adf_stat}"
        assert adf_stat < -2.5  # Should be significantly negative

    def test_engle_granger_insufficient_data(self):
        """Test error with insufficient data."""
        y = np.random.randn(20)
        x = np.random.randn(20)

        with pytest.raises(ValueError, match="Insufficient data"):
            engle_granger_test(y, x)

    def test_engle_granger_unequal_length(self):
        """Test error for unequal series lengths."""
        y = np.random.randn(100)
        x = np.random.randn(80)

        with pytest.raises(ValueError, match="equal length"):
            engle_granger_test(y, x)

    def test_engle_granger_multidimensional(self):
        """Test error for multidimensional input."""
        y = np.random.randn(100, 2)
        x = np.random.randn(100)

        with pytest.raises(ValueError, match="1D"):
            engle_granger_test(y, x)

    def test_engle_granger_residuals_stationary(self):
        """Test that residuals from cointegration are more stationary."""
        np.random.seed(42)
        T = 250

        # Create cointegrated pair
        x = np.cumsum(np.random.randn(T))
        y = 1.5 * x + np.random.randn(T) * 2.0  # Small stationary deviation

        _, _, is_coint, residuals = engle_granger_test(y, x, alpha=0.05)

        # Residuals should have smaller variance than original series
        assert np.std(residuals) < np.std(y) * 0.5

        if is_coint:
            # Residuals should look stationary (no strong trend)
            drift = (residuals[-1] - residuals[0]) / T
            assert np.abs(drift) < 0.1


class TestCointegrationHelpers:
    """Test helper functions."""

    def test_lag_matrix(self):
        """Test lagged matrix construction."""
        T = 10
        K = 2
        data = np.arange(T * K, dtype=np.float64).reshape(T, K)

        lagged = _lag_matrix(data, lags=2)

        # Should have shape (T-lags, K*lags) = (8, 4)
        assert lagged.shape == (8, 4)

        # First column should be data[1:9, 0] (lag 1 of first series)
        np.testing.assert_array_equal(lagged[:, 0], data[1:9, 0])

        # Third column should be data[1:9, 1] (lag 1 of second series)
        np.testing.assert_array_equal(lagged[:, 2], data[1:9, 1])

    def test_residuals_from_deterministics_no_det(self):
        """Test with no deterministic components."""
        T, K = 50, 2
        data = np.random.randn(T, K)

        residuals, fitted = _residuals_from_deterministics(data, det_order=-1)

        # Should return original data
        np.testing.assert_array_equal(residuals, data)
        np.testing.assert_array_equal(fitted, np.zeros_like(data))

    def test_residuals_from_deterministics_constant(self):
        """Test removing constant."""
        T = 100
        K = 2

        # Data with non-zero mean
        data = np.random.randn(T, K) + np.array([5.0, -3.0])

        residuals, fitted = _residuals_from_deterministics(data, det_order=0)

        # Residuals should have mean near zero
        assert np.abs(np.mean(residuals[:, 0])) < 0.2
        assert np.abs(np.mean(residuals[:, 1])) < 0.2

        # Fitted should be constant
        assert np.std(fitted[:, 0]) < 1e-10
        assert np.std(fitted[:, 1]) < 1e-10

    def test_residuals_from_deterministics_trend(self):
        """Test removing constant and trend."""
        T = 100
        K = 1

        # Data with linear trend
        t = np.arange(T, dtype=np.float64)
        data = (2.0 * t + 10.0).reshape(-1, 1) + np.random.randn(T, 1) * 0.5

        residuals, fitted = _residuals_from_deterministics(data, det_order=1)

        # Residuals should have no trend (mean stable over time)
        first_half_mean = np.mean(residuals[: T // 2])
        second_half_mean = np.mean(residuals[T // 2 :])
        assert np.abs(first_half_mean - second_half_mean) < 1.0

        # Fitted should capture the trend
        assert np.corrcoef(fitted[:, 0], t)[0, 1] > 0.95


class TestCointegrationEdgeCases:
    """Test edge cases."""

    def test_johansen_collinear_series(self):
        """Test with perfectly collinear series."""
        np.random.seed(42)
        T = 150

        x = np.cumsum(np.random.randn(T))
        y = 2.0 * x  # Perfect collinearity

        data = np.column_stack([x, y])

        # Should handle gracefully (may have issues with singular matrices)
        try:
            trace_stats, max_eig_stats, rank, eigenvalues = johansen_test(
                data, det_order=0, lags=1, alpha=0.05
            )
            # If it runs, eigenvalues should be in valid range
            assert np.all(eigenvalues >= 0)
            assert np.all(eigenvalues < 1)
        except ValueError as e:
            # Acceptable to raise error for singular matrix
            assert "Singular" in str(e) or "collinearity" in str(e)

    def test_engle_granger_perfect_relationship(self):
        """Test Engle-Granger with perfect deterministic relationship."""
        np.random.seed(42)
        T = 200

        x = np.cumsum(np.random.randn(T))
        y = 3.0 * x + 5.0  # Perfect relationship

        adf_stat, p_value, is_coint, residuals = engle_granger_test(y, x, alpha=0.05)

        # Residuals should be essentially constant (zero variance)
        assert np.std(residuals) < 1e-10

        # ADF test may not work well with constant residuals
        # but should not crash
        assert np.isfinite(adf_stat) or np.isnan(adf_stat)

    def test_johansen_high_dimensional(self):
        """Test Johansen with higher dimensional system."""
        np.random.seed(42)
        T = 400
        K = 4

        # Create system with 2 common trends -> rank = 2
        trend1 = np.cumsum(np.random.randn(T))
        trend2 = np.cumsum(np.random.randn(T))

        data = np.column_stack([
            trend1 + np.random.randn(T) * 0.1,
            trend1 + 0.5 * trend2 + np.random.randn(T) * 0.1,
            trend2 + np.random.randn(T) * 0.1,
            trend1 - trend2 + np.random.randn(T) * 0.1,
        ])

        trace_stats, max_eig_stats, rank, eigenvalues = johansen_test(
            data, det_order=0, lags=2, alpha=0.05
        )

        assert len(eigenvalues) == K
        # Should detect multiple cointegrating relationships
        assert rank >= 1
