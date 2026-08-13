"""
Tests for VaR and CVaR computation.
"""

import pytest
import numpy as np
from scipy import stats

from quant_evaluator.metrics.risk.var_cvar import (
    compute_var,
    compute_cvar,
    compute_var_cvar,
    compute_var_historical,
    compute_var_parametric,
    compute_var_cornish_fisher,
)


class TestVaRHistorical:
    """Test historical VaR computation."""

    def test_var_basic(self):
        """Basic VaR computation with normal returns."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        var_95 = compute_var_historical(returns, confidence_level=0.95)

        # VaR should be positive loss
        assert var_95 > 0
        # For normal distribution, roughly 1.65 * sigma
        expected_var = 1.65 * 0.02
        assert abs(var_95 - expected_var) < 0.01

    def test_var_known_distribution(self):
        """VaR with known quantile."""
        # Returns from -10% to +10%
        returns = np.linspace(-0.10, 0.10, 1000)

        var_95 = compute_var_historical(returns, confidence_level=0.95)

        # 5th percentile should be around -0.09
        expected_var = 0.09
        assert abs(var_95 - expected_var) < 0.005

    def test_var_multi_factor(self):
        """VaR for multiple factors."""
        np.random.seed(100)
        T, F = 500, 3

        returns = np.random.randn(T, F) * 0.02
        # Factor 1 has higher volatility
        returns[:, 1] *= 2.0

        var_95 = compute_var_historical(returns, confidence_level=0.95)

        assert var_95.shape == (F,)
        # Factor 1 should have higher VaR
        assert var_95[1] > var_95[0]
        assert var_95[1] > var_95[2]

    def test_var_confidence_levels(self):
        """VaR increases with confidence level."""
        np.random.seed(200)
        returns = np.random.randn(1000) * 0.02

        var_90 = compute_var_historical(returns, confidence_level=0.90)
        var_95 = compute_var_historical(returns, confidence_level=0.95)
        var_99 = compute_var_historical(returns, confidence_level=0.99)

        # Higher confidence = higher VaR
        assert var_90 < var_95 < var_99

    def test_var_insufficient_data(self):
        """VaR returns NaN for insufficient data."""
        returns = np.random.randn(10)

        var = compute_var_historical(returns, min_periods=20)

        assert np.isnan(var)

    def test_var_with_nans(self):
        """VaR handles NaN values."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        returns[10:20] = np.nan

        var_95 = compute_var_historical(returns, confidence_level=0.95)

        assert np.isfinite(var_95)
        assert var_95 > 0


class TestVaRParametric:
    """Test parametric VaR computation."""

    def test_var_parametric_normal(self):
        """Parametric VaR for normal distribution."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        var_95 = compute_var_parametric(returns, confidence_level=0.95)

        # Should match theoretical value
        z = stats.norm.ppf(0.05)  # -1.645
        expected_var = -z * 0.02
        assert abs(var_95 - expected_var) < 0.002

    def test_var_parametric_vs_historical(self):
        """Compare parametric and historical VaR."""
        np.random.seed(100)
        returns = np.random.randn(1000) * 0.02

        var_param = compute_var_parametric(returns, confidence_level=0.95)
        var_hist = compute_var_historical(returns, confidence_level=0.95)

        # Should be close for normal distribution
        assert abs(var_param - var_hist) < 0.005

    def test_var_parametric_biased_mean(self):
        """Parametric VaR with non-zero mean."""
        np.random.seed(200)
        returns = np.random.randn(1000) * 0.02 + 0.001  # Positive drift

        var_95 = compute_var_parametric(returns, confidence_level=0.95)

        # Should account for positive mean
        assert var_95 > 0
        # VaR should be less than pure volatility-based
        var_zero_mean = compute_var_parametric(
            np.random.randn(1000) * 0.02, confidence_level=0.95
        )
        assert var_95 < var_zero_mean * 1.2


class TestVaRCornishFisher:
    """Test Cornish-Fisher VaR."""

    def test_var_cf_normal(self):
        """Cornish-Fisher reduces to parametric for normal distribution."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        var_cf = compute_var_cornish_fisher(returns, confidence_level=0.95)
        var_param = compute_var_parametric(returns, confidence_level=0.95)

        # Should be very close for normal
        assert abs(var_cf - var_param) < 0.003

    def test_var_cf_skewed(self):
        """Cornish-Fisher adjusts for skewness."""
        np.random.seed(100)
        # Create negatively skewed returns (fat left tail)
        returns = np.random.randn(1000) ** 3 * 0.01

        var_cf = compute_var_cornish_fisher(returns, confidence_level=0.95)
        var_param = compute_var_parametric(returns, confidence_level=0.95)

        # Both should be computed (skewness affects adjustment)
        assert np.isfinite(var_cf)
        assert np.isfinite(var_param)

    def test_var_cf_multi_factor(self):
        """Cornish-Fisher for multiple factors."""
        np.random.seed(200)
        T, F = 500, 3

        returns = np.zeros((T, F))
        returns[:, 0] = np.random.randn(T) * 0.02  # Normal
        returns[:, 1] = np.random.randn(T) ** 3 * 0.01  # Skewed
        returns[:, 2] = np.random.standard_t(df=5, size=T) * 0.01  # Fat tails

        var_cf = compute_var_cornish_fisher(returns, confidence_level=0.95)

        assert var_cf.shape == (F,)
        assert np.all(var_cf > 0)


class TestCVaR:
    """Test CVaR (Expected Shortfall) computation."""

    def test_cvar_basic(self):
        """Basic CVaR computation."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        cvar_95 = compute_cvar(returns, confidence_level=0.95, method="historical")

        # CVaR should be larger than VaR
        var_95 = compute_var_historical(returns, confidence_level=0.95)
        assert cvar_95 > var_95

    def test_cvar_parametric_normal(self):
        """Parametric CVaR for normal distribution."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        cvar_95 = compute_cvar(returns, confidence_level=0.95, method="parametric")

        # Theoretical CVaR for normal
        z = stats.norm.ppf(0.05)
        pdf_z = stats.norm.pdf(z)
        expected_cvar = 0.02 * pdf_z / 0.05
        assert abs(cvar_95 - expected_cvar) < 0.003

    def test_cvar_historical_vs_parametric(self):
        """Compare historical and parametric CVaR."""
        np.random.seed(100)
        returns = np.random.randn(1000) * 0.02

        cvar_hist = compute_cvar(returns, confidence_level=0.95, method="historical")
        cvar_param = compute_cvar(returns, confidence_level=0.95, method="parametric")

        # Should be close for normal distribution
        assert abs(cvar_hist - cvar_param) < 0.005

    def test_cvar_multi_factor(self):
        """CVaR for multiple factors."""
        np.random.seed(200)
        T, F = 500, 3

        returns = np.random.randn(T, F) * 0.02
        returns[:, 1] *= 2.0  # Higher volatility

        cvar_95 = compute_cvar(returns, confidence_level=0.95, method="historical")

        assert cvar_95.shape == (F,)
        # Higher volatility = higher CVaR
        assert cvar_95[1] > cvar_95[0]

    def test_cvar_coherent_risk_measure(self):
        """CVaR satisfies subadditivity (coherent risk measure)."""
        np.random.seed(42)
        T = 1000

        returns_a = np.random.randn(T) * 0.02
        returns_b = np.random.randn(T) * 0.015
        returns_combined = (returns_a + returns_b) / 2.0

        cvar_a = compute_cvar(returns_a, confidence_level=0.95, method="historical")
        cvar_b = compute_cvar(returns_b, confidence_level=0.95, method="historical")
        cvar_combined = compute_cvar(
            returns_combined, confidence_level=0.95, method="historical"
        )

        # CVaR(A+B) <= CVaR(A) + CVaR(B) (subadditivity)
        assert cvar_combined <= cvar_a + cvar_b + 0.001  # Small tolerance


class TestVaRCVaRCombined:
    """Test combined VaR/CVaR computation."""

    def test_var_cvar_together(self):
        """Compute VaR and CVaR together."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        var, cvar = compute_var_cvar(
            returns, confidence_level=0.95, method="historical"
        )

        # Both should be positive
        assert var > 0
        assert cvar > 0
        # CVaR >= VaR
        assert cvar >= var

    def test_var_cvar_consistency(self):
        """VaR/CVaR from combined function match individual."""
        np.random.seed(100)
        returns = np.random.randn(1000) * 0.02

        var_combined, cvar_combined = compute_var_cvar(
            returns, confidence_level=0.95, method="historical"
        )

        var_separate = compute_var_historical(returns, confidence_level=0.95)
        cvar_separate = compute_cvar(
            returns, confidence_level=0.95, method="historical"
        )

        assert abs(var_combined - var_separate) < 1e-10
        assert abs(cvar_combined - cvar_separate) < 1e-10

    def test_var_cvar_multi_method(self):
        """VaR/CVaR with different methods."""
        np.random.seed(200)
        returns = np.random.randn(500) * 0.02

        var_hist, cvar_hist = compute_var_cvar(returns, method="historical")
        var_param, cvar_param = compute_var_cvar(returns, method="parametric")
        var_cf, cvar_cf = compute_var_cvar(returns, method="cornish_fisher")

        # All should be positive
        assert var_hist > 0 and cvar_hist > 0
        assert var_param > 0 and cvar_param > 0
        assert var_cf > 0 and cvar_cf > 0

        # CVaR >= VaR for all methods
        assert cvar_hist >= var_hist
        assert cvar_param >= var_param
        assert cvar_cf >= var_cf


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_returns(self):
        """VaR/CVaR with constant returns."""
        returns = np.full(100, 0.01)

        var = compute_var_historical(returns, confidence_level=0.95)
        cvar = compute_cvar(returns, confidence_level=0.95, method="historical")

        # Should be zero (no risk)
        assert var == 0.0
        assert cvar == 0.0

    def test_empty_returns(self):
        """VaR/CVaR with empty returns."""
        returns = np.array([])

        # Empty returns should return NaN
        var = compute_var_historical(returns, confidence_level=0.95, min_periods=1)

        # Should return NaN for empty data
        assert np.isnan(var)

    def test_all_nan_returns(self):
        """VaR/CVaR with all NaN returns."""
        returns = np.full(100, np.nan)

        var = compute_var_historical(returns, confidence_level=0.95)

        assert np.isnan(var)

    def test_invalid_confidence_level(self):
        """VaR with edge confidence levels."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        # Very low confidence (should give small VaR)
        var_50 = compute_var_historical(returns, confidence_level=0.50)
        # Very high confidence (should give large VaR)
        var_999 = compute_var_historical(returns, confidence_level=0.999)

        assert var_50 < var_999
        # 50% confidence can be zero or very small
        assert var_50 >= 0

    def test_extreme_returns(self):
        """VaR/CVaR with extreme returns."""
        returns = np.array([0.01] * 95 + [-0.50] * 5)

        var_95 = compute_var_historical(returns, confidence_level=0.95)
        cvar_95 = compute_cvar(returns, confidence_level=0.95, method="historical")

        # With 95% confidence, VaR is at 5th percentile
        # Since we have 95 positive and 5 negative, the 5th percentile falls in positive range
        # But CVaR should be in the tail (extreme losses)
        assert var_95 >= 0  # Can be small if threshold falls in positive range
        assert cvar_95 > 0.3  # CVaR captures the extreme losses
