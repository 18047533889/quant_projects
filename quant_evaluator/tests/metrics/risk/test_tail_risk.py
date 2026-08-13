"""
Tests for tail risk measures.
"""

import pytest
import numpy as np

from quant_evaluator.metrics.risk.tail_risk import (
    compute_tail_ratio,
    compute_gain_loss_ratio,
    compute_upside_potential_ratio,
    compute_omega_ratio,
    compute_expected_shortfall_ratio,
    compute_tail_dependence,
)


class TestTailRatio:
    """Test tail ratio computation."""

    def test_tail_ratio_symmetric(self):
        """Tail ratio for symmetric distribution."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        tail_ratio = compute_tail_ratio(returns, upper_percentile=0.95, lower_percentile=0.05)

        # For symmetric distribution, should be close to 1.0
        assert 0.8 < tail_ratio < 1.2

    def test_tail_ratio_positive_skew(self):
        """Tail ratio for positively skewed distribution."""
        np.random.seed(100)
        # Positive skew (right tail larger)
        returns = np.random.exponential(scale=0.02, size=1000) - 0.02

        tail_ratio = compute_tail_ratio(returns)

        # Right tail larger than left tail
        assert tail_ratio > 1.0

    def test_tail_ratio_negative_skew(self):
        """Tail ratio for negatively skewed distribution."""
        np.random.seed(200)
        # Negative skew (left tail larger)
        returns = -(np.random.exponential(scale=0.02, size=1000) - 0.02)

        tail_ratio = compute_tail_ratio(returns)

        # Left tail larger than right tail
        assert tail_ratio < 1.0

    def test_tail_ratio_multi_factor(self):
        """Tail ratio for multiple factors."""
        np.random.seed(42)
        T, F = 500, 3

        returns = np.zeros((T, F))
        returns[:, 0] = np.random.randn(T) * 0.02  # Symmetric
        returns[:, 1] = np.random.exponential(scale=0.02, size=T) - 0.02  # Positive skew
        returns[:, 2] = -(np.random.exponential(scale=0.02, size=T) - 0.02)  # Negative skew

        tail_ratio = compute_tail_ratio(returns)

        assert tail_ratio.shape == (F,)
        # Factor 1 should have highest ratio
        assert tail_ratio[1] > tail_ratio[0] > tail_ratio[2]

    def test_tail_ratio_custom_percentiles(self):
        """Tail ratio with custom percentiles."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        tail_90_10 = compute_tail_ratio(returns, upper_percentile=0.90, lower_percentile=0.10)
        tail_99_01 = compute_tail_ratio(returns, upper_percentile=0.99, lower_percentile=0.01)

        # Both should be computed
        assert np.isfinite(tail_90_10)
        assert np.isfinite(tail_99_01)


class TestGainLossRatio:
    """Test gain/loss ratio computation."""

    def test_gain_loss_symmetric(self):
        """Gain/loss ratio for symmetric distribution."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        gl_ratio = compute_gain_loss_ratio(returns)

        # For symmetric distribution, should be close to 1.0
        assert 0.8 < gl_ratio < 1.2

    def test_gain_loss_positive_bias(self):
        """Gain/loss ratio with positive bias."""
        np.random.seed(100)
        returns = np.random.randn(1000) * 0.02 + 0.005  # Positive drift

        gl_ratio = compute_gain_loss_ratio(returns)

        # More gains than losses
        assert gl_ratio > 1.0

    def test_gain_loss_negative_bias(self):
        """Gain/loss ratio with negative bias."""
        np.random.seed(200)
        returns = np.random.randn(1000) * 0.02 - 0.005  # Negative drift

        gl_ratio = compute_gain_loss_ratio(returns)

        # More losses than gains
        assert gl_ratio < 1.0

    def test_gain_loss_custom_threshold(self):
        """Gain/loss ratio with custom threshold."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        gl_zero = compute_gain_loss_ratio(returns, threshold=0.0)
        gl_positive = compute_gain_loss_ratio(returns, threshold=0.005)

        # Both should be computed
        assert np.isfinite(gl_zero)
        assert np.isfinite(gl_positive)

    def test_gain_loss_multi_factor(self):
        """Gain/loss ratio for multiple factors."""
        np.random.seed(42)
        T, F = 500, 3

        returns = np.random.randn(T, F) * 0.02
        returns[:, 1] += 0.005  # Positive bias

        gl_ratio = compute_gain_loss_ratio(returns)

        assert gl_ratio.shape == (F,)
        # Factor 1 should have higher ratio
        assert gl_ratio[1] > gl_ratio[0]


class TestUpsidePotentialRatio:
    """Test Upside Potential Ratio computation."""

    def test_upr_basic(self):
        """Basic UPR computation."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        upr = compute_upside_potential_ratio(returns, minimum_acceptable_return=0.0)

        # Should be positive
        assert upr > 0

    def test_upr_positive_returns(self):
        """UPR with mostly positive returns."""
        np.random.seed(100)
        returns = np.abs(np.random.randn(1000) * 0.02)

        upr = compute_upside_potential_ratio(returns, minimum_acceptable_return=0.0)

        # All positive returns = no downside deviation = NaN
        assert np.isnan(upr)

    def test_upr_negative_returns(self):
        """UPR with mostly negative returns."""
        np.random.seed(200)
        returns = -np.abs(np.random.randn(1000) * 0.02)

        upr = compute_upside_potential_ratio(returns, minimum_acceptable_return=0.0)

        # Low UPR for negative returns
        assert upr < 0.5

    def test_upr_mar_adjustment(self):
        """UPR with different MAR thresholds."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02 + 0.002

        upr_zero = compute_upside_potential_ratio(returns, minimum_acceptable_return=0.0)
        upr_positive = compute_upside_potential_ratio(returns, minimum_acceptable_return=0.001)

        # Both should be computed
        assert np.isfinite(upr_zero)
        assert np.isfinite(upr_positive)

    def test_upr_multi_factor(self):
        """UPR for multiple factors."""
        np.random.seed(42)
        T, F = 500, 3

        returns = np.random.randn(T, F) * 0.02

        upr = compute_upside_potential_ratio(returns)

        assert upr.shape == (F,)
        assert np.all(upr > 0)


class TestOmegaRatio:
    """Test Omega ratio computation."""

    def test_omega_basic(self):
        """Basic Omega ratio computation."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        omega = compute_omega_ratio(returns, threshold=0.0)

        # Should be close to 1.0 for symmetric distribution
        assert 0.8 < omega < 1.2

    def test_omega_positive_returns(self):
        """Omega ratio with mostly positive returns."""
        np.random.seed(100)
        returns = np.random.randn(1000) * 0.02 + 0.005

        omega = compute_omega_ratio(returns, threshold=0.0)

        # High Omega for positive returns
        assert omega > 1.5

    def test_omega_negative_returns(self):
        """Omega ratio with mostly negative returns."""
        np.random.seed(200)
        returns = np.random.randn(1000) * 0.02 - 0.005

        omega = compute_omega_ratio(returns, threshold=0.0)

        # Low Omega for negative returns
        assert omega < 0.8

    def test_omega_threshold_effect(self):
        """Omega ratio with different thresholds."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02 + 0.001

        omega_zero = compute_omega_ratio(returns, threshold=0.0)
        omega_negative = compute_omega_ratio(returns, threshold=-0.01)
        omega_positive = compute_omega_ratio(returns, threshold=0.01)

        # Higher threshold = lower Omega
        assert omega_negative > omega_zero > omega_positive

    def test_omega_multi_factor(self):
        """Omega ratio for multiple factors."""
        np.random.seed(42)
        T, F = 500, 3

        returns = np.random.randn(T, F) * 0.02

        omega = compute_omega_ratio(returns)

        assert omega.shape == (F,)
        assert np.all(omega > 0)


class TestExpectedShortfallRatio:
    """Test Expected Shortfall Ratio computation."""

    def test_es_ratio_basic(self):
        """Basic ES ratio computation."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        es_ratio = compute_expected_shortfall_ratio(
            returns, confidence_level=0.95, periods_per_year=252
        )

        # Should be finite
        assert np.isfinite(es_ratio)

    def test_es_ratio_positive_returns(self):
        """ES ratio with positive expected returns."""
        np.random.seed(100)
        returns = np.random.randn(1000) * 0.02 + 0.003

        es_ratio = compute_expected_shortfall_ratio(returns, periods_per_year=252)

        # Should be positive
        assert es_ratio > 0

    def test_es_ratio_similar_to_sharpe(self):
        """ES ratio similar in spirit to Sharpe ratio."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02 + 0.001

        from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio

        es_ratio = compute_expected_shortfall_ratio(returns, periods_per_year=252)
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)

        # Both should have same sign for return/risk
        assert (es_ratio > 0) == (sharpe > 0)

    def test_es_ratio_multi_factor(self):
        """ES ratio for multiple factors."""
        np.random.seed(42)
        T, F = 500, 3

        returns = np.random.randn(T, F) * 0.02

        es_ratio = compute_expected_shortfall_ratio(returns, periods_per_year=252)

        assert es_ratio.shape == (F,)


class TestTailDependence:
    """Test tail dependence computation."""

    def test_tail_dep_independent(self):
        """Tail dependence for independent series."""
        np.random.seed(42)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        returns_y = np.random.randn(T) * 0.02

        lower_dep, upper_dep = compute_tail_dependence(
            returns_x, returns_y, quantile=0.05
        )

        # Should be close to quantile (0.05) for independent
        assert 0.0 < lower_dep < 0.15
        assert 0.0 < upper_dep < 0.15

    def test_tail_dep_perfect_correlation(self):
        """Tail dependence for perfectly correlated series."""
        np.random.seed(100)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        returns_y = returns_x.copy()  # Perfect correlation

        lower_dep, upper_dep = compute_tail_dependence(
            returns_x, returns_y, quantile=0.05
        )

        # Should be close to 1.0 for perfect correlation
        assert lower_dep > 0.8
        assert upper_dep > 0.8

    def test_tail_dep_partial_correlation(self):
        """Tail dependence for partially correlated series."""
        np.random.seed(200)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        returns_y = 0.5 * returns_x + 0.5 * np.random.randn(T) * 0.02

        lower_dep, upper_dep = compute_tail_dependence(
            returns_x, returns_y, quantile=0.05
        )

        # Should be between independent and perfect
        assert 0.1 < lower_dep < 0.8
        assert 0.1 < upper_dep < 0.8

    def test_tail_dep_asymmetric(self):
        """Tail dependence can be asymmetric."""
        np.random.seed(42)
        T = 1000

        returns_x = np.random.randn(T) * 0.02

        # Create asymmetric dependence (stronger in lower tail)
        returns_y = np.where(
            returns_x < 0,
            0.8 * returns_x + 0.2 * np.random.randn(T) * 0.02,  # Strong dependence in lower tail
            0.2 * returns_x + 0.8 * np.random.randn(T) * 0.02,  # Weak dependence in upper tail
        )

        lower_dep, upper_dep = compute_tail_dependence(
            returns_x, returns_y, quantile=0.05
        )

        # Lower tail dependence should be stronger
        assert lower_dep > upper_dep

    def test_tail_dep_custom_quantile(self):
        """Tail dependence with custom quantile."""
        np.random.seed(42)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        returns_y = 0.5 * returns_x + 0.5 * np.random.randn(T) * 0.02

        lower_01, upper_01 = compute_tail_dependence(
            returns_x, returns_y, quantile=0.01
        )
        lower_10, upper_10 = compute_tail_dependence(
            returns_x, returns_y, quantile=0.10
        )

        # Both should be computed
        assert np.isfinite(lower_01)
        assert np.isfinite(upper_01)
        assert np.isfinite(lower_10)
        assert np.isfinite(upper_10)

    def test_tail_dep_insufficient_data(self):
        """Tail dependence with insufficient data."""
        returns_x = np.random.randn(10)
        returns_y = np.random.randn(10)

        lower_dep, upper_dep = compute_tail_dependence(
            returns_x, returns_y, min_periods=50
        )

        # Should return NaN
        assert np.isnan(lower_dep)
        assert np.isnan(upper_dep)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tail_ratio_zero_tail(self):
        """Tail ratio when lower tail is zero."""
        returns = np.full(100, 0.01)

        tail_ratio = compute_tail_ratio(returns)

        # Should handle gracefully (returns NaN)
        assert np.isnan(tail_ratio) or tail_ratio > 0

    def test_gain_loss_all_gains(self):
        """Gain/loss ratio with all gains."""
        returns = np.abs(np.random.randn(100) * 0.02)

        gl_ratio = compute_gain_loss_ratio(returns)

        # Should return NaN (no losses)
        assert np.isnan(gl_ratio)

    def test_gain_loss_all_losses(self):
        """Gain/loss ratio with all losses."""
        returns = -np.abs(np.random.randn(100) * 0.02)

        gl_ratio = compute_gain_loss_ratio(returns)

        # Should return NaN (no gains)
        assert np.isnan(gl_ratio)

    def test_upr_no_downside(self):
        """UPR with no downside."""
        returns = np.abs(np.random.randn(100) * 0.02)

        upr = compute_upside_potential_ratio(returns, minimum_acceptable_return=0.0)

        # Should return NaN (no downside deviation)
        assert np.isnan(upr)

    def test_omega_no_losses(self):
        """Omega ratio with no losses."""
        returns = np.abs(np.random.randn(100) * 0.02)

        omega = compute_omega_ratio(returns, threshold=0.0)

        # Should return NaN (division by zero)
        assert np.isnan(omega)

    def test_tail_dep_length_mismatch(self):
        """Tail dependence with mismatched lengths."""
        returns_x = np.random.randn(100)
        returns_y = np.random.randn(50)

        with pytest.raises(ValueError):
            compute_tail_dependence(returns_x, returns_y)

    def test_tail_dep_multidimensional_error(self):
        """Tail dependence requires 1D input."""
        returns_x = np.random.randn(100, 2)
        returns_y = np.random.randn(100, 2)

        with pytest.raises(ValueError):
            compute_tail_dependence(returns_x, returns_y)

    def test_insufficient_periods(self):
        """All metrics handle insufficient periods."""
        returns = np.random.randn(10)

        tail_ratio = compute_tail_ratio(returns, min_periods=50)
        gl_ratio = compute_gain_loss_ratio(returns, min_periods=50)
        upr = compute_upside_potential_ratio(returns, min_periods=50)
        omega = compute_omega_ratio(returns, min_periods=50)

        # All should return NaN
        assert np.isnan(tail_ratio)
        assert np.isnan(gl_ratio)
        assert np.isnan(upr)
        assert np.isnan(omega)
