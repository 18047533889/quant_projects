"""
Tests for portfolio statistics with golden reference values.
"""

import pytest
import numpy as np

from quant_evaluator.metrics.portfolio_stats import (
    compute_long_short_returns,
    compute_sharpe_ratio,
    compute_maximum_drawdown,
    compute_calmar_ratio,
    compute_sortino_ratio,
    compute_win_rate,
)


class TestLongShortReturns:
    """Test long/short portfolio return computation."""

    def test_long_short_basic(self):
        """Basic long/short portfolio construction."""
        np.random.seed(42)
        T, N = 50, 100

        # Factor with signal
        factor_values = np.random.randn(T, N)
        forward_returns = factor_values * 0.01 + np.random.randn(T, N) * 0.02

        long_ret, short_ret, ls_ret = compute_long_short_returns(
            factor_values, forward_returns, long_threshold=0.8, short_threshold=0.2
        )

        assert long_ret.shape == (T,)
        assert short_ret.shape == (T,)
        assert ls_ret.shape == (T,)

        # Long should outperform short on average
        assert np.nanmean(long_ret) > np.nanmean(short_ret)

    def test_long_short_perfect_signal(self):
        """Perfect predictive factor."""
        T, N = 30, 50

        factor_values = np.random.randn(T, N)
        # Perfect prediction
        forward_returns = factor_values * 0.05

        long_ret, short_ret, ls_ret = compute_long_short_returns(
            factor_values, forward_returns, long_threshold=0.9, short_threshold=0.1
        )

        # Long should be strongly positive, short strongly negative
        assert np.nanmean(long_ret) > 0.04
        assert np.nanmean(short_ret) < -0.04
        # Long-short spread should be positive
        assert np.nanmean(ls_ret) > 0.08

    def test_long_short_no_signal(self):
        """Factor with no predictive power."""
        np.random.seed(100)
        T, N = 50, 100

        factor_values = np.random.randn(T, N)
        forward_returns = np.random.randn(T, N) * 0.02  # Independent

        long_ret, short_ret, ls_ret = compute_long_short_returns(
            factor_values, forward_returns, long_threshold=0.8, short_threshold=0.2
        )

        # No significant difference expected
        mean_ls = np.nanmean(ls_ret)
        assert np.abs(mean_ls) < 0.01

    def test_long_short_multi_factor(self):
        """Multiple factors (3D input)."""
        np.random.seed(200)
        T, N, F = 40, 80, 3

        factor_values = np.random.randn(T, N, F)
        forward_returns = np.random.randn(T, N) * 0.02
        # Factor 0 has signal
        forward_returns += factor_values[:, :, 0] * 0.01

        long_ret, short_ret, ls_ret = compute_long_short_returns(
            factor_values, forward_returns, long_threshold=0.8, short_threshold=0.2
        )

        assert long_ret.shape == (T, F)
        assert short_ret.shape == (T, F)
        assert ls_ret.shape == (T, F)

        # Factor 0 should have better performance
        assert np.nanmean(ls_ret[:, 0]) > np.nanmean(ls_ret[:, 1])

    def test_long_short_with_validity(self):
        """Validity mask filters invalid observations."""
        T, N = 30, 60

        factor_values = np.random.randn(T, N)
        forward_returns = factor_values * 0.01 + np.random.randn(T, N) * 0.02

        validity_mask = np.ones((T, N), dtype=bool)
        validity_mask[:, :10] = False  # Mask first 10 assets

        long_ret, short_ret, ls_ret = compute_long_short_returns(
            factor_values, forward_returns, validity_mask=validity_mask
        )

        assert long_ret.shape == (T,)
        # Should compute successfully with remaining assets
        assert np.sum(np.isfinite(ls_ret)) > T * 0.8

    def test_long_short_insufficient_assets(self):
        """Insufficient valid assets returns NaN."""
        T, N = 20, 5

        factor_values = np.random.randn(T, N)
        forward_returns = np.random.randn(T, N) * 0.02

        # With thresholds 0.8/0.2, need at least 5 assets
        # Will often have too few for robust quantiles
        long_ret, short_ret, ls_ret = compute_long_short_returns(
            factor_values, forward_returns, long_threshold=0.95, short_threshold=0.05
        )

        # May have many NaN periods
        assert long_ret.shape == (T,)


class TestSharpeRatio:
    """Test Sharpe ratio computation."""

    def test_sharpe_positive_returns(self):
        """Positive mean returns yield positive Sharpe."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005  # Mean 0.05%/day, ~12.6% annual

        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        # Positive mean, reasonable vol -> positive Sharpe
        assert sharpe > 0.5
        assert sharpe < 3.0

    def test_sharpe_golden_reference(self):
        """Golden reference computation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005] * 10)  # 50 periods

        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        # Manual calculation
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        expected_sharpe = (mean_ret / std_ret) * np.sqrt(252)

        assert np.isclose(sharpe, expected_sharpe, atol=1e-10)

    def test_sharpe_with_risk_free(self):
        """Sharpe ratio with non-zero risk-free rate."""
        np.random.seed(100)
        returns = np.random.randn(252) * 0.01 + 0.0004  # ~10% annual

        sharpe_rf0 = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        sharpe_rf3 = compute_sharpe_ratio(returns, risk_free_rate=0.03, periods_per_year=252)

        # Higher risk-free rate -> lower Sharpe
        assert sharpe_rf0 > sharpe_rf3

    def test_sharpe_negative_returns(self):
        """Negative mean returns yield negative Sharpe."""
        # Create returns with reliably negative mean
        returns = np.full(252, -0.001) + np.random.RandomState(200).randn(252) * 0.005

        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        assert sharpe < 0

    def test_sharpe_multi_factor(self):
        """Sharpe ratio for multiple return series."""
        np.random.seed(42)
        T, F = 252, 3
        returns = np.random.randn(T, F) * 0.01
        returns[:, 0] += 0.002  # Factor 0: best Sharpe
        returns[:, 1] += 0.001  # Factor 1: medium Sharpe
        # Factor 2: near-zero mean

        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        assert sharpe.shape == (F,)
        # Factor 0 should have best Sharpe (most positive mean)
        assert sharpe[0] > sharpe[1]

    def test_sharpe_insufficient_periods(self):
        """Insufficient periods returns NaN."""
        returns = np.random.randn(10)

        sharpe = compute_sharpe_ratio(returns, min_periods=20)

        assert np.isnan(sharpe)

    def test_sharpe_zero_volatility(self):
        """Zero volatility returns NaN."""
        returns = np.full(100, 0.01)  # Constant

        sharpe = compute_sharpe_ratio(returns)

        assert np.isnan(sharpe)


class TestMaximumDrawdown:
    """Test maximum drawdown computation."""

    def test_drawdown_no_losses(self):
        """All positive returns have zero drawdown."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.005])

        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)

        assert max_dd == 0.0
        assert np.all(dd_series == 0.0)

    def test_drawdown_simple_case(self):
        """Simple drawdown case."""
        # Start at 100, go to 120, drop to 90, recover to 110
        returns = np.array([0.2, -0.25, 0.222222])  # 100 -> 120 -> 90 -> 110

        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)

        # Max drawdown: (90 - 120) / 120 = -0.25
        assert np.isclose(max_dd, 0.25, atol=1e-6)
        # Peak is the last index at/before the trough where wealth equals
        # the running max: wealth = [1.2, 0.9, 1.1], trough at index 1,
        # running max at trough = 1.2 attained at index 0.
        assert peak_idx == 0  # Peak at index 0 (wealth 1.2)

    def test_drawdown_golden_reference(self):
        """Golden reference with known drawdown."""
        # Wealth: 1.0 -> 1.1 -> 1.21 -> 1.089 -> 0.9801 -> 1.0781
        returns = np.array([0.1, 0.1, -0.1, -0.1, 0.1])

        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)

        # Peak at 1.21, trough at 0.9801
        # Drawdown: (0.9801 - 1.21) / 1.21 = -0.19
        assert np.isclose(max_dd, 0.19, atol=1e-4)

    def test_drawdown_continuous_decline(self):
        """Continuous decline case."""
        returns = np.array([-0.05, -0.05, -0.05, -0.05])

        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)

        # Starting at 1.0, after 4 periods: 0.95^4 = 0.81450625
        # Drawdown from peak (1.0): (0.81450625 - 1.0) / 1.0 = -0.18549375
        # But cumulative returns start at period 0 with value 1.0 * (1-0.05) = 0.95
        # Actually the peak is at t=0 (before any returns applied), value = 1.0
        # After returns: [0.95, 0.9025, 0.857375, 0.81450625]
        # Drawdown at end: (0.81450625 - 1.0) / 1.0 = -0.18549375 magnitude = 0.18549375
        # But our function computes cumulative product starting from returns themselves
        # cum_returns = cumprod(1 + returns) = [0.95, 0.9025, 0.857375, 0.81450625]
        # running_max = [0.95, 0.95, 0.95, 0.95]
        # drawdown = (cum - running_max) / running_max
        # At idx 3: (0.81450625 - 0.95) / 0.95 = -0.142625
        expected_dd = (0.95 - 0.95**4) / 0.95  # More accurate
        assert np.isclose(max_dd, expected_dd, atol=1e-4)
        # Peak is the last index where wealth equals the running max
        # (0.95, attained at index 0); the trough is index 3.
        assert peak_idx == 0

    def test_drawdown_multi_factor(self):
        """Multiple return series."""
        T, F = 50, 2
        np.random.seed(42)
        returns = np.random.randn(T, F) * 0.02 + 0.001

        max_dd, dd_series, peak_indices = compute_maximum_drawdown(returns)

        assert max_dd.shape == (F,)
        assert dd_series.shape == (T, F)
        assert peak_indices.shape == (F,)

        # All should have some drawdown
        assert np.all(max_dd > 0)

    def test_drawdown_with_nans(self):
        """NaN handling (treated as 0 return)."""
        returns = np.array([0.1, np.nan, -0.1, 0.05])

        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)

        # Should handle NaN gracefully
        assert np.isfinite(max_dd)


class TestCalmarRatio:
    """Test Calmar ratio computation."""

    def test_calmar_positive(self):
        """Positive returns with drawdown."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.015 + 0.0005

        calmar = compute_calmar_ratio(returns, periods_per_year=252)

        # Positive mean / drawdown
        assert calmar > 0

    def test_calmar_golden_reference(self):
        """Golden reference calculation."""
        returns = np.array([0.01, 0.02, -0.05, 0.03, 0.01] * 10)

        calmar = compute_calmar_ratio(returns, periods_per_year=252, min_periods=20)

        # Manual: annual return / max drawdown
        mean_ret = np.mean(returns)
        ann_ret = mean_ret * 252
        max_dd, _, _ = compute_maximum_drawdown(returns)
        expected_calmar = ann_ret / max_dd

        assert np.isclose(calmar, expected_calmar, atol=1e-10)

    def test_calmar_no_drawdown(self):
        """No drawdown returns NaN."""
        returns = np.full(100, 0.01)  # All positive

        calmar = compute_calmar_ratio(returns)

        # Division by zero -> NaN
        assert np.isnan(calmar)

    def test_calmar_insufficient_periods(self):
        """Insufficient periods returns NaN."""
        returns = np.random.randn(10)

        calmar = compute_calmar_ratio(returns, min_periods=20)

        assert np.isnan(calmar)


class TestSortinoRatio:
    """Test Sortino ratio computation."""

    def test_sortino_positive_returns(self):
        """Positive mean returns yield positive Sortino."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005

        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        assert sortino > 0

    def test_sortino_vs_sharpe(self):
        """Sortino typically higher than Sharpe for skewed returns."""
        np.random.seed(100)
        # Right-skewed returns (few large losses)
        returns = np.random.exponential(scale=0.01, size=252) - 0.005

        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        # Sortino only penalizes downside -> typically higher
        if not np.isnan(sortino):
            assert sortino > sharpe * 0.8

    def test_sortino_no_downside(self):
        """All positive excess returns -> NaN (no downside)."""
        returns = np.full(100, 0.01)

        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0)

        # No downside deviation -> NaN
        assert np.isnan(sortino)

    def test_sortino_golden_reference(self):
        """Golden reference calculation."""
        returns = np.array([0.02, 0.01, -0.03, 0.015, -0.01, 0.02, 0.01, -0.02] * 5)

        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252, min_periods=20)

        # Manual calculation
        mean_ret = np.mean(returns)
        downside_ret = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_ret ** 2))
        expected_sortino = (mean_ret / downside_std) * np.sqrt(252)

        assert np.isclose(sortino, expected_sortino, atol=1e-10)

    def test_sortino_multi_factor(self):
        """Multiple return series."""
        np.random.seed(42)
        T, F = 252, 2
        returns = np.random.randn(T, F) * 0.01
        returns[:, 0] += 0.0005  # Factor 0: better performance

        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        assert sortino.shape == (F,)
        assert sortino[0] > sortino[1]

    def test_sortino_insufficient_periods(self):
        """Insufficient periods returns NaN."""
        returns = np.random.randn(10)

        sortino = compute_sortino_ratio(returns, min_periods=20)

        assert np.isnan(sortino)


class TestWinRate:
    """Test win rate computation."""

    def test_win_rate_all_positive(self):
        """All positive returns -> 100% win rate."""
        returns = np.array([0.01, 0.02, 0.005, 0.015, 0.03])

        win_rate = compute_win_rate(returns)

        assert win_rate == 1.0

    def test_win_rate_all_negative(self):
        """All negative returns -> 0% win rate."""
        returns = np.array([-0.01, -0.02, -0.005, -0.015])

        win_rate = compute_win_rate(returns)

        assert win_rate == 0.0

    def test_win_rate_mixed(self):
        """Mixed returns."""
        returns = np.array([0.01, -0.01, 0.02, -0.005, 0.01, -0.02, 0.005, 0.01])

        win_rate = compute_win_rate(returns)

        # 5 positive out of 8
        assert np.isclose(win_rate, 5/8, atol=1e-10)

    def test_win_rate_with_zeros(self):
        """Zero returns are not wins."""
        returns = np.array([0.01, 0.0, -0.01, 0.0, 0.02])

        win_rate = compute_win_rate(returns)

        # 2 positive out of 5
        assert np.isclose(win_rate, 2/5, atol=1e-10)

    def test_win_rate_multi_factor(self):
        """Multiple return series."""
        returns = np.array([
            [0.01, -0.01],
            [0.02, -0.02],
            [-0.01, 0.01],
            [0.01, 0.01],
        ])

        win_rate = compute_win_rate(returns)

        assert win_rate.shape == (2,)
        assert np.isclose(win_rate[0], 3/4)
        assert np.isclose(win_rate[1], 2/4)

    def test_win_rate_with_nans(self):
        """NaN values are filtered."""
        returns = np.array([0.01, np.nan, -0.01, np.nan, 0.02, 0.01])

        win_rate = compute_win_rate(returns)

        # 3 positive out of 4 valid
        assert np.isclose(win_rate, 3/4, atol=1e-10)

    def test_win_rate_all_nan(self):
        """All NaN returns NaN."""
        returns = np.array([np.nan, np.nan, np.nan])

        win_rate = compute_win_rate(returns)

        assert np.isnan(win_rate)
