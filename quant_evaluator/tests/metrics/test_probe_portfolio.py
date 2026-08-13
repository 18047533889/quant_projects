"""
Tests for probe portfolio construction and backtesting metrics.
"""

import pytest
import numpy as np

from quant_evaluator.metrics.probe_portfolio import (
    construct_long_short_portfolio,
    compute_equal_weighted_returns,
    compute_cap_weighted_returns,
    compute_long_short_equal_weighted,
    compute_long_short_cap_weighted,
    compute_portfolio_weights,
    compute_portfolio_concentration,
    compute_turnover_from_positions,
)


class TestConstructLongShortPortfolio:
    """Test portfolio position construction."""

    def test_basic_construction(self):
        """Basic long/short position construction."""
        np.random.seed(42)
        T, N = 30, 100

        factor_values = np.random.randn(T, N)

        long_pos, short_pos = construct_long_short_portfolio(
            factor_values, long_threshold=0.8, short_threshold=0.2
        )

        assert long_pos.shape == (T, N)
        assert short_pos.shape == (T, N)
        assert long_pos.dtype == bool
        assert short_pos.dtype == bool

        # Should have roughly 20% long and 20% short per period
        for t in range(T):
            n_long = np.sum(long_pos[t, :])
            n_short = np.sum(short_pos[t, :])
            assert 15 <= n_long <= 25  # Roughly 20%
            assert 15 <= n_short <= 25

    def test_no_overlap(self):
        """Long and short positions should not overlap."""
        np.random.seed(100)
        T, N = 20, 50

        factor_values = np.random.randn(T, N)

        long_pos, short_pos = construct_long_short_portfolio(
            factor_values, long_threshold=0.7, short_threshold=0.3
        )

        # No asset should be both long and short
        overlap = long_pos & short_pos
        assert np.sum(overlap) == 0

    def test_extreme_thresholds(self):
        """Extreme thresholds (top 5%, bottom 5%)."""
        np.random.seed(200)
        T, N = 25, 200

        factor_values = np.random.randn(T, N)

        long_pos, short_pos = construct_long_short_portfolio(
            factor_values, long_threshold=0.95, short_threshold=0.05
        )

        # Should have roughly 5% in each
        for t in range(T):
            n_long = np.sum(long_pos[t, :])
            n_short = np.sum(short_pos[t, :])
            assert 5 <= n_long <= 15  # Roughly 10 (5%)
            assert 5 <= n_short <= 15

    def test_with_validity_mask(self):
        """Validity mask filters invalid values."""
        T, N = 20, 60

        factor_values = np.random.randn(T, N)
        validity_mask = np.ones((T, N), dtype=bool)
        validity_mask[:, :10] = False  # Mask first 10 assets

        long_pos, short_pos = construct_long_short_portfolio(
            factor_values, validity_mask=validity_mask
        )

        # First 10 assets should never be selected
        assert np.sum(long_pos[:, :10]) == 0
        assert np.sum(short_pos[:, :10]) == 0

        # Remaining assets should have positions
        assert np.sum(long_pos[:, 10:]) > 0
        assert np.sum(short_pos[:, 10:]) > 0

    def test_multi_factor_3d(self):
        """3D factor values (multiple factors)."""
        np.random.seed(42)
        T, N, F = 30, 80, 3

        factor_values = np.random.randn(T, N, F)

        long_pos, short_pos = construct_long_short_portfolio(
            factor_values, long_threshold=0.8, short_threshold=0.2
        )

        assert long_pos.shape == (T, N, F)
        assert short_pos.shape == (T, N, F)

        # Each factor should have independent positions
        for f in range(F):
            assert np.sum(long_pos[:, :, f]) > 0
            assert np.sum(short_pos[:, :, f]) > 0

    def test_insufficient_assets(self):
        """Too few assets returns empty positions."""
        T, N = 10, 1

        factor_values = np.random.randn(T, N)

        long_pos, short_pos = construct_long_short_portfolio(
            factor_values, long_threshold=0.8, short_threshold=0.2
        )

        # Cannot form portfolio with only 1 asset
        assert np.sum(long_pos) == 0
        assert np.sum(short_pos) == 0


class TestEqualWeightedReturns:
    """Test equal-weighted portfolio returns."""

    def test_basic_equal_weighted(self):
        """Basic equal-weighted return computation."""
        T, N = 30, 50

        # All long positions
        positions = np.ones((T, N), dtype=bool)
        forward_returns = np.random.randn(T, N) * 0.01 + 0.001

        portfolio_returns = compute_equal_weighted_returns(positions, forward_returns)

        assert portfolio_returns.shape == (T,)
        assert np.all(np.isfinite(portfolio_returns))

        # Should equal mean returns across assets
        for t in range(T):
            expected = np.mean(forward_returns[t, :])
            assert np.isclose(portfolio_returns[t], expected, atol=1e-10)

    def test_partial_positions(self):
        """Partial position mask."""
        T, N = 20, 40

        positions = np.zeros((T, N), dtype=bool)
        positions[:, :10] = True  # Only first 10 assets

        forward_returns = np.random.randn(T, N) * 0.01

        portfolio_returns = compute_equal_weighted_returns(positions, forward_returns)

        # Should only average first 10 assets
        for t in range(T):
            expected = np.mean(forward_returns[t, :10])
            assert np.isclose(portfolio_returns[t], expected, atol=1e-10)

    def test_no_positions(self):
        """No positions returns NaN."""
        T, N = 15, 30

        positions = np.zeros((T, N), dtype=bool)
        forward_returns = np.random.randn(T, N) * 0.01

        portfolio_returns = compute_equal_weighted_returns(positions, forward_returns)

        # All NaN
        assert np.all(np.isnan(portfolio_returns))

    def test_multi_factor_equal_weighted(self):
        """3D positions (multiple factors)."""
        T, N, F = 25, 60, 2

        positions = np.random.rand(T, N, F) > 0.7  # Sparse positions
        forward_returns = np.random.randn(T, N) * 0.01

        portfolio_returns = compute_equal_weighted_returns(positions, forward_returns)

        assert portfolio_returns.shape == (T, F)

    def test_with_nans_in_returns(self):
        """NaN returns are filtered."""
        T, N = 20, 40

        positions = np.ones((T, N), dtype=bool)
        forward_returns = np.random.randn(T, N) * 0.01
        forward_returns[:, :5] = np.nan  # First 5 assets have NaN

        portfolio_returns = compute_equal_weighted_returns(positions, forward_returns)

        # Should compute from remaining 35 assets
        for t in range(T):
            expected = np.mean(forward_returns[t, 5:])
            assert np.isclose(portfolio_returns[t], expected, atol=1e-10)


class TestCapWeightedReturns:
    """Test cap-weighted portfolio returns."""

    def test_basic_cap_weighted(self):
        """Basic cap-weighted return computation."""
        np.random.seed(42)
        T, N = 30, 50

        positions = np.ones((T, N), dtype=bool)
        forward_returns = np.random.randn(T, N) * 0.01
        market_caps = np.random.uniform(1e9, 1e11, (T, N))

        portfolio_returns = compute_cap_weighted_returns(
            positions, forward_returns, market_caps, normalize=True
        )

        assert portfolio_returns.shape == (T,)
        assert np.all(np.isfinite(portfolio_returns))

        # Verify weighted average
        for t in range(T):
            weights = market_caps[t, :] / np.sum(market_caps[t, :])
            expected = np.sum(forward_returns[t, :] * weights)
            assert np.isclose(portfolio_returns[t], expected, atol=1e-10)

    def test_cap_weighted_vs_equal(self):
        """Cap-weighted differs from equal-weighted."""
        np.random.seed(100)
        T, N = 20, 40

        positions = np.ones((T, N), dtype=bool)
        forward_returns = np.random.randn(T, N) * 0.01

        # Large-cap skewed
        market_caps = np.random.exponential(scale=1e10, size=(T, N))

        eq_returns = compute_equal_weighted_returns(positions, forward_returns)
        cap_returns = compute_cap_weighted_returns(
            positions, forward_returns, market_caps, normalize=True
        )

        # Should differ in most periods
        differences = np.abs(eq_returns - cap_returns)
        assert np.mean(differences) > 1e-6

    def test_cap_weighted_single_large_cap(self):
        """Single large cap dominates."""
        T, N = 15, 30

        positions = np.ones((T, N), dtype=bool)
        forward_returns = np.ones((T, N)) * 0.01
        forward_returns[:, 0] = 0.05  # First asset has higher return

        market_caps = np.ones((T, N)) * 1e9
        market_caps[:, 0] = 1e12  # First asset 1000x larger

        portfolio_returns = compute_cap_weighted_returns(
            positions, forward_returns, market_caps, normalize=True
        )

        # Should be dominated by first asset's return
        for t in range(T):
            # First asset weight ~ 1000 / 1029 ≈ 0.972
            assert portfolio_returns[t] > 0.045  # Close to 0.05

    def test_cap_weighted_no_normalize(self):
        """Cap-weighted without normalization."""
        T, N = 10, 20

        positions = np.ones((T, N), dtype=bool)
        forward_returns = np.ones((T, N)) * 0.01
        market_caps = np.ones((T, N)) * 1e9

        portfolio_returns = compute_cap_weighted_returns(
            positions, forward_returns, market_caps, normalize=False
        )

        # Should scale with total cap
        for t in range(T):
            expected = 0.01 * N * 1e9
            assert np.isclose(portfolio_returns[t], expected, rtol=1e-6)

    def test_cap_weighted_partial_positions(self):
        """Partial position mask with caps."""
        T, N = 20, 40

        positions = np.zeros((T, N), dtype=bool)
        positions[:, :10] = True  # Only first 10 assets

        forward_returns = np.random.randn(T, N) * 0.01
        market_caps = np.random.uniform(1e9, 1e11, (T, N))

        portfolio_returns = compute_cap_weighted_returns(
            positions, forward_returns, market_caps, normalize=True
        )

        # Should only consider first 10 assets
        for t in range(T):
            weights = market_caps[t, :10] / np.sum(market_caps[t, :10])
            expected = np.sum(forward_returns[t, :10] * weights)
            assert np.isclose(portfolio_returns[t], expected, atol=1e-10)

    def test_cap_weighted_multi_factor(self):
        """3D positions (multiple factors)."""
        T, N, F = 25, 50, 2

        positions = np.random.rand(T, N, F) > 0.6
        forward_returns = np.random.randn(T, N) * 0.01
        market_caps = np.random.uniform(1e9, 1e11, (T, N))

        portfolio_returns = compute_cap_weighted_returns(
            positions, forward_returns, market_caps, normalize=True
        )

        assert portfolio_returns.shape == (T, F)

    def test_cap_weighted_zero_caps(self):
        """Zero market caps are filtered."""
        T, N = 15, 30

        positions = np.ones((T, N), dtype=bool)
        forward_returns = np.random.randn(T, N) * 0.01
        market_caps = np.random.uniform(1e9, 1e11, (T, N))
        market_caps[:, :5] = 0  # First 5 have zero cap

        portfolio_returns = compute_cap_weighted_returns(
            positions, forward_returns, market_caps, normalize=True
        )

        # Should only use assets with positive cap
        assert np.all(np.isfinite(portfolio_returns))


class TestLongShortEqualWeighted:
    """Test integrated long/short equal-weighted construction."""

    def test_basic_long_short_equal(self):
        """Basic long/short equal-weighted."""
        np.random.seed(42)
        T, N = 40, 100

        factor_values = np.random.randn(T, N)
        # Add signal
        forward_returns = factor_values * 0.01 + np.random.randn(T, N) * 0.02

        long_ret, short_ret, ls_ret = compute_long_short_equal_weighted(
            factor_values, forward_returns, long_threshold=0.8, short_threshold=0.2
        )

        assert long_ret.shape == (T,)
        assert short_ret.shape == (T,)
        assert ls_ret.shape == (T,)

        # Long should outperform short
        assert np.nanmean(long_ret) > np.nanmean(short_ret)

    def test_long_short_equal_perfect_signal(self):
        """Perfect predictive signal."""
        T, N = 30, 80

        factor_values = np.random.randn(T, N)
        forward_returns = factor_values * 0.05  # Perfect prediction

        long_ret, short_ret, ls_ret = compute_long_short_equal_weighted(
            factor_values, forward_returns, long_threshold=0.9, short_threshold=0.1
        )

        # Strong positive spread
        assert np.nanmean(ls_ret) > 0.08

    def test_long_short_equal_multi_factor(self):
        """Multiple factors."""
        np.random.seed(200)
        T, N, F = 35, 90, 3

        factor_values = np.random.randn(T, N, F)
        forward_returns = np.random.randn(T, N) * 0.02
        # Factor 0 has signal
        forward_returns += factor_values[:, :, 0] * 0.01

        long_ret, short_ret, ls_ret = compute_long_short_equal_weighted(
            factor_values, forward_returns
        )

        assert long_ret.shape == (T, F)
        assert short_ret.shape == (T, F)
        assert ls_ret.shape == (T, F)

        # Factor 0 should have best performance
        assert np.nanmean(ls_ret[:, 0]) > np.nanmean(ls_ret[:, 1])


class TestLongShortCapWeighted:
    """Test integrated long/short cap-weighted construction."""

    def test_basic_long_short_cap(self):
        """Basic long/short cap-weighted."""
        np.random.seed(42)
        T, N = 40, 100

        factor_values = np.random.randn(T, N)
        forward_returns = factor_values * 0.01 + np.random.randn(T, N) * 0.02
        market_caps = np.random.uniform(1e9, 1e11, (T, N))

        long_ret, short_ret, ls_ret = compute_long_short_cap_weighted(
            factor_values, forward_returns, market_caps,
            long_threshold=0.8, short_threshold=0.2
        )

        assert long_ret.shape == (T,)
        assert short_ret.shape == (T,)
        assert ls_ret.shape == (T,)

        # Long should outperform short
        assert np.nanmean(long_ret) > np.nanmean(short_ret)

    def test_long_short_cap_vs_equal(self):
        """Cap-weighted differs from equal-weighted."""
        np.random.seed(100)
        T, N = 30, 80

        factor_values = np.random.randn(T, N)
        forward_returns = factor_values * 0.01 + np.random.randn(T, N) * 0.02
        market_caps = np.random.exponential(scale=1e10, size=(T, N))

        _, _, ls_equal = compute_long_short_equal_weighted(
            factor_values, forward_returns
        )

        _, _, ls_cap = compute_long_short_cap_weighted(
            factor_values, forward_returns, market_caps
        )

        # Should differ
        differences = np.abs(ls_equal - ls_cap)
        assert np.nanmean(differences) > 1e-5

    def test_long_short_cap_multi_factor(self):
        """Multiple factors."""
        np.random.seed(300)
        T, N, F = 35, 90, 2

        factor_values = np.random.randn(T, N, F)
        forward_returns = np.random.randn(T, N) * 0.02
        market_caps = np.random.uniform(1e9, 1e11, (T, N))

        long_ret, short_ret, ls_ret = compute_long_short_cap_weighted(
            factor_values, forward_returns, market_caps
        )

        assert long_ret.shape == (T, F)
        assert short_ret.shape == (T, F)
        assert ls_ret.shape == (T, F)


class TestPortfolioWeights:
    """Test portfolio weight computation."""

    def test_equal_weights(self):
        """Equal weights without market caps."""
        T, N = 20, 40

        positions = np.zeros((T, N), dtype=bool)
        positions[:, :10] = True  # 10 positions

        weights = compute_portfolio_weights(positions, normalize=True)

        assert weights.shape == (T, N)

        # Each selected asset should have weight 1/10 = 0.1
        for t in range(T):
            selected_weights = weights[t, :10]
            assert np.allclose(selected_weights, 0.1, atol=1e-10)
            # Sum to 1
            assert np.isclose(np.sum(weights[t, :]), 1.0, atol=1e-10)

    def test_cap_weights(self):
        """Cap-weighted with market caps."""
        T, N = 15, 30

        positions = np.ones((T, N), dtype=bool)
        market_caps = np.ones((T, N)) * 1e9
        market_caps[:, 0] = 9e9  # First asset 9x larger

        weights = compute_portfolio_weights(positions, market_caps, normalize=True)

        # First asset weight: 9e9 / (9e9 + 29*1e9) = 9/38 ≈ 0.2368
        expected_first = 9.0 / 38.0
        for t in range(T):
            assert np.isclose(weights[t, 0], expected_first, atol=1e-10)
            # Sum to 1
            assert np.isclose(np.sum(weights[t, :]), 1.0, atol=1e-10)

    def test_weights_multi_factor(self):
        """3D positions."""
        T, N, F = 20, 40, 2

        positions = np.random.rand(T, N, F) > 0.7

        weights = compute_portfolio_weights(positions, normalize=True)

        assert weights.shape == (T, N, F)

        # Each factor's weights should sum to 1 (where positions exist)
        for t in range(T):
            for f in range(F):
                if np.sum(positions[t, :, f]) > 0:
                    assert np.isclose(np.sum(weights[t, :, f]), 1.0, atol=1e-10)


class TestPortfolioConcentration:
    """Test portfolio concentration (HHI)."""

    def test_single_asset_concentration(self):
        """Single asset has HHI = 1."""
        T, N = 15, 30

        positions = np.zeros((T, N), dtype=bool)
        positions[:, 0] = True  # Only one asset

        hhi = compute_portfolio_concentration(positions)

        # HHI = 1 for single asset
        assert np.allclose(hhi, 1.0, atol=1e-10)

    def test_equal_weighted_concentration(self):
        """Equal-weighted portfolio HHI."""
        T, N = 20, 100

        positions = np.zeros((T, N), dtype=bool)
        positions[:, :10] = True  # 10 equal assets

        hhi = compute_portfolio_concentration(positions)

        # HHI = 10 * (1/10)^2 = 0.1
        assert np.allclose(hhi, 0.1, atol=1e-10)

    def test_concentration_with_caps(self):
        """Cap-weighted concentration."""
        T, N = 15, 30

        positions = np.ones((T, N), dtype=bool)
        market_caps = np.ones((T, N)) * 1e9
        market_caps[:, 0] = 9e9  # First asset dominates

        hhi = compute_portfolio_concentration(positions, market_caps)

        # First asset weight: 9e9 / (9e9 + 29*1e9) = 9/38
        # Other assets: 1e9 / 38e9 = 1/38 each
        # HHI = (9/38)^2 + 29 * (1/38)^2 = 81/1444 + 29/1444 = 110/1444 ≈ 0.07617
        w_first = 9.0 / 38.0
        w_other = 1.0 / 38.0
        expected_hhi = w_first**2 + 29 * w_other**2
        assert np.allclose(hhi, expected_hhi, atol=1e-6)

    def test_no_positions_concentration(self):
        """No positions returns NaN."""
        T, N = 10, 20

        positions = np.zeros((T, N), dtype=bool)

        hhi = compute_portfolio_concentration(positions)

        assert np.all(np.isnan(hhi))


class TestTurnoverFromPositions:
    """Test turnover computation from positions."""

    def test_no_turnover(self):
        """Static positions have zero turnover."""
        T, N = 20, 40

        positions = np.zeros((T, N), dtype=bool)
        positions[:, :10] = True  # Static

        turnover = compute_turnover_from_positions(positions)

        # All zeros
        assert np.allclose(turnover, 0.0, atol=1e-10)

    def test_full_turnover(self):
        """Complete position change."""
        T, N = 10, 20

        positions = np.zeros((T, N), dtype=bool)
        # Alternate between first and second half
        for t in range(T):
            if t % 2 == 0:
                positions[t, :10] = True
            else:
                positions[t, 10:] = True

        turnover = compute_turnover_from_positions(positions, normalize=True)

        # Full turnover = 2.0 (sell all, buy all)
        assert np.allclose(turnover, 2.0, atol=1e-10)

    def test_partial_turnover(self):
        """Partial position change."""
        T, N = 15, 30

        positions = np.zeros((T, N), dtype=bool)
        positions[0, :10] = True
        positions[1, 5:15] = True  # 5 overlap, 5 new, 5 removed

        turnover = compute_turnover_from_positions(positions, normalize=True)

        # Period 0->1: 5 removed (0.5 -> 0), 5 new (0 -> 0.5)
        # Turnover = |0-0.5|*5 + |0.5-0.5|*5 + |0.5-0|*5 = 0.5*5 + 0 + 0.5*5 = 5.0
        # But with normalization, first period has 10 assets (weight 0.1 each)
        # Second period has 10 assets (weight 0.1 each)
        # Weight change per asset leaving: 0.1 -> 0, per asset entering: 0 -> 0.1
        # Turnover = 5 * 0.1 + 5 * 0.1 = 1.0
        assert np.isclose(turnover[0], 1.0, atol=1e-6)

    def test_turnover_multi_factor(self):
        """Multiple factors."""
        T, N, F = 20, 40, 2

        positions = np.random.rand(T, N, F) > 0.7

        turnover = compute_turnover_from_positions(positions, normalize=True)

        assert turnover.shape == (T-1, F)
