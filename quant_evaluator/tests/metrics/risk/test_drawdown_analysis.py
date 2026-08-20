"""
Tests for drawdown analysis.
"""

import pytest
import numpy as np

from quant_evaluator.metrics.risk.drawdown_analysis import (
    compute_drawdown_series,
    compute_drawdown_statistics,
    identify_drawdown_periods,
    compute_drawdown_duration,
    compute_ulcer_index,
    compute_pain_index,
)


class TestDrawdownSeries:
    """Test drawdown series computation."""

    def test_drawdown_no_loss(self):
        """Drawdown with only positive returns."""
        returns = np.array([0.01, 0.02, 0.01, 0.015])

        dd_series, cum_ret, running_max = compute_drawdown_series(returns)

        # No drawdown
        assert np.all(dd_series >= -1e-10)
        # Cumulative returns increasing
        assert np.all(cum_ret[1:] >= cum_ret[:-1])
        # Running max matches cumulative
        assert np.allclose(running_max, cum_ret)

    def test_drawdown_simple_loss(self):
        """Simple drawdown and recovery."""
        returns = np.array([0.10, -0.05, -0.05, 0.10, 0.05])

        dd_series, cum_ret, running_max = compute_drawdown_series(returns)

        # Peak at index 0: 1.10
        # After two losses: 1.10 * 0.95 * 0.95 = 0.99275
        # Drawdown = (0.99275 - 1.10) / 1.10 ≈ -0.0975
        expected_dd_at_2 = (1.10 * 0.95 * 0.95 - 1.10) / 1.10
        assert abs(dd_series[2] - expected_dd_at_2) < 0.001

        # Should recover eventually
        assert dd_series[-1] > dd_series[2]

    def test_drawdown_multi_factor(self):
        """Drawdown for multiple factors."""
        np.random.seed(42)
        T, F = 100, 3

        returns = np.random.randn(T, F) * 0.02

        dd_series, cum_ret, running_max = compute_drawdown_series(returns)

        assert dd_series.shape == (T, F)
        assert cum_ret.shape == (T, F)
        assert running_max.shape == (T, F)

        # Drawdowns should be negative or zero
        assert np.all(dd_series <= 1e-10)

    def test_drawdown_with_nans(self):
        """Drawdown handles NaN returns (treats as 0)."""
        returns = np.array([0.10, -0.05, np.nan, 0.05, -0.03])

        dd_series, cum_ret, running_max = compute_drawdown_series(returns)

        # NaN treated as 0 return (no change)
        assert np.isfinite(dd_series[2])
        assert cum_ret[2] == cum_ret[1]  # No change at NaN


class TestDrawdownStatistics:
    """Test comprehensive drawdown statistics."""

    def test_drawdown_stats_basic(self):
        """Basic drawdown statistics."""
        # Create returns with known drawdown
        returns = np.array([0.10, -0.20, -0.10, 0.05, 0.10, 0.15])

        stats = compute_drawdown_statistics(returns, min_periods=5)

        # Should have max drawdown
        assert stats["max_drawdown"] > 0.25
        # Should identify index of max drawdown
        assert 1 <= stats["max_drawdown_idx"] <= 2
        # Some time underwater
        assert 0 < stats["time_underwater_pct"] < 100

    def test_drawdown_stats_no_drawdown(self):
        """Statistics with no drawdown."""
        returns = np.full(50, 0.01)

        stats = compute_drawdown_statistics(returns)

        # Zero drawdown
        assert stats["max_drawdown"] == 0.0
        assert stats["time_underwater_pct"] == 0.0

    def test_drawdown_stats_multi_factor(self):
        """Drawdown statistics for multiple factors."""
        np.random.seed(42)
        T, F = 200, 3

        returns = np.random.randn(T, F) * 0.02
        # Factor 1 has larger losses
        returns[50:60, 1] = -0.05

        stats = compute_drawdown_statistics(returns)

        assert stats["max_drawdown"].shape == (F,)
        # Factor 1 should have larger max drawdown
        assert stats["max_drawdown"][1] > stats["max_drawdown"][0]

    def test_drawdown_stats_all_keys(self):
        """Verify all expected keys present."""
        np.random.seed(100)
        returns = np.random.randn(100) * 0.02

        stats = compute_drawdown_statistics(returns)

        expected_keys = [
            "max_drawdown",
            "max_drawdown_idx",
            "avg_drawdown",
            "drawdown_volatility",
            "drawdown_99",
            "time_underwater_pct",
        ]

        for key in expected_keys:
            assert key in stats
            assert np.isfinite(stats[key])


class TestIdentifyDrawdownPeriods:
    """Test drawdown period identification."""

    def test_identify_single_period(self):
        """Identify single drawdown period."""
        returns = np.array([0.10, -0.15, -0.10, 0.10, 0.20])

        periods = identify_drawdown_periods(returns, threshold=0.05, min_duration=1)

        # Should identify one period
        assert len(periods) >= 1

        period = periods[0]
        assert "peak_idx" in period
        assert "trough_idx" in period
        assert "recovery_idx" in period
        assert "duration" in period
        assert "drawdown" in period

        # Drawdown should exceed threshold
        assert period["drawdown"] > 0.05

    def test_identify_multiple_periods(self):
        """Identify multiple drawdown periods."""
        returns = np.array([
            0.10, -0.10, -0.05, 0.15, 0.10,  # First period
            -0.15, -0.10, 0.20, 0.10           # Second period
        ])

        periods = identify_drawdown_periods(returns, threshold=0.05, min_duration=1)

        # Should identify at least one period
        assert len(periods) >= 1

        # Periods should be in chronological order
        for i in range(len(periods) - 1):
            assert periods[i]["peak_idx"] < periods[i + 1]["peak_idx"]

    def test_identify_ongoing_drawdown(self):
        """Identify ongoing drawdown (not recovered)."""
        returns = np.array([0.10, 0.05, -0.10, -0.15, -0.05])

        periods = identify_drawdown_periods(returns, threshold=0.05, min_duration=1)

        # Should have at least one period
        assert len(periods) >= 1

        # Last period should not be recovered
        last_period = periods[-1]
        assert last_period["recovery_idx"] == -1

    def test_identify_min_duration_filter(self):
        """Filter short drawdown periods."""
        returns = np.array([
            0.01, -0.08, 0.09,  # Short drawdown (2 periods)
            0.01, -0.06, -0.05, -0.04, 0.20  # Longer drawdown (4 periods)
        ])

        periods_all = identify_drawdown_periods(
            returns, threshold=0.05, min_duration=1
        )
        periods_filtered = identify_drawdown_periods(
            returns, threshold=0.05, min_duration=3
        )

        # Filtered should have fewer periods
        assert len(periods_filtered) <= len(periods_all)

    def test_identify_threshold_filter(self):
        """Filter small drawdowns by threshold."""
        returns = np.array([0.05, -0.03, 0.04, -0.12, -0.08, 0.20])

        periods_low = identify_drawdown_periods(returns, threshold=0.02)
        periods_high = identify_drawdown_periods(returns, threshold=0.10)

        # Higher threshold = fewer periods
        assert len(periods_high) <= len(periods_low)


class TestDrawdownDuration:
    """Test drawdown duration statistics."""

    def test_duration_basic(self):
        """Basic drawdown duration."""
        returns = np.array([0.10, -0.10, -0.05, 0.05, 0.10, 0.05])

        duration_stats = compute_drawdown_duration(returns, min_periods=5)

        assert "max_drawdown_duration" in duration_stats
        assert "avg_drawdown_duration" in duration_stats
        assert "current_drawdown_duration" in duration_stats

        # Should have some duration
        assert duration_stats["max_drawdown_duration"] > 0

    def test_duration_recovered(self):
        """Duration when portfolio has recovered."""
        returns = np.array([0.05, -0.10, 0.10, 0.05])

        duration_stats = compute_drawdown_duration(returns)

        # Currently not in drawdown
        assert duration_stats["current_drawdown_duration"] == 0.0

    def test_duration_ongoing(self):
        """Duration with ongoing drawdown."""
        returns = np.array([0.10, 0.05, -0.10, -0.05, -0.03])

        duration_stats = compute_drawdown_duration(returns, min_periods=5)

        # Currently in drawdown
        assert duration_stats["current_drawdown_duration"] > 0

    def test_duration_multi_factor(self):
        """Duration for multiple factors."""
        np.random.seed(42)
        T, F = 100, 3

        returns = np.random.randn(T, F) * 0.02

        duration_stats = compute_drawdown_duration(returns)

        assert isinstance(duration_stats["max_drawdown_duration"], np.ndarray)
        assert duration_stats["max_drawdown_duration"].shape == (F,)


class TestUlcerIndex:
    """Test Ulcer Index computation."""

    def test_ulcer_index_no_drawdown(self):
        """Ulcer Index with no drawdown."""
        returns = np.full(50, 0.01)

        ulcer = compute_ulcer_index(returns)

        # Should be zero or near-zero
        assert ulcer < 0.1

    def test_ulcer_index_with_drawdown(self):
        """Ulcer Index increases with drawdown."""
        returns_small = np.array([0.01] * 50)
        returns_large = np.array([0.10, -0.15, -0.10, 0.05] + [0.01] * 46)

        ulcer_small = compute_ulcer_index(returns_small)
        ulcer_large = compute_ulcer_index(returns_large)

        # Larger drawdown = higher Ulcer Index
        assert ulcer_large > ulcer_small

    def test_ulcer_index_sustained_drawdown(self):
        """Ulcer Index penalizes sustained drawdowns."""
        # Short sharp drawdown
        returns_sharp = np.array([0.10, -0.20, 0.15] + [0.01] * 47)
        # Prolonged mild drawdown
        returns_prolonged = np.array([0.10] + [-0.02] * 20 + [0.01] * 29)

        ulcer_sharp = compute_ulcer_index(returns_sharp)
        ulcer_prolonged = compute_ulcer_index(returns_prolonged)

        # Both should have non-zero Ulcer Index
        assert ulcer_sharp > 0
        assert ulcer_prolonged > 0

    def test_ulcer_index_multi_factor(self):
        """Ulcer Index for multiple factors."""
        np.random.seed(100)
        T, F = 200, 3

        returns = np.random.randn(T, F) * 0.02

        ulcer = compute_ulcer_index(returns)

        assert ulcer.shape == (F,)
        assert np.all(ulcer >= 0)


class TestPainIndex:
    """Test Pain Index computation."""

    def test_pain_index_no_drawdown(self):
        """Pain Index with no drawdown."""
        returns = np.full(50, 0.01)

        pain = compute_pain_index(returns)

        # Should be zero or near-zero
        assert pain < 0.1

    def test_pain_index_with_drawdown(self):
        """Pain Index increases with drawdown."""
        returns_small = np.array([0.01] * 50)
        returns_large = np.array([0.10, -0.15, -0.10, 0.05] + [0.01] * 46)

        pain_small = compute_pain_index(returns_small)
        pain_large = compute_pain_index(returns_large)

        # Larger drawdown = higher Pain Index
        assert pain_large > pain_small

    def test_pain_index_vs_ulcer(self):
        """Pain Index vs Ulcer Index relationship."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02

        pain = compute_pain_index(returns)
        ulcer = compute_ulcer_index(returns)

        # Both should be positive
        assert pain > 0
        assert ulcer > 0

        # Ulcer Index (RMS) typically higher than Pain Index (mean)
        # for distributions with outliers
        assert ulcer >= pain * 0.5  # Rough relationship

    def test_pain_index_multi_factor(self):
        """Pain Index for multiple factors."""
        np.random.seed(200)
        T, F = 200, 3

        returns = np.random.randn(T, F) * 0.02

        pain = compute_pain_index(returns)

        assert pain.shape == (F,)
        assert np.all(pain >= 0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self):
        """Statistics with insufficient data."""
        returns = np.random.randn(5)

        stats = compute_drawdown_statistics(returns, min_periods=10)

        # Should return NaN
        assert np.isnan(stats["max_drawdown"])

    def test_all_losses(self):
        """Drawdown with continuous losses."""
        returns = np.full(50, -0.01)

        dd_series, _, _ = compute_drawdown_series(returns)

        # Drawdown should keep increasing (becoming more negative)
        assert dd_series[-1] < dd_series[0]
        assert dd_series[-1] < -0.3  # Significant drawdown

    def test_extreme_volatility(self):
        """Drawdown with extreme volatility."""
        returns = np.array([0.50, -0.40, 0.30, -0.35, 0.40])

        stats = compute_drawdown_statistics(returns, min_periods=5)

        # Should handle extreme values
        assert np.isfinite(stats["max_drawdown"])
        assert stats["max_drawdown"] > 0.3

    def test_single_period(self):
        """Drawdown with single period."""
        returns = np.array([0.10])

        dd_series, _, _ = compute_drawdown_series(returns)

        # No drawdown from peak
        assert dd_series[0] == 0.0

    def test_identify_multidimensional_error(self):
        """identify_drawdown_periods requires 1D input."""
        returns = np.random.randn(50, 3)

        with pytest.raises(ValueError):
            identify_drawdown_periods(returns)

    def test_total_wipeout_nan_from_wipeout_onward(self):
        """Return of -1.0 -> drawdown NaN from that index onward, finite before."""
        returns = np.array([0.10, -0.05, -1.0, 0.50, 0.20])

        dd_series, cum_ret, running_max = compute_drawdown_series(returns)

        # Before the wipeout: finite, ordinary drawdowns
        assert np.all(np.isfinite(dd_series[:2]))
        expected_dd1 = (1.10 * 0.95 - 1.10) / 1.10
        assert abs(dd_series[1] - expected_dd1) < 1e-10

        # From the wipeout index onward: wealth <= 0 -> drawdown unmeasurable (NaN)
        assert np.all(np.isnan(dd_series[2:]))

    def test_negative_wealth_flip_does_not_resurrect_data(self):
        """Wealth < 0 flipped positive again stays NaN (fail-closed)."""
        # -1.5 gives factor -0.5: wealth 1.10 -> -0.55 -> +0.55 (two negative factors)
        returns = np.array([0.10, -1.5, -1.5, 0.10])

        dd_series, cum_ret, _ = compute_drawdown_series(returns)

        # cum_returns genuinely flips positive again, but drawdown must stay NaN
        assert cum_ret[2] > 0
        assert np.all(np.isfinite(dd_series[:1]))
        assert np.all(np.isnan(dd_series[1:]))

    def test_normal_series_unchanged(self):
        """Normal series: behavior identical to the plain ratio formula."""
        returns = np.array([0.10, -0.05, -0.05, 0.10, 0.05])

        dd_series, cum_ret, running_max = compute_drawdown_series(returns)

        # Reference: the historical formula (cum - max) / max on the same inputs
        expected = (cum_ret - running_max) / running_max
        assert np.allclose(dd_series, expected)
        assert np.all(np.isfinite(dd_series))
        # Known small case: index 2 drawdown pinned
        expected_dd_at_2 = (1.10 * 0.95 * 0.95 - 1.10) / 1.10
        assert abs(dd_series[2] - expected_dd_at_2) < 0.001

    def test_wipeout_mask_per_column_2d(self):
        """2D (T, F) input: wipeout masks only the affected column."""
        returns = np.array([
            [0.10, 0.05, -0.02],
            [-1.0, 0.05, -0.03],
            [0.50, -0.05, 0.10],
            [0.10, 0.05, 0.10],
        ])

        dd_series, cum_ret, running_max = compute_drawdown_series(returns)

        assert dd_series.shape == (4, 3)

        # Column 0: wiped out at index 1 -> NaN from index 1 onward
        assert np.isfinite(dd_series[0, 0])
        assert np.all(np.isnan(dd_series[1:, 0]))

        # Columns 1 and 2: never wiped out -> all finite, matching the formula
        for f in (1, 2):
            assert np.all(np.isfinite(dd_series[:, f]))
            expected = (cum_ret[:, f] - running_max[:, f]) / running_max[:, f]
            assert np.allclose(dd_series[:, f], expected)
            assert np.all(dd_series[:, f] <= 1e-10)
