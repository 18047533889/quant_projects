"""
Tests for temporal metrics with golden reference values.
"""

import pytest
import numpy as np
from scipy import stats

from quant_evaluator.metrics.temporal import (
    compute_autocorrelation,
    compute_ic_autocorrelation,
    compute_rank_stability,
    compute_mean_rank_stability,
    compute_factor_turnover_rate,
    compute_half_life,
)


class TestAutocorrelation:
    """Test autocorrelation function."""

    def test_acf_white_noise(self):
        """White noise has zero autocorrelation."""
        np.random.seed(42)
        series = np.random.randn(200)

        acf = compute_autocorrelation(series, max_lag=10, min_obs=30)

        assert acf.shape == (11,)
        assert np.isclose(acf[0], 1.0, atol=1e-10)  # Lag 0
        # Higher lags should be close to 0
        assert np.all(np.abs(acf[1:]) < 0.2)

    def test_acf_ar1_process(self):
        """AR(1) process has exponential decay."""
        np.random.seed(100)
        phi = 0.7
        T = 500

        # Generate AR(1): x_t = phi * x_{t-1} + epsilon
        series = np.zeros(T)
        series[0] = np.random.randn()
        for t in range(1, T):
            series[t] = phi * series[t-1] + np.random.randn()

        acf = compute_autocorrelation(series, max_lag=5, min_obs=30)

        # ACF should be approximately phi^lag
        for lag in range(1, 6):
            expected = phi ** lag
            assert np.isclose(acf[lag], expected, atol=0.15)

    def test_acf_perfect_autocorrelation(self):
        """Repeated pattern has high autocorrelation."""
        pattern = np.array([1, 2, 3, 4, 5])
        series = np.tile(pattern, 40)  # 200 observations

        acf = compute_autocorrelation(series, max_lag=5, min_obs=30)

        # At lag=5 (pattern length), should have high correlation
        assert acf[5] > 0.9

    def test_acf_multiple_series(self):
        """Multiple series ACF."""
        np.random.seed(42)
        series = np.random.randn(150, 3)

        acf = compute_autocorrelation(series, max_lag=8, min_obs=30)

        assert acf.shape == (9, 3)
        assert np.allclose(acf[0, :], 1.0, atol=1e-10)

    def test_acf_insufficient_obs(self):
        """Insufficient observations returns NaN."""
        series = np.random.randn(20)

        acf = compute_autocorrelation(series, max_lag=5, min_obs=50)

        assert np.all(np.isnan(acf))

    def test_acf_singleton_time_and_factor_axes_are_preserved(self):
        """Two-dimensional inputs preserve singleton T and F axes."""
        series = np.array([[2.0]])

        acf = compute_autocorrelation(series, max_lag=3, min_obs=1)

        assert acf.shape == (4, 1)
        assert acf[0, 0] == 1.0
        assert np.all(np.isnan(acf[1:, 0]))

    def test_acf_singleton_factor_warmup_shape_and_nans(self):
        """Insufficient two-dimensional input keeps factor shape and NaNs."""
        series = np.array([[2.0]])

        acf = compute_autocorrelation(series, max_lag=3, min_obs=2)

        assert acf.shape == (4, 1)
        assert np.all(np.isnan(acf))

    def test_acf_one_dimensional_max_lag_zero_returns_vector(self):
        """A one-dimensional input never collapses to a scalar."""
        series = np.arange(5.0)

        acf = compute_autocorrelation(series, max_lag=0, min_obs=2)

        assert acf.shape == (1,)
        assert acf[0] == 1.0

    def test_acf_with_nans(self):
        """NaN values are filtered."""
        np.random.seed(42)
        series = np.random.randn(100)
        series[::10] = np.nan  # 10% NaN

        acf = compute_autocorrelation(series, max_lag=5, min_obs=30)

        # Should compute on valid observations
        assert np.isfinite(acf[0])


class TestICAutocorrelation:
    """Test IC autocorrelation."""

    def test_ic_acf_wrapper(self):
        """IC ACF is wrapper around compute_autocorrelation."""
        np.random.seed(42)
        ic_series = np.random.randn(100, 2)

        acf = compute_ic_autocorrelation(ic_series, max_lag=10, min_obs=30)

        # Should match direct call
        expected = compute_autocorrelation(ic_series, max_lag=10, min_obs=30)
        assert np.allclose(acf, expected, equal_nan=True)


class TestRankStability:
    """Test rank stability."""

    def test_rank_stability_perfect(self):
        """Perfect rank stability when ordering unchanged."""
        T, N, F = 50, 100, 1
        np.random.seed(42)

        # Constant ranks across time
        base_ranks = np.arange(N, dtype=float)
        factor_values = np.tile(base_ranks, (T, 1)).reshape(T, N, F)

        stability = compute_rank_stability(factor_values, lag=1, method="spearman")

        assert stability.shape == (T - 1, F)
        # Perfect rank correlation
        assert np.allclose(stability[:, 0], 1.0, atol=1e-10)

    def test_rank_stability_reversed(self):
        """Reversed ranks yield negative stability."""
        T, N, F = 10, 50, 1

        factor_values = np.zeros((T, N, F))
        for t in range(T):
            if t % 2 == 0:
                factor_values[t, :, 0] = np.arange(N)
            else:
                factor_values[t, :, 0] = np.arange(N)[::-1]

        stability = compute_rank_stability(factor_values, lag=1, method="spearman")

        # Alternating pattern -> negative correlation
        assert np.all(stability[:, 0] < -0.5)

    def test_rank_stability_random(self):
        """Random values yield low stability."""
        T, N, F = 30, 80, 1
        np.random.seed(100)

        factor_values = np.random.randn(T, N, F)

        stability = compute_rank_stability(factor_values, lag=1, method="spearman")

        # Random ranks -> low correlation
        mean_stability = np.nanmean(stability[:, 0])
        assert abs(mean_stability) < 0.2

    def test_rank_stability_lag_validation(self):
        """Invalid lag raises error."""
        factor_values = np.random.randn(10, 50, 1)

        with pytest.raises(ValueError, match="lag .* must be less than T"):
            compute_rank_stability(factor_values, lag=15, method="spearman")

    def test_rank_stability_pearson_method(self):
        """Pearson method for value stability."""
        T, N, F = 20, 60, 1
        np.random.seed(42)

        # Highly correlated values across time
        factor_values = np.random.randn(T, N, F) + np.random.randn(1, N, F) * 3

        stability = compute_rank_stability(factor_values, lag=1, method="pearson")

        # Should show positive stability
        assert np.nanmean(stability[:, 0]) > 0.3


class TestMeanRankStability:
    """Test time-averaged rank stability."""

    def test_mean_rank_stability_basic(self):
        """Basic mean rank stability."""
        T, N, F = 50, 100, 2
        np.random.seed(42)

        factor_values = np.random.randn(T, N, F)

        mean_stability = compute_mean_rank_stability(
            factor_values, lag=1, method="spearman", min_periods=20
        )

        assert mean_stability.shape == (F,)
        # With random data, should be near zero
        assert np.all(np.abs(mean_stability) < 0.3)

    def test_mean_rank_stability_insufficient_periods(self):
        """Insufficient periods returns NaN."""
        T, N, F = 15, 50, 1
        factor_values = np.random.randn(T, N, F)

        mean_stability = compute_mean_rank_stability(
            factor_values, lag=1, method="spearman", min_periods=20
        )

        assert np.isnan(mean_stability[0])


class TestFactorTurnoverRate:
    """Test factor turnover rate."""

    def test_turnover_zero_stable_top(self):
        """Zero turnover when top quantile unchanged."""
        T, N, F = 10, 100, 1

        # Top 10 assets always the same
        factor_values = np.zeros((T, N, F))
        for t in range(T):
            factor_values[t, :10, 0] = 10.0
            factor_values[t, 10:, 0] = np.random.randn(N - 10)

        turnover = compute_factor_turnover_rate(factor_values, quantile=0.9)

        assert turnover.shape == (T - 1, F)
        # No turnover in top 10%
        assert np.all(turnover[:, 0] < 0.15)

    def test_turnover_high_random(self):
        """High turnover with random values."""
        T, N, F = 20, 200, 1
        np.random.seed(42)

        factor_values = np.random.randn(T, N, F)

        turnover = compute_factor_turnover_rate(factor_values, quantile=0.9)

        # Random reshuffling -> expect ~18% turnover (only top 10% being tracked)
        mean_turnover = np.nanmean(turnover[:, 0])
        assert 0.1 < mean_turnover < 0.3

    def test_turnover_bottom_quantile(self):
        """Bottom quantile turnover."""
        T, N, F = 15, 150, 1
        np.random.seed(100)

        factor_values = np.random.randn(T, N, F)

        turnover = compute_factor_turnover_rate(factor_values, quantile=0.1)

        assert turnover.shape == (T - 1, F)
        # Random data should have turnover
        assert np.nanmean(turnover[:, 0]) > 0.1

    def test_turnover_quantile_validation(self):
        """Invalid quantile raises error."""
        factor_values = np.random.randn(10, 50, 1)

        with pytest.raises(ValueError, match="quantile must be in"):
            compute_factor_turnover_rate(factor_values, quantile=1.5)


class TestHalfLife:
    """Test IC half-life estimation."""

    def test_half_life_high_persistence(self):
        """High persistence yields long half-life."""
        np.random.seed(42)
        phi = 0.9  # High AR(1) coefficient
        T = 200

        # Generate AR(1) IC series
        ic_series = np.zeros((T, 1))
        ic_series[0, 0] = np.random.randn()
        for t in range(1, T):
            ic_series[t, 0] = phi * ic_series[t-1, 0] + np.random.randn() * 0.1

        half_life = compute_half_life(ic_series, min_periods=60)

        # Half-life = -log(2) / log(phi)
        expected_hl = -np.log(2) / np.log(phi)
        assert np.isclose(half_life[0], expected_hl, rtol=0.3)
        assert half_life[0] > 5.0  # Should be long

    def test_half_life_low_persistence(self):
        """Low persistence yields short half-life."""
        np.random.seed(100)
        phi = 0.3  # Low AR(1) coefficient
        T = 200

        ic_series = np.zeros((T, 1))
        ic_series[0, 0] = np.random.randn()
        for t in range(1, T):
            ic_series[t, 0] = phi * ic_series[t-1, 0] + np.random.randn() * 0.5

        half_life = compute_half_life(ic_series, min_periods=60)

        expected_hl = -np.log(2) / np.log(phi)
        assert np.isclose(half_life[0], expected_hl, rtol=0.4)
        assert half_life[0] < 3.0  # Should be short

    def test_half_life_white_noise(self):
        """White noise (phi=0) returns NaN."""
        np.random.seed(42)
        ic_series = np.random.randn(100, 1)

        half_life = compute_half_life(ic_series, min_periods=60)

        # phi ~0 -> invalid half-life
        assert np.isnan(half_life[0])

    def test_half_life_unit_root(self):
        """Unit root (phi~1) has very long or invalid half-life."""
        np.random.seed(42)
        T = 150
        # Random walk: phi=1
        ic_series = np.cumsum(np.random.randn(T, 1), axis=0)

        half_life = compute_half_life(ic_series, min_periods=60)

        # OLS on random walk may give phi < 1 (biased downward)
        # If phi is estimated as >= 1, we get NaN; if < 1, we get very long half-life
        assert np.isnan(half_life[0]) or half_life[0] > 50

    def test_half_life_insufficient_periods(self):
        """Insufficient periods returns NaN."""
        ic_series = np.random.randn(30, 1)

        half_life = compute_half_life(ic_series, min_periods=60)

        assert np.isnan(half_life[0])

    def test_half_life_multiple_factors(self):
        """Half-life for multiple factors."""
        np.random.seed(42)
        T = 200

        # Factor 1: high persistence, Factor 2: low persistence
        ic_f1 = np.zeros(T)
        ic_f2 = np.zeros(T)

        ic_f1[0] = np.random.randn()
        ic_f2[0] = np.random.randn()

        for t in range(1, T):
            ic_f1[t] = 0.85 * ic_f1[t-1] + np.random.randn() * 0.1
            ic_f2[t] = 0.2 * ic_f2[t-1] + np.random.randn() * 0.5

        ic_series = np.column_stack([ic_f1, ic_f2])

        half_life = compute_half_life(ic_series, min_periods=60)

        assert half_life.shape == (2,)
        # Factor 1 should have longer half-life
        assert half_life[0] > half_life[1]
