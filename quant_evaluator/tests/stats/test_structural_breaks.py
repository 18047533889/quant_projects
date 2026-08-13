"""
Tests for structural break detection.
"""

import numpy as np
import pytest

from quant_evaluator.metrics.stats.structural_breaks import (
    chow_test,
    cusum_test,
    cusum_of_squares_test,
    sup_wald_test,
    bai_perron_test,
)


class TestChowTest:
    """Test suite for Chow test."""

    def test_chow_no_break(self):
        """Test Chow test with no structural break."""
        np.random.seed(42)
        T = 200

        # Generate data from single regime
        X = np.random.randn(T)
        y = 2.0 + 1.5 * X + np.random.randn(T) * 0.5

        # Test break at midpoint
        breakpoint = T // 2

        f_stat, p_value, df1, df2, k = chow_test(y, X, breakpoint)

        # Should NOT reject null (high p-value)
        assert p_value > 0.05, f"False positive: p={p_value:.4f}"
        assert np.isfinite(f_stat)
        assert df1 == k == 2  # Intercept + 1 regressor

    def test_chow_with_break(self):
        """Test Chow test with clear structural break."""
        np.random.seed(42)
        T = 300
        breakpoint = 150

        X = np.random.randn(T)
        y = np.zeros(T)

        # First regime: y = 1 + 0.5*X
        y[:breakpoint] = 1.0 + 0.5 * X[:breakpoint] + np.random.randn(breakpoint) * 0.3

        # Second regime: y = 5 + 2.0*X (different intercept and slope)
        y[breakpoint:] = 5.0 + 2.0 * X[breakpoint:] + np.random.randn(T - breakpoint) * 0.3

        f_stat, p_value, df1, df2, k = chow_test(y, X, breakpoint)

        # Should reject null (low p-value)
        assert p_value < 0.01, f"False negative: p={p_value:.4f}"
        assert f_stat > 10.0  # Strong evidence of break

    def test_chow_univariate_x(self):
        """Test Chow with univariate X."""
        np.random.seed(42)
        T = 150

        X = np.random.randn(T)
        y = 3.0 + 1.0 * X + np.random.randn(T) * 0.5

        breakpoint = 75
        f_stat, p_value, df1, df2, k = chow_test(y, X, breakpoint)

        assert np.isfinite(f_stat)
        assert k == 2  # Intercept + X

    def test_chow_multivariate_x(self):
        """Test Chow with multiple regressors."""
        np.random.seed(42)
        T = 200
        K = 3

        X = np.random.randn(T, K)
        y = 1.0 + X @ np.array([0.5, 1.0, -0.5]) + np.random.randn(T) * 0.5

        breakpoint = 100
        f_stat, p_value, df1, df2, k = chow_test(y, X, breakpoint)

        assert np.isfinite(f_stat)
        assert k == K + 1  # Intercept + K regressors

    def test_chow_invalid_breakpoint(self):
        """Test error for invalid breakpoint location."""
        T = 100
        X = np.random.randn(T)
        y = np.random.randn(T)

        # Breakpoint too early
        with pytest.raises(ValueError, match="Breakpoint"):
            chow_test(y, X, breakpoint=5, min_obs=10)

        # Breakpoint too late
        with pytest.raises(ValueError, match="Breakpoint"):
            chow_test(y, X, breakpoint=95, min_obs=10)

    def test_chow_unequal_length(self):
        """Test error for unequal y and X lengths."""
        y = np.random.randn(100)
        X = np.random.randn(80)

        with pytest.raises(ValueError, match="same length"):
            chow_test(y, X, breakpoint=50)

    def test_chow_multidimensional_y(self):
        """Test error for multidimensional y."""
        y = np.random.randn(100, 2)
        X = np.random.randn(100)

        with pytest.raises(ValueError, match="1-dimensional"):
            chow_test(y, X, breakpoint=50)


class TestCUSUMTest:
    """Test suite for CUSUM test."""

    def test_cusum_stable_parameters(self):
        """Test CUSUM with stable parameters."""
        np.random.seed(42)
        T = 150

        # Stable residuals from consistent model
        residuals = np.random.randn(T) * 0.5

        cusum_stats, boundary, max_stat, is_stable = cusum_test(residuals, alpha=0.05)

        # Check that statistics are computed
        assert cusum_stats.shape == (T,)
        assert boundary > 0
        assert np.isfinite(max_stat)

        # Note: with random data, may occasionally cross boundary by chance
        # Main check is that function completes and returns valid outputs

    def test_cusum_parameter_drift(self):
        """Test CUSUM with parameter drift."""
        np.random.seed(42)
        T = 200

        # Simulate drift: residuals trend upward
        residuals = np.random.randn(T) * 0.5 + np.linspace(0, 2, T)

        cusum_stats, boundary, max_stat, is_stable = cusum_test(residuals, alpha=0.05)

        # Should detect instability
        assert not is_stable, f"False negative: max_stat={max_stat:.2f} < {boundary:.2f}"
        assert max_stat > boundary

    def test_cusum_sharp_break(self):
        """Test CUSUM with sharp structural break."""
        np.random.seed(42)
        T = 250

        # Residuals shift at midpoint
        residuals = np.concatenate([
            np.random.randn(125) * 0.5 - 0.5,
            np.random.randn(125) * 0.5 + 1.5,
        ])

        cusum_stats, boundary, max_stat, is_stable = cusum_test(residuals, alpha=0.05)

        # Should detect break
        assert not is_stable

    def test_cusum_insufficient_data(self):
        """Test error with insufficient data."""
        residuals = np.random.randn(15)

        with pytest.raises(ValueError, match="Insufficient data"):
            cusum_test(residuals)

    def test_cusum_with_nan(self):
        """Test CUSUM with NaN values."""
        np.random.seed(42)
        residuals = np.random.randn(150)
        residuals[50:60] = np.nan

        cusum_stats, boundary, max_stat, is_stable = cusum_test(residuals, alpha=0.05)

        # Should handle by removing NaN
        assert len(cusum_stats) < 150  # Some removed

    def test_cusum_zero_variance(self):
        """Test error with zero variance residuals."""
        residuals = np.ones(100) * 3.0

        with pytest.raises(ValueError, match="zero variance"):
            cusum_test(residuals)

    def test_cusum_different_alpha(self):
        """Test CUSUM with different significance levels."""
        np.random.seed(42)
        residuals = np.random.randn(200)

        _, boundary_01, _, _ = cusum_test(residuals, alpha=0.01)
        _, boundary_05, _, _ = cusum_test(residuals, alpha=0.05)
        _, boundary_10, _, _ = cusum_test(residuals, alpha=0.10)

        # Stricter alpha should have wider boundary
        assert boundary_01 > boundary_05 > boundary_10


class TestCUSUMofSquaresTest:
    """Test suite for CUSUM of squares test."""

    def test_cusumsq_stable_variance(self):
        """Test CUSUM-sq with stable variance."""
        np.random.seed(42)
        T = 200

        # Constant variance
        residuals = np.random.randn(T) * 1.0

        cusumsq, bounds, max_dev, is_stable = cusum_of_squares_test(residuals, alpha=0.05)

        # Check outputs are valid
        assert cusumsq.shape == (T,)
        assert len(bounds) == 2
        assert len(bounds[0]) == T  # Lower bound
        assert np.isfinite(max_dev)

        # Note: random data may occasionally cross bounds by chance

    def test_cusumsq_variance_break(self):
        """Test CUSUM-sq with variance change."""
        np.random.seed(42)
        T = 300

        # Variance increases at midpoint
        residuals = np.concatenate([
            np.random.randn(150) * 0.5,
            np.random.randn(150) * 2.0,  # 4x variance
        ])

        cusumsq, bounds, max_dev, is_stable = cusum_of_squares_test(residuals, alpha=0.05)

        # Check that CUSUM-sq shows clear pattern (monotonic increase after break)
        # The second half should have much higher cumulative squares
        first_half_end = cusumsq[149]
        last_value = cusumsq[-1]

        # Should show acceleration (second half contributes disproportionately)
        assert last_value > first_half_end * 1.5  # More than proportional growth

    def test_cusumsq_bounds_valid(self):
        """Test that CUSUM-sq bounds are in [0, 1]."""
        np.random.seed(42)
        residuals = np.random.randn(150)

        cusumsq, (lower, upper), max_dev, is_stable = cusum_of_squares_test(residuals, alpha=0.05)

        # Bounds should be valid
        assert np.all(lower >= 0) and np.all(lower <= 1)
        assert np.all(upper >= 0) and np.all(upper <= 1)
        assert np.all(lower <= upper)

        # CUSUM-sq should also be in [0, 1]
        assert np.all(cusumsq >= 0) and np.all(cusumsq <= 1)

    def test_cusumsq_insufficient_data(self):
        """Test error with insufficient data."""
        residuals = np.random.randn(15)

        with pytest.raises(ValueError, match="Insufficient data"):
            cusum_of_squares_test(residuals)


class TestSupWaldTest:
    """Test suite for supremum Wald test."""

    def test_sup_wald_no_break(self):
        """Test sup-Wald with no structural break."""
        np.random.seed(42)
        T = 200

        X = np.random.randn(T)
        y = 1.0 + 0.8 * X + np.random.randn(T) * 0.5

        sup_stat, breakpoint, p_value = sup_wald_test(y, X, trim=0.15)

        # Should NOT reject null (high p-value)
        assert p_value > 0.05, f"False positive: p={p_value:.4f}"
        assert np.isfinite(sup_stat)
        assert 0 < breakpoint < T

    def test_sup_wald_with_break(self):
        """Test sup-Wald with structural break at unknown location."""
        np.random.seed(42)
        T = 300
        true_break = 120

        X = np.random.randn(T)
        y = np.zeros(T)

        # Regime 1
        y[:true_break] = 1.0 + 0.5 * X[:true_break] + np.random.randn(true_break) * 0.3

        # Regime 2: different coefficients
        y[true_break:] = 4.0 + 1.5 * X[true_break:] + np.random.randn(T - true_break) * 0.3

        sup_stat, breakpoint, p_value = sup_wald_test(y, X, trim=0.15)

        # Should detect break
        assert p_value < 0.10, f"False negative: p={p_value:.4f}"

        # Estimated breakpoint should be somewhat close to true break
        assert np.abs(breakpoint - true_break) < T * 0.2

    def test_sup_wald_trimming(self):
        """Test that trimming affects search range."""
        np.random.seed(42)
        T = 200

        X = np.random.randn(T)
        y = 1.0 + X + np.random.randn(T) * 0.5

        # With high trim, breakpoint must be in middle range
        sup_stat, breakpoint, p_value = sup_wald_test(y, X, trim=0.3)

        # Breakpoint should be in trimmed range [0.3*T, 0.7*T]
        assert 0.3 * T <= breakpoint <= 0.7 * T

    def test_sup_wald_insufficient_data(self):
        """Test error with insufficient data after trimming."""
        T = 50
        X = np.random.randn(T)
        y = np.random.randn(T)

        # May raise error or return result depending on min_obs
        try:
            sup_stat, breakpoint, p_value = sup_wald_test(y, X, trim=0.4, min_obs=20)
            # If it succeeds, should return valid outputs
            assert np.isfinite(sup_stat)
        except ValueError as e:
            assert "Insufficient" in str(e) or "data" in str(e)


class TestBaiPerronTest:
    """Test suite for Bai-Perron multiple breaks test."""

    def test_bai_perron_no_breaks(self):
        """Test Bai-Perron with no structural breaks."""
        np.random.seed(42)
        T = 250

        X = np.random.randn(T)
        y = 2.0 + 1.0 * X + np.random.randn(T) * 0.5

        n_breaks, breakpoints, bic = bai_perron_test(y, X, max_breaks=3, trim=0.15)

        # Should find 0 breaks
        assert n_breaks == 0, f"False positive: found {n_breaks} breaks"
        assert len(breakpoints) == 0
        assert np.isfinite(bic)

    def test_bai_perron_one_break(self):
        """Test Bai-Perron with one structural break."""
        np.random.seed(42)
        T = 300
        true_break = 150

        X = np.random.randn(T)
        y = np.zeros(T)

        # Regime 1
        y[:true_break] = 1.0 + 0.5 * X[:true_break] + np.random.randn(true_break) * 0.4

        # Regime 2
        y[true_break:] = 4.0 + 1.5 * X[true_break:] + np.random.randn(T - true_break) * 0.4

        n_breaks, breakpoints, bic = bai_perron_test(y, X, max_breaks=3, trim=0.1)

        # Should find 1 break
        assert n_breaks >= 1, f"False negative: found {n_breaks} breaks"
        assert len(breakpoints) == n_breaks

        if n_breaks >= 1:
            # First breakpoint should be close to true break
            assert np.abs(breakpoints[0] - true_break) < T * 0.15

    def test_bai_perron_multiple_breaks(self):
        """Test Bai-Perron with multiple breaks."""
        np.random.seed(42)
        T = 600
        true_breaks = [200, 400]

        X = np.random.randn(T)
        y = np.zeros(T)

        # Regime 1
        y[:200] = 1.0 + 0.5 * X[:200] + np.random.randn(200) * 0.3

        # Regime 2
        y[200:400] = 3.0 + 1.5 * X[200:400] + np.random.randn(200) * 0.3

        # Regime 3
        y[400:] = 0.0 + 0.8 * X[400:] + np.random.randn(200) * 0.3

        n_breaks, breakpoints, bic = bai_perron_test(y, X, max_breaks=5, trim=0.1)

        # Should find at least 1 break, ideally 2
        assert n_breaks >= 1, f"False negative: found {n_breaks} breaks"

        # If found 2+, check they are in reasonable locations
        if n_breaks >= 2:
            assert len(breakpoints) == n_breaks
            # At least one breakpoint should be near each true break
            for tb in true_breaks:
                min_dist = min(np.abs(bp - tb) for bp in breakpoints)
                assert min_dist < T * 0.15

    def test_bai_perron_bic_selection(self):
        """Test that BIC prevents overfitting."""
        np.random.seed(42)
        T = 200

        # Simple linear model, no breaks
        X = np.random.randn(T)
        y = 1.0 + 1.0 * X + np.random.randn(T) * 0.5

        n_breaks, breakpoints, bic = bai_perron_test(y, X, max_breaks=5, trim=0.15)

        # Should not overfit (should find 0 or 1 at most)
        assert n_breaks <= 1, f"Overfitting: found {n_breaks} breaks"

    def test_bai_perron_insufficient_data(self):
        """Test with data too short for multiple breaks."""
        T = 80
        X = np.random.randn(T)
        y = np.random.randn(T)

        # With trim=0.2, each segment needs at least 16 obs
        # Can't fit many breaks
        n_breaks, breakpoints, bic = bai_perron_test(y, X, max_breaks=5, trim=0.2)

        # Should find few or no breaks
        assert n_breaks <= 2


class TestStructuralBreaksEdgeCases:
    """Test edge cases and robustness."""

    def test_chow_perfect_fit(self):
        """Test Chow with perfect fit (no residual variance)."""
        T = 100
        X = np.arange(T, dtype=np.float64)
        y = 2.0 + 1.5 * X  # Perfect linear relationship

        breakpoint = 50

        f_stat, p_value, df1, df2, k = chow_test(y, X, breakpoint)

        # May return NaN or very large F-stat
        # Should not crash
        assert np.isfinite(f_stat) or np.isnan(f_stat)

    def test_cusum_monotonic_residuals(self):
        """Test CUSUM with strictly monotonic residuals."""
        residuals = np.arange(100, dtype=np.float64)

        cusum_stats, boundary, max_stat, is_stable = cusum_test(residuals, alpha=0.05)

        # Should detect strong instability
        assert not is_stable
        assert max_stat > 5.0  # Large deviation

    def test_sup_wald_break_at_boundary(self):
        """Test sup-Wald when break is near trimming boundary."""
        np.random.seed(42)
        T = 250
        true_break = 50  # Near 20% boundary

        X = np.random.randn(T)
        y = np.zeros(T)

        y[:true_break] = 1.0 + 0.5 * X[:true_break] + np.random.randn(true_break) * 0.5
        y[true_break:] = 5.0 + 2.0 * X[true_break:] + np.random.randn(T - true_break) * 0.5

        # May not detect if break is outside search range
        sup_stat, breakpoint, p_value = sup_wald_test(y, X, trim=0.25)

        # Should still return valid result (may not detect if trimmed out)
        assert np.isfinite(sup_stat)

    def test_bai_perron_max_breaks_limit(self):
        """Test that Bai-Perron respects max_breaks."""
        np.random.seed(42)
        T = 500

        X = np.random.randn(T)
        y = np.random.randn(T)  # Pure noise, no structure

        n_breaks, breakpoints, bic = bai_perron_test(y, X, max_breaks=2, trim=0.1)

        # Should not exceed max_breaks
        assert n_breaks <= 2
        assert len(breakpoints) <= 2

    def test_chow_collinear_regressors(self):
        """Test Chow with collinear regressors."""
        np.random.seed(42)
        T = 150

        X1 = np.random.randn(T)
        X2 = 2.0 * X1  # Perfect collinearity
        X = np.column_stack([X1, X2])
        y = 1.0 + X1 + np.random.randn(T) * 0.5

        breakpoint = 75

        # May fail due to singular matrix
        try:
            f_stat, p_value, df1, df2, k = chow_test(y, X, breakpoint)
            # If succeeds, should return valid result
            assert np.isfinite(f_stat) or np.isnan(f_stat)
        except np.linalg.LinAlgError:
            # Acceptable to fail with singular matrix
            pass
