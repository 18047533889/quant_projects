"""
Tests for distribution metrics with golden reference values.
"""

import pytest
import numpy as np
from scipy import stats

from quant_evaluator.metrics.distribution import (
    compute_skewness,
    compute_kurtosis,
    detect_outliers_iqr,
    detect_outliers_zscore,
    compute_outlier_ratio,
    compute_higher_moments,
)


class TestSkewness:
    """Test skewness computation."""

    def test_skewness_normal(self):
        """Normal distribution has near-zero skewness."""
        np.random.seed(42)
        values = np.random.randn(1000, 50, 3)  # (T, N, F)

        skew = compute_skewness(values, axis=0, min_obs=100)

        assert skew.shape == (50, 3)
        # Normal distribution: skewness near 0
        assert np.all(np.abs(skew) < 0.3)

    def test_skewness_right_tailed(self):
        """Right-tailed distribution has positive skewness."""
        np.random.seed(100)
        # Exponential distribution (right-tailed)
        values = np.random.exponential(scale=1.0, size=(500, 100, 1))

        skew = compute_skewness(values, axis=0, min_obs=50)

        assert skew.shape == (100, 1)
        # Exponential: positive skewness
        assert np.all(skew > 1.0)

    def test_skewness_left_tailed(self):
        """Left-tailed distribution has negative skewness."""
        np.random.seed(200)
        # Negative exponential (left-tailed)
        values = -np.random.exponential(scale=1.0, size=(500, 100, 1))

        skew = compute_skewness(values, axis=0, min_obs=50)

        assert skew.shape == (100, 1)
        # Negative exponential: negative skewness
        assert np.all(skew < -1.0)

    def test_skewness_scipy_parity(self):
        """Verify parity with scipy.stats.skew."""
        np.random.seed(42)
        values = np.random.randn(200, 50, 2)

        our_skew = compute_skewness(values, axis=0, min_obs=10)
        scipy_skew = stats.skew(values, axis=0, bias=False)

        assert np.allclose(our_skew, scipy_skew, atol=1e-10)

    def test_skewness_insufficient_obs(self):
        """Insufficient observations returns NaN."""
        values = np.random.randn(5, 10, 2)

        skew = compute_skewness(values, axis=0, min_obs=20)

        assert np.all(np.isnan(skew))

    def test_skewness_with_nans(self):
        """NaN values are properly filtered."""
        np.random.seed(42)
        values = np.random.randn(100, 20, 1)
        values[::10, :, :] = np.nan  # 10% NaN

        skew = compute_skewness(values, axis=0, min_obs=50)

        assert skew.shape == (20, 1)
        # Should compute with remaining valid data
        assert np.all(np.isfinite(skew))

    def test_skewness_cross_sectional(self):
        """Skewness along cross-sectional axis."""
        np.random.seed(42)
        values = np.random.randn(100, 1000, 1)

        skew = compute_skewness(values, axis=1, min_obs=100)

        assert skew.shape == (100, 1)
        # Cross-sectional skewness of normal data
        assert np.all(np.abs(skew) < 0.5)


class TestKurtosis:
    """Test kurtosis computation."""

    def test_kurtosis_normal(self):
        """Normal distribution has near-zero excess kurtosis."""
        np.random.seed(42)
        values = np.random.randn(1000, 50, 2)

        kurt = compute_kurtosis(values, axis=0, min_obs=100, excess=True)

        assert kurt.shape == (50, 2)
        # Normal distribution: excess kurtosis near 0
        assert np.all(np.abs(kurt) < 0.5)

    def test_kurtosis_heavy_tailed(self):
        """Heavy-tailed distribution has positive excess kurtosis."""
        np.random.seed(100)
        # t-distribution with df=3 (heavy tails)
        values = stats.t.rvs(df=3, size=(500, 100, 1))

        kurt = compute_kurtosis(values, axis=0, min_obs=50, excess=True)

        assert kurt.shape == (100, 1)
        # t(3): excess kurtosis > 0
        assert np.mean(kurt) > 0.5

    def test_kurtosis_uniform(self):
        """Uniform distribution has negative excess kurtosis."""
        np.random.seed(200)
        # Uniform distribution (light tails)
        values = np.random.uniform(low=-1, high=1, size=(500, 100, 1))

        kurt = compute_kurtosis(values, axis=0, min_obs=50, excess=True)

        assert kurt.shape == (100, 1)
        # Uniform: excess kurtosis ~ -1.2
        assert np.mean(kurt) < -0.8

    def test_kurtosis_scipy_parity(self):
        """Verify parity with scipy.stats.kurtosis."""
        np.random.seed(42)
        values = np.random.randn(200, 50, 2)

        our_kurt = compute_kurtosis(values, axis=0, min_obs=10, excess=True)
        scipy_kurt = stats.kurtosis(values, axis=0, bias=False, fisher=True)

        assert np.allclose(our_kurt, scipy_kurt, atol=1e-10)

    def test_kurtosis_not_excess(self):
        """Non-excess kurtosis (Fisher=False)."""
        np.random.seed(42)
        values = np.random.randn(500, 50, 1)

        kurt = compute_kurtosis(values, axis=0, min_obs=50, excess=False)

        # Normal distribution: kurtosis ~ 3 (mean should be close to 3)
        assert np.abs(np.mean(kurt) - 3.0) < 0.2

    def test_kurtosis_insufficient_obs(self):
        """Insufficient observations returns NaN."""
        values = np.random.randn(5, 10, 2)

        kurt = compute_kurtosis(values, axis=0, min_obs=20)

        assert np.all(np.isnan(kurt))


class TestOutliersIQR:
    """Test IQR-based outlier detection."""

    def test_outliers_iqr_normal(self):
        """Normal distribution with few outliers."""
        np.random.seed(42)
        values = np.random.randn(1000, 1, 1)

        outliers = detect_outliers_iqr(values, axis=0, multiplier=1.5, min_obs=100)

        assert outliers.shape == (1000, 1, 1)
        ratio = np.sum(outliers) / 1000
        # Normal: ~0.7% outliers with 1.5*IQR
        assert 0.0 < ratio < 0.02

    def test_outliers_iqr_with_extreme(self):
        """Detect injected extreme outliers."""
        np.random.seed(100)
        values = np.random.randn(100, 1, 1)
        # Inject outliers
        values[0, 0, 0] = 100.0
        values[1, 0, 0] = -100.0

        outliers = detect_outliers_iqr(values, axis=0, multiplier=1.5, min_obs=10)

        # First two should be detected
        assert outliers[0, 0, 0]
        assert outliers[1, 0, 0]

    def test_outliers_iqr_cross_sectional(self):
        """Cross-sectional outlier detection."""
        np.random.seed(200)
        values = np.random.randn(10, 1000, 1)
        # Inject cross-sectional outlier
        values[5, 0, 0] = 50.0

        outliers = detect_outliers_iqr(values, axis=1, multiplier=1.5, min_obs=100)

        assert outliers.shape == (10, 1000, 1)
        # Outlier should be detected in that time period
        assert outliers[5, 0, 0]

    def test_outliers_iqr_multiplier(self):
        """Higher multiplier detects fewer outliers."""
        np.random.seed(42)
        values = np.random.randn(500, 1, 1)

        outliers_15 = detect_outliers_iqr(values, axis=0, multiplier=1.5)
        outliers_30 = detect_outliers_iqr(values, axis=0, multiplier=3.0)

        ratio_15 = np.sum(outliers_15) / 500
        ratio_30 = np.sum(outliers_30) / 500

        # Higher multiplier -> fewer outliers
        assert ratio_15 > ratio_30

    def test_outliers_iqr_insufficient_obs(self):
        """Insufficient observations marks nothing as outlier."""
        values = np.random.randn(5, 10, 1)

        outliers = detect_outliers_iqr(values, axis=0, multiplier=1.5, min_obs=100)

        assert np.sum(outliers) == 0

    def test_outliers_iqr_with_nans(self):
        """NaN values are not marked as outliers."""
        np.random.seed(42)
        values = np.random.randn(100, 1, 1)
        values[::10, 0, 0] = np.nan

        outliers = detect_outliers_iqr(values, axis=0, multiplier=1.5, min_obs=50)

        # NaN positions should not be outliers
        assert not np.any(outliers[::10, 0, 0])


class TestOutliersZScore:
    """Test z-score-based outlier detection."""

    def test_outliers_zscore_normal(self):
        """Normal distribution with few outliers at 3-sigma."""
        np.random.seed(42)
        values = np.random.randn(1000, 1, 1)

        outliers = detect_outliers_zscore(values, axis=0, threshold=3.0, min_obs=100)

        assert outliers.shape == (1000, 1, 1)
        ratio = np.sum(outliers) / 1000
        # Normal: ~0.3% beyond 3-sigma
        assert 0.0 <= ratio < 0.01

    def test_outliers_zscore_with_extreme(self):
        """Detect injected extreme outliers."""
        np.random.seed(100)
        values = np.random.randn(100, 1, 1)
        # Inject outliers (5+ sigma)
        values[0, 0, 0] = 10.0
        values[1, 0, 0] = -10.0

        outliers = detect_outliers_zscore(values, axis=0, threshold=3.0, min_obs=10)

        # First two should be detected
        assert outliers[0, 0, 0]
        assert outliers[1, 0, 0]

    def test_outliers_zscore_threshold(self):
        """Lower threshold detects more outliers."""
        np.random.seed(42)
        values = np.random.randn(1000, 1, 1)

        outliers_2 = detect_outliers_zscore(values, axis=0, threshold=2.0)
        outliers_3 = detect_outliers_zscore(values, axis=0, threshold=3.0)

        ratio_2 = np.sum(outliers_2) / 1000
        ratio_3 = np.sum(outliers_3) / 1000

        # Lower threshold -> more outliers
        assert ratio_2 > ratio_3
        # Normal: ~5% beyond 2-sigma, ~0.3% beyond 3-sigma
        assert 0.03 < ratio_2 < 0.07

    def test_outliers_zscore_constant(self):
        """Constant values (zero std) mark nothing as outlier."""
        values = np.full((100, 1, 1), 5.0)

        outliers = detect_outliers_zscore(values, axis=0, threshold=3.0)

        assert np.sum(outliers) == 0

    def test_outliers_zscore_insufficient_obs(self):
        """Insufficient observations marks nothing as outlier."""
        values = np.random.randn(5, 10, 1)

        outliers = detect_outliers_zscore(values, axis=0, threshold=3.0, min_obs=100)

        assert np.sum(outliers) == 0


class TestOutlierRatio:
    """Test outlier ratio computation."""

    def test_outlier_ratio_basic(self):
        """Basic outlier ratio calculation."""
        outlier_mask = np.zeros((100, 50, 2), dtype=bool)
        outlier_mask[:10, :, 0] = True  # 10% outliers in factor 0
        outlier_mask[:5, :, 1] = True   # 5% outliers in factor 1

        ratio = compute_outlier_ratio(outlier_mask, axis=0)

        assert ratio.shape == (50, 2)
        assert np.allclose(ratio[:, 0], 0.1)
        assert np.allclose(ratio[:, 1], 0.05)

    def test_outlier_ratio_cross_sectional(self):
        """Cross-sectional outlier ratio."""
        outlier_mask = np.zeros((10, 1000, 1), dtype=bool)
        outlier_mask[5, :100, 0] = True  # 10% outliers in time 5

        ratio = compute_outlier_ratio(outlier_mask, axis=1)

        assert ratio.shape == (10, 1)
        assert ratio[5, 0] == 0.1
        assert np.all(ratio[:5, 0] == 0.0)


class TestHigherMoments:
    """Test combined higher moments computation."""

    def test_higher_moments_normal(self):
        """Normal distribution moments."""
        np.random.seed(42)
        values = np.random.randn(1000, 100, 2)

        mean, std, skew, kurt = compute_higher_moments(values, axis=0, min_obs=100)

        assert mean.shape == (100, 2)
        assert std.shape == (100, 2)
        assert skew.shape == (100, 2)
        assert kurt.shape == (100, 2)

        # Normal distribution properties
        assert np.abs(np.mean(mean)) < 0.1  # Mean near 0
        assert np.abs(np.mean(std) - 1.0) < 0.1  # Std near 1
        assert np.abs(np.mean(skew)) < 0.2  # Skewness near 0
        assert np.abs(np.mean(kurt)) < 0.3  # Excess kurtosis near 0

    def test_higher_moments_shifted(self):
        """Shifted and scaled distribution."""
        np.random.seed(100)
        values = np.random.randn(500, 50, 1) * 2.0 + 5.0  # N(5, 4)

        mean, std, skew, kurt = compute_higher_moments(values, axis=0, min_obs=50)

        # Mean ~ 5, std ~ 2
        assert np.abs(np.mean(mean) - 5.0) < 0.2
        assert np.abs(np.mean(std) - 2.0) < 0.2
        # Skewness and kurtosis unchanged by affine transform
        assert np.abs(np.mean(skew)) < 0.3
        assert np.abs(np.mean(kurt)) < 0.3

    def test_higher_moments_insufficient_obs(self):
        """Insufficient observations returns NaN."""
        values = np.random.randn(10, 5, 2)

        mean, std, skew, kurt = compute_higher_moments(values, axis=0, min_obs=50)

        assert np.all(np.isnan(mean))
        assert np.all(np.isnan(std))
        assert np.all(np.isnan(skew))
        assert np.all(np.isnan(kurt))

    def test_higher_moments_golden_reference(self):
        """Golden reference with fixed seed."""
        np.random.seed(12345)
        values = np.random.randn(200, 1, 1)

        mean, std, skew, kurt = compute_higher_moments(values, axis=0, min_obs=20)

        # Golden values (verified with scipy)
        expected_mean = np.mean(values[:, 0, 0])
        expected_std = np.std(values[:, 0, 0], ddof=1)
        expected_skew = stats.skew(values[:, 0, 0], bias=False)
        expected_kurt = stats.kurtosis(values[:, 0, 0], bias=False, fisher=True)

        assert np.isclose(mean[0, 0], expected_mean, atol=1e-10)
        assert np.isclose(std[0, 0], expected_std, atol=1e-10)
        assert np.isclose(skew[0, 0], expected_skew, atol=1e-10)
        assert np.isclose(kurt[0, 0], expected_kurt, atol=1e-10)
