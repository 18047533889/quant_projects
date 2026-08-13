"""
Tests for exposure metrics with golden reference values.
"""

import pytest
import numpy as np

from quant_evaluator.metrics.exposure import (
    compute_factor_loadings,
    compute_sector_exposure,
    compute_style_exposure,
    compute_concentration_hhi,
)


class TestFactorLoadings:
    """Test factor loading estimation via cross-sectional regression."""

    def test_loadings_single_risk_factor(self):
        """Single risk factor regression."""
        T, N = 10, 50
        np.random.seed(42)

        # Factor = 0.5 + 2.0 * risk_factor + noise
        risk_factors = np.random.randn(T, N, 1)
        factor_values = 0.5 + 2.0 * risk_factors[:, :, 0] + np.random.randn(T, N) * 0.1

        loadings, r_squared, residuals = compute_factor_loadings(
            factor_values, risk_factors, intercept=True, min_obs=10
        )

        assert loadings.shape == (T, 2)  # intercept + 1 risk factor
        assert r_squared.shape == (T,)
        assert residuals.shape == (T, N)

        # Loadings should be close to [0.5, 2.0]
        mean_intercept = np.nanmean(loadings[:, 0])
        mean_beta = np.nanmean(loadings[:, 1])

        assert np.isclose(mean_intercept, 0.5, atol=0.2)
        assert np.isclose(mean_beta, 2.0, atol=0.2)

        # R^2 should be high
        assert np.all(r_squared > 0.8)

    def test_loadings_multiple_risk_factors(self):
        """Multiple risk factors."""
        T, N, K = 5, 100, 3
        np.random.seed(100)

        risk_factors = np.random.randn(T, N, K)
        true_betas = np.array([1.5, -0.8, 0.3])

        # factor = intercept + sum(beta_k * risk_factor_k) + noise
        factor_values = np.zeros((T, N))
        for t in range(T):
            factor_values[t, :] = 1.0 + risk_factors[t, :, :] @ true_betas + np.random.randn(N) * 0.1

        loadings, r_squared, residuals = compute_factor_loadings(
            factor_values, risk_factors, intercept=True, min_obs=10
        )

        assert loadings.shape == (T, K + 1)

        # Check estimated betas
        mean_betas = np.nanmean(loadings[:, 1:], axis=0)
        assert np.allclose(mean_betas, true_betas, atol=0.2)

    def test_loadings_no_intercept(self):
        """Regression without intercept."""
        T, N = 8, 60
        np.random.seed(50)

        risk_factors = np.random.randn(T, N, 1)
        factor_values = 3.0 * risk_factors[:, :, 0]  # No intercept

        loadings, r_squared, residuals = compute_factor_loadings(
            factor_values, risk_factors, intercept=False, min_obs=10
        )

        assert loadings.shape == (T, 1)  # No intercept column
        mean_beta = np.nanmean(loadings[:, 0])
        assert np.isclose(mean_beta, 3.0, atol=0.1)

    def test_loadings_insufficient_obs(self):
        """Insufficient observations returns NaN."""
        T, N = 3, 5
        risk_factors = np.random.randn(T, N, 1)
        factor_values = np.random.randn(T, N)

        loadings, r_squared, residuals = compute_factor_loadings(
            factor_values, risk_factors, intercept=True, min_obs=10
        )

        # All periods have < 10 obs
        assert np.all(np.isnan(loadings))

    def test_loadings_with_nans(self):
        """NaN values are filtered per period."""
        T, N = 5, 50
        np.random.seed(42)

        risk_factors = np.random.randn(T, N, 1)
        factor_values = 2.0 * risk_factors[:, :, 0] + np.random.randn(T, N) * 0.1

        # Introduce NaNs
        factor_values[0, :10] = np.nan
        risk_factors[1, :5, 0] = np.nan

        loadings, r_squared, residuals = compute_factor_loadings(
            factor_values, risk_factors, intercept=True, min_obs=10
        )

        # Should still get valid estimates
        assert np.sum(~np.isnan(loadings[:, 1])) >= 3


class TestSectorExposure:
    """Test sector exposure computation."""

    def test_sector_exposure_equal_weight(self):
        """Equal-weighted sector exposure."""
        T, N = 10, 60
        np.random.seed(42)

        factor_values = np.random.randn(T, N)
        # 3 sectors: 0, 1, 2
        sector_labels = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=float)

        sector_exposure, sector_counts = compute_sector_exposure(
            factor_values, sector_labels, weights=None
        )

        assert sector_exposure.shape == (T, 3)
        assert sector_counts.shape == (T, 3)

        # Each sector should have 20 assets
        assert np.all(sector_counts == 20)

        # Check manual calculation for first period
        expected_sector_0 = np.mean(factor_values[0, :20])
        assert np.isclose(sector_exposure[0, 0], expected_sector_0, atol=1e-10)

    def test_sector_exposure_weighted(self):
        """Weighted sector exposure."""
        T, N = 5, 30
        np.random.seed(100)

        factor_values = np.random.randn(T, N)
        sector_labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=float)

        # Custom weights
        weights = np.random.rand(T, N) + 0.5

        sector_exposure, sector_counts = compute_sector_exposure(
            factor_values, sector_labels, weights=weights
        )

        assert sector_exposure.shape == (T, 3)

        # Manual calculation for sector 0, period 0
        sector_0_mask = sector_labels == 0
        factor_s0 = factor_values[0, sector_0_mask]
        weight_s0 = weights[0, sector_0_mask]
        expected = np.sum(factor_s0 * weight_s0) / np.sum(weight_s0)

        assert np.isclose(sector_exposure[0, 0], expected, atol=1e-10)

    def test_sector_exposure_with_nans(self):
        """NaN values are excluded from sector calculation."""
        T, N = 3, 20
        factor_values = np.ones((T, N))
        factor_values[0, :5] = np.nan

        sector_labels = np.array([0] * 10 + [1] * 10, dtype=float)

        sector_exposure, sector_counts = compute_sector_exposure(
            factor_values, sector_labels, weights=None
        )

        # Sector 0 should have 5 valid assets in period 0
        assert sector_counts[0, 0] == 5
        # Sector 1 should have 10 valid assets
        assert sector_counts[0, 1] == 10


class TestStyleExposure:
    """Test style factor exposure."""

    def test_style_exposure_single_style(self):
        """Single style factor exposure."""
        T, N = 50, 100
        np.random.seed(42)

        # Factor has exposure to style factor
        style_factors = np.random.randn(T, N, 1)
        factor_values = 1.5 * style_factors[:, :, 0] + np.random.randn(T, N) * 0.5

        mean_exposures = compute_style_exposure(
            factor_values, style_factors, style_names=("momentum",)
        )

        assert mean_exposures.shape == (1,)
        assert np.isclose(mean_exposures[0], 1.5, atol=0.3)

    def test_style_exposure_multiple_styles(self):
        """Multiple style factors."""
        T, N = 40, 120
        np.random.seed(100)

        style_factors = np.random.randn(T, N, 3)
        true_exposures = np.array([2.0, -1.0, 0.5])

        factor_values = np.zeros((T, N))
        for t in range(T):
            factor_values[t, :] = (
                0.5 + style_factors[t, :, :] @ true_exposures + np.random.randn(N) * 0.3
            )

        mean_exposures = compute_style_exposure(
            factor_values,
            style_factors,
            style_names=("value", "momentum", "quality"),
        )

        assert mean_exposures.shape == (3,)
        assert np.allclose(mean_exposures, true_exposures, atol=0.3)

    def test_style_exposure_name_mismatch_raises(self):
        """Mismatched style names raises error."""
        style_factors = np.random.randn(10, 50, 2)
        factor_values = np.random.randn(10, 50)

        with pytest.raises(ValueError, match="has 2 factors but 3 names"):
            compute_style_exposure(
                factor_values, style_factors, style_names=("a", "b", "c")
            )


class TestConcentrationHHI:
    """Test Herfindahl-Hirschman Index."""

    def test_hhi_equal_weight_uniform(self):
        """Equal-weight uniform exposure yields low HHI."""
        T, N = 5, 100
        # All assets have same factor value
        factor_values = np.ones((T, N))

        hhi = compute_concentration_hhi(factor_values, weights=None)

        assert hhi.shape == (T,)
        # With equal weights and uniform exposure, HHI = 1/N
        expected_hhi = 1.0 / N
        assert np.allclose(hhi, expected_hhi, atol=1e-10)

    def test_hhi_concentrated_exposure(self):
        """Concentrated exposure yields high HHI."""
        T, N = 3, 50
        factor_values = np.zeros((T, N))
        # Only first asset has non-zero value
        factor_values[:, 0] = 10.0

        hhi = compute_concentration_hhi(factor_values, weights=None)

        # All exposure concentrated in one asset -> HHI = 1.0
        assert np.allclose(hhi, 1.0, atol=1e-10)

    def test_hhi_weighted(self):
        """Weighted HHI computation."""
        T, N = 5, 10
        np.random.seed(42)

        factor_values = np.random.randn(T, N)
        # One asset has very high weight
        weights = np.ones((T, N))
        weights[:, 0] = 10.0

        hhi = compute_concentration_hhi(factor_values, weights=weights)

        # High weight on first asset -> higher HHI
        assert np.all(hhi > 0.1)

    def test_hhi_with_nans(self):
        """NaN values are excluded."""
        T, N = 3, 20
        factor_values = np.random.randn(T, N)
        factor_values[0, :10] = np.nan

        hhi = compute_concentration_hhi(factor_values, weights=None)

        # Should compute HHI on valid assets only
        assert np.all(np.isfinite(hhi))
