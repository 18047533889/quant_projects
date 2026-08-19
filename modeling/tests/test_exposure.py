"""
Tests for exposure decomposition and soft neutralization.
"""
import pytest
import numpy as np

from modeling_adapters.exposure import compute_exposure_residual, soft_neutralization
from modeling_adapters.errors import ContractViolation


class TestComputeExposureResidual:
    def test_basic_ols_residual(self):
        np.random.seed(42)
        T, N, K = 100, 50, 3
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        residuals = compute_exposure_residual(factor_values, exposures, method="ols")
        assert residuals.shape == factor_values.shape

    def test_residual_with_weights(self):
        np.random.seed(42)
        T, N, K = 50, 30, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        weights = np.abs(np.random.randn(T, N))
        residuals = compute_exposure_residual(factor_values, exposures, method="weighted_ols", weights=weights)
        assert residuals.shape == factor_values.shape

    def test_ridge_residual(self):
        np.random.seed(42)
        T, N, K = 50, 30, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        residuals = compute_exposure_residual(factor_values, exposures, method="ridge")
        assert residuals.shape == factor_values.shape

    def test_mismatched_time_dimension(self):
        factor_values = np.random.randn(100, 50)
        exposures = np.random.randn(80, 50, 3)
        with pytest.raises(ContractViolation, match="same time dimension"):
            compute_exposure_residual(factor_values, exposures)

    def test_mismatched_asset_dimension(self):
        factor_values = np.random.randn(100, 50)
        exposures = np.random.randn(100, 40, 3)
        with pytest.raises(ContractViolation, match="same asset dimension"):
            compute_exposure_residual(factor_values, exposures)

    def test_wrong_exposure_ndim(self):
        factor_values = np.random.randn(100, 50)
        exposures = np.random.randn(100, 50)
        with pytest.raises(ContractViolation, match="must be 3D"):
            compute_exposure_residual(factor_values, exposures)

    def test_residual_with_nan(self):
        np.random.seed(42)
        T, N, K = 50, 30, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        factor_values[10, 5] = np.nan
        exposures[20, 10, 0] = np.nan
        residuals = compute_exposure_residual(factor_values, exposures, method="ols")
        assert np.isnan(residuals[10, 5])

    def test_cross_sectional_independence(self):
        np.random.seed(42)
        T, N, K = 50, 30, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        residuals = compute_exposure_residual(factor_values, exposures, method="ols")
        t = 10
        y = factor_values[t, :]
        X = exposures[t, :, :]
        valid = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        if np.sum(valid) >= K + 1:
            beta = np.linalg.lstsq(X[valid, :], y[valid], rcond=None)[0]
            expected_residual = y.copy()
            expected_residual[valid] = y[valid] - X[valid, :] @ beta
            np.testing.assert_allclose(residuals[t, valid], expected_residual[valid], rtol=1e-5)


class TestSoftNeutralization:
    def test_soft_neutralization_alpha_zero(self):
        np.random.seed(42)
        T, N, K = 50, 30, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        result = soft_neutralization(factor_values, exposures, alpha=0.0)
        np.testing.assert_array_equal(result, factor_values)

    def test_soft_neutralization_alpha_one(self):
        np.random.seed(42)
        T, N, K = 50, 30, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        result_soft = soft_neutralization(factor_values, exposures, alpha=1.0)
        result_full = compute_exposure_residual(factor_values, exposures, method="ols")
        np.testing.assert_allclose(result_soft, result_full, rtol=1e-10)

    def test_soft_neutralization_partial(self):
        np.random.seed(42)
        T, N, K = 50, 30, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        alpha = 0.5
        result = soft_neutralization(factor_values, exposures, alpha=alpha)
        residuals = compute_exposure_residual(factor_values, exposures, method="ols")
        expected = (1 - alpha) * factor_values + alpha * residuals
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_soft_neutralization_invalid_alpha(self):
        factor_values = np.random.randn(50, 30)
        exposures = np.random.randn(50, 30, 2)
        with pytest.raises(ValueError, match="alpha must be in"):
            soft_neutralization(factor_values, exposures, alpha=-0.1)

    def test_soft_neutralization_preserves_shape(self):
        np.random.seed(42)
        T, N, K = 50, 30, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = soft_neutralization(factor_values, exposures, alpha=alpha)
            assert result.shape == factor_values.shape


class TestExposureDecompositionIntegration:
    def test_pipeline_winsorize_then_neutralize(self):
        from modeling_adapters.preprocess.stateless import winsorize
        np.random.seed(42)
        T, N, K = 100, 50, 3
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        factor_values[10, 5] = 50
        factor_winsorized = winsorize(factor_values, lower=0.01, upper=0.99, axis=1)
        factor_neutral = compute_exposure_residual(factor_winsorized, exposures, method="ols")
        assert factor_neutral.shape == factor_values.shape

    def test_multiple_exposure_sets(self):
        np.random.seed(42)
        T, N = 100, 50
        factor_values = np.random.randn(T, N)
        exposures_industry = np.random.randn(T, N, 10)
        residual_industry = compute_exposure_residual(factor_values, exposures_industry, method="ols")
        exposures_full = np.random.randn(T, N, 11)
        residual_full = compute_exposure_residual(factor_values, exposures_full, method="ols")
        assert np.nanvar(residual_full) <= np.nanvar(residual_industry)

    def test_exposure_stability_over_time(self):
        np.random.seed(42)
        T, N, K = 100, 50, 2
        exposures = np.random.randn(T, N, K)
        true_beta = np.array([0.5, -0.3])
        factor_values = np.zeros((T, N))
        for t in range(T):
            factor_values[t, :] = exposures[t, :, :] @ true_beta + np.random.randn(N) * 0.1
        residuals = compute_exposure_residual(factor_values, exposures, method="ols")
        original_corr = np.abs(np.corrcoef(factor_values.ravel(), exposures[:, :, 0].ravel())[0, 1])
        residual_corr = np.abs(np.corrcoef(residuals.ravel(), exposures[:, :, 0].ravel())[0, 1])
        assert residual_corr < original_corr / 2


class TestExposureHardening:
    def test_inf_y_is_invalid_not_residual(self):
        """An inf in the factor must be masked out, not admitted as a valid
        observation emitting a literal inf residual (weighted_ols probe)."""
        np.random.seed(3)
        T, N, K = 10, 8, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        weights = np.ones((T, N))
        factor_values[3, 2] = np.inf

        residuals = compute_exposure_residual(
            factor_values, exposures, method="weighted_ols", weights=weights
        )
        # The inf asset is masked; remaining residuals are finite/NaN, never inf
        assert not np.any(np.isposinf(residuals))
        assert not np.any(np.isneginf(residuals))
        # Row still produced residuals for the other assets
        assert np.any(np.isfinite(residuals[3]))

    def test_ridge_neutralizes_small_scale_exposures(self):
        """Scale-blind lambda=0.01 previously left small-scale exposures
        essentially un-neutralized (residual ~= raw factor)."""
        np.random.seed(4)
        T, N, K = 60, 40, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K) * 1e-4  # tiny scale

        residuals = compute_exposure_residual(factor_values, exposures, method="ridge")

        # Residual must genuinely differ from the raw factor (neutralization
        # happened), while staying finite.
        diff = np.nanmean(np.abs(residuals - factor_values))
        assert np.isfinite(diff)
        assert diff > 0.1

    def test_weighted_ols_rejects_negative_weights(self):
        np.random.seed(5)
        T, N, K = 10, 8, 2
        factor_values = np.random.randn(T, N)
        exposures = np.random.randn(T, N, K)
        weights = np.ones((T, N))
        weights[0, 0] = -0.5

        with pytest.raises(ContractViolation, match="non-negative weights"):
            compute_exposure_residual(
                factor_values, exposures, method="weighted_ols", weights=weights
            )
