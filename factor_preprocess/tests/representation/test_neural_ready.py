"""
Test suite for neural-ready feature preparation.
"""
import pytest
import numpy as np
from factor_preprocess.representation.neural_ready import (
    build_neural_ready,
    prepare_embeddings,
    NeuralReadyConfig,
    NeuralReadyResult,
)


class TestNeuralReadyConfig:
    """Test NeuralReadyConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = NeuralReadyConfig(scaling="robust", handle_missing="zero")
        assert config.scaling == "robust"
        assert config.handle_missing == "zero"

    def test_invalid_scaling(self):
        """Test that invalid scaling is rejected."""
        with pytest.raises(ValueError, match="scaling must be one of"):
            NeuralReadyConfig(scaling="invalid")

    def test_invalid_handle_missing(self):
        """Test that invalid handle_missing is rejected."""
        with pytest.raises(ValueError, match="handle_missing must be one of"):
            NeuralReadyConfig(handle_missing="invalid")

    def test_invalid_robust_quantile_range(self):
        """Test that invalid robust quantile range is rejected."""
        with pytest.raises(ValueError, match="robust_quantile_range"):
            NeuralReadyConfig(robust_quantile_range=(75.0, 25.0))

    def test_invalid_minmax_range_length(self):
        """Test that invalid minmax range length is rejected."""
        with pytest.raises(ValueError, match="minmax_range must be"):
            NeuralReadyConfig(minmax_range=(0.0,))

    def test_invalid_outlier_quantile_range(self):
        """Test that invalid outlier quantile range is rejected."""
        with pytest.raises(ValueError, match="outlier_quantile_range"):
            NeuralReadyConfig(outlier_quantile_range=(99.0, 1.0))


class TestBuildNeuralReady:
    """Test neural-ready feature building."""

    def test_robust_scaling(self):
        """Test robust scaling using median and IQR."""
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
        config = NeuralReadyConfig(scaling="robust", handle_missing="zero", clip_outliers=False)
        result = build_neural_ready(values, config)

        # Check that scaling was applied
        assert result.scaling_params["method"] == "robust"
        assert "medians" in result.scaling_params
        assert "iqr" in result.scaling_params

        # Median of column 0 should be 3.0
        assert result.scaling_params["medians"][0] == 3.0

    def test_minmax_scaling(self):
        """Test min-max scaling."""
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
        config = NeuralReadyConfig(
            scaling="minmax", minmax_range=(0.0, 1.0), handle_missing="zero", clip_outliers=False
        )
        result = build_neural_ready(values, config)

        # Check that values are in [0, 1] range
        assert np.min(result.X) >= 0.0
        assert np.max(result.X) <= 1.0
        assert result.scaling_params["method"] == "minmax"

    def test_standard_scaling(self):
        """Test standard scaling (z-score)."""
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
        config = NeuralReadyConfig(scaling="standard", handle_missing="zero", clip_outliers=False)
        result = build_neural_ready(values, config)

        # Each column should have mean ~0 and std ~1
        for col in range(result.X.shape[1]):
            col_data = result.X[:, col]
            np.testing.assert_allclose(np.mean(col_data), 0.0, atol=1e-10)
            np.testing.assert_allclose(np.std(col_data, ddof=1), 1.0, atol=1e-10)

        assert result.scaling_params["method"] == "standard"

    def test_no_scaling(self):
        """Test with no scaling."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = NeuralReadyConfig(scaling="none", handle_missing="zero", clip_outliers=False)
        result = build_neural_ready(values, config)

        np.testing.assert_array_equal(result.X, values)
        assert result.scaling_params["method"] == "none"

    def test_zero_fill_missing(self):
        """Test zero-filling missing values."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = NeuralReadyConfig(scaling="none", handle_missing="zero", clip_outliers=False)
        result = build_neural_ready(values, config)

        assert not np.any(np.isnan(result.X))
        assert result.X[1, 0] == 0.0

    def test_mean_fill_missing(self):
        """Test mean-filling missing values."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = NeuralReadyConfig(scaling="none", handle_missing="mean", clip_outliers=False)
        result = build_neural_ready(values, config)

        # Mean of [1, 5] is 3.0
        assert result.X[1, 0] == 3.0

    def test_forward_fill_missing(self):
        """Test forward-filling missing values."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [np.nan, 6.0], [5.0, 8.0]])
        config = NeuralReadyConfig(scaling="none", handle_missing="forward_fill", clip_outliers=False)
        result = build_neural_ready(values, config)

        # Row 1 and 2 should be forward-filled with 1.0
        assert result.X[1, 0] == 1.0
        assert result.X[2, 0] == 1.0
        assert result.X[3, 0] == 5.0

    def test_clip_outliers(self):
        """Test outlier clipping."""
        values = np.array([
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
            [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [100.0, 11.0]
        ])
        config = NeuralReadyConfig(
            scaling="none",
            clip_outliers=True,
            outlier_quantile_range=(5.0, 95.0),
            handle_missing="zero",
        )
        result = build_neural_ready(values, config)

        # Extreme value should be clipped
        assert result.X[9, 0] < 100.0
        assert "n_values_clipped" in result.preprocessing_stats
        assert result.preprocessing_stats["n_values_clipped"] > 0

    def test_no_outlier_clipping(self):
        """Test without outlier clipping."""
        values = np.array([[1.0, 2.0], [2.0, 3.0], [100.0, 4.0]])
        config = NeuralReadyConfig(scaling="none", clip_outliers=False, handle_missing="zero")
        result = build_neural_ready(values, config)

        assert result.X[2, 0] == 100.0

    def test_per_sample_normalization(self):
        """Test per-sample normalization."""
        values = np.array([[3.0, 4.0], [1.0, 0.0], [5.0, 12.0]])
        config = NeuralReadyConfig(
            scaling="none",
            normalize_per_sample=True,
            handle_missing="zero",
            clip_outliers=False,
        )
        result = build_neural_ready(values, config)

        # Each row should have unit L2 norm
        for i in range(result.X.shape[0]):
            row_norm = np.linalg.norm(result.X[i, :])
            np.testing.assert_allclose(row_norm, 1.0, atol=1e-10)

    def test_no_per_sample_normalization(self):
        """Test without per-sample normalization."""
        values = np.array([[3.0, 4.0], [1.0, 0.0]])
        config = NeuralReadyConfig(
            scaling="none",
            normalize_per_sample=False,
            handle_missing="zero",
            clip_outliers=False,
        )
        result = build_neural_ready(values, config)

        # Row norm should not be 1
        row_norm = np.linalg.norm(result.X[0, :])
        assert row_norm == 5.0  # sqrt(3^2 + 4^2) = 5

    def test_feature_names_default(self):
        """Test default feature names."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = NeuralReadyConfig()
        result = build_neural_ready(values, config)

        assert result.feature_names == ["f0", "f1"]

    def test_feature_names_custom(self):
        """Test custom feature names."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = NeuralReadyConfig()
        result = build_neural_ready(values, config, feature_names=["alpha", "beta"])

        assert result.feature_names == ["alpha", "beta"]

    def test_feature_names_length_mismatch(self):
        """Test that feature_names length mismatch raises ValueError."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = NeuralReadyConfig()

        with pytest.raises(ValueError, match="feature_names length"):
            build_neural_ready(values, config, feature_names=["alpha"])

    def test_constant_column_handling(self):
        """Test that constant columns are handled gracefully."""
        values = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        config = NeuralReadyConfig(scaling="robust")
        result = build_neural_ready(values, config)

        # Constant column should not cause division by zero
        assert np.all(np.isfinite(result.X))

    def test_empty_values_rejected(self):
        """Test that empty values are rejected."""
        values = np.array([]).reshape(0, 2)
        config = NeuralReadyConfig()

        with pytest.raises(ValueError, match="cannot be empty"):
            build_neural_ready(values, config)

    def test_invalid_ndim(self):
        """Test that non-2D arrays are rejected."""
        values = np.array([1.0, 2.0, 3.0])
        config = NeuralReadyConfig()

        with pytest.raises(ValueError, match="must be 2D"):
            build_neural_ready(values, config)

    def test_validate_method(self):
        """Test validate method."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = NeuralReadyConfig()
        result = build_neural_ready(values, config)

        assert result.validate() is True

    def test_preprocessing_stats_tracking(self):
        """Test that preprocessing statistics are tracked."""
        values = np.array([[1.0, np.nan], [np.nan, 4.0], [5.0, 6.0]])
        config = NeuralReadyConfig(handle_missing="zero", clip_outliers=False)
        result = build_neural_ready(values, config)

        assert "n_missing_original" in result.preprocessing_stats
        assert result.preprocessing_stats["n_missing_original"] == 2
        assert "n_missing_final" in result.preprocessing_stats
        assert result.preprocessing_stats["n_missing_final"] == 0

    def test_all_nan_column_handling(self):
        """All-NaN column under mean fill must fail closed, not fake-zero."""
        values = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        config = NeuralReadyConfig(handle_missing="mean", clip_outliers=False)

        with pytest.raises(ValueError, match="all-NaN columns"):
            build_neural_ready(values, config)

    def test_all_nan_column_forward_fill_fails_closed(self):
        """All-NaN column under forward_fill must fail closed, not fake-zero."""
        values = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        config = NeuralReadyConfig(handle_missing="forward_fill", clip_outliers=False)

        with pytest.raises(ValueError, match="all-NaN columns"):
            build_neural_ready(values, config)

    def test_clip_outliers_inf_uses_finite_bounds(self):
        """Clip bounds must come from finite values only; inf gets clipped."""
        values = np.array([
            [1.0], [2.0], [3.0], [4.0], [5.0],
            [6.0], [7.0], [8.0], [9.0], [np.inf],
        ])
        config = NeuralReadyConfig(
            handle_missing="zero",
            clip_outliers=True,
            outlier_quantile_range=(10.0, 90.0),
            scaling="none",
        )
        result = build_neural_ready(values, config)

        assert np.all(np.isfinite(result.X))
        # The inf row must have been clipped to the finite upper bound.
        upper = result.preprocessing_stats["clip_bounds"][0]["upper"]
        assert result.X[-1, 0] == upper

    def test_clip_outliers_all_inf_column_skipped(self):
        """A column with no finite values is skipped, not turned into NaN bounds."""
        values = np.array([[1.0, np.inf], [2.0, -np.inf], [3.0, np.inf]])
        config = NeuralReadyConfig(
            # forward_fill is a no-op here (no NaN); "zero" would run
            # nan_to_num, which converts ±inf to ±float-max before clipping.
            handle_missing="forward_fill",
            clip_outliers=True,
            scaling="none",
        )
        result = build_neural_ready(values, config)

        # Column 1 had no finite values: no clip bounds recorded, untouched.
        assert len(result.preprocessing_stats["clip_bounds"]) == 1
        assert np.isinf(result.X[:, 1]).all()


class TestPrepareEmbeddings:
    """Test embedding preparation for categorical features."""

    def test_basic_embedding_preparation(self):
        """Test basic embedding configuration."""
        categorical_features = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 0.0], [3.0, 2.0]])
        result = prepare_embeddings(categorical_features)

        assert result["n_categorical_features"] == 2
        assert len(result["embedding_configs"]) == 2

        # First feature has 3 unique values (1, 2, 3)
        assert result["embedding_configs"][0]["cardinality"] == 3

        # Second feature has 3 unique values (0, 1, 2)
        assert result["embedding_configs"][1]["cardinality"] == 3

    def test_embedding_dimension_heuristic(self):
        """Test embedding dimension heuristic."""
        # High cardinality feature
        categorical_features = np.arange(100).reshape(-1, 1).astype(float)
        result = prepare_embeddings(categorical_features)

        # With 100 unique values, suggested dim should be min(50, (100+1)//2) = 50
        assert result["embedding_configs"][0]["embedding_dim"] == 50

    def test_custom_embedding_dimension(self):
        """Test custom embedding dimension."""
        categorical_features = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
        result = prepare_embeddings(categorical_features, embedding_dim=10)

        # All features should use the specified embedding dimension
        for config in result["embedding_configs"]:
            assert config["embedding_dim"] == 10

    def test_nan_handling_in_embeddings(self):
        """Test that NaN values are excluded from cardinality."""
        categorical_features = np.array([[1.0, 0.0], [np.nan, 1.0], [2.0, 0.0], [1.0, np.nan]])
        result = prepare_embeddings(categorical_features)

        # First feature has 2 unique values (1, 2), excluding NaN
        assert result["embedding_configs"][0]["cardinality"] == 2

        # Second feature has 2 unique values (0, 1), excluding NaN
        assert result["embedding_configs"][1]["cardinality"] == 2

    def test_total_embedding_params(self):
        """Test total embedding parameters calculation."""
        categorical_features = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
        result = prepare_embeddings(categorical_features, embedding_dim=5)

        # Feature 0: 3 unique values * 5 dim = 15 params
        # Feature 1: 3 unique values * 5 dim = 15 params
        # Total: 30 params
        assert result["total_embedding_params"] == 30

    def test_invalid_ndim(self):
        """Test that non-2D arrays are rejected."""
        categorical_features = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be 2D"):
            prepare_embeddings(categorical_features)

    def test_single_categorical_feature(self):
        """Test with single categorical feature."""
        categorical_features = np.array([[1.0], [2.0], [3.0], [1.0]])
        result = prepare_embeddings(categorical_features)

        assert result["n_categorical_features"] == 1
        assert result["embedding_configs"][0]["cardinality"] == 3

    def test_low_cardinality_embedding_dim(self):
        """Test embedding dimension for low cardinality features."""
        categorical_features = np.array([[0.0], [1.0], [0.0], [1.0]])
        result = prepare_embeddings(categorical_features)

        # With 2 unique values, suggested dim should be (2+1)//2 = 1
        assert result["embedding_configs"][0]["embedding_dim"] == 1
