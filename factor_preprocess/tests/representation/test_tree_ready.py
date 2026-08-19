"""
Test suite for tree-ready feature preparation.
"""
import pytest
import numpy as np
from factor_preprocess.representation.tree_ready import (
    build_tree_ready,
    suggest_tree_params,
    TreeReadyConfig,
    TreeReadyResult,
)


class TestTreeReadyConfig:
    """Test TreeReadyConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = TreeReadyConfig(handle_missing="keep", add_missing_indicator=True)
        assert config.handle_missing == "keep"
        assert config.add_missing_indicator is True

    def test_invalid_handle_missing(self):
        """Test that invalid handle_missing is rejected."""
        with pytest.raises(ValueError, match="handle_missing must be one of"):
            TreeReadyConfig(handle_missing="invalid")

    def test_invalid_encoding(self):
        """Test that invalid categorical_encoding is rejected."""
        with pytest.raises(ValueError, match="categorical_encoding must be one of"):
            TreeReadyConfig(categorical_encoding="invalid")

    def test_invalid_outlier_threshold(self):
        """Test that negative outlier threshold is rejected."""
        with pytest.raises(ValueError, match="outlier_std_threshold must be positive"):
            TreeReadyConfig(outlier_std_threshold=-1.0)


class TestBuildTreeReady:
    """Test tree-ready feature building."""

    def test_keep_missing(self):
        """Test keeping NaN for native tree handling."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = TreeReadyConfig(handle_missing="keep")
        result = build_tree_ready(values, config)

        assert result.X.shape == (3, 2)
        assert np.isnan(result.X[1, 0])

    def test_flag_missing(self):
        """Test replacing NaN with flag value."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = TreeReadyConfig(handle_missing="flag")
        result = build_tree_ready(values, config)

        assert not np.any(np.isnan(result.X))
        assert result.X[1, 0] == -999.0
        assert "missing_flag_value" in result.preprocessing_stats

    def test_fill_median(self):
        """Test filling missing with median."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = TreeReadyConfig(handle_missing="fill_median")
        result = build_tree_ready(values, config)

        assert not np.any(np.isnan(result.X))
        # Median of [1, 5] is 3.0
        assert result.X[1, 0] == 3.0

    def test_missing_indicator(self):
        """Test adding missing value indicators."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = TreeReadyConfig(handle_missing="keep", add_missing_indicator=True)
        result = build_tree_ready(values, config)

        assert result.missing_indicators is not None
        assert result.missing_indicators.shape == values.shape
        assert result.missing_indicators[1, 0] == 1.0
        assert result.missing_indicators[0, 0] == 0.0

    def test_no_missing_indicator(self):
        """Test without missing indicators."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = TreeReadyConfig(add_missing_indicator=False)
        result = build_tree_ready(values, config)

        assert result.missing_indicators is None

    def test_clip_outliers(self):
        """Test outlier clipping."""
        # Create data with clear outlier
        values = np.array([
            [1.0, 2.0], [1.5, 3.0], [2.0, 4.0], [2.5, 5.0],
            [1.8, 2.5], [2.2, 3.5], [1.9, 4.5], [2.1, 5.5],
            [100.0, 2.0]  # Clear outlier
        ])
        config = TreeReadyConfig(clip_outliers=True, outlier_std_threshold=2.0)
        result = build_tree_ready(values, config)

        # Extreme value should be clipped
        assert result.X[8, 0] < 100.0
        assert "n_values_clipped" in result.preprocessing_stats
        assert result.preprocessing_stats["n_values_clipped"] > 0

    def test_no_outlier_clipping(self):
        """Test without outlier clipping."""
        values = np.array([[1.0, 2.0], [2.0, 3.0], [100.0, 4.0]])
        config = TreeReadyConfig(clip_outliers=False)
        result = build_tree_ready(values, config)

        # Extreme value should remain
        assert result.X[2, 0] == 100.0

    def test_categorical_encoding_none(self):
        """Test with no categorical encoding."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = TreeReadyConfig(categorical_encoding="none")
        result = build_tree_ready(values, config)

        assert result.X.shape == values.shape

    def test_categorical_encoding_not_implemented(self):
        """Test that categorical encoding raises NotImplementedError."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        categorical_mask = np.array([True, False])
        config = TreeReadyConfig(categorical_encoding="ordinal")

        with pytest.raises(NotImplementedError, match="categorical_encoding"):
            build_tree_ready(values, config, categorical_mask=categorical_mask)

    def test_feature_names_default(self):
        """Test default feature names."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = TreeReadyConfig()
        result = build_tree_ready(values, config)

        assert result.feature_names == ["f0", "f1"]

    def test_feature_names_custom(self):
        """Test custom feature names."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = TreeReadyConfig()
        result = build_tree_ready(values, config, feature_names=["alpha", "beta"])

        assert result.feature_names == ["alpha", "beta"]

    def test_feature_names_length_mismatch(self):
        """Test that feature_names length mismatch raises ValueError."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = TreeReadyConfig()

        with pytest.raises(ValueError, match="feature_names length"):
            build_tree_ready(values, config, feature_names=["alpha"])

    def test_empty_values_rejected(self):
        """Test that empty values are rejected."""
        values = np.array([]).reshape(0, 2)
        config = TreeReadyConfig()

        with pytest.raises(ValueError, match="cannot be empty"):
            build_tree_ready(values, config)

    def test_invalid_ndim(self):
        """Test that non-2D arrays are rejected."""
        values = np.array([1.0, 2.0, 3.0])
        config = TreeReadyConfig()

        with pytest.raises(ValueError, match="must be 2D"):
            build_tree_ready(values, config)

    def test_validate_method(self):
        """Test validate method."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = TreeReadyConfig()
        result = build_tree_ready(values, config)

        assert result.validate() is True

    def test_preprocessing_stats_tracking(self):
        """Test that preprocessing statistics are tracked."""
        values = np.array([[1.0, np.nan], [np.nan, 4.0], [5.0, 6.0]])
        config = TreeReadyConfig(handle_missing="flag")
        result = build_tree_ready(values, config)

        assert "n_missing_original" in result.preprocessing_stats
        assert result.preprocessing_stats["n_missing_original"] == 2
        assert "n_missing_final" in result.preprocessing_stats

    def test_all_nan_column_handling(self):
        """All-NaN column under median fill must fail closed, not fake-zero."""
        values = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        config = TreeReadyConfig(handle_missing="fill_median")

        with pytest.raises(ValueError, match="all-NaN columns"):
            build_tree_ready(values, config)


class TestSuggestTreeParams:
    """Test tree hyperparameter suggestions."""

    def test_small_dataset(self):
        """Test suggestions for small dataset."""
        X = np.random.randn(500, 10)
        params = suggest_tree_params(X)

        assert "max_depth" in params
        assert "learning_rate" in params
        assert params["max_depth"] <= 5

    def test_medium_dataset(self):
        """Test suggestions for medium dataset."""
        X = np.random.randn(5000, 20)
        params = suggest_tree_params(X)

        assert params["max_depth"] >= 3
        assert 0.01 <= params["learning_rate"] <= 0.2

    def test_large_dataset(self):
        """Test suggestions for large dataset."""
        X = np.random.randn(50000, 30)
        params = suggest_tree_params(X)

        assert params["max_depth"] >= 5
        assert params["subsample"] < 1.0

    def test_many_features(self):
        """Test suggestions for many features."""
        X = np.random.randn(1000, 100)
        params = suggest_tree_params(X)

        assert params["colsample_bytree"] < 1.0

    def test_few_features(self):
        """Test suggestions for few features."""
        X = np.random.randn(1000, 5)
        params = suggest_tree_params(X)

        assert params["colsample_bytree"] == 1.0

    def test_invalid_ndim(self):
        """Test that non-2D arrays are rejected."""
        X = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be 2D"):
            suggest_tree_params(X)

    def test_includes_notes(self):
        """Test that suggestions include usage notes."""
        X = np.random.randn(1000, 10)
        params = suggest_tree_params(X)

        assert "notes" in params
        assert "heuristic" in params["notes"].lower()
