"""
Test suite for linear-ready feature preparation.
"""
import pytest
import numpy as np
from factor_preprocess.representation.linear_ready import (
    build_linear_ready,
    assess_collinearity,
    LinearReadyConfig,
    LinearReadyResult,
)


class TestLinearReadyConfig:
    """Test LinearReadyConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = LinearReadyConfig(fill_method="zero", standardize=True)
        assert config.fill_method == "zero"
        assert config.standardize is True

    def test_invalid_fill_method(self):
        """Test that invalid fill_method is rejected."""
        with pytest.raises(ValueError, match="fill_method must be one of"):
            LinearReadyConfig(fill_method="invalid")


class TestBuildLinearReady:
    """Test linear-ready feature building."""

    def test_basic_zero_fill(self):
        """Test basic zero-fill for missing values."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = LinearReadyConfig(fill_method="zero", standardize=False)
        result = build_linear_ready(values, config)

        assert result.X.shape == (3, 2)
        assert not np.any(np.isnan(result.X))
        assert result.X[1, 0] == 0.0

    def test_mean_fill(self):
        """Test mean-fill for missing values."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = LinearReadyConfig(fill_method="mean", standardize=False)
        result = build_linear_ready(values, config)

        # Column 0 mean (excluding NaN): (1 + 5) / 2 = 3.0
        assert result.X[1, 0] == 3.0

    def test_drop_rows_with_nan(self):
        """Test dropping rows with any NaN."""
        values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        config = LinearReadyConfig(fill_method="drop", standardize=False)
        result = build_linear_ready(values, config)

        assert result.X.shape == (2, 2)
        assert result.n_samples == 2
        assert "n_rows_dropped" in result.fill_stats

    def test_drop_all_rows_raises(self):
        """Test that dropping all rows raises ValueError."""
        values = np.array([[np.nan, 2.0], [1.0, np.nan], [np.nan, np.nan]])
        config = LinearReadyConfig(fill_method="drop", standardize=False)

        with pytest.raises(ValueError, match="All rows contain NaN"):
            build_linear_ready(values, config)

    def test_standardization(self):
        """Test standardization to mean=0, std=1."""
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        config = LinearReadyConfig(fill_method="zero", standardize=True)
        result = build_linear_ready(values, config)

        # Each column should have mean ~0 and std ~1
        for col in range(result.X.shape[1]):
            col_data = result.X[:, col]
            np.testing.assert_allclose(np.mean(col_data), 0.0, atol=1e-10)
            np.testing.assert_allclose(np.std(col_data, ddof=1), 1.0, atol=1e-10)

    def test_add_intercept(self):
        """Test adding intercept column."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = LinearReadyConfig(add_intercept=True, standardize=False)
        result = build_linear_ready(values, config)

        assert result.X.shape == (2, 3)
        assert result.n_features == 3
        assert result.has_intercept is True
        np.testing.assert_array_equal(result.X[:, 0], np.ones(2))

    def test_feature_names_default(self):
        """Test default feature names."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = LinearReadyConfig(standardize=False)
        result = build_linear_ready(values, config)

        assert result.feature_names == ["f0", "f1"]

    def test_feature_names_custom(self):
        """Test custom feature names."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = LinearReadyConfig(standardize=False)
        result = build_linear_ready(values, config, feature_names=["alpha", "beta"])

        assert result.feature_names == ["alpha", "beta"]

    def test_feature_names_with_intercept(self):
        """Test feature names with intercept."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = LinearReadyConfig(add_intercept=True, standardize=False)
        result = build_linear_ready(values, config, feature_names=["alpha", "beta"])

        assert result.feature_names == ["intercept", "alpha", "beta"]

    def test_feature_names_length_mismatch(self):
        """Test that feature_names length mismatch raises ValueError."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = LinearReadyConfig(standardize=False)

        with pytest.raises(ValueError, match="feature_names length"):
            build_linear_ready(values, config, feature_names=["alpha"])

    def test_constant_column_standardization(self):
        """Test that constant columns are handled gracefully."""
        values = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        config = LinearReadyConfig(standardize=True)
        result = build_linear_ready(values, config)

        # Constant column should not cause division by zero
        assert np.all(np.isfinite(result.X))

    def test_empty_values_rejected(self):
        """Test that empty values are rejected."""
        values = np.array([]).reshape(0, 2)
        config = LinearReadyConfig()

        with pytest.raises(ValueError, match="cannot be empty"):
            build_linear_ready(values, config)

    def test_invalid_ndim(self):
        """Test that non-2D arrays are rejected."""
        values = np.array([1.0, 2.0, 3.0])
        config = LinearReadyConfig()

        with pytest.raises(ValueError, match="must be 2D"):
            build_linear_ready(values, config)

    def test_validate_method(self):
        """Test validate method."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = LinearReadyConfig(standardize=False)
        result = build_linear_ready(values, config)

        assert result.validate() is True

    def test_fill_stats_tracking(self):
        """Test that fill statistics are tracked."""
        values = np.array([[1.0, np.nan], [np.nan, 4.0], [5.0, 6.0]])
        config = LinearReadyConfig(fill_method="zero", standardize=False)
        result = build_linear_ready(values, config)

        assert "n_missing_before" in result.fill_stats
        assert result.fill_stats["n_missing_before"] == 2
        assert "n_missing_after" in result.fill_stats
        assert result.fill_stats["n_missing_after"] == 0


class TestAssessCollinearity:
    """Test collinearity assessment."""

    def test_no_collinearity(self):
        """Test features with no collinearity."""
        X = np.array([[1.0, 0.0], [2.0, 0.5], [3.0, -0.5]])
        result = assess_collinearity(X, threshold=0.99)

        assert result["has_collinearity"] is False
        assert result["n_high_corr_pairs"] == 0

    def test_high_collinearity(self):
        """Test features with high collinearity."""
        X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        result = assess_collinearity(X, threshold=0.99)

        assert result["has_collinearity"] is True
        assert result["n_high_corr_pairs"] > 0
        assert result["max_correlation"] >= 0.99

    def test_invalid_ndim(self):
        """Test that non-2D arrays are rejected."""
        X = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be 2D"):
            assess_collinearity(X)

    def test_small_matrix(self):
        """Test behavior with small matrices."""
        X = np.array([[1.0]])
        result = assess_collinearity(X)

        assert result["has_collinearity"] is False
        assert result["max_correlation"] == 0.0

    def test_custom_threshold(self):
        """Test custom correlation threshold."""
        X = np.array([[1.0, 0.9], [2.0, 1.8], [3.0, 2.7]])
        result = assess_collinearity(X, threshold=0.95)

        # These should be highly correlated
        assert result["max_correlation"] > 0.95

    def test_constant_column_undefined_pairs_reported(self):
        """A constant column gives NaN correlations — must be counted, not hidden."""
        X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        result = assess_collinearity(X, threshold=0.99)

        # Column 0 is constant: its correlation with column 1 is NaN.
        assert result["n_undefined_pairs"] == 1
        assert result["n_high_corr_pairs"] == 0
        # max_correlation must ignore the NaN pair, not become NaN.
        assert np.isfinite(result["max_correlation"])
