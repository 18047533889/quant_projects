"""
Test suite for cross-sectional transforms.
"""
import pytest
import numpy as np
from factor_preprocess.transforms.cross_sectional import (
    cs_rank,
    cs_zscore,
    cs_demean,
    cs_winsor,
    cs_scale,
)


class TestCsRank:
    """Test cross-sectional ranking."""

    def test_basic_ranking(self):
        """Test basic ranking behavior."""
        values = np.array([3.0, 1.0, 2.0])
        result = cs_rank(values)
        expected = np.array([3.0, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_average_ties(self):
        """Test average-tie ranking."""
        values = np.array([1.0, 2.0, 2.0, 3.0])
        result = cs_rank(values, method="average")
        # Ranks: 1, 2.5, 2.5, 4
        expected = np.array([1.0, 2.5, 2.5, 4.0])
        np.testing.assert_array_equal(result, expected)

    def test_nan_preservation(self):
        """Test that NaN inputs produce NaN outputs."""
        values = np.array([1.0, np.nan, 3.0, 2.0])
        result = cs_rank(values)
        assert np.isnan(result[1])
        # Other ranks should be 1, 3, 2
        assert result[0] == 1.0
        assert result[2] == 3.0
        assert result[3] == 2.0

    def test_percentile_ranks(self):
        """Test percentile ranking."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = cs_rank(values, pct=True)
        expected = np.array([0.0, 1/3, 2/3, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_constant_values(self):
        """Test constant values produce mid-rank."""
        values = np.array([5.0, 5.0, 5.0])
        result = cs_rank(values, method="average")
        # All get average rank (1+2+3)/3 = 2
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_2d_array(self):
        """Test ranking along axis in 2D array."""
        values = np.array([
            [3.0, 1.0, 2.0],
            [1.0, 3.0, 2.0],
        ])
        result = cs_rank(values, axis=1)
        expected = np.array([
            [3.0, 1.0, 2.0],
            [1.0, 3.0, 2.0],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test empty array returns empty."""
        values = np.array([])
        result = cs_rank(values)
        assert result.size == 0


class TestCsZscore:
    """Test cross-sectional z-score."""

    def test_basic_zscore(self):
        """Test basic z-score normalization."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cs_zscore(values)
        # Should have mean ~0 and std ~1
        np.testing.assert_allclose(np.nanmean(result), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.nanstd(result, ddof=1), 1.0, atol=1e-10)

    def test_nan_preservation(self):
        """Test that NaN inputs produce NaN outputs."""
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = cs_zscore(values)
        assert np.isnan(result[1])
        # Others should be finite
        assert np.all(np.isfinite(result[[0, 2, 3, 4]]))

    def test_constant_values(self):
        """Test constant values produce constant_value."""
        values = np.array([5.0, 5.0, 5.0])
        result = cs_zscore(values, constant_value=0.0)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_2d_array(self):
        """Test z-score along axis in 2D array."""
        values = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        result = cs_zscore(values, axis=1)
        # Each row should have mean ~0 and std ~1
        for i in range(result.shape[0]):
            np.testing.assert_allclose(np.nanmean(result[i]), 0.0, atol=1e-10)
            np.testing.assert_allclose(np.nanstd(result[i], ddof=1), 1.0, atol=1e-10)


class TestCsDemean:
    """Test cross-sectional demeaning."""

    def test_basic_demean(self):
        """Test basic demeaning."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cs_demean(values)
        # Should have mean ~0
        np.testing.assert_allclose(np.nanmean(result), 0.0, atol=1e-10)

    def test_nan_preservation(self):
        """Test that NaN inputs produce NaN outputs."""
        values = np.array([1.0, np.nan, 3.0, 4.0])
        result = cs_demean(values)
        assert np.isnan(result[1])

    def test_2d_array(self):
        """Test demeaning along axis in 2D array."""
        values = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        result = cs_demean(values, axis=1)
        # Each row should have mean ~0
        for i in range(result.shape[0]):
            np.testing.assert_allclose(np.nanmean(result[i]), 0.0, atol=1e-10)


class TestCsWinsor:
    """Test cross-sectional winsorization."""

    def test_basic_winsor(self):
        """Test basic winsorization."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = cs_winsor(values, lower=0.1, upper=0.9)
        # 10th percentile clips lower values, 90th percentile clips upper values
        # Result should have clipped extremes
        assert result[0] > values[0]  # 1.0 should be clipped up
        assert result[-1] < values[-1]  # 10.0 should be clipped down

    def test_nan_preservation(self):
        """Test that NaN inputs produce NaN outputs."""
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = cs_winsor(values)
        assert np.isnan(result[1])

    def test_invalid_quantiles(self):
        """Test that invalid quantiles are rejected."""
        values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Invalid quantiles"):
            cs_winsor(values, lower=0.9, upper=0.1)

    def test_2d_array(self):
        """Test winsorization along axis in 2D array."""
        values = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        result = cs_winsor(values, lower=0.2, upper=0.8, axis=1)
        # Each row should have min/max clipped
        assert result.shape == values.shape


class TestCsScale:
    """Test cross-sectional scaling."""

    def test_basic_scale(self):
        """Test basic scaling to target std."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cs_scale(values, target_std=2.0)
        # Should have std ~2.0
        np.testing.assert_allclose(np.nanstd(result, ddof=1), 2.0, atol=1e-10)

    def test_constant_values(self):
        """Test constant values remain constant."""
        values = np.array([5.0, 5.0, 5.0])
        result = cs_scale(values)
        np.testing.assert_array_equal(result, values)

    def test_nan_preservation(self):
        """Test that NaN inputs produce NaN outputs."""
        values = np.array([1.0, np.nan, 3.0, 4.0])
        result = cs_scale(values)
        assert np.isnan(result[1])

    def test_2d_array(self):
        """Test scaling along axis in 2D array."""
        values = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        result = cs_scale(values, axis=1, target_std=1.0)
        # Each row should have std ~1.0
        for i in range(result.shape[0]):
            np.testing.assert_allclose(np.nanstd(result[i], ddof=1), 1.0, atol=1e-10)
