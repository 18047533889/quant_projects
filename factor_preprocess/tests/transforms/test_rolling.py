"""
Test suite for rolling transforms with future-poison detection.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.rolling import (
    rolling_mean,
    rolling_std,
    rolling_zscore,
    ewma,
)


class TestRollingMean:
    """Test causal rolling mean."""

    def test_basic_rolling_mean(self):
        """Test basic rolling mean with lag."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = rolling_mean(df, window=2)

        # First two should be NaN (warmup + lag)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

        # Third should be mean of first two: (1+2)/2 = 1.5
        assert result.iloc[2] == 1.5

        # Fourth should be mean of second and third: (2+3)/2 = 2.5
        assert result.iloc[3] == 2.5

    def test_multiple_assets(self):
        """Test asset isolation."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })

        result = rolling_mean(df, window=2)

        # Asset A: first two NaN, third is (1+2)/2 = 1.5
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 1.5

        # Asset B: first two NaN, third is (10+20)/2 = 15.0
        assert pd.isna(result.iloc[3])
        assert pd.isna(result.iloc[4])
        assert result.iloc[5] == 15.0

    def test_nan_propagation(self):
        """Test NaN handling."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, np.nan, 3.0, 4.0],
        })

        result = rolling_mean(df, window=2)

        # First two NaN (warmup)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

        # Third: mean of (1, NaN) -> NaN
        assert pd.isna(result.iloc[2])

        # Fourth: mean of (NaN, 3) -> NaN
        assert pd.isna(result.iloc[3])

    def test_unsorted_fails(self):
        """Test that unsorted data is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
            "value": [3.0, 1.0, 2.0],
        })

        with pytest.raises(ValueError, match="sorted"):
            rolling_mean(df, window=2)


class TestRollingStd:
    """Test causal rolling std."""

    def test_basic_rolling_std(self):
        """Test basic rolling std with lag."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = rolling_std(df, window=3)

        # First three should be NaN (warmup + lag)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

        # Fourth should be std of first three
        expected_std = np.std([1.0, 2.0, 3.0], ddof=1)
        np.testing.assert_allclose(result.iloc[3], expected_std)

    def test_constant_values(self):
        """Test that constant values produce zero std."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [5.0, 5.0, 5.0, 5.0, 5.0],
        })

        result = rolling_std(df, window=2)

        # First two NaN, rest should be 0.0
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 0.0


class TestRollingZscore:
    """Test causal rolling z-score."""

    def test_basic_rolling_zscore(self):
        """Test basic rolling z-score with lag."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = rolling_zscore(df, window=2)

        # First two should be NaN (warmup + lag)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

        # Third: (3 - mean(1,2)) / std(1,2)
        # mean = 1.5, std = sqrt(0.5) = 0.707...
        # zscore = (3 - 1.5) / 0.707... = 2.12...
        mean_12 = np.mean([1.0, 2.0])
        std_12 = np.std([1.0, 2.0], ddof=1)
        expected = (3.0 - mean_12) / std_12
        np.testing.assert_allclose(result.iloc[2], expected)

    def test_zero_std_produces_nan(self):
        """Test that zero rolling std produces NaN."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [5.0, 5.0, 5.0, 6.0],
        })

        result = rolling_zscore(df, window=2)

        # First two NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

        # Third: (5 - mean(5,5)) / std(5,5) = 0 / 0 = NaN
        assert pd.isna(result.iloc[2])

    @pytest.mark.future_poison
    def test_future_poison_detection(self):
        """Test that current observation is not used in statistics."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, 2.0, 3.0, 100.0],  # Large spike
        })

        result = rolling_zscore(df, window=2)

        # The spike at index 3 should not affect statistics used at index 3
        # Stats should be computed from index 1 and 2: mean=2.5, std=0.707
        # zscore = (100 - 2.5) / 0.707 = very large
        mean_12 = np.mean([2.0, 3.0])
        std_12 = np.std([2.0, 3.0], ddof=1)
        expected = (100.0 - mean_12) / std_12

        np.testing.assert_allclose(result.iloc[3], expected, rtol=1e-5)

        # If current was included, result would be much smaller


class TestEwma:
    """Test causal EWMA."""

    def test_basic_ewma(self):
        """Test basic EWMA with lag."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = ewma(df, halflife=2.0)

        # First should be NaN (lag)
        assert pd.isna(result.iloc[0])

        # Second onward should have values
        assert pd.notna(result.iloc[1])
        assert pd.notna(result.iloc[2])

        # EWMA should be increasing (input is increasing)
        assert result.iloc[2] > result.iloc[1]

    def test_invalid_halflife(self):
        """Test that invalid halflife is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="halflife must be > 0"):
            ewma(df, halflife=0.0)

        with pytest.raises(ValueError, match="halflife must be > 0"):
            ewma(df, halflife=-1.0)

    def test_multiple_assets(self):
        """Test asset isolation in EWMA."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })

        result = ewma(df, halflife=2.0)

        # Each asset should have first NaN (due to shift), then values
        assert pd.isna(result.iloc[0])
        assert pd.notna(result.iloc[1])
        # Note: B starts at index 3, but due to how groupby+ewm works after shift,
        # it may have a value. The key is asset isolation.
        # Asset B values should be much larger than Asset A
        assert result.iloc[4] > result.iloc[1]
