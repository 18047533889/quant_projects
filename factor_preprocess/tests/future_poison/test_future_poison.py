"""
Future-poison detection tests.

These tests verify that rolling transforms do not leak future information.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.rolling import (
    rolling_mean,
    rolling_std,
    rolling_zscore,
)


@pytest.mark.future_poison
class TestFuturePoisonDetection:
    """Test suite for temporal leakage detection."""

    def test_rolling_mean_excludes_current(self):
        """Test that rolling_mean excludes current observation."""
        # Create data with a known pattern
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [10.0, 10.0, 10.0, 10.0, 100.0],  # Spike at end
        })

        result = rolling_mean(df, window=2)

        # At index 4, rolling mean should be of values at indices 2 and 3
        # which are both 10.0, so mean should be 10.0
        # If current (100.0) were included, mean would be much higher
        expected_at_4 = 10.0
        np.testing.assert_allclose(result.iloc[4], expected_at_4)

    def test_rolling_std_excludes_current(self):
        """Test that rolling_std excludes current observation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 100.0],  # Spike at end
        })

        result = rolling_std(df, window=2)

        # At index 4, std should be of values at indices 2 and 3: [3.0, 4.0]
        # std([3.0, 4.0]) = 0.707...
        # If current (100.0) were included, std would be much larger
        expected_std = np.std([3.0, 4.0], ddof=1)
        np.testing.assert_allclose(result.iloc[4], expected_std)

    def test_rolling_zscore_uses_lagged_stats(self):
        """Test that rolling_zscore uses lagged statistics."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],  # Spike at end
        })

        result = rolling_zscore(df, window=3)

        # At index 5, stats should be from indices 2, 3, 4: [3.0, 4.0, 5.0]
        # mean = 4.0, std = 1.0
        # zscore = (100 - 4) / 1 = 96.0
        mean_345 = np.mean([3.0, 4.0, 5.0])
        std_345 = np.std([3.0, 4.0, 5.0], ddof=1)
        expected_zscore = (100.0 - mean_345) / std_345

        np.testing.assert_allclose(result.iloc[5], expected_zscore, rtol=1e-5)

        # If current were included in stats, zscore would be much smaller

    def test_no_forward_looking_values(self):
        """Test that values at time t do not affect values at time t-1."""
        # Create data where future changes should not affect past
        df1 = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, 2.0, 3.0, 4.0],
        })

        df2 = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, 2.0, 3.0, 100.0],  # Different last value
        })

        result1 = rolling_mean(df1, window=2)
        result2 = rolling_mean(df2, window=2)

        # Values at indices 0, 1, 2 should be identical
        # (they should not be affected by the difference at index 3)
        np.testing.assert_array_equal(result1.iloc[:3], result2.iloc[:3])

    def test_asset_isolation_prevents_leakage(self):
        """Test that assets do not leak information to each other."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        })

        result = rolling_mean(df, window=2)

        # Asset A's rolling mean should not be affected by Asset B's values
        # At index 2 (A's third observation): mean of [1.0, 2.0] = 1.5
        assert result.iloc[2] == 1.5

        # At index 5 (B's third observation): mean of [100.0, 200.0] = 150.0
        assert result.iloc[5] == 150.0

    def test_warmup_period_is_nan(self):
        """Test that warmup period produces NaN (no peeking)."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = rolling_mean(df, window=3)

        # First 3 should be NaN (window=3 requires 3 past observations)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

        # Fourth onward should have values
        assert pd.notna(result.iloc[3])
        assert pd.notna(result.iloc[4])
