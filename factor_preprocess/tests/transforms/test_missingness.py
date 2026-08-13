"""
Test suite for missingness transforms.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.missingness import (
    forward_fill,
    missing_indicator,
    missing_run_length,
    missing_rate,
    linear_interpolate,
    time_weighted_interpolate,
    impute_with_fallback,
)


class TestForwardFill:
    """Test forward fill transform."""

    def test_basic_forward_fill(self):
        """Test basic forward fill."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, np.nan, np.nan, 2.0, np.nan],
        })

        result = forward_fill(df)

        expected = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_forward_fill_with_max_lag(self):
        """Test forward fill with bounded lag."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0],
        })

        result = forward_fill(df, max_lag=2)

        # Should fill up to 2 periods, then stop
        expected = pd.Series([1.0, 1.0, 1.0, np.nan, np.nan, 2.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_no_prior_value(self):
        """Test that NaN remains if no prior value exists."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [np.nan, np.nan, 1.0],
        })

        result = forward_fill(df)

        # First two should remain NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 1.0

    def test_per_asset_isolation(self):
        """Test that assets are processed independently."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [1.0, np.nan, np.nan, np.nan, 10.0, np.nan],
        })

        result = forward_fill(df)

        # Asset A should fill 1.0 forward
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 1.0
        assert result.iloc[2] == 1.0

        # Asset B should have NaN at start, then fill 10.0
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == 10.0
        assert result.iloc[5] == 10.0

    def test_invalid_max_lag(self):
        """Test that invalid max_lag is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, np.nan, 2.0],
        })

        with pytest.raises(ValueError, match="max_lag must be >= 1"):
            forward_fill(df, max_lag=0)

    def test_unsorted_raises_error(self):
        """Test that unsorted data raises error."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.date_range("2020-01-03", periods=3)[::-1],
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="must be sorted"):
            forward_fill(df)


class TestMissingIndicator:
    """Test missing indicator transform."""

    def test_basic_missing_indicator(self):
        """Test basic missing indicator."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, np.nan, 3.0, np.nan, 5.0],
        })

        result = missing_indicator(df)

        expected = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_all_present(self):
        """Test indicator when all values are present."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        result = missing_indicator(df)

        expected = pd.Series([0.0, 0.0, 0.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_all_missing(self):
        """Test indicator when all values are missing."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [np.nan, np.nan, np.nan],
        })

        result = missing_indicator(df)

        expected = pd.Series([1.0, 1.0, 1.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)


class TestMissingRunLength:
    """Test missing run length transform."""

    def test_basic_run_length(self):
        """Test basic run length computation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 8,
            "date": pd.date_range("2020-01-01", periods=8),
            "value": [1.0, np.nan, np.nan, np.nan, 2.0, np.nan, 3.0, np.nan],
        })

        result = missing_run_length(df)

        expected = pd.Series([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_no_missing_values(self):
        """Test run length when no missing values."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        result = missing_run_length(df)

        expected = pd.Series([0.0, 0.0, 0.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_all_missing(self):
        """Test run length when all missing."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [np.nan, np.nan, np.nan, np.nan],
        })

        result = missing_run_length(df)

        expected = pd.Series([1.0, 2.0, 3.0, 4.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_per_asset_isolation(self):
        """Test that run length resets per asset."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [np.nan, np.nan, 1.0, 2.0, np.nan, np.nan],
        })

        result = missing_run_length(df)

        # Asset A: two missing then present
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == 0.0

        # Asset B: present then two missing
        assert result.iloc[3] == 0.0
        assert result.iloc[4] == 1.0
        assert result.iloc[5] == 2.0


class TestMissingRate:
    """Test missing rate transform."""

    def test_basic_missing_rate(self):
        """Test basic missing rate computation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [1.0, np.nan, np.nan, 2.0, 3.0, 4.0],
        })

        result = missing_rate(df, window=3)

        # At index 3: lagged window is [1.0, NaN, NaN], rate = 2/3
        np.testing.assert_allclose(result.iloc[3], 2.0 / 3.0, rtol=1e-5)

        # At index 4: lagged window is [NaN, NaN, 2.0], rate = 2/3
        np.testing.assert_allclose(result.iloc[4], 2.0 / 3.0, rtol=1e-5)

        # At index 5: lagged window is [NaN, 2.0, 3.0], rate = 1/3
        np.testing.assert_allclose(result.iloc[5], 1.0 / 3.0, rtol=1e-5)

    def test_excludes_current_observation(self):
        """Test that missing rate excludes current observation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, np.nan],  # Missing at end
        })

        result = missing_rate(df, window=3)

        # At index 4: lagged window is [2.0, 3.0, 4.0], rate = 0/3 = 0.0
        # Should NOT include the current NaN
        np.testing.assert_allclose(result.iloc[4], 0.0, atol=1e-10)

    def test_warmup_period(self):
        """Test that warmup period produces NaN."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, np.nan, 2.0, np.nan, 3.0],
        })

        result = missing_rate(df, window=3)

        # First 3 should be NaN (warmup)
        assert pd.isna(result.iloc[0])
        # Note: iloc[1] and [2] may have values if min_periods allows it
        # After warmup should have values
        assert pd.notna(result.iloc[3])
        assert pd.notna(result.iloc[4])

    def test_invalid_window(self):
        """Test that invalid window is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="window must be >= 1"):
            missing_rate(df, window=0)


@pytest.mark.future_poison
class TestMissingnessFuturePoison:
    """Test that missingness transforms exclude future information."""

    def test_missing_rate_excludes_current(self):
        """Test that missing_rate excludes current observation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, np.nan],  # NaN at end
        })

        result = missing_rate(df, window=3)

        # At index 4, should compute from lagged [2.0, 3.0, 4.0], not include current NaN
        # If current were included, rate would be 1/3 instead of 0/3
        np.testing.assert_allclose(result.iloc[4], 0.0, atol=1e-10)

    def test_forward_fill_no_future_leak(self):
        """Test that forward_fill doesn't use future values."""
        df1 = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, np.nan, np.nan, 2.0],
        })

        df2 = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, np.nan, np.nan, 10.0],  # Different last value
        })

        result1 = forward_fill(df1)
        result2 = forward_fill(df2)

        # First 3 values should be identical (not affected by last value)
        np.testing.assert_array_equal(result1.iloc[:3].values, result2.iloc[:3].values)

    def test_missing_indicator_is_current(self):
        """Test that missing_indicator reflects current state only."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, np.nan, 2.0, np.nan],
        })

        result = missing_indicator(df)

        # Should exactly match current state, no temporal dependencies
        expected = pd.Series([0.0, 1.0, 0.0, 1.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)


class TestLinearInterpolate:
    """Test linear interpolation transform."""

    def test_basic_linear_interpolate(self):
        """Test basic linear interpolation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, np.nan, np.nan, 4.0, 5.0],
        })

        result = linear_interpolate(df)

        # Should interpolate linearly: 1.0 -> 2.0 -> 3.0 -> 4.0
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_interpolate_with_max_gap(self):
        """Test interpolation with bounded gap size."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 7,
            "date": pd.date_range("2020-01-01", periods=7),
            "value": [1.0, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0],
        })

        result = linear_interpolate(df, max_gap=2)

        # Should only fill gaps up to 2 consecutive NaNs
        # Gap from index 1-4 is 4 NaNs, too large - but 'limit' parameter
        # means max number of consecutive NaNs to fill, not gap size
        # So with limit=2, first 2 NaNs after 1.0 get filled
        assert result.iloc[0] == 1.0
        assert pd.notna(result.iloc[1])  # First NaN gets filled
        assert pd.notna(result.iloc[2])  # Second NaN gets filled
        assert pd.isna(result.iloc[3])   # Third NaN remains (exceeds limit)
        assert pd.isna(result.iloc[4])   # Fourth NaN remains
        assert result.iloc[5] == 6.0

    def test_no_extrapolation(self):
        """Test that interpolation does not extrapolate."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [np.nan, 2.0, 3.0, 4.0, np.nan],
        })

        result = linear_interpolate(df)

        # Should not extrapolate at boundaries
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == 3.0
        assert result.iloc[3] == 4.0
        assert pd.isna(result.iloc[4])

    def test_per_asset_isolation(self):
        """Test that assets are interpolated independently."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [1.0, np.nan, 3.0, 10.0, np.nan, 30.0],
        })

        result = linear_interpolate(df)

        # Asset A should interpolate to 2.0
        assert result.iloc[1] == 2.0

        # Asset B should interpolate to 20.0
        assert result.iloc[4] == 20.0


class TestTimeWeightedInterpolate:
    """Test time-weighted interpolation transform."""

    def test_basic_time_weighted_interpolate(self):
        """Test basic time-weighted interpolation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "value": [1.0, np.nan, np.nan, 4.0, 5.0],
        })

        result = time_weighted_interpolate(df)

        # With uniform daily spacing, should match linear interpolation
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_irregular_time_spacing(self):
        """Test interpolation with irregular time spacing."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-10"]),
            "value": [1.0, np.nan, np.nan, 10.0],
        })

        result = time_weighted_interpolate(df)

        # Should interpolate based on actual time distances
        # Day 2: 1 day after day 1, 8 days before day 10 -> closer to 1.0
        # Day 5: 4 days after day 1, 5 days before day 10 -> roughly middle
        assert pd.notna(result.iloc[1])
        assert pd.notna(result.iloc[2])
        assert result.iloc[1] < result.iloc[2]  # Day 2 closer to start
        assert result.iloc[2] < result.iloc[3]  # Day 5 closer to middle

    def test_interpolate_with_max_gap(self):
        """Test time-weighted interpolation with max gap."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "value": [1.0, np.nan, np.nan, np.nan, np.nan, 6.0],
        })

        result = time_weighted_interpolate(df, max_gap=2)

        # Should only fill up to 2 consecutive NaNs
        # The 'limit' parameter controls max consecutive NaNs to fill
        assert result.iloc[0] == 1.0
        assert pd.notna(result.iloc[1])  # First NaN filled
        assert pd.notna(result.iloc[2])  # Second NaN filled
        assert pd.isna(result.iloc[3])   # Third NaN not filled (exceeds limit)
        assert pd.isna(result.iloc[4])   # Fourth NaN not filled
        assert result.iloc[5] == 6.0

    def test_per_asset_isolation(self):
        """Test that assets are interpolated independently."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [1.0, np.nan, 3.0, 10.0, np.nan, 30.0],
        })

        result = time_weighted_interpolate(df)

        # Asset A should interpolate to 2.0
        assert result.iloc[1] == 2.0

        # Asset B should interpolate to 20.0
        assert result.iloc[4] == 20.0


class TestImputeWithFallback:
    """Test imputation with fallback strategies."""

    def test_basic_impute_forward_fill(self):
        """Test basic forward fill imputation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, np.nan, np.nan, 2.0, np.nan],
        })

        result = impute_with_fallback(df, fallback_strategy="forward_fill")

        expected = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_impute_with_zero_fallback(self):
        """Test zero fallback for remaining NaNs."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [np.nan, 1.0, np.nan, 2.0, np.nan],
        })

        result = impute_with_fallback(df, fallback_strategy="zero", max_lag=1)

        # Forward fill with max_lag=1, then fill remaining with 0.0
        expected = pd.Series([0.0, 1.0, 1.0, 2.0, 2.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_impute_with_median_fallback(self):
        """Test median fallback for remaining NaNs."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [np.nan, 1.0, 2.0, 3.0, 4.0, np.nan],
        })

        result = impute_with_fallback(df, fallback_strategy="median", max_lag=0)

        # No forward fill (max_lag=0 means no fill), then use median
        # Median of [1.0, 2.0, 3.0, 4.0] = 2.5
        assert result.iloc[0] == 2.5
        assert result.iloc[5] == 2.5

    def test_impute_with_mean_fallback(self):
        """Test mean fallback for remaining NaNs."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [np.nan, 1.0, 2.0, 3.0, 4.0, np.nan],
        })

        result = impute_with_fallback(df, fallback_strategy="mean", max_lag=0)

        # No forward fill, then use mean
        # Mean of [1.0, 2.0, 3.0, 4.0] = 2.5
        assert result.iloc[0] == 2.5
        assert result.iloc[5] == 2.5

    def test_per_asset_fallback(self):
        """Test that fallback is computed per-asset."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [np.nan, 1.0, 2.0, np.nan, 10.0, 20.0],
        })

        result = impute_with_fallback(df, fallback_strategy="mean", max_lag=0)

        # Asset A mean: 1.5
        assert result.iloc[0] == 1.5

        # Asset B mean: 15.0
        assert result.iloc[3] == 15.0

    def test_invalid_fallback_strategy(self):
        """Test that invalid fallback strategy is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, np.nan, 2.0],
        })

        with pytest.raises(ValueError, match="Unknown fallback_strategy"):
            impute_with_fallback(df, fallback_strategy="invalid")
