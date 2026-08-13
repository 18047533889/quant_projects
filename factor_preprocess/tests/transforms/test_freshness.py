"""
Test suite for freshness transforms.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.freshness import (
    days_since_update,
    observation_age,
    freshness_score,
    stale_data_indicator,
)


class TestDaysSinceUpdate:
    """Test days since update transform."""

    def test_basic_days_since_update(self):
        """Test basic days since last valid observation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, np.nan, np.nan, np.nan, 2.0],
        })

        result = days_since_update(df)

        # Day 0: valid, days_since = 0
        # Day 1: missing, 1 day since last valid (day 0)
        # Day 2: missing, 2 days since last valid (day 0)
        # Day 3: missing, 3 days since last valid (day 0)
        # Day 4: valid, days_since = 0
        expected = pd.Series([0.0, 1.0, 2.0, 3.0, 0.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_no_prior_valid_observation(self):
        """Test that NaN is returned when no prior valid observation exists."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [np.nan, np.nan, 1.0, 2.0],
        })

        result = days_since_update(df)

        # First two should be NaN (no prior valid observation)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Next two should be 0 (valid observations)
        assert result.iloc[2] == 0.0
        assert result.iloc[3] == 0.0

    def test_per_asset_isolation(self):
        """Test that assets are processed independently."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [1.0, np.nan, np.nan, np.nan, 10.0, np.nan],
        })

        result = days_since_update(df)

        # Asset A: valid, then 1 day, then 2 days
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 1.0
        assert result.iloc[2] == 2.0

        # Asset B: no prior valid, valid, then 1 day
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == 0.0
        assert result.iloc[5] == 1.0

    def test_unsorted_raises_error(self):
        """Test that unsorted data raises error."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.date_range("2020-01-03", periods=3)[::-1],
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="must be sorted"):
            days_since_update(df)


class TestObservationAge:
    """Test observation age transform."""

    def test_basic_observation_age(self):
        """Test basic observation age computation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "observation_date": pd.to_datetime(["2020-01-01", "2019-12-30", "2019-12-25"]),
            "current_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        })

        result = observation_age(df, observation_date_col="observation_date", current_date_col="current_date")

        # Row 0: same day, age = 0
        # Row 1: 3 days between 2019-12-30 and 2020-01-02
        # Row 2: 9 days between 2019-12-25 and 2020-01-03
        expected = pd.Series([0.0, 3.0, 9.0], index=df.index)
        pd.testing.assert_series_equal(result, expected)

    def test_missing_dates_produce_nan(self):
        """Test that missing dates produce NaN."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "observation_date": pd.to_datetime([pd.NaT, "2020-01-01", "2020-01-01"]),
            "current_date": pd.to_datetime(["2020-01-01", pd.NaT, "2020-01-02"]),
        })

        result = observation_age(df, observation_date_col="observation_date", current_date_col="current_date")

        # Row 0: missing observation_date -> NaN
        # Row 1: missing current_date -> NaN
        # Row 2: valid, 1 day difference
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 1.0

    def test_negative_age_for_future_observations(self):
        """Test that future observations produce negative age."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 1,
            "date": pd.date_range("2020-01-01", periods=1),
            "observation_date": pd.to_datetime(["2020-01-05"]),
            "current_date": pd.to_datetime(["2020-01-01"]),
        })

        result = observation_age(df, observation_date_col="observation_date", current_date_col="current_date")

        # Observation is 4 days in the future, age = -4
        assert result.iloc[0] == -4.0


class TestFreshnessScore:
    """Test freshness score transform."""

    def test_basic_freshness_score(self):
        """Test basic freshness score computation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, np.nan, np.nan, 2.0],
        })

        result = freshness_score(df, halflife_days=1.0)

        # Day 0: valid, freshness = 1.0
        # Day 1: 1 day since update, freshness = exp(-1 * ln(2) / 1) = 0.5
        # Day 2: 2 days since update, freshness = exp(-2 * ln(2) / 1) = 0.25
        # Day 3: valid, freshness = 1.0
        expected_1 = np.exp(-1 * np.log(2) / 1.0)  # ~0.5
        expected_2 = np.exp(-2 * np.log(2) / 1.0)  # ~0.25

        assert result.values[0] == 1.0
        np.testing.assert_allclose(result.values[1], expected_1, rtol=1e-5)
        np.testing.assert_allclose(result.values[2], expected_2, rtol=1e-5)
        assert result.values[3] == 1.0

    def test_halflife_property(self):
        """Test that score is 0.5 at halflife."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 11,
            "date": pd.date_range("2020-01-01", periods=11),
            "value": [1.0] + [np.nan] * 10,
        })

        halflife = 5.0
        result = freshness_score(df, halflife_days=halflife)

        # At day 5 (index 5), should be ~0.5
        np.testing.assert_allclose(result.values[5], 0.5, rtol=1e-5)

    def test_no_valid_observation_gives_zero(self):
        """Test that no valid observation gives zero freshness."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [np.nan, np.nan, np.nan],
        })

        result = freshness_score(df, halflife_days=1.0)

        # All should be 0.0 (no valid observation)
        expected = pd.Series([0.0, 0.0, 0.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_invalid_halflife(self):
        """Test that invalid halflife is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="halflife_days must be > 0"):
            freshness_score(df, halflife_days=-1.0)


class TestStaleDataIndicator:
    """Test stale data indicator transform."""

    def test_basic_stale_indicator(self):
        """Test basic stale data indicator."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0],
        })

        result = stale_data_indicator(df, max_days=2)

        # Day 0: fresh (0 days), indicator = 0
        # Day 1: 1 day, indicator = 0
        # Day 2: 2 days, indicator = 0
        # Day 3: 3 days > 2, indicator = 1
        # Day 4: 4 days > 2, indicator = 1
        # Day 5: fresh (0 days), indicator = 0
        expected = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_no_prior_observation_gives_nan(self):
        """Test that no prior observation gives NaN."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [np.nan, np.nan, 1.0],
        })

        result = stale_data_indicator(df, max_days=1)

        # First two should be NaN (no prior observation)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Third should be 0 (fresh)
        assert result.iloc[2] == 0.0

    def test_max_days_zero(self):
        """Test that max_days=0 flags any missing as stale."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, np.nan, 2.0],
        })

        result = stale_data_indicator(df, max_days=0)

        # Day 0: fresh, indicator = 0
        # Day 1: 1 day > 0, indicator = 1
        # Day 2: fresh, indicator = 0
        expected = pd.Series([0.0, 1.0, 0.0], index=df.index, name="value")
        pd.testing.assert_series_equal(result, expected)

    def test_invalid_max_days(self):
        """Test that invalid max_days is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="max_days must be >= 0"):
            stale_data_indicator(df, max_days=-1)


@pytest.mark.future_poison
class TestFreshnessFuturePoison:
    """Test that freshness transforms do not use future information."""

    def test_days_since_update_no_future_leak(self):
        """Test that days_since_update doesn't use future values."""
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

        result1 = days_since_update(df1)
        result2 = days_since_update(df2)

        # First 3 values should be identical (not affected by last value)
        np.testing.assert_array_equal(result1.iloc[:3].values, result2.iloc[:3].values)

    def test_freshness_score_only_depends_on_past(self):
        """Test that freshness score only depends on past observations."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, np.nan, np.nan, np.nan, 100.0],  # Different last value
        })

        result = freshness_score(df, halflife_days=1.0)

        # At index 3, freshness should be based on days since index 0
        # Should be 3 days, score = exp(-3 * ln(2))
        expected_3 = np.exp(-3 * np.log(2) / 1.0)
        np.testing.assert_allclose(result.values[3], expected_3, rtol=1e-5)

        # The value at index 4 (100.0) should not affect index 3
        # Recompute with different value at index 4
        df2 = df.copy()
        df2.loc[df2.index[4], "value"] = np.nan

        result2 = freshness_score(df2, halflife_days=1.0)

        # Index 3 should be identical
        assert result.values[3] == result2.values[3]

    def test_stale_indicator_causal(self):
        """Test that stale indicator is causal."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, np.nan, np.nan, 2.0],
        })

        result = stale_data_indicator(df, max_days=1)

        # At index 2, should be based on days since index 0 (2 days)
        # Should be stale (2 > 1), regardless of value at index 3
        assert result.values[2] == 1.0

    def test_observation_age_is_point_in_time(self):
        """Test that observation_age is point-in-time correct."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "observation_date": pd.to_datetime(["2020-01-01", "2019-12-30", "2019-12-25"]),
            "current_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        })

        result = observation_age(df, observation_date_col="observation_date", current_date_col="current_date")

        # Each row is independent, no temporal dependency
        # Changing any row should not affect others
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 3.0
        assert result.iloc[2] == 9.0
