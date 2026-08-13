"""
Test suite for HP filter trend extraction with causality checks.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.decomposition.trend import (
    hp_filter,
    hp_decompose,
)


class TestHPFilter:
    """Test Hodrick-Prescott filter."""

    def test_basic_hp_filter(self):
        """Test basic HP filter with synthetic trend + cycle."""
        n = 100
        t = np.arange(n)
        trend = 0.5 * t
        cycle = 5 * np.sin(2 * np.pi * t / 20)
        series = trend + cycle

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = hp_filter(df, lambda_param=1600)

        # First should be NaN (causal lag)
        assert pd.isna(result.iloc[0])

        # HP filter should extract smooth trend
        # Check that filtered trend is closer to true trend than raw series
        valid_idx = result.notna()
        extracted_trend = result[valid_idx].values
        true_trend = trend[valid_idx]

        # Correlation with true trend should be high
        corr = np.corrcoef(extracted_trend, true_trend)[0, 1]
        assert corr > 0.95

    def test_multiple_assets(self):
        """Test asset isolation in HP filter."""
        n = 50
        df = pd.DataFrame({
            "asset_id": ["A"] * n + ["B"] * n,
            "date": pd.date_range("2020-01-01", periods=n).tolist() * 2,
            "value": np.concatenate([
                np.arange(n) * 1.0,  # Linear trend for A
                np.arange(n) * 2.0,  # Steeper trend for B
            ]),
        })

        result = hp_filter(df, lambda_param=1600)

        # First of each asset should be NaN
        assert pd.isna(result.iloc[0])  # Asset A
        assert pd.isna(result.iloc[n])  # Asset B

        # Trends should be different between assets
        asset_a_trend = result.iloc[1:n].mean()
        asset_b_trend = result.iloc[n+1:].mean()
        assert asset_b_trend > asset_a_trend * 1.5

    def test_nan_propagation(self):
        """Test that NaN in input produces NaN in output."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "value": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        result = hp_filter(df, lambda_param=1600)

        # First is NaN (lag)
        assert pd.isna(result.iloc[0])
        # HP filter handles internal NaN values by working on valid segments
        # The third position has NaN in lagged input (value[2] shifted to lagged[3])
        assert pd.isna(result.iloc[3])  # Because lagged[3] = value[2] = NaN

    def test_unsorted_fails(self):
        """Test that unsorted data is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
            "value": [3.0, 1.0, 2.0],
        })

        with pytest.raises(ValueError, match="sorted"):
            hp_filter(df, lambda_param=1600)

    def test_invalid_lambda(self):
        """Test that invalid lambda is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "value": np.arange(10) * 1.0,
        })

        with pytest.raises(ValueError, match="lambda_param must be > 0"):
            hp_filter(df, lambda_param=0)

        with pytest.raises(ValueError, match="lambda_param must be > 0"):
            hp_filter(df, lambda_param=-100)

    def test_short_series(self):
        """Test behavior with short time series."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        result = hp_filter(df, lambda_param=1600)

        # Should handle gracefully (may be all NaN for very short series)
        assert len(result) == 3
        assert pd.isna(result.iloc[0])

    @pytest.mark.future_poison
    def test_causality_current_not_used(self):
        """Test that current observation is not used in trend calculation."""
        # Create series with a sudden spike at the end
        n = 50
        series = np.ones(n) * 100.0
        series[-1] = 1000.0  # Large spike

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = hp_filter(df, lambda_param=1600)

        # The trend at the last observation should not be influenced by the spike
        # It should be close to 100.0, not pulled toward 1000.0
        last_trend = result.iloc[-1]
        assert pd.notna(last_trend)
        assert abs(last_trend - 100.0) < 50.0  # Should be much closer to 100 than 1000


class TestHPDecompose:
    """Test HP decomposition into trend and cycle."""

    def test_basic_decomposition(self):
        """Test that decomposition satisfies y = trend + cycle."""
        n = 100
        t = np.arange(n)
        series = 0.5 * t + 5 * np.sin(2 * np.pi * t / 20)

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        trend, cycle = hp_decompose(df, lambda_param=1600)

        # First should be NaN for both
        assert pd.isna(trend.iloc[0])
        assert pd.isna(cycle.iloc[0])

        # Check decomposition identity: lagged_y = trend + cycle
        lagged_y = df.groupby("asset_id", sort=False)["value"].shift(1)
        valid_idx = trend.notna()

        reconstructed = trend[valid_idx] + cycle[valid_idx]
        original = lagged_y[valid_idx]

        np.testing.assert_allclose(reconstructed.values, original.values, rtol=1e-5)

    def test_cycle_properties(self):
        """Test that cycle component has expected properties."""
        n = 100
        t = np.arange(n)
        trend_component = 0.5 * t
        cycle_component = 5 * np.sin(2 * np.pi * t / 20)
        series = trend_component + cycle_component

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        trend, cycle = hp_decompose(df, lambda_param=1600)

        valid_idx = cycle.notna()
        extracted_cycle = cycle[valid_idx].values

        # Cycle should oscillate around zero
        cycle_mean = np.mean(extracted_cycle)
        assert abs(cycle_mean) < 1.0

        # Cycle should have variance (not constant)
        cycle_std = np.std(extracted_cycle)
        assert cycle_std > 0.1

    def test_different_lambda_values(self):
        """Test that different lambda values produce different smoothness."""
        n = 100
        series = np.random.randn(n).cumsum()

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        trend_low, _ = hp_decompose(df, lambda_param=100)
        trend_high, _ = hp_decompose(df, lambda_param=10000)

        # Higher lambda should produce smoother trend (lower variance)
        valid_idx = trend_low.notna() & trend_high.notna()

        var_low = np.var(np.diff(trend_low[valid_idx].values))
        var_high = np.var(np.diff(trend_high[valid_idx].values))

        assert var_high < var_low

    def test_multiple_assets_decomposition(self):
        """Test decomposition with multiple assets."""
        n = 50
        df = pd.DataFrame({
            "asset_id": ["A"] * n + ["B"] * n,
            "date": pd.date_range("2020-01-01", periods=n).tolist() * 2,
            "value": np.concatenate([
                np.arange(n) * 1.0,
                np.arange(n) * 2.0,
            ]),
        })

        trend, cycle = hp_decompose(df, lambda_param=1600)

        # Check that trends are isolated per asset
        asset_a_trend = trend.iloc[1:n].mean()
        asset_b_trend = trend.iloc[n+1:].mean()

        assert asset_b_trend > asset_a_trend * 1.5
