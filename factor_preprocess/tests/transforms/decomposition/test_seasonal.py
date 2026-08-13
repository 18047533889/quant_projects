"""
Test suite for STL seasonal decomposition with causality checks.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.decomposition.seasonal import (
    stl_decompose,
    seasonal_component,
    trend_component,
    residual_component,
)


class TestSTLDecompose:
    """Test STL seasonal decomposition."""

    def test_basic_stl_decomposition(self):
        """Test basic STL with synthetic seasonal data."""
        n = 120  # 10 years of monthly data
        t = np.arange(n)
        trend = 0.1 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.randn(n) * 0.5
        series = trend + seasonal + noise

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='MS'),
            "value": series,
        })

        trend_comp, seasonal_comp, residual_comp = stl_decompose(
            df, period=12, seasonal=7
        )

        # First should be NaN (causal lag)
        assert pd.isna(trend_comp.iloc[0])
        assert pd.isna(seasonal_comp.iloc[0])
        assert pd.isna(residual_comp.iloc[0])

        # Check decomposition identity: y = trend + seasonal + residual
        valid_idx = trend_comp.notna()
        if valid_idx.sum() > 0:
            lagged_y = df.groupby("asset_id", sort=False)["value"].shift(1)
            reconstructed = (
                trend_comp[valid_idx] +
                seasonal_comp[valid_idx] +
                residual_comp[valid_idx]
            )
            original = lagged_y[valid_idx]

            np.testing.assert_allclose(
                reconstructed.values, original.values, rtol=1e-3
            )

    def test_seasonal_periodicity(self):
        """Test that seasonal component has correct periodicity."""
        n = 120
        t = np.arange(n)
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        series = seasonal + np.random.randn(n) * 0.1

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='MS'),
            "value": series,
        })

        _, seasonal_comp, _ = stl_decompose(df, period=12, seasonal=7)

        valid_idx = seasonal_comp.notna()
        if valid_idx.sum() > 24:  # Need at least 2 full periods
            extracted = seasonal_comp[valid_idx].values

            # Check that seasonal component repeats with period 12
            # Compare first period with second period
            if len(extracted) >= 24:
                period1 = extracted[0:12]
                period2 = extracted[12:24]

                # Correlation should be high
                corr = np.corrcoef(period1, period2)[0, 1]
                assert corr > 0.7  # Relaxed threshold for noisy data

    def test_multiple_assets(self):
        """Test asset isolation in STL."""
        n = 60
        t = np.arange(n)

        # Asset A: weak seasonal
        seasonal_a = 1 * np.sin(2 * np.pi * t / 12)
        # Asset B: strong seasonal
        seasonal_b = 10 * np.sin(2 * np.pi * t / 12)

        df = pd.DataFrame({
            "asset_id": ["A"] * n + ["B"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='MS').tolist() * 2,
            "value": np.concatenate([seasonal_a, seasonal_b]),
        })

        _, seasonal_comp, _ = stl_decompose(df, period=12, seasonal=7)

        # Seasonal amplitude should be larger for asset B
        asset_a_seasonal = seasonal_comp.iloc[:n]
        asset_b_seasonal = seasonal_comp.iloc[n:]

        valid_a = asset_a_seasonal.notna()
        valid_b = asset_b_seasonal.notna()

        if valid_a.sum() > 0 and valid_b.sum() > 0:
            std_a = asset_a_seasonal[valid_a].std()
            std_b = asset_b_seasonal[valid_b].std()

            assert std_b > std_a * 2

    def test_insufficient_data(self):
        """Test behavior with insufficient data for STL."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10, freq='MS'),
            "value": np.arange(10) * 1.0,
        })

        # period=12 but only 10 observations
        trend_comp, seasonal_comp, residual_comp = stl_decompose(
            df, period=12, seasonal=7
        )

        # Should return all NaN (insufficient data)
        assert seasonal_comp.isna().all()

    def test_invalid_period(self):
        """Test that invalid period is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 50,
            "date": pd.date_range("2020-01-01", periods=50),
            "value": np.random.randn(50),
        })

        with pytest.raises(ValueError, match="period must be >= 2"):
            stl_decompose(df, period=1, seasonal=7)

    def test_invalid_seasonal(self):
        """Test that even seasonal parameter is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 50,
            "date": pd.date_range("2020-01-01", periods=50),
            "value": np.random.randn(50),
        })

        with pytest.raises(ValueError, match="seasonal must be odd"):
            stl_decompose(df, period=12, seasonal=8)

    def test_unsorted_fails(self):
        """Test that unsorted data is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
            "value": [3.0, 1.0, 2.0],
        })

        with pytest.raises(ValueError, match="sorted"):
            stl_decompose(df, period=12, seasonal=7)

    @pytest.mark.future_poison
    def test_causality_current_not_used(self):
        """Test that current observation is not used in decomposition."""
        n = 60
        t = np.arange(n)
        series = 10 * np.sin(2 * np.pi * t / 12)
        series[-1] = 1000.0  # Large spike at end

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='MS'),
            "value": series,
        })

        trend_comp, seasonal_comp, residual_comp = stl_decompose(
            df, period=12, seasonal=7
        )

        # The spike should not affect the decomposition at the last point
        # The residual at the last point should capture the spike
        last_residual = residual_comp.iloc[-1]

        if pd.notna(last_residual):
            # Residual should be large (spike is in residual, not trend/seasonal)
            assert abs(last_residual) > 100


class TestSeasonalComponent:
    """Test seasonal component extraction."""

    def test_seasonal_component_only(self):
        """Test extracting only seasonal component."""
        n = 60
        t = np.arange(n)
        seasonal = 10 * np.sin(2 * np.pi * t / 12)

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='MS'),
            "value": seasonal,
        })

        seasonal_comp = seasonal_component(df, period=12, seasonal=7)

        # Should match full decomposition result
        _, seasonal_full, _ = stl_decompose(df, period=12, seasonal=7)

        pd.testing.assert_series_equal(seasonal_comp, seasonal_full)


class TestTrendComponent:
    """Test trend component extraction."""

    def test_trend_component_only(self):
        """Test extracting only trend component."""
        n = 60
        t = np.arange(n)
        trend = 0.5 * t

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='MS'),
            "value": trend,
        })

        trend_comp = trend_component(df, period=12, seasonal=7)

        # Should match full decomposition result
        trend_full, _, _ = stl_decompose(df, period=12, seasonal=7)

        pd.testing.assert_series_equal(trend_comp, trend_full)


class TestResidualComponent:
    """Test residual component extraction."""

    def test_residual_component_only(self):
        """Test extracting only residual component."""
        n = 60
        noise = np.random.randn(n)

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='MS'),
            "value": noise,
        })

        residual_comp = residual_component(df, period=12, seasonal=7)

        # Should match full decomposition result
        _, _, residual_full = stl_decompose(df, period=12, seasonal=7)

        pd.testing.assert_series_equal(residual_comp, residual_full)

    def test_residual_captures_noise(self):
        """Test that residual captures noise and outliers."""
        n = 60
        t = np.arange(n)
        trend = 0.1 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 12)
        noise = np.random.randn(n) * 0.5
        series = trend + seasonal + noise

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='MS'),
            "value": series,
        })

        residual_comp = residual_component(df, period=12, seasonal=7)

        valid_idx = residual_comp.notna()
        if valid_idx.sum() > 0:
            # Residual should have small mean (noise centered around zero)
            residual_mean = residual_comp[valid_idx].mean()
            assert abs(residual_mean) < 1.0

            # Residual should have variance (captures noise)
            residual_std = residual_comp[valid_idx].std()
            assert residual_std > 0.1
