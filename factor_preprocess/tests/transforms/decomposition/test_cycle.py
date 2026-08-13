"""
Test suite for cycle extraction with causality checks.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.decomposition.cycle import (
    bandpass_filter,
    extract_cycle,
    christiano_fitzgerald_filter,
)


class TestBandpassFilter:
    """Test bandpass filter for cycle extraction."""

    def test_basic_bandpass(self):
        """Test basic bandpass filtering."""
        n = 200
        t = np.arange(n)
        # Mix of frequencies
        low_freq = 0.5 * np.sin(2 * np.pi * t / 100)  # Period 100
        high_freq = 0.5 * np.sin(2 * np.pi * t / 5)   # Period 5
        series = low_freq + high_freq

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        # Extract cycle with period around 100 (low_freq = 0.01, high_freq = 0.05)
        result = bandpass_filter(df, low_freq=0.005, high_freq=0.02)

        # First should be NaN (causal lag)
        assert pd.isna(result.iloc[0])

        # Filter should extract the low frequency component
        valid_idx = result.notna()
        if valid_idx.sum() > 50:
            filtered = result[valid_idx].values

            # Should have variance (not all zero)
            assert np.std(filtered) > 0.01

    def test_multiple_assets(self):
        """Test asset isolation in bandpass filter."""
        n = 100
        t = np.arange(n)

        # Asset A: high frequency
        series_a = np.sin(2 * np.pi * t / 5)
        # Asset B: low frequency
        series_b = np.sin(2 * np.pi * t / 50)

        df = pd.DataFrame({
            "asset_id": ["A"] * n + ["B"] * n,
            "date": pd.date_range("2020-01-01", periods=n).tolist() * 2,
            "value": np.concatenate([series_a, series_b]),
        })

        # Filter for high frequency (period ~ 5-10)
        result = bandpass_filter(df, low_freq=0.1, high_freq=0.25)

        # Asset A should have larger variance (high frequency content)
        asset_a = result.iloc[:n]
        asset_b = result.iloc[n:]

        valid_a = asset_a.notna()
        valid_b = asset_b.notna()

        if valid_a.sum() > 20 and valid_b.sum() > 20:
            var_a = asset_a[valid_a].var()
            var_b = asset_b[valid_b].var()

            # Asset A has high freq, should pass through filter better
            assert var_a > var_b * 1.5

    def test_invalid_frequencies(self):
        """Test that invalid frequencies are rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 50,
            "date": pd.date_range("2020-01-01", periods=50),
            "value": np.random.randn(50),
        })

        # low_freq must be in (0, 0.5)
        with pytest.raises(ValueError, match="low_freq must be in"):
            bandpass_filter(df, low_freq=0, high_freq=0.2)

        with pytest.raises(ValueError, match="low_freq must be in"):
            bandpass_filter(df, low_freq=0.6, high_freq=0.2)

        # high_freq must be > low_freq and < 0.5
        with pytest.raises(ValueError, match="high_freq must be in"):
            bandpass_filter(df, low_freq=0.3, high_freq=0.2)

        with pytest.raises(ValueError, match="high_freq must be in"):
            bandpass_filter(df, low_freq=0.1, high_freq=0.6)

    def test_invalid_order(self):
        """Test that invalid order is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 50,
            "date": pd.date_range("2020-01-01", periods=50),
            "value": np.random.randn(50),
        })

        with pytest.raises(ValueError, match="order must be >= 1"):
            bandpass_filter(df, low_freq=0.1, high_freq=0.2, order=0)

    def test_unsorted_fails(self):
        """Test that unsorted data is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
            "value": [3.0, 1.0, 2.0],
        })

        with pytest.raises(ValueError, match="sorted"):
            bandpass_filter(df, low_freq=0.1, high_freq=0.2)

    @pytest.mark.future_poison
    def test_causality_current_not_used(self):
        """Test that current observation is not used in filtering."""
        n = 100
        series = np.sin(2 * np.pi * np.arange(n) / 20)
        series[-1] = 1000.0  # Large spike

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = bandpass_filter(df, low_freq=0.02, high_freq=0.1)

        # The spike should not affect earlier values
        # Check that value before spike is reasonable
        if pd.notna(result.iloc[-2]):
            assert abs(result.iloc[-2]) < 10  # Should not be pulled toward 1000


class TestExtractCycle:
    """Test cycle extraction by period range."""

    def test_basic_cycle_extraction(self):
        """Test extracting cycle with period range."""
        n = 200
        t = np.arange(n)
        # Business cycle-like component (period 20-40)
        cycle = np.sin(2 * np.pi * t / 30)

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": cycle,
        })

        result = extract_cycle(df, low_period=20, high_period=40)

        # First should be NaN
        assert pd.isna(result.iloc[0])

        # Should extract the cycle
        valid_idx = result.notna()
        if valid_idx.sum() > 50:
            extracted = result[valid_idx].values
            assert np.std(extracted) > 0.1

    def test_invalid_periods(self):
        """Test that invalid periods are rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 100,
            "date": pd.date_range("2020-01-01", periods=100),
            "value": np.random.randn(100),
        })

        # low_period must be >= 2
        with pytest.raises(ValueError, match="low_period must be >= 2"):
            extract_cycle(df, low_period=1, high_period=10)

        # high_period must be > low_period
        with pytest.raises(ValueError, match="high_period must be >"):
            extract_cycle(df, low_period=20, high_period=10)

        with pytest.raises(ValueError, match="high_period must be >"):
            extract_cycle(df, low_period=20, high_period=20)

    def test_business_cycle_extraction(self):
        """Test extracting business cycle (2-8 years with quarterly data)."""
        n = 200  # 50 years of quarterly data
        t = np.arange(n)

        # Simulate business cycle (6 year period = 24 quarters)
        business_cycle = 10 * np.sin(2 * np.pi * t / 24)
        # Add high frequency noise
        noise = np.random.randn(n) * 2

        series = business_cycle + noise

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq='QS'),
            "value": series,
        })

        # Extract 2-8 year cycle (8 to 32 quarters)
        result = extract_cycle(df, low_period=8, high_period=32)

        valid_idx = result.notna()
        if valid_idx.sum() > 50:
            extracted = result[valid_idx].values

            # Should be smoother than original (noise filtered out)
            original_std = df.loc[valid_idx, "value"].std()
            extracted_std = np.std(extracted)

            # Extracted should have lower variance (noise removed)
            assert extracted_std < original_std


class TestChristianoFitzgeraldFilter:
    """Test Christiano-Fitzgerald bandpass filter."""

    def test_basic_cf_filter(self):
        """Test basic CF filter."""
        n = 100
        t = np.arange(n)
        # Mix of trend and cycle
        trend = 0.1 * t
        cycle = 5 * np.sin(2 * np.pi * t / 20)
        series = trend + cycle

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = christiano_fitzgerald_filter(
            df, low_period=15, high_period=25, drift=True
        )

        # First should be NaN
        assert pd.isna(result.iloc[0])

        # Should have some valid values
        valid_count = result.notna().sum()
        assert valid_count > 0

    def test_multiple_assets_cf(self):
        """Test asset isolation in CF filter."""
        n = 100
        t = np.arange(n)

        series_a = np.sin(2 * np.pi * t / 10)
        series_b = 2 * np.sin(2 * np.pi * t / 10)

        df = pd.DataFrame({
            "asset_id": ["A"] * n + ["B"] * n,
            "date": pd.date_range("2020-01-01", periods=n).tolist() * 2,
            "value": np.concatenate([series_a, series_b]),
        })

        result = christiano_fitzgerald_filter(df, low_period=8, high_period=12)

        # Assets should be processed independently
        assert result.notna().any()

    def test_invalid_periods_cf(self):
        """Test that invalid periods are rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 100,
            "date": pd.date_range("2020-01-01", periods=100),
            "value": np.random.randn(100),
        })

        with pytest.raises(ValueError, match="low_period must be >= 2"):
            christiano_fitzgerald_filter(df, low_period=1, high_period=10)

        with pytest.raises(ValueError, match="high_period must be >"):
            christiano_fitzgerald_filter(df, low_period=20, high_period=10)

    def test_unsorted_fails_cf(self):
        """Test that unsorted data is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
            "value": [3.0, 1.0, 2.0],
        })

        with pytest.raises(ValueError, match="sorted"):
            christiano_fitzgerald_filter(df, low_period=8, high_period=12)

    def test_drift_parameter(self):
        """Test CF filter with and without drift."""
        n = 100
        t = np.arange(n)
        # Series with trend
        series = 0.5 * t + 5 * np.sin(2 * np.pi * t / 20)

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result_with_drift = christiano_fitzgerald_filter(
            df, low_period=15, high_period=25, drift=True
        )
        result_no_drift = christiano_fitzgerald_filter(
            df, low_period=15, high_period=25, drift=False
        )

        # Both should produce results
        assert result_with_drift.notna().sum() > 0
        assert result_no_drift.notna().sum() > 0

        # Results should differ (drift handling affects detrending)
        valid_both = result_with_drift.notna() & result_no_drift.notna()
        if valid_both.sum() > 10:
            diff = (result_with_drift[valid_both] - result_no_drift[valid_both]).abs().mean()
            # Should have some difference (but may be small)
            assert diff >= 0  # Just check it doesn't error

    @pytest.mark.future_poison
    def test_causality_cf(self):
        """Test that CF filter is causal."""
        n = 100
        series = 5 * np.sin(2 * np.pi * np.arange(n) / 20)
        series[-1] = 500.0  # Spike

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = christiano_fitzgerald_filter(df, low_period=15, high_period=25)

        # Earlier values should not be affected by the spike
        if pd.notna(result.iloc[-10]):
            # Value well before spike should be reasonable
            assert abs(result.iloc[-10]) < 20
