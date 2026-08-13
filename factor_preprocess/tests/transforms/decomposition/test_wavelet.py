"""
Test suite for wavelet decomposition with causality checks.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.decomposition.wavelet import (
    wavelet_decompose,
    wavelet_smooth,
    wavelet_denoise,
)


class TestWaveletDecompose:
    """Test wavelet decomposition."""

    def test_basic_wavelet_decomposition(self):
        """Test basic wavelet decomposition."""
        n = 128  # Power of 2 for clean wavelet decomposition
        t = np.arange(n)
        # Mix of low and high frequency
        low_freq = 10 * np.sin(2 * np.pi * t / 32)
        high_freq = 2 * np.sin(2 * np.pi * t / 4)
        series = low_freq + high_freq

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = wavelet_decompose(df, wavelet='db4', level=3)

        # Should have approximation and detail components
        assert 'a3' in result
        assert 'd3' in result or 'd1' in result

        # First should be NaN (causal lag)
        for key, series_val in result.items():
            assert pd.isna(series_val.iloc[0])

        # Should have some valid values
        approx = result['a3']
        assert approx.notna().sum() > 0

    def test_multiple_assets_wavelet(self):
        """Test asset isolation in wavelet decomposition."""
        n = 64
        t = np.arange(n)

        series_a = np.sin(2 * np.pi * t / 16)
        series_b = 5 * np.sin(2 * np.pi * t / 16)

        df = pd.DataFrame({
            "asset_id": ["A"] * n + ["B"] * n,
            "date": pd.date_range("2020-01-01", periods=n).tolist() * 2,
            "value": np.concatenate([series_a, series_b]),
        })

        result = wavelet_decompose(df, wavelet='db4', level=2)

        # Assets should be processed independently
        approx = result['a2']

        asset_a_approx = approx.iloc[:n]
        asset_b_approx = approx.iloc[n:]

        valid_a = asset_a_approx.notna()
        valid_b = asset_b_approx.notna()

        if valid_a.sum() > 10 and valid_b.sum() > 10:
            mean_a = asset_a_approx[valid_a].abs().mean()
            mean_b = asset_b_approx[valid_b].abs().mean()

            # Asset B has larger amplitude
            assert mean_b > mean_a * 2

    def test_different_wavelets(self):
        """Test different wavelet families."""
        n = 64
        series = np.random.randn(n).cumsum()

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        # Test common wavelets
        for wavelet in ['db4', 'sym4', 'coif3']:
            result = wavelet_decompose(df, wavelet=wavelet, level=2)
            assert 'a2' in result
            assert result['a2'].notna().sum() > 0

    def test_invalid_wavelet(self):
        """Test that invalid wavelet is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 64,
            "date": pd.date_range("2020-01-01", periods=64),
            "value": np.random.randn(64),
        })

        with pytest.raises(ValueError, match="Unknown wavelet"):
            wavelet_decompose(df, wavelet='invalid_wavelet')

    def test_unsorted_fails_wavelet(self):
        """Test that unsorted data is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
            "value": [3.0, 1.0, 2.0],
        })

        with pytest.raises(ValueError, match="sorted"):
            wavelet_decompose(df, wavelet='db4')

    def test_short_series_wavelet(self):
        """Test behavior with short time series."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 8,
            "date": pd.date_range("2020-01-01", periods=8),
            "value": np.arange(8) * 1.0,
        })

        result = wavelet_decompose(df, wavelet='db4', level=1)

        # Should handle gracefully
        assert 'a1' in result
        assert len(result['a1']) == 8

    @pytest.mark.future_poison
    def test_causality_wavelet(self):
        """Test that wavelet decomposition is causal."""
        n = 64
        series = np.sin(2 * np.pi * np.arange(n) / 16)
        series[-1] = 100.0  # Large spike

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = wavelet_decompose(df, wavelet='db4', level=2)

        approx = result['a2']

        # Earlier values should not be affected by spike
        if pd.notna(approx.iloc[-10]):
            assert abs(approx.iloc[-10]) < 10


class TestWaveletSmooth:
    """Test wavelet smoothing."""

    def test_basic_wavelet_smooth(self):
        """Test basic wavelet smoothing."""
        n = 128
        t = np.arange(n)
        # Smooth signal + noise
        signal = 10 * np.sin(2 * np.pi * t / 32)
        noise = np.random.randn(n) * 2
        series = signal + noise

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = wavelet_smooth(df, wavelet='db4', level=3)

        # First should be NaN
        assert pd.isna(result.iloc[0])

        # Smoothed result should have lower variance than original
        valid_idx = result.notna()
        if valid_idx.sum() > 50:
            smoothed_std = result[valid_idx].std()
            original_std = df.loc[valid_idx, "value"].std()

            # Smoothing should reduce variance
            assert smoothed_std < original_std

    def test_higher_level_smoother(self):
        """Test that higher level produces smoother result."""
        n = 128
        series = np.random.randn(n).cumsum()

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        smooth_level1 = wavelet_smooth(df, wavelet='db4', level=1)
        smooth_level3 = wavelet_smooth(df, wavelet='db4', level=3)

        valid_idx = smooth_level1.notna() & smooth_level3.notna()

        if valid_idx.sum() > 50:
            # Higher level should be smoother (lower variance in differences)
            diff_var_level1 = np.var(np.diff(smooth_level1[valid_idx].values))
            diff_var_level3 = np.var(np.diff(smooth_level3[valid_idx].values))

            assert diff_var_level3 < diff_var_level1

    def test_multiple_assets_smooth(self):
        """Test asset isolation in wavelet smoothing."""
        n = 64
        series_a = np.random.randn(n)
        series_b = np.random.randn(n) * 5

        df = pd.DataFrame({
            "asset_id": ["A"] * n + ["B"] * n,
            "date": pd.date_range("2020-01-01", periods=n).tolist() * 2,
            "value": np.concatenate([series_a, series_b]),
        })

        result = wavelet_smooth(df, wavelet='db4', level=2)

        # Assets should be independent
        assert len(result) == 2 * n


class TestWaveletDenoise:
    """Test wavelet denoising."""

    def test_basic_wavelet_denoise(self):
        """Test basic wavelet denoising."""
        n = 128
        t = np.arange(n)
        # Clean signal + noise
        signal = 10 * np.sin(2 * np.pi * t / 32)
        noise = np.random.randn(n) * 1
        series = signal + noise

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = wavelet_denoise(df, wavelet='db4', level=3, threshold_mode='soft')

        # First should be NaN
        assert pd.isna(result.iloc[0])

        # Denoised result should have some valid values
        valid_idx = result.notna()
        assert valid_idx.sum() > 50

        # Denoising should reduce high-frequency variance
        if valid_idx.sum() > 50:
            denoised_std = np.std(np.diff(result[valid_idx].values))
            noisy_std = np.std(np.diff(df.loc[valid_idx, "value"].values))

            # Denoised should have smoother (lower) differences
            assert denoised_std < noisy_std

    def test_soft_vs_hard_threshold(self):
        """Test soft vs hard thresholding."""
        n = 128
        series = np.random.randn(n).cumsum()

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        soft = wavelet_denoise(df, wavelet='db4', threshold_mode='soft')
        hard = wavelet_denoise(df, wavelet='db4', threshold_mode='hard')

        # Both should produce results
        assert soft.notna().sum() > 0
        assert hard.notna().sum() > 0

        # Results should differ
        valid_both = soft.notna() & hard.notna()
        if valid_both.sum() > 10:
            diff = (soft[valid_both] - hard[valid_both]).abs().sum()
            assert diff > 0

    def test_invalid_threshold_mode(self):
        """Test that invalid threshold mode is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 64,
            "date": pd.date_range("2020-01-01", periods=64),
            "value": np.random.randn(64),
        })

        with pytest.raises(ValueError, match="threshold_mode must be"):
            wavelet_denoise(df, wavelet='db4', threshold_mode='invalid')

    def test_threshold_scale(self):
        """Test effect of threshold scale parameter."""
        n = 128
        t = np.arange(n)
        signal = 10 * np.sin(2 * np.pi * t / 32)
        noise = np.random.randn(n) * 2
        series = signal + noise

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        # Low threshold = less aggressive denoising
        result_low = wavelet_denoise(df, wavelet='db4', threshold_scale=0.5)
        # High threshold = more aggressive denoising
        result_high = wavelet_denoise(df, wavelet='db4', threshold_scale=2.0)

        valid_both = result_low.notna() & result_high.notna()

        if valid_both.sum() > 50:
            # Higher threshold should produce smoother result
            diff_var_low = np.var(np.diff(result_low[valid_both].values))
            diff_var_high = np.var(np.diff(result_high[valid_both].values))

            assert diff_var_high < diff_var_low

    def test_unsorted_fails_denoise(self):
        """Test that unsorted data is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
            "value": [3.0, 1.0, 2.0],
        })

        with pytest.raises(ValueError, match="sorted"):
            wavelet_denoise(df, wavelet='db4')

    def test_multiple_assets_denoise(self):
        """Test asset isolation in wavelet denoising."""
        n = 64
        series_a = np.random.randn(n)
        series_b = np.random.randn(n) * 5

        df = pd.DataFrame({
            "asset_id": ["A"] * n + ["B"] * n,
            "date": pd.date_range("2020-01-01", periods=n).tolist() * 2,
            "value": np.concatenate([series_a, series_b]),
        })

        result = wavelet_denoise(df, wavelet='db4')

        # Assets should be independent
        assert len(result) == 2 * n

        # Asset B should have larger values (larger input variance)
        asset_a = result.iloc[:n]
        asset_b = result.iloc[n:]

        valid_a = asset_a.notna()
        valid_b = asset_b.notna()

        if valid_a.sum() > 10 and valid_b.sum() > 10:
            std_a = asset_a[valid_a].std()
            std_b = asset_b[valid_b].std()

            assert std_b > std_a * 2

    @pytest.mark.future_poison
    def test_causality_denoise(self):
        """Test that wavelet denoising is causal."""
        n = 128
        series = np.random.randn(n)
        series[-1] = 100.0  # Large spike

        df = pd.DataFrame({
            "asset_id": ["A"] * n,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": series,
        })

        result = wavelet_denoise(df, wavelet='db4')

        # Earlier values should not be affected by spike
        if pd.notna(result.iloc[-10]):
            assert abs(result.iloc[-10]) < 10
