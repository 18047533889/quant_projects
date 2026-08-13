"""
Test suite for multichannel representation.
"""
import pytest
import numpy as np
from factor_preprocess.representation.multichannel import (
    build_multichannel,
    MultichannelConfig,
    MultichannelResult,
)


class TestMultichannelConfig:
    """Test MultichannelConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = MultichannelConfig(channels=["raw", "rank", "zscore"])
        assert config.channels == ["raw", "rank", "zscore"]
        assert config.rank_pct is True

    def test_empty_channels(self):
        """Test that empty channels list is rejected."""
        with pytest.raises(ValueError, match="At least one channel"):
            MultichannelConfig(channels=[])

    def test_invalid_channel_type(self):
        """Test that invalid channel types are rejected."""
        with pytest.raises(ValueError, match="Invalid channel types"):
            MultichannelConfig(channels=["raw", "invalid"])


class TestBuildMultichannel:
    """Test multichannel building."""

    def test_single_channel_raw(self):
        """Test building single raw channel."""
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        config = MultichannelConfig(channels=["raw"])
        result = build_multichannel(values, config, axis=0)

        assert "raw" in result.channels
        np.testing.assert_array_equal(result.channels["raw"], values)
        assert result.n_channels == 1
        assert result.n_factors == 2

    def test_multiple_channels_2d(self):
        """Test building multiple channels from 2D array."""
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        config = MultichannelConfig(channels=["raw", "rank", "zscore"])
        result = build_multichannel(values, config, axis=0)

        assert len(result.channels) == 3
        assert "raw" in result.channels
        assert "rank" in result.channels
        assert "zscore" in result.channels
        assert result.n_channels == 3
        assert result.shape == values.shape

    def test_multiple_channels_3d(self):
        """Test building multiple channels from 3D array."""
        # Shape: (dates, assets, factors)
        values = np.random.randn(10, 20, 5)
        config = MultichannelConfig(channels=["raw", "rank"])
        result = build_multichannel(values, config, axis=1)

        assert result.shape == values.shape
        assert result.n_factors == 5
        assert result.n_channels == 2

    def test_rank_channel_percentile(self):
        """Test rank channel with percentile ranks."""
        values = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
        config = MultichannelConfig(channels=["rank"], rank_pct=True)
        result = build_multichannel(values, config, axis=0)

        rank_channel = result.channels["rank"]
        # Each column should be ranked independently
        # Column 0: [1, 2, 3] -> [0, 0.5, 1]
        # Column 1: [5, 4, 3] -> [1, 0.5, 0]
        assert np.all((rank_channel >= 0) & (rank_channel <= 1))

    def test_zscore_channel(self):
        """Test zscore channel normalization."""
        values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        config = MultichannelConfig(channels=["zscore"])
        result = build_multichannel(values, config, axis=0)

        zscore_channel = result.channels["zscore"]
        # Each column should have mean ~0 and std ~1
        for col in range(zscore_channel.shape[1]):
            col_data = zscore_channel[:, col]
            np.testing.assert_allclose(np.mean(col_data), 0.0, atol=1e-10)
            np.testing.assert_allclose(np.std(col_data, ddof=1), 1.0, atol=1e-10)

    def test_residual_channel_stub(self):
        """Test residual channel (stub implementation)."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = MultichannelConfig(channels=["residual"])
        result = build_multichannel(values, config, axis=0)

        # Stub returns zeros
        np.testing.assert_array_equal(result.channels["residual"], np.zeros_like(values))

    def test_residual_with_basis_not_implemented(self):
        """Test that residual_basis raises NotImplementedError."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = MultichannelConfig(channels=["residual"], residual_basis="industry")

        with pytest.raises(NotImplementedError, match="residual_basis"):
            build_multichannel(values, config, axis=0)

    def test_nan_preservation(self):
        """Test that NaN is preserved across channels."""
        values = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
        config = MultichannelConfig(channels=["raw", "rank", "zscore"])
        result = build_multichannel(values, config, axis=0)

        # NaN should be preserved in all channels
        assert np.isnan(result.channels["raw"][0, 1])
        assert np.isnan(result.channels["rank"][0, 1])
        assert np.isnan(result.channels["zscore"][0, 1])

    def test_empty_values_rejected(self):
        """Test that empty values are rejected."""
        values = np.array([])
        config = MultichannelConfig(channels=["raw"])

        with pytest.raises(ValueError, match="must be 2D or 3D"):
            build_multichannel(values, config)

    def test_invalid_ndim(self):
        """Test that 1D or 4D arrays are rejected."""
        values = np.array([1.0, 2.0, 3.0])
        config = MultichannelConfig(channels=["raw"])

        with pytest.raises(ValueError, match="must be 2D or 3D"):
            build_multichannel(values, config)


class TestMultichannelResult:
    """Test MultichannelResult methods."""

    def test_get_channel(self):
        """Test retrieving a specific channel."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = MultichannelConfig(channels=["raw", "rank"])
        result = build_multichannel(values, config, axis=0)

        raw_channel = result.get_channel("raw")
        np.testing.assert_array_equal(raw_channel, values)

    def test_get_channel_missing(self):
        """Test that missing channel raises KeyError."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = MultichannelConfig(channels=["raw"])
        result = build_multichannel(values, config, axis=0)

        with pytest.raises(KeyError, match="not found"):
            result.get_channel("nonexistent")

    def test_as_stacked_2d(self):
        """Test stacking channels for 2D input."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = MultichannelConfig(channels=["raw", "rank"])
        result = build_multichannel(values, config, axis=0)

        stacked = result.as_stacked()
        # Should concatenate along last axis: (2, 2) + (2, 2) -> (2, 4)
        assert stacked.shape == (2, 4)

    def test_as_stacked_3d(self):
        """Test stacking channels for 3D input."""
        values = np.random.randn(5, 10, 3)
        config = MultichannelConfig(channels=["raw", "rank"])
        result = build_multichannel(values, config, axis=1)

        stacked = result.as_stacked()
        # Should concatenate: (5, 10, 3) + (5, 10, 3) -> (5, 10, 6)
        assert stacked.shape == (5, 10, 6)

    def test_as_interleaved_2d(self):
        """Test interleaving channels for 2D input."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = MultichannelConfig(channels=["raw", "rank"])
        result = build_multichannel(values, config, axis=0)

        interleaved = result.as_interleaved()
        # Should interleave: factor1_raw, factor1_rank, factor2_raw, factor2_rank
        assert interleaved.shape == (2, 4)

    def test_as_interleaved_3d(self):
        """Test interleaving channels for 3D input."""
        values = np.random.randn(5, 10, 3)
        config = MultichannelConfig(channels=["raw", "rank", "zscore"])
        result = build_multichannel(values, config, axis=1)

        interleaved = result.as_interleaved()
        # 3 factors * 3 channels = 9 features
        assert interleaved.shape == (5, 10, 9)

    def test_channel_order_preserved(self):
        """Test that channel order is preserved."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = MultichannelConfig(channels=["zscore", "raw", "rank"])
        result = build_multichannel(values, config, axis=0)

        assert result.channel_order == ["zscore", "raw", "rank"]

        stacked = result.as_stacked()
        # First 2 columns should be zscore, next 2 raw, last 2 rank
        np.testing.assert_array_equal(stacked[:, :2], result.channels["zscore"])
        np.testing.assert_array_equal(stacked[:, 2:4], result.channels["raw"])
