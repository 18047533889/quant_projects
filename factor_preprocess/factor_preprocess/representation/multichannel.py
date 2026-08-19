"""
Multichannel representation: raw, rank, zscore, residual channels.

Each channel provides a different view of the factor data for model consumption.
"""
import numpy as np
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

from factor_preprocess.transforms.cross_sectional import cs_rank, cs_zscore


ChannelType = Literal["raw", "rank", "zscore", "residual"]


@dataclass(frozen=True)
class MultichannelConfig:
    """Configuration for multichannel representation."""

    channels: List[ChannelType]
    rank_pct: bool = True
    zscore_ddof: int = 1
    zscore_constant_value: float = 0.0
    residual_basis: Optional[str] = None  # Future: industry, sector, market

    def __post_init__(self):
        """Validate configuration."""
        if not self.channels:
            raise ValueError("At least one channel must be specified")

        valid_channels = {"raw", "rank", "zscore", "residual"}
        invalid = set(self.channels) - valid_channels
        if invalid:
            raise ValueError(f"Invalid channel types: {invalid}")


@dataclass(frozen=True)
class MultichannelResult:
    """Result of multichannel transformation."""

    channels: Dict[str, np.ndarray]
    channel_order: List[str]
    shape: tuple
    n_factors: int
    n_channels: int

    def get_channel(self, channel_name: str) -> np.ndarray:
        """Retrieve a specific channel by name."""
        if channel_name not in self.channels:
            raise KeyError(f"Channel '{channel_name}' not found")
        return self.channels[channel_name]

    def as_stacked(self) -> np.ndarray:
        """
        Stack all channels into a single array.

        Returns
        -------
        np.ndarray
            Shape (dates, assets, n_channels * n_factors)
        """
        arrays = [self.channels[ch] for ch in self.channel_order]
        return np.concatenate(arrays, axis=-1)

    def as_interleaved(self) -> np.ndarray:
        """
        Interleave channels by factor.

        Returns
        -------
        np.ndarray
            Shape (dates, assets, n_factors * n_channels)
            Order: factor1_raw, factor1_rank, ..., factor2_raw, factor2_rank, ...
        """
        if len(self.shape) == 3:
            dates, assets, factors = self.shape
            result = np.empty((dates, assets, factors * self.n_channels))

            for f_idx in range(factors):
                for ch_idx, ch_name in enumerate(self.channel_order):
                    col_idx = f_idx * self.n_channels + ch_idx
                    result[:, :, col_idx] = self.channels[ch_name][:, :, f_idx]

            return result
        else:
            # 2D case
            assets, factors = self.shape
            result = np.empty((assets, factors * self.n_channels))

            for f_idx in range(factors):
                for ch_idx, ch_name in enumerate(self.channel_order):
                    col_idx = f_idx * self.n_channels + ch_idx
                    result[:, col_idx] = self.channels[ch_name][:, f_idx]

            return result


def build_multichannel(
    values: np.ndarray,
    config: MultichannelConfig,
    axis: int = -2,
) -> MultichannelResult:
    """
    Build multichannel representation from factor values.

    Parameters
    ----------
    values : np.ndarray
        Factor values. Shape (dates, assets, n_factors) or (assets, n_factors)
    config : MultichannelConfig
        Channel configuration
    axis : int
        Cross-sectional axis for rank/zscore transforms (default -2 for assets)

    Returns
    -------
    MultichannelResult
        Multichannel representation with requested channels

    Notes
    -----
    - raw: original factor values
    - rank: percentile ranks [0, 1] or integer ranks
    - zscore: cross-sectional z-scores
    - residual: NOT IMPLEMENTED — requesting it raises NotImplementedError

    All channels preserve NaN from input and apply transforms cross-sectionally.
    """
    if values.ndim not in (2, 3):
        raise ValueError(f"values must be 2D or 3D, got shape {values.shape}")

    if values.size == 0:
        raise ValueError("values cannot be empty")

    channels = {}

    # Build each requested channel
    for channel_type in config.channels:
        if channel_type == "raw":
            channels["raw"] = values.copy()

        elif channel_type == "rank":
            channels["rank"] = cs_rank(values, axis=axis, pct=config.rank_pct)

        elif channel_type == "zscore":
            channels["zscore"] = cs_zscore(
                values,
                axis=axis,
                ddof=config.zscore_ddof,
                constant_value=config.zscore_constant_value,
            )

        elif channel_type == "residual":
            # Residualization is not implemented.  Emitting a placeholder
            # (e.g. zeros) would inject a fake all-zero "residual" signal
            # into downstream models with no warning, so fail closed.
            # Future: orthogonalize against industry/sector/market.
            raise NotImplementedError(
                "residual channel is not implemented; request it only once "
                "a residualization basis is supported"
            )

    # Determine shape
    if values.ndim == 3:
        n_factors = values.shape[2]
    else:
        n_factors = values.shape[1]

    return MultichannelResult(
        channels=channels,
        channel_order=config.channels,
        shape=values.shape,
        n_factors=n_factors,
        n_channels=len(config.channels),
    )
