"""
Feature bundle contract for model input.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class AxisRef:
    """Reference to a time or asset axis."""
    axis_name: str
    axis_values: Any  # numpy array, list, or reference
    axis_dtype: str


@dataclass(frozen=True)
class ChannelRef:
    """Reference to a feature channel."""
    channel_name: str
    channel_type: str  # "feature", "missing", "freshness", "exposure"
    feature_ids: List[str]


@dataclass(frozen=True)
class FeatureBundle:
    """
    Transformed features ready for modeling.

    Contains feature values, metadata, and provenance for reproducibility.
    """
    bundle_id: str

    # Axes
    time_axis: AxisRef
    asset_axis: AxisRef

    # Channels
    channels: Dict[str, ChannelRef]

    # Values (can be numpy array or external reference)
    values: Any  # Shape depends on layout
    layout: str = "wide"  # "wide", "long", "block"
    dtype: str = "float64"

    # Source factor metadata
    source_factor_ids: List[str] = field(default_factory=list)

    # Fitted state references (if any fitted transforms were applied)
    fitted_state_refs: List[str] = field(default_factory=list)

    # Timing metadata
    transform_start_time: Optional[datetime] = None
    transform_end_time: Optional[datetime] = None

    # Missingness metadata
    has_missing_channel: bool = False
    has_freshness_channel: bool = False

    # Provenance
    policy_id: Optional[str] = None
    schema_version: str = "0.1.0"
    producer: str = "factor_preprocess"
    producer_version: str = "0.1.0"
    created_at: Optional[datetime] = None
    config_hash: Optional[str] = None

    def __post_init__(self):
        """Validate bundle on construction."""
        if not self.bundle_id:
            raise ValueError("FeatureBundle.bundle_id cannot be empty")
        if not self.channels:
            raise ValueError("FeatureBundle must have at least one channel")

        # Validate layout
        valid_layouts = {"wide", "long", "block"}
        if self.layout not in valid_layouts:
            raise ValueError(f"layout must be one of {valid_layouts}")

    def get_channel(self, channel_name: str) -> Optional[ChannelRef]:
        """Retrieve a specific channel by name."""
        return self.channels.get(channel_name)

    def has_channel(self, channel_name: str) -> bool:
        """Check if bundle has a specific channel."""
        return channel_name in self.channels
