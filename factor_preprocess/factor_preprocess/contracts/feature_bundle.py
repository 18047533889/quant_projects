"""
Feature bundle contract for model input.
"""
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class AxisRef:
    """Reference to a time or asset axis.

    ``axis_values`` accepts a numpy array, list, or reference at construction
    time; storage is made immutable in ``__post_init__``: numpy arrays are
    marked read-only in place, and lists are converted to tuples.
    """
    axis_name: str
    axis_values: Any  # numpy array (read-only), tuple, or reference
    axis_dtype: str

    def __post_init__(self):
        """Freeze axis_values so a frozen AxisRef cannot be mutated in place."""
        values = self.axis_values
        if isinstance(values, np.ndarray):
            # Mark the array read-only in place; any later assignment through
            # it raises ValueError.  Callers handing us an array they still
            # own should pass a copy if they need to keep writing to it.
            try:
                values.flags.writeable = False
            except ValueError:
                # E.g. a view onto a read-only base or a non-contiguous slice
                # of a buffer that cannot be locked; fall back to a frozen copy.
                frozen = values.copy()
                frozen.flags.writeable = False
                object.__setattr__(self, "axis_values", frozen)
        elif isinstance(values, list):
            object.__setattr__(self, "axis_values", tuple(values))


@dataclass(frozen=True)
class ChannelRef:
    """Reference to a feature channel."""
    channel_name: str
    channel_type: str  # "feature", "missing", "freshness", "exposure"
    feature_ids: Tuple[str, ...]

    def __post_init__(self):
        """Validate channel reference (FP-P0-11 fail-closed hardening)."""
        if not self.channel_name:
            raise ValueError("ChannelRef.channel_name cannot be empty")
        valid_types = {"feature", "missing", "freshness", "exposure"}
        if self.channel_type not in valid_types:
            raise ValueError(
                f"ChannelRef.channel_type must be one of {sorted(valid_types)}, "
                f"got {self.channel_type!r}"
            )
        if not self.feature_ids:
            raise ValueError(
                f"ChannelRef '{self.channel_name}' must declare at least one "
                "feature_id"
            )
        if len(self.feature_ids) != len(set(self.feature_ids)):
            raise ValueError(
                f"ChannelRef '{self.channel_name}' has duplicate feature_ids"
            )
        # Constructor accepts any iterable (lists keep working); storage is an
        # immutable tuple so a frozen ChannelRef cannot be mutated in place.
        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))


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

    # Channels (read-only mapping; constructed from any mapping)
    channels: Mapping[str, ChannelRef]

    # Values (can be numpy array or external reference)
    values: Any  # Shape depends on layout
    layout: str = "wide"  # "wide", "long", "block"
    dtype: str = "float64"

    # Source factor metadata (immutable tuple; constructed from any iterable)
    source_factor_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Fitted state references (if any fitted transforms were applied)
    fitted_state_refs: Tuple[str, ...] = field(default_factory=tuple)

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
        """Validate bundle on construction (FP-P0-11 fail-closed hardening).

        Containers are also converted to immutable equivalents: ``channels``
        becomes a ``MappingProxyType`` over a private dict, and the list-typed
        fields become tuples, so a frozen bundle cannot be mutated in place.
        """
        # Deep immutability: constructors still accept plain dicts/lists, but
        # storage is frozen before any validation logic reads it.
        object.__setattr__(
            self, "channels", MappingProxyType(dict(self.channels))
        )
        object.__setattr__(self, "source_factor_ids", tuple(self.source_factor_ids))
        object.__setattr__(self, "fitted_state_refs", tuple(self.fitted_state_refs))

        if not self.bundle_id:
            raise ValueError("FeatureBundle.bundle_id cannot be empty")
        if not self.channels:
            raise ValueError("FeatureBundle must have at least one channel")

        # Validate layout
        valid_layouts = {"wide", "long", "block"}
        if self.layout not in valid_layouts:
            raise ValueError(f"layout must be one of {valid_layouts}")

        # Channel dict keys must match each ChannelRef's channel_name; a
        # mismatch silently routes lookups to the wrong channel metadata.
        for key, channel in self.channels.items():
            if key != channel.channel_name:
                raise ValueError(
                    f"channels key {key!r} does not match ChannelRef "
                    f"channel_name {channel.channel_name!r}"
                )

        # Missing/freshness flags must not lie about channel presence
        # (downstream consumers gate imputation/staleness logic on them).
        has_missing = any(
            ch.channel_type == "missing" for ch in self.channels.values()
        )
        has_freshness = any(
            ch.channel_type == "freshness" for ch in self.channels.values()
        )
        if self.has_missing_channel and not has_missing:
            raise ValueError(
                "has_missing_channel=True but no channel with "
                "channel_type='missing' is present"
            )
        if self.has_freshness_channel and not has_freshness:
            raise ValueError(
                "has_freshness_channel=True but no channel with "
                "channel_type='freshness' is present"
            )

        # Wide layout has a fixed (time, asset, feature) shape contract.
        if self.layout == "wide" and self.values is not None:
            values = np.asarray(self.values)
            feature_channels = [
                ch for ch in self.channels.values()
                if ch.channel_type == "feature"
            ]
            if feature_channels:
                n_features = len(feature_channels[0].feature_ids)
                if values.ndim != 3:
                    raise ValueError(
                        "wide layout requires values with shape "
                        "(time, asset, feature); use 'long'/'block' for "
                        f"other shapes (got ndim={values.ndim})"
                    )
                if values.shape[2] != n_features:
                    raise ValueError(
                        f"wide layout feature axis size {values.shape[2]} does "
                        f"not match feature channel with {n_features} "
                        "feature_ids"
                    )
            time_len = len(self.time_axis.axis_values)
            if values.shape[0] != time_len:
                raise ValueError(
                    f"time_axis length {time_len} does not match values "
                    f"time dimension {values.shape[0]}"
                )
            asset_len = len(self.asset_axis.axis_values)
            if values.shape[1] != asset_len:
                raise ValueError(
                    f"asset_axis length {asset_len} does not match values "
                    f"asset dimension {values.shape[1]}"
                )

    def get_channel(self, channel_name: str) -> Optional[ChannelRef]:
        """Retrieve a specific channel by name."""
        return self.channels.get(channel_name)

    def has_channel(self, channel_name: str) -> bool:
        """Check if bundle has a specific channel."""
        return channel_name in self.channels
