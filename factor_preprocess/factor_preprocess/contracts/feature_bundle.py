"""
Feature bundle contract for model input.

Provides immutable, snapshot-safe containers for transformed features.
"""
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from factor_preprocess.errors import InvalidContractError
from factor_preprocess.contracts.state import _freeze


def _freeze_value(value: Any) -> Any:
    """
    Create an immutable snapshot of a value.

    - ndarray: copy then set writeable=False
    - list/tuple of str: convert to tuple
    - dict: convert to MappingProxyType
    - other: return as-is
    """
    if isinstance(value, np.ndarray):
        frozen = value.copy()
        frozen.flags.writeable = False
        return frozen
    elif isinstance(value, list):
        return tuple(value)
    elif isinstance(value, dict):
        # Deep snapshot: copy before wrapping (see FP-P0-02).
        return MappingProxyType(dict(value))
    return value


@dataclass(frozen=True)
class AxisRef:
    """
    Reference to a time or asset axis.

    The axis_values are snapshot-copied and frozen to prevent
    caller-owned ndarray mutation.
    """
    axis_name: str
    axis_values: Any  # numpy array, list, or reference
    axis_dtype: str

    def __post_init__(self):
        """Freeze axis_values to prevent mutation."""
        # Frozen dataclass allows __init__ but we need to freeze mutable fields
        # Use object.__setattr__ since frozen dataclass blocks direct assignment
        if isinstance(self.axis_values, np.ndarray):
            frozen = self.axis_values.copy()
            frozen.flags.writeable = False
            object.__setattr__(self, 'axis_values', frozen)
        elif isinstance(self.axis_values, list):
            object.__setattr__(self, 'axis_values', tuple(self.axis_values))


@dataclass(frozen=True)
class ChannelRef:
    """Reference to a feature channel."""
    channel_name: str
    channel_type: str  # "feature", "missing", "freshness", "exposure"
    feature_ids: Tuple[str, ...]  # Immutable tuple

    def __post_init__(self):
        """Freeze feature_ids to tuple."""
        if isinstance(self.feature_ids, list):
            object.__setattr__(self, 'feature_ids', tuple(self.feature_ids))


class FeatureManifest:
    """
    Column index mapping for multi-channel layouts.

    Maps feature_ids to column indices in the values array,
    enabling proper multi-feature channel support.
    """

    def __init__(
        self,
        feature_ids: List[str],
        channel_offsets: Dict[str, int],
        channel_sizes: Dict[str, int],
    ):
        """
        Initialize feature manifest.

        Parameters
        ----------
        feature_ids : List[str]
            Ordered list of all feature IDs across channels
        channel_offsets : Dict[str, int]
            Starting column index for each channel
        channel_sizes : Dict[str, int]
            Number of columns per channel
        """
        self._feature_ids: Tuple[str, ...] = tuple(feature_ids)
        # Snapshot-copy caller dicts before wrapping so later mutation of the
        # caller's dict cannot leak into this manifest (deep snapshot).
        self._channel_offsets: MappingProxyType[str, int] = MappingProxyType(dict(channel_offsets))
        self._channel_sizes: MappingProxyType[str, int] = MappingProxyType(dict(channel_sizes))

        # FP-P0-01: duplicate feature IDs would silently collapse to the last
        # column. Reject them unless they are channel-qualified (RAW::x vs
        # RANK::x) and therefore distinct.
        if len(self._feature_ids) != len(set(self._feature_ids)):
            raise InvalidContractError(
                f"Duplicate feature IDs in manifest: "
                f"{[fid for fid in set(self._feature_ids) if self._feature_ids.count(fid) > 1]}"
            )

        # FP-P0-03: shape/layout validation.
        if not self._channel_offsets:
            raise InvalidContractError("FeatureManifest requires at least one channel")
        if set(self._channel_offsets) != set(self._channel_sizes):
            raise InvalidContractError(
                "channel_offsets and channel_sizes must have the same channel set"
            )

        max_end = 0
        for ch in self._channel_offsets:
            offset = self._channel_offsets[ch]
            size = self._channel_sizes[ch]
            if offset < 0:
                raise InvalidContractError(
                    f"Channel '{ch}' offset must be >= 0, got {offset}"
                )
            if size <= 0:
                raise InvalidContractError(
                    f"Channel '{ch}' size must be > 0, got {size}"
                )
            max_end = max(max_end, offset + size)

        # Channels must not overlap each other.
        occupied = {}
        for ch in sorted(self._channel_offsets, key=lambda c: self._channel_offsets[c]):
            offset = self._channel_offsets[ch]
            size = self._channel_sizes[ch]
            for col in range(offset, offset + size):
                if col in occupied:
                    raise InvalidContractError(
                        f"Channel '{ch}' overlaps channel '{occupied[col]}' at column {col}"
                    )
                occupied[col] = ch

        # Feature dimension must be fully accounted for by the channels.
        if max_end != len(self._feature_ids):
            raise InvalidContractError(
                f"Max channel end {max_end} does not match feature count "
                f"{len(self._feature_ids)} (offset+size must tile exactly)"
            )

        # Build reverse lookup: feature_id -> column index
        self._feature_to_col: Dict[str, int] = {}
        for i, fid in enumerate(self._feature_ids):
            self._feature_to_col[fid] = i

    @property
    def feature_ids(self) -> Tuple[str, ...]:
        """All feature IDs in order."""
        return self._feature_ids

    @property
    def total_features(self) -> int:
        """Total number of features."""
        return len(self._feature_ids)

    @property
    def channel_offsets(self) -> MappingProxyType[str, int]:
        """Starting column index for each channel."""
        return self._channel_offsets

    @property
    def channel_sizes(self) -> MappingProxyType[str, int]:
        """Number of columns per channel."""
        return self._channel_sizes

    def get_column_index(self, feature_id: str) -> int:
        """
        Get column index for a feature ID.

        Parameters
        ----------
        feature_id : str
            Feature identifier

        Returns
        -------
        int
            Column index in the values array
        """
        if feature_id not in self._feature_to_col:
            raise KeyError(f"Feature '{feature_id}' not found in manifest")
        return self._feature_to_col[feature_id]

    def get_channel_slice(self, channel_name: str) -> slice:
        """
        Get slice for extracting a channel's columns.

        Parameters
        ----------
        channel_name : str
            Channel name

        Returns
        -------
        slice
            Slice object for column selection
        """
        if channel_name not in self._channel_offsets:
            raise KeyError(f"Channel '{channel_name}' not found in manifest")
        offset = self._channel_offsets[channel_name]
        size = self._channel_sizes[channel_name]
        return slice(offset, offset + size)

    def get_feature_indices(self, feature_ids: List[str]) -> List[int]:
        """
        Get column indices for multiple feature IDs.

        Parameters
        ----------
        feature_ids : List[str]
            Feature identifiers

        Returns
        -------
        List[int]
            Column indices
        """
        return [self.get_column_index(fid) for fid in feature_ids]

    def __contains__(self, feature_id: str) -> bool:
        """Check if feature exists."""
        return feature_id in self._feature_to_col

    def __len__(self) -> int:
        """Number of features."""
        return self.total_features

    def __repr__(self) -> str:
        return (
            f"FeatureManifest(features={self.total_features}, "
            f"channels={list(self._channel_offsets.keys())})"
        )


@dataclass(frozen=True)
class FeatureBundle:
    """
    Transformed features ready for modeling.

    Contains feature values, metadata, and provenance for reproducibility.

    Immutability guarantees:
    - values: ndarray is snapshot-copied and frozen (writeable=False)
    - channels: wrapped in MappingProxyType (copied first, see FP-P0-02)
    - source_factor_ids, fitted_state_refs: converted to tuples
    - AxisRef.axis_values: snapshot-copied and frozen
    - ChannelRef.feature_ids: converted to tuple

    Layout: 'NF' = [asset, feature]; 'TNF' = [time, asset, feature].
    The feature axis is always the trailing axis, so channel/feature
    extraction slices the last axis (see FP-P0-04).
    """
    bundle_id: str

    # Axes
    time_axis: AxisRef
    asset_axis: AxisRef

    # Channels
    channels: MappingProxyType  # Dict[str, ChannelRef] -> immutable

    # Values (snapshot-copied if ndarray, writeable=False)
    values: Any  # Shape depends on layout
    # Layout of the values array: 'NF' = [feature, ...]; 'TNF' = [time, asset, feature].
    # Any other value (e.g. 'wide') leaves extraction unchanged for back-compat.
    layout: str = "NF"
    dtype: str = "float64"

    # Source factor metadata (immutable tuple)
    source_factor_ids: Tuple[str, ...] = ()

    # Fitted state references (immutable tuple)
    fitted_state_refs: Tuple[str, ...] = ()

    # Column index mapping for multi-channel layouts
    manifest: Optional[FeatureManifest] = None

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
        """Validate and freeze bundle on construction."""
        if not self.bundle_id:
            raise ValueError("FeatureBundle.bundle_id cannot be empty")
        if not self.channels:
            raise ValueError("FeatureBundle must have at least one channel")

        # Validate layout
        valid_layouts = {"NF", "TNF", "wide", "long", "block"}
        if self.layout not in valid_layouts:
            raise ValueError(f"layout must be one of {valid_layouts}")

        # FP-P0-03: validate manifest against the values feature dimension.
        if self.manifest is not None:
            if not isinstance(self.values, np.ndarray) or self.values.ndim < 1:
                raise InvalidContractError(
                    "manifest requires values to be an ndarray with a feature axis"
                )
            feature_dim = self.values.shape[-1]
            if self.manifest.total_features != feature_dim:
                raise InvalidContractError(
                    f"manifest.total_features={self.manifest.total_features} does not match "
                    f"values feature dimension {feature_dim}"
                )

        # Freeze values: snapshot ndarray then set writeable=False
        if isinstance(self.values, np.ndarray):
            frozen_values = self.values.copy()
            frozen_values.flags.writeable = False
            object.__setattr__(self, 'values', frozen_values)
        elif isinstance(self.values, list):
            # Convert list to immutable tuple
            object.__setattr__(self, 'values', tuple(self.values))

        # Freeze channels to MappingProxyType
        if isinstance(self.channels, dict):
            # Ensure all ChannelRefs have immutable feature_ids
            frozen_channels = {}
            for name, ref in self.channels.items():
                if isinstance(ref, ChannelRef):
                    frozen_channels[name] = ref
                else:
                    # Convert dict to ChannelRef if needed
                    frozen_channels[name] = ChannelRef(**ref)
            object.__setattr__(self, 'channels', MappingProxyType(frozen_channels))
        elif isinstance(self.channels, MappingProxyType):
            # Deep snapshot: snapshot-copy before wrapping so later mutation of
            # the caller's mapping cannot leak into the bundle (FP-P0-02).
            snapshot = dict(self.channels)
            object.__setattr__(self, 'channels', MappingProxyType(snapshot))
        else:
            # Convert whatever we got to MappingProxyType
            object.__setattr__(self, 'channels', MappingProxyType(dict(self.channels)))

        # Freeze source_factor_ids and fitted_state_refs to tuples
        if isinstance(self.source_factor_ids, list):
            object.__setattr__(self, 'source_factor_ids', tuple(self.source_factor_ids))
        if isinstance(self.fitted_state_refs, list):
            object.__setattr__(self, 'fitted_state_refs', tuple(self.fitted_state_refs))

        # Auto-generate manifest for NF/TNF layouts if not provided
        if self.manifest is None and self.layout in ("NF", "TNF"):
            self._auto_generate_manifest()
            # Auto-generated manifest may expose a mismatch (e.g. fewer feature
            # channels than values columns). Surface it as a contract error.
            if self.manifest is not None:
                if not isinstance(self.values, np.ndarray) or self.values.ndim < 1:
                    raise InvalidContractError(
                        "manifest requires values to be an ndarray with a feature axis"
                    )
                feature_dim = self.values.shape[-1]
                if self.manifest.total_features != feature_dim:
                    raise InvalidContractError(
                        f"manifest.total_features={self.manifest.total_features} does not match "
                        f"values feature dimension {feature_dim}"
                    )

    def _auto_generate_manifest(self):
        """Auto-generate FeatureManifest for wide layout."""
        if not isinstance(self.values, np.ndarray) or self.values.ndim < 2:
            return

        # Build manifest from channels
        feature_ids = []
        channel_offsets = {}
        channel_sizes = {}
        offset = 0

        for name, ref in self.channels.items():
            if ref.channel_type == "feature":
                fids = list(ref.feature_ids) if isinstance(ref.feature_ids, tuple) else ref.feature_ids
                channel_offsets[name] = offset
                channel_sizes[name] = len(fids)
                feature_ids.extend(fids)
                offset += len(fids)

        if feature_ids:
            manifest = FeatureManifest(
                feature_ids=feature_ids,
                channel_offsets=channel_offsets,
                channel_sizes=channel_sizes,
            )
            object.__setattr__(self, 'manifest', manifest)

    def get_channel(self, channel_name: str) -> Optional[ChannelRef]:
        """Retrieve a specific channel by name."""
        return self.channels.get(channel_name)

    def has_channel(self, channel_name: str) -> bool:
        """Check if bundle has a specific channel."""
        return channel_name in self.channels

    def get_channel_values(self, channel_name: str) -> np.ndarray:
        """
        Extract values for a specific channel using manifest.

        Parameters
        ----------
        channel_name : str
            Channel name

        Returns
        -------
        np.ndarray
            Values for the specified channel

        Raises
        ------
        KeyError
            If channel not found or manifest not available
        """
        if self.manifest is None:
            raise KeyError("No manifest available for channel extraction")
        if channel_name not in self.channels:
            raise KeyError(f"Channel '{channel_name}' not found")

        slice_obj = self.manifest.get_channel_slice(channel_name)
        # FP-P0-04: the feature axis is always the last axis regardless of
        # layout. Use trailing-axis indexing so a [T,N,F] array does not get
        # sliced along the asset axis.
        return self.values[..., slice_obj]

    def get_feature_values(self, feature_id: str) -> np.ndarray:
        """
        Extract values for a specific feature using manifest.

        Parameters
        ----------
        feature_id : str
            Feature identifier

        Returns
        -------
        np.ndarray
            Values for the specified feature

        Raises
        ------
        KeyError
            If feature not found or manifest not available
        """
        if self.manifest is None:
            raise KeyError("No manifest available for feature extraction")

        col_idx = self.manifest.get_column_index(feature_id)
        # Feature axis is always the last axis (see FP-P0-04).
        return self.values[..., col_idx]

    def is_immutable(self) -> bool:
        """
        Verify bundle immutability.

        Returns
        -------
        bool
            True if all fields are properly frozen
        """
        checks = []

        # Check values
        if isinstance(self.values, np.ndarray):
            checks.append(not self.values.flags.writeable)
        else:
            checks.append(True)

        # Check channels
        checks.append(isinstance(self.channels, MappingProxyType))

        # Check tuples
        checks.append(isinstance(self.source_factor_ids, tuple))
        checks.append(isinstance(self.fitted_state_refs, tuple))

        return all(checks)

    def __repr__(self) -> str:
        channel_names = list(self.channels.keys())
        values_shape = self.values.shape if isinstance(self.values, np.ndarray) else None
        return (
            f"FeatureBundle(id={self.bundle_id}, layout={self.layout}, "
            f"channels={channel_names}, shape={values_shape})"
        )
