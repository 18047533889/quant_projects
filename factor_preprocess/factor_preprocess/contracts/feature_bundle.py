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

    def __len__(self) -> int:
        """Number of labels on this axis."""
        return len(self.axis_values)


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


# Valid auxiliary channel types (FP-P0-02).
_AUX_CHANNEL_TYPES = {"missing", "freshness", "exposure"}


class FeatureManifest:
    """
    Unified column index mapping for multi-channel layouts (FP-P0-02).

    Maps every channel (feature, missing, freshness, exposure) to a column
    slice in the values array, so every advertised channel is addressable via
    ``get_channel_values``. ``feature_ids`` is the union of the feature-type
    channels; ``total_width`` is the sum of ALL channel sizes (the physical
    feature-axis width).

    FP-P0-01: channels must tile the feature axis exactly with no holes and
    no overlap.
    """

    def __init__(
        self,
        feature_ids: List[str],
        channel_offsets: Dict[str, int],
        channel_sizes: Dict[str, int],
        _allow_aux_channels: bool = False,
        feature_channels: Optional[List[str]] = None,
    ):
        """
        Initialize feature manifest.

        Parameters
        ----------
        feature_ids : List[str]
            Ordered list of feature IDs (feature-type channels only).
        channel_offsets : Dict[str, int]
            Starting column index for each channel.
        channel_sizes : Dict[str, int]
            Number of columns per channel.
        _allow_aux_channels : bool
            Internal: when True, the manifest may carry auxiliary channels
            (missing/freshness/exposure) whose columns are not features, so
            ``total_width`` may exceed ``len(feature_ids)``.
        feature_channels : List[str] | None
            Ordered list of channel names (in channel-offset order) whose
            entries are features. When provided (multi-channel layouts with
            auxiliary channels present), the physical column index of a
            feature is its position within its channel plus the channel
            offset, NOT its position in ``feature_ids``. When None, the
            feature channels are assumed to tile the axis starting at column
            0 with no auxiliary channels ahead, so logical index == physical
            index. Required whenever auxiliary channels come before feature
            channels (FP-P0-12).
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

        self._total_width = max_end

        # Feature dimension must be fully accounted for by the channels when
        # this is a feature-only manifest (no auxiliary channels).
        if not _allow_aux_channels and max_end != len(self._feature_ids):
            raise InvalidContractError(
                f"Max channel end {max_end} does not match feature count "
                f"{len(self._feature_ids)} (offset+size must tile exactly)"
            )

        # FP-P0-01: exact tiling — no holes. The no-overlap + max_end==F
        # checks alone allow holes (e.g. columns {0,2} for F=3). The occupied
        # columns must be exactly {0..width-1}.
        if set(occupied) != set(range(max_end)):
            raise InvalidContractError(
                f"Channels must tile the feature axis exactly with no holes; "
                f"occupied columns {sorted(occupied)} != {list(range(max_end))}"
            )

        # Build reverse lookup: feature_id -> physical column index.
        self._feature_to_col: Dict[str, int] = {}

        if feature_channels is not None:
            # FP-P0-12: with auxiliary channels (missing/freshness/exposure)
            # present, physical column index != logical feature index. A
            # feature channel's entries are consecutive starting at its
            # channel offset, so fid -> offset + local position. Fail closed
            # on any mismatch between the declared feature channels and the
            # passed feature_ids.
            if len(feature_channels) == 0:
                raise InvalidContractError(
                    "feature_channels cannot be empty when provided"
                )
            if len(set(feature_channels)) != len(feature_channels):
                raise InvalidContractError(
                    "feature_channels contains duplicate channel names"
                )
            for ch in feature_channels:
                if ch not in self._channel_offsets:
                    raise InvalidContractError(
                        f"feature channel '{ch}' not present in channel_offsets"
                    )
                if self._channel_sizes[ch] <= 0:
                    raise InvalidContractError(
                        f"feature channel '{ch}' must have size > 0"
                    )

            # The concatenated feature ids of the feature channels (in offset
            # order) must exactly equal the passed feature_ids. The caller
            # guarantees ordering; the feature_ids are consumed positionally
            # across the feature channels, so a length mismatch fails closed.
            total = sum(self._channel_sizes[ch] for ch in feature_channels)
            if total != len(self._feature_ids):
                raise InvalidContractError(
                    "feature_channels total size does not match len(feature_ids): "
                    f"{total} != {len(self._feature_ids)}"
                )

            # Positional mapping: features are consecutive within each channel.
            pos = 0
            for ch in feature_channels:
                offset = self._channel_offsets[ch]
                size = self._channel_sizes[ch]
                for local in range(size):
                    if pos >= len(self._feature_ids):
                        raise InvalidContractError("feature channel exceeds feature_ids")
                    fid = self._feature_ids[pos]
                    self._feature_to_col[fid] = offset + local
                    pos += 1
            if pos != len(self._feature_ids):
                raise InvalidContractError(
                    "feature_channels do not account for all feature_ids "
                    f"({pos} != {len(self._feature_ids)})"
                )
        else:
            # No auxiliary channels: feature channels tile the axis from
            # column 0, so logical index == physical index.
            for i, fid in enumerate(self._feature_ids):
                self._feature_to_col[fid] = i

    @classmethod
    def from_channels(cls, channels: Dict[str, ChannelRef]) -> "FeatureManifest":
        """
        Build a unified manifest from ALL channels (feature + auxiliary).

        Every channel is assigned a contiguous slice; feature_ids is the union
        of the feature-type channels. This is the physical contract for
        multi-channel layouts (FP-P0-02).
        """
        feature_ids: List[str] = []
        feature_channels: List[str] = []
        channel_offsets: Dict[str, int] = {}
        channel_sizes: Dict[str, int] = {}
        offset = 0
        for name, ref in channels.items():
            fids = tuple(ref.feature_ids)
            channel_offsets[name] = offset
            channel_sizes[name] = len(fids)
            if ref.channel_type == "feature":
                feature_channels.append(name)
                feature_ids.extend(fids)
            offset += len(fids)
        return cls(
            feature_ids,
            channel_offsets,
            channel_sizes,
            _allow_aux_channels=True,
            feature_channels=feature_channels,
        )

    @property
    def feature_ids(self) -> Tuple[str, ...]:
        """All feature IDs in order (feature-type channels only)."""
        return self._feature_ids

    @property
    def total_features(self) -> int:
        """Number of feature-type features."""
        return len(self._feature_ids)

    @property
    def total_width(self) -> int:
        """Physical feature-axis width (sum of ALL channel sizes)."""
        return self._total_width

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
        Get the physical column index for a feature ID.

        The column index is the position of the feature within its channel
        plus the channel's offset, so it is correct even when auxiliary
        channels (missing/freshness/exposure) precede feature channels
        (FP-P0-12).

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
            f"width={self._total_width}, "
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
    extraction slices the last axis.

    FP-P0-03: the values shape is validated against the axes and manifest:
    - TNF -> ndim==3, T==len(time_axis), N==len(asset_axis), F==manifest width
    - NF  -> ndim==2, N==len(asset_axis), F==manifest width
    Duplicate axis labels are rejected unless ``allow_duplicate_axis_labels``.

    FP-P0-02: ``has_missing_channel`` / ``has_freshness_channel`` /
    ``has_exposure_channel`` must agree with the actual channels present, and
    every advertised channel must be addressable via ``get_channel_values``.
    """
    bundle_id: str

    # Axes
    time_axis: AxisRef
    asset_axis: AxisRef

    # Channels
    channels: MappingProxyType  # Dict[str, ChannelRef] -> immutable

    # Values (snapshot-copied if ndarray, writeable=False)
    values: Any  # Shape depends on layout
    # Layout of the values array: 'NF' = [asset, feature]; 'TNF' = [time, asset, feature].
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

    # Missingness metadata (must agree with actual channels, FP-P0-02)
    has_missing_channel: bool = False
    has_freshness_channel: bool = False
    has_exposure_channel: bool = False

    # FP-P0-03: reject duplicate axis labels unless explicitly allowed.
    allow_duplicate_axis_labels: bool = False

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

        # FP-P0-02: has_missing/has_freshness/has_exposure must agree with the
        # actual channels present (fail-closed).
        actual_types = {ref.channel_type for ref in self.channels.values()}
        expected_missing = "missing" in actual_types
        expected_freshness = "freshness" in actual_types
        expected_exposure = "exposure" in actual_types
        if self.has_missing_channel != expected_missing:
            raise InvalidContractError(
                f"has_missing_channel={self.has_missing_channel} does not agree "
                f"with actual channels (missing present: {expected_missing})"
            )
        if self.has_freshness_channel != expected_freshness:
            raise InvalidContractError(
                f"has_freshness_channel={self.has_freshness_channel} does not agree "
                f"with actual channels (freshness present: {expected_freshness})"
            )
        if self.has_exposure_channel != expected_exposure:
            raise InvalidContractError(
                f"has_exposure_channel={self.has_exposure_channel} does not agree "
                f"with actual channels (exposure present: {expected_exposure})"
            )

        # Auto-generate manifest for NF/TNF layouts if not provided
        if self.manifest is None and self.layout in ("NF", "TNF"):
            self._auto_generate_manifest()

        # FP-P0-03: validate manifest + axes against the values shape.
        if self.manifest is not None:
            self._validate_shape()

    def _validate_shape(self):
        """Validate values ndim/axis lengths against layout and manifest."""
        if not isinstance(self.values, np.ndarray) or self.values.ndim < 1:
            raise InvalidContractError(
                "manifest requires values to be an ndarray with a feature axis"
            )

        width = self.manifest.total_width
        if self.layout == "TNF":
            if self.values.ndim != 3:
                raise InvalidContractError(
                    f"TNF layout requires values.ndim==3, got {self.values.ndim}"
                )
            t, n, f = self.values.shape
            if f != width:
                raise InvalidContractError(
                    f"manifest width {width} does not match values feature "
                    f"dimension {f}"
                )
            if t != len(self.time_axis):
                raise InvalidContractError(
                    f"TNF time axis length {t} != len(time_axis) {len(self.time_axis)}"
                )
            if n != len(self.asset_axis):
                raise InvalidContractError(
                    f"TNF asset axis length {n} != len(asset_axis) {len(self.asset_axis)}"
                )
        elif self.layout == "NF":
            if self.values.ndim != 2:
                raise InvalidContractError(
                    f"NF layout requires values.ndim==2, got {self.values.ndim}"
                )
            n, f = self.values.shape
            if f != width:
                raise InvalidContractError(
                    f"manifest width {width} does not match values feature "
                    f"dimension {f}"
                )
            if n != len(self.asset_axis):
                raise InvalidContractError(
                    f"NF asset axis length {n} != len(asset_axis) {len(self.asset_axis)}"
                )
        else:
            # Non-NF/TNF layouts: only check the feature dimension.
            feature_dim = self.values.shape[-1]
            if feature_dim != width:
                raise InvalidContractError(
                    f"manifest width {width} does not match values feature "
                    f"dimension {feature_dim}"
                )

        # FP-P0-03: duplicate axis labels rejected unless explicitly allowed.
        if not self.allow_duplicate_axis_labels:
            for axis in (self.time_axis, self.asset_axis):
                labels = list(axis.axis_values)
                if len(labels) != len(set(labels)):
                    raise InvalidContractError(
                        f"Axis '{axis.axis_name}' has duplicate labels; "
                        f"set allow_duplicate_axis_labels=True to permit"
                    )

    def _auto_generate_manifest(self):
        """Auto-generate unified FeatureManifest from ALL channels."""
        if not isinstance(self.values, np.ndarray) or self.values.ndim < 2:
            return

        manifest = FeatureManifest.from_channels(dict(self.channels))
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
        # The feature axis is always the last axis regardless of layout. Use
        # trailing-axis indexing so a [T,N,F] array does not get sliced along
        # the asset axis.
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
        # Feature axis is always the last axis.
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
