"""Bounded factor-axis tile source contract for future single-request evaluation.

This module validates transport and identity only. It does not merge independent
EvaluationBundles or claim to implement evaluate_source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from .errors import InvalidContractError, SnapshotMismatchError
from .factor_batch import AxisRef, FactorBatch


@dataclass(frozen=True)
class FactorTile:
    """One exact half-open factor range from an immutable source snapshot."""

    start: int
    end: int
    batch: FactorBatch
    snapshot_id: str

    def __post_init__(self) -> None:
        if (type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.end <= self.start):
            raise InvalidContractError("tile range must be a nonempty nonnegative half-open range")
        if not isinstance(self.batch, FactorBatch) or self.batch.num_factors != self.end - self.start:
            raise InvalidContractError("tile batch width does not match its range")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise InvalidContractError("tile snapshot_id must be a nonempty string")


@runtime_checkable
class FactorTileSource(Protocol):
    """Reader for bounded factor tiles over one full time/asset panel."""

    factor_ids: tuple[str, ...]
    time_axis: AxisRef
    asset_axis: AxisRef
    dtype: str
    snapshot_id: str
    max_tile_size: int

    def read_tile(self, start: int, end: int) -> FactorTile: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class FactorTileSourceMetadata:
    factor_ids: tuple[str, ...]
    time_axis: AxisRef
    asset_axis: AxisRef
    dtype: str
    snapshot_id: str
    max_tile_size: int


def _same_axis(left: AxisRef, right: AxisRef) -> bool:
    if not isinstance(right, AxisRef):
        return False
    if (left.name, left.dtype, left.size) != (right.name, right.dtype, right.size):
        return False
    if left.values is None or right.values is None:
        return False
    return (left.values.dtype == right.values.dtype
            and np.array_equal(left.values, right.values))


def capture_factor_tile_source(source: FactorTileSource) -> FactorTileSourceMetadata:
    """Freeze source metadata before reads; fail closed on incomplete axes."""
    return _capture_factor_tile_source(source)


def _capture_factor_tile_source(source, *, known_ids=None):
    # Only the same already validated built-in tuple bypasses repeated ID
    # scans. Lists/replacements still receive full validation; every other
    # metadata field retains its original read-boundary checks.
    try:
        raw_ids = source.factor_ids
        ids = tuple(raw_ids)
        time_axis, asset_axis = source.time_axis, source.asset_axis
        dtype, snapshot_id = source.dtype, source.snapshot_id
        maximum = source.max_tile_size
        reader, closer = source.read_tile, source.close
    except (AttributeError, TypeError) as exc:
        raise InvalidContractError("factor tile source metadata is incomplete") from exc
    trusted_ids = type(raw_ids) is tuple and raw_ids is known_ids
    if not trusted_ids and (not ids or any(not isinstance(fid, str) or not fid.strip() for fid in ids)
                           or len(set(ids)) != len(ids)):
        raise InvalidContractError("source factor_ids must be ordered, nonempty and unique")
    if (not isinstance(time_axis, AxisRef) or not isinstance(asset_axis, AxisRef)
            or time_axis.values is None or asset_axis.values is None):
        raise InvalidContractError("source requires full time and asset coordinates")
    if (time_axis.size <= 0 or asset_axis.size <= 0
            or not isinstance(snapshot_id, str) or not snapshot_id.strip()):
        raise InvalidContractError("source axes and snapshot_id must be nonempty")
    if type(maximum) is not int or maximum <= 0:
        raise InvalidContractError("max_tile_size must be a positive integer")
    if not callable(reader) or not callable(closer):
        raise InvalidContractError("source must provide read_tile and close")
    try:
        normalized_dtype = str(np.dtype(dtype))
    except (TypeError, ValueError) as exc:
        raise InvalidContractError("source dtype is invalid") from exc
    if not np.issubdtype(np.dtype(normalized_dtype), np.number) or np.issubdtype(
            np.dtype(normalized_dtype), np.complexfloating):
        raise InvalidContractError("source dtype must be real numeric")
    return FactorTileSourceMetadata(ids, time_axis, asset_axis, normalized_dtype,
                                    snapshot_id, maximum)


def _assert_source_unchanged(source: FactorTileSource, metadata: FactorTileSourceMetadata) -> None:
    current = _capture_factor_tile_source(source, known_ids=metadata.factor_ids)
    if (current.factor_ids != metadata.factor_ids
            or current.dtype != metadata.dtype
            or current.snapshot_id != metadata.snapshot_id
            or current.max_tile_size != metadata.max_tile_size
            or not _same_axis(metadata.time_axis, current.time_axis)
            or not _same_axis(metadata.asset_axis, current.asset_axis)):
        raise SnapshotMismatchError("factor tile source metadata changed during evaluation")


def read_validated_factor_tile(
    source: FactorTileSource, metadata: FactorTileSourceMetadata, start: int, end: int,
) -> FactorTile:
    """Read one exact range and reject reorder, axis drift or snapshot changes."""
    if (type(start) is not int or type(end) is not int or start < 0
            or end <= start or end > len(metadata.factor_ids)
            or end - start > metadata.max_tile_size):
        raise InvalidContractError("requested factor tile range is invalid")
    _assert_source_unchanged(source, metadata)
    tile = source.read_tile(start, end)
    _assert_source_unchanged(source, metadata)
    if not isinstance(tile, FactorTile) or (tile.start, tile.end) != (start, end):
        raise InvalidContractError("source returned a different factor tile range")
    if tile.snapshot_id != metadata.snapshot_id:
        raise SnapshotMismatchError("factor tile snapshot_id mismatch")
    batch = tile.batch
    if batch.factor_ids != metadata.factor_ids[start:end]:
        raise InvalidContractError("factor tile ids are missing, duplicated or reordered")
    if (not _same_axis(metadata.time_axis, batch.time_axis)
            or not _same_axis(metadata.asset_axis, batch.asset_axis)
            or str(batch.values.dtype) != metadata.dtype):
        raise InvalidContractError("factor tile axes or dtype differ from source")
    return tile


def iter_validated_factor_tiles(
    source: FactorTileSource, *, max_tile_size: int | None = None,
) -> Iterator[FactorTile]:
    """Yield exact complete coverage, optionally capping the in-memory tile width."""
    metadata = capture_factor_tile_source(source)
    if max_tile_size is not None and (type(max_tile_size) is not int or max_tile_size <= 0):
        raise InvalidContractError("max_tile_size override must be a positive integer")
    width = min(metadata.max_tile_size, max_tile_size or metadata.max_tile_size)
    for start in range(0, len(metadata.factor_ids), width):
        end = min(start + width, len(metadata.factor_ids))
        yield read_validated_factor_tile(source, metadata, start, end)
