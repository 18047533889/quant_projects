"""Fail-closed contracts for bounded factor-axis source reads."""
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import (
    FactorTile, capture_factor_tile_source, iter_validated_factor_tiles,
    read_validated_factor_tile,
)
from quant_evaluator.contracts.errors import InvalidContractError, SnapshotMismatchError


class Source:
    def __init__(self):
        self.factor_ids = ("a", "b", "c", "d", "e")
        self.time_axis = AxisRef("time", "int64", 2, np.array([1, 2], dtype=np.int64))
        self.asset_axis = AxisRef("asset", "int64", 3, np.array([10, 11, 12], dtype=np.int64))
        self.dtype = "float64"
        self.snapshot_id = "verified-cos-manifest-sha256"
        self.max_tile_size = 2
        self.reads = []
        self.closed = False
        self.modify = None

    def read_tile(self, start, end):
        self.reads.append((start, end))
        ids = self.factor_ids[start:end]
        time_axis, asset_axis = self.time_axis, self.asset_axis
        snapshot_id = self.snapshot_id
        dtype = self.dtype
        if self.modify == "reorder":
            ids = tuple(reversed(ids))
        elif self.modify == "axis":
            time_axis = AxisRef("time", "int64", 2, np.array([1, 3], dtype=np.int64))
        elif self.modify == "dtype":
            dtype = "float32"
        elif self.modify == "tile_snapshot":
            snapshot_id = "another-snapshot"
        values = np.ones((2, 3, end - start), dtype=dtype)
        batch = FactorBatch(factor_ids=ids, time_axis=time_axis,
                            asset_axis=asset_axis, values=values)
        tile = FactorTile(start, end, batch, snapshot_id)
        if self.modify == "source_drift":
            self.snapshot_id = "changed-during-read"
        return tile

    def close(self):
        self.closed = True


def test_contiguous_complete_ranges_and_immutable_values():
    source = Source()
    tiles = list(iter_validated_factor_tiles(source))
    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert tuple(fid for tile in tiles for fid in tile.batch.factor_ids) == source.factor_ids
    assert all(not tile.batch.values.flags.writeable for tile in tiles)


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 0), (0, 3), (4, 6),
                                        (True, 1), (0, False), (0, 1.0)])
def test_invalid_ranges_fail_before_read(start, end):
    source = Source()
    metadata = capture_factor_tile_source(source)
    with pytest.raises(InvalidContractError):
        read_validated_factor_tile(source, metadata, start, end)
    assert source.reads == []


@pytest.mark.parametrize("modify,error", [
    ("reorder", InvalidContractError),
    ("axis", InvalidContractError),
    ("dtype", InvalidContractError),
    ("tile_snapshot", SnapshotMismatchError),
    ("source_drift", SnapshotMismatchError),
])
def test_tile_identity_mismatch_fails_closed(modify, error):
    source = Source()
    source.modify = modify
    metadata = capture_factor_tile_source(source)
    with pytest.raises(error):
        read_validated_factor_tile(source, metadata, 0, 2)


def test_source_metadata_drift_before_read_fails_closed():
    source = Source()
    metadata = capture_factor_tile_source(source)
    source.factor_ids = ("a", "b", "c")
    with pytest.raises(SnapshotMismatchError):
        read_validated_factor_tile(source, metadata, 0, 2)
    assert source.reads == []


@pytest.mark.parametrize("change", [
    lambda s: setattr(s, "factor_ids", ("a", "a")),
    lambda s: setattr(s, "snapshot_id", ""),
    lambda s: setattr(s, "max_tile_size", 0),
    lambda s: setattr(s, "time_axis", AxisRef("time", "int64", 2)),
    lambda s: setattr(s, "dtype", "complex128"),
])
def test_invalid_source_metadata_fails_closed(change):
    source = Source()
    change(source)
    with pytest.raises(InvalidContractError):
        capture_factor_tile_source(source)


def test_caller_can_cap_tile_width_below_source_maximum():
    source = Source()
    tiles = list(iter_validated_factor_tiles(source, max_tile_size=1))
    assert source.reads == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert tuple(fid for tile in tiles for fid in tile.batch.factor_ids) == source.factor_ids


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_invalid_caller_tile_width_fails_before_read(width):
    source = Source()
    with pytest.raises(InvalidContractError):
        list(iter_validated_factor_tiles(source, max_tile_size=width))
    assert source.reads == []
