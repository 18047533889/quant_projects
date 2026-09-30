from types import SimpleNamespace
import numpy as np
import pytest
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.factor_tile_source import capture_factor_tile_source, _assert_source_unchanged
from quant_evaluator.contracts.errors import InvalidContractError, SnapshotMismatchError

def make_source(ids):
    return SimpleNamespace(factor_ids=ids,
        time_axis=AxisRef("time", "int64", 2, np.arange(2, dtype=np.int64)),
        asset_axis=AxisRef("asset", "int64", 2, np.arange(2, dtype=np.int64)),
        dtype="float64", snapshot_id="snapshot", max_tile_size=2,
        read_tile=lambda *_: None, close=lambda: None)

def test_validated_tuple_skips_repeated_string_scans():
    class CountedString(str):
        calls = 0
        def strip(self, *args):
            type(self).calls += 1
            return super().strip(*args)
    ids = tuple(CountedString(f"f{i}") for i in range(100))
    source = make_source(ids)
    metadata = capture_factor_tile_source(source)
    assert CountedString.calls == 100
    for _ in range(10):
        _assert_source_unchanged(source, metadata)
    assert CountedString.calls == 100
    source.factor_ids = tuple(reversed(ids))
    with pytest.raises(SnapshotMismatchError):
        _assert_source_unchanged(source, metadata)
    assert CountedString.calls == 200

def test_mutable_ids_and_snapshot_changes_still_rejected():
    source = make_source(["f0", "f1"])
    metadata = capture_factor_tile_source(source)
    source.factor_ids[1] = "f0"
    with pytest.raises(InvalidContractError):
        _assert_source_unchanged(source, metadata)
    source = make_source(("f0", "f1"))
    metadata = capture_factor_tile_source(source)
    source.snapshot_id = "changed"
    with pytest.raises(SnapshotMismatchError):
        _assert_source_unchanged(source, metadata)
