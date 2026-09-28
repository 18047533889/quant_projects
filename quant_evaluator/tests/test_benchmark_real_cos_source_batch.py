from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness


def _fixtures():
    times = pd.date_range("2024-01-01", periods=3, freq="B")
    time_axis = AxisRef("time", "datetime64[ns]", 3,
                        times.to_numpy(dtype="datetime64[ns]"))
    asset_axis = AxisRef("asset", "str", 4,
                         np.asarray(["A", "B", "C", "D"]))
    values = np.arange(3 * 4 * 2, dtype=np.float64).reshape(3, 4, 2)
    batch = FactorBatch(("f0", "f1"), time_axis, asset_axis, values)
    label = LabelBundle(
        "ret", np.ones((3, 4), dtype=np.float64), 1,
        decision_time=tuple(times), label_start_time=tuple(times),
        label_end_time=tuple(times + pd.Timedelta(days=1)),
        asset_axis=asset_axis,
    )
    records = (
        ("f0", "cos://test/f0", "0" * 64, 1),
        ("f1", "cos://test/f1", "1" * 64, 1),
    )
    source_rows = tuple((*row, "etag-" + row[0], "2" * 64) for row in records)
    return times, asset_axis, batch, label, records, source_rows


def test_real_cos_source_repeats_share_snapshot_and_preserve_tile_axes(monkeypatch):
    times, asset_axis, batch, label, records, source_rows = _fixtures()
    monkeypatch.setattr(harness.tiles, "iter_frames", lambda *args, **kwargs: iter(()))

    def make_tile(stream, expected, dates, assets, labels):
        ids = tuple(row[0] for row in expected)
        indices = [batch.factor_ids.index(fid) for fid in ids]
        tile_batch = FactorBatch(
            ids, batch.time_axis, batch.asset_axis,
            batch.values[:, :, indices],
        )
        return tile_batch

    monkeypatch.setattr(harness.tiles, "make_tile", make_tile)
    sources = [
        harness.RealCosSource(
            records, source_rows, pd.DatetimeIndex(times), asset_axis.values,
            label, "a" * 64, 1, 16,
        )
        for _ in range(2)
    ]
    assert sources[0] is not sources[1]
    assert sources[0].snapshot_id == sources[1].snapshot_id
    for source in sources:
        for start in range(2):
            tile = source.read_tile(start, start + 1)
            assert tile.snapshot_id == source.snapshot_id
            assert tile.batch.factor_ids == (f"f{start}",)
            assert np.array_equal(tile.batch.time_axis.values, source.time_axis.values)
            assert np.array_equal(tile.batch.asset_axis.values, source.asset_axis.values)
        assert source.reads == [(0, 1), (1, 2)]


def test_run_backend_forwards_object_cap_and_validates_receipt(monkeypatch):
    _, asset_axis, _, label, records, source_rows = _fixtures()
    sources = []

    def fake_evaluate(source, labels, *, metrics, backend, max_tile_size, gpu_policy):
        sources.append(source)
        for start in range(0, len(source.factor_ids), max_tile_size):
            source.reads.append(
                (start, min(start + max_tile_size, len(source.factor_ids))))
        return SimpleNamespace(metadata={
            "factor_tiles_processed": len(source.reads),
            "backend_used": "cuda" if backend == "cuda_strict" else "cpu",
        })

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", fake_evaluate)
    results = []
    for backend in ("cpu", "cuda_strict"):
        result, timing = harness.run_backend(
            backend, records, source_rows, pd.date_range("2024-01-01", periods=3, freq="B"),
            asset_axis.values, label, "a" * 64, 1, 16, GPUExecutionPolicy(),
        )
        results.append((result, timing))

    assert sources[0] is not sources[1]
    assert all(source.max_object_mib == 16 for source in sources)
    assert all(source.reads == [(0, 1), (1, 2)] for source in sources)
    assert [timing["backend_used"] for _, timing in results] == ["cpu", "cuda"]
    assert [timing["factor_tiles_processed"] for _, timing in results] == [2, 2]


def test_preflight_enforces_ram_and_bounded_disk_headroom(monkeypatch, tmp_path):
    monkeypatch.setattr(harness.tiles, "cos_cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        harness.shutil, "disk_usage",
        lambda path: SimpleNamespace(free=8 * 1024**3),
    )
    monkeypatch.setattr(harness, "available_ram_bytes",
                        lambda: 40 * 1024**3)
    assert harness.preflight(128, 4096)["pass"] is True

    monkeypatch.setattr(harness, "available_ram_bytes",
                        lambda: 31 * 1024**3)
    assert harness.preflight(128, 4096)["pass"] is False


def test_compare_reads_fingerprint_from_public_bundle_metadata():
    def bundle(fingerprint="a" * 64):
        return SimpleNamespace(
            factor_ids=("f0", "f1"),
            metadata={"source_request_fingerprint": fingerprint},
            scalar_metrics={name: np.array([0.1, 0.2])
                            for name in harness.METRICS},
            observation_counts={name: np.array([3, 3])
                                for name in harness.METRICS},
        )

    result = harness.compare(bundle(), bundle())
    assert result["pass"] is True
    assert result["compared_factor_count"] == 2
    assert result["compared_metric_count"] == 6
    with pytest.raises(ValueError, match="fingerprints differ"):
        harness.compare(bundle(), bundle("b" * 64))
