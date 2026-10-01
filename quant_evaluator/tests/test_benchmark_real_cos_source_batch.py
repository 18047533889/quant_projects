import json
import gc
import weakref
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
    reader_calls = []

    def fake_reader(name):
        def read(rows, *args, **kwargs):
            reader_calls.append((name, tuple(row[0] for row in rows)))
            return iter(())
        return read

    monkeypatch.setattr(harness.tiles, "iter_frames", fake_reader("serial"))
    monkeypatch.setattr(harness.tiles, "iter_frames_prefetched", fake_reader("prefetch"))

    def make_tile(stream, expected, dates, assets, labels, *, load_phases=None):
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
            label, "a" * 64, 1, 16, prefetch_objects=bool(index),
        )
        for index in range(2)
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
        assert [item["tile_range"] for item in source.tile_read_timings] == [[0, 1], [1, 2]]
    assert reader_calls == [
        ("serial", ("f0",)), ("serial", ("f1",)),
        ("prefetch", ("f0",)), ("prefetch", ("f1",)),
    ]


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
            harness.DEFAULT_METRICS, prefetch_objects=True,
        )
        results.append((result, timing))

    assert sources[0] is not sources[1]
    assert all(source.max_object_mib == 16 for source in sources)
    assert all(source.reads == [(0, 1), (1, 2)] for source in sources)
    assert [timing["backend_used"] for _, timing in results] == ["cpu", "cuda"]
    assert [timing["factor_tiles_processed"] for _, timing in results] == [2, 2]
    assert [timing["prefetch_objects"] for _, timing in results] == [True, True]
    assert [timing["total_wall_seconds"] for _, timing in results] == [
        timing["seconds"] for _, timing in results]


def test_run_backend_auto_validates_effective_not_requested_tile_width(monkeypatch):
    _, asset_axis, _, label, _, _ = _fixtures()
    records = tuple((f"f{i}", f"cos://test/f{i}", "0" * 64, 1) for i in range(5))
    source_rows = tuple((*row, "etag", "2" * 64) for row in records)

    def fake_evaluate(source, labels, *, metrics, backend, max_tile_size, gpu_policy):
        source.reads.extend([(0, 2), (2, 4), (4, 5)])
        return SimpleNamespace(metadata={
            "factor_tiles_processed": 3, "backend_used": "cuda",
            "effective_max_tile_size": 2,
        })

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", fake_evaluate)
    _, receipt = harness.run_backend(
        "auto", records, source_rows,
        pd.date_range("2024-01-01", periods=3, freq="B"),
        asset_axis.values, label, "a" * 64, 4, 16, GPUExecutionPolicy(),
        harness.DEFAULT_METRICS, expected_auto_cuda=True,
    )
    assert receipt["tile_ranges"] == [(0, 2), (2, 4), (4, 5)]


@pytest.mark.parametrize("source_adapter", ("legacy", "cos"))
@pytest.mark.parametrize("fail_evaluation", (False, True))
def test_run_backend_closes_both_source_adapters_on_success_and_failure(
        monkeypatch, source_adapter, fail_evaluation):
    _, asset_axis, _, label, _, _ = _fixtures()
    records = (("f0", "cos://test/f0", "0" * 64, 1),
               ("f1", "cos://test/f1", "1" * 64, 1))
    source_rows = tuple((*row, "etag", "2" * 64) for row in records)
    sources = []

    class Source:
        def __init__(self, *args, **kwargs):
            self.factor_ids = tuple(row[0] for row in records)
            self.snapshot_id = "a" * 64
            self.manifest_sha256 = "b" * 64
            self.max_tile_size = 1
            self.prefetch_mode = "auto"
            self.prefetch_window = 2
            self.prefetch_objects = False
            self.reads = []
            self.tile_read_timings = []
            self.closed = 0
            sources.append(self)
        def close(self):
            self.closed += 1

    if source_adapter == "cos":
        def make_cos(*args, **kwargs):
            source = Source()
            source.prefetch_mode = kwargs["prefetch"]
            return source
        monkeypatch.setattr(harness, "_make_cos_source", make_cos)
    else:
        monkeypatch.setattr(harness, "RealCosSource", Source)

    def fake_evaluate(source, labels, *, metrics, backend, max_tile_size, gpu_policy):
        if fail_evaluation:
            raise RuntimeError("evaluation failed")
        source.reads.extend([(0, 1), (1, 2)])
        return SimpleNamespace(metadata={"factor_tiles_processed": 2,
            "backend_used": "cpu", "effective_max_tile_size": 1})

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", fake_evaluate)
    call = lambda: harness.run_backend(
        "cpu", records, source_rows, pd.date_range("2024-01-01", periods=3, freq="B"),
        asset_axis.values, label, "a" * 64, 1, 16, GPUExecutionPolicy(),
        harness.DEFAULT_METRICS, source_adapter=source_adapter)
    if fail_evaluation:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            call()
    else:
        _, receipt = call()
        assert receipt["source_adapter"] == source_adapter
        assert receipt["source_snapshot_id"] == "a" * 64
        assert receipt["source_request_identity_sha256"]
        assert receipt["prefetch_mode"] == ("auto" if source_adapter == "cos" else "off")
    assert len(sources) == 1 and sources[0].closed == 1


def test_cos_builder_uses_registered_descriptors_and_existing_make_tile_semantics(monkeypatch):
    import data_access.core.engine as engine_module
    import data_access.registry.loader as registry_module
    import data_access.store as store_module
    from quant_evaluator.adapters.cos_factor_tile_source import (
        BoundCosFactor, DataAccessReadContext, VerifiedFactorPayload,
    )

    _, asset_axis, _, label, _, _ = _fixtures()
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    records = (("f0", "cos://test/path/f0.parquet", "0" * 64, 12),
               ("f1", "cos://test/path/f1.parquet", "1" * 64, 12))
    source_assets = ("A.SZ", "B.SH", "C.SZ", "D.SH")
    frame = pd.DataFrame({"timestamp": dates,
                          **{name: [1.0, 2.0, 3.0] for name in source_assets}})
    axis_frame = frame.set_index("timestamp")
    row = (*records[0], "etag-f0", harness.tiles.axis_hash(axis_frame))
    row1 = (*records[1], "etag-f1", harness.tiles.axis_hash(axis_frame))
    rows = (row, row1)
    contexts = []

    class Engine:
        def __init__(self, **kwargs): self.kwargs, self.closed = kwargs, False
        def close(self): self.closed = True
    class Registry(dict): pass
    class Store:
        def __init__(self, registry, engine): self.registry, self.engine = registry, engine
    monkeypatch.setattr(engine_module, "DuckDBEngine", Engine)
    monkeypatch.setattr(registry_module, "DatasetRegistry", Registry)
    monkeypatch.setattr(store_module, "DataAccessStore", Store)
    monkeypatch.setattr(harness.tiles, "_ds",
        lambda name, uri, filename, fmt: SimpleNamespace(name=name, uri=uri,
                                                          filename=filename, fmt=fmt))

    captured = {}
    class FakeSource:
        pass
    def fake_from_data_access(cls, **kwargs):
        captured.update(kwargs)
        manifest_ctx = kwargs["manifest_context_factory"]()
        factor_ctx = kwargs["factor_context_factory"](
            BoundCosFactor("f0", records[0][1], records[0][2], records[0][3]))
        contexts.extend([manifest_ctx, factor_ctx])
        assert manifest_ctx.manifest_dataset == factor_ctx.manifest_dataset == "source_manifest"
        assert factor_ctx.factor_dataset == "factor_panel"
        assert factor_ctx.store.registry["factor_panel"].uri == "cos://test/path"
        return FakeSource()
    monkeypatch.setattr(harness.CosFactorTileSource, "from_data_access",
                        classmethod(fake_from_data_access))
    captured_tile = {}
    source_frames = [frame.copy(), frame.copy()]
    source_frame_refs = [weakref.ref(item) for item in source_frames]
    payloads = [VerifiedFactorPayload(item, {"source_etag": etag})
                for item, etag in zip(source_frames, ("etag-f0", "etag-f1"))]
    del source_frames

    def consume_stream(stream, expected, received_dates, received_assets,
                       received_labels):
        iterator = iter(stream)
        first = next(iterator)
        assert first[1] == rows[0]
        pd.testing.assert_frame_equal(first[0], axis_frame, check_freq=False)
        first = None
        second = next(iterator)
        gc.collect()
        # The original payload DataFrame for factor 0 must be collectible
        # before the second factor is consumed, not merely on callback return.
        assert payloads[0] is None
        assert source_frame_refs[0]() is None
        assert second[1] == rows[1]
        pd.testing.assert_frame_equal(second[0], axis_frame, check_freq=False)
        second = None
        with pytest.raises(StopIteration):
            next(iterator)
        captured_tile.update(expected=expected, dates=received_dates,
                             assets=received_assets, labels=received_labels)
        return "batch"

    monkeypatch.setattr(harness.tiles, "make_tile", consume_stream)

    source = harness._make_cos_source(
        records, rows, dates, asset_axis.values, label, "a" * 64, 2, 16,
        prefetch="auto", prefetch_workers=4, max_prefetch_memory_mib=1024)
    batch = captured["make_tile"](
        0, 2, [BoundCosFactor(*records[0]), BoundCosFactor(*records[1])],
        payloads)
    assert batch == "batch"
    assert payloads == [None, None]
    assert source_frame_refs[0]() is None and source_frame_refs[1]() is None
    assert captured_tile["expected"] == rows
    assert captured["prefetch"] == "auto" and captured["max_tile_size"] == 2
    assert captured["max_source_memory_bytes"] == 4096 * 1024**2
    assert captured["prefetch_workers"] == 4
    assert captured["max_prefetch_memory_bytes"] == 1024**3
    assert captured["extra_assembly_bytes_per_cell"] > 0
    assert [ctx.store.engine.kwargs for ctx in contexts] == [{"threads": 1}, {"threads": 2}]
    for context in contexts:
        context.close()
        assert context.store.engine.closed


@pytest.mark.parametrize("mode", ("off", "on", "auto"))
@pytest.mark.parametrize("workers", (1, 2, 4))
def test_run_backend_forwards_explicit_cos_prefetch_policy(monkeypatch, mode, workers):
    _, asset_axis, _, label, _, _ = _fixtures()
    records = (("f0", "cos://test/f0", "0" * 64, 1),)
    source_rows = ((*records[0], "etag", "2" * 64),)
    source = SimpleNamespace(
        factor_ids=("f0",), snapshot_id="a" * 64, manifest_sha256="b" * 64,
        max_tile_size=1, prefetch_mode=mode, prefetch_window=2 if mode != "off" else 1,
        reads=[(0, 1)], tile_read_timings=[], close=lambda: None,
        max_source_memory_bytes=1024, estimated_peak_source_bytes=512)
    seen = []
    monkeypatch.setattr(harness, "_make_cos_source",
                        lambda *args, **kwargs: seen.append((kwargs["prefetch"],
                            kwargs["prefetch_workers"], kwargs["max_prefetch_memory_mib"]))
                            or source)
    monkeypatch.setattr(harness, "evaluate_factor_source_batch",
        lambda *args, **kwargs: SimpleNamespace(metadata={
            "factor_tiles_processed": 1, "backend_used": "cpu",
            "effective_max_tile_size": 1}))
    _, receipt = harness.run_backend(
        "cpu", records, source_rows, pd.date_range("2024-01-01", periods=3, freq="B"),
        asset_axis.values, label, "a" * 64, 1, 16, GPUExecutionPolicy(),
        harness.DEFAULT_METRICS, source_adapter="cos", cos_prefetch=mode,
        cos_prefetch_workers=workers, max_prefetch_memory_mib=1024)
    assert seen == [(mode, workers, 1024)]
    assert receipt["cos_prefetch"] == mode
    assert receipt["prefetch_mode"] == mode
    assert receipt["prefetch_objects"] is (mode != "off")
    assert harness._source_prefetch_report_fields(receipt) == {
        "prefetch_objects": mode != "off", "prefetch_mode": mode,
        "prefetch_window": 1 if mode == "off" else 2,
    }


def test_top_level_prefetch_fields_are_projected_from_effective_run_receipt():
    fields = harness._source_prefetch_report_fields({
        "prefetch_objects": True, "prefetch_mode": "auto", "prefetch_window": 2,
    })
    assert fields == {"prefetch_objects": True, "prefetch_mode": "auto",
                      "prefetch_window": 2}


def test_cos_prefetch_cli_conflicts_with_legacy_prefetch_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark", "--factors", "61",
        "--source-adapter", "cos", "--prefetch-objects"])
    with pytest.raises(SystemExit):
        harness.main()


def test_auto_reference_requires_opposite_order_and_matching_hashes(tmp_path):
    metric = "pearson_ic"
    digest = "a" * 64

    def report(order, shape=(2586, 5461, 61)):
        return {
            "status": "complete", "kind": "real_cos_whole_source_batch_ab.v1",
            "manifest_sha256": "b" * 64, "metric_ids": [metric],
            "factor_dtype": "float64", "tile_size": 16,
            "shape": list(shape), "run_order": list(order),
            "runs": [{"backend_used": "cpu"}, {"backend_used": "cuda"}],
            "comparison": {"pass": True, "compared_metric_count": 61,
                           "metrics": {metric: {
                               "pass": True, "artifact_kind": "scalar",
                               "cuda_values_sha256": digest,
                               "cpu_values_sha256": "c" * 64,
                               "cuda_shape": [61], "finite_value_count": 61,
                           }}},
        }

    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(report(("cpu", "cuda_strict"))), encoding="utf-8")
    right.write_text(json.dumps(report(("cuda_strict", "cpu"))), encoding="utf-8")
    assert harness.certified_cuda_hashes((left, right), (metric,), "b" * 64)["shape"] == [2586, 5461, 61]
    second_shape = (2400, 5000, 61)
    left.write_text(json.dumps(report(("cpu", "cuda_strict"), second_shape)), encoding="utf-8")
    right.write_text(json.dumps(report(("cuda_strict", "cpu"), second_shape)), encoding="utf-8")
    assert harness.certified_cuda_hashes(
        (left, right), (metric,), "b" * 64)["shape"] == list(second_shape)
    unsupported_shape = (2400, 5001, 61)
    left.write_text(json.dumps(report(("cpu", "cuda_strict"), unsupported_shape)), encoding="utf-8")
    right.write_text(json.dumps(report(("cuda_strict", "cpu"), unsupported_shape)), encoding="utf-8")
    with pytest.raises(ValueError, match="opposite-order"):
        harness.certified_cuda_hashes((left, right), (metric,), "b" * 64)
    bad = report(("cpu", "cuda_strict"))
    left.write_text(json.dumps(report(("cpu", "cuda_strict"))), encoding="utf-8")
    right.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="opposite-order"):
        harness.certified_cuda_hashes((left, right), (metric,), "b" * 64)
    bad["run_order"] = ["cuda_strict", "cpu"]
    bad["comparison"]["metrics"][metric]["cuda_values_sha256"] = "d" * 64
    right.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        harness.certified_cuda_hashes((left, right), (metric,), "b" * 64)



def test_observation_count_hashes_and_auto_reference_compatibility():
    counts = np.asarray([3, 2], dtype=np.int64)
    bundle = SimpleNamespace(
        factor_ids=("f0", "f1"),
        metadata={"source_request_fingerprint": "a" * 64},
        scalar_metrics={"pearson_ic": np.asarray([0.1, 0.2])},
        observation_counts={"pearson_ic": counts},
    )
    compared = harness.compare(bundle, bundle, harness.PEARSON_SINGLE)
    metric = compared["metrics"]["pearson_ic"]
    expected_hash = harness.hashlib.sha256(counts.tobytes()).hexdigest()
    assert metric["cpu_observation_counts_sha256"] == expected_hash
    assert metric["cuda_observation_counts_sha256"] == expected_hash

    reference_metric = {
        "artifact_kind": "scalar", "cuda_shape": [2], "finite_value_count": 2,
        "cuda_values_sha256": harness.hashlib.sha256(
            np.ascontiguousarray(bundle.scalar_metrics["pearson_ic"]).tobytes()
        ).hexdigest(),
        "cuda_observation_counts_sha256": expected_hash,
    }
    reference = {"comparison": {"metrics": {"pearson_ic": reference_metric}}}
    assert harness.verify_auto_against_reference(
        bundle, harness.PEARSON_SINGLE, reference)["pass"]
    reference_metric["cuda_observation_counts_sha256"] = "0" * 64
    assert not harness.verify_auto_against_reference(
        bundle, harness.PEARSON_SINGLE, reference)["pass"]
    del reference_metric["cuda_observation_counts_sha256"]
    assert harness.verify_auto_against_reference(
        bundle, harness.PEARSON_SINGLE, reference)["pass"]


def test_preflight_enforces_ram_and_bounded_disk_headroom(monkeypatch, tmp_path):
    monkeypatch.setattr(harness.tiles, "cos_cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        harness.shutil, "disk_usage",
        lambda path: SimpleNamespace(free=8 * 1024**3),
    )
    monkeypatch.setattr(harness, "available_ram_bytes",
                        lambda: 40 * 1024**3)
    assert harness.preflight(128, 4096)["pass"] is True
    assert harness.preflight(128, 4096, 37 * 1024)["pass"] is False

    monkeypatch.setattr(harness, "available_ram_bytes",
                        lambda: 31 * 1024**3)
    assert harness.preflight(128, 4096)["pass"] is False


def test_compare_reads_fingerprint_from_public_bundle_metadata():
    def bundle(fingerprint="a" * 64):
        return SimpleNamespace(
            factor_ids=("f0", "f1"),
            metadata={"source_request_fingerprint": fingerprint},
            scalar_metrics={name: np.array([0.1, 0.2])
                            for name in ("pearson_ic",)},
            observation_counts={name: np.array([3, 3])
                                for name in ("pearson_ic",)},
        )

    result = harness.compare(bundle(), bundle(), harness.PEARSON_SINGLE)
    assert result["pass"] is True
    assert result["compared_factor_count"] == 2
    assert result["compared_metric_count"] == 2
    with pytest.raises(ValueError, match="fingerprints differ"):
        harness.compare(bundle(), bundle("b" * 64), harness.PEARSON_SINGLE)


def test_auto_direct_comparison_rejects_same_shape_count_mismatch():
    def bundle(counts):
        return SimpleNamespace(
            factor_ids=("f0", "f1"),
            metadata={"source_request_fingerprint": "a" * 64},
            scalar_metrics={"pearson_ic": np.asarray([0.1, 0.2])},
            observation_counts={"pearson_ic": np.asarray(counts, dtype=np.int64)},
        )

    result = harness.compare(
        bundle([3, 3]), bundle([3, 2]), harness.PEARSON_SINGLE)
    metric = result["metrics"]["pearson_ic"]
    assert metric["observation_counts_shape_valid"] is True
    assert metric["observation_counts_equal"] is False
    assert result["pass"] is False


def test_compare_pearson_chain_checks_full_series_counts_masks_and_fingerprint():
    factors = ("f0", "f1")
    scalars = {
        metric: np.asarray([0.1, 0.2])
        for metric in ("pearson_ic", "pearson_ic_std", "pearson_ic_ir")
    }
    series = np.asarray([[0.1, np.nan], [0.2, 0.3], [np.nan, 0.4]])
    def bundle(*, values=series, counts=None, fingerprint="a" * 64):
        return SimpleNamespace(
            factor_ids=factors,
            metadata={"source_request_fingerprint": fingerprint},
            scalar_metrics=scalars,
            series_metrics={"pearson_ic_series": values},
            observation_counts={
                metric: np.asarray([2, 2]) for metric in harness.PEARSON_CHAIN
            } if counts is None else counts,
        )

    result = harness.compare(
        bundle(), bundle(), harness.PEARSON_CHAIN, expected_days=3)
    assert result["pass"]
    assert result["compared_metric_count"] == 12
    assert result["metrics"]["pearson_ic_series"]["artifact_kind"] == "series"
    assert result["metrics"]["pearson_ic_series"]["cpu_shape"] == [3, 2]
    assert result["metrics"]["pearson_ic_series"]["finite_value_count"] == 4
    assert len(result["metrics"]["pearson_ic_series"]["cpu_values_sha256"]) == 64

    wrong_mask = series.copy()
    wrong_mask[0, 1] = 0.0
    assert not harness.compare(
        bundle(), bundle(values=wrong_mask), harness.PEARSON_CHAIN,
        expected_days=3)["pass"]
    assert not harness.compare(
        bundle(), bundle(values=series[:2]), harness.PEARSON_CHAIN,
        expected_days=3)["pass"]
    wrong_counts = {metric: np.asarray([2, 2]) for metric in harness.PEARSON_CHAIN}
    wrong_counts["pearson_ic_series"] = np.asarray([2, 1])
    assert not harness.compare(
        bundle(), bundle(counts=wrong_counts), harness.PEARSON_CHAIN,
        expected_days=3)["pass"]
    with pytest.raises(ValueError, match="fingerprints differ"):
        harness.compare(
            bundle(fingerprint="z" * 64), bundle(fingerprint="z" * 64),
            harness.PEARSON_CHAIN, expected_days=3)


def test_pearson_chain_rejects_gpu_worker_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "benchmark", "--factors", "61", "--metrics",
        ",".join(harness.PEARSON_CHAIN), "--gpu-worker", "--output", "/tmp/unused.json"])
    with pytest.raises(SystemExit):
        harness.main()


def test_pearson_chain_benchmark_accepts_only_complete_permutations():
    assert harness.is_pearson_chain(harness.PEARSON_CHAIN)
    assert harness.is_pearson_chain(tuple(reversed(harness.PEARSON_CHAIN)))
    assert not harness.is_pearson_chain(harness.PEARSON_CHAIN[:-1])
    assert not harness.is_pearson_chain(("pearson_ic",) * 4)


def test_all_source_benchmark_profile_covers_public_source_metric_set():
    from quant_evaluator.api.factor_source import _SOURCE_METRICS

    assert len(harness.ALL_SOURCE_METRICS) == len(_SOURCE_METRICS) == 15
    assert frozenset(harness.ALL_SOURCE_METRICS) == _SOURCE_METRICS


def test_gpu_tile_width_ab_interleaves_and_checks_every_run(monkeypatch, tmp_path):
    reports, widths = [], []
    assert harness.DEFAULT_METRICS == ("rank_ic", "quantile_spread", "factor_turnover_rate")
    monkeypatch.setattr("sys.argv", ["benchmark", "--factors", "61",
                                  "--gpu-tile-widths", "8", "16",
                                  "--metrics", "pearson_ic", "--output", str(tmp_path / "summary.json")])
    monkeypatch.setattr(harness, "preflight", lambda *a: {"pass": True})
    def prepare_axes(args, directory):
        index = harness.Path(directory) / "source-axis-index.json"
        index.write_text("bounded-index", encoding="utf-8")
        return index, (3, 4, 2)
    monkeypatch.setattr(harness.source_width_preparation, "prepare_axis_index", prepare_axes)
    forwarded_indexes = []

    def fake_run(command, **kwargs):
        assert command[command.index("--metrics") + 1] == "pearson_ic"
        forwarded_indexes.append(command[command.index("--axis-index") + 1])
        width = int(command[command.index("--tile-size") + 1])
        widths.append(width)
        output = harness.Path(command[command.index("--output") + 1])
        receipt = {
            "kind": "real_cos_source_gpu_worker.v1",
            "tile_size": width, "metric_ids": ["pearson_ic"], "manifest_sha256": "a" * 64,
            "shape": [3, 4, 2], "factor_ids": ["f0", "f1"],
            "source_request_fingerprint": "b" * 64,
            "run": {"seconds": width / 10, "backend_used": "cuda",
                    "prefetch_objects": True, "prefetch_mode": "auto",
                    "prefetch_window": 2},
            "scalar_metrics": {name: [0.1, 0.2] for name in ("pearson_ic",)},
            "observation_counts": {name: [3, 3] for name in ("pearson_ic",)},
        }
        output.write_text(json.dumps(receipt), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    monkeypatch.setattr(harness, "emit_report",
                        lambda report, output: reports.append(json.loads(json.dumps(report))))
    harness.main()
    assert widths == [8, 16, 16, 8]
    assert len(set(forwarded_indexes)) == 1
    assert [item["status"] for item in reports] == ["partial"] * 4 + ["complete"]
    assert len(reports[-1]["comparisons_to_first_run"]) == 3
    assert all(item["pass"] for item in reports[-1]["comparisons_to_first_run"])
    assert reports[-1]["median_seconds_by_width"] == {"8": 0.8, "16": 1.6}
    assert reports[-1]["metric_ids"] == ["pearson_ic"]
    assert reports[-1]["prefetch_objects"] is True
    assert reports[-1]["prefetch_mode"] == "auto"
    assert reports[-1]["prefetch_window"] == 2
    assert all("factor_ids" not in item and "scalar_metrics" not in item
               and "observation_counts" not in item for item in reports[-1]["runs"])


def test_gpu_tile_width_ab_failure_keeps_partial_receipt(monkeypatch, tmp_path):
    args = SimpleNamespace(
        gpu_tile_widths=(8, 16), factors=61, days=0, assets=5500,
        max_object_mib=128, max_total_mib=4096, axis_index=None,
        output=tmp_path / "summary.json",
    )
    monkeypatch.setattr(harness, "preflight", lambda *a: {"pass": True})
    def prepare_axes(args, directory):
        index = harness.Path(directory) / "source-axis-index.json"
        index.write_text("bounded-index", encoding="utf-8")
        return index, (3, 4, 61)
    monkeypatch.setattr(harness.source_width_preparation, "prepare_axis_index", prepare_axes)
    monkeypatch.setattr(
        harness.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=9, stderr="private stderr cos://secret", stdout="private stdout"),
    )
    with pytest.raises(SystemExit) as error:
        harness.run_gpu_tile_width_ab(args, harness.DEFAULT_METRICS)
    assert error.value.code == 1
    receipt = json.loads(args.output.read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted"
    assert receipt["runs"] == []
    assert receipt["failure"] == {"category": "worker_error", "worker_index": 0, "exit_code": 9}
    assert "private" not in json.dumps(receipt) and "cos://secret" not in json.dumps(receipt)
    assert "median_seconds_by_width" not in receipt
def test_cos_width8_over_default_source_budget_rejects_before_worker(monkeypatch, tmp_path):
    args = SimpleNamespace(
        gpu_tile_widths=(4, 8), factors=48, days=2586, assets=5461,
        max_object_mib=128, max_total_mib=4096, axis_index=None,
        max_source_memory_mib=4096, max_prefetch_memory_mib=512,
        cos_prefetch_workers=2, cos_prefetch="auto", source_adapter="cos",
        output=tmp_path / "summary.json")
    monkeypatch.setattr(harness, "preflight", lambda *a: {"pass": True})
    def prepare_axes(_args, directory):
        index = harness.Path(directory) / "source-axis-index.json"
        index.write_text("bounded-index", encoding="utf-8")
        return index, (2586, 5461, 48)
    monkeypatch.setattr(harness.source_width_preparation, "prepare_axis_index", prepare_axes)
    monkeypatch.setattr(harness.subprocess, "run",
                        lambda *a, **k: pytest.fail("worker launched before width admission"))
    with pytest.raises(SystemExit) as error:
        harness.run_gpu_tile_width_ab(args, harness.DEFAULT_METRICS)
    assert error.value.code == 1
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["status"] == "resource_rejected"
    assert report["failure"] == {
        "category": "source_memory_budget", "rejected_widths": [8]}
    assert report["source_memory_preflight"]["4"]["admitted"] is True
    assert report["source_memory_preflight"]["8"]["admitted"] is False


def test_source_axis_index_roundtrip_binds_manifest_selection_and_factor_axes(tmp_path):
    times, asset_axis, _, _, records, source_rows = _fixtures()
    index = harness.tiles.build_axis_index(
        "a" * 64, records, times, asset_axis.values, source_rows)
    path = tmp_path / "axes.json"
    harness.tiles.write_axis_index_atomic(path, index)
    dates, assets, restored_rows = harness.tiles.read_axis_index(
        path, "a" * 64, records)
    assert dates.equals(pd.DatetimeIndex(times))
    assert assets == set(asset_axis.values)
    assert restored_rows == source_rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0][2] = "f" * 64
    payload["sources"][0][2] = "f" * 64
    body = {key: value for key, value in payload.items() if key != "body_sha256"}
    payload["body_sha256"] = harness.hashlib.sha256(
        harness.tiles._axis_index_bytes(body)).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selection or manifest changed"):
        harness.tiles.read_axis_index(path, "a" * 64, records)
    payload = json.loads(path.read_text(encoding="utf-8"))
    index = harness.tiles.build_axis_index(
        "a" * 64, records, times, asset_axis.values, source_rows)
    harness.tiles.write_axis_index_atomic(path, index)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dates_ns"][0] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="axis index checksum mismatch"):
        harness.tiles.read_axis_index(path, "a" * 64, records)

def test_prepare_axis_index_intersects_and_builds_once(monkeypatch, tmp_path):
    prep = harness.source_width_preparation
    times, asset_axis, _, _, records, source_rows = _fixtures()
    manifest_sha = "a" * 64
    counts = {"manifest": 0, "selection": 0, "frames": 0, "intersect": 0,
              "build": 0, "write": 0, "read": 0}
    monkeypatch.setattr(prep.tiles, "MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(prep.tiles, "read_manifest",
                        lambda sha: counts.__setitem__("manifest", counts["manifest"] + 1) or object())
    monkeypatch.setattr(prep.tiles, "select_source_records",
                        lambda *a: counts.__setitem__("selection", counts["selection"] + 1) or records)
    monkeypatch.setattr(prep.tiles, "iter_frames",
                        lambda *a, **k: counts.__setitem__("frames", counts["frames"] + 1) or iter(()))
    def intersect(stream, count):
        counts["intersect"] += 1
        assert count == len(records)
        return times, set(asset_axis.values), source_rows
    monkeypatch.setattr(prep.tiles, "intersect_axes", intersect)
    monkeypatch.setattr(prep.tiles, "build_axis_index",
                        lambda *a: counts.__setitem__("build", counts["build"] + 1) or {"kind": "index"})
    def write(path, index):
        counts["write"] += 1
        path.write_text("verified-index", encoding="utf-8")
    monkeypatch.setattr(prep.tiles, "write_axis_index_atomic", write)
    def read(path, sha, selected):
        counts["read"] += 1
        assert sha == manifest_sha and selected == records
        return times, set(asset_axis.values), source_rows
    monkeypatch.setattr(prep.tiles, "read_axis_index", read)
    monkeypatch.setattr(prep.tiles, "load_labels", lambda dates, assets, days, nassets:
                        (dates, sorted(assets)[:nassets], object()))
    args = SimpleNamespace(axis_index=None, factors=2, max_object_mib=128,
                           max_total_mib=4096, days=0, assets=4)
    path, shape = prep.prepare_axis_index(args, tmp_path)
    assert path.is_file() and shape == (3, 4, 2)
    assert counts == {"manifest": 1, "selection": 1, "frames": 1, "intersect": 1,
                      "build": 1, "write": 1, "read": 1}

def test_prepare_axis_index_reuses_verified_index_without_intersection(monkeypatch, tmp_path):
    prep = harness.source_width_preparation
    times, asset_axis, _, _, records, source_rows = _fixtures()
    index_path = tmp_path / "existing.json"
    index_path.write_text("existing", encoding="utf-8")
    calls = []
    monkeypatch.setattr(prep.tiles, "MANIFEST_SHA256", "a" * 64)
    monkeypatch.setattr(prep.tiles, "read_manifest", lambda sha: object())
    monkeypatch.setattr(prep.tiles, "select_source_records", lambda *a: records)
    monkeypatch.setattr(prep.tiles, "read_axis_index",
                        lambda path, sha, selected: calls.append((path, sha, selected)) or
                        (times, set(asset_axis.values), source_rows))
    monkeypatch.setattr(prep.tiles, "intersect_axes",
                        lambda *a: pytest.fail("verified index must skip intersection"))
    monkeypatch.setattr(prep.tiles, "build_axis_index",
                        lambda *a: pytest.fail("verified index must skip build"))
    monkeypatch.setattr(prep.tiles, "load_labels", lambda dates, assets, days, nassets:
                        (dates, sorted(assets)[:nassets], object()))
    args = SimpleNamespace(axis_index=index_path, factors=2, max_object_mib=128,
                           max_total_mib=4096, days=0, assets=4)
    path, shape = prep.prepare_axis_index(args, tmp_path)
    assert path == index_path and shape == (3, 4, 2)
    assert calls == [(index_path, "a" * 64, records)]

def test_prepare_axis_index_rejects_tampered_axis_index(monkeypatch, tmp_path):
    prep = harness.source_width_preparation
    times, asset_axis, _, _, records, source_rows = _fixtures()
    index = harness.tiles.build_axis_index(
        "a" * 64, records, times, asset_axis.values, source_rows)
    path = tmp_path / "tampered.json"
    harness.tiles.write_axis_index_atomic(path, index)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dates_ns"][0] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(prep.tiles, "MANIFEST_SHA256", "a" * 64)
    monkeypatch.setattr(prep.tiles, "read_manifest", lambda sha: object())
    monkeypatch.setattr(prep.tiles, "select_source_records", lambda *a: records)
    monkeypatch.setattr(prep.tiles, "intersect_axes",
                        lambda *a: pytest.fail("tampered index must not rebuild silently"))
    args = SimpleNamespace(axis_index=path, factors=2, max_object_mib=128,
                           max_total_mib=4096, days=0, assets=4)
    with pytest.raises(ValueError, match="axis index checksum mismatch"):
        prep.prepare_axis_index(args, tmp_path)

@pytest.mark.parametrize("stderr,timeout,category", [
    ("private cos://bucket/key OutOfMemoryError", False, "gpu_out_of_memory"),
    ("private /srv/data MemoryError", False, "source_memory_admission"),
    ("private path: insufficient RAM or COS cache disk headroom", False,
     "resource_preflight"),
    ("private cos://bucket/key trace", False, "worker_error"),
    ("private stderr", True, "worker_timeout"),
])
def test_worker_failure_categories_never_include_private_output(stderr, timeout, category):
    safe = harness._safe_worker_failure(stderr, timeout=timeout)
    assert safe == category
    assert "private" not in safe and "cos://" not in safe and "/srv/" not in safe
def test_f8_rank_pair_profile_compares_full_scalar_and_series_outputs():
    factor_ids = tuple(f"f{index}" for index in range(8))
    scalar = np.linspace(-0.2, 0.2, 8)
    series = np.arange(24, dtype=np.float64).reshape(3, 8)

    def bundle(*, series_values=series):
        return SimpleNamespace(
            factor_ids=factor_ids,
            metadata={"source_request_fingerprint": "a" * 64},
            scalar_metrics={"rank_ic": scalar},
            series_metrics={"rank_ic_series": series_values},
            observation_counts={
                "rank_ic": np.asarray([3] * 8),
                "rank_ic_series": np.asarray([3] * 8),
            },
        )

    result = harness.compare(bundle(), bundle(), harness.RANK_PAIR, expected_days=3)
    assert result["pass"]
    assert result["compared_factor_count"] == 8
    assert result["compared_metric_count"] == 8 + 3 * 8
    assert result["metrics"]["rank_ic_series"]["artifact_kind"] == "series"
    changed = series.copy()
    changed[0, 0] = np.nan
    assert not harness.compare(
        bundle(), bundle(series_values=changed), harness.RANK_PAIR,
        expected_days=3)["pass"]


def test_f8_auto_reference_requires_complete_opposite_order_pair(tmp_path):
    metrics = {
        "rank_ic": {
            "pass": True, "cpu_values_sha256": "a" * 64,
            "cuda_values_sha256": "b" * 64, "cuda_shape": [8],
            "finite_value_count": 8,
        },
        "rank_ic_series": {
            "pass": True, "cpu_values_sha256": "c" * 64,
            "cuda_values_sha256": "d" * 64, "cuda_shape": [2586, 8],
            "finite_value_count": 2586 * 8,
        },
    }
    comparison = {"pass": True, "compared_metric_count": 8 + 2586 * 8,
                  "metrics": metrics}
    common = {
        "status": "complete", "kind": "real_cos_whole_source_batch_ab.v1",
        "manifest_sha256": "e" * 64, "shape": list(harness.F8_SOURCE_SHAPE),
        "factor_dtype": "float64", "tile_size": 2,
        "metric_ids": list(harness.RANK_PAIR), "comparison": comparison,
        "runs": [{"backend_used": "cpu"}, {"backend_used": "cuda"}],
        "source_adapter": "cos", "cos_prefetch": "auto",
        "prefetch_objects": True, "prefetch_mode": "auto", "prefetch_window": 2,
    }
    reports = []
    for index, order in enumerate((
            ["cpu", "cuda_strict"], ["cuda_strict", "cpu"])):
        report = dict(common, run_order=order)
        path = tmp_path / f"f8-ab-{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        reports.append(path)

    reference = harness.certified_cuda_hashes(
        tuple(reports), harness.RANK_PAIR, "e" * 64)
    assert reference["shape"] == list(harness.F8_SOURCE_SHAPE)


@pytest.mark.parametrize("profile_args", [
    ["--factors", "8", "--metrics", "rank-pair"],
    ["--factors", "8", "--metrics", "rank-pair", "--days", "2586",
     "--assets", "5461", "--tile-size", "4"],
])
def test_f8_source_benchmark_rejects_unbounded_profile(monkeypatch, profile_args):
    monkeypatch.setattr("sys.argv", ["benchmark", *profile_args])
    with pytest.raises(SystemExit):
        harness.main()


@pytest.mark.parametrize("width", [2, 8])
def test_f8_source_benchmark_admits_only_bounded_ab_before_preflight(monkeypatch, width):
    monkeypatch.setattr("sys.argv", ["benchmark", "--factors", "8",
        "--metrics", "rank-pair", "--days", "2586", "--assets", "5461",
        "--tile-size", str(width)])
    calls = []
    def denied(*args):
        calls.append(args)
        return {"pass": False}
    monkeypatch.setattr(harness, "preflight", denied)
    with pytest.raises(SystemExit, match="insufficient RAM or COS cache disk headroom"):
        harness.main()
    assert len(calls) == 1


@pytest.mark.parametrize("mode", [
    ["--verify-auto"],
    ["--auto-references", "a.json", "b.json", "--output", "out.json"],
])
def test_exploratory_f8_tile8_cannot_enter_certified_auto_modes(monkeypatch, mode):
    monkeypatch.setattr("sys.argv", ["benchmark", "--factors", "8",
        "--metrics", "rank-pair", "--days", "2586", "--assets", "5461",
        "--tile-size", "8", *mode])
    monkeypatch.setattr(harness, "preflight", lambda *a: pytest.fail("unexpected preflight"))
    monkeypatch.setattr(harness.tiles, "read_manifest", lambda *a: pytest.fail("unexpected COS read"))
    with pytest.raises(SystemExit) as exc:
        harness.main()
    assert exc.value.code == 2


def test_cap8_auto_tile2_references_reject_wrong_cli_tile_before_io(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["benchmark", "--factors", "8", "--metrics",
        "rank-pair", "--days", "2586", "--assets", "5461", "--tile-size", "2",
        "--source-adapter", "cos", "--auto-cap8-tile2-references", "a.json", "b.json",
        "--output", str(tmp_path / "out.json")])
    monkeypatch.setattr(harness, "preflight", lambda *a: pytest.fail("unexpected preflight"))
    with pytest.raises(SystemExit) as exc:
        harness.main()
    assert exc.value.code == 2


def test_cap8_auto_mode_requires_cos_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["benchmark", "--factors", "8", "--metrics",
        "rank-pair", "--days", "2586", "--assets", "5461", "--tile-size", "8",
        "--auto-cap8-tile2-references", "a.json", "b.json", "--output",
        str(tmp_path / "out.json")])
    monkeypatch.setattr(harness, "preflight", lambda *a: pytest.fail("unexpected preflight"))
    with pytest.raises(SystemExit) as exc:
        harness.main()
    assert exc.value.code == 2
