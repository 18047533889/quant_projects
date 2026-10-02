import threading
import time

import numpy as np
import pytest

from quant_evaluator.adapters.cos_factor_tile_source import (
    BoundCosFactor, BoundManifestHelpers, CosFactorTileSource, DataAccessReadContext,
)
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import iter_validated_factor_tiles


def _source(prefetch="auto", *, tamper=False, prefetch_workers=2,
            max_prefetch_memory_bytes=512 * 1024**2, max_source_memory_bytes=4 * 1024**3):
    records = tuple(BoundCosFactor(f"f{i}", f"cos://bucket/{i}", f"{i}" * 64,
                                   100) for i in range(5))
    times = AxisRef("time", "int64", 2, np.array([1, 2], dtype=np.int64))
    assets = AxisRef("asset", "str", 2, np.array(["A", "B"]))
    active = maximum = 0
    lock = threading.Lock()
    calls = []

    def read_factor(record, snapshot):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005)
        with lock:
            active -= 1
            calls.append(record.factor_id)
        identity = {"uri": record.uri, "sha256": record.sha256,
                    "size_bytes": record.size_bytes, "manifest_sha256": "m" * 64}
        if tamper and record.factor_id == "f1":
            identity["sha256"] = "x" * 64
        return int(record.factor_id[1:]), identity

    verified = []
    def verify(snapshot):
        assert snapshot == {"immutable": True}
        verified.append(True)

    def make_tile(start, end, selected, payloads):
        assert [r.factor_id for r in selected] == [f"f{i}" for i in range(start, end)]
        values = np.stack([np.full((2, 2), p.payload, dtype=np.float64)
                           for p in payloads], axis=-1)
        return FactorBatch(tuple(r.factor_id for r in selected), times, assets, values)

    source = CosFactorTileSource(
        records=records, time_axis=times, asset_axis=assets, dtype="float64",
        manifest_snapshot={"immutable": True}, manifest_sha256="m" * 64,
        max_tile_size=2, read_factor=read_factor, verify_manifest=verify,
        make_tile=make_tile, prefetch=prefetch,
        prefetch_workers=prefetch_workers,
        max_prefetch_memory_bytes=max_prefetch_memory_bytes,
        max_source_memory_bytes=max_source_memory_bytes,
    )
    return source, calls, verified, lambda: maximum


def test_bound_manifest_helpers_reject_noncallables():
    with pytest.raises(TypeError, match="must be callable"):
        BoundManifestHelpers(None, lambda *_: None, lambda *_: None)


def test_prefetch_auto_is_bounded_ordered_and_preserves_tiles():
    source, calls, verified, maximum = _source()
    try:
        tiles = list(iter_validated_factor_tiles(source))
        assert [tile.batch.factor_ids for tile in tiles] == [("f0", "f1"), ("f2", "f3"), ("f4",)]
        assert sorted(calls) == ["f0", "f1", "f2", "f3", "f4"]
        assert maximum() <= 2
        assert source.prefetch_mode == "auto" and source.prefetch_window == 2
        assert len(verified) == 2
    finally:
        source.close()
    source.close()


def test_prefetch_retains_two_reads_ahead_at_tile_boundary():
    source, _, _, maximum = _source()
    try:
        first = source.read_tile(0, 2)
        assert first.batch.factor_ids == ("f0", "f1")
        assert [index for index, _ in source._pending] == [2, 3]
        assert source._submitted == 4
        assert maximum() <= 2

        second = source.read_tile(2, 4)
        assert second.batch.factor_ids == ("f2", "f3")
        np.testing.assert_array_equal(second.batch.values[0, 0], [2.0, 3.0])
        assert [index for index, _ in source._pending] == [4]
    finally:
        source.close()


def test_prefetch_off_is_serial():
    source, calls, _, maximum = _source("off")
    try:
        list(iter_validated_factor_tiles(source))
        assert calls == ["f0", "f1", "f2", "f3", "f4"]
        assert maximum() == 1
        assert source.prefetch_window == 1
    finally:
        source.close()


@pytest.mark.parametrize("workers", (1, 2, 4))
def test_prefetch_worker_window_and_read_concurrency_are_configurable(workers):
    source, _, _, maximum = _source(prefetch_workers=workers,
        max_prefetch_memory_bytes=1024**3, max_source_memory_bytes=12 * 1024**3)
    try:
        first = source.read_tile(0, 2)
        assert first.batch.factor_ids == ("f0", "f1")
        assert source.prefetch_workers == workers
        assert source.prefetch_window == workers
        assert maximum() <= workers
        assert len(source._pending) <= workers
        assert source.estimated_prefetch_bytes == workers * 256 * 1024**2
    finally:
        source.close()


def test_identity_failure_fails_closed_and_close_is_idempotent():
    source, _, _, _ = _source(tamper=True)
    with pytest.raises(ValueError, match="identity"):
        source.read_tile(0, 2)
    source.close()
    source.close()
    with pytest.raises(RuntimeError, match="closed"):
        source.read_tile(0, 1)


def test_source_rejects_out_of_order_reads():
    source, _, _, _ = _source()
    try:
        with pytest.raises(ValueError, match="next bounded"):
            source.read_tile(1, 2)
    finally:
        source.close()


def test_close_after_partial_read_performs_final_manifest_verification():
    source, _, verified, _ = _source()
    source.read_tile(0, 2)
    source.close()
    assert len(verified) == 2


def test_request_snapshot_id_includes_selected_source_axes():
    left, *_ = _source()
    right, *_ = _source()
    assert left.snapshot_id == right.snapshot_id
    records = tuple(BoundCosFactor(f"g{i}", f"cos://bucket/{i}", f"{i}" * 64,
                                   100) for i in range(5))
    right.factor_ids = tuple(r.factor_id for r in records)
    right.records = records
    assert left.snapshot_id != CosFactorTileSource._request_snapshot_id(
        right.manifest_sha256, right.factor_ids, right.time_axis, right.asset_axis,
        right.dtype, right.records)
    left.close()
    right.close()


def test_source_enforces_tile_plus_prefetch_memory_budget():
    source, *_ = _source()
    source.close()
    times = AxisRef("time", "int64", 100, np.arange(100, dtype=np.int64))
    assets = AxisRef("asset", "str", 100, np.asarray([f"A{i}" for i in range(100)]))
    with pytest.raises(MemoryError, match="max_prefetch_memory_bytes"):
        CosFactorTileSource(
            records=(BoundCosFactor("f", "cos://b/f", "a" * 64, 1),),
            time_axis=times, asset_axis=assets, dtype="float64",
            manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=1,
            read_factor=lambda *_: None, verify_manifest=lambda *_: None,
            make_tile=lambda *_: None, max_prefetch_memory_bytes=1)


def test_four_workers_require_explicit_prefetch_budget_and_are_admitted_when_raised(monkeypatch):
    # Admission must fail before any worker can perform a read.
    with monkeypatch.context() as scoped:
        scoped.setattr("quant_evaluator.adapters.cos_factor_tile_source.ThreadPoolExecutor",
                       lambda **_: pytest.fail("workers created before budget rejection"))
        with pytest.raises(MemoryError, match="max_prefetch_memory_bytes"):
            _source(prefetch_workers=4)
    source, *_ = _source(prefetch_workers=4,
        max_prefetch_memory_bytes=1024**3, max_source_memory_bytes=12 * 1024**3)
    try:
        assert source.prefetch_workers == 4
        assert source.estimated_prefetch_bytes == 1024**3
    finally:
        source.close()


@pytest.mark.parametrize("workers", (0, 3, 5, True))
def test_invalid_prefetch_worker_count_is_rejected(workers):
    with pytest.raises(ValueError, match="prefetch_workers"):
        _source(prefetch_workers=workers)


def test_data_access_builder_uses_injected_helpers_without_factor_optimizer(monkeypatch):
    import builtins

    times = AxisRef("time", "int64", 2, np.array([1, 2], dtype=np.int64))
    assets = AxisRef("asset", "str", 2, np.array(["A", "B"]))
    manifest = type("Snapshot", (), {"manifest_sha256": "m" * 64,
        "factors": {"f0": {"uri": "cos://bucket/f0", "sha256": "a" * 64,
                            "bytes": 8, "verified": True}}})()
    controller_closed = []
    worker_closed = []
    verify_calls = []

    def controller_factory():
        return DataAccessReadContext(object(), "manifest", close=lambda: controller_closed.append(True))

    def factor_factory(record):
        return DataAccessReadContext(object(), "manifest", "factor", factor_params={"id": record.factor_id},
                                     close=lambda: worker_closed.append(True))

    def read_bound_manifest(store, dataset, **kwargs):
        assert dataset == "manifest"
        assert kwargs["allow_research"] is True
        return manifest

    def verify_bound_manifest_unchanged(store, dataset, snapshot, **kwargs):
        assert snapshot is manifest
        verify_calls.append(True)

    class Factor:
        source_uri, content_sha256, downloaded_bytes = "cos://bucket/f0", "a" * 64, 8
        source_etag = "etag-f0"
        table = type("Table", (), {"to_pandas": lambda self: 7})()
    bound = type("Bound", (), {"factor": Factor(), "manifest_sha256": "m" * 64})()
    def read_bound(*args, **kwargs):
        assert kwargs["manifest_snapshot"] is manifest
        assert kwargs["allow_research"] is True
        return bound
    helpers = BoundManifestHelpers(
        read_bound_manifest=read_bound_manifest, read_bound_factor=read_bound,
        verify_bound_manifest_unchanged=verify_bound_manifest_unchanged)
    original_import = builtins.__import__
    def reject_factor_optimizer(name, *args, **kwargs):
        if name == "factor_optimizer" or name.startswith("factor_optimizer."):
            raise AssertionError("QE adapter attempted to import optional factor_optimizer")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_factor_optimizer)

    def make_tile(start, end, records, payloads):
        assert [item.payload for item in payloads] == [7]
        return FactorBatch(("f0",), times, assets,
                           np.ones((2, 2, 1), dtype=np.float64))

    source = CosFactorTileSource.from_data_access(
        factor_ids=("f0",), time_axis=times, asset_axis=assets, dtype="float64",
        make_tile=make_tile, bound_manifest_helpers=helpers,
        manifest_context_factory=controller_factory,
        factor_context_factory=factor_factory, max_tile_size=1, prefetch="off")
    tile = source.read_tile(0, 1)
    assert tile.snapshot_id != "m" * 64
    assert worker_closed == [True]
    source.close()
    assert controller_closed == [True]
    assert len(verify_calls) == 2  # first read and final post-read snapshot check
    phases = source.stage_telemetry.snapshot()
    assert phases["context_setup"]["count"] == 2
    assert phases["bound_factor_read"]["count"] == 1
    assert phases["arrow_to_pandas_axis"]["count"] == 1
    assert phases["manifest_verify"]["count"] == 2
    assert all(value["total_seconds"] >= 0 for value in phases.values())


def test_stage_telemetry_keeps_fixed_phase_set_and_records_failures():
    from quant_evaluator.adapters.cos_factor_tile_source import SourceStageTelemetry
    telemetry = SourceStageTelemetry()
    with pytest.raises(OSError, match="read failed"):
        with telemetry.measure("bound_factor_read"):
            raise OSError("read failed")
    snapshot = telemetry.snapshot()
    assert tuple(snapshot) == SourceStageTelemetry.PHASES
    assert snapshot["bound_factor_read"]["count"] == 1
    with pytest.raises(ValueError, match="unknown"):
        telemetry.record("raw_uri", 1, 0.0)
    with pytest.raises(ValueError, match="unknown"):
        with telemetry.measure("raw_uri"):
            pytest.fail("unknown telemetry phase entered its context")


@pytest.mark.parametrize("count,seconds", [
    (True, 0.0), (-1, 0.0), (1.5, 0.0), (0, -1.0), (0, float("nan")),
    (0, float("inf")), (0, True),
])
def test_stage_telemetry_rejects_invalid_counts_and_durations(count, seconds):
    from quant_evaluator.adapters.cos_factor_tile_source import SourceStageTelemetry
    with pytest.raises(ValueError):
        SourceStageTelemetry().record("context_setup", count, seconds)


def test_stage_telemetry_concurrent_fixed_counts_and_detached_snapshots():
    from quant_evaluator.adapters.cos_factor_tile_source import SourceStageTelemetry
    telemetry = SourceStageTelemetry()

    def record_many():
        for _ in range(500):
            telemetry.record("bound_factor_read", 1, 0.25)

    workers = [threading.Thread(target=record_many) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    snapshot = telemetry.snapshot()
    assert snapshot["bound_factor_read"] == {"count": 4000, "total_seconds": 1000.0}
    snapshot["bound_factor_read"]["count"] = -1
    assert telemetry.snapshot()["bound_factor_read"]["count"] == 4000


def test_prefetch_future_failure_waits_for_and_retires_other_reads():
    records = tuple(BoundCosFactor(f"f{i}", f"cos://bucket/{i}", f"{i}" * 64, 100)
                    for i in range(4))
    times = AxisRef("time", "int64", 1, np.array([1], dtype=np.int64))
    assets = AxisRef("asset", "str", 1, np.array(["A"]))
    lock = threading.Lock()
    active = 0

    def read_factor(record, _snapshot):
        nonlocal active
        with lock:
            active += 1
        try:
            time.sleep(0.01)
            if record.factor_id == "f1":
                raise OSError("read failed")
            return 1, {"uri": record.uri, "sha256": record.sha256,
                       "size_bytes": record.size_bytes, "manifest_sha256": "m" * 64}
        finally:
            with lock:
                active -= 1

    source = CosFactorTileSource(
        records=records, time_axis=times, asset_axis=assets, dtype="float64",
        manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=2,
        read_factor=read_factor, verify_manifest=lambda *_: None,
        make_tile=lambda *_: None, max_source_memory_bytes=1024**3)
    with pytest.raises(OSError, match="read failed"):
        source.read_tile(0, 2)
    assert active == 0
    assert source._closed and source._executor is None and not source._pending


def test_final_manifest_failure_still_closes_source_context():
    closed = []
    source, *_ = _source()
    source._controller_close = lambda: closed.append(True)
    source.verify_manifest = lambda *_: (_ for _ in ()).throw(RuntimeError("manifest changed"))
    with pytest.raises(RuntimeError, match="manifest changed"):
        source.close()
    assert closed == [True]
    assert source._executor is None and source._closed


@pytest.mark.parametrize("status,factor_uri_matches", [
    ("materialized_not_evaluated", True),
    ("blocked", True),
    ("materialized_not_evaluated", False),
])
def test_data_access_builder_relies_on_bound_manifest_uri_sha_and_status(
        monkeypatch, status, factor_uri_matches):
    research_manifest = pytest.importorskip("factor_optimizer.research_manifest")
    import data_access.cos.remote as remote
    import data_access.cos.research as cos_research

    from types import SimpleNamespace

    manifest_uri = "cos://bucket/manifest.json"
    factor_uri = "cos://bucket/factor.parquet"
    manifest_sha = "c" * 64
    factor_sha = "a" * 64
    time_axis = AxisRef("time", "int64", 2, np.asarray([1, 2], dtype=np.int64))
    asset_axis = AxisRef("asset", "str", 1, np.asarray(["A"]))
    frame = SimpleNamespace()
    row = {"uri": factor_uri, "sha256": factor_sha, "bytes": 8,
           "verified": True, "status": status, "fe_dsl": "col('close')"}
    factors = {"f0": row}
    closes = []

    class Table:
        def __init__(self, kind): self.kind = kind
        def to_pylist(self): return [{"factors": factors}]
        def to_pandas(self): return frame

    class Store:
        def __init__(self, descriptors): self.descriptors = descriptors
        def authorize_dataset(self, _dataset): pass
        def _authorize_factor_params(self, _dataset, _params): pass
        def get_dataset(self, name): return self.descriptors[name]

    def resolve_paths(dataset, *, params=None):
        if dataset.name == "source_manifest": return [manifest_uri]
        return [factor_uri if factor_uri_matches else "cos://bucket/other.parquet"]

    def read_object(store, dataset, *, params=None, allow_research=False,
                    max_object_mib=64, **kwargs):
        if dataset == "source_manifest":
            return SimpleNamespace(source_uri=manifest_uri, content_sha256=manifest_sha,
                                   table=Table("manifest"))
        return SimpleNamespace(source_uri=factor_uri, source_etag="etag",
                               content_sha256=factor_sha, downloaded_bytes=8,
                               table=Table("factor"))

    monkeypatch.setattr(remote, "resolve_remote_paths", resolve_paths)
    monkeypatch.setattr(cos_research, "read_declared_cos_object", read_object)

    def manifest_context_factory():
        descriptor = SimpleNamespace(name="source_manifest")
        return DataAccessReadContext(Store({"source_manifest": descriptor}),
                                     "source_manifest", close=lambda: closes.append("manifest"))

    def factor_context_factory(record):
        assert record.sha256 == factor_sha and record.size_bytes == 8
        descriptor = SimpleNamespace(name="source_manifest")
        factor_descriptor = SimpleNamespace(name="factor_panel")
        return DataAccessReadContext(
            Store({"source_manifest": descriptor, "factor_panel": factor_descriptor}),
            "source_manifest", "factor_panel", factor_params={"id": "f0"},
            close=lambda: closes.append("factor"))

    def make_tile(start, end, records, payloads):
        return FactorBatch(("f0",), time_axis, asset_axis,
                           np.ones((2, 1, 1), dtype=np.float64))

    source = CosFactorTileSource.from_data_access(
        factor_ids=("f0",), time_axis=time_axis, asset_axis=asset_axis,
        dtype="float64", make_tile=make_tile,
        bound_manifest_helpers=BoundManifestHelpers(
            read_bound_manifest=research_manifest.read_bound_manifest,
            read_bound_factor=research_manifest.read_bound_factor,
            verify_bound_manifest_unchanged=research_manifest.verify_bound_manifest_unchanged),
        manifest_context_factory=manifest_context_factory,
        factor_context_factory=factor_context_factory,
        expected_manifest_sha256=manifest_sha, max_tile_size=1, prefetch="off")
    if status == "materialized_not_evaluated" and factor_uri_matches:
        tile = source.read_tile(0, 1)
        assert tile.snapshot_id != manifest_sha
        source.close()
        assert closes == ["factor", "manifest"]
    else:
        with pytest.raises(ValueError):
            source.read_tile(0, 1)
        assert source._closed
        assert closes == ["factor", "manifest"]



def test_source_memory_bound_accounts_assembly_and_validity_copies():
    records = tuple(BoundCosFactor(f"f{i}", f"cos://b/{i}", "a" * 64, 1)
                    for i in range(2))
    times = AxisRef("time", "int64", 2, np.array([1, 2], dtype=np.int64))
    assets = AxisRef("asset", "str", 2, np.array(["A", "B"]))
    args = dict(
        records=records, time_axis=times, asset_axis=assets, dtype="float64",
        manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=2,
        read_factor=lambda *_: None, verify_manifest=lambda *_: None,
        make_tile=lambda *_: None, prefetch="off",
    )
    # 3 value panels + raw/frozen bool masks, with no extra callback scratch.
    expected = 3 * (2 * 2 * 8 * 2) + (2 * 2 * 2 * 2)
    with pytest.raises(MemoryError, match="tile assembly"):
        CosFactorTileSource(**args, max_source_memory_bytes=expected - 1)
    source = CosFactorTileSource(**args, max_source_memory_bytes=expected)
    assert source.estimated_assembly_bytes == expected
    source.close()

    with_extra = CosFactorTileSource(
        **args, max_source_memory_bytes=expected + 7 * 2 * 2 * 2,
        extra_assembly_bytes_per_cell=7)
    assert with_extra.estimated_assembly_bytes == expected + 7 * 2 * 2 * 2
    with_extra.close()


def test_f61_tile16_assembly_bound_does_not_fit_four_gibibytes():
    records = tuple(BoundCosFactor(f"f{i}", f"cos://b/{i}", "a" * 64, 1)
                    for i in range(16))
    times = AxisRef("time", "int64", 2586, np.arange(2586, dtype=np.int64))
    assets = AxisRef("asset", "int64", 5461, np.arange(5461, dtype=np.int64))
    with pytest.raises(MemoryError, match="tile assembly"):
        CosFactorTileSource(
            records=records, time_axis=times, asset_axis=assets, dtype="float64",
            manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=16,
            read_factor=lambda *_: None, verify_manifest=lambda *_: None,
            make_tile=lambda *_: None, prefetch="auto",
            max_source_memory_bytes=4 * 1024**3)



@pytest.mark.parametrize("extra", [True, -1, 1.5])
def test_extra_assembly_bound_must_be_nonnegative_integer(extra):
    times = AxisRef("time", "int64", 1, np.array([1], dtype=np.int64))
    assets = AxisRef("asset", "int64", 1, np.array([1], dtype=np.int64))
    with pytest.raises(ValueError, match="extra_assembly_bytes_per_cell"):
        CosFactorTileSource(
            records=(BoundCosFactor("f", "cos://b/f", "a" * 64, 1),),
            time_axis=times, asset_axis=assets, dtype="float64",
            manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=1,
            read_factor=lambda *_: None, verify_manifest=lambda *_: None,
            make_tile=lambda *_: None, extra_assembly_bytes_per_cell=extra)


def test_source_budget_uses_available_factors_not_unused_tile_capacity():
    times = AxisRef("time", "int64", 2, np.arange(2, dtype=np.int64))
    assets = AxisRef("asset", "int64", 3, np.arange(3, dtype=np.int64))
    expected = 6 * (3 * 8 + 2)
    with CosFactorTileSource(
        records=(BoundCosFactor("f", "cos://b/f", "a" * 64, 1),),
        time_axis=times, asset_axis=assets, dtype="float64",
        manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=32,
        read_factor=lambda *_: None, verify_manifest=lambda *_: None,
        make_tile=lambda *_: None, prefetch="off",
        max_source_memory_bytes=expected,
    ) as source:
        assert source.estimated_assembly_bytes == expected
        assert source.estimated_peak_source_bytes == expected
