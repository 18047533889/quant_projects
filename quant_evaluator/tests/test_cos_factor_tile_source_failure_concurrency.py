"""Deterministic failure cleanup tests for DataAccess-backed COS prefetch."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.adapters.cos_factor_tile_source import (
    BoundCosFactor, BoundManifestHelpers, CosFactorTileSource, DataAccessReadContext,
)
from quant_evaluator.contracts.factor_batch import AxisRef


def test_worker_failure_waits_for_active_sibling_and_closes_owned_contexts():
    manifest_sha = "m" * 64
    rows = {f"f{i}": {"uri": f"cos://bucket/f{i}", "sha256": str(i) * 64,
                      "bytes": 16} for i in range(2)}
    snapshot = SimpleNamespace(manifest_sha256=manifest_sha, factors=rows)
    sibling_started, fail_first, release_sibling = (
        threading.Event(), threading.Event(), threading.Event())
    controller_closed = threading.Event()
    closed_contexts, close_lock, verify_calls = [], threading.Lock(), []

    def remember_close(name):
        with close_lock:
            closed_contexts.append(name)

    def manifest_context():
        return DataAccessReadContext(
            store=object(), manifest_dataset="manifest",
            close=controller_closed.set)

    def factor_context(record):
        name = record.factor_id
        return DataAccessReadContext(
            store=object(), manifest_dataset="manifest", factor_dataset="factor",
            close=lambda: remember_close(name))

    def read_manifest(store, dataset, **kwargs):
        assert dataset == "manifest" and kwargs["allow_research"] is True
        return snapshot

    def read_factor(store, manifest_dataset, factor_dataset, factor_id, **kwargs):
        assert manifest_dataset == "manifest" and factor_dataset == "factor"
        assert kwargs["manifest_snapshot"] is snapshot
        if factor_id == "f0":
            assert sibling_started.wait(5), "sibling worker did not start"
            fail_first.set()
            raise RuntimeError("injected first-worker failure")
        sibling_started.set()
        assert release_sibling.wait(5), "test did not release sibling worker"
        factor = SimpleNamespace(
            source_uri=rows[factor_id]["uri"],
            content_sha256=rows[factor_id]["sha256"],
            downloaded_bytes=rows[factor_id]["bytes"], source_etag="etag",
            table=SimpleNamespace(to_pandas=lambda: np.array([1.0])))
        return SimpleNamespace(factor=factor, manifest_sha256=manifest_sha)

    def verify_manifest(*_args, **_kwargs):
        verify_calls.append(True)

    helpers = BoundManifestHelpers(read_manifest, read_factor, verify_manifest)
    times = AxisRef("time", "int64", 1, np.array([1], dtype=np.int64))
    assets = AxisRef("asset", "str", 1, np.array(["A"]))
    source = CosFactorTileSource.from_data_access(
        factor_ids=("f0", "f1"), time_axis=times, asset_axis=assets,
        dtype="float64", make_tile=lambda *_: pytest.fail("failed read made a tile"),
        bound_manifest_helpers=helpers, manifest_context_factory=manifest_context,
        factor_context_factory=factor_context, prefetch="auto", prefetch_workers=2)
    executor = source._executor
    read_done, close_started, close_done = (
        threading.Event(), threading.Event(), threading.Event())
    read_errors = []

    def read_tile():
        try:
            source.read_tile(0, 1)
        except BaseException as exc:
            read_errors.append(exc)
        finally:
            read_done.set()

    def close_source():
        close_started.set()
        source.close()
        close_done.set()

    reader = threading.Thread(target=read_tile, name="test-cos-source-reader")
    closer = threading.Thread(target=close_source, name="test-cos-source-closer")
    reader.start()
    try:
        assert sibling_started.wait(5), "sibling worker did not start"
        assert fail_first.wait(5), "first worker did not reach its failure"
        closer.start()
        assert close_started.wait(5)
        assert not read_done.wait(0.05)
        assert not close_done.wait(0.05)
    finally:
        release_sibling.set()
        reader.join(5)
        if closer.ident is not None:
            closer.join(5)

    assert not reader.is_alive() and not closer.is_alive()
    assert len(read_errors) == 1 and isinstance(read_errors[0], RuntimeError)
    assert str(read_errors[0]) == "injected first-worker failure"
    assert sorted(closed_contexts) == ["f0", "f1"]
    assert controller_closed.is_set()
    # One start-of-read manifest check is expected; failure must skip final verify.
    assert len(verify_calls) == 1
    assert not source._pending and source._executor is None
    assert executor is not None and all(not worker.is_alive() for worker in executor._threads)



def _source_for_close_callbacks(verify, controller_close):
    times = AxisRef("time", "int64", 1, np.array([1], dtype=np.int64))
    assets = AxisRef("asset", "str", 1, np.array(["A"]))
    return CosFactorTileSource(
        records=(BoundCosFactor("f", "cos://bucket/f", "a" * 64, 16),),
        time_axis=times, asset_axis=assets, dtype="float64",
        manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=1,
        read_factor=lambda *_: None, verify_manifest=verify,
        make_tile=lambda *_: None, prefetch="auto", prefetch_workers=2,
        _close_controller=controller_close)


def test_concurrent_normal_close_waits_for_single_cleanup_and_is_idempotent():
    verify_entered, release_verify = threading.Event(), threading.Event()
    first_close_started, first_done = threading.Event(), threading.Event()
    verify_calls, close_calls = [], []

    def verify(_snapshot):
        verify_calls.append(True)
        verify_entered.set()
        assert release_verify.wait(5), "test did not release manifest verification"

    source = _source_for_close_callbacks(verify, lambda: close_calls.append(True))

    def first_close():
        first_close_started.set()
        source.close()
        first_done.set()

    first = threading.Thread(target=first_close)
    first.start()
    second = None
    try:
        assert first_close_started.wait(5)
        assert verify_entered.wait(5), "first close did not claim verification"
        second_started, second_done = threading.Event(), threading.Event()

        def second_close():
            second_started.set()
            source.close()
            second_done.set()

        second = threading.Thread(target=second_close)
        second.start()
        assert second_started.wait(5)
        assert not second_done.wait(0.05)
    finally:
        release_verify.set()
        first.join(5)
        if second is not None:
            second.join(5)

    assert not first.is_alive()
    assert second is not None and not second.is_alive()
    source.close()
    assert verify_calls == [True]
    assert close_calls == [True]


def test_controller_close_error_is_shared_by_waiters_and_repeated_close():
    failure = RuntimeError("controller close failed")
    verify_calls, close_calls = [], []

    def close_controller():
        close_calls.append(True)
        raise failure

    source = _source_for_close_callbacks(
        lambda _snapshot: verify_calls.append(True), close_controller)

    with pytest.raises(RuntimeError) as first:
        source.close()
    with pytest.raises(RuntimeError) as repeated:
        source.close()

    assert first.value is failure and repeated.value is failure
    assert verify_calls == [True]
    assert close_calls == [True]


def test_read_error_is_preserved_when_controller_cleanup_also_fails():
    read_failure = RuntimeError("factor read failed")
    cleanup_failure = OSError("controller cleanup failed")
    verify_calls, close_calls = [], []

    def read_factor(*_args):
        raise read_failure

    def close_controller():
        close_calls.append(True)
        raise cleanup_failure

    times = AxisRef("time", "int64", 1, np.array([1], dtype=np.int64))
    assets = AxisRef("asset", "str", 1, np.array(["A"]))
    source = CosFactorTileSource(
        records=(BoundCosFactor("f", "cos://bucket/f", "a" * 64, 16),),
        time_axis=times, asset_axis=assets, dtype="float64",
        manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=1,
        read_factor=read_factor,
        verify_manifest=lambda _snapshot: verify_calls.append(True),
        make_tile=lambda *_: pytest.fail("failed read made a tile"),
        prefetch="off", _close_controller=close_controller)

    with pytest.raises(RuntimeError) as read_error:
        source.read_tile(0, 1)

    assert read_error.value is read_failure
    assert close_calls == [True]
    assert len(verify_calls) == 1
    assert source._cleanup_error is cleanup_failure
    assert source._cleanup_done.is_set()
    with pytest.raises(OSError) as cleanup_error:
        source.close()
    assert cleanup_error.value is cleanup_failure
    assert close_calls == [True]


def test_cleanup_owner_callbacks_can_reenter_close():
    source_ref, verify_calls, close_calls = {}, [], []

    def verify(snapshot):
        verify_calls.append(snapshot)
        source_ref["source"].close()

    def close_controller():
        close_calls.append(True)
        source_ref["source"].close()

    source = _source_for_close_callbacks(verify, close_controller)
    source_ref["source"] = source
    source.close()

    assert source._cleanup_done.is_set()
    assert source._cleanup_owner is None
    assert verify_calls == [{}]
    assert close_calls == [True]
    source.close()
    assert verify_calls == [{}]
    assert close_calls == [True]
