from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from factor_engine.storage.sources import data_access_source as module


def test_snapshot_io_does_not_hold_cache_metadata_lock(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def describe(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(snapshot_id="old")

    store = SimpleNamespace(manifest_version=lambda *a, **k: {}, describe_dataset=describe)
    monkeypatch.setattr(module, "_get_store", lambda: store)
    source = module.DataAccessSource(dataset="test")
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(source.refresh_snapshot, force=True)
        try:
            assert entered.wait(2)
            acquired = source._cache_lock.acquire(timeout=0.2)
            if acquired:
                source._cache_lock.release()
            assert acquired, "snapshot network I/O blocks cache metadata operations"
        finally:
            release.set()
        assert task.result(timeout=3) == "old"


def test_stale_refresh_cannot_overwrite_newer_read_snapshot(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def describe(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(snapshot_id="stale")

    store = SimpleNamespace(manifest_version=lambda *a, **k: {}, describe_dataset=describe)
    monkeypatch.setattr(module, "_get_store", lambda: store)
    source = module.DataAccessSource(dataset="test")
    source._data_snapshot_id = "initial"
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(source.refresh_snapshot, force=True)
        try:
            assert entered.wait(2)
            acquired = source._cache_lock.acquire(timeout=0.2)
            assert acquired, "metadata lock is held during source I/O"
            try:
                source._record_read_snapshot("newer")
            finally:
                source._cache_lock.release()
        finally:
            release.set()
        with pytest.raises(module.ApprovedSnapshotMismatch, match="changed during snapshot refresh"):
            task.result(timeout=3)
    assert source._data_snapshot_id == "newer"
