from __future__ import annotations

from types import SimpleNamespace

from storage.sources import data_access_source as module
from storage.sources.data_access_source import DataAccessSource


class _Store:
    def __init__(self, snapshot_id: str):
        self.snapshot_id = snapshot_id

    def describe_dataset(self, dataset, *, params=None, instrument_filter=None):
        return SimpleNamespace(snapshot_id=self.snapshot_id)


def test_snapshot_change_invalidates_all_caches(monkeypatch):
    store = _Store("snap-1")
    monkeypatch.setattr(module, "_get_store", lambda: store)
    source = DataAccessSource(dataset="demo", params={})
    source._data_snapshot_id = "snap-1"
    source._column_cache["close"] = object()
    source._panel_cache["close"] = object()
    source._lazy_bundle = object()

    store.snapshot_id = "snap-2"
    assert source.refresh_snapshot(force=True) == "snap-2"
    assert source.column_cache_stats()["cached_columns"] == 0
    assert source.column_cache_stats()["cached_panels"] == 0
    assert source._lazy_bundle is None


class _ManifestStore:
    """带 manifest_version 的假 store（不走昂贵的 describe_dataset）。"""

    def __init__(self, version: str):
        self.version = version
        self.describe_calls = 0

    def manifest_version(self, dataset, **params):
        return {
            "dataset": dataset,
            "has_manifest": True,
            "fresh": True,
            "dataset_version": "dv1",
            "partition_version": self.version,
            "file_count": 1,
        }

    def describe_dataset(self, dataset, *, params=None, instrument_filter=None):
        self.describe_calls += 1
        return SimpleNamespace(snapshot_id="snap-describe")


def test_snapshot_change_uses_cheap_manifest_token(monkeypatch):
    store = _ManifestStore("pv1")
    monkeypatch.setattr(module, "_get_store", lambda: store)
    source = DataAccessSource(dataset="demo", params={})
    source._data_snapshot_id = "snap-1"

    # 首次刷新：记录 token，不清缓存，不触发 describe
    assert source.refresh_snapshot(force=True) == "snap-1"
    assert source._manifest_token == "manifest:dv1:pv1"
    assert store.describe_calls == 0

    # token 未变：缓存保留
    source._column_cache["close"] = object()
    assert source.refresh_snapshot(force=True) == "snap-1"
    assert source.column_cache_stats()["cached_columns"] == 1
    assert store.describe_calls == 0

    # token 变化：清缓存，仍不触发 describe
    store.version = "pv2"
    assert source.refresh_snapshot(force=True) == "snap-1"
    assert source.column_cache_stats()["cached_columns"] == 0
    assert store.describe_calls == 0


def test_cache_is_lru_bounded(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_DATA_CACHE_MAX_COLUMNS", "2")
    source = DataAccessSource(dataset="demo", params={})
    source._put_cache(source._column_cache, "a", 1)
    source._put_cache(source._column_cache, "b", 2)
    source._put_cache(source._column_cache, "c", 3)
    assert list(source._column_cache) == ["b", "c"]


def test_close_rejects_future_reads(monkeypatch):
    source = DataAccessSource(dataset="demo", params={})
    source.close()
    try:
        source.refresh_snapshot(force=True)
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("closed source accepted a read")
