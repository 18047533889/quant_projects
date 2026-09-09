from types import SimpleNamespace
import threading

import pandas as pd
import pytest

from factor_engine.storage.sources import data_access_source as module
from factor_engine.storage.sources.data_access_source import ApprovedSnapshotMismatch, DataAccessSource


class Broker:
    def __init__(self):
        self.leases = []

    def current_read_budget(self):
        return 64 * 1024 * 1024

    def acquire_memory(self, *args, **kwargs):
        lease = SimpleNamespace(released=False)
        lease.release = lambda: setattr(lease, "released", True)
        self.leases.append(lease)
        return lease


class Handle:
    def __init__(self, digest, dataset="demo", snapshot_id="snap-a"):
        self.snapshot = SimpleNamespace(snapshot_id=snapshot_id)
        self.read_identity = None if digest is None else SimpleNamespace(
            dataset=dataset, source_snapshot=digest
        )
        self.closed = self.converted = False

    def close(self):
        self.closed = True

    def to_arrow(self):
        self.converted = True
        return object()


def series(value):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "000001.SZ")],
        names=["timestamp", "instrument"],
    )
    return pd.Series([value], index=index)


def source_for(monkeypatch, handle, fetched):
    broker = Broker()
    monkeypatch.setattr(module, "_data_cache_authority", lambda: broker)
    source = DataAccessSource(dataset="demo")
    source._max_cache_columns = 1
    source._max_cache_bytes = 64 * 1024 * 1024
    source._data_snapshot_id = "snap-a"
    source.refresh_snapshot = lambda **kwargs: source._data_snapshot_id
    source._adapter_options = lambda store: (
        SimpleNamespace(time_column="time", instrument_column="instrument"), False, None
    )
    source._resolve_columns = lambda names: (list(names), {})
    source._normalize_contract_columns = lambda values, names, **kwargs: None
    store = SimpleNamespace(read=lambda *args, **kwargs: handle)
    monkeypatch.setattr(module, "_get_store", lambda: store)
    monkeypatch.setattr(
        "data_access.read.adapters.arrow_table_to_multiindex_columns",
        lambda *args, **kwargs: dict(fetched),
    )
    return source, broker


def test_public_load_columns_keeps_hit_when_miss_evicts_it(monkeypatch):
    close, high = series(10.0), series(12.0)
    source, broker = source_for(monkeypatch, Handle("a" * 32), {"high": high})
    assert source._put_cache(source._column_cache, "close", close)
    close_lease = broker.leases[-1]

    result = source.load_columns(["close", "high"])

    assert result["close"] is close
    assert result["high"] is high
    assert list(source._column_cache) == ["high"]
    assert close_lease.released is False


def test_public_load_columns_keeps_hit_when_concurrent_cache_write_evicts_it(monkeypatch):
    close, high = series(10.0), series(12.0)
    handle = Handle("a" * 32)
    source, _ = source_for(monkeypatch, handle, {"high": high})
    assert source._put_cache(source._column_cache, "close", close)
    entered, proceed = threading.Event(), threading.Event()

    def read(*args, **kwargs):
        entered.set()
        assert proceed.wait(2)
        return handle

    monkeypatch.setattr(module, "_get_store", lambda: SimpleNamespace(read=read))
    outcome = {}
    thread = threading.Thread(
        target=lambda: outcome.update(source.load_columns(["high", "close"]))
    )
    thread.start()
    assert entered.wait(2)
    source._put_cache(source._column_cache, "other", series(99.0))
    proceed.set()
    thread.join(2)

    assert not thread.is_alive()
    assert outcome["close"] is close
    assert outcome["high"] is high


def test_public_load_columns_rejects_snapshot_change_with_hit(monkeypatch):
    handle = Handle("a" * 32, snapshot_id="snap-b")
    source, _ = source_for(monkeypatch, handle, {"high": series(12.0)})
    assert source._put_cache(source._column_cache, "close", series(10.0))

    with pytest.raises(ApprovedSnapshotMismatch, match="request.*snapshot"):
        source.load_columns(["close", "high"])

    assert handle.closed and not handle.converted
    assert not source._column_cache


@pytest.mark.parametrize(
    ("digest", "dataset"),
    [("b" * 32, "demo"), (None, "demo"), ("a" * 32, "other")],
)
def test_eager_rejects_actual_identity_before_conversion(monkeypatch, digest, dataset):
    handle = Handle(digest, dataset=dataset)
    source, _ = source_for(monkeypatch, handle, {"close": series(10.0)})
    source.bind_approved_content_digest("a" * 32)

    with pytest.raises(ApprovedSnapshotMismatch, match="read handle.*approved content"):
        source.load_columns(["close"])

    assert handle.closed and not handle.converted
    assert not source._column_cache


def test_eager_accepts_matching_actual_identity(monkeypatch):
    close, handle = series(10.0), Handle("a" * 32)
    source, _ = source_for(monkeypatch, handle, {"close": close})
    source.bind_approved_content_digest("a" * 32)

    assert source.load_columns(["close"])["close"] is close
    assert handle.converted


def test_binding_exact_content_discards_unproven_existing_cache(monkeypatch):
    source, _ = source_for(monkeypatch, Handle("a" * 32), {})
    assert source._put_cache(source._column_cache, "close", series(10.0))

    source.bind_approved_content_digest("a" * 32)

    assert not source._column_cache


def test_production_missing_snapshot_closes_handle_before_conversion(monkeypatch):
    handle = Handle("a" * 32, snapshot_id=None)
    source, _ = source_for(monkeypatch, handle, {"close": series(10.0)})
    source._data_snapshot_id = None
    source.production = True

    with pytest.raises(RuntimeError, match="production read"):
        source.load_columns(["close"])

    assert handle.closed and not handle.converted


def test_unknown_snapshot_cache_hit_cannot_mix_with_eager_read(monkeypatch):
    handle = Handle("a" * 32)
    source, _ = source_for(monkeypatch, handle, {"high": series(12.0)})
    assert source._put_cache(source._column_cache, "close", series(10.0))
    source._data_snapshot_id = None

    with pytest.raises(ApprovedSnapshotMismatch, match="request.*snapshot"):
        source.load_columns(["close", "high"])

    assert handle.closed and not handle.converted
    assert not source._column_cache


def test_concurrent_snapshot_advance_blocks_late_cache_insertion(monkeypatch):
    handle = Handle("a" * 32)
    source, _ = source_for(monkeypatch, handle, {"close": series(10.0)})
    replacement = series(99.0)

    def convert(*args, **kwargs):
        source._record_read_snapshot("snap-b")
        source._put_cache(source._column_cache, "replacement", replacement)
        return {"close": series(10.0)}

    monkeypatch.setattr(
        "data_access.read.adapters.arrow_table_to_multiindex_columns", convert
    )

    with pytest.raises(ApprovedSnapshotMismatch, match="before request results"):
        source.load_columns(["close"])

    assert list(source._column_cache) == ["replacement"]
    assert source._column_cache["replacement"] is replacement


def test_concurrent_approval_binding_blocks_unapproved_inflight_read(monkeypatch):
    handle = Handle("b" * 32)
    source, _ = source_for(monkeypatch, handle, {"close": series(10.0)})

    def convert(*args, **kwargs):
        source.bind_approved_content_digest("a" * 32)
        return {"close": series(10.0)}

    monkeypatch.setattr(
        "data_access.read.adapters.arrow_table_to_multiindex_columns", convert
    )

    with pytest.raises(ApprovedSnapshotMismatch, match="approved content changed"):
        source.load_columns(["close"])

    assert not source._column_cache


def test_load_column_uses_locked_multi_column_path(monkeypatch):
    source, _ = source_for(monkeypatch, Handle("a" * 32), {})
    expected = series(10.0)
    calls = []
    source.load_columns = lambda names: calls.append(names) or {"close": expected}

    assert source.load_column("close") is expected
    assert calls == [["close"]]


def test_approval_binding_during_normalization_blocks_cache_publication(monkeypatch):
    source, _ = source_for(monkeypatch, Handle("b" * 32), {"close": series(10.0)})

    def normalize(values, names, **kwargs):
        source.bind_approved_content_digest("a" * 32)

    source._normalize_contract_columns = normalize
    with pytest.raises(ApprovedSnapshotMismatch, match="approved content changed"):
        source.load_columns(["close"])
    assert not source._column_cache


@pytest.mark.parametrize("method", ["load_column_panel", "prefetch_panels"])
def test_approval_binding_during_panel_conversion_cannot_repopulate_cache(monkeypatch, method):
    source, _ = source_for(monkeypatch, Handle("b" * 32), {"close": series(10.0)})
    original = pd.Series.unstack

    def unstack(value, *args, **kwargs):
        source.bind_approved_content_digest("a" * 32)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "unstack", unstack)
    with pytest.raises(ApprovedSnapshotMismatch, match="panel conversion"):
        getattr(source, method)("close" if method == "load_column_panel" else ["close"])
    assert not source._panel_cache


@pytest.mark.parametrize("method", ["load_columns", "prefetch_columns", "_prefetch_lazy_bundle"])
def test_approved_legacy_lazy_requests_use_exact_checked_handle(monkeypatch, method):
    handle = Handle("b" * 32)
    source, _ = source_for(monkeypatch, handle, {"close": series(10.0)})
    source.enable_lazy_scan(True)
    source.bind_approved_content_digest("a" * 32)
    with pytest.raises(ApprovedSnapshotMismatch, match="read handle.*approved content"):
        getattr(source, method)(["close"])
    assert handle.closed and not handle.converted
    assert not source._column_cache


def test_all_cached_read_rechecks_approval_before_return(monkeypatch):
    from collections import OrderedDict

    source, _ = source_for(monkeypatch, Handle("b" * 32), {})

    class Cache(OrderedDict):
        def __getitem__(self, key):
            value = super().__getitem__(key)
            source.bind_approved_content_digest("a" * 32)
            return value

    source._column_cache = Cache(close=series(10.0))
    with pytest.raises(ApprovedSnapshotMismatch, match="cached request"):
        source.load_columns(["close"])


def test_normalization_algorithm_changes_persistent_source_identity(monkeypatch):
    source, _ = source_for(monkeypatch, Handle("a" * 32), {})
    before = source.source_dependency_hash()
    monkeypatch.setattr(module, "UNIT_NORMALIZATION_ALGORITHM_ID", "test-next-algorithm")
    assert source.source_dependency_hash() != before


def test_normalization_algorithm_change_invalidates_only_source_warm_cache(monkeypatch):
    source, _ = source_for(monkeypatch, Handle("a" * 32), {"close": series(11.0)})
    stale = series(10.0)
    source._column_cache["close"] = stale
    monkeypatch.setattr(module, "UNIT_NORMALIZATION_ALGORITHM_ID", "test-next-algorithm")

    result = source.load_columns(["close"])

    assert result["close"].iloc[0] == 11.0
    assert result["close"] is not stale
    assert source._cache_normalization_identity == "test-next-algorithm"
