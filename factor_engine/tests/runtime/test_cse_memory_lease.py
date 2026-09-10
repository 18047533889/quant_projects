from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.buffer_store import GovernedBufferStore
from factor_engine.runtime.spill_store import SpillStore


class CountingLease:
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


def _attach(store, key, lease):
    target = store.capture_lease_target(key)
    assert target is not None
    assert target.bytes > 0
    assert store.attach_memory_lease(key, lease, target=target)
    del target


def _collect() -> None:
    for _ in range(3):
        gc.collect()


def test_release_waits_for_external_ndarray_and_derived_view():
    store = GovernedBufferStore()
    root = np.arange(64, dtype=np.float64)
    resident = root[4:40]
    external_view = resident[::2]
    store.put("x", resident, bytes_=resident.nbytes)
    lease = CountingLease()
    _attach(store, "x", lease)

    store.release("x")
    del resident, root
    _collect()
    assert lease.release_calls == 0

    del external_view
    _collect()
    assert lease.release_calls == 1


def test_series_numpy_view_keeps_lease_after_logical_clear():
    store = GovernedBufferStore()
    series = pd.Series(np.arange(32, dtype=np.int64))
    external = series.to_numpy(copy=False)[2:]
    store.put("series", series, bytes_=int(series.memory_usage(deep=True)))
    lease = CountingLease()
    _attach(store, "series", lease)

    store.clear()
    del series
    _collect()
    assert lease.release_calls == 0

    del external
    _collect()
    assert lease.release_calls == 1


@pytest.mark.parametrize("axis_owner", ["index_values", "multiindex_codes"])
def test_series_axis_buffer_keeps_lease_after_value_buffer_dies(axis_owner):
    if axis_owner == "index_values":
        index = pd.Index(np.arange(32, dtype=np.int64) + 100)
    else:
        index = pd.MultiIndex.from_arrays(
            [np.arange(32, dtype=np.int64), np.arange(32, dtype=np.int64) % 3]
        )
    series = pd.Series(np.arange(32, dtype=np.float64), index=index)
    external_axis = (
        series.index.to_numpy(copy=False)
        if axis_owner == "index_values"
        else series.index.codes[0]
    )
    store = GovernedBufferStore()
    store.put("series", series, bytes_=int(series.memory_usage(deep=True)))
    lease = CountingLease()
    _attach(store, "series", lease)

    store.clear()
    del series, index, store
    _collect()
    assert lease.release_calls == 0
    del external_axis
    _collect()
    assert lease.release_calls == 1


def test_replacement_does_not_release_old_generation_or_attach_to_new_one():
    store = GovernedBufferStore()
    old = np.arange(16)
    old_external = old.view()
    store.put("same", old, bytes_=old.nbytes)
    stale = store.capture_lease_target("same")
    new = np.arange(24)
    store.put("same", new, bytes_=new.nbytes)
    old_lease = CountingLease()
    assert store.attach_memory_lease("same", old_lease, target=stale)
    del stale

    new_lease = CountingLease()
    _attach(store, "same", new_lease)
    store.release("same")
    del old, new
    _collect()
    assert old_lease.release_calls == 0
    assert new_lease.release_calls == 1

    del old_external
    _collect()
    assert old_lease.release_calls == 1


@pytest.mark.parametrize("action", ["spill", "evict"])
def test_spill_and_eviction_drop_only_logical_reference(tmp_path, action):
    spill_store = SpillStore(root_dir=str(tmp_path / "spill")) if action == "spill" else None
    store = GovernedBufferStore(spill_store=spill_store)
    resident = np.arange(128, dtype=np.float64)
    external = resident.view()
    store.put("x", resident, bytes_=resident.nbytes)
    lease = CountingLease()
    _attach(store, "x", lease)

    if action == "spill":
        assert store.spill("x").status == "SPILLED"
    else:
        assert store.evict_if_over_budget(1) == resident.nbytes
    del resident
    _collect()
    assert lease.release_calls == 0

    del external
    _collect()
    assert lease.release_calls == 1


def test_store_gc_does_not_release_while_external_buffer_lives():
    store = GovernedBufferStore()
    resident = np.arange(20)
    external = resident.view()
    store.put("x", resident, bytes_=resident.nbytes)
    lease = CountingLease()
    _attach(store, "x", lease)

    del resident, store
    _collect()
    assert lease.release_calls == 0
    del external
    _collect()
    assert lease.release_calls == 1


def test_missing_or_invalid_attach_releases_transferred_lease_exactly_once():
    store = GovernedBufferStore()
    missing = CountingLease()
    assert not store.attach_memory_lease("missing", missing)
    assert missing.release_calls == 1

    store.put("native", [1, 2, 3], bytes_=24)
    unsupported = store.capture_lease_target("native")
    assert unsupported is not None
    assert unsupported.physical_ownership_supported is False

    other = GovernedBufferStore()
    other.put("native", np.arange(3), bytes_=24)
    foreign = CountingLease()
    foreign_target = other.capture_lease_target("native")
    assert foreign_target is not None
    assert not store.attach_memory_lease("native", foreign, target=foreign_target)
    assert foreign.release_calls == 1


def test_unknown_owner_bypassing_pretransfer_check_is_retained_fail_closed():
    store = GovernedBufferStore()
    store.put("native", [1, 2, 3], bytes_=24)
    target = store.capture_lease_target("native")
    assert target is not None and not target.physical_ownership_supported
    lease = CountingLease()
    assert not store.attach_memory_lease("native", lease, target=target)
    store.clear()
    del target, store
    _collect()
    assert lease.release_calls == 0


def test_partial_finalizer_registration_failure_retains_lease_fail_closed(monkeypatch):
    import factor_engine.runtime.buffer_store as buffer_store

    store = GovernedBufferStore()
    resident = np.arange(12)[2:]
    store.put("x", resident, bytes_=resident.nbytes)
    target = store.capture_lease_target("x")
    assert target is not None
    real_finalize = buffer_store.weakref.finalize
    calls = 0

    def fail_second(owner, callback):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected finalizer allocation failure")
        return real_finalize(owner, callback)

    monkeypatch.setattr(buffer_store.weakref, "finalize", fail_second)
    lease = CountingLease()
    assert not store.attach_memory_lease("x", lease, target=target)
    store.clear()
    del target, resident
    _collect()
    assert lease.release_calls == 0
