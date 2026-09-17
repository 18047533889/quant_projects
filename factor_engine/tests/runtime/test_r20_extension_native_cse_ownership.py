from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.buffer_store import GovernedBufferStore


class Lease:
    def __init__(self): self.release_calls = 0
    def release(self): self.release_calls += 1


def collect():
    for _ in range(3): gc.collect()


@pytest.mark.parametrize("dtype", ["Int64", "boolean"])
def test_numpy_backed_extension_array_tracks_external_physical_buffer(dtype):
    frame = pd.DataFrame({"x": pd.Series([1, 2, None] if dtype == "Int64" else [True, False, None], dtype=dtype)})
    block = frame._mgr.blocks[0].values
    external = next(x for x in (getattr(block, "_data", None), getattr(block, "_ndarray", None)) if isinstance(x, np.ndarray)).view()
    store = GovernedBufferStore(); store.put("x", frame, bytes_=int(frame.memory_usage(deep=True).sum()))
    target = store.capture_lease_target("x"); assert target is not None and target.ownership_kind == "weakref_physical"
    lease = Lease(); assert store.attach_memory_lease("x", lease, target=target)
    store.release("x"); del target, frame, block; collect(); assert lease.release_calls == 0
    del external; collect(); assert lease.release_calls == 1


def test_polars_native_storage_is_accounted_conservatively_for_process_lifetime():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"symbol": ["a", "b"], "value": [1.0, 2.0]})
    store = GovernedBufferStore(); store.put("x", frame, bytes_=frame.estimated_size())
    target = store.capture_lease_target("x"); assert target is not None
    assert target.ownership_kind == "process_lifetime_native"
    lease = Lease(); assert store.attach_memory_lease("x", lease, target=target)
    store.release("x"); del target, frame, store; collect()
    assert lease.release_calls == 0


def test_arrow_backed_pandas_storage_is_accounted_conservatively():
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"symbol": pd.Series(["a", "b"], dtype="string[pyarrow]")})
    store = GovernedBufferStore(); store.put("x", frame, bytes_=int(frame.memory_usage(deep=True).sum()))
    target = store.capture_lease_target("x"); assert target is not None
    assert target.ownership_kind == "process_lifetime_native"
    lease = Lease(); assert store.attach_memory_lease("x", lease, target=target)
    store.clear(); del target, frame, store; collect()
    assert lease.release_calls == 0


def test_cse_memory_lease_bundle_releases_all_parts_once():
    from factor_engine.runtime.adaptive_batch_scheduler import _CSEMemoryLeaseBundle

    left, right = Lease(), Lease()
    bundle = _CSEMemoryLeaseBundle(left, right)
    bundle.release(); bundle.release()
    assert left.release_calls == 1
    assert right.release_calls == 1


def test_cse_transfer_admits_measured_overflow_before_attachment():
    from types import SimpleNamespace
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    base, overflow = Lease(), Lease()
    transferred = []
    class Reservation:
        task = SimpleNamespace(admissible_peak_bytes=40)
        def transfer_memory_ownership(self, kind, nbytes, *, lease_id):
            transferred.append((nbytes, lease_id)); return base
    target = SimpleNamespace(bytes=100, physical_ownership_supported=True)
    attached = []
    class Store:
        def capture_lease_target(self, key): return target
        def attach_memory_lease(self, key, lease, *, target):
            attached.append((key, lease)); return True
    scheduler = object.__new__(AdaptiveBatchScheduler)
    scheduler._lease_scope = "test"
    overflow_requests = []
    def acquire(kind, nbytes, *, lease_id):
        overflow_requests.append((nbytes, lease_id)); return overflow
    scheduler._acquire_wave = acquire
    ctx = SimpleNamespace(shared_buffers=Store())
    assert scheduler._transfer_cse_memory_ownership(ctx, "cse:sid", Reservation())
    assert transferred == [(40, "test:cse:sid")]
    assert overflow_requests == [(60, "test:cse-overflow:sid")]
    attached[0][1].release()
    assert base.release_calls == 1 and overflow.release_calls == 1
