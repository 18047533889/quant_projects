from __future__ import annotations

from dataclasses import dataclass
import time
import tracemalloc

import pytest

from factor_engine.runtime.auto_dag_admission import admit_run_dag
from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind


@dataclass(frozen=True)
class _Record:
    definition_bytes: object
    name: object = "factor"


class _Manifest:
    def __init__(self, count, size=0, *, yielded_count=None):
        self.count = count
        self.size = size
        self.yielded_count = count if yielded_count is None else yielded_count
        self.scanned = 0
        self.calls = []

    def __len__(self):
        return self.count

    def records(self, *, start, limit):
        self.calls.append((start, limit))
        for _ in range(min(limit, self.yielded_count)):
            self.scanned += 1
            yield _Record(self.size, f"f{self.scanned}")


class _Lease:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _Broker:
    def __init__(self, budget, *, admit=True):
        self.budget = budget
        self.admit = admit
        self.execution_budget_calls = 0
        self.acquisitions = []

    def execution_budget(self):
        self.execution_budget_calls += 1
        return self.budget

    def acquire_memory(self, kind, nbytes, *, lease_id):
        self.acquisitions.append((kind, nbytes, lease_id))
        return _Lease() if self.admit else None


def _future():
    return time.monotonic() + 60


def test_small_batch_skips_budget_scan_and_lease_even_at_zero_budget():
    manifest = _Manifest(512, size=10)
    broker = _Broker(0)
    lease, summary = admit_run_dag(
        manifest, broker, run_id="r", fallback_items=512, deadline=_future()
    )
    assert lease is None
    assert summary["reason"] == "SMALL_BATCH_ALREADY_FITS"
    assert summary["execution_scope"] == "bounded_waves"
    assert summary["cross_wave_value_reuse"] is False
    assert manifest.scanned == 0
    assert broker.execution_budget_calls == 0
    assert broker.acquisitions == []


def test_zero_budget_rejects_without_reading_manifest_records():
    manifest = _Manifest(513, size=0)
    broker = _Broker(0)
    lease, summary = admit_run_dag(
        manifest, broker, run_id="r", fallback_items=512, deadline=_future()
    )
    assert lease is None
    assert summary["reason"] == "METADATA_REQUIRES_BOUNDED_WAVES"
    assert summary["scanned_roots"] == manifest.scanned == 0
    assert summary["estimated_metadata_peak_bytes"] == 1024 * 1024
    assert broker.acquisitions == []


def test_budget_shortfall_stops_manifest_before_later_records():
    manifest = _Manifest(100_000, size=100)
    # ceiling=1,050,000; base + one record charge exceeds it.
    broker = _Broker(4_200_000)
    lease, summary = admit_run_dag(
        manifest, broker, run_id="r", fallback_items=512, deadline=_future()
    )
    assert lease is None
    assert summary["reason"] == "METADATA_REQUIRES_BOUNDED_WAVES"
    assert manifest.scanned == 1
    assert summary["scanned_roots"] == 1


def test_one_hundred_thousand_light_definitions_are_streamed_and_admitted():
    manifest = _Manifest(100_000, size=0)
    expected = 1024 * 1024 + 100_000 * 8192
    broker = _Broker(expected * 4 + 4)
    tracemalloc.start()
    try:
        lease, summary = admit_run_dag(
            manifest, broker, run_id="big", fallback_items=512, deadline=_future()
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert lease is not None
    assert manifest.calls == [(0, 100_000)]
    assert manifest.scanned == 100_000
    assert peak < 8 * 1024 * 1024
    assert summary["execution_scope"] == "whole_run_dag"
    assert summary["requested_roots"] == summary["scanned_roots"] == 100_000
    assert summary["estimated_metadata_peak_bytes"] == expected
    assert summary["definition_bytes"] == 0
    assert broker.acquisitions == [
        (MemoryLeaseKind.MANIFEST_BUFFER, expected, "run-dag-metadata:big")
    ]


def test_rejected_metadata_lease_preserves_bounded_wave_claim():
    manifest = _Manifest(513, size=0)
    broker = _Broker(10**12, admit=False)
    lease, summary = admit_run_dag(
        manifest, broker, run_id="r", fallback_items=512, deadline=_future()
    )
    assert lease is None
    assert summary["reason"] == "METADATA_LEASE_NOT_ADMITTED"
    assert summary["execution_scope"] == "bounded_waves"
    assert summary["cross_wave_value_reuse"] is False
    assert len(broker.acquisitions) == 1


def test_zero_descriptor_budget_rejects_without_manifest_scan():
    manifest = _Manifest(513)
    lease, summary = admit_run_dag(
        manifest, _Broker(10**12), run_id="r", fallback_items=512,
        deadline=_future(), descriptor_budget_bytes=0,
    )
    assert lease is None
    assert summary["reason"] == "ARTIFACT_DESCRIPTORS_REQUIRE_BOUNDED_WAVES"
    assert summary["descriptor_budget_bytes"] == 0
    assert summary["estimated_descriptor_bytes"] == 0
    assert summary["scanned_roots"] == manifest.scanned == 0


def test_descriptor_budget_is_checked_during_same_bounded_scan():
    manifest = _Manifest(100_000)
    broker = _Broker(10**12)
    one = 8192 + 4 * len("f1".encode())
    lease, summary = admit_run_dag(
        manifest, broker, run_id="r", fallback_items=512,
        deadline=_future(), descriptor_budget_bytes=2 * one,
    )
    assert lease is None
    assert summary["reason"] == "ARTIFACT_DESCRIPTORS_REQUIRE_BOUNDED_WAVES"
    assert summary["scanned_roots"] == manifest.scanned == 2
    assert summary["estimated_descriptor_bytes"] > one
    assert broker.acquisitions == []


def test_descriptor_estimate_and_budget_are_recorded_on_admission():
    manifest = _Manifest(513)
    expected = sum(8192 + 4 * len(f"f{i}".encode()) for i in range(1, 514))
    budget = 2 * expected
    lease, summary = admit_run_dag(
        manifest, _Broker(10**12), run_id="r", fallback_items=512,
        deadline=_future(), descriptor_budget_bytes=budget,
    )
    assert lease is not None
    assert summary["estimated_descriptor_bytes"] == expected
    assert summary["descriptor_budget_bytes"] == budget


@pytest.mark.parametrize("invalid", [1.5, "2", object()])
def test_invalid_descriptor_budget_is_rejected_before_scan(invalid):
    manifest = _Manifest(513)
    with pytest.raises(ValueError, match="descriptor_budget_bytes"):
        admit_run_dag(
            manifest, _Broker(10**12), run_id="r", fallback_items=512,
            deadline=_future(), descriptor_budget_bytes=invalid,
        )
    assert manifest.scanned == 0


def test_expired_deadline_stops_before_reading_manifest_records():
    manifest = _Manifest(513)
    lease, summary = admit_run_dag(
        manifest, _Broker(10**12), run_id="r", fallback_items=512,
        deadline=time.monotonic() - 1,
    )
    assert lease is None
    assert summary["reason"] == "CATALOG_ADMISSION_DEADLINE"
    assert manifest.scanned == 0


def test_cancellation_is_checked_before_first_and_each_512_records():
    manifest = _Manifest(1025)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        admit_run_dag(
            manifest, _Broker(10**12), run_id="r", fallback_items=512,
            deadline=_future(), check_cancellation=cancel,
        )
    assert calls == 3
    assert manifest.scanned == 1024


@pytest.mark.parametrize("invalid", [-1, 1.5, "2", None])
def test_invalid_definition_size_is_rejected(invalid):
    with pytest.raises(ValueError, match="definition byte count"):
        admit_run_dag(
            _Manifest(513, size=invalid), _Broker(10**12), run_id="r",
            fallback_items=512, deadline=_future(),
        )


def test_manifest_count_drift_is_rejected_without_lease():
    manifest = _Manifest(513, yielded_count=512)
    broker = _Broker(10**12)
    with pytest.raises(ValueError, match="root count changed"):
        admit_run_dag(
            manifest, broker, run_id="r", fallback_items=512, deadline=_future()
        )
    assert broker.acquisitions == []


@pytest.mark.parametrize("budget", [None, "bad", object()])
def test_invalid_or_missing_live_budget_falls_back_without_scanning(budget):
    broker = object() if budget is None else _Broker(budget)
    manifest = _Manifest(513)
    lease, summary = admit_run_dag(
        manifest, broker, run_id="r", fallback_items=512, deadline=_future()
    )
    assert lease is None
    assert summary["reason"] == "LIVE_BUDGET_UNAVAILABLE"
    assert manifest.scanned == 0
