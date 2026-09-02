from __future__ import annotations

import threading
import time

import pytest

from data_access.core.exceptions import DeadlineExceeded, ResourceAdmissionError
from data_access.r30.execution_lease import ExecutionLease
from data_access.r30.resolution_lease import ResolutionLease
from data_access.runtime.resource_governor import GlobalResourceGovernor, ResourceReservation


class _PartialGovernor:
    def __init__(self) -> None:
        self.calls = 0
        self.held = 0

    def acquire_remote_discovery_slot(self) -> bool:
        self.calls += 1
        if self.calls == 3:
            return False
        self.held += 1
        return True

    def release_remote_discovery_slot(self) -> None:
        self.held -= 1


def _lease() -> ExecutionLease:
    return ExecutionLease("root", 100, 0, 0, 0, 0, 0).acquire()


def test_resolution_partial_acquire_rolls_back_only_acquired_now():
    governor = _PartialGovernor()
    lease = ResolutionLease("r", max_resolution_slots=8, governor=governor)
    assert lease.acquire_resolution() is True
    assert lease.acquire_resolution(3) is False
    assert lease.to_dict()["resolution_inflight"] == 1
    assert governor.held == 1


def test_child_release_returns_parent_budget_exactly_once():
    parent = _lease()
    child = parent.request_child({"memory": 60})
    assert parent.to_dict()["remaining"]["memory"] == 40
    child.release()
    child.release()
    assert parent.to_dict()["remaining"]["memory"] == 100


def test_concurrent_child_allocate_release_never_exceeds_parent():
    parent = _lease()
    errors = []

    def work() -> None:
        for _ in range(100):
            try:
                child = parent.request_child({"memory": 10})
            except RuntimeError:
                continue
            if parent.to_dict()["remaining"]["memory"] < 0:
                errors.append("negative")
            child.release()

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert parent.to_dict()["remaining"]["memory"] == 100


def test_expired_resolution_and_slot_wait_fail():
    with pytest.raises(DeadlineExceeded):
        ResolutionLease("expired", absolute_deadline=time.monotonic() - 1).acquire_resolution()
    governor = GlobalResourceGovernor(max_duckdb_concurrency=1)
    # R32-P0-010 权威 API：acquire_duckdb_slot / acquire_remote_slot（deadline 已过
    # → raise DeadlineExceeded）。*_slot_lease 是 R48 前的旧名，不存在了。
    with pytest.raises(DeadlineExceeded):
        governor.acquire_duckdb_slot(deadline=time.monotonic() - 1)
    with pytest.raises(DeadlineExceeded):
        governor.acquire_remote_slot(deadline=time.monotonic() - 1)


def test_host_backed_duplicate_and_broken_bridge_fail_closed():
    # 同 P0-008：fail-closed 语义只在 strict/production 下生效（R39 #51 权威）。
    from data_access.runtime.mode_identity import (
        reset_runtime_mode_identity,
        set_runtime_mode_identity,
    )

    token = set_runtime_mode_identity("production", source="test")
    try:
        governor = GlobalResourceGovernor(max_total_reserved_memory=1000)
        governor.set_host_lease_request(lambda *_: object())
        governor.admit(ResourceReservation("q", "p", estimated_memory=1))
        with pytest.raises(ResourceAdmissionError):
            governor.admit(ResourceReservation("q", "p", estimated_memory=1))

        broken = GlobalResourceGovernor(max_total_reserved_memory=1000)
        broken.set_host_lease_request(lambda *_: (_ for _ in ()).throw(RuntimeError("host down")))
        with pytest.raises(ResourceAdmissionError):
            broken.admit(ResourceReservation("broken", "p", estimated_memory=1))
        assert broken.active_count() == 0
    finally:
        reset_runtime_mode_identity(token)
