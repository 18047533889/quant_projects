from __future__ import annotations

import pytest


class _Lease:
    def __init__(self): self.released = False
    def release(self): self.released = True


class _Broker:
    def __init__(self): self.requests = []
    def acquire_memory(self, kind, amount, *, lease_id):
        self.requests.append((kind, amount, lease_id))
        return _Lease()


def _coordinator(broker):
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator
    coordinator = HostResourceCoordinator.__new__(HostResourceCoordinator)
    coordinator._broker = broker
    coordinator.set_active_job_lease(None)
    return coordinator


def test_da_child_lease_reserves_memory_not_cumulative_scan_bytes():
    broker = _Broker()
    lease = _coordinator(broker).request_da_child_lease(100, 900)
    assert broker.requests[0][1] == 100
    lease.release()
    assert lease.released is True


def test_scan_only_uses_minimum_nonzero_memory_charge():
    broker = _Broker()
    _coordinator(broker).request_da_child_lease(0, 10_000)
    assert broker.requests[0][1] == 1


def test_negative_and_unknown_cost_contract(monkeypatch):
    from data_access.core.exceptions import RemoteMetadataUnavailable
    from data_access.runtime.resource_governor import ResourceReservation
    negative = ResourceReservation("negative", "p", -20, -10)
    assert negative.estimated_memory == negative.estimated_scan_bytes == 0
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    with pytest.raises(RemoteMetadataUnavailable):
        ResourceReservation("unknown", "p", None, 1)


def test_host_success_still_enforces_scan_cap_before_callback():
    from data_access.core.exceptions import ResourceAdmissionError
    from data_access.runtime.resource_governor import GlobalResourceGovernor, ResourceReservation
    calls = []
    governor = GlobalResourceGovernor(max_total_scan_bytes_inflight=100)
    governor.set_host_lease_request(lambda memory, scan: calls.append((memory, scan)))
    with pytest.raises(ResourceAdmissionError, match="scan bytes"):
        governor.admit(ResourceReservation("wide", "p", 101, 1))
    assert calls == []


def test_concurrent_scan_recheck_rolls_back_host_lease():
    from data_access.core.exceptions import ResourceAdmissionError
    from data_access.runtime.resource_governor import GlobalResourceGovernor, ResourceReservation
    governor = GlobalResourceGovernor(max_total_scan_bytes_inflight=100)
    lease = _Lease()
    def callback(memory, scan):
        with governor._lock:
            governor._active["incumbent"] = ResourceReservation("incumbent", "p", 60, 1)
        return lease
    governor.set_host_lease_request(callback)
    with pytest.raises(ResourceAdmissionError, match="scan bytes"):
        governor.admit(ResourceReservation("candidate", "p", 50, 1))
    assert lease.released is True
    assert "candidate" not in governor._active


def test_release_restores_scan_capacity_and_host_remains_memory_authority():
    from data_access.runtime.resource_governor import GlobalResourceGovernor, ResourceReservation
    leases = []
    governor = GlobalResourceGovernor(
        max_total_reserved_memory=1, max_total_scan_bytes_inflight=100,
    )
    governor.set_host_lease_request(
        lambda memory, scan: leases.append(_Lease()) or leases[-1]
    )
    governor.admit(ResourceReservation("first", "p", 100, 10_000))
    assert governor.inflight_scan_bytes() == 100
    governor.release("first")
    assert governor.inflight_scan_bytes() == 0
    assert leases[0].released is True
    governor.admit(ResourceReservation("second", "p", 100, 10_000))
