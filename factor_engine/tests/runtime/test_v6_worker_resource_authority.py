from __future__ import annotations

from types import SimpleNamespace


def test_worker_autopilot_is_read_only_parent_proxy_view(monkeypatch):
    from factor_engine.runtime.resource_autopilot_service import (
        WorkerResourceDecisionView,
        start_resource_autopilot,
        stop_resource_autopilot,
    )
    from factor_engine.runtime.resource_broker_ipc import ResourceBrokerProxy

    proxy = object.__new__(ResourceBrokerProxy)
    proxy.summary = lambda: {"authority": "parent"}
    proxy.resource_decision = lambda **kwargs: SimpleNamespace(source="parent")
    stop_resource_autopilot()
    view = start_resource_autopilot(proxy)
    assert isinstance(view, WorkerResourceDecisionView)
    assert view.tick_count() == 0
    assert view.summary()["parent_broker"] == {"authority": "parent"}
    assert not hasattr(view, "_thread")
    stop_resource_autopilot()


def test_host_coordinator_reuses_explicit_proxy_and_rejects_switch():
    import pytest
    from factor_engine.runtime.host_resource_coordinator import (
        get_host_coordinator,
        reset_host_coordinator,
    )
    from factor_engine.runtime.resource_broker_ipc import ResourceBrokerProxy

    first = object.__new__(ResourceBrokerProxy)
    second = object.__new__(ResourceBrokerProxy)
    reset_host_coordinator()
    coordinator = get_host_coordinator(broker=first)
    assert coordinator.broker is first
    assert get_host_coordinator(broker=first) is coordinator
    with pytest.raises(RuntimeError, match="another broker authority"):
        get_host_coordinator(broker=second)
    reset_host_coordinator()


def test_real_prepared_read_charges_and_releases_parent_broker(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_access.registry.loader import DatasetRegistry, StaticDataset
    from data_access.store import DataAccessStore, get_shared_engine
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator
    from factor_engine.runtime.resource_broker import ResourceBroker
    from factor_engine.runtime.resource_broker_ipc import ParentBrokerIPC

    pq.write_table(pa.table({"Close": [1.0, 2.0]}), tmp_path / "part.parquet")
    dataset = StaticDataset(
        name="daily", access_mode="published", layout="plain",
        time_column=None, instrument_column=None, hive_partitioning=False,
        union_by_name=False, root=str(tmp_path), glob="part.parquet",
        schema={"Close": "float64"},
    )
    store = DataAccessStore(
        registry=DatasetRegistry({"daily": dataset}), engine=get_shared_engine(),
    )
    parent = ResourceBroker(
        hard_memory_limit=1024**3, safety_factor=0.8,
        min_host_reserve_gb=0, min_host_reserve_fraction=0,
    )
    ipc = ParentBrokerIPC(parent)
    proxy = ipc.create_proxy()
    coordinator = HostResourceCoordinator(broker=proxy)
    applied = coordinator.apply_da_envelope()
    assert applied["host_lease_bridge"] == "installed"
    before = parent.summary()
    prepared = store.prepare_read(
        "daily", columns=("Close",), params={}, run_mode="production",
        snapshot_policy="fail_if_changed",
    )
    during = parent.summary()
    assert during["active_memory_leases"] == before["active_memory_leases"] + 1
    assert during["memory_lease_bytes"] > before["memory_lease_bytes"]
    result = store.execute_prepared_read(prepared)
    assert result is not None
    store._pipeline.release_reservation(prepared.resource_reservation)
    after = parent.summary()
    assert after["active_memory_leases"] == before["active_memory_leases"]
    assert after["memory_lease_bytes"] == before["memory_lease_bytes"]
    coordinator._da_governor().set_host_lease_request(None)
    ipc.close()
