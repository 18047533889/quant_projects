from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import replace

import pytest

from factor_engine.runtime.auto_memory_budget import AutoMemoryBudget, MemoryLeaseKind
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.resource_broker_ipc import BrokerRPCError, ParentBrokerIPC


def _broker() -> ResourceBroker:
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3, cpu_slots=4,
        min_host_reserve_gb=0, min_host_reserve_fraction=0,
    )
    broker._refresh_auto_budget = lambda: AutoMemoryBudget(
        hard_memory_limit=8 * 1024**3, emergency_reserve=0,
        safe_live_budget=1024**3, execution_budget=1024**3,
        safety_factor=.8, measurement_state="TEST_INJECTED",
    )
    return broker


def _stay_alive() -> None:
    time.sleep(10)


def test_epoch_end_refuses_outstanding_lease_then_returns_verified_zero_receipt():
    ipc = ParentBrokerIPC(_broker())
    proxy = ipc.create_proxy()
    try:
        epoch_id = proxy.begin_epoch()
        lease = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, 1, lease_id="epoch-lease")
        assert lease is not None
        with pytest.raises(BrokerRPCError, match="outstanding leases"):
            proxy.end_epoch()
        lease.release()
        receipt = proxy.end_epoch()
        assert receipt.epoch_id == epoch_id
        assert receipt.active_tokens == 0
        assert ipc.verify_epoch_receipt(receipt)
        assert not ipc.verify_epoch_receipt(replace(receipt, sequence=receipt.sequence + 1))
        assert not ipc.verify_epoch_receipt(replace(receipt, active_tokens=False))
        assert not ipc.verify_epoch_receipt(replace(receipt, active_tokens=0.0))
        assert not ipc.verify_epoch_receipt(replace(receipt, signature=None))
        assert not ipc.verify_epoch_receipt(replace(receipt, signature="é" * 64))
        assert not ipc.verify_epoch_receipt(replace(receipt, client_id="\ud800" * 32))
        assert not ipc.verify_epoch_receipt(replace(receipt, epoch_id="G" * 32))
        assert not ipc.verify_epoch_receipt(replace(receipt, sequence=0))
    finally:
        ipc.close()


def test_replayed_old_token_cannot_affect_a_later_epoch():
    ipc = ParentBrokerIPC(_broker())
    proxy = ipc.create_proxy()
    try:
        proxy.begin_epoch()
        old_epoch = proxy._epoch_id
        old_capability = proxy._epoch_capability
        lease = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, 1, lease_id="old")
        assert lease is not None
        old_token = lease.token
        lease.release()
        first = proxy.end_epoch()

        second_id = proxy.begin_epoch()
        assert proxy._rpc("release", token=old_token) is False
        current = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, 1, lease_id="current")
        assert current is not None
        with pytest.raises(RuntimeError, match="capability mismatch"):
            ipc._dispatch(proxy.client_id, "release", {
                "token": current.token,
                "_epoch_id": old_epoch,
                "_epoch_capability": old_capability,
            })
        assert ipc.active_tokens() == 1
        current.release()
        second = proxy.end_epoch()
        assert second.epoch_id == second_id != first.epoch_id
        assert second.sequence > first.sequence
        assert ipc.verify_epoch_receipt(second)
    finally:
        ipc.close()


def test_same_parent_clients_have_isolated_epochs():
    ipc = ParentBrokerIPC(_broker())
    first = ipc.create_proxy()
    second = ipc.create_proxy()
    try:
        first.begin_epoch()
        second.begin_epoch()
        lease = first.acquire_memory(MemoryLeaseKind.COMPUTE, 1, lease_id="first-only")
        assert lease is not None
        second_receipt = second.end_epoch()
        assert ipc.verify_epoch_receipt(second_receipt)
        with pytest.raises(BrokerRPCError, match="outstanding leases"):
            first.end_epoch()
        lease.release()
        assert ipc.verify_epoch_receipt(first.end_epoch())
    finally:
        ipc.close()


def test_process_exit_reclaim_clears_epoch_and_lease_without_receipt():
    ipc = ParentBrokerIPC(_broker())
    proxy = ipc.create_proxy()
    process = mp.get_context("spawn").Process(target=_stay_alive)
    process.start()
    try:
        ipc.bind_client_process(proxy.client_id, process.pid)
        proxy.begin_epoch()
        lease = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, 1, lease_id="orphan")
        assert lease is not None and ipc.active_tokens() == 1
        process.terminate()
        process.join(2.0)
        assert ipc.reclaim_client(proxy.client_id) == 1
        assert ipc.active_tokens() == 0
        assert proxy.client_id not in ipc._client_epochs
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        ipc.close()


def test_legacy_client_without_epoch_remains_compatible():
    ipc = ParentBrokerIPC(_broker())
    proxy = ipc.create_proxy()
    try:
        lease = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, 1, lease_id="legacy")
        assert lease is not None
        lease.release()
        assert ipc.active_tokens() == 0
    finally:
        ipc.close()


def test_epoch_cannot_begin_until_legacy_leases_are_zero():
    ipc = ParentBrokerIPC(_broker())
    proxy = ipc.create_proxy()
    try:
        lease = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, 1, lease_id="legacy-first")
        assert lease is not None
        with pytest.raises(BrokerRPCError, match="outstanding legacy leases"):
            proxy.begin_epoch()
        lease.release()
        proxy.begin_epoch()
        assert ipc.verify_epoch_receipt(proxy.end_epoch())
    finally:
        ipc.close()
