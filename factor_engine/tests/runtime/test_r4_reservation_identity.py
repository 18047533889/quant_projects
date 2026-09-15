"""Legacy bool reservations must not acknowledge a different contract by ID."""
from dataclasses import replace
from factor_engine.tests.r38.test_p011_resource_broker_authority import _broker
from factor_engine.runtime.task_resource_contract import TaskResourceContract


def test_different_contract_with_same_id_is_not_an_idempotent_reservation():
    broker = _broker()
    small = TaskResourceContract(peak_memory_bytes=10, cpu_tokens=1)
    huge = replace(small, peak_memory_bytes=10**12, cpu_tokens=1000)
    try:
        assert broker.reserve(small, task_id="same") is True
        assert broker.reserve(huge, task_id="same") is False
        assert broker._running["same"] == small
        assert broker._cpu.in_use == 1
        assert broker.reserve(replace(small), task_id="same") is True
    finally:
        broker.release(small, task_id="same")
    assert broker._cpu.in_use == 0


def test_mismatched_legacy_release_does_not_release_another_live_contract():
    broker = _broker()
    small = TaskResourceContract(peak_memory_bytes=10, cpu_tokens=1)
    other = replace(small, peak_memory_bytes=20)
    assert broker.reserve(small, task_id="same") is True
    try:
        broker.release(other, task_id="same")
        assert broker._running["same"] == small
        assert broker._cpu.in_use == 1
    finally:
        broker.release(small, task_id="same")
    assert broker._cpu.in_use == 0
