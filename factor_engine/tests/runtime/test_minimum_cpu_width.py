from dataclasses import replace
import threading

import pytest

from factor_engine.runtime.resource_broker import ResourceBroker, ResourceSnapshot
from factor_engine.runtime.resource_autopilot import ResourceDecision
from factor_engine.runtime.resource_monitor import ResourceSignals
from factor_engine.runtime.resource_errors import CPUWidthUnavailable
from factor_engine.runtime.task_resource_contract import TaskResourceContract


@pytest.fixture
def broker(monkeypatch):
    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4,
        min_host_reserve_gb=0, min_host_reserve_fraction=0)
    snapshot = ResourceSnapshot(timestamp_ms=0, hard_cpu_slots=4, system_cpu_util=0,
        our_cpu_util=0, external_cpu_util=0, loadavg=0, hard_memory_limit=8 * 1024**3,
        cgroup_memory_current=None, host_mem_available=8 * 1024**3,
        process_rss=0, worker_rss=0, process_family_rss=0, process_family_pss=0,
        spill_free_bytes=8 * 1024**3, spill_total_bytes=16 * 1024**3, disk_busy=0)
    monkeypatch.setattr(broker, "_refresh", lambda **kwargs: snapshot)
    return broker


def decision(stage="NORMAL"):
    return ResourceDecision(target_concurrency=4, target_cpu_tokens=1,
        read_wave_bytes=1024**2, factor_block_bytes=1024**2, result_queue_bytes=1024**2,
        io_concurrency=4, remote_concurrency=4, cache_budget_bytes=1024**2,
        spill_budget_bytes=1024**2, pressure_state=stage)


def task(width=2):
    return TaskResourceContract(cpu_tokens=width, backend_threads=width,
        peak_memory_bytes=1024, output_bytes=32, backend="polars")


@pytest.mark.parametrize("stage", ["NORMAL", "PRESSURE_1"])
def test_controller_floor_preserves_real_width_and_allows_two_runs(broker, stage):
    for attempt in range(2):
        broker._cpu.set_soft_budget(1)
        contract = task()
        assert broker.try_reserve(contract, task_id="root") is None
        resolved = broker.request_minimum_cpu_width(contract, decision=decision(stage))
        assert resolved.target_concurrency == 1
        assert resolved.target_cpu_tokens == 2
        assert contract.cpu_tokens == contract.backend_threads == 2
        lease = broker.try_reserve(contract, task_id="root")
        assert lease is not None
        assert broker._cpu.in_use == 2
        lease.release()
        assert broker._cpu.in_use == 0 and not broker._running


@pytest.mark.parametrize("stage", ["PRESSURE_2", "PRESSURE_3", "CRITICAL"])
def test_elevated_pressure_cannot_raise_soft_target(broker, stage):
    broker._cpu.set_soft_budget(1)
    with pytest.raises(CPUWidthUnavailable):
        broker.request_minimum_cpu_width(task(), decision=decision(stage))
    assert broker._cpu.soft_budget == 1


def test_hard_limit_memory_and_running_work_not_bypassed(broker):
    broker._cpu.set_soft_budget(1)
    for contract in (task(5), replace(task(), peak_memory_bytes=16 * 1024**3)):
        with pytest.raises(CPUWidthUnavailable):
            broker.request_minimum_cpu_width(contract, decision=decision())
    lease = broker.try_reserve(task(1), task_id="other")
    assert lease is not None
    try:
        with pytest.raises(CPUWidthUnavailable):
            broker.request_minimum_cpu_width(task(), decision=decision())
    finally:
        lease.release()


def test_static_swap_occupancy_does_not_repeatedly_throttle(broker):
    controller = broker._resource_controller()
    for _ in range(20):
        result = controller.tick(ResourceSignals(swap_current=96, swap_max=100), broker.resource_envelope())
        assert result.pressure_state == "NORMAL"
        assert result.target_cpu_tokens >= 4
    # Genuine memory PSI still escalates pressure.
    result = controller.tick(ResourceSignals(memory_psi_some=1), broker.resource_envelope())
    assert result.pressure_state != "NORMAL"


def test_stale_normal_decision_cannot_overwrite_new_pressure(broker):
    controller = broker._resource_controller()
    stale = decision("NORMAL")
    current = controller.tick(ResourceSignals(memory_psi_some=1, cpu_psi_some=1),
                              broker.resource_envelope())
    assert current.pressure_state not in {"NORMAL", "PRESSURE_1"}
    before = broker._cpu.soft_budget
    with pytest.raises(CPUWidthUnavailable):
        broker.request_minimum_cpu_width(task(), decision=stale)
    assert controller.last_decision() is current
    assert broker._cpu.soft_budget == before


def test_background_tick_serializes_with_foreground_admission(broker):
    controller = broker._resource_controller()
    entered, finished = threading.Event(), threading.Event()
    envelope = broker.resource_envelope()
    def tick():
        entered.set()
        controller.tick(ResourceSignals(memory_psi_some=1, cpu_psi_some=1), envelope)
        finished.set()
    with broker._lock:
        worker = threading.Thread(target=tick)
        worker.start()
        assert entered.wait(1)
        assert not finished.wait(0.03)  # tick must honor the existing broker lock
    worker.join(1)
    assert finished.is_set()
    latest = controller.last_decision()
    with pytest.raises(CPUWidthUnavailable):
        broker.request_minimum_cpu_width(task(), decision=decision("NORMAL"))
    assert controller.last_decision() is latest
