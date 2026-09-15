"""Resource accounting rejects invalid token quantities and bounds diagnostics."""
import copy

import pytest

import factor_engine.runtime.resource_broker as module
from factor_engine.runtime.task_resource_contract import TaskResourceContract
from factor_engine.tests.r38.test_p011_resource_broker_authority import _broker


@pytest.mark.parametrize("field", ["cpu_tokens", "io_tokens"])
@pytest.mark.parametrize("value", [-1, -2, 0.5, True, float("nan"), float("inf")])
def test_invalid_token_quantities_rejected_before_admission(field, value):
    with pytest.raises((TypeError, ValueError)):
        TaskResourceContract(**{field: value})


def test_numpy_integer_token_quantities_are_supported_and_normalized():
    import numpy as np
    task = TaskResourceContract(cpu_tokens=np.int64(2), io_tokens=np.int64(0))
    assert type(task.cpu_tokens) is int and task.cpu_tokens == 2
    assert type(task.io_tokens) is int and task.io_tokens == 0


def test_zero_token_contract_has_no_accounting_effect():
    broker = _broker()
    task = TaskResourceContract(cpu_tokens=0, io_tokens=0)
    lease = broker.try_reserve(task, task_id="zero")
    assert lease is not None
    lease.release()
    lease.release()
    assert broker._cpu.in_use == broker._io.in_use == 0


def test_broker_repeated_admission_and_pressure_diagnostics_are_bounded(monkeypatch):
    broker = _broker()
    monkeypatch.setattr(broker, "can_admit", lambda _: False)
    task = TaskResourceContract()
    for i in range(1000):
        assert broker.try_reserve(task, task_id=str(i)) is None
        broker.adapt_uncertainty(underpredict_streak=3, overpredict_streak=0)
    assert len(broker._admission_events) == 50
    assert len(broker._pressure_log) == 20
    summary = broker.summary()
    assert type(summary["admission_events"]) is list
    assert type(summary["pressure_log"]) is list
    summary["admission_events"].clear()
    summary["pressure_log"].clear()
    assert len(broker._admission_events) == 50
    assert len(broker._pressure_log) == 20


def test_degradation_diagnostics_keep_total_count_but_bounded_history(monkeypatch):
    state = copy.deepcopy(module._degraded)
    state[0] = 0
    state[1].clear()
    monkeypatch.setattr(module, "_degraded", state)
    monkeypatch.delenv("FE_STRICT_RESOURCE_BROKER", raising=False)
    for _ in range(1000):
        assert module.require_broker(None, run_mode="research", return_fallback=False) is None
    snapshot = module.broker_degraded_observable()
    assert snapshot["count"] == 1000
    assert snapshot["verdicts"] == ["degraded"] * 20
    assert len(module._degraded[1]) == 20
    snapshot["verdicts"].clear()
    assert len(module.broker_degraded_observable()["verdicts"]) == 20

def _child_read_diagnostics(connection):
    try:
        connection.send(module.broker_degraded_observable())
    finally:
        connection.close()


def test_diagnostic_snapshot_does_not_deadlock_after_fork():
    import multiprocessing as mp

    if "fork" not in mp.get_all_start_methods():
        pytest.skip("requires fork")
    context = mp.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_child_read_diagnostics, args=(sender,))
    try:
        with module._degraded_lock:
            process.start()
            sender.close()
            process.join(timeout=5)
        assert not process.is_alive(), "fork child inherited a locked diagnostic observer"
        assert process.exitcode == 0
        assert receiver.poll(1)
        assert type(receiver.recv()["verdicts"]) is list
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        receiver.close()
        sender.close()
        if process.exitcode is not None:
            process.close()
