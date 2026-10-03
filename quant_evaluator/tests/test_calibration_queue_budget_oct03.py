"""Static regressions for cold-calibration queue budget semantics."""
from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime import backend_calibration as calibration
from quant_evaluator.runtime import calibration_measurement_coordinator as coordinator


def _inputs():
    times = AxisRef("time", "int64", 2, np.array([1, 2], dtype=np.int64))
    assets = AxisRef("asset", "str", 2, np.array(["a", "b"]))
    batch = FactorBatch(("f",), times, assets, np.arange(4.0).reshape(2, 2, 1))
    label = LabelBundle(
        "queue-budget-test", np.arange(4.0).reshape(2, 2), 1,
        decision_time=(1, 2), label_start_time=(2, 3), label_end_time=(3, 4),
        asset_axis=assets,
    )
    return batch, label


def _install_fake_identity(monkeypatch, state):
    monkeypatch.setattr(calibration, "_PROCESS_SOURCE_DRIFTED", False)

    def identity(_batch, _label, *, metrics, calibration_policy, gpu_policy, started=None):
        return {
            "request_digest": state["request_digest"],
            "source_digest": state["source"],
            "source_metadata": {"mode": "strict_full_content"},
            "runtime": state["runtime"], "key": state["key"],
            "input_fingerprint_seconds": 0.0, "setup_seconds": 0.001,
            "device_admission": None, "device_info": {"device_id": 0},
            "process_source_drifted": False,
        }

    monkeypatch.setattr(calibration, "calibration_identity", identity)
    monkeypatch.setattr(calibration, "_runtime_fingerprint",
                        lambda _device: state["runtime"])


def _install_fake_evaluator(monkeypatch):
    import quant_evaluator.runtime.evaluator as evaluator_module

    calls = []

    def evaluate(batch, _label, *, metrics, backend, gpu_policy):
        calls.append(backend)
        return SimpleNamespace(factor_ids=tuple(batch.factor_ids))

    monkeypatch.setattr(evaluator_module, "evaluate", evaluate)
    return calls


def _state():
    return {
        "request_digest": "a" * 64, "source": "source-a",
        "runtime": {"threads": 2}, "key": "queue-key",
    }


def _invoke(batch, label, cache, *, budget_seconds):
    return calibration.evaluate_calibrated_batch(
        batch, label, metrics=("coverage",),
        calibration_policy=calibration.CalibrationPolicy(
            max_wall_time_seconds=budget_seconds, repetitions=1, warmups=0,
        ),
        cache=cache, gpu_policy=GPUExecutionPolicy(),
    )


def test_queue_timeout_does_not_call_evaluator_and_lock_remains_usable(monkeypatch):
    batch, label = _inputs()
    _install_fake_identity(monkeypatch, _state())
    calls = _install_fake_evaluator(monkeypatch)
    cache = calibration.BoundedCalibrationCache()
    original_reserve = coordinator.serial_calibration_measurement
    waiter_requested = threading.Event()
    monkeypatch.setattr(
        coordinator, "serial_calibration_measurement",
        lambda timeout_seconds=None: (
            waiter_requested.set(), original_reserve(timeout_seconds)
        )[1],
    )
    errors = []

    def worker():
        try:
            _invoke(batch, label, cache, budget_seconds=0.1)
        except BaseException as exc:
            errors.append(exc)

    with original_reserve():
        waiter = threading.Thread(target=worker, name="queue-budget-timeout")
        waiter.start()
        assert waiter_requested.wait(timeout=5)
        waiter.join(timeout=5)
        assert not waiter.is_alive()

    assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    assert "timed out waiting" in str(errors[0])
    assert calls == []
    # The timed-out waiter did not retain or poison the process-wide lock.
    with original_reserve():
        pass


class _ReleaseAfterWaitLock:
    """A lock proxy that lets tests release a queued caller deterministically."""

    def __init__(self):
        self._lock = threading.Lock()
        self.waiter_entered = threading.Event()

    def acquire(self, *args, **kwargs):
        if threading.current_thread().name.startswith("queue-budget-"):
            self.waiter_entered.set()
            # Deliberately model a late grant even after the supplied timeout;
            # this isolates the post-acquire budget/cache recheck behavior.
            return self._lock.acquire()
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()


def _start_queued_call(monkeypatch, lock, target, errors):
    monkeypatch.setattr(coordinator._COORDINATOR, "_lock", lock)
    owner = coordinator.serial_calibration_measurement()
    owner.__enter__()
    result = {}

    def worker():
        try:
            result["value"] = target()
        except BaseException as exc:
            errors.append(exc)

    waiter = threading.Thread(target=worker, name="queue-budget-waiter")
    waiter.start()
    assert lock.waiter_entered.wait(timeout=5)
    return owner, waiter, result


def _install_fake_clock(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(
        calibration, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    return clock


def test_queued_miss_rechecks_budget_before_first_cold_call(monkeypatch):
    batch, label = _inputs()
    _install_fake_identity(monkeypatch, _state())
    calls = _install_fake_evaluator(monkeypatch)
    cache = calibration.BoundedCalibrationCache()
    lock = _ReleaseAfterWaitLock()
    errors = []
    clock = _install_fake_clock(monkeypatch)
    owner, waiter, _result = _start_queued_call(
        monkeypatch, lock,
        lambda: _invoke(batch, label, cache, budget_seconds=0.01), errors,
    )
    try:
        clock = _install_fake_clock(monkeypatch)
        clock["now"] = 101.0
    finally:
        owner.__exit__(None, None, None)
    waiter.join(timeout=5)

    assert not waiter.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    assert "budget exhausted between backend calls" in str(errors[0])
    assert calls == []


def test_queued_cache_hit_can_return_after_soft_budget_elapsed(monkeypatch):
    batch, label = _inputs()
    state = _state()
    _install_fake_identity(monkeypatch, state)
    calls = _install_fake_evaluator(monkeypatch)
    cache = calibration.BoundedCalibrationCache()
    lock = _ReleaseAfterWaitLock()
    errors = []
    clock = _install_fake_clock(monkeypatch)
    owner, waiter, result = _start_queued_call(
        monkeypatch, lock,
        lambda: _invoke(batch, label, cache, budget_seconds=0.01), errors,
    )
    clock = _install_fake_clock(monkeypatch)
    try:
        clock["now"] = 101.0
        cache.put(state["key"], {
            "winner": "cuda_strict", "parity": "pass",
            "selection_basis": "steady_state_median",
            "cpu_median_seconds": 2.0, "cuda_median_seconds": 1.0,
            "repetitions": 1, "warmups": 0,
        })
    finally:
        owner.__exit__(None, None, None)
    waiter.join(timeout=5)

    assert not waiter.is_alive()
    assert not errors
    assert result["value"].metadata["status"] == "cache_hit"
    assert calls == ["cuda_strict"]
