"""Concurrency regressions for exact cold calibration coordination."""
from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np
import pytest

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
        "singleflight-test", np.arange(4.0).reshape(2, 2), 1,
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
            "runtime": state["runtime"],
            "key": state["key"],
            "input_fingerprint_seconds": 0.0,
            "setup_seconds": 0.001,
            "device_admission": state["admission"],
            "device_info": state["device_info"],
            "process_source_drifted": False,
        }

    monkeypatch.setattr(calibration, "calibration_identity", identity)
    monkeypatch.setattr(calibration, "_runtime_fingerprint",
                        lambda _device: state["runtime"])
    return identity


def _install_fake_public_evaluator(monkeypatch, *, first_call_entered=None,
                                   release_first_call=None):
    import quant_evaluator.runtime.evaluator as evaluator_module

    calls = []
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    def evaluate(batch, _label, *, metrics, backend, gpu_policy):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
            calls.append(backend)
            call_number = len(calls)
        try:
            if call_number == 1 and first_call_entered is not None:
                first_call_entered.set()
                if not release_first_call.wait(timeout=5):
                    raise AssertionError("first fake evaluation was not released")
            return SimpleNamespace(factor_ids=tuple(batch.factor_ids))
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(evaluator_module, "evaluate", evaluate)
    return calls, lambda: maximum_active


def _worker(target, errors):
    try:
        target()
    except BaseException as exc:
        errors.append(exc)


def test_same_key_cold_misses_share_one_measurement_and_second_uses_cache(monkeypatch):
    batch, label = _inputs()
    state = {
        "request_digest": "a" * 64, "source": "source-a", "runtime": {"threads": 2},
        "key": "same-key", "admission": None, "device_info": {"device_id": 0},
    }
    _install_fake_identity(monkeypatch, state)
    first_call_entered = threading.Event()
    release_first_call = threading.Event()
    calls, maximum_active = _install_fake_public_evaluator(
        monkeypatch, first_call_entered=first_call_entered,
        release_first_call=release_first_call,
    )

    original_reserve = coordinator.serial_calibration_measurement
    second_requested_reservation = threading.Event()

    def observed_reserve(timeout_seconds=None):
        if threading.current_thread().name == "calibration-cold-waiter":
            second_requested_reservation.set()
        return original_reserve(timeout_seconds)

    monkeypatch.setattr(coordinator, "serial_calibration_measurement", observed_reserve)
    cache = calibration.BoundedCalibrationCache()
    policy = calibration.CalibrationPolicy(repetitions=2, warmups=0)
    results = {}
    errors = []

    def invoke(slot):
        results[slot] = calibration.evaluate_calibrated_batch(
            batch, label, metrics=("coverage",), calibration_policy=policy,
            cache=cache, gpu_policy=GPUExecutionPolicy(),
        )

    first = threading.Thread(
        target=lambda: _worker(lambda: invoke("first"), errors),
        name="calibration-cold-owner",
    )
    second = threading.Thread(
        target=lambda: _worker(lambda: invoke("second"), errors),
        name="calibration-cold-waiter",
    )
    first.start()
    assert first_call_entered.wait(timeout=5)
    second.start()
    try:
        assert second_requested_reservation.wait(timeout=5)
        # The second caller completed preflight and is queued, while the first
        # calibration still owns the reservation and is blocked in its first call.
        assert calls == ["cpu"]
    finally:
        release_first_call.set()

    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert results["first"].metadata["status"] == "calibrated"
    assert results["second"].metadata["status"] == "cache_hit"
    assert results["second"].metadata["cache_key"] == "same-key"
    assert results["second"].metadata["winner"] == results["first"].metadata["winner"]
    assert calls == ["cpu", "cuda_strict", "cuda_strict", "cpu",
                     results["first"].metadata["winner"]]
    assert maximum_active() == 1
    assert len(cache) == 1


def test_waiter_rechecks_identity_and_uses_new_key_cache_after_release(monkeypatch):
    batch, label = _inputs()
    state = {
        "request_digest": "b" * 64, "source": "source-before", "runtime": {"threads": 2},
        "key": "pre-wait-key", "admission": None, "device_info": {"device_id": 0},
    }
    _install_fake_identity(monkeypatch, state)
    calls, _ = _install_fake_public_evaluator(monkeypatch)
    cache = calibration.BoundedCalibrationCache()
    policy = calibration.CalibrationPolicy(repetitions=2, warmups=0)
    original_reserve = coordinator.serial_calibration_measurement
    waiter_requested_reservation = threading.Event()

    def observed_reserve(timeout_seconds=None):
        if threading.current_thread().name == "identity-change-waiter":
            waiter_requested_reservation.set()
        return original_reserve(timeout_seconds)

    errors = []
    result_box = {}

    def invoke():
        result_box["result"] = calibration.evaluate_calibrated_batch(
            batch, label, metrics=("coverage",), calibration_policy=policy,
            cache=cache, gpu_policy=GPUExecutionPolicy(),
        )

    with original_reserve():
        monkeypatch.setattr(coordinator, "serial_calibration_measurement", observed_reserve)
        waiter = threading.Thread(
            target=lambda: _worker(invoke, errors), name="identity-change-waiter",
        )
        waiter.start()
        assert waiter_requested_reservation.wait(timeout=5)

        # Simulate source/runtime/request identity changing while queued, and
        # publish a qualified record under the freshly computed key.
        state.update(request_digest="c" * 64, source="source-after",
                     runtime={"threads": 4}, key="post-wait-key")
        cache.put("post-wait-key", {
            "winner": "cpu", "parity": "pass", "selection_basis": "steady_state_median",
            "cpu_median_seconds": 1.0, "cuda_median_seconds": 5.0,
            "repetitions": 2, "warmups": 0,
        })

    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert not errors
    result = result_box["result"]
    assert result.metadata["status"] == "cache_hit"
    assert result.metadata["cache_key"] == "post-wait-key"
    assert result.metadata["winner"] == "cpu"
    assert calls == ["cpu"]


def test_waiter_rechecks_device_admission_and_returns_cpu_only(monkeypatch):
    batch, label = _inputs()
    state = {
        "request_digest": "d" * 64, "source": "source-device", "runtime": {"threads": 2},
        "key": "device-key", "admission": None, "device_info": {"device_id": 0},
    }
    _install_fake_identity(monkeypatch, state)
    calls, _ = _install_fake_public_evaluator(monkeypatch)
    cache = calibration.BoundedCalibrationCache()
    policy = calibration.CalibrationPolicy(repetitions=2, warmups=0)
    original_reserve = coordinator.serial_calibration_measurement
    waiter_requested_reservation = threading.Event()
    errors = []
    result_box = {}

    def observed_reserve(timeout_seconds=None):
        if threading.current_thread().name == "device-change-waiter":
            waiter_requested_reservation.set()
        return original_reserve(timeout_seconds)

    def invoke():
        result_box["result"] = calibration.evaluate_calibrated_batch(
            batch, label, metrics=("coverage",), calibration_policy=policy,
            cache=cache, gpu_policy=GPUExecutionPolicy(),
        )

    with original_reserve():
        monkeypatch.setattr(coordinator, "serial_calibration_measurement", observed_reserve)
        waiter = threading.Thread(
            target=lambda: _worker(invoke, errors), name="device-change-waiter",
        )
        waiter.start()
        assert waiter_requested_reservation.wait(timeout=5)
        state.update(runtime={"threads": 4}, key="new-device-key",
                     admission="cuda_unavailable", device_info=None)

    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert not errors
    result = result_box["result"]
    assert result.metadata["status"] == "cpu_only_device_admission"
    assert result.metadata["winner"] == "cpu"
    assert result.metadata["device_admission"] == "cuda_unavailable"
    assert calls == ["cpu"]
    assert len(cache) == 0


def test_existing_cache_hit_never_enters_coordinator_even_if_held(monkeypatch):
    batch, label = _inputs()
    state = {
        "request_digest": "e" * 64, "source": "source-cache", "runtime": {"threads": 2},
        "key": "cached-key", "admission": None, "device_info": {"device_id": 0},
    }
    _install_fake_identity(monkeypatch, state)
    calls, _ = _install_fake_public_evaluator(monkeypatch)
    cache = calibration.BoundedCalibrationCache()
    cache.put("cached-key", {
        "winner": "cuda_strict", "parity": "pass",
        "selection_basis": "steady_state_median",
        "cpu_median_seconds": 5.0, "cuda_median_seconds": 1.0,
        "repetitions": 2, "warmups": 0,
    })
    policy = calibration.CalibrationPolicy(repetitions=2, warmups=0)

    with coordinator.serial_calibration_measurement():
        def fail_if_reserved(**_kwargs):
            raise AssertionError("cache-hit inference tried to reserve cold-measurement slot")

        monkeypatch.setattr(coordinator, "serial_calibration_measurement", fail_if_reserved)
        result = calibration.evaluate_calibrated_batch(
            batch, label, metrics=("coverage",), calibration_policy=policy,
            cache=cache, gpu_policy=GPUExecutionPolicy(),
        )

    assert result.metadata["status"] == "cache_hit"
    assert result.metadata["winner"] == "cuda_strict"
    assert calls == ["cuda_strict"]
