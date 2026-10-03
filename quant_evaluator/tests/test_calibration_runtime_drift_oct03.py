"""A fast calibration must not be cached across observed runtime drift."""
import pytest
from quant_evaluator.runtime import backend_calibration as c
from quant_evaluator.runtime import evaluator as public_runtime
from quant_evaluator.tests.test_backend_calibration import _inputs, _result


@pytest.mark.parametrize("warmups", [0, 1])
@pytest.mark.parametrize("change_on_call", [1, 2, 3, 4])
def test_observed_thread_change_during_calibration_falls_back_without_cache(
        monkeypatch, change_on_call, warmups):
    clock = [0.0]
    runtime = [2]
    calls = []
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DRIFTED", False)
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(c, "_source_fingerprint", lambda: "stable-source")
    monkeypatch.setattr(c, "_device_admission", lambda policy: (None, {"id": 0}))
    monkeypatch.setattr(c, "_runtime_fingerprint", lambda device: {"threads": runtime[0]})

    def evaluate(batch, label, *, metrics, backend, gpu_policy):
        calls.append(backend)
        clock[0] += 2.0 if backend == "cpu" else 1.0
        if len(calls) == change_on_call:
            runtime[0] = 8
        result = _result(batch)
        result.execution_route = backend
        return result

    monkeypatch.setattr(public_runtime, "evaluate", evaluate)
    batch, label = _inputs()
    cache = c.BoundedCalibrationCache()
    output = c.evaluate_calibrated_batch(
        batch, label, metrics=("ic",),
        calibration_policy=c.CalibrationPolicy(repetitions=3, warmups=warmups), cache=cache)
    assert output.metadata["status"] == "runtime_changed_cpu_fallback"
    assert output.metadata["winner"] == "cpu"
    assert output.metadata["calibration_record"]["runtime_stable"] is False
    assert output.metadata["calibration_record"]["parity"] == "pass"
    assert len(cache) == 0
    assert output.bundle.execution_route == "cpu"
    assert len(calls) == (2 if change_on_call <= 2 else 4)


def test_stable_runtime_still_admits_calibrated_winner(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DRIFTED", False)
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(c, "_source_fingerprint", lambda: "stable-source")
    monkeypatch.setattr(c, "_device_admission", lambda policy: (None, {"id": 0}))
    monkeypatch.setattr(c, "_runtime_fingerprint", lambda device: {"threads": 2})
    def evaluate(batch, label, *, metrics, backend, gpu_policy):
        clock[0] += 2.0 if backend == "cpu" else 1.0
        return _result(batch)
    monkeypatch.setattr(public_runtime, "evaluate", evaluate)
    batch, label = _inputs()
    cache = c.BoundedCalibrationCache()
    output = c.evaluate_calibrated_batch(batch, label, metrics=("ic",),
        calibration_policy=c.CalibrationPolicy(repetitions=2, warmups=0), cache=cache)
    assert output.metadata["status"] == "calibrated"
    assert output.metadata["winner"] == "cuda_strict"
    assert output.metadata["calibration_record"]["runtime_stable"] is True
    assert len(cache) == 1
