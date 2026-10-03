"""Public default-auto candidate validation against runtime identity drift."""
import numpy as np
import pytest

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime import auto_calibration, backend_calibration
from quant_evaluator.runtime.backend_calibration import (
    BoundedCalibrationCache, CalibrationPolicy,
)
from quant_evaluator.runtime import measured_auto_registry as registry
from quant_evaluator.runtime.evaluator import evaluate


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear_registry_for_tests()
    yield
    registry.clear_registry_for_tests()


def _inputs():
    time_axis = AxisRef("time", "int64", 4, np.arange(4, dtype=np.int64))
    asset_axis = AxisRef("asset", "str", 3, np.array(["a", "b", "c"]))
    batch = FactorBatch(
        factor_ids=("f1",), time_axis=time_axis, asset_axis=asset_axis,
        values=np.arange(12, dtype=np.float64).reshape(4, 3, 1),
    )
    label = LabelBundle(
        target_id="forward_1d", values=np.arange(12, dtype=np.float64).reshape(4, 3),
        horizon=1, decision_time=(0, 1, 2, 3), label_start_time=(1, 2, 3, 4),
        label_end_time=(2, 3, 4, 5), asset_axis=asset_axis,
    )
    return batch, label


def _require_cuda_for_static_fallback():
    reason, _ = backend_calibration._device_admission(GPUExecutionPolicy())
    if reason:
        pytest.skip("static fallback route requires admitted CUDA: " + str(reason))


def _install_candidate(monkeypatch, batch, label, *, runtime, winner="cpu"):
    policy = CalibrationPolicy()
    gpu_policy = GPUExecutionPolicy()
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DRIFTED", False)
    monkeypatch.setattr(backend_calibration, "_source_fingerprint", lambda: "stable-source")
    monkeypatch.setattr(
        backend_calibration, "_device_admission",
        lambda _: (None, {"device_id": 0, "name": "test-device"}),
    )
    monkeypatch.setattr(backend_calibration, "_runtime_fingerprint", lambda info: {
        "device": info, "runtime": dict(runtime),
    })
    identity = backend_calibration.calibration_identity(
        batch, label, metrics=("coverage",), calibration_policy=policy,
        gpu_policy=gpu_policy,
    )
    cache = BoundedCalibrationCache()
    record = {
        "winner": winner, "parity": "pass",
        "selection_basis": "steady_state_median",
        "cpu_median_seconds": 1.0 if winner == "cpu" else 5.0,
        "cuda_median_seconds": 5.0 if winner == "cpu" else 1.0,
        "repetitions": policy.repetitions, "warmups": policy.warmups,
    }
    cache.put(identity["key"], record)
    assert registry.register_candidate(
        cache, identity["key"], status="calibrated", winner=winner,
        calibration_record=record, source_check=identity["source_metadata"],
        process_source_drifted=False, calibration_policy=policy,
        gpu_policy=gpu_policy,
        descriptor=auto_calibration.measured_auto_descriptor(
            batch, label, ("coverage",)),
        setup_seconds=0.1, batch=batch, label=label, metrics=("coverage",),
        request_digest=identity["request_digest"],
    )
    monkeypatch.setattr(
        "quant_evaluator.runtime.evaluator._select_public_auto_backend",
        lambda *args, **kwargs: ("cuda_strict", "static_test_route"),
    )
    return cache, identity["key"]


def test_default_auto_consumes_exact_cached_winner(monkeypatch):
    batch, label = _inputs()
    _cache, _key = _install_candidate(monkeypatch, batch, label, runtime={"threads": 4})

    result = evaluate(batch, label, metrics=("coverage",))

    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "measured_auto_exact_candidate"


def test_runtime_drift_evicts_candidate_until_qualified_again(monkeypatch):
    _require_cuda_for_static_fallback()
    batch, label = _inputs()
    runtime = {"threadpool": (("openblas", 4),)}
    _cache, _key = _install_candidate(monkeypatch, batch, label, runtime=runtime)

    runtime["threadpool"] = (("openblas", 8),)
    changed = evaluate(batch, label, metrics=("coverage",))
    assert changed.metadata["backend_used"] == "cuda"
    assert changed.metadata["auto_backend_reason"] == "static_test_route"
    assert changed.metadata["auto_backend_rejection_reason"] == "identity_mismatch"
    assert registry.registry_size() == 0

    runtime["threadpool"] = (("openblas", 4),)
    restored = evaluate(batch, label, metrics=("coverage",))
    assert restored.metadata["backend_used"] == "cuda"
    assert restored.metadata["auto_backend_reason"] == "static_test_route"
    assert restored.metadata["auto_backend_rejection_reason"] == "no_candidate"

    _requalified_cache, _requalified_key = _install_candidate(
        monkeypatch, batch, label, runtime=runtime)
    requalified = evaluate(batch, label, metrics=("coverage",))
    assert requalified.metadata["backend_used"] == "cpu"
    assert requalified.metadata["auto_backend_reason"] == "measured_auto_exact_candidate"


def test_default_auto_rejects_candidate_when_cached_record_no_longer_matches(monkeypatch):
    _require_cuda_for_static_fallback()
    batch, label = _inputs()
    cache, key = _install_candidate(monkeypatch, batch, label, runtime={"threads": 4})
    cache.put(key, {
        "winner": "cuda_strict", "parity": "pass",
        "selection_basis": "steady_state_median",
        "cpu_median_seconds": 1.0, "cuda_median_seconds": 5.0,
        "repetitions": 3, "warmups": 1,
    })

    result = evaluate(batch, label, metrics=("coverage",))

    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["auto_backend_reason"] == "static_test_route"
    assert result.metadata["auto_backend_rejection_reason"] == "calibration_cache_missing_or_stale"


def test_default_auto_rejects_candidate_when_calibration_policy_identity_differs(monkeypatch):
    _require_cuda_for_static_fallback()
    batch, label = _inputs()
    _cache, _key = _install_candidate(monkeypatch, batch, label, runtime={"threads": 4})
    policy_type = auto_calibration.CalibrationPolicy

    def changed_policy(**fields):
        fields["repetitions"] = 2
        return policy_type(**fields)

    monkeypatch.setattr(auto_calibration, "CalibrationPolicy", changed_policy)

    result = evaluate(batch, label, metrics=("coverage",))

    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["auto_backend_reason"] == "static_test_route"
    assert result.metadata["auto_backend_rejection_reason"] == "identity_mismatch"
