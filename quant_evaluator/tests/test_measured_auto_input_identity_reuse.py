from datetime import datetime, timedelta, tzinfo

import numpy as np

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime import auto_calibration, backend_calibration
from quant_evaluator.runtime.backend_calibration import (
    BoundedCalibrationCache, CalibrationPolicy,
)
from quant_evaluator.runtime import measured_auto_registry as registry


def _inputs(values=None, *, context_refs=None, time_axis=None):
    time_axis = time_axis or AxisRef("time", "int64", 4, np.arange(4, dtype=np.int64))
    asset_axis = AxisRef("asset", "str", 3, np.array(["a", "b", "c"]))
    batch = FactorBatch(
        factor_ids=("f1",), time_axis=time_axis, asset_axis=asset_axis,
        values=(np.arange(12, dtype=np.float64).reshape(4, 3, 1)
                if values is None else values),
        context_refs={} if context_refs is None else context_refs,
    )
    label = LabelBundle(
        target_id="forward_1d", values=np.arange(12, dtype=np.float64).reshape(4, 3),
        horizon=1, decision_time=(0, 1, 2, 3), label_start_time=(1, 2, 3, 4),
        label_end_time=(2, 3, 4, 5), asset_axis=asset_axis,
    )
    return batch, label


def _install_candidate(monkeypatch, batch, label, *, source="source-a", admission=None):
    registry.clear_registry_for_tests()
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DRIFTED", False)
    calls = {"fingerprint": 0, "source": 0, "device": 0, "runtime": 0}

    real_fingerprint = backend_calibration._request_fingerprint

    def fingerprint(*args, **kwargs):
        calls["fingerprint"] += 1
        return real_fingerprint(*args, **kwargs)

    def source_fingerprint():
        calls["source"] += 1
        return source

    def device_admission(_policy):
        calls["device"] += 1
        return admission, None if admission else {"device_id": 0, "name": "test"}

    monkeypatch.setattr(backend_calibration, "_request_fingerprint", fingerprint)
    monkeypatch.setattr(backend_calibration, "_source_fingerprint", source_fingerprint)
    monkeypatch.setattr(backend_calibration, "_device_admission", device_admission)
    def runtime_fingerprint(info):
        calls["runtime"] += 1
        return {"device": info, "runtime": "fixed"}

    monkeypatch.setattr(backend_calibration, "_runtime_fingerprint", runtime_fingerprint)

    policy = CalibrationPolicy()
    gpu_policy = GPUExecutionPolicy()
    initial = backend_calibration.calibration_identity(
        batch, label, metrics=("coverage",), calibration_policy=policy,
        gpu_policy=gpu_policy,
    )
    cache = BoundedCalibrationCache()
    record = {
        "winner": "cpu", "parity": "pass",
        "selection_basis": "steady_state_median",
        "cpu_median_seconds": 1.0, "cuda_median_seconds": 5.0,
        "repetitions": policy.repetitions, "warmups": policy.warmups,
    }
    cache.put(initial["key"], record)
    assert registry.register_candidate(
        cache, initial["key"], status="calibrated", winner="cpu",
        calibration_record=record, source_check={"mode": "strict_full_content"},
        process_source_drifted=False, calibration_policy=policy,
        gpu_policy=gpu_policy,
        descriptor=auto_calibration.measured_auto_descriptor(
            batch, label, ("coverage",)),
        setup_seconds=0.1, batch=batch, label=label, metrics=("coverage",),
        request_digest=initial["request_digest"],
    )
    calls["fingerprint"] = 0
    calls["source"] = 0
    calls["device"] = 0
    calls["runtime"] = 0
    return calls, gpu_policy, cache


def test_same_live_objects_reuse_digest_but_recheck_source_and_device(monkeypatch):
    batch, label = _inputs()
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label)
    selected = auto_calibration.select_measured_auto_backend(
        batch, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    selected_again = auto_calibration.select_measured_auto_backend(
        batch, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[:3] == ("cpu", "measured_auto_exact_candidate", True)
    assert selected_again[:3] == selected[:3]
    assert calls == {"fingerprint": 0, "source": 2, "device": 2, "runtime": 2}


def test_same_schema_new_values_still_use_full_fingerprint(monkeypatch):
    batch, label = _inputs()
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label)
    changed = FactorBatch(
        factor_ids=batch.factor_ids, time_axis=batch.time_axis,
        asset_axis=batch.asset_axis, values=np.asarray(batch.values) + 1.0,
    )
    selected = auto_calibration.select_measured_auto_backend(
        changed, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[0] == "cuda_strict"
    assert selected[2] is False
    assert selected[3]["rejection_reason"] == "identity_mismatch"
    assert calls["fingerprint"] == 1


def test_same_objects_reject_source_drift_and_device_failure(monkeypatch):
    batch, label = _inputs()
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label, source="source-a")
    monkeypatch.setattr(backend_calibration, "_source_fingerprint",
                        lambda: "source-b")
    selected = auto_calibration.select_measured_auto_backend(
        batch, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[0] == "cuda_strict"
    assert selected[3]["rejection_reason"] == "source_drift"
    assert calls["fingerprint"] == 0

    batch2, label2 = _inputs()
    calls2, gpu_policy2, _cache2 = _install_candidate(
        monkeypatch, batch2, label2, admission="insufficient_cuda_memory")
    selected2 = auto_calibration.select_measured_auto_backend(
        batch2, label2, metrics=("coverage",), gpu_policy=gpu_policy2,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected2[0] == "cuda_strict"
    assert selected2[3]["rejection_reason"] == "device_admission_failed"
    assert calls2["fingerprint"] == 0
    assert calls2["device"] == 1


def test_dead_identity_weakrefs_fall_back_to_content_hash(monkeypatch):
    import gc
    import weakref

    batch, label = _inputs()
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label)
    batch_ref, label_ref = weakref.ref(batch), weakref.ref(label)
    del batch, label
    gc.collect()
    assert batch_ref() is None
    assert label_ref() is None
    assert registry.registry_size() == 1

    reconstructed_batch, reconstructed_label = _inputs()
    selected = auto_calibration.select_measured_auto_backend(
        reconstructed_batch, reconstructed_label, metrics=("coverage",),
        gpu_policy=gpu_policy, static_backend="cuda_strict", static_reason="static",
    )
    assert selected[:3] == ("cpu", "measured_auto_exact_candidate", True)
    assert calls["fingerprint"] == 1
    assert calls["source"] == 1
    assert calls["device"] == 1
    assert calls["runtime"] == 1


def test_metrics_mismatch_cannot_reuse_candidate_digest(monkeypatch):
    batch, label = _inputs()
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label)
    selected = auto_calibration.select_measured_auto_backend(
        batch, label, metrics=("coverage", "ic"), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[0] == "cuda_strict"
    assert selected[3]["rejection_reason"] == "no_candidate"
    assert calls == {"fingerprint": 0, "source": 0, "device": 0, "runtime": 0}


def test_contract_subclass_falls_back_to_full_fingerprint(monkeypatch):
    batch, label = _inputs()

    class DerivedBatch(FactorBatch):
        pass

    derived = DerivedBatch(
        factor_ids=batch.factor_ids, time_axis=batch.time_axis,
        asset_axis=batch.asset_axis, values=batch.values,
    )
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, derived, label)
    selected = auto_calibration.select_measured_auto_backend(
        derived, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[0] == "cpu", selected
    assert selected[2] is True
    assert calls["fingerprint"] == 1


def test_axis_subclass_falls_back_to_full_fingerprint(monkeypatch):
    batch, label = _inputs()

    class DerivedAxis(AxisRef):
        pass

    derived_time = DerivedAxis(
        name=batch.time_axis.name, dtype=batch.time_axis.dtype,
        size=batch.time_axis.size, values=batch.time_axis.values,
    )
    derived = FactorBatch(
        factor_ids=batch.factor_ids, time_axis=derived_time,
        asset_axis=batch.asset_axis, values=batch.values,
    )
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, derived, label)
    selected = auto_calibration.select_measured_auto_backend(
        derived, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[0] == "cpu", selected
    assert selected[2] is True
    assert calls["fingerprint"] == 1


class _MutableTimezone(tzinfo):
    def __init__(self):
        self.hours = 0

    def utcoffset(self, value):
        return timedelta(hours=self.hours)

    def dst(self, value):
        return timedelta(0)

    def tzname(self, value):
        return f"UTC{self.hours:+d}"


def test_mutable_timezone_in_context_uses_full_fingerprint(monkeypatch):
    zone = _MutableTimezone()
    batch, label = _inputs(context_refs={
        "captured_at": datetime(2024, 1, 1, tzinfo=zone),
    })
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label)
    zone.hours = 4
    selected = auto_calibration.select_measured_auto_backend(
        batch, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[0] == "cuda_strict"
    assert selected[3]["rejection_reason"] == "identity_mismatch"
    assert calls["fingerprint"] == 1


def test_object_datetime_axis_uses_full_fingerprint(monkeypatch):
    zone = _MutableTimezone()
    time_axis = AxisRef(
        "time", "object", 4,
        np.array([
            datetime(2024, 1, day, tzinfo=zone) for day in range(1, 5)
        ], dtype=object),
    )
    batch, label = _inputs(time_axis=time_axis)
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label)
    zone.hours = 5
    selected = auto_calibration.select_measured_auto_backend(
        batch, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[0] == "cuda_strict"
    assert selected[3]["rejection_reason"] == "identity_mismatch"
    assert calls["fingerprint"] == 1


def test_nested_primitive_context_remains_digest_reuse_eligible(monkeypatch):
    batch, label = _inputs(context_refs={
        "versions": {"factor": ["v1", "v2"]},
        "groups": ["g1", "g2"],
    })
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label)
    selected = auto_calibration.select_measured_auto_backend(
        batch, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[:3] == ("cpu", "measured_auto_exact_candidate", True)
    assert calls["fingerprint"] == 0


def test_immutability_walk_limit_falls_back_to_full_fingerprint(monkeypatch):
    batch, label = _inputs(context_refs={"items": list(range(1100))})
    calls, gpu_policy, _cache = _install_candidate(monkeypatch, batch, label)
    selected = auto_calibration.select_measured_auto_backend(
        batch, label, metrics=("coverage",), gpu_policy=gpu_policy,
        static_backend="cuda_strict", static_reason="static",
    )
    assert selected[:3] == ("cpu", "measured_auto_exact_candidate", True)
    assert calls["fingerprint"] == 1


def test_immutability_walk_preflights_container_width(monkeypatch):
    from quant_evaluator.contracts.metric_artifacts import FrozenMapping
    from quant_evaluator.runtime import immutable_input_identity

    monkeypatch.setattr(immutable_input_identity, "MAX_IMMUTABILITY_NODES", 4)
    assert not immutable_input_identity._deeply_immutable(tuple(range(100_000)))
    assert not immutable_input_identity._deeply_immutable(
        FrozenMapping({f"k{i}": i for i in range(100)})
    )
