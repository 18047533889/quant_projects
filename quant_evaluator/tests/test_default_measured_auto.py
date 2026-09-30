from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime import auto_calibration, backend_calibration
from quant_evaluator.runtime.backend_calibration import (
    BoundedCalibrationCache, CalibrationPolicy, _parity_mismatch,
)
from quant_evaluator.runtime import measured_auto_registry as registry
from quant_evaluator.runtime.evaluator import evaluate


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear_registry_for_tests()
    yield
    registry.clear_registry_for_tests()


def inputs(values=None):
    time_axis = AxisRef("time", "int64", 4, np.arange(4, dtype=np.int64))
    asset_axis = AxisRef("asset", "str", 3, np.array(["a", "b", "c"]))
    batch = FactorBatch(
        factor_ids=("f1", "f2"), time_axis=time_axis, asset_axis=asset_axis,
        values=(np.arange(24, dtype=np.float64).reshape(4, 3, 2) if values is None else values),
    )
    label = LabelBundle(
        target_id="forward_1d", values=np.arange(12, dtype=np.float64).reshape(4, 3),
        horizon=1, decision_time=(0, 1, 2, 3), label_start_time=(1, 2, 3, 4),
        label_end_time=(2, 3, 4, 5), asset_axis=asset_axis,
    )
    return batch, label


def add_candidate(batch, label, *, winner="cpu"):
    cache = BoundedCalibrationCache()
    policy = CalibrationPolicy()
    gpu_policy = GPUExecutionPolicy()
    key = "measured-key"
    timings = {"cpu": 1.0, "cuda_strict": 5.0} if winner == "cpu" else {
        "cpu": 5.0, "cuda_strict": 1.0,
    }
    record = {
        "winner": winner, "parity": "pass",
        "selection_basis": "steady_state_median",
        "cpu_median_seconds": timings["cpu"],
        "cuda_median_seconds": timings["cuda_strict"],
        "repetitions": policy.repetitions, "warmups": policy.warmups,
    }
    cache.put(key, record)
    descriptor = auto_calibration.measured_auto_descriptor(batch, label, ("coverage",))
    assert registry.register_candidate(
        cache, key, status="calibrated", winner=winner,
        calibration_record=record, source_check={"mode": "strict_full_content"},
        process_source_drifted=False, calibration_policy=policy,
        gpu_policy=gpu_policy, descriptor=descriptor, setup_seconds=0.1,
    )
    return cache


def mock_static(monkeypatch, backend="cpu", reason="static_test_route"):
    monkeypatch.setattr(
        "quant_evaluator.runtime.evaluator._select_public_auto_backend",
        lambda *args, **kwargs: (backend, reason),
    )


def test_no_candidate_does_not_fingerprint_inputs_or_scan_source(monkeypatch):
    batch, label = inputs()
    mock_static(monkeypatch)
    monkeypatch.setattr(backend_calibration, "_request_fingerprint",
                        lambda *args, **kwargs: pytest.fail("unexpected input fingerprint"))
    monkeypatch.setattr(backend_calibration, "_source_fingerprint",
                        lambda *args, **kwargs: pytest.fail("unexpected source scan"))

    result = evaluate(batch, label, metrics=("coverage",))
    assert result.metadata["auto_backend_reason"] == "static_test_route"
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_validation_seconds"] == 0.0
    assert result.metadata["auto_backend_rejection_reason"] == "no_candidate"


def test_exact_candidate_is_adopted_and_marked_in_receipt(monkeypatch):
    batch, label = inputs()
    _cache = add_candidate(batch, label, winner="cpu")
    mock_static(monkeypatch, backend="cuda_strict")
    calls = []

    def exact_identity(*args, **kwargs):
        calls.append(kwargs)
        return {
            "key": "measured-key", "device_admission": None,
            "process_source_drifted": False,
            "source_metadata": {"mode": "strict_full_content"},
        }

    monkeypatch.setattr(auto_calibration, "calibration_identity", exact_identity)
    result = evaluate(batch, label, metrics=("coverage",))
    assert calls and calls[0]["calibration_policy"].repetitions == 3
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "measured_auto_exact_candidate"
    assert result.metadata["auto_backend_policy"] == "measured_auto_registry_exact_v1"
    assert result.metadata["auto_backend_profile"] == "exact_content_process_local"
    assert result.metadata["execution_receipt"]["auto_backend_reason"] == "measured_auto_exact_candidate"
    assert result.metadata["auto_backend_validation_seconds"] >= 0.0
    assert result.metadata["auto_backend_rejection_reason"] is None


@pytest.mark.parametrize("identity", [
    {"key": "changed-input-key", "device_admission": None,
     "process_source_drifted": False, "source_metadata": {"mode": "strict_full_content"}},
    {"key": "measured-key", "device_admission": "insufficient_cuda_memory",
     "process_source_drifted": False, "source_metadata": {"mode": "strict_full_content"}},
    {"key": "measured-key", "device_admission": None,
     "process_source_drifted": True, "source_metadata": {"mode": "strict_full_content"}},
])
def test_changed_identity_or_current_admission_falls_back(monkeypatch, identity):
    batch, label = inputs()
    # No supplied value_hash means same cheap descriptor; only the full
    # identity callback can reject changed array contents/source/runtime.
    _cache = add_candidate(batch, label, winner="cuda_strict")
    mock_static(monkeypatch, backend="cpu")
    monkeypatch.setattr(auto_calibration, "calibration_identity",
                        lambda *args, **kwargs: identity)

    result = evaluate(batch, label, metrics=("coverage",))
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "static_test_route"


def test_same_static_route_skips_exact_validation(monkeypatch):
    batch, label = inputs()
    _cache = add_candidate(batch, label, winner="cpu")
    mock_static(monkeypatch, backend="cpu")
    monkeypatch.setattr(auto_calibration, "calibration_identity",
                        lambda *args, **kwargs: pytest.fail("same route must skip validation"))

    result = evaluate(batch, label, metrics=("coverage",))
    assert result.metadata["auto_backend_reason"] == "static_test_route"
    assert result.metadata["auto_backend_validation_seconds"] == 0.0
    assert result.metadata["auto_backend_rejection_reason"] == "candidate_matches_static_route"


def test_advanced_request_stays_on_static_route(monkeypatch):
    batch, label = inputs()
    _cache = add_candidate(batch, label, winner="cuda_strict")
    mock_static(monkeypatch, backend="cpu")
    monkeypatch.setattr(auto_calibration, "calibration_identity",
                        lambda *args, **kwargs: pytest.fail("advanced request must bypass measured auto"))

    result = evaluate(batch, label, metrics=("coverage",), quantile_builder_parameters={"n_quantiles": 2})
    assert result.metadata["auto_backend_reason"] == "static_test_route"
    assert result.metadata["backend_used"] == "cpu"


def test_descriptor_keeps_raw_metric_alias_and_public_duplicates_still_fail():
    batch, label = inputs()
    alias = auto_calibration.measured_auto_descriptor(batch, label, ("ic",))
    canonical = auto_calibration.measured_auto_descriptor(batch, label, ("pearson_ic",))
    assert alias.metrics == ("ic",)
    assert alias != canonical
    with pytest.raises(InvalidContractError, match="unique"):
        evaluate(batch, label, metrics=("coverage", "coverage"))


def test_real_exact_fingerprint_rejects_changed_values_and_reuses_measured_policy(monkeypatch):
    batch, label = inputs()
    policy = CalibrationPolicy(repetitions=2, max_wall_time_seconds=360.0)
    gpu_policy = GPUExecutionPolicy()
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DRIFTED", False)
    monkeypatch.setattr(backend_calibration, "_source_fingerprint", lambda: "stable-source")
    monkeypatch.setattr(backend_calibration, "_device_admission",
                        lambda _policy: (None, {"device_id": 0, "name": "test-gpu"}))
    monkeypatch.setattr(backend_calibration, "_runtime_fingerprint",
                        lambda info: {"device": info, "runtime": "test-runtime"})
    exact = backend_calibration.calibration_identity(
        batch, label, metrics=("coverage",), calibration_policy=policy,
        gpu_policy=gpu_policy,
    )
    cache = BoundedCalibrationCache()
    record = {
        "winner": "cuda_strict", "parity": "pass",
        "selection_basis": "steady_state_median",
        "cpu_median_seconds": 5.0, "cuda_median_seconds": 1.0,
        "repetitions": 2, "warmups": 1,
    }
    cache.put(exact["key"], record)
    assert registry.register_candidate(
        cache, exact["key"], status="calibrated", winner="cuda_strict",
        calibration_record=record, source_check={"mode": "strict_full_content"},
        process_source_drifted=False, calibration_policy=policy,
        gpu_policy=gpu_policy,
        descriptor=auto_calibration.measured_auto_descriptor(batch, label, ("coverage",)),
        setup_seconds=0.1,
    )
    changed = FactorBatch(
        factor_ids=batch.factor_ids, time_axis=batch.time_axis,
        asset_axis=batch.asset_axis, values=np.array(batch.values) + 0.5,
    )
    mock_static(monkeypatch, backend="cpu")
    seen_policies = []
    real_identity = backend_calibration.calibration_identity

    def observe_identity(*args, **kwargs):
        seen_policies.append(kwargs["calibration_policy"])
        return real_identity(*args, **kwargs)

    monkeypatch.setattr(auto_calibration, "calibration_identity", observe_identity)
    result = evaluate(changed, label, metrics=("coverage",))
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "static_test_route"
    assert len(seen_policies) == 1
    assert seen_policies[0].repetitions == 2
    assert seen_policies[0].max_wall_time_seconds == 360.0


def test_single_backend_missing_result_field_fails_parity():
    cpu = SimpleNamespace(factor_ids=("f1",), metric_values={})
    cuda = SimpleNamespace(factor_ids=("f1",))
    assert _parity_mismatch(cpu, cuda, CalibrationPolicy()) == (
        "metric_values: result field is missing on one backend"
    )


def test_both_injected_results_may_omit_all_parity_fields():
    cpu = SimpleNamespace(factor_ids=("f1",))
    cuda = SimpleNamespace(factor_ids=("f1",))
    assert _parity_mismatch(cpu, cuda, CalibrationPolicy()) is None


def test_validation_cost_at_or_above_median_advantage_keeps_static_route(monkeypatch):
    batch, label = inputs()
    _cache = add_candidate(batch, label, winner="cuda_strict")
    mock_static(monkeypatch, backend="cpu", reason="static_test_route")
    monkeypatch.setattr(auto_calibration, "calibration_identity", lambda *args, **kwargs: {
        "key": "measured-key", "device_admission": None,
        "process_source_drifted": False,
        "source_metadata": {"mode": "strict_full_content"},
        "setup_seconds": 4.0,
    })
    result = evaluate(batch, label, metrics=("coverage",))
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "static_test_route"
    assert result.metadata["auto_backend_validation_seconds"] == 4.0
    assert result.metadata["auto_backend_rejection_reason"] == "validation_cost_exceeds_median_advantage"



def test_full_fingerprint_covers_factor_mask_axes_and_label(monkeypatch):
    from dataclasses import replace

    batch, label = inputs()
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DRIFTED", False)
    monkeypatch.setattr(backend_calibration, "_source_fingerprint", lambda: "fixed-source")
    monkeypatch.setattr(backend_calibration, "_device_admission",
                        lambda _policy: (None, {"device_id": 0, "name": "test-gpu"}))
    monkeypatch.setattr(backend_calibration, "_runtime_fingerprint",
                        lambda info: {"device": info, "runtime": "fixed-runtime"})
    policy = CalibrationPolicy(max_wall_time_seconds=360.0)

    def key_for(candidate_batch=batch, candidate_label=label):
        return backend_calibration.calibration_identity(
            candidate_batch, candidate_label, metrics=("coverage",),
            calibration_policy=policy, gpu_policy=GPUExecutionPolicy(),
        )["key"]

    baseline = key_for()
    changed_mask = FactorBatch(
        factor_ids=batch.factor_ids, time_axis=batch.time_axis,
        asset_axis=batch.asset_axis, values=batch.values,
        validity=np.zeros(batch.values.shape, dtype=bool),
    )
    changed_time = FactorBatch(
        factor_ids=batch.factor_ids,
        time_axis=AxisRef("time", "int64", 4, np.arange(4, dtype=np.int64) + 10),
        asset_axis=batch.asset_axis, values=batch.values,
    )
    changed_assets = np.array(["x", "y", "z"])
    changed_axis = AxisRef("asset", "str", 3, changed_assets)
    changed_asset_batch = FactorBatch(
        factor_ids=batch.factor_ids, time_axis=batch.time_axis,
        asset_axis=changed_axis, values=batch.values,
    )
    changed_label = replace(label, values=label.values + 1.0)
    changed_label_mask = replace(label, validity=np.zeros(label.values.shape, dtype=bool))
    changed_label_axis = replace(label, asset_axis=changed_axis)

    for candidate_batch, candidate_label in (
        (changed_mask, label), (changed_time, label),
        (changed_asset_batch, label), (batch, changed_label),
        (batch, changed_label_mask), (batch, changed_label_axis),
    ):
        assert key_for(candidate_batch, candidate_label) != baseline


def test_source_runtime_and_device_identity_change_full_key(monkeypatch):
    batch, label = inputs()
    source = ["source-a"]
    device = [{"device_id": 0, "driver": 1}]
    runtime = ["runtime-a"]
    monkeypatch.setattr(backend_calibration, "_source_fingerprint", lambda: source[0])
    monkeypatch.setattr(backend_calibration, "_device_admission",
                        lambda _policy: (None, dict(device[0])))
    monkeypatch.setattr(backend_calibration, "_runtime_fingerprint",
                        lambda info: {"device": info, "runtime": runtime[0]})
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DRIFTED", False)
    policy = CalibrationPolicy(max_wall_time_seconds=360.0)

    def key():
        return backend_calibration.calibration_identity(
            batch, label, metrics=("coverage",), calibration_policy=policy,
            gpu_policy=GPUExecutionPolicy(),
        )["key"]

    baseline = key()
    runtime[0] = "runtime-b"
    assert key() != baseline
    runtime[0] = "runtime-a"
    device[0] = {"device_id": 0, "driver": 2}
    assert key() != baseline
    device[0] = {"device_id": 0, "driver": 1}
    source[0] = "source-b"
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DRIFTED", False)
    assert key() != baseline



def test_validation_timing_is_diagnostic_and_does_not_change_receipt_hash(monkeypatch):
    batch, label = inputs()
    _cache = add_candidate(batch, label, winner="cpu")
    mock_static(monkeypatch, backend="cuda_strict", reason="static_test_route")
    elapsed = [1.0]

    def identity(*args, **kwargs):
        return {
            "key": "measured-key", "device_admission": None,
            "process_source_drifted": False,
            "source_metadata": {"mode": "strict_full_content"},
            "setup_seconds": elapsed[0],
        }

    monkeypatch.setattr(auto_calibration, "calibration_identity", identity)
    first = evaluate(batch, label, metrics=("coverage",))
    elapsed[0] = 2.0
    second = evaluate(batch, label, metrics=("coverage",))

    assert first.metadata["auto_backend_validation_seconds"] == 1.0
    assert second.metadata["auto_backend_validation_seconds"] == 2.0
    assert first.metadata["execution_receipt"]["receipt_hash"] == second.metadata["execution_receipt"]["receipt_hash"]
    assert "auto_backend_validation_seconds" not in first.metadata["execution_receipt"]
    assert "auto_backend_rejection_reason" not in first.metadata["execution_receipt"]


def test_explicit_backend_keeps_v1_receipt_hash_route_contract():
    from quant_evaluator.contracts._hashutil import stable_content_hex

    batch, label = inputs()
    result = evaluate(batch, label, metrics=("coverage",), backend="cpu")
    receipt = result.metadata["execution_receipt"]
    route = {
        key: receipt[key]
        for key in (
            "backend_requested", "backend_strategy", "backend_used",
            "auto_backend_policy", "auto_backend_profile", "auto_backend_reason",
            "metric_backends",
        )
    }
    assert receipt["receipt_hash"] == stable_content_hex(
        tag="EvaluationExecutionReceipt.v1",
        fields={"config_hash": receipt["config_hash"], **route},
    )
    assert "auto_backend_validation_seconds" not in receipt
    assert "auto_backend_rejection_reason" not in receipt
