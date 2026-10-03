"""CPU-only contract tests for the optional GPU factor tile cap."""
import numpy as np
from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.api import factor_source as factor_source_api
from types import SimpleNamespace

import pytest

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.gpu_executor import GPUExecutor


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_gpu_tile_cap_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="max_factor_tile_size"):
        GPUExecutionPolicy(max_factor_tile_size=value)


def test_candidates_keep_default_and_filter_custom_caps():
    assert GPUExecutionPolicy().factor_tile_candidates() == (128, 64, 32, 16, 8, 4, 2, 1)
    assert GPUExecutionPolicy(max_factor_tile_size=10).factor_tile_candidates() == (8, 4, 2, 1)
    assert GPUExecutionPolicy(max_factor_tile_size=3).factor_tile_candidates() == (2, 1)


def test_session_cap_does_not_bypass_memory_admission():
    session = DeviceEvaluationSession(GPUExecutionPolicy(max_factor_tile_size=10))
    session._vram_budget = 10_000
    session._estimate_working_set = lambda _m, _t, _n, tile, _d, **_kw: tile * 100
    assert session.estimate_tile(("rank_ic",), 2, 3, 8) == 8
    session._vram_budget = 350
    assert session.estimate_tile(("rank_ic",), 2, 3, 8) == 2


def test_resident_label_estimator_uses_same_cap_candidates():
    class Session:
        policy = GPUExecutionPolicy(max_factor_tile_size=3)
        _pool = SimpleNamespace(used_bytes=lambda: 0)
        _vram_budget = 10_000
        _estimate_working_set = lambda self, _m, _t, _n, tile, _d: tile * 100

    executor = GPUExecutor(Session())
    assert executor._factor_tile_size_with_resident_labels(
        ("rank_ic",), 2, 3, 10, 8, candidate_limit=10) == 2


def test_calibration_policy_identity_fields_include_tile_cap():
    from dataclasses import fields

    names = {item.name for item in fields(GPUExecutionPolicy)}
    assert "max_factor_tile_size" in names
    assert GPUExecutionPolicy().max_factor_tile_size != GPUExecutionPolicy(
        max_factor_tile_size=8).max_factor_tile_size


def _source_inputs():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.factor_tile_source import FactorTile
    from quant_evaluator.contracts.label_bundle import LabelBundle

    time = np.arange(4, dtype=np.int64)
    assets = np.arange(5, dtype=np.int64)
    ta = AxisRef("time", "int64", len(time), time)
    aa = AxisRef("asset", "int64", len(assets), assets)
    values = np.arange(4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3)
    batch = FactorBatch(("a", "b", "c"), ta, aa, values)
    label = LabelBundle("ret", values[:, :, 0], 1,
                        decision_time=tuple(time), label_start_time=tuple(time),
                        label_end_time=tuple(time + 1), asset_axis=aa)

    class Source:
        factor_ids, time_axis, asset_axis, dtype = batch.factor_ids, ta, aa, batch.dtype
        snapshot_id = "tile-policy-test"
        max_tile_size = 3

        def __init__(self):
            self.reads = []

        def read_tile(self, start, end):
            self.reads.append((start, end))
            tile_batch = FactorBatch(self.factor_ids[start:end], ta, aa,
                                     values[:, :, start:end])
            return FactorTile(start, end, tile_batch, self.snapshot_id)

        def close(self):
            pass

    return Source(), label


def test_source_auto_cap_below_certified_width_falls_back_without_shrinking_cpu(monkeypatch):
    evidence = SimpleNamespace(effective_tile_width=2, minimum_effective_vram_bytes=0,
                               legacy_reason="certified", evidence_id="test",
                               evidence_artifacts=(), evidence_status="measured_source_ab")
    monkeypatch.setattr(factor_source_api, "select_source_auto_route", lambda **_kw: evidence)
    source, label = _source_inputs()
    result = factor_source_api.evaluate_factor_source_batch(
        source, label, metrics=("coverage",), backend="auto",
        gpu_policy=GPUExecutionPolicy(max_factor_tile_size=1))
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "source_policy_tile_cap_outside_certified_tile"
    assert result.metadata["execution_receipt"]["effective_max_tile_size"] == 3
    assert source.reads == [(0, 3)]


def test_source_gpu_routes_apply_policy_cap_but_cpu_route_does_not(monkeypatch):
    calls = []

    class Session:
        def __init__(self, policy):
            self.policy = policy

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    class Executor:
        def __init__(self, _session):
            pass

        def run_source_tiled(self, source, label, metrics, *, max_tile_size, source_metadata):
            calls.append(max_tile_size)
            return SimpleNamespace(metadata={"factor_tile_size": max_tile_size})

    monkeypatch.setattr(factor_source_api, "select_source_auto_route", lambda **_kw: None)
    monkeypatch.setattr("quant_evaluator.runtime.device_session.DeviceEvaluationSession", Session)
    monkeypatch.setattr("quant_evaluator.runtime.gpu_executor.GPUExecutor", Executor)

    strict_source, label = _source_inputs()
    strict = factor_source_api.evaluate_factor_source_batch(
        strict_source, label, metrics=("coverage",), backend="cuda_strict",
        gpu_policy=GPUExecutionPolicy(max_factor_tile_size=1))
    assert calls == [1]
    assert strict.metadata["execution_receipt"]["max_factor_tile_size"] == 1
    assert strict.metadata["execution_receipt"]["effective_gpu_factor_tile_size"] == 1

    cpu_source, label = _source_inputs()
    cpu = factor_source_api.evaluate_factor_source_batch(
        cpu_source, label, metrics=("coverage",), backend="cpu",
        gpu_policy=GPUExecutionPolicy(max_factor_tile_size=1))
    assert calls == [1]
    assert cpu_source.reads == [(0, 3)]
    assert cpu.metadata["execution_receipt"]["effective_max_tile_size"] == 3


def test_static_materialized_auto_rejects_custom_tile_cap(monkeypatch):
    from quant_evaluator.runtime import evaluator

    batch, label = _source_inputs()
    batch = SimpleNamespace(num_times=4, num_assets=5, num_factors=3,
                            values=np.asarray(batch.read_tile(0, 3).batch.values))
    monkeypatch.setattr(evaluator, "_AUTO_CUDA_METRICS", frozenset({"coverage"}))
    monkeypatch.setattr(evaluator, "_auto_public_shape_profile", lambda _batch: "test-profile")
    route = evaluator._select_public_auto_backend(
        batch, label, ("coverage",), metric_parameters=None,
        context=None, quantile_builder_parameters=None, portfolio_returns=None,
        holding_returns=None, trade_eligibility=None, calendar_snapshot=None,
        exposure_panel=None, generalization_evidence=None, evaluator=None,
        gpu_policy=GPUExecutionPolicy(max_factor_tile_size=4))
    assert route == ("cpu", "gpu_factor_tile_cap_outside_certified_profile")


def test_source_auto_keeps_certified_width_when_policy_cap_is_larger(monkeypatch):
    evidence = SimpleNamespace(effective_tile_width=2, minimum_effective_vram_bytes=0,
                               legacy_reason="certified", evidence_id="test",
                               evidence_artifacts=(), evidence_status="measured_source_ab")
    monkeypatch.setattr(factor_source_api, "select_source_auto_route", lambda **_kw: evidence)
    monkeypatch.setattr("quant_evaluator.runtime.evaluator._auto_batch_cuda_rejection",
                        lambda *_args: None)
    calls = []

    class Session:
        def __init__(self, policy):
            self.policy = policy

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    class Executor:
        def __init__(self, _session):
            pass

        def run_source_tiled(self, _source, _label, _metrics, *, max_tile_size,
                             source_metadata):
            calls.append(max_tile_size)
            return SimpleNamespace(metadata={"factor_tile_size": max_tile_size})

    monkeypatch.setattr("quant_evaluator.runtime.device_session.DeviceEvaluationSession", Session)
    monkeypatch.setattr("quant_evaluator.runtime.gpu_executor.GPUExecutor", Executor)
    source, label = _source_inputs()
    result = factor_source_api.evaluate_factor_source_batch(
        source, label, metrics=("coverage",), backend="auto",
        gpu_policy=GPUExecutionPolicy(max_factor_tile_size=3))
    assert calls == [2]
    receipt = result.metadata["execution_receipt"]
    assert receipt["max_factor_tile_size"] == 3
    assert receipt["effective_max_tile_size"] == 2
    assert receipt["effective_gpu_factor_tile_size"] == 2


def test_calibration_keys_distinguish_caps_and_still_admit_device(monkeypatch):
    from quant_evaluator.runtime import backend_calibration as calibration

    monkeypatch.setattr(calibration, "_source_fingerprint", lambda: "tile-policy-source")
    monkeypatch.setattr(calibration, "_runtime_fingerprint", lambda device: {"device": device})
    monkeypatch.setattr(calibration, "_device_admission",
                        lambda policy: (None, {"device_id": policy.device_ids[0]}))
    monkeypatch.setattr(calibration, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(calibration, "_PROCESS_SOURCE_DRIFTED", False)
    source, label = _source_inputs()
    batch = source.read_tile(0, 3).batch
    policy = calibration.CalibrationPolicy(repetitions=1, warmups=0)
    identities = [calibration.calibration_identity(
        batch, label, metrics=("coverage",), calibration_policy=policy,
        gpu_policy=GPUExecutionPolicy(max_factor_tile_size=cap))
        for cap in (None, 1, 2)]
    assert len({item["key"] for item in identities}) == 3
    assert all(item["device_admission"] is None for item in identities)
    cache = calibration.BoundedCalibrationCache()
    cache.put(identities[0]["key"], {"winner": "cuda_strict"})
    assert cache.get(identities[1]["key"]) is None


@pytest.mark.parametrize("cap,current,expected", [(3, 128, 3), (1, 128, 1), (10, 8, 4), (None, 128, 64)])
def test_oom_retile_cannot_return_a_width_above_policy_cap(cap, current, expected):
    session = DeviceEvaluationSession(GPUExecutionPolicy(max_factor_tile_size=cap))
    assert session.retile_on_oom(current) == expected
    assert session._final_tile == expected
    assert session._oom_retries == 1
