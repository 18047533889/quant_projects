"""Public-facade cache-aware routing for the certified F32 coverage route."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.contracts._array_hash_cache import ArrayHashStateCache
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.runtime import measured_auto_registry as registry
from quant_evaluator.api.requests import EvaluationRequest


@pytest.fixture(autouse=True)
def clean_measured_registry():
    registry.clear_registry_for_tests()
    yield
    registry.clear_registry_for_tests()


def _readonly(values):
    source = np.asarray(values)
    return np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)


def _inputs(shape=(5, 7, 32)):
    times, assets, factors = shape
    time_axis = AxisRef("time", "int64", times, np.arange(times, dtype=np.int64))
    asset_axis = AxisRef("asset", "str", assets,
                         np.asarray([f"a{i}" for i in range(assets)]))
    source = np.arange(times * assets * factors, dtype=np.float64).reshape(shape)
    values = _readonly(np.sin(source / 11.0))
    source_labels = np.arange(times * assets, dtype=np.float64).reshape(times, assets)
    label_values = _readonly(np.cos(source_labels / 7.0))
    batch = FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(factors)),
        time_axis=time_axis, asset_axis=asset_axis, values=values,
    )
    labels = LabelBundle(
        target_id="forward_1d", values=label_values, horizon=1,
        decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=asset_axis,
    )
    return batch, labels


def _patch_small_profile(monkeypatch, shape):
    evaluator = import_module("quant_evaluator.runtime.evaluator")
    auto_policy = import_module("quant_evaluator.runtime.auto_cuda_policy")
    monkeypatch.setattr(evaluator, "_AUTO_REAL_COS_F32_SHAPE", shape)
    monkeypatch.setattr(auto_policy, "_AUTO_REAL_COS_F32_SHAPE", shape)
    return evaluator


def _install_caches(monkeypatch):
    from quant_evaluator.contracts import array_identity
    from quant_evaluator.contracts import _hashutil as hashutil

    raw_cache = ArrayHashStateCache(min_nbytes=0, capacity=32)
    json_cache = ArrayHashStateCache(min_nbytes=0, capacity=32)
    monkeypatch.setattr(array_identity, "_RAW_ARRAY_HASH_STATE_CACHE", raw_cache)
    monkeypatch.setattr(hashutil, "_ARRAY_HASH_STATE_CACHE", json_cache)
    return raw_cache, json_cache


def test_public_facade_cold_identity_cache_does_not_block_resource_route(monkeypatch):
    evaluator = _patch_small_profile(monkeypatch, (5, 7, 32))
    _install_caches(monkeypatch)
    gate_calls = []
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *args: gate_calls.append(args) or None)
    batch, labels = _inputs()

    assert evaluate(batch, labels, metrics=("coverage",), _prepare_only=True,
                    _include_auto_route_reason=True) == (
        "cuda_strict", "certified_single_metric_real_cos_f32_coverage")
    assert gate_calls


def test_original_f32_coverage_shape_cold_still_obeys_resource_rejection(monkeypatch):
    evaluator = import_module("quant_evaluator.runtime.evaluator")
    gate_calls = []
    monkeypatch.setattr(
        evaluator, "_auto_batch_cuda_rejection",
        lambda *args: gate_calls.append(args) or "insufficient_cuda_memory",
    )
    monkeypatch.setattr(evaluator, "FactorBatch", SimpleNamespace)
    monkeypatch.setattr(evaluator, "LabelBundle", SimpleNamespace)
    batch = SimpleNamespace(
        num_times=2586, num_assets=5461, num_factors=32,
        values=np.empty((1,), dtype=np.float64), validity=None,
        factor_ids=("f0",),
    )
    labels = SimpleNamespace(values=np.empty((1,), dtype=np.float64),
                              content_hash="precomputed-label-hash")
    assert evaluator._select_public_auto_backend(
        batch, labels, ("coverage",), metric_parameters={}, context=None,
        quantile_builder_parameters={}, portfolio_returns=None, holding_returns=None,
        trade_eligibility=None, calendar_snapshot=None, exposure_panel=None,
        generalization_evidence=None, evaluator=None,
    ) == ("cpu", "insufficient_cuda_memory")
    assert gate_calls


def test_public_facade_cold_and_warm_auto_cuda_match_cpu_identity(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device available")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    evaluator = _patch_small_profile(monkeypatch, (5, 7, 32))
    _install_caches(monkeypatch)
    gate_calls = []
    real_gate = evaluator._auto_batch_cuda_rejection

    def checked_gate(*args):
        gate_calls.append(args)
        return real_gate(*args)

    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection", checked_gate)
    batch, labels = _inputs()

    rejection = real_gate(
        None, evaluator._AUTO_REAL_COS_F32_COVERAGE_MIN_EFFECTIVE_VRAM_BYTES)
    if rejection is not None:
        pytest.skip(f"certified CUDA resource gate unavailable: {rejection}")
    cold = evaluate(batch, labels, metrics=("coverage",))
    assert cold.metadata["backend_used"] == "cuda"
    assert cold.metadata["auto_backend_reason"] == (
        "certified_single_metric_real_cos_f32_coverage")
    assert gate_calls
    reference = evaluate(batch, labels, metrics=("coverage",), backend="cpu")
    warm = evaluate(batch, labels, metrics=("coverage",))
    assert gate_calls
    assert warm.metadata["backend_used"] == "cuda"
    assert warm.metadata["auto_backend_reason"] == (
        "certified_single_metric_real_cos_f32_coverage")
    for result in (cold, warm):
        assert result.config_hash == reference.config_hash
        assert result.metadata["provenance"] == reference.metadata["provenance"]
        assert result.grouped_metrics == reference.grouped_metrics
        assert result.factor_ids == reference.factor_ids


def test_evaluation_request_fields_are_included_in_cache_probe(monkeypatch):
    evaluator = _patch_small_profile(monkeypatch, (5, 7, 32))
    _install_caches(monkeypatch)
    gate_calls = []
    monkeypatch.setattr(
        evaluator, "_auto_batch_cuda_rejection",
        lambda *args: gate_calls.append(args) or None,
    )
    batch, labels = _inputs()
    request = EvaluationRequest(
        batch_or_factor_ids=batch, label_bundle=labels, metric_ids=("coverage",))

    cold = evaluate(request, _prepare_only=True, _include_auto_route_reason=True)
    assert cold == (
        "cuda_strict", "certified_single_metric_real_cos_f32_coverage")
    assert gate_calls

    # Request fields remain in the shared exact config hash; cache residency
    # no longer decides the route.
    evaluate(request, backend="cpu")
    warm = evaluate(request, _prepare_only=True, _include_auto_route_reason=True)
    assert warm == (
        "cuda_strict", "certified_single_metric_real_cos_f32_coverage")
    assert gate_calls
