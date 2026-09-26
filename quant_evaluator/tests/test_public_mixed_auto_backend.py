"""Exact whole-request auto route for the real-COS mixed three metrics."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate

module = import_module("quant_evaluator.runtime.evaluator")
METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")


def _select(monkeypatch, shape=(2586, 5461, 2), metrics=METRICS, rejection=None,
            **overrides):
    seen = []
    def check(policy, minimum):
        seen.append(minimum)
        return rejection
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", check)
    batch = SimpleNamespace(
        num_times=shape[0], num_assets=shape[1], num_factors=shape[2],
        values=np.empty(0, dtype=np.float64))
    labels = SimpleNamespace(values=np.empty(0, dtype=np.float64))
    options = dict(
        metric_parameters={}, context=None, quantile_builder_parameters={},
        portfolio_returns=None, holding_returns=None, trade_eligibility=None,
        calendar_snapshot=None, exposure_panel=None,
        generalization_evidence=None, evaluator=None)
    options.update(overrides)
    route = module._select_public_auto_backend(
        batch, labels, metrics, **options)
    return route, seen


def test_exact_shape_and_set_only(monkeypatch):
    route, seen = _select(monkeypatch)
    assert route == ("cuda_strict", "certified_batch_real_cos_mixed_three")
    assert seen == [module._auto_large_min_effective_vram_bytes(
        SimpleNamespace(num_times=2586, num_assets=5461, num_factors=2),
        "rank_ic")]
    assert seen[0] > 11_234_754_560  # full-request observed peak
    assert _select(monkeypatch, metrics=tuple(reversed(METRICS)))[0] == route
    for shape in ((2585, 5461, 2), (2586, 5460, 2),
                  (2586, 5461, 1), (2601, 5461, 2),
                  (701, 5314, 2)):
        assert _select(monkeypatch, shape=shape)[0][0] == "cpu"
    for ids in (METRICS[:-1], METRICS + (METRICS[0],),
                METRICS + ("coverage",)):
        assert _select(monkeypatch, metrics=ids)[0][0] == "cpu"
    assert _select(monkeypatch, metric_parameters={
        "rank_ic": {"min_assets": 25}}) == (
            ("cpu", "special_input_or_parameters"), [])


@pytest.mark.parametrize("rejection", [
    "cuda_unavailable", "gpu_model_not_certified",
    "insufficient_cuda_memory", "gpu_policy_outside_certified_range",
])
def test_hardware_rejection_uses_cpu(monkeypatch, rejection):
    assert _select(monkeypatch, rejection=rejection)[0] == ("cpu", rejection)


def _inputs():
    rng = np.random.default_rng(260927)
    t, n = 32, 60
    x = rng.normal(size=(t, n, 2))
    y = rng.normal(size=(t, n))
    batch = FactorBatch(("f0", "f1"), AxisRef("time", "int", t),
                        AxisRef("asset", "int", n), x)
    labels = LabelBundle(
        "next_ret", y, 1, decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)))
    return batch, labels


def _assert_parity(cpu, auto, factor_ids):
    assert cpu.config_hash == auto.config_hash
    assert auto.metadata["execution_receipt"]["config_hash"] == auto.config_hash
    assert auto.metadata["metric_backends"] == {
        metric: auto.metadata["backend_used"] for metric in METRICS}
    for metric in METRICS:
        left, right = cpu.artifacts[metric], auto.artifacts[metric]
        assert type(left) is type(right)
        assert left.metric_id == right.metric_id
        assert left.producer_version == right.producer_version
        assert left.factor_axis == right.factor_axis
        assert left.provenance == right.provenance
        np.testing.assert_array_equal(
            np.isfinite(left.values), np.isfinite(right.values))
        np.testing.assert_allclose(
            left.values, right.values, rtol=1e-8, atol=1e-10,
            equal_nan=True)
        for factor_id in factor_ids:
            a, b = cpu.get_metric(metric, factor_id), auto.get_metric(metric, factor_id)
            assert (a.metric_id, a.valid, a.observation_count,
                    a.metric_version, a.sample_unit, a.warnings) == (
                    b.metric_id, b.valid, b.observation_count,
                    b.metric_version, b.sample_unit, b.warnings)
            assert a.value == pytest.approx(b.value, rel=1e-8, abs=1e-10)


def test_uncertified_shape_cpu_receipt_and_hash():
    batch, labels = _inputs()
    cpu = evaluate(batch, labels, metrics=METRICS, backend="cpu")
    auto = evaluate(batch, labels, metrics=METRICS, backend="auto")
    assert auto.metadata["backend_used"] == "cpu"
    assert auto.metadata["auto_backend_reason"] == "shape_outside_certified_range"
    assert auto.metadata["backend_strategy"] == "auto"
    assert auto.metadata["execution_receipt"]["receipt_hash"] != (
        cpu.metadata["execution_receipt"]["receipt_hash"])
    _assert_parity(cpu, auto, batch.factor_ids)


def test_unavailable_cuda_cpu_receipt_and_hash(monkeypatch):
    monkeypatch.setattr(module, "_AUTO_REAL_COS_MIXED_THREE_SHAPE", (32, 60, 2))
    monkeypatch.setattr(module, "_auto_public_shape_profile",
                        lambda batch: module._AUTO_LARGE_PROFILE)
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda *args: "cuda_unavailable")
    batch, labels = _inputs()
    cpu = evaluate(batch, labels, metrics=METRICS, backend="cpu")
    auto = evaluate(batch, labels, metrics=METRICS, backend="auto")
    assert auto.metadata["backend_used"] == "cpu"
    assert auto.metadata["auto_backend_reason"] == "cuda_unavailable"
    _assert_parity(cpu, auto, batch.factor_ids)


def test_available_cuda_parity_receipt_and_hash(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    monkeypatch.setattr(module, "_AUTO_REAL_COS_MIXED_THREE_SHAPE", (32, 60, 2))
    monkeypatch.setattr(module, "_auto_public_shape_profile",
                        lambda batch: module._AUTO_LARGE_PROFILE)
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda *args: None)
    batch, labels = _inputs()
    cpu = evaluate(batch, labels, metrics=METRICS, backend="cpu")
    auto = evaluate(batch, labels, metrics=METRICS, backend="auto")
    assert auto.metadata["backend_used"] == "cuda"
    assert auto.metadata["auto_backend_reason"] == (
        "certified_batch_real_cos_mixed_three")
    assert auto.metadata["execution_receipt"]["backend_used"] == "cuda"
    assert auto.metadata["execution_receipt"]["receipt_hash"] != (
        cpu.metadata["execution_receipt"]["receipt_hash"])
    _assert_parity(cpu, auto, batch.factor_ids)
