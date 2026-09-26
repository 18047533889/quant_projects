"""Bounded public auto-backend policy and complete output parity."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import sys

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate

module = import_module("quant_evaluator.runtime.evaluator")

_CERTIFIED = {
    "daily_quantile_monotonicity_rate",
    "daily_quantile_monotonicity_series",
    "ic_ir", "ic_median", "ic_std",
    "quantile_monotonicity", "quantile_returns_daily",
    "quantile_returns_full", "quantile_spread",
    "rank_ic", "rank_ic_series", "turnover",
}


def _policy(metric_ids=("rank_ic",), **overrides):
    batch = SimpleNamespace(
        num_times=701, num_assets=5314, num_factors=2,
        values=np.empty(0, dtype=np.float64),
    )
    labels = SimpleNamespace(values=np.empty(0, dtype=np.float64))
    options = dict(
        metric_parameters={}, context=None, quantile_builder_parameters={},
        portfolio_returns=None, holding_returns=None, trade_eligibility=None,
        calendar_snapshot=None, exposure_panel=None,
        generalization_evidence=None, evaluator=None,
    )
    options.update(overrides)
    return module._select_public_auto_backend(batch, labels, metric_ids, **options)


def test_auto_certified_cold_and_warm_metric_set(monkeypatch):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert module._AUTO_CUDA_METRICS == _CERTIFIED
    for metric_id in _CERTIFIED:
        assert _policy((metric_id,)) == ("cuda_strict", "certified_single_metric_shape")
    assert _policy(("pearson_ic_ir",)) == ("cpu", "metric_not_certified")
    assert _policy(("factor_turnover_rate",)) == ("cpu", "metric_not_certified")


@pytest.mark.parametrize("metrics, expected", [
    (("rank_ic", "quantile_spread"), "metric_set_not_certified"),
    (("pearson_ic",), "metric_not_certified"),
])
def test_auto_metric_gate(monkeypatch, metrics, expected):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _policy(metrics) == ("cpu", expected)


@pytest.mark.parametrize("field, value", [
    ("metric_parameters", {"rank_ic": {"min_assets": 25}}),
    ("context", object()),
    ("quantile_builder_parameters", {"n_quantiles": 5}),
    ("portfolio_returns", object()),
    ("holding_returns", object()),
    ("trade_eligibility", object()),
    ("calendar_snapshot", object()),
    ("exposure_panel", object()),
    ("generalization_evidence", object()),
    ("evaluator", object()),
])
def test_auto_special_inputs_stay_cpu(monkeypatch, field, value):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _policy(**{field: value}) == ("cpu", "special_input_or_parameters")


_BATCHES = {
    "rank_chain": ("rank_ic", "rank_ic_series", "ic_std", "ic_ir"),
    "quantile_chain": ("quantile_returns_full", "quantile_returns_daily",
                       "quantile_spread", "quantile_monotonicity",
                       "daily_quantile_monotonicity_rate"),
    "mixed_core": ("rank_ic", "rank_ic_series", "ic_ir",
                   "quantile_spread", "turnover", "factor_turnover_rate"),
}


def test_auto_exact_batch_sets_and_single_metric_policy(monkeypatch):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda policy: None)
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert set(module._AUTO_CUDA_BATCHES.values()) == set(_BATCHES)
    for name, metrics in _BATCHES.items():
        assert _policy(metrics) == ("cuda_strict", f"certified_batch_{name}")
        assert _policy(tuple(reversed(metrics))) == (
            "cuda_strict", f"certified_batch_{name}")
        assert _policy(metrics[:-1]) == ("cpu", "metric_set_not_certified")
        assert _policy(metrics + (metrics[0],)) == (
            "cpu", "metric_set_not_certified")
    # The portfolio batch needs a typed trajectory and is deliberately closed.
    portfolio = ("rank_ic", "quantile_spread", "sharpe_ratio",
                 "sortino_ratio", "max_drawdown", "calmar_ratio")
    assert _policy(portfolio) == ("cpu", "metric_set_not_certified")
    assert _policy(("rank_ic",)) == ("cuda_strict", "certified_single_metric_shape")


def test_auto_batch_hardware_and_memory_gate(monkeypatch):
    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    hardware = {"name": b"NVIDIA L20", "free": 40 * 1024 ** 3}
    runtime = SimpleNamespace(
        getDeviceProperties=lambda index: {"name": hardware["name"]},
        memGetInfo=lambda: (hardware["free"], 48 * 1024 ** 3),
    )
    fake_cp = SimpleNamespace(cuda=SimpleNamespace(Device=Device, runtime=runtime))
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    assert module._auto_batch_cuda_rejection(None) is None
    hardware["name"] = b"NVIDIA L20S"
    assert module._auto_batch_cuda_rejection(None) == "gpu_model_not_certified"
    hardware["name"] = b"NVIDIA L20"
    hardware["free"] = 11 * 1024 ** 3
    assert module._auto_batch_cuda_rejection(None) == "insufficient_cuda_memory"
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    hardware["free"] = 20 * 1024 ** 3
    assert module._auto_batch_cuda_rejection(
        GPUExecutionPolicy(max_vram_fraction=0.4)) == "insufficient_cuda_memory"
    hardware["free"] = 40 * 1024 ** 3
    assert module._auto_batch_cuda_rejection(
        GPUExecutionPolicy(max_vram_fraction=0.4)) is None
    hardware["free"] = 10 * 1024 ** 3
    assert module._auto_batch_cuda_rejection(
        None, module._AUTO_SINGLE_MIN_EFFECTIVE_VRAM_BYTES
    ) == "insufficient_cuda_memory"
    hardware["free"] = 12 * 1024 ** 3
    assert module._auto_batch_cuda_rejection(
        None, module._AUTO_SINGLE_MIN_EFFECTIVE_VRAM_BYTES
    ) is None
    assert module._auto_batch_cuda_rejection(None) == "insufficient_cuda_memory"


def test_auto_batch_hardware_rejection_uses_cpu(monkeypatch):
    monkeypatch.setattr(
        module, "_auto_batch_cuda_rejection",
        lambda *args: "gpu_model_not_certified",
    )
    assert _policy(_BATCHES["rank_chain"]) == (
        "cpu", "gpu_model_not_certified")
    assert _policy(("rank_ic",)) == ("cpu", "gpu_model_not_certified")
    monkeypatch.setattr(
        module, "_auto_batch_cuda_rejection",
        lambda *args: "insufficient_cuda_memory",
    )
    assert _policy(("rank_ic",)) == ("cpu", "insufficient_cuda_memory")


def test_auto_shape_dtype_and_device_gates(monkeypatch):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _policy() == ("cuda_strict", "certified_single_metric_shape")
    batch = SimpleNamespace(num_times=599, num_assets=5314, num_factors=2,
                            values=np.empty(0, dtype=np.float64))
    labels = SimpleNamespace(values=np.empty(0, dtype=np.float64))
    kwargs = dict(metric_parameters={}, context=None, quantile_builder_parameters={},
                  portfolio_returns=None, holding_returns=None, trade_eligibility=None,
                  calendar_snapshot=None, exposure_panel=None,
                  generalization_evidence=None, evaluator=None)
    select = module._select_public_auto_backend
    assert select(batch, labels, ("rank_ic",), **kwargs) == (
        "cpu", "shape_outside_certified_range")
    batch.num_times = 701
    batch.num_assets = 4999
    assert select(batch, labels, ("rank_ic",), **kwargs)[1] == "shape_outside_certified_range"
    batch.num_assets = 5314
    batch.num_factors = 1
    assert select(batch, labels, ("rank_ic",), **kwargs)[1] == "shape_outside_certified_range"
    batch.num_factors = 2
    batch.values = np.empty(0, dtype=np.float32)
    assert select(batch, labels, ("rank_ic",), **kwargs)[1] == "dtype_outside_certified_range"
    batch.values = np.empty(0, dtype=np.float64)
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda *args: "cuda_unavailable")
    assert select(batch, labels, ("rank_ic",), **kwargs) == ("cpu", "cuda_unavailable")



def _region_policy(monkeypatch, t, n, f, metric_id="rank_ic", rejection=None):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: rejection)
    batch = SimpleNamespace(num_times=t, num_assets=n, num_factors=f,
                            values=np.empty(0, dtype=np.float64))
    labels = SimpleNamespace(values=np.empty(0, dtype=np.float64))
    options = dict(metric_parameters={}, context=None, quantile_builder_parameters={},
                   portfolio_returns=None, holding_returns=None, trade_eligibility=None,
                   calendar_snapshot=None, exposure_panel=None,
                   generalization_evidence=None, evaluator=None)
    return module._select_public_auto_backend(batch, labels, (metric_id,), **options)


@pytest.mark.parametrize("t,n,f,route", [
    (1000, 5000, 2, "cuda_strict"),
    (1800, 5300, 2, "cuda_strict"),
    (2586, 5461, 2, "cuda_strict"),
    (2600, 5500, 2, "cuda_strict"),
    (2601, 5461, 2, "cuda_strict"),
    (3200, 6000, 2, "cuda_strict"),
    (2400, 5461, 1, "cuda_strict"),
    (2000, 5000, 1, "cuda_strict"),
    (3200, 6000, 1, "cuda_strict"),
    (999, 5000, 2, "cpu"),
    (1000, 5000, 1, "cpu"),
    (1999, 5461, 1, "cpu"),
    (3201, 5461, 2, "cpu"),
    (2586, 4999, 2, "cpu"),
    (2586, 6001, 2, "cpu"),
    (2586, 5461, 3, "cpu"),
])
def test_large_region_shape_boundaries(monkeypatch, t, n, f, route):
    selected, reason = _region_policy(monkeypatch, t, n, f)
    assert selected == route
    if route == "cuda_strict":
        assert reason == ("certified_single_metric_real_cos_region"
                          if t <= 2600 and n <= 5500
                          else "bounded_extrapolation_real_cos_headroom")
    else:
        assert reason == "shape_outside_certified_range"


def test_large_region_metric_and_vram_gates(monkeypatch):
    assert _region_policy(monkeypatch, 2586, 5461, 2, "quantile_spread") == (
        "cuda_strict", "certified_single_metric_real_cos_region")
    assert _region_policy(monkeypatch, 2586, 5461, 2, "factor_turnover_rate") == (
        "cpu", "metric_not_certified")
    assert _region_policy(monkeypatch, 2586, 5461, 2, "ic_ir") == (
        "cpu", "metric_not_certified_for_profile")
    batch = SimpleNamespace(num_times=2586, num_assets=5461, num_factors=2)
    rank_budget = module._auto_large_min_effective_vram_bytes(batch, "rank_ic")
    assert rank_budget == 768 * 2586 * 5461 * 2
    assert rank_budget > 16 * 1024 ** 3
    assert module._auto_large_min_effective_vram_bytes(batch, "quantile_spread") == 8 * 1024 ** 3
    batch.num_times, batch.num_factors = 2400, 1
    assert module._auto_large_min_effective_vram_bytes(batch, "rank_ic") == 1024 * 2400 * 5461
    assert _region_policy(monkeypatch, 2586, 5461, 2,
                          rejection="insufficient_cuda_memory") == (
        "cpu", "insufficient_cuda_memory")


def _inputs(t, n):
    rng = np.random.default_rng(20260926)
    factors = rng.normal(size=(t, n, 2))
    labels = 0.002 * factors[:, :, 0] + rng.normal(size=(t, n))
    batch = FactorBatch(("f0", "f1"), AxisRef("time", "int", t),
                        AxisRef("asset", "int", n), factors)
    label = LabelBundle("next_ret", labels, 1, decision_time=tuple(range(t)),
                        label_start_time=tuple(range(1, t + 1)),
                        label_end_time=tuple(range(2, t + 2)))
    return batch, label


def test_auto_small_public_request_equals_cpu(monkeypatch):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    batch, labels = _inputs(30, 60)
    cpu = evaluate(batch, labels, metrics=("rank_ic",), backend="cpu")
    auto = evaluate(batch, labels, metrics=("rank_ic",), backend="auto")
    default = evaluate(batch, labels, metrics=("rank_ic",))
    assert default.metadata["backend_requested"] == "default"
    assert default.metadata["backend_strategy"] == "auto"
    assert default.metadata["backend_used"] == "cpu"
    assert default.metadata["auto_backend_reason"] == "shape_outside_certified_range"
    assert default.metadata["auto_backend_policy"] == module._AUTO_CUDA_POLICY_VERSION
    assert cpu.metadata["backend_requested"] == "cpu"
    assert cpu.metadata["backend_strategy"] == "explicit"
    assert default.metadata["execution_receipt"]["config_hash"] == default.config_hash
    assert default.metadata["execution_receipt"]["receipt_hash"] != auto.metadata["execution_receipt"]["receipt_hash"]
    assert default.config_hash == auto.config_hash == cpu.config_hash
    assert auto.metadata["backend_requested"] == "auto"
    assert auto.metadata["backend_used"] == "cpu"
    assert auto.metadata["auto_backend_reason"] == "shape_outside_certified_range"
    assert auto.metadata["metric_backends"] == {"rank_ic": "cpu"}
    assert auto.config_hash == cpu.config_hash
    np.testing.assert_equal(auto.artifacts["rank_ic"].values,
                            cpu.artifacts["rank_ic"].values)
    np.testing.assert_equal(default.artifacts["rank_ic"].values,
                            cpu.artifacts["rank_ic"].values)
    for factor_id in batch.factor_ids:
        assert auto.get_metric("rank_ic", factor_id) == cpu.get_metric("rank_ic", factor_id)


def test_default_public_request_records_cuda_unavailable_cpu_fallback(monkeypatch):
    monkeypatch.setattr(
        module, "_select_public_auto_backend",
        lambda *args, **kwargs: ("cpu", "cuda_unavailable"),
    )
    batch, labels = _inputs(30, 60)
    default = evaluate(batch, labels, metrics=("rank_ic",))
    cpu = evaluate(batch, labels, metrics=("rank_ic",), backend="cpu")
    assert default.metadata["backend_requested"] == "default"
    assert default.metadata["backend_used"] == "cpu"
    assert default.metadata["auto_backend_reason"] == "cuda_unavailable"
    assert default.metadata["execution_receipt"]["backend_used"] == "cpu"
    np.testing.assert_equal(default.artifacts["rank_ic"].values, cpu.artifacts["rank_ic"].values)
    assert default.config_hash == cpu.config_hash


def test_default_region_vram_fallback_preserves_explicit_cpu(monkeypatch):
    monkeypatch.setattr(module, "_auto_public_shape_profile",
                        lambda batch: module._AUTO_LARGE_PROFILE)
    captured = []
    def reject(policy, minimum):
        captured.append(minimum)
        return "insufficient_cuda_memory"
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", reject)
    batch, labels = _inputs(30, 60)
    default = evaluate(batch, labels, metrics=("rank_ic",))
    cpu = evaluate(batch, labels, metrics=("rank_ic",), backend="cpu")
    assert captured == [module._auto_large_min_effective_vram_bytes(batch, "rank_ic")]
    assert default.metadata["auto_backend_profile"] == module._AUTO_LARGE_PROFILE
    assert default.metadata["auto_backend_reason"] == "insufficient_cuda_memory"
    assert default.metadata["backend_used"] == "cpu"
    assert cpu.metadata["backend_strategy"] == "explicit"
    assert default.config_hash == cpu.config_hash

def test_auto_alias_uses_canonical_policy_and_preserves_requested_name(monkeypatch):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert module._resolve_alias("ic.rank.mean") == "rank_ic"
    assert _policy((module._resolve_alias("ic.rank.mean"),)) == (
        "cuda_strict", "certified_single_metric_shape")
    batch, labels = _inputs(30, 60)
    out = evaluate(batch, labels, metrics=("ic.rank.mean",), backend="auto")
    assert out.metadata["auto_backend_reason"] == "shape_outside_certified_range"
    assert out.metadata["metric_backends"] == {"ic.rank.mean": "cpu"}
    assert "ic.rank.mean" in out.artifacts


def test_auto_mixed_public_request_uses_one_cpu_plan(monkeypatch):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    batch, labels = _inputs(30, 60)
    out = evaluate(batch, labels, metrics=("rank_ic", "coverage"), backend="auto")
    assert out.metadata["backend_used"] == "cpu"
    assert out.metadata["auto_backend_reason"] == "metric_set_not_certified"
    assert out.metadata["metric_backends"] == {
        "rank_ic": "cpu", "coverage": "cpu"}


def test_auto_rank_batch_large_public_request_matches_cpu():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if module._auto_batch_cuda_rejection(None) is not None:
        pytest.skip("L20 or effective VRAM gate unavailable")
    batch, labels = _inputs(701, 5314)
    cpu = evaluate(batch, labels, metrics=_BATCHES["rank_chain"], backend="cpu")
    auto = evaluate(batch, labels, metrics=_BATCHES["rank_chain"], backend="auto")
    assert auto.metadata["backend_requested"] == "auto"
    assert auto.metadata["backend_used"] == "cuda"
    assert auto.metadata["auto_backend_reason"] == "certified_batch_rank_chain"
    assert auto.metadata["metric_backends"] == {
        metric_id: "cuda" for metric_id in _BATCHES["rank_chain"]}
    assert auto.config_hash == cpu.config_hash
    for metric_id in _BATCHES["rank_chain"]:
        np.testing.assert_allclose(
            auto.artifacts[metric_id].values, cpu.artifacts[metric_id].values,
            rtol=1e-8, atol=1e-10, equal_nan=True,
        )
        for factor_id in batch.factor_ids:
            a = auto.get_metric(metric_id, factor_id)
            c = cpu.get_metric(metric_id, factor_id)
            if a is None or c is None:
                assert a is c
                continue
            assert a.observation_count == c.observation_count
            assert a.valid == c.valid


def test_auto_large_public_request_matches_cpu_and_cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if module._auto_batch_cuda_rejection(
            None, module._AUTO_SINGLE_MIN_EFFECTIVE_VRAM_BYTES) is not None:
        pytest.skip("L20 or effective VRAM gate unavailable")
    batch, labels = _inputs(701, 5314)
    cpu = evaluate(batch, labels, metrics=("rank_ic",), backend="cpu")
    cuda = evaluate(batch, labels, metrics=("rank_ic",), backend="cuda")
    auto = evaluate(batch, labels, metrics=("rank_ic",), backend="auto")
    default = evaluate(batch, labels, metrics=("rank_ic",))
    assert default.metadata["backend_requested"] == "default"
    assert default.metadata["backend_strategy"] == "auto"
    assert default.metadata["backend_used"] == "cuda"
    assert default.metadata["auto_backend_reason"] == "certified_single_metric_shape"
    assert default.config_hash == auto.config_hash == cpu.config_hash == cuda.config_hash
    assert auto.metadata["backend_requested"] == "auto"
    assert auto.metadata["backend_used"] == "cuda"
    assert auto.metadata["auto_backend_policy"] == module._AUTO_CUDA_POLICY_VERSION
    assert auto.metadata["auto_backend_reason"] == "certified_single_metric_shape"
    assert auto.metadata["metric_backends"] == {"rank_ic": "cuda"}
    assert auto.config_hash == cpu.config_hash == cuda.config_hash
    for factor_id in batch.factor_ids:
        a = auto.get_metric("rank_ic", factor_id)
        c = cpu.get_metric("rank_ic", factor_id)
        g = cuda.get_metric("rank_ic", factor_id)
        assert a.observation_count == c.observation_count == g.observation_count
        assert a.valid == c.valid == g.valid
        assert a.value == pytest.approx(c.value, rel=1e-8, abs=1e-10)
        assert a.value == pytest.approx(g.value, rel=1e-8, abs=1e-10)
    np.testing.assert_allclose(auto.artifacts["rank_ic"].values,
                               cpu.artifacts["rank_ic"].values,
                               rtol=1e-8, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(default.artifacts["rank_ic"].values,
                               cuda.artifacts["rank_ic"].values,
                               rtol=1e-8, atol=1e-10, equal_nan=True)
