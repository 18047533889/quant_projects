"""Exact F32 Pearson-chain auto admission, receipt reason, and exclusions."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

module = import_module("quant_evaluator.runtime.evaluator")
PEARSON_CHAIN = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
OPTIONS = dict(metric_parameters={}, context=None, quantile_builder_parameters={},
               portfolio_returns=None, holding_returns=None, trade_eligibility=None,
               calendar_snapshot=None, exposure_panel=None,
               generalization_evidence=None, evaluator=None)


def _inputs(t=2586, n=5461, f=32, factor_dtype=np.float64, label_dtype=np.float64):
    batch = SimpleNamespace(num_times=t, num_assets=n, num_factors=f,
                            values=np.empty(0, dtype=factor_dtype))
    labels = SimpleNamespace(values=np.empty(0, dtype=label_dtype))
    return batch, labels


def _route(batch, labels, metrics=PEARSON_CHAIN, **overrides):
    return module._select_public_auto_backend(
        batch, labels, metrics, **(OPTIONS | overrides))


def test_exact_f32_pearson_chain_receipt_reason_and_vram_gate(monkeypatch):
    budgets = []
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda policy, minimum: budgets.append(minimum) or None)
    batch, labels = _inputs()
    reason = "certified_batch_real_cos_f32_pearson_chain"
    assert _route(batch, labels) == ("cuda_strict", reason)
    assert _route(batch, labels, tuple(reversed(PEARSON_CHAIN))) == (
        "cuda_strict", reason)
    assert budgets == [14 * 1024**3, 14 * 1024**3]


@pytest.mark.parametrize("metrics, reason", [
    (PEARSON_CHAIN[:-1], "metric_set_not_certified"),
    (PEARSON_CHAIN + (PEARSON_CHAIN[0],), "metric_set_not_certified"),
    (("pearson_ic",), "metric_not_certified_for_profile"),
])
def test_pearson_subsets_and_duplicates_stay_cpu(monkeypatch, metrics, reason):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    batch, labels = _inputs()
    assert _route(batch, labels, metrics) == ("cpu", reason)


@pytest.mark.parametrize("shape, reason", [
    ((701, 5314, 2), "metric_not_certified_for_profile"),
    ((2586, 5461, 8), "metric_not_certified_for_profile"),
    ((2586, 5461, 12), "metric_not_certified_for_profile"),
    ((2585, 5461, 32), "shape_outside_certified_range"),
    ((2586, 5460, 32), "shape_outside_certified_range"),
    ((2586, 5461, 31), "shape_outside_certified_range"),
    ((2586, 5461, 24), "metric_not_certified_for_profile"),
    ((2586, 5461, 2), "metric_not_certified_for_profile"),
])
def test_neighboring_profiles_do_not_inherit_pearson_chain(monkeypatch, shape, reason):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    batch, labels = _inputs(*shape)
    assert _route(batch, labels) == ("cpu", reason)


@pytest.mark.parametrize("factor_dtype,label_dtype", [
    (np.float32, np.float64), (np.float64, np.float32),
])
def test_pearson_dtype_guard(monkeypatch, factor_dtype, label_dtype):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    batch, labels = _inputs(factor_dtype=factor_dtype, label_dtype=label_dtype)
    assert _route(batch, labels) == ("cpu", "dtype_outside_certified_range")


@pytest.mark.parametrize("override", [
    {"metric_parameters": {"pearson_ic": {"min_assets": 10}}},
    {"quantile_builder_parameters": {"n_quantiles": 5}},
    {"context": object()}, {"portfolio_returns": object()},
    {"holding_returns": object()}, {"trade_eligibility": object()},
    {"calendar_snapshot": object()}, {"exposure_panel": object()},
    {"generalization_evidence": object()}, {"evaluator": object()},
])
def test_pearson_special_inputs_stay_cpu(monkeypatch, override):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    batch, labels = _inputs()
    assert _route(batch, labels, **override) == ("cpu", "special_input_or_parameters")


@pytest.mark.parametrize("rejection", [
    "gpu_model_not_certified", "insufficient_cuda_memory",
    "gpu_policy_outside_certified_range", "cuda_unavailable",
])
def test_pearson_device_and_budget_rejections_are_preserved(monkeypatch, rejection):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda policy, minimum: rejection)
    batch, labels = _inputs()
    assert _route(batch, labels) == ("cpu", rejection)
