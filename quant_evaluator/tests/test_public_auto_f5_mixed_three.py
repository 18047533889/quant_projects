"""Exact F5 default mixed-three auto certificate and neighboring guards."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

module = import_module("quant_evaluator.runtime.evaluator")
MIXED_THREE = ("rank_ic", "quantile_spread", "factor_turnover_rate")
OPTIONS = dict(metric_parameters={}, context=None, quantile_builder_parameters={},
               portfolio_returns=None, holding_returns=None, trade_eligibility=None,
               calendar_snapshot=None, exposure_panel=None,
               generalization_evidence=None, evaluator=None)


def _route(t=2586, n=5461, f=5, metrics=MIXED_THREE,
           factor_dtype=np.float64, label_dtype=np.float64, **overrides):
    batch = SimpleNamespace(num_times=t, num_assets=n, num_factors=f,
                            values=np.empty(0, dtype=factor_dtype))
    labels = SimpleNamespace(values=np.empty(0, dtype=label_dtype))
    return module._select_public_auto_backend(
        batch, labels, metrics, **(OPTIONS | overrides))


def test_exact_f5_mixed_three_reason_and_conservative_budget(monkeypatch):
    budgets = []
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda policy, minimum: budgets.append(minimum) or None)
    reason = "certified_batch_real_cos_f5_mixed_three"
    assert _route() == ("cuda_strict", reason)
    assert _route(metrics=tuple(reversed(MIXED_THREE))) == ("cuda_strict", reason)
    assert budgets == [14 * 1024**3, 14 * 1024**3]
    assert int(module._AUTO_CUDA_POLICY_VERSION.rsplit("_v", 1)[1]) >= 23


@pytest.mark.parametrize("shape", [
    (2585, 5461, 5), (2586, 5460, 5), (2586, 5461, 4),
    (2586, 5461, 6), (701, 5314, 5),
])
def test_neighboring_shapes_remain_cpu(monkeypatch, shape):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _route(*shape) == ("cpu", "shape_outside_certified_range")


@pytest.mark.parametrize("metrics, reason", [
    (MIXED_THREE[:-1], "metric_set_not_certified"),
    (MIXED_THREE + (MIXED_THREE[0],), "metric_set_not_certified"),
    (("rank_ic",), "metric_not_certified_for_profile"),
    (("rank_ic", "rank_ic_series", "ic_std", "ic_ir"),
     "metric_not_certified_for_profile"),
])
def test_f5_subsets_and_other_groups_remain_cpu(monkeypatch, metrics, reason):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _route(metrics=metrics) == ("cpu", reason)


@pytest.mark.parametrize("factor_dtype,label_dtype", [
    (np.float32, np.float64), (np.float64, np.float32),
])
def test_f5_requires_float64_on_both_inputs(monkeypatch, factor_dtype, label_dtype):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _route(factor_dtype=factor_dtype, label_dtype=label_dtype) == (
        "cpu", "dtype_outside_certified_range")


@pytest.mark.parametrize("override", [
    {"metric_parameters": {"rank_ic": {"min_assets": 10}}},
    {"quantile_builder_parameters": {"n_quantiles": 5}},
    {"context": object()}, {"portfolio_returns": object()},
    {"holding_returns": object()}, {"trade_eligibility": object()},
    {"calendar_snapshot": object()}, {"exposure_panel": object()},
    {"generalization_evidence": object()}, {"evaluator": object()},
])
def test_f5_special_inputs_remain_cpu(monkeypatch, override):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _route(**override) == ("cpu", "special_input_or_parameters")


@pytest.mark.parametrize("rejection", [
    "gpu_model_not_certified", "insufficient_cuda_memory",
    "gpu_policy_outside_certified_range", "cuda_unavailable",
])
def test_f5_device_and_budget_rejections(monkeypatch, rejection):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda policy, minimum: rejection)
    assert _route() == ("cpu", rejection)


def test_f32_pearson_certificate_is_preserved(monkeypatch):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    pearson = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
    assert _route(f=32, metrics=pearson) == (
        "cuda_strict", "certified_batch_real_cos_f32_pearson_chain")
