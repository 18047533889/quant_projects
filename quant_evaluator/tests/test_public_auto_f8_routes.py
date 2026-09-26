"""Exact-shape F8 public auto-route contract."""
from importlib import import_module
from types import SimpleNamespace
import sys

import numpy as np
import pytest

module = import_module("quant_evaluator.runtime.evaluator")
F8_SHAPE = (2586, 5461, 8)
F8_METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")


def _select(monkeypatch, shape, metrics, *, dtype=np.float64, policy=None,
            parameters=None, rejection=None, mock_cuda=True):
    batch = SimpleNamespace(
        num_times=shape[0], num_assets=shape[1], num_factors=shape[2],
        values=np.empty(0, dtype=dtype),
    )
    labels = SimpleNamespace(values=np.empty(0, dtype=dtype))
    options = dict(
        metric_parameters=parameters or {}, context=None,
        quantile_builder_parameters={}, portfolio_returns=None,
        holding_returns=None, trade_eligibility=None, calendar_snapshot=None,
        exposure_panel=None, generalization_evidence=None, evaluator=None,
        gpu_policy=policy,
    )
    if mock_cuda is True:
        monkeypatch.setattr(
            module, "_auto_batch_cuda_rejection", lambda *args: rejection,
        )
    return module._select_public_auto_backend(batch, labels, metrics, **options)


@pytest.mark.parametrize("metric,minimum", [
    ("rank_ic", 13_999_136_256),
    ("rank_ic_series", 13_999_136_256),
    ("ic_ir", 13_999_136_256),
    ("quantile_spread", 8 * 1024 ** 3),
    ("factor_turnover_rate", 8 * 1024 ** 3),
])
def test_f8_exact_single_metric_route_and_memory_threshold(monkeypatch, metric, minimum):
    seen = []
    monkeypatch.setattr(
        module, "_auto_batch_cuda_rejection",
        lambda policy, threshold: seen.append(threshold) or None,
    )
    assert _select(monkeypatch, F8_SHAPE, (metric,), mock_cuda=None) == (
        "cuda_strict", "certified_single_metric_real_cos_f8",
    )
    assert seen == [minimum]


@pytest.mark.parametrize("shape", [
    (2585, 5461, 8), (2586, 5460, 8), (2586, 5461, 7),
    (2586, 5461, 9),
])
def test_f8_route_does_not_extrapolate_shape(monkeypatch, shape):
    assert _select(monkeypatch, shape, ("factor_turnover_rate",)) == (
        "cpu", "shape_outside_certified_range",
    )


def test_f8_route_keeps_other_metrics_and_partial_sets_on_cpu(monkeypatch):
    assert _select(monkeypatch, F8_SHAPE, ("turnover",)) == (
        "cpu", "metric_not_certified_for_profile",
    )
    assert _select(monkeypatch, F8_SHAPE, ("rank_ic", "quantile_spread")) == (
        "cpu", "metric_set_not_certified",
    )
    seen = []
    monkeypatch.setattr(
        module, "_auto_batch_cuda_rejection",
        lambda policy, threshold: seen.append(threshold) or None,
    )
    assert _select(monkeypatch, F8_SHAPE, F8_METRICS, mock_cuda=None) == (
        "cuda_strict", "certified_batch_real_cos_f8_mixed_three",
    )
    assert seen == [14 * 1024 ** 3]


def test_f8_route_retains_default_input_and_dtype_gates(monkeypatch):
    assert _select(monkeypatch, F8_SHAPE, ("rank_ic",), dtype=np.float32) == (
        "cpu", "dtype_outside_certified_range",
    )
    assert _select(
        monkeypatch, F8_SHAPE, ("rank_ic",),
        parameters={"rank_ic": {"min_assets": 25}},
    ) == ("cpu", "special_input_or_parameters")


def test_f12_exact_single_metric_routes(monkeypatch):
    shape = (2586, 5461, 12)
    seen = []
    monkeypatch.setattr(
        module, "_auto_batch_cuda_rejection",
        lambda policy, threshold: seen.append(threshold) or None,
    )
    assert _select(monkeypatch, shape, ("rank_ic",), mock_cuda=None) == (
        "cuda_strict", "certified_single_metric_real_cos_f12",
    )
    assert seen == [14 * 1024 ** 3]
    seen.clear()
    assert _select(monkeypatch, shape, ("quantile_spread",), mock_cuda=None) == (
        "cuda_strict", "certified_single_metric_real_cos_f12",
    )
    assert seen == [8 * 1024 ** 3]
    seen.clear()
    assert _select(monkeypatch, shape, ("factor_turnover_rate",), mock_cuda=None) == (
        "cuda_strict", "certified_single_metric_real_cos_f12",
    )
    assert seen == [8 * 1024 ** 3]
    assert _select(monkeypatch, shape, F8_METRICS) == (
        "cpu", "metric_not_certified_for_profile",
    )
    for metric in ("rank_ic_series", "ic_ir"):
        assert _select(monkeypatch, shape, (metric,)) == (
            "cpu", "metric_not_certified_for_profile",
        )
    assert _select(monkeypatch, (2586, 5461, 11), ("rank_ic",)) == (
        "cpu", "shape_outside_certified_range",
    )


def test_f8_route_accounts_for_policy_fraction_and_keeps_f2_f1_routes(monkeypatch):
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy

    class Device:
        def __init__(self, index):
            self.index = index
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None

    hardware = {"free": 26 * 1024 ** 3}
    runtime = SimpleNamespace(
        getDeviceProperties=lambda index: {"name": b"NVIDIA L20"},
        memGetInfo=lambda: (hardware["free"], 48 * 1024 ** 3),
    )
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(Device=Device, runtime=runtime),
    ))
    half = GPUExecutionPolicy(max_vram_fraction=0.5)
    assert _select(monkeypatch, F8_SHAPE, ("rank_ic",), policy=half,
                   mock_cuda=False) == ("cpu", "insufficient_cuda_memory")
    assert _select(monkeypatch, F8_SHAPE, F8_METRICS, policy=half,
                   mock_cuda=False) == ("cpu", "insufficient_cuda_memory")
    hardware["free"] = 28 * 1024 ** 3
    assert _select(monkeypatch, F8_SHAPE, F8_METRICS, policy=half,
                   mock_cuda=False) == (
        "cuda_strict", "certified_batch_real_cos_f8_mixed_three",
    )
    assert _select(monkeypatch, F8_SHAPE, ("rank_ic",), policy=half,
                   mock_cuda=False) == (
        "cuda_strict", "certified_single_metric_real_cos_f8",
    )
    assert _select(monkeypatch, (2586, 5461, 12), ("rank_ic",),
                   policy=half, mock_cuda=False) == (
        "cuda_strict", "certified_single_metric_real_cos_f12",
    )
    assert _select(monkeypatch, (2586, 5461, 2), ("rank_ic",)) == (
        "cuda_strict", "certified_single_metric_real_cos_region",
    )
    assert _select(monkeypatch, (2400, 5461, 1), ("quantile_spread",)) == (
        "cuda_strict", "certified_single_metric_real_cos_region",
    )
    assert _select(monkeypatch, (2586, 5461, 2), ("factor_turnover_rate",)) == (
        "cpu", "metric_not_certified",
    )
