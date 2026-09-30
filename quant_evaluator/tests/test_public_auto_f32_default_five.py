"""Exact F32 default-five auto admission and fail-closed neighboring guards."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

module = import_module("quant_evaluator.runtime.evaluator")
DEFAULT_FIVE = ("rank_ic", "pearson_ic", "ic_ir", "quantile_spread", "coverage")
OPTIONS = dict(metric_parameters={}, context=None, quantile_builder_parameters={},
               portfolio_returns=None, holding_returns=None, trade_eligibility=None,
               calendar_snapshot=None, exposure_panel=None,
               generalization_evidence=None, evaluator=None)


def _route(shape=(2586, 5461, 32), metrics=DEFAULT_FIVE,
           factor_dtype=np.float64, label_dtype=np.float64, **overrides):
    batch = SimpleNamespace(num_times=shape[0], num_assets=shape[1],
                            num_factors=shape[2],
                            values=np.empty(0, dtype=factor_dtype))
    labels = SimpleNamespace(values=np.empty(0, dtype=label_dtype))
    return module._select_public_auto_backend(
        batch, labels, metrics, **(OPTIONS | overrides))


def test_exact_default_five_route_order_and_dedicated_memory_gate(monkeypatch):
    budgets = []
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda policy, minimum: budgets.append(minimum) or None)
    reason = "certified_batch_real_cos_f32_default_five"
    assert _route() == ("cuda_strict", reason)
    assert _route(metrics=tuple(reversed(DEFAULT_FIVE))) == ("cuda_strict", reason)
    assert budgets == [module._AUTO_REAL_COS_F32_DEFAULT_FIVE_MIN_EFFECTIVE_VRAM_BYTES] * 2
    assert budgets[0] == 12 * 1024 ** 3


@pytest.mark.parametrize("metrics", [
    DEFAULT_FIVE[:-1],
    DEFAULT_FIVE + (DEFAULT_FIVE[0],),
    ("rank_ic", "pearson_ic", "ic_ir", "quantile_spread", "coverage", "turnover"),
    ("rank_ic", "pearson_ic", "ic_ir", "quantile_spread"),
])
def test_subsets_duplicates_and_neighboring_sets_stay_cpu(monkeypatch, metrics):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _route(metrics=metrics) == ("cpu", "metric_set_not_certified")


@pytest.mark.parametrize("shape", [
    ((2585, 5461, 32), "shape_outside_certified_range"),
    ((2586, 5460, 32), "shape_outside_certified_range"),
    ((2586, 5461, 31), "shape_outside_certified_range"),
    ((2586, 5461, 33), "shape_outside_certified_range"),
    ((2586, 5461, 24), "metric_not_certified_for_profile"),
    ((2586, 5461, 13), "metric_not_certified_for_profile"),
])
def test_non_f32_exact_profiles_stay_cpu(monkeypatch, shape):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    shape, reason = shape
    assert _route(shape=shape) == ("cpu", reason)


@pytest.mark.parametrize("factor_dtype,label_dtype", [
    (np.float32, np.float64), (np.float64, np.float32),
])
def test_both_inputs_must_be_float64(monkeypatch, factor_dtype, label_dtype):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _route(factor_dtype=factor_dtype, label_dtype=label_dtype) == (
        "cpu", "dtype_outside_certified_range")


@pytest.mark.parametrize("override", [
    {"metric_parameters": {"coverage": {"min_assets": 10}}},
    {"quantile_builder_parameters": {"n_quantiles": 5}},
    {"context": object()}, {"portfolio_returns": object()},
    {"holding_returns": object()}, {"trade_eligibility": object()},
    {"calendar_snapshot": object()}, {"exposure_panel": object()},
    {"generalization_evidence": object()}, {"evaluator": object()},
])
def test_special_inputs_or_parameters_stay_cpu(monkeypatch, override):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _route(**override) == ("cpu", "special_input_or_parameters")


@pytest.mark.parametrize("rejection", [
    "gpu_model_not_certified", "insufficient_cuda_memory",
    "gpu_policy_outside_certified_range", "cuda_unavailable",
])
def test_device_and_memory_rejections_stay_cpu(monkeypatch, rejection):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection",
                        lambda policy, minimum: rejection)
    assert _route() == ("cpu", rejection)


@pytest.mark.parametrize("shape", [
    (2586, 5461, 1), (2586, 5461, 2), (2586, 5461, 5),
    (2586, 5461, 8), (2586, 5461, 12), (2586, 5461, 13),
    (2586, 5461, 24),
])
def test_other_certified_factor_profiles_do_not_inherit_f32_default_five(
        monkeypatch, shape):
    monkeypatch.setattr(module, "_auto_batch_cuda_rejection", lambda *args: None)
    assert _route(shape=shape) == ("cpu", "metric_not_certified_for_profile")


def test_default_five_memory_gate_boundary_and_fraction(monkeypatch):
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    hardware = {"free": 12 * 1024 ** 3}

    class Runtime:
        @staticmethod
        def getDeviceProperties(index):
            return {"name": b"NVIDIA L20"}

        @staticmethod
        def memGetInfo():
            return hardware["free"], 48 * 1024 ** 3

    import sys
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(Device=Device, runtime=Runtime)))
    minimum = module._AUTO_REAL_COS_F32_DEFAULT_FIVE_MIN_EFFECTIVE_VRAM_BYTES
    selector = module._auto_batch_cuda_rejection
    assert minimum == 12 * 1024 ** 3
    full_fraction = GPUExecutionPolicy(max_vram_fraction=1.0)
    hardware["free"] = minimum
    assert selector(full_fraction, minimum) is None
    hardware["free"] = minimum - 1
    assert selector(full_fraction, minimum) == "insufficient_cuda_memory"
    default_free = (minimum * 4 + 2) // 3
    hardware["free"] = default_free
    assert selector(None, minimum) is None
    hardware["free"] = default_free - 1
    assert selector(None, minimum) == "insufficient_cuda_memory"
    hardware["free"] = minimum
    assert selector(GPUExecutionPolicy(max_vram_fraction=0.5), minimum) == (
        "insufficient_cuda_memory")
