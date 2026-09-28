"""Keep public backend benchmark groups aligned with CUDA and registry capabilities."""

import pytest

from quant_evaluator.runtime.gpu_executor import GPUExecutor
from quant_evaluator.scripts import benchmark_public_backend_routes as routes


def test_metric_groups_include_new_cuda_base_metric():
    groups = routes.metric_groups()
    assert "rank_ic_positive_ratio" in groups["base"]
    assert set(groups["base"]) == (
        GPUExecutor.SUPPORTED_METRICS - set(groups["portfolio"]) - set(groups["exposure"])
    )


def test_metric_groups_report_missing_cuda_special_metric(monkeypatch):
    monkeypatch.setattr(
        GPUExecutor, "SUPPORTED_METRICS",
        GPUExecutor.SUPPORTED_METRICS - {"sharpe_ratio"},
    )
    with pytest.raises(RuntimeError, match="missing_cuda=\\['sharpe_ratio'\\]"):
        routes.metric_groups()


def test_metric_groups_report_unregistered_cuda_metric(monkeypatch):
    monkeypatch.setattr(
        GPUExecutor, "SUPPORTED_METRICS",
        GPUExecutor.SUPPORTED_METRICS | {"unregistered_probe"},
    )
    with pytest.raises(RuntimeError, match="unregistered_cuda=\\['unregistered_probe'\\]"):
        routes.metric_groups()
