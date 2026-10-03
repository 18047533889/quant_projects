"""Forwarding contracts for evaluate_many's shared quantile builder options."""

from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.api import evaluate_many as api
from quant_evaluator.contracts.errors import InvalidContractError, UnsupportedMetricError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _inputs():
    t, n, f = 6, 12, 2
    times = np.arange(t, dtype=np.int64)
    assets = AxisRef("asset", "str", n,
                     np.asarray([f"S{i:02d}" for i in range(n)]))
    factors = FactorBatch(
        ("f0", "f1"), AxisRef("time", "int64", t, times), assets,
        np.arange(t * n * f, dtype=np.float64).reshape(t, n, f),
    )
    labels = tuple(LabelBundle(
        f"h{horizon}", np.arange(t * n, dtype=np.float64).reshape(t, n), horizon,
        decision_time=tuple(times), observation_time=tuple(times),
        label_start_time=tuple(times), label_end_time=tuple(times + horizon),
        asset_axis=assets,
    ) for horizon in (1, 2))
    return factors, labels


def test_cpu_fallback_forwards_global_builder_parameters(monkeypatch):
    factors, labels = _inputs()
    builder = {"n_quantiles": 3, "min_assets": 2}
    calls = []

    def fake_evaluate(_factors, label, **kwargs):
        calls.append((label.target_id, kwargs))
        if kwargs.get("_prepare_only"):
            return "cpu", "test_cpu_route"
        return label.target_id

    monkeypatch.setattr(api, "evaluate", fake_evaluate)
    result = api.evaluate_many(
        factors, labels, metrics=("rank_ic_series",), backend="auto",
        quantile_builder_parameters=builder,
    )

    assert result == {label.target_id: label.target_id for label in labels}
    assert [target for target, _ in calls] == ["h1", "h1", "h2"]
    assert all(kwargs["quantile_builder_parameters"] is builder
               for _, kwargs in calls)
    assert all(kwargs["metric_parameters"] is None for _, kwargs in calls)


def test_shared_cuda_executor_receives_isolated_global_builder_and_metric_options(monkeypatch):
    factors, labels = _inputs()
    alias = "ic.rank.daily"
    builder = {"n_quantiles": 3, "min_assets": 2}
    metric_parameters = {
        alias: {"min_periods": 4},
        "quantile_curvature": {"n_quantiles": 5},
    }
    evaluate_calls = []
    executor_instances = []
    executor_batches = []

    def fake_evaluate(_factors, label, **kwargs):
        evaluate_calls.append((label.target_id, kwargs))
        if kwargs.get("_prepare_only"):
            return "cuda_strict", "test_cuda_route"
        gpu_bundle = kwargs["_gpu_result_override"]
        return SimpleNamespace(target_id=label.target_id,
                               series_metrics=gpu_bundle.series_metrics,
                               kwargs=kwargs)

    class FakeSession:
        def __init__(self, _policy):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.closed = True

    class FakeExecutor:
        def __init__(self, session):
            self.session = session
            executor_instances.append(self)

        def run_tiled_many(self, _factors, selected_labels, metrics):
            executor_batches.append((tuple(lb.target_id for lb in selected_labels), metrics))
            bundle = SimpleNamespace(
                scalar_metrics={"quantile_curvature": np.zeros(2)},
                series_metrics={"rank_ic_series": np.zeros((6, 2))},
                vector_metrics={}, observation_counts={},
            )
            return [bundle for _ in selected_labels]

    monkeypatch.setattr(api, "evaluate", fake_evaluate)
    monkeypatch.setattr(api, "diagnose_all_factors", lambda _factors: object())
    from quant_evaluator.runtime import device_session, gpu_executor
    monkeypatch.setattr(device_session, "DeviceEvaluationSession", FakeSession)
    monkeypatch.setattr(gpu_executor, "GPUExecutor", FakeExecutor)

    result = api.evaluate_many(
        factors, labels, metrics=(alias, "quantile_curvature"), backend="cuda_strict",
        metric_parameters=metric_parameters,
        quantile_builder_parameters=builder,
    )

    assert set(result) == {"h1", "h2"}
    assert executor_batches == [(("h1", "h2"),
                                 ("rank_ic_series", "quantile_curvature"))]
    executor = executor_instances[0]
    assert executor.quantile_builder_parameters == builder
    assert executor.quantile_builder_parameters is not builder
    assert executor.metric_parameters == {
        "rank_ic_series": {"min_periods": 4},
        "quantile_curvature": {"n_quantiles": 5},
    }
    assert executor.metric_parameters["rank_ic_series"] is not metric_parameters[alias]
    assert executor.metric_parameters["quantile_curvature"] is not \
        metric_parameters["quantile_curvature"]
    assert executor.quantile_builder_parameters["n_quantiles"] == 3
    assert executor.metric_parameters["quantile_curvature"]["n_quantiles"] == 5
    assert all(kwargs["quantile_builder_parameters"] is builder
               for _, kwargs in evaluate_calls)
    assert all(kwargs["metric_parameters"] is metric_parameters
               for _, kwargs in evaluate_calls)
    for label in labels:
        assert alias in result[label.target_id].series_metrics
        assert result[label.target_id].series_metrics[alias] is \
            result[label.target_id].series_metrics["rank_ic_series"]


def test_cuda_window_size_is_rejected_before_shared_session_allocation(monkeypatch):
    factors, labels = _inputs()
    session_constructions = []

    def unexpected_session(*_args, **_kwargs):
        session_constructions.append(True)
        raise AssertionError("invalid CUDA builder options must fail before allocation")

    from quant_evaluator.runtime import device_session
    monkeypatch.setattr(device_session, "DeviceEvaluationSession", unexpected_session)

    with pytest.raises(UnsupportedMetricError, match="window_size"):
        api.evaluate_many(
            factors, labels, metrics=("quantile_returns_full",),
            backend="cuda_strict",
            quantile_builder_parameters={"window_size": 10},
        )
    assert session_constructions == []


def test_cuda_global_q_without_quantile_consumer_is_rejected(monkeypatch):
    factors, labels = _inputs()
    session_constructions = []

    def unexpected_session(*_args, **_kwargs):
        session_constructions.append(True)
        raise AssertionError("invalid CUDA builder options must fail before allocation")

    from quant_evaluator.runtime import device_session
    monkeypatch.setattr(device_session, "DeviceEvaluationSession", unexpected_session)

    with pytest.raises(UnsupportedMetricError, match="GPU-profile quantile metric"):
        api.evaluate_many(
            factors, labels, metrics=("rank_ic_series",),
            backend="cuda_strict",
            quantile_builder_parameters={"n_quantiles": 3},
        )
    assert session_constructions == []
