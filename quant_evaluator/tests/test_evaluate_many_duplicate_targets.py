"""A batch evaluation must never silently discard a label result."""

import numpy as np
import pytest

from quant_evaluator.contracts.label_bundle import LabelBundle

from quant_evaluator.api import evaluate_many as api_module


@pytest.mark.parametrize("backend", ["cpu", "auto", "cuda_strict"])
def test_duplicate_target_ids_fail_before_any_evaluation(monkeypatch, backend):
    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("duplicate labels must be rejected before evaluation")

    monkeypatch.setattr(api_module, "evaluate", unexpected_evaluation)
    labels = tuple(LabelBundle(
        "same", np.array([float(seed), float(seed + 1)]), 1,
        decision_time=(0, 1), label_start_time=(0, 1), label_end_time=(1, 2),
    ) for seed in (0, 10))

    with pytest.raises(ValueError, match="unique target_id"):
        api_module.evaluate_many(object(), labels, backend=backend)


def test_shared_cuda_route_reuses_factor_diagnostics_without_gpu(monkeypatch):
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.runtime.evaluator import evaluate as real_evaluate

    times = np.arange(6)
    assets = np.asarray(["A", "B", "C", "D"])
    axis_t = AxisRef("time", "int64", len(times), times)
    axis_n = AxisRef("asset", "str", len(assets), assets)
    factors = FactorBatch(
        ("f0", "f1"), axis_t, axis_n,
        np.arange(48, dtype=np.float64).reshape(6, 4, 2),
    )
    labels = tuple(LabelBundle(
        f"h{horizon}", np.arange(24, dtype=np.float64).reshape(6, 4) + horizon,
        horizon, decision_time=tuple(times), observation_time=tuple(times),
        label_start_time=tuple(times + 1), label_end_time=tuple(times + 2),
        asset_axis=axis_n,
    ) for horizon in (1, 2))
    expected = {
        label.target_id: real_evaluate(factors, label, backend="cpu", metrics=("coverage",))
        for label in labels
    }

    diagnostic_calls = []
    original_diagnose = api_module.diagnose_all_factors

    def diagnose(batch):
        diagnostic_calls.append(batch)
        return original_diagnose(batch)

    monkeypatch.setattr(api_module, "diagnose_all_factors", diagnose)

    def route_or_evaluate(batch, label, **kwargs):
        if kwargs.get("_prepare_only"):
            return "cuda_strict", "test_route"
        kwargs.pop("_auto_route_override")
        kwargs.pop("_gpu_result_override")
        kwargs["backend"] = "cpu"
        return real_evaluate(batch, label, **kwargs)

    monkeypatch.setattr(api_module, "evaluate", route_or_evaluate)

    class Session:
        def __init__(self, policy):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Executor:
        def __init__(self, session):
            pass

        def run_tiled_many(self, batch, label_bundles, metrics):
            from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
            return [BatchEvaluationBundle(batch.factor_ids, label.target_id)
                    for label in label_bundles]

    from quant_evaluator.runtime import device_session, gpu_executor
    monkeypatch.setattr(device_session, "DeviceEvaluationSession", Session)
    monkeypatch.setattr(gpu_executor, "GPUExecutor", Executor)

    actual = api_module.evaluate_many(
        factors, labels, backend="cuda_strict", metrics=("coverage",),
    )

    assert len(diagnostic_calls) == 1 and diagnostic_calls[0] is factors
    for label in labels:
        assert actual[label.target_id].diagnostics == expected[label.target_id].diagnostics
    assert actual["h1"].diagnostics is not actual["h2"].diagnostics
