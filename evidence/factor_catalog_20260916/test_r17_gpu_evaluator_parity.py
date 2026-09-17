"""Tiny same-input CPU/CUDA parity, not an all-metric certification."""
import numpy as np
import pytest


@pytest.mark.parametrize("ties_and_missing", [False, True])
def test_public_cuda_rank_ic_matches_cpu_per_factor(ties_and_missing):
    cp = pytest.importorskip("cupy")
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate

    rng = np.random.default_rng(173)
    times, assets, factors = 80, 24, 2
    values = rng.normal(size=(times, assets, factors))
    labels = rng.normal(size=(times, assets))
    if ties_and_missing:
        values = np.round(values)
        labels = np.round(labels)
        values[::7, ::3, :] = np.nan
        labels[::5, ::4] = np.nan
    lb = LabelBundle(
        target_id="r", values=np.ascontiguousarray(labels), horizon=1,
        decision_time=tuple(range(times)), label_start_time=tuple(range(times)),
        label_end_time=tuple(range(1, times + 1)),
    )
    def batch(array, ids):
        return FactorBatch(
            factor_ids=ids, time_axis=AxisRef("t", "int", times),
            asset_axis=AxisRef("a", "str", assets),
            values=np.ascontiguousarray(array),
        )
    gpu = evaluate(batch(values, ("g0", "g1")), lb,
                   metrics=("rank_ic",), backend="cuda")
    # Public EvaluationBundle intentionally owns host arrays after bounded D2H.
    assert gpu.metadata["backend_used"] == "cuda"
    assert gpu.metadata["h2d_bytes"] > 0
    assert gpu.metadata["d2h_bytes"] > 0
    actual = np.asarray(gpu.scalar_metrics["rank_ic"])
    expected = np.array([
        float(evaluate(batch(values[:, :, i:i + 1], (f"g{i}",)), lb,
                       metrics=("rank_ic",)).metric_values["rank_ic"].value)
        for i in range(factors)
    ])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
