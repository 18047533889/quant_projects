"""CUDA multi-label factor-tile reuse and pairwise-mask isolation."""

import numpy as np
import pytest

pytest.importorskip("cupy")

from quant_evaluator.api.evaluate_many import evaluate_many
from quant_evaluator.api.horizons import evaluate_horizons
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.device_session import DeviceEvaluationSession


def _inputs():
    rng = np.random.default_rng(20260927)
    t, n, f = 24, 40, 5
    x = rng.integers(0, 12, size=(t, n, f)).astype(float)
    x[4, 3, 1] = np.nan
    times = np.arange(t)
    asset_axis = AxisRef("asset", "str", n, np.asarray([f"S{i:03d}" for i in range(n)]))
    factors = FactorBatch(
        tuple(f"f{i}" for i in range(f)), AxisRef("time", "int64", t, times),
        asset_axis, x,
    )
    labels = []
    for horizon in (1, 5):
        y = rng.integers(0, 9, size=(t, n)).astype(float)
        valid = np.ones((t, n), dtype=bool)
        if horizon == 1:
            valid[:, :5] = False
        else:
            valid[:, 20:30] = False
            y[4, 3] = np.nan
        labels.append(LabelBundle(
            f"h{horizon}", y, horizon,
            decision_time=tuple(times), observation_time=tuple(times),
            label_start_time=tuple(times),
            label_end_time=tuple(times + horizon),
            asset_axis=asset_axis, validity=valid,
        ))
    return factors, labels


def test_evaluate_many_cuda_reuses_tiles_and_isolates_pairwise_ranks(monkeypatch):
    factors, labels = _inputs()
    from quant_evaluator.runtime.evaluator import evaluate
    cpu = {
        lb.target_id: evaluate(factors, lb, backend="cpu", metrics=("rank_ic_series",))
        for lb in labels
    }
    uploads = []
    opens = []
    original_stage = DeviceEvaluationSession.stage_factors
    original_open = DeviceEvaluationSession._open

    def stage(self, values, factor_ids, layout="T,F,N"):
        uploads.append(tuple(factor_ids))
        return original_stage(self, values, factor_ids, layout)

    def opened(self):
        opens.append(True)
        return original_open(self)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", stage)
    monkeypatch.setattr(DeviceEvaluationSession, "_open", opened)
    gpu = evaluate_many(factors, labels, backend="cuda_strict", metrics=("rank_ic_series",))
    assert len(opens) == 1
    assert uploads == [("f0", "f1"), ("f2", "f3"), ("f4",)]
    # Transfer counters describe the one shared session, not either result alone.
    expected_h2d = factors.values.nbytes + len(labels) * 3 * labels[0].values.nbytes
    for result in gpu.values():
        assert result.metadata["device_session_counter_scope"] == "shared_session_total"
        assert result.metadata["shared_session_label_count"] == 2
        assert result.metadata["shared_session_factor_tiles_processed"] == 3
        assert result.metadata["h2d_bytes"] == expected_h2d
    for lb in labels:
        np.testing.assert_allclose(
            gpu[lb.target_id].artifacts["rank_ic_series"].values,
            cpu[lb.target_id].artifacts["rank_ic_series"].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )
        assert gpu[lb.target_id].metadata["multi_label_factor_tile_reuse"] is True


@pytest.mark.parametrize("policy", ["common", "per_horizon"])
@pytest.mark.parametrize("ic_method", ["spearman", "pearson"])
def test_evaluate_horizons_cuda_reuses_tiles_with_selected_masks(monkeypatch, policy, ic_method):
    factors, label_list = _inputs()
    labels = {lb.horizon: lb for lb in label_list}
    cpu = evaluate_horizons(
        factors, labels, as_of=25, sample_policy=policy,
        min_assets=10, min_periods=3, backend="cpu", ic_method=ic_method,
    )
    uploads = []
    original_stage = DeviceEvaluationSession.stage_factors

    def stage(self, values, factor_ids, layout="T,F,N"):
        uploads.append(tuple(factor_ids))
        return original_stage(self, values, factor_ids, layout)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", stage)
    gpu = evaluate_horizons(
        factors, labels, as_of=25, sample_policy=policy,
        min_assets=10, min_periods=3, backend="cuda_strict", ic_method=ic_method,
    )
    assert uploads == [("f0", "f1"), ("f2", "f3"), ("f4",)]
    for horizon in labels:
        np.testing.assert_allclose(
            gpu.daily_ic_artifacts[horizon].values,
            cpu.daily_ic_artifacts[horizon].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )


def test_evaluate_many_cuda_retiles_after_second_label_oom(monkeypatch):
    import cupy as cp
    from quant_evaluator.runtime.gpu_executor import GPUExecutor

    factors, labels = _inputs()
    reference = evaluate_many(factors, labels, backend="cpu", metrics=("rank_ic_series",))

    original_run = GPUExecutor.run
    attempts = []
    failed = False

    def run(self, factor_ids, metrics, label_id="next_ret"):
        nonlocal failed
        attempts.append((len(factor_ids), label_id))
        if len(factor_ids) > 2 and label_id == "h5" and not failed:
            failed = True
            raise cp.cuda.memory.OutOfMemoryError(100, 100, 100)
        return original_run(self, factor_ids, metrics, label_id)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 4)
    monkeypatch.setattr(GPUExecutor, "run", run)
    result = evaluate_many(factors, labels, backend="cuda_strict", metrics=("rank_ic_series",))
    assert attempts[:2] == [(4, "h1"), (4, "h5")]
    assert result["h1"].metadata["oom_retries"] == 1
    for lb in labels:
        np.testing.assert_allclose(
            result[lb.target_id].artifacts["rank_ic_series"].values,
            reference[lb.target_id].artifacts["rank_ic_series"].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )
