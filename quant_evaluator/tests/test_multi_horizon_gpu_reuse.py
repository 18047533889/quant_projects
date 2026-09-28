"""CUDA multi-label factor-tile reuse and pairwise-mask isolation."""

import numpy as np
import pytest

pytest.importorskip("cupy")

from quant_evaluator.api.evaluate_many import evaluate_many
from quant_evaluator.api.horizons import evaluate_horizons
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.device_session import DeviceEvaluationSession


def test_gpu_execution_policy_resident_label_preference_is_opt_in_and_boolean():
    assert GPUExecutionPolicy().prefer_resident_labels is False
    assert GPUExecutionPolicy(prefer_resident_labels=True).prefer_resident_labels is True
    with pytest.raises(ValueError, match="prefer_resident_labels must be a bool"):
        GPUExecutionPolicy(prefer_resident_labels=1)


def _inputs(n=40):
    rng = np.random.default_rng(20260927)
    t, f = 24, 5
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
    label_uploads = []
    opens = []
    original_stage = DeviceEvaluationSession.stage_factors
    original_stage_labels = DeviceEvaluationSession.stage_labels
    original_open = DeviceEvaluationSession._open

    def stage(self, values, factor_ids, layout="T,F,N"):
        uploads.append(tuple(factor_ids))
        return original_stage(self, values, factor_ids, layout)

    def stage_label(self, values, target_id="next_ret"):
        label_uploads.append(target_id)
        return original_stage_labels(self, values, target_id)

    def opened(self):
        opens.append(True)
        return original_open(self)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", stage)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_labels", stage_label)
    monkeypatch.setattr(DeviceEvaluationSession, "_open", opened)
    gpu = evaluate_many(factors, labels, backend="cuda_strict", metrics=("rank_ic_series",),
                        gpu_policy=GPUExecutionPolicy(prefer_resident_labels=True))
    assert len(opens) == 1
    assert uploads == [("f0", "f1"), ("f2", "f3"), ("f4",)]
    assert label_uploads == ["h1", "h5"]
    # Factor tiles and each horizon label are uploaded exactly once.
    expected_h2d = factors.values.nbytes + sum(lb.values.nbytes for lb in labels)
    for result in gpu.values():
        assert result.metadata["device_session_counter_scope"] == "shared_session_total"
        assert result.metadata["shared_session_label_count"] == 2
        assert result.metadata["shared_session_factor_tiles_processed"] == 3
        assert result.metadata["label_staging_preference"] == "resident"
        assert result.metadata["label_staging_mode"] == "resident"
        assert result.metadata["h2d_bytes"] == expected_h2d
    for lb in labels:
        np.testing.assert_allclose(
            gpu[lb.target_id].artifacts["rank_ic_series"].values,
            cpu[lb.target_id].artifacts["rank_ic_series"].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )
        assert gpu[lb.target_id].metadata["multi_label_factor_tile_reuse"] is True


@pytest.mark.parametrize("metric_id", ["rank_ic_series", "ic.rank.daily"])
def test_evaluate_many_metric_alias_shares_cuda_and_matches_cpu(monkeypatch, metric_id):
    factors, labels = _inputs()
    canonical_cpu = evaluate_many(
        factors, labels, backend="cpu", metrics=("rank_ic_series",))
    cpu = evaluate_many(factors, labels, backend="cpu", metrics=(metric_id,))

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
    gpu = evaluate_many(factors, labels, backend="cuda_strict", metrics=(metric_id,))

    assert len(opens) == 1
    assert uploads == [("f0", "f1"), ("f2", "f3"), ("f4",)]
    for lb in labels:
        actual = gpu[lb.target_id]
        alias_or_canonical_cpu = cpu[lb.target_id]
        reference = canonical_cpu[lb.target_id]
        assert actual.config_hash == alias_or_canonical_cpu.config_hash
        assert metric_id in actual.artifacts
        assert actual.metadata["metric_backends"] == {metric_id: "cuda"}
        assert actual.metadata["multi_label_factor_tile_reuse"] is True
        np.testing.assert_allclose(
            actual.artifacts[metric_id].values,
            alias_or_canonical_cpu.artifacts[metric_id].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )
        np.testing.assert_allclose(
            actual.artifacts[metric_id].values,
            reference.artifacts["rank_ic_series"].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )


def test_evaluate_many_quantile_chain_reuses_uploads_and_matches_cpu(monkeypatch):
    factors, labels = _inputs(n=80)
    metrics = ("quantile_returns_full", "quantile_returns_daily",
               "quantile_spread", "quantile_monotonicity",
               "daily_quantile_monotonicity_rate")
    cpu = evaluate_many(factors, labels, backend="cpu", metrics=metrics)
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
    gpu = evaluate_many(factors, labels, backend="cuda_strict", metrics=metrics,
                        gpu_policy=GPUExecutionPolicy(prefer_resident_labels=True))
    assert len(opens) == 1
    assert uploads == [("f0", "f1"), ("f2", "f3"), ("f4",)]
    assert any(np.isfinite(cpu["h1"].artifacts["quantile_returns_daily"].values.ravel()))
    for label in labels:
        left, right = cpu[label.target_id], gpu[label.target_id]
        assert right.config_hash == left.config_hash
        assert right.metadata["multi_label_factor_tile_reuse"] is True
        assert right.metadata["shared_session_factor_tiles_processed"] == 3
        assert right.metadata["label_staging_preference"] == "resident"
        assert right.metadata["label_staging_mode"] == "resident"
        for metric in metrics:
            a, b = left.artifacts[metric], right.artifacts[metric]
            np.testing.assert_allclose(b.values, a.values, rtol=1e-8,
                                       atol=1e-10, equal_nan=True)
            for field in ("counts", "valid_mask"):
                expected, actual = getattr(a, field, None), getattr(b, field, None)
                if expected is not None and actual is not None:
                    np.testing.assert_array_equal(actual, expected)


def test_evaluate_many_forwards_quantile_parameters_to_shared_gpu():
    factors, labels = _inputs()
    options = {"quantile_returns_daily": {"n_quantiles": 1}}
    cpu = evaluate_many(
        factors, labels, backend="cpu", metrics=("quantile_returns_daily",),
        metric_parameters=options)
    gpu = evaluate_many(
        factors, labels, backend="cuda_strict", metrics=("quantile_returns_daily",),
        metric_parameters=options)
    for label in labels:
        left = cpu[label.target_id].artifacts["quantile_returns_daily"]
        right = gpu[label.target_id].artifacts["quantile_returns_daily"]
        assert left.values.shape == (24, 1, 5)
        np.testing.assert_allclose(right.values, left.values, equal_nan=True)
        np.testing.assert_array_equal(right.counts, left.counts)
        assert gpu[label.target_id].metadata["multi_label_factor_tile_reuse"] is True


def test_evaluate_many_mixed_panel_metrics_share_cuda_and_match_cpu(monkeypatch):
    factors, labels = _inputs(n=80)
    metrics = ("rank_ic", "quantile_spread", "factor_turnover_rate")
    cpu = evaluate_many(factors, labels, backend="cpu", metrics=metrics)
    uploads = []
    original_stage = DeviceEvaluationSession.stage_factors

    def stage(self, values, factor_ids, layout="T,F,N"):
        uploads.append(tuple(factor_ids))
        return original_stage(self, values, factor_ids, layout)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", stage)
    gpu = evaluate_many(factors, labels, backend="cuda_strict", metrics=metrics,
                        gpu_policy=GPUExecutionPolicy(prefer_resident_labels=True))
    assert uploads == [("f0", "f1"), ("f2", "f3"), ("f4",)]
    for label in labels:
        left, right = cpu[label.target_id], gpu[label.target_id]
        assert right.config_hash == left.config_hash
        assert right.metadata["multi_label_factor_tile_reuse"] is True
        for metric in metrics:
            a, b = left.artifacts[metric], right.artifacts[metric]
            np.testing.assert_allclose(b.values, a.values, rtol=1e-8,
                                       atol=1e-10, equal_nan=True)


def test_evaluate_many_rank_ic_positive_ratio_reuses_cuda_uploads_with_parity(monkeypatch):
    factors, labels = _inputs()
    metrics = ("rank_ic_positive_ratio", "ic.rank.mean")
    parameters = {
        "rank_ic_positive_ratio": {"min_periods": 12},
        "ic.rank.mean": {"min_periods": 8},
    }
    cpu = evaluate_many(
        factors, labels, backend="cpu", metrics=metrics,
        metric_parameters=parameters,
    )
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
    gpu = evaluate_many(
        factors, labels, backend="cuda_strict", metrics=metrics,
        metric_parameters=parameters,
        gpu_policy=GPUExecutionPolicy(prefer_resident_labels=True),
    )

    assert len(opens) == 1
    assert uploads == [("f0", "f1"), ("f2", "f3"), ("f4",)]
    expected_h2d = factors.values.nbytes + sum(lb.values.nbytes for lb in labels)
    for label in labels:
        left, right = cpu[label.target_id], gpu[label.target_id]
        assert right.config_hash == left.config_hash
        assert right.metadata["metric_backends"] == {
            "rank_ic_positive_ratio": "cuda", "ic.rank.mean": "cuda",
        }
        assert right.metadata["multi_label_factor_tile_reuse"] is True
        assert right.metadata["shared_session_label_count"] == len(labels)
        assert right.metadata["shared_session_factor_tiles_processed"] == 3
        assert right.metadata["device_session_counter_scope"] == "shared_session_total"
        assert right.metadata["label_staging_preference"] == "resident"
        assert right.metadata["label_staging_mode"] == "resident"
        assert right.metadata["h2d_bytes"] == expected_h2d
        for metric in metrics:
            expected, actual = left.artifacts[metric], right.artifacts[metric]
            np.testing.assert_allclose(
                actual.values, expected.values, rtol=0, atol=1e-12,
                equal_nan=True,
            )
            np.testing.assert_array_equal(
                actual.provenance["observation_counts"],
                expected.provenance["observation_counts"],
            )
            for factor_id in factors.factor_ids:
                cpu_value = left.get_metric(metric, factor_id)
                gpu_value = right.get_metric(metric, factor_id)
                assert gpu_value.metric_id == cpu_value.metric_id
                assert gpu_value.valid == cpu_value.valid
                assert gpu_value.observation_count == cpu_value.observation_count
                if cpu_value.valid:
                    assert gpu_value.value == pytest.approx(cpu_value.value, abs=1e-12)

    # Distinct label streams must remain distinct despite sharing each tile.
    assert not np.allclose(
        gpu["h1"].artifacts["rank_ic_positive_ratio"].values,
        gpu["h5"].artifacts["rank_ic_positive_ratio"].values,
        rtol=0, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("metric_id", ["quantile_spread", "ic.rank.mean"])
def test_evaluate_many_auto_unverified_shape_keeps_cpu_without_device(monkeypatch, metric_id):
    factors, labels = _inputs(n=80)
    monkeypatch.setattr(
        DeviceEvaluationSession, "_open",
        lambda self: (_ for _ in ()).throw(AssertionError("unexpected GPU session")))
    results = evaluate_many(
        factors, labels, backend="auto", metrics=(metric_id,))
    assert set(results) == {"h1", "h5"}
    assert all(result.metadata["backend_used"] == "cpu"
               for result in results.values())
    assert all(result.metadata["auto_backend_reason"] == "shape_outside_certified_range"
               for result in results.values())
    assert all(metric_id in result.artifacts for result in results.values())


def test_evaluate_many_auto_shared_cuda_reuses_preflight_decision(monkeypatch):
    factors, labels = _inputs()
    from quant_evaluator.runtime import evaluator as evaluator_module

    selector_calls = []

    def select(*args, **kwargs):
        selector_calls.append(args[1].target_id)
        return "cuda_strict", "test_preflight_route"

    monkeypatch.setattr(evaluator_module, "_select_public_auto_backend", select)
    cpu = evaluate_many(factors, labels, backend="cpu", metrics=("rank_ic_series",))
    result = evaluate_many(factors, labels, backend="auto", metrics=("rank_ic_series",))

    # One routing/admission decision per label, with no second probe while
    # wrapping the shared GPU result.
    assert selector_calls == ["h1", "h5"]
    for label in labels:
        np.testing.assert_allclose(
            result[label.target_id].artifacts["rank_ic_series"].values,
            cpu[label.target_id].artifacts["rank_ic_series"].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )
        assert result[label.target_id].metadata["backend_used"] == "cuda"
        assert result[label.target_id].metadata["backend_strategy"] == "auto"
        assert result[label.target_id].metadata["auto_backend_reason"] == "test_preflight_route"


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




def test_evaluate_many_cuda_defaults_to_per_tile_label_staging(monkeypatch):
    factors, labels = _inputs()
    label_uploads = []
    original_stage_labels = DeviceEvaluationSession.stage_labels

    def stage_labels(self, values, target_id="next_ret"):
        label_uploads.append(target_id)
        return original_stage_labels(self, values, target_id)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_labels", stage_labels)
    results = evaluate_many(factors, labels, backend="cuda_strict",
                            metrics=("rank_ic_series",))
    assert label_uploads == ["h1", "h5"] * 3
    for result in results.values():
        assert result.metadata["label_staging_preference"] == "per_tile"
        assert result.metadata["label_staging_mode"] == "per_tile"


def test_evaluate_many_cuda_uses_per_tile_labels_when_budget_rejects_residency(monkeypatch):
    factors, labels = _inputs()
    from quant_evaluator.runtime.evaluator import evaluate
    cpu = {
        lb.target_id: evaluate(factors, lb, backend="cpu", metrics=("rank_ic_series",))
        for lb in labels
    }
    uploads = []
    label_uploads = []
    original_stage_factors = DeviceEvaluationSession.stage_factors
    original_stage_labels = DeviceEvaluationSession.stage_labels
    original_open = DeviceEvaluationSession._open

    def stage_factors(self, values, factor_ids, layout="T,F,N"):
        uploads.append(tuple(factor_ids))
        return original_stage_factors(self, values, factor_ids, layout)

    def stage_labels(self, values, target_id="next_ret"):
        label_uploads.append(target_id)
        return original_stage_labels(self, values, target_id)

    def opened(self):
        original_open(self)
        self._vram_budget = 1

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", stage_factors)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_labels", stage_labels)
    monkeypatch.setattr(DeviceEvaluationSession, "_open", opened)
    gpu = evaluate_many(factors, labels, backend="cuda_strict", metrics=("rank_ic_series",),
                        gpu_policy=GPUExecutionPolicy(prefer_resident_labels=True))

    assert uploads == [("f0", "f1"), ("f2", "f3"), ("f4",)]
    assert label_uploads == ["h1", "h5"] * 3
    for lb in labels:
        actual = gpu[lb.target_id]
        assert actual.metadata["label_staging_preference"] == "resident"
        assert actual.metadata["label_staging_mode"] == "per_tile"
        np.testing.assert_allclose(
            actual.artifacts["rank_ic_series"].values,
            cpu[lb.target_id].artifacts["rank_ic_series"].values,
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
    assert result["h1"].metadata["label_staging_preference"] == "per_tile"
    assert result["h1"].metadata["label_staging_mode"] == "per_tile"
    for lb in labels:
        np.testing.assert_allclose(
            result[lb.target_id].artifacts["rank_ic_series"].values,
            reference[lb.target_id].artifacts["rank_ic_series"].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )



def test_evaluate_many_cuda_falls_back_after_resident_label_oom(monkeypatch):
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
    result = evaluate_many(factors, labels, backend="cuda_strict", metrics=("rank_ic_series",),
                           gpu_policy=GPUExecutionPolicy(prefer_resident_labels=True))
    assert attempts[:2] == [(4, "h1"), (4, "h5")]
    assert result["h1"].metadata["oom_retries"] == 0
    assert result["h1"].metadata["label_staging_preference"] == "resident"
    assert result["h1"].metadata["label_staging_mode"] == "per_tile_fallback_after_oom"
    for lb in labels:
        np.testing.assert_allclose(
            result[lb.target_id].artifacts["rank_ic_series"].values,
            reference[lb.target_id].artifacts["rank_ic_series"].values,
            rtol=0, atol=1e-12, equal_nan=True,
        )
