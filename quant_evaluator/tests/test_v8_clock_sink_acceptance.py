import json
from pathlib import Path

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_instance import EvaluationScenario, MetricInstance
from quant_evaluator.contracts.resampling import ResamplingPlan
from quant_evaluator.jobs.handler import QEJobHandler
from quant_evaluator.jobs.refs import QEJobShard, write_manifest
from quant_evaluator.metrics.robustness import (
    compute_block_bootstrap_ci,
    compute_hac_variance,
)
from quant_evaluator.metrics.statistical_evidence import (
    build_hac_evidence,
    build_paired_block_bootstrap_difference,
)
from quant_platform.app.contracts import JobSpec


def test_t33_missing_positions_never_acquire_compressed_hac_or_bootstrap_clock():
    complete = np.linspace(-0.2, 0.3, 40)
    gap_early = complete.copy(); gap_early[9] = np.nan
    gap_late = complete.copy(); gap_late[27] = np.nan

    for gapped in (gap_early, gap_late):
        hac = build_hac_evidence(gapped, max_lag=2, min_periods=20)
        paired = build_paired_block_bootstrap_difference(
            gapped, np.zeros_like(gapped), block_length=4,
            repetitions=20, min_periods=20,
        )
        assert hac.status == "INSUFFICIENT"
        assert hac.standard_error is None and hac.t_statistic is None
        assert paired.status == "INSUFFICIENT"
        assert paired.confidence_interval == (None, None)
        assert np.isnan(compute_hac_variance(gapped, max_lag=2)[0])

    lo, hi = compute_block_bootstrap_ci(
        np.column_stack((complete, gap_early, gap_late)),
        block_length=4, num_bootstrap=20, random_seed=33,
    )
    assert np.isfinite((lo[0], hi[0])).all()
    assert np.isnan(lo[1:]).all() and np.isnan(hi[1:]).all()


def test_t36_singleton_repeats_short_span_and_oversized_blocks_cannot_be_evidence():
    values = np.linspace(-0.1, 0.2, 30)
    with pytest.raises(ValueError, match="repetitions"):
        build_paired_block_bootstrap_difference(
            values, np.zeros_like(values), block_length=5,
            repetitions=1, min_periods=20,
        )
    with pytest.raises(ValueError, match="num_bootstrap"):
        compute_block_bootstrap_ci(values, block_length=5, num_bootstrap=1)
    with pytest.raises(ValueError, match="insufficient time span"):
        ResamplingPlan(tuple(range(7)), "clock", 4, 10, 36)
    paired = build_paired_block_bootstrap_difference(
        values[:6], np.zeros(6), block_length=7,
        repetitions=20, min_periods=5,
    )
    assert paired.status == "INSUFFICIENT"
    assert paired.mean_difference is None


def _sink_job(tmp_path):
    rng = np.random.default_rng(76)
    t, n, f = 8, 30, 4
    factor_ids = tuple(f"factor:f{i}" for i in range(f))
    batch = FactorBatch(
        factor_ids, AxisRef("time", "int", t, np.arange(t)),
        AxisRef("asset", "str", n, np.asarray([f"a{i}" for i in range(n)])),
        np.ascontiguousarray(rng.normal(size=(t, n, f))),
    )
    labels = LabelBundle(
        "return", np.ascontiguousarray(rng.normal(size=(t, n))), 1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)),
    )
    instance = MetricInstance(
        "rank_ic_series", scenario_id="s", horizon=1,
        price_convention="vwap_to_vwap",
    )
    shard = QEJobShard(
        factor_ids, "label:h1", (instance.to_dict(),),
        (("s", "scenario:s"),), str(tmp_path / "sink"), 0,
        tier="extended", cost_budget=10, allowed_output_modes=("FULL_DIAGNOSTIC",),
    )
    manifest = write_manifest(tmp_path / "manifests", shard)
    handler = QEJobHandler(
        factor_batch_resolver=lambda _: batch,
        label_resolver=lambda _: labels,
        scenario_resolver=lambda _: EvaluationScenario(labels),
    )
    return handler, JobSpec(
        "qe_metric_instance_shard", "t76-safe-key",
        input_artifact_refs=(manifest,),
    ), factor_ids


def test_t76_oom_retile_preserves_numbers_and_never_recounts_committed_tile(tmp_path, monkeypatch):
    cp = pytest.importorskip("cupy")
    import quant_evaluator
    from quant_evaluator.api.requests import EvaluationBundle
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession
    from quant_evaluator.runtime.gpu_executor import GPUExecutor

    handler, spec, factor_ids = _sink_job(tmp_path)
    public_evaluate = quant_evaluator.evaluate
    attempts = []
    failed = {"once": False}
    gpu_run = GPUExecutor.run

    def routed(request):
        return public_evaluate(request, backend="cuda_strict")

    def run_with_one_oom(self, ids, metrics, label_id="next_ret"):
        attempts.append(tuple(ids))
        if tuple(ids) == factor_ids[2:] and not failed["once"]:
            failed["once"] = True
            raise cp.cuda.memory.OutOfMemoryError(1, 1, 1)
        return gpu_run(self, ids, metrics, label_id)

    monkeypatch.setattr(quant_evaluator, "evaluate", routed)
    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(GPUExecutor, "run", run_with_one_oom)
    result = handler(spec)
    assert attempts == [factor_ids[:2], factor_ids[2:], factor_ids[2:3], factor_ids[3:]]
    assert attempts.count(factor_ids[:2]) == 1
    assert len(result.output_artifact_refs) == 1
    payload = json.loads(Path(result.output_artifact_refs[0]).read_text())
    restored = EvaluationBundle.from_dict(payload)

    # Independent no-OOM production evaluation is the numeric oracle.
    monkeypatch.setattr(quant_evaluator, "evaluate", public_evaluate)
    batch = handler.factor_batch_resolver(factor_ids)
    labels = handler.label_resolver("label:h1")
    expected = public_evaluate(
        batch, labels, backend="cuda_strict", metrics=("rank_ic_series",),
    )
    np.testing.assert_allclose(
        restored.artifacts["rank_ic_series"].values,
        expected.artifacts["rank_ic_series"].values,
        rtol=0, atol=1e-12, equal_nan=True,
    )
