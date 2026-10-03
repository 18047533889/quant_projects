from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.scripts.source_profile_abba import (
    IndependentOracleMismatchError, produce_source_route_profile_abba,
)
from test_source_profile_measurement_oct04 import sample as sample_fixture


@pytest.fixture
def sample(request):
    # Pure orchestration fixture only; these contexts do not constitute live evidence.
    return request.getfixturevalue("sample_fixture")


def _run_setup(sample):
    context, cpu, cuda, cpu_receipt, cuda_receipt, _ = sample
    order = []
    oracle_calls = []

    def observer(**kwargs):
        return context

    def runner(backend, *, context_observer, **kwargs):
        order.append(backend)
        index = len(order) - 1
        observed_before = context_observer(phase="before", **kwargs)
        observed_after = context_observer(phase="after", **kwargs)
        bundle = cpu if backend == "cpu" else cuda
        base = cpu_receipt if backend == "cpu" else cuda_receipt
        receipt = dict(base, seconds=float(base["seconds"] + index / 100.0),
                       total_wall_seconds=float(base["seconds"] + index / 100.0),
                       context_before=observed_before,
                       context_after=observed_after)
        return bundle, receipt

    def oracle(*, bundle, run_receipt, context, backend, run_index):
        oracle_calls.append((backend, run_index))
        return cpu

    result = produce_source_route_profile_abba(
        run_backend_fn=runner, context_observer=observer,
        run_kwargs={"request": "test"}, oracle=oracle,
        comparison_fn=lambda *args, **kwargs: sample[5])
    return result, order, oracle_calls


def test_runs_real_pair_builder_in_abba_order_and_executes_oracle_for_every_run(sample):
    result, order, oracle_calls = _run_setup(sample)

    assert order == ["cpu", "cuda_strict", "cuda_strict", "cpu"]
    assert [backend for backend, _ in oracle_calls] == ["cpu", "cuda", "cuda", "cpu"]
    assert [index for _, index in oracle_calls] == [0, 1, 2, 3]
    assert len(result.records) == 2
    assert result.records[0].execution_order == ("cpu", "cuda")
    assert result.records[1].execution_order == ("cuda", "cpu")
    assert all(run["pass"] for run in result.oracle_reports)
    report = result.oracle_reports[0]["metrics"]["rank_ic_series"]
    assert report["expected_coverage"] == 15
    assert report["observed_coverage"] == 15
    assert report["finite_mask_equal"] is True
    assert report["nan_mask_equal"] is True
    assert report["positive_infinity_mask_equal"] is True
    assert report["negative_infinity_mask_equal"] is True
    assert report["observation_counts_equal"] is True
    assert report["max_abs_error"] == 0.0


@pytest.mark.parametrize("oracle", [None, True], ids=["missing", "boolean"])
def test_missing_or_noncallable_oracle_fails_before_any_backend_run(sample, oracle):
    context, *_ = sample
    calls = []
    with pytest.raises(ValueError, match="independent oracle"):
        produce_source_route_profile_abba(
            run_backend_fn=lambda *args, **kwargs: calls.append(args),
            run_kwargs={}, oracle=None)
    assert calls == []


def test_oracle_mismatch_blocks_profile_record_creation(sample):
    context, cpu, cuda, cpu_receipt, cuda_receipt, _ = sample
    calls = []

    def observer(**kwargs):
        return context

    def runner(backend, *, context_observer, **kwargs):
        i = len(calls)
        calls.append(backend)
        before = context_observer(phase="before", **kwargs)
        after = context_observer(phase="after", **kwargs)
        bundle = cpu if backend == "cpu" else cuda
        base = cpu_receipt if backend == "cpu" else cuda_receipt
        receipt = dict(base, context_before=before, context_after=after)
        return bundle, receipt

    def bad_oracle(*, bundle, run_receipt, context, backend, run_index):
        values = cpu.scalar_metrics["rank_ic"].copy()
        values[0] += 1e-6
        return {"rank_ic": {"values": values,
                            "observation_counts": cpu.observation_counts["rank_ic"]},
                "rank_ic_series": {
                    "values": cpu.series_metrics["rank_ic_series"],
                    "observation_counts": cpu.observation_counts["rank_ic_series"]}}

    with pytest.raises(IndependentOracleMismatchError, match="rank_ic") as caught:
        produce_source_route_profile_abba(
            run_backend_fn=runner, context_observer=observer,
            run_kwargs={}, oracle=bad_oracle,
            comparison_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("pairwise comparator must not mask oracle mismatch")))
    assert calls == ["cpu"]
    assert caught.value.report["pass"] is False
    assert set(caught.value.report["metrics"]) == {"rank_ic", "rank_ic_series"}


@pytest.mark.parametrize("bad_counts", [
    np.array([4, -1, 4, 2, 4], dtype=np.int64),
    np.array([4, 3, 4, 2, 2**63], dtype=np.uint64),
])
def test_oracle_rejects_negative_or_int64_overflow_counts(sample, bad_counts):
    _, cpu, *_ = sample
    def oracle(*, bundle, **kwargs):
        return {
            "rank_ic": {"values": cpu.scalar_metrics["rank_ic"], "observation_counts": bad_counts},
            "rank_ic_series": {"values": cpu.series_metrics["rank_ic_series"],
                               "observation_counts": cpu.observation_counts["rank_ic_series"]},
        }
    with pytest.raises(IndependentOracleMismatchError, match="counts are malformed"):
        _run_with_oracle(sample, oracle)


def _run_with_oracle(sample, oracle, *, observer=None, runner_mutator=None, progress_observer=None):
    context, cpu, cuda, cpu_receipt, cuda_receipt, comparison = sample
    run_index = 0
    def stable_observer(**kwargs):
        return context
    def runner(backend, *, context_observer, **kwargs):
        nonlocal run_index
        index = run_index
        run_index += 1
        before = context_observer(phase="before", **kwargs)
        after = context_observer(phase="after", **kwargs)
        bundle = cpu if backend == "cpu" else cuda
        base = cpu_receipt if backend == "cpu" else cuda_receipt
        receipt = dict(base, context_before=before, context_after=after)
        if runner_mutator is not None:
            receipt = runner_mutator(receipt, index, backend)
        return bundle, receipt
    return produce_source_route_profile_abba(
        run_backend_fn=runner, context_observer=observer or stable_observer,
        run_kwargs={"request": "test"}, oracle=oracle,
        comparison_fn=lambda *args, **kwargs: comparison,
        progress_observer=progress_observer)


@pytest.mark.parametrize("mutation", ["nan", "positive_inf", "negative_inf"])
def test_oracle_rejects_nonfinite_category_mask_drift(sample, mutation):
    _, cpu, *_ = sample
    def oracle(*, bundle, **kwargs):
        values = cpu.scalar_metrics["rank_ic"].copy()
        values[0] = {"nan": np.nan, "positive_inf": np.inf,
                     "negative_inf": -np.inf}[mutation]
        return {
            "rank_ic": {"values": values,
                        "observation_counts": cpu.observation_counts["rank_ic"]},
            "rank_ic_series": {"values": cpu.series_metrics["rank_ic_series"],
                               "observation_counts": cpu.observation_counts["rank_ic_series"]},
        }
    with pytest.raises(IndependentOracleMismatchError, match="rank_ic"):
        _run_with_oracle(sample, oracle)


def test_oracle_rejects_scalar_and_series_shape_mismatch(sample):
    _, cpu, *_ = sample
    def oracle(*, bundle, **kwargs):
        return {
            "rank_ic": {"values": cpu.scalar_metrics["rank_ic"][:-1],
                        "observation_counts": cpu.observation_counts["rank_ic"]},
            "rank_ic_series": {"values": cpu.series_metrics["rank_ic_series"][:, :-1],
                               "observation_counts": cpu.observation_counts["rank_ic_series"]},
        }
    with pytest.raises(IndependentOracleMismatchError, match="rank_ic"):
        _run_with_oracle(sample, oracle)


@pytest.mark.parametrize("drift", ["within_run", "between_runs"])
def test_context_drift_fails_closed(sample, drift):
    context, *_ = sample
    calls = 0
    def observer(**kwargs):
        nonlocal calls
        calls += 1
        if ((drift == "within_run" and calls == 2)
                or (drift == "between_runs" and calls == 3)):
            return replace(context, requested_tile_size=context.requested_tile_size + 1)
        return context
    with pytest.raises(ValueError, match="context changed|do not share"):
        _run_with_oracle(sample, lambda *, bundle, **kwargs: sample[1], observer=observer)


def test_wrong_backend_receipt_fails_closed(sample):
    def mutate(receipt, index, backend):
        if index == 1:
            receipt["backend_used"] = "cpu"
        return receipt
    with pytest.raises(ValueError, match="different backend"):
        _run_with_oracle(sample, lambda *, bundle, **kwargs: sample[1],
                         runner_mutator=mutate)
@pytest.mark.parametrize("bad_counts", [
    np.array([4, 3, 4, 2, 4], dtype=np.bool_),
    np.array([4, 3, 4, 2], dtype=np.int64),
])
def test_oracle_rejects_bool_or_wrong_shape_counts(sample, bad_counts):
    _, cpu, *_ = sample
    def oracle(*, bundle, **kwargs):
        return {
            "rank_ic": {"values": cpu.scalar_metrics["rank_ic"], "observation_counts": bad_counts},
            "rank_ic_series": {"values": cpu.series_metrics["rank_ic_series"],
                               "observation_counts": cpu.observation_counts["rank_ic_series"]},
        }
    with pytest.raises(IndependentOracleMismatchError, match="counts are malformed"):
        _run_with_oracle(sample, oracle)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_default_observer_calls_live_capture_seam_for_each_phase(monkeypatch, sample, phase):
    import quant_evaluator.scripts.source_profile_abba as abba
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    metadata = object()
    seen = {}
    class CaptureSentinel(Exception):
        pass
    monkeypatch.setattr(abba, "capture_factor_tile_source", lambda source: metadata)
    import quant_evaluator.api.factor_source as factor_source
    monkeypatch.setattr(factor_source, "_source_request_fingerprint",
                        lambda m, l, metrics: "fingerprint")
    def capture(**kwargs):
        seen.update(kwargs)
        raise CaptureSentinel
    monkeypatch.setattr(abba.source_profile_router, "capture_source_route_profile_context", capture)
    with pytest.raises(CaptureSentinel):
        abba.live_source_profile_context_observer(
            phase=phase, source=object(), labels=object(), metrics=("rank_ic",),
            requested_tile_size=8, policy=GPUExecutionPolicy())
    assert seen["metadata"] is metadata
    assert seen["metrics"] == ("rank_ic",)
    assert seen["request_fingerprint"] == "fingerprint"
    assert seen["requested_tile_size"] == 8
    assert seen["policy"] == GPUExecutionPolicy()


@pytest.mark.parametrize("field,value", [
    ("oom_retries", 1),
    ("execution_schedule_sha256", "0" * 64),
])
def test_receipt_zero_oom_or_schedule_drift_fails_closed(sample, field, value):
    def mutate(receipt, index, backend):
        if index == 0:
            receipt[field] = value
        return receipt
    with pytest.raises(ValueError):
        _run_with_oracle(sample, lambda *, bundle, **kwargs: sample[1],
                         runner_mutator=mutate)


def test_order_dependent_winner_drift_fails_closed(sample):
    def mutate(receipt, index, backend):
        receipt["seconds"] = {0: 0.1, 1: 0.2, 2: 0.1, 3: 0.2}[index]
        receipt["total_wall_seconds"] = receipt["seconds"]
        return receipt
    with pytest.raises(ValueError, match="winner"):
        _run_with_oracle(sample, lambda *, bundle, **kwargs: sample[1],
                         runner_mutator=mutate)

def test_invalid_progress_observer_fails_before_backend_run(sample):
    calls = []
    with pytest.raises(TypeError, match="progress_observer"):
        produce_source_route_profile_abba(
            run_backend_fn=lambda *args, **kwargs: calls.append(args),
            run_kwargs={}, oracle=lambda **kwargs: {}, progress_observer=object())
    assert calls == []


def test_progress_callback_error_stops_before_profile_qualification(sample):
    calls = []

    def mutate(receipt, index, backend):
        calls.append(backend)
        return receipt

    def fail(event):
        raise RuntimeError("observer stopped")

    with pytest.raises(RuntimeError, match="observer stopped"):
        _run_with_oracle(
            sample, lambda *, bundle, **kwargs: sample[1],
            runner_mutator=mutate, progress_observer=fail)
    assert calls == ["cpu"]


def test_progress_keeps_four_safe_events_before_pair_aggregation_failure(sample):
    events = []

    def reorder_winners(receipt, index, backend):
        receipt["seconds"] = {0: 0.1, 1: 0.2, 2: 0.1, 3: 0.2}[index]
        receipt["total_wall_seconds"] = receipt["seconds"]
        return receipt

    with pytest.raises(ValueError, match="winner"):
        _run_with_oracle(
            sample, lambda *, bundle, **kwargs: sample[1],
            runner_mutator=reorder_winners,
            progress_observer=lambda event: events.append(event))

    assert len(events) == 4
    assert all(set(event) == {
        "phase", "run_index", "backend_used", "seconds", "total_wall_seconds",
        "oom_retries", "actual_source_tile_size", "actual_gpu_factor_tile_size",
        "effective_max_tile_size", "oracle_report",
    } for event in events)
    assert [event["backend_used"] for event in events] == ["cpu", "cuda", "cuda", "cpu"]
    assert all(event["phase"] == "run_validated" for event in events)
    assert all("source_snapshot_id" not in event for event in events)
    assert all("pass" in event["oracle_report"] for event in events)


def test_retry_rejection_keeps_only_its_validated_progress_event(sample):
    events = []
    calls = []

    def mutate(receipt, index, backend):
        calls.append(backend)
        if index == 0:
            receipt["oom_retries"] = 1
        return receipt

    with pytest.raises(ValueError, match="zero OOM retries"):
        _run_with_oracle(
            sample, lambda *, bundle, **kwargs: sample[1],
            runner_mutator=mutate, progress_observer=lambda event: events.append(event))

    assert calls == ["cpu"]
    assert len(events) == 1
    assert events[0]["oom_retries"] == 1


def test_progress_callback_cannot_mutate_internal_oracle_qualification(sample):
    def tamper(event):
        event["oracle_report"]["pass"] = False
        event["oracle_report"]["metrics"]["rank_ic"]["pass"] = False

    result = _run_with_oracle(
        sample, lambda *, bundle, **kwargs: sample[1], progress_observer=tamper)
    assert len(result.records) == 2
    assert all(report["pass"] for report in result.oracle_reports)
