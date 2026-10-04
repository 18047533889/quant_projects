from __future__ import annotations

from dataclasses import replace
import gc
import json
import re
import weakref

import numpy as np
import pytest

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.scripts.source_profile_abba import IndependentOracleMismatchError
from test_source_profile_measurement_oct04 import sample as sample_fixture


def _api():
    try:
        from quant_evaluator.scripts.source_profile_randomized_runner import (
            produce_source_route_profile_randomized_trials,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "quant_evaluator.scripts.source_profile_randomized_runner":
            raise
        pytest.fail("required behavior is missing: randomized source-profile runner")
    return produce_source_route_profile_randomized_trials


def _setup(sample, *, observer=None, mutate_receipt=None, oracle=None,
           comparator=None, bundle_mutator=None, on_call=None):
    context, cpu, cuda, cpu_receipt, cuda_receipt, _ = sample
    calls = []
    bundles = []

    def stable_observer(**_kwargs):
        return context

    def run_backend(backend, *, context_observer, **kwargs):
        index = len(calls)
        calls.append(backend)
        if on_call is not None:
            on_call(index)
        before = context_observer(phase="before", **kwargs)
        after = context_observer(phase="after", **kwargs)
        template = cpu if backend == "cpu" else cuda
        bundle = BatchEvaluationBundle(
            factor_ids=template.factor_ids,
            label_id=template.label_id,
            scalar_metrics={key: value.copy() for key, value in template.scalar_metrics.items()},
            series_metrics={key: value.copy() for key, value in template.series_metrics.items()},
            vector_metrics={}, metadata=dict(template.metadata),
            observation_counts={key: value.copy()
                                for key, value in template.observation_counts.items()},
        )
        if bundle_mutator is not None:
            bundle_mutator(bundle, index, backend)
        bundles.append(weakref.ref(bundle))
        receipt_template = cpu_receipt if backend == "cpu" else cuda_receipt
        receipt = dict(receipt_template, context_before=before, context_after=after)
        if mutate_receipt is not None:
            receipt = mutate_receipt(receipt, index, backend)
        return bundle, receipt

    def oracle_fn(*, bundle, backend, **_kwargs):
        if oracle is not None:
            return oracle(bundle=bundle, backend=backend)
        # The CPU fixture is the independent expected result for both arms.
        return cpu

    result = _api()(
        oracle=oracle_fn, run_kwargs={"request": "fixture"}, seed=20261004,
        pairs=3, run_backend_fn=run_backend,
        context_observer=observer or stable_observer,
        comparison_fn=comparator,
    )
    return result, calls, bundles


def test_runs_three_seeded_pairs_with_live_oracle_measurement_and_bounded_progress(sample_fixture):
    context, *_ = sample_fixture
    calls, events, bundle_refs = [], [], []

    def observer(**_kwargs):
        return context

    def on_call(index):
        calls.append(index)
        if index == 2:
            gc.collect()
            assert all(ref() is None for ref in bundle_refs[:2])

    # Keep this test's run seam local so it can observe collection at the next pair.
    _, cpu, cuda, cpu_receipt, cuda_receipt, _ = sample_fixture

    def run_backend(backend, *, context_observer, **kwargs):
        index = len(calls)
        on_call(index)
        before = context_observer(phase="before", **kwargs)
        after = context_observer(phase="after", **kwargs)
        template = cpu if backend == "cpu" else cuda
        bundle = BatchEvaluationBundle(
            factor_ids=template.factor_ids, label_id=template.label_id,
            scalar_metrics={k: v.copy() for k, v in template.scalar_metrics.items()},
            series_metrics={k: v.copy() for k, v in template.series_metrics.items()},
            metadata=dict(template.metadata),
            observation_counts={k: v.copy() for k, v in template.observation_counts.items()},
        )
        bundle_refs.append(weakref.ref(bundle))
        base = cpu_receipt if backend == "cpu" else cuda_receipt
        seconds = float(1.0 if backend == "cpu" else 2.0)
        receipt = dict(base, seconds=seconds, total_wall_seconds=seconds,
                       context_before=before, context_after=after)
        return bundle, receipt

    result = _api()(
        oracle=lambda *, bundle, **_kwargs: cpu,
        run_kwargs={"request": "fixture"}, seed=20261004, pairs=3,
        run_backend_fn=run_backend, context_observer=observer,
        progress_observer=events.append,
    )

    assert calls == list(range(6))
    assert len(result["schedule"]) == len(result["trials"]) == 3
    assert result["aggregate"]["aggregate_pass"] is True
    assert result["aggregate"]["winner"] == "cpu"
    assert all(trial["timing_scope"] == "evaluate_factor_source_batch_wall_v1"
               for trial in result["trials"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", trial["context_fingerprint"])
               for trial in result["trials"])
    assert len(events) == 6
    assert all(len(json.dumps(event, allow_nan=False)) < 2048 for event in events)
    assert not any(isinstance(value, np.ndarray) for event in events
                   for value in event.values())
    assert not any(key in event for event in events
                   for key in ("receipt", "bundle", "source_snapshot_id", "url"))
    assert all(ref() is None for ref in bundle_refs)
    assert not any(key in result for key in ("records", "auto_qualified", "profile"))


@pytest.mark.parametrize(("drift_calls", "message"), [(2, "context changed"),
                                                       ((3, 4), "do not share")])
def test_live_context_drift_fails_closed(sample_fixture, drift_calls, message):
    context, *_ = sample_fixture
    seen = 0

    def observer(**_kwargs):
        nonlocal seen
        seen += 1
        changed_at = (drift_calls,) if type(drift_calls) is int else drift_calls
        if seen in changed_at:
            return replace(context, source_content_sha256="9" * 64)
        return context

    with pytest.raises(ValueError, match=message):
        _setup(sample_fixture, observer=observer)


def test_wrong_actual_backend_is_rejected_before_it_can_be_aggregated(sample_fixture):
    def mutate(receipt, index, _backend):
        if _backend == "cuda_strict":
            receipt["backend_used"] = "cpu"
        return receipt

    with pytest.raises(ValueError, match="different backend"):
        _setup(sample_fixture, mutate_receipt=mutate)


def test_oracle_mismatch_propagates_and_never_returns_aggregate(sample_fixture):
    with pytest.raises(IndependentOracleMismatchError):
        _setup(sample_fixture, oracle=lambda **_kwargs: {})


def test_pairwise_parity_failure_propagates_and_never_returns_aggregate(sample_fixture):
    _, cpu, *_ = sample_fixture

    def change_cuda(bundle, _index, backend):
        if backend == "cuda_strict":
            bundle.scalar_metrics["rank_ic"][0] += 0.1

    def expected(*, bundle, backend):
        return cpu if backend == "cpu" else bundle

    with pytest.raises(ValueError, match="parity"):
        _setup(sample_fixture, oracle=expected, bundle_mutator=change_cuda)
