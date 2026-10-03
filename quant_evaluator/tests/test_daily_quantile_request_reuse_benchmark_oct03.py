"""Behavioral smoke for the bounded daily quantile A/B harness."""
import pytest
from types import SimpleNamespace
import numpy as np
from quant_evaluator.scripts.benchmark_daily_quantile_request_reuse_oct03 import benchmark


def test_tiny_evaluator_reuse_ab_is_equivalent_and_reduces_calls():
    report = benchmark(time_count=24, asset_count=100, factor_count=2,
                       rounds=3, seed=82110, cpu_kernel="numpy")
    assert report["shape"] == [24, 100, 2]
    assert report["baseline"]["kernel_calls_per_evaluate"] == 7
    assert report["optimized"]["kernel_calls_per_evaluate"] == 1
    assert report["baseline"]["outputs_match_optimized"] is True
    assert len(report["samples"]["baseline_seconds"]) == 6
    assert len(report["samples"]["optimized_seconds"]) == 6
    assert report["estimated_peak_bytes"] <= report["limits"]["estimated_peak_bytes"]
    assert report["process_peak_rss_bytes"] > 0
    assert len(report["sources"]["baseline_evaluator_sha256"]) == 64
    assert len(report["sources"]["daily_quantile_cache_sha256"]) == 64


def test_benchmark_harness_rejects_unbounded_shape():
    from quant_evaluator.scripts.benchmark_daily_quantile_request_reuse_oct03 import _check_shape
    with pytest.raises(ValueError, match="40M-cell cap"):
        _check_shape(1000, 5461, 48)
    with pytest.raises(ValueError, match="positive builtin integer"):
        _check_shape(1, 100, True)


def test_benchmark_comparison_requires_exact_counts_masks_and_observation_counts():
    from quant_evaluator.scripts.benchmark_daily_quantile_request_reuse_oct03 import (
        METRICS, _compare,
    )
    def bundle(count=10_000_000_000, mask=True, obs=10_000_000_000, value=0.25):
        return SimpleNamespace(artifacts={
            metric: SimpleNamespace(values=np.array([value]), counts=np.array([count]),
                                    valid_mask=np.array([mask]),
                                    provenance={"observation_counts": (obs,)})
            for metric in METRICS
        })
    _compare(bundle(), bundle(value=0.25 + 1e-13))
    for kwargs in ({"count": 10_000_000_001}, {"mask": False}, {"obs": 10_000_000_001}):
        with pytest.raises(AssertionError):
            _compare(bundle(), bundle(**kwargs))


def test_benchmark_round_count_is_bounded_before_evaluation():
    from quant_evaluator.scripts.benchmark_daily_quantile_request_reuse_oct03 import benchmark
    for rounds in (2, 21):
        with pytest.raises(ValueError, match="3 through 20"):
            benchmark(24, 100, 2, rounds=rounds)


@pytest.mark.parametrize("mutated_call", [3, 16])
def test_every_warmup_and_timed_result_is_checked_against_stable_reference(
        monkeypatch, mutated_call):
    import quant_evaluator.scripts.benchmark_daily_quantile_request_reuse_oct03 as harness
    original = harness._run_one
    call_number = 0

    def mutate_one_result(*args, **kwargs):
        nonlocal call_number
        call_number += 1
        result, elapsed, calls = original(*args, **kwargs)
        if call_number == mutated_call:
            from dataclasses import replace
            artifact = result.artifacts["quantile_spread"]
            result.artifacts["quantile_spread"] = replace(
                artifact, values=np.asarray(artifact.values) + 1e-4)
        return result, elapsed, calls

    monkeypatch.setattr(harness, "_run_one", mutate_one_result)
    with pytest.raises(AssertionError, match="quantile_spread.values"):
        harness.benchmark(24, 100, 2, rounds=3, seed=82111)
