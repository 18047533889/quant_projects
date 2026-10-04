from __future__ import annotations

from statistics import median

import pytest

try:
    from quant_evaluator.scripts.source_profile_paired_trials import (
        aggregate_paired_trials,
        build_randomized_paired_schedule,
    )
except ModuleNotFoundError as exc:
    if exc.name != "quant_evaluator.scripts.source_profile_paired_trials":
        raise
    _missing_module = exc.name

    def _not_implemented(*_args, **_kwargs):
        pytest.fail(f"required behavior is missing: {_missing_module}")

    aggregate_paired_trials = build_randomized_paired_schedule = _not_implemented


def _trial(order, cpu_s, cuda_s, *, context="f" * 64, oracle=True, parity=True,
           scope="evaluate_factor_source_batch_wall_v1"):
    return {
        "execution_order": order,
        "context_fingerprint": context,
        "timing_scope": scope,
        "cpu_seconds": cpu_s,
        "cuda_seconds": cuda_s,
        "cpu_oracle_pass": oracle,
        "cuda_oracle_pass": oracle,
        "parity_pass": parity,
    }


def test_schedule_is_seeded_bounded_and_counterbalanced():
    schedule = build_randomized_paired_schedule(seed=20261004, pairs=7)

    assert schedule == build_randomized_paired_schedule(seed=20261004, pairs=7)
    assert len(schedule) == 7
    assert all(sorted(pair) == ["cpu", "cuda_strict"] for pair in schedule)
    cpu_first = sum(pair[0] == "cpu" for pair in schedule)
    assert abs(cpu_first - (len(schedule) - cpu_first)) <= 1


@pytest.mark.parametrize("seed", [True, False, 1.0, "1"])
def test_schedule_rejects_non_builtin_int_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        build_randomized_paired_schedule(seed=seed, pairs=3)


@pytest.mark.parametrize("pairs", [True, 2, 17, 3.0, "3"])
def test_schedule_rejects_unbounded_or_invalid_pair_count(pairs):
    with pytest.raises(ValueError, match="pairs"):
        build_randomized_paired_schedule(seed=1, pairs=pairs)


def test_aggregate_requires_three_or_more_trials():
    with pytest.raises(ValueError, match="minimum_pairs"):
        aggregate_paired_trials([_trial(("cpu", "cuda_strict"), 1, 2)] * 2)


def test_aggregate_requires_consistent_context_and_full_api_wall_scope():
    context_drift = [
        _trial(("cpu", "cuda_strict"), 1, 2),
        _trial(("cuda_strict", "cpu"), 1, 2, context="e" * 64),
        _trial(("cpu", "cuda_strict"), 1, 2),
    ]
    with pytest.raises(ValueError, match="context"):
        aggregate_paired_trials(context_drift)

    microbenchmark = [_trial(("cpu", "cuda_strict"), 1, 2,
                             scope="kernel_only") for _ in range(3)]
    with pytest.raises(ValueError, match="timing_scope"):
        aggregate_paired_trials(microbenchmark)


@pytest.mark.parametrize("order", [("cpu", "cpu"), ("cuda", "cpu"), ("cpu", "cuda")])
def test_aggregate_rejects_invalid_api_execution_order(order):
    trials = [_trial(order, 1, 2) for _ in range(3)]
    with pytest.raises(ValueError, match="execution_order"):
        aggregate_paired_trials(trials)


@pytest.mark.parametrize("bad_time", [True, 0, -1, float("inf"), float("nan"), "1"])
def test_aggregate_rejects_nonpositive_nonfinite_or_wrong_type_timing(bad_time):
    trials = [_trial(("cpu", "cuda_strict"), bad_time, 2) for _ in range(3)]
    with pytest.raises(ValueError, match="seconds"):
        aggregate_paired_trials(trials)


def test_any_oracle_failure_or_parity_failure_blocks_aggregate_winner():
    oracle_failure = [
        _trial(("cpu", "cuda_strict"), 1, 2),
        _trial(("cuda_strict", "cpu"), 1, 2, oracle=False),
        _trial(("cpu", "cuda_strict"), 1, 2),
    ]
    result = aggregate_paired_trials(oracle_failure)
    assert result["aggregate_pass"] is False
    assert result["winner"] is None

    parity_failure = [
        _trial(("cpu", "cuda_strict"), 1, 2),
        _trial(("cuda_strict", "cpu"), 1, 2, parity=False),
        _trial(("cpu", "cuda_strict"), 1, 2),
    ]
    result = aggregate_paired_trials(parity_failure)
    assert result["aggregate_pass"] is False
    assert result["winner"] is None


def test_winner_requires_strict_paired_majority_and_global_median_same_direction():
    # CUDA wins two pairs 2:1, but its global median runtime is much slower.
    trials = [
        _trial(("cpu", "cuda_strict"), 100, 99),
        _trial(("cuda_strict", "cpu"), 1, 0.9),
        _trial(("cpu", "cuda_strict"), 2, 200),
    ]
    result = aggregate_paired_trials(trials)
    assert result["aggregate_pass"] is True
    assert result["pair_wins"] == {"cpu": 1, "cuda": 2, "tie": 0}
    assert result["median_seconds"] == {"cpu": median([100, 1, 2]),
                                        "cuda": median([99, 0.9, 200])}
    assert result["median_seconds"]["cpu"] < result["median_seconds"]["cuda"]
    assert result["winner"] is None

    cpu_wins = [
        _trial(("cpu", "cuda_strict"), 8, 10),
        _trial(("cuda_strict", "cpu"), 7, 10),
        _trial(("cpu", "cuda_strict"), 9, 10),
    ]
    assert aggregate_paired_trials(cpu_wins)["winner"] == "cpu"


def test_ties_or_no_strict_majority_do_not_infer_winner():
    tied_median = [
        _trial(("cpu", "cuda_strict"), 1, 1),
        _trial(("cuda_strict", "cpu"), 1, 1),
        _trial(("cpu", "cuda_strict"), 1, 1),
    ]
    result = aggregate_paired_trials(tied_median)
    assert result["aggregate_pass"] is True
    assert result["winner"] is None

    no_majority = [
        _trial(("cpu", "cuda_strict"), 1, 2),
        _trial(("cuda_strict", "cpu"), 2, 1),
        _trial(("cpu", "cuda_strict"), 1, 1),
    ]
    assert aggregate_paired_trials(no_majority)["winner"] is None


@pytest.mark.parametrize("context", [None, ""])
def test_aggregate_rejects_missing_or_empty_context_fingerprint(context):
    trials = [_trial(("cpu", "cuda_strict"), 1, 2, context=context)] * 3
    with pytest.raises(ValueError, match="context"):
        aggregate_paired_trials(trials)


def test_aggregate_rejects_malformed_oracle_flags():
    bad_oracle = [_trial(("cpu", "cuda_strict"), 1, 2, oracle=1)] * 3
    with pytest.raises(ValueError, match="oracle"):
        aggregate_paired_trials(bad_oracle)


@pytest.mark.parametrize("cpu,cuda", [(1.7e308, 1.79e308), (5e-324, 1e-323)])
def test_even_medians_preserve_finite_positive_extreme_timings(cpu, cuda):
    # Adding two middle values can overflow; halving each can underflow.
    import json
    import math
    result = aggregate_paired_trials([
        _trial(("cpu", "cuda_strict"), cpu, cuda),
        _trial(("cuda_strict", "cpu"), cpu, cuda),
    ] * 2)
    assert result["median_seconds"] == {"cpu": cpu, "cuda": cuda}
    assert all(math.isfinite(v) and v > 0 for v in result["median_seconds"].values())
    assert result["winner"] == "cpu"
    json.dumps(result, allow_nan=False)


def test_aggregate_rejects_huge_integer_timing_with_contract_error():
    with pytest.raises(ValueError, match="seconds"):
        aggregate_paired_trials([_trial(("cpu", "cuda_strict"), 10**1000, 2)] * 3)


def test_aggregate_rejects_unhashable_order_with_contract_error():
    with pytest.raises(ValueError, match="execution_order"):
        aggregate_paired_trials([_trial(([], "cpu"), 1, 2)] * 3)


def test_aggregate_accesses_only_bounded_sequence_positions():
    # A Sequence may override iteration to produce unbounded records.
    from collections.abc import Sequence

    class HostileIterator(Sequence):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if not 0 <= index < 3:
                raise AssertionError("read outside declared sequence bound")
            return _trial(("cpu", "cuda_strict"), 1, 2)

        def __iter__(self):
            raise AssertionError("unbounded iterator must not be used")

    result = aggregate_paired_trials(HostileIterator())
    assert result["pair_count"] == 3
    assert result["winner"] == "cpu"
