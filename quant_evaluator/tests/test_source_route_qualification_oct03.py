"""Mutation and independent-oracle tests for modular route qualification."""
from dataclasses import replace
import pytest

from quant_evaluator.runtime.source_route_qualification import (
    BackendMeasurement,
    CounterbalancedABRecord,
    SourceRouteContext,
    validate_source_route_qualification,
)


def _h(char):
    return char * 64


@pytest.fixture
def evidence():
    context = SourceRouteContext(
        request_content_sha256=_h("a"), source_identity_sha256=_h("b"),
        source_content_sha256=_h("c"), executable_source_sha256=_h("d"),
        runtime_fingerprint_sha256=_h("e"), package_fingerprint_sha256=_h("f"),
        thread_fingerprint_sha256=_h("1"), device_fingerprint_sha256=_h("2"),
        config_fingerprint_sha256=_h("3"), request_shape=(100, 40, 8),
        metric_ids=("rank_ic", "rank_ic_series"), requested_tile_size=8,
        expected_coverage_count=320, effective_tile_size=4,
    )

    def measurement(backend, seconds):
        return BackendMeasurement(backend, seconds, 0, _h("4"), True, 320, 320)

    records = (
        CounterbalancedABRecord(context, ("cpu", "cuda"),
                                 measurement("cpu", 12.0), measurement("cuda", 9.0)),
        CounterbalancedABRecord(context, ("cuda", "cpu"),
                                 measurement("cpu", 13.0), measurement("cuda", 8.5)),
    )
    return context, records


def test_accepts_bound_counterbalanced_pair_and_matches_independent_oracle(evidence):
    context, records = evidence
    qualified = validate_source_route_qualification(records, expected_context=context)
    # Independent oracle: compare all declared inputs and arithmetic directly.
    assert qualified.context == context
    assert qualified.winning_backend == "cuda"
    assert qualified.run_orders == (("cpu", "cuda"), ("cuda", "cpu"))
    assert qualified.timings_seconds == ((12.0, 9.0), (13.0, 8.5))
    assert qualified.correctness_digest_sha256 == _h("4")


def test_accepts_cpu_winner_consistently_when_cuda_is_slower(evidence):
    context, records = evidence
    cpu_wins = tuple(replace(r, cpu=replace(r.cpu, seconds=r.cpu.seconds - 5.0))
                     for r in records)
    qualified = validate_source_route_qualification(cpu_wins, expected_context=context)
    assert qualified.winning_backend == "cpu"


@pytest.mark.parametrize("malformed_context", [
    "bool_tile", "float_tile", "float_shape",
])
def test_validates_record_context_types_before_dataclass_equality(evidence, malformed_context):
    context, records = evidence
    actual_context = context
    expected = context
    if malformed_context == "bool_tile":
        expected = replace(context, requested_tile_size=1, effective_tile_size=1)
        actual_context = replace(expected, requested_tile_size=True)
    elif malformed_context == "float_tile":
        actual_context = replace(context, requested_tile_size=8.0)
    else:
        actual_context = replace(context, request_shape=(100, 40, 8.0))
    malformed_record = replace(records[0], context=actual_context)
    with pytest.raises(ValueError):
        validate_source_route_qualification((malformed_record, records[1]),
                                            expected_context=expected)


@pytest.mark.parametrize("field,value", [
    ("request_content_sha256", "not-a-sha"),
    ("source_identity_sha256", "A" * 64),
    ("source_content_sha256", ""),
    ("executable_source_sha256", "0" * 63),
    ("runtime_fingerprint_sha256", None),
    ("package_fingerprint_sha256", True),
    ("thread_fingerprint_sha256", "g" * 64),
    ("device_fingerprint_sha256", "z" * 64),
    ("config_fingerprint_sha256", "1" * 64 + "x"),
    ("request_shape", (100, True, 8)),
    ("requested_tile_size", True),
    ("effective_tile_size", 9),
    ("expected_coverage_count", True),
    ("metric_ids", ("rank_ic", "rank_ic")),
    ("metric_ids", (" rank_ic",)),
    ("request_shape", (100, 40)),
])
def test_rejects_invalid_or_stale_context_fields(evidence, field, value):
    context, records = evidence
    bad = replace(context, **{field: value})
    with pytest.raises(ValueError):
        validate_source_route_qualification(records, expected_context=bad)


@pytest.mark.parametrize("seconds", [True, 0, -1, float("nan"), float("inf"), 10**1000])
def test_rejects_boolean_nonpositive_or_nonfinite_timing(evidence, seconds):
    context, records = evidence
    rec = records[0]
    bad = replace(rec, cuda=replace(rec.cuda, seconds=seconds))
    with pytest.raises(ValueError):
        validate_source_route_qualification((bad, records[1]), expected_context=context)


@pytest.mark.parametrize("mutation", [
    "snapshot_only", "wrong_code", "runtime_drift", "device_drift", "config_drift",
    "same_order", "cuda_slower_first", "cuda_slower_second", "oom_bool", "oom_retry",
    "correctness_flag", "correctness_digest", "coverage_bool", "coverage_short",
    "coverage_cross_order", "winner_changes_with_order", "cpu_oom", "backend_bool",
    "cross_backend_correctness", "one_record", "list_records", "backend_label",
])
def test_rejects_unbound_malformed_or_nonqualifying_records(evidence, mutation):
    context, records = evidence
    first, second = records
    expected = context
    if mutation == "snapshot_only":
        expected = replace(context, source_content_sha256=_h("9"))
    elif mutation == "wrong_code":
        first = replace(first, context=replace(context, executable_source_sha256=_h("9")))
    elif mutation == "runtime_drift":
        second = replace(second, context=replace(context, runtime_fingerprint_sha256=_h("9")))
    elif mutation == "device_drift":
        second = replace(second, context=replace(context, device_fingerprint_sha256=_h("9")))
    elif mutation == "config_drift":
        first = replace(first, context=replace(context, config_fingerprint_sha256=_h("9")))
    elif mutation == "same_order":
        second = replace(second, execution_order=first.execution_order)
    elif mutation == "cuda_slower_first":
        first = replace(first, cuda=replace(first.cuda, seconds=14.0))
    elif mutation == "cuda_slower_second":
        second = replace(second, cuda=replace(second.cuda, seconds=14.0))
    elif mutation == "oom_bool":
        first = replace(first, cuda=replace(first.cuda, oom_count=False))
    elif mutation == "oom_retry":
        second = replace(second, cuda=replace(second.cuda, oom_count=1))
    elif mutation == "correctness_flag":
        first = replace(first, cpu=replace(first.cpu, correctness_validated=False))
    elif mutation == "correctness_digest":
        second = replace(second, cuda=replace(second.cuda, correctness_digest_sha256=_h("8")))
    elif mutation == "coverage_bool":
        first = replace(first, cpu=replace(first.cpu, coverage_expected=True))
    elif mutation == "coverage_short":
        second = replace(second, cuda=replace(second.cuda, coverage_observed=319))
    elif mutation == "coverage_cross_order":
        second = replace(second,
                         cpu=replace(second.cpu, coverage_expected=319, coverage_observed=319),
                         cuda=replace(second.cuda, coverage_expected=319, coverage_observed=319))
    elif mutation == "winner_changes_with_order":
        second = replace(second, cpu=replace(second.cpu, seconds=7.0))
    elif mutation == "cpu_oom":
        first = replace(first, cpu=replace(first.cpu, oom_count=1))
    elif mutation == "backend_bool":
        first = replace(first, cpu=replace(first.cpu, backend=True))
    elif mutation == "cross_backend_correctness":
        first = replace(first, cuda=replace(first.cuda, correctness_digest_sha256=_h("7")))
    elif mutation == "one_record":
        with pytest.raises(ValueError):
            validate_source_route_qualification((first,), expected_context=context)
        return
    elif mutation == "list_records":
        with pytest.raises(ValueError):
            validate_source_route_qualification([first, second], expected_context=context)
        return
    elif mutation == "backend_label":
        first = replace(first, cpu=replace(first.cpu, backend="cuda"))
    with pytest.raises(ValueError):
        validate_source_route_qualification((first, second), expected_context=expected)


def test_finite_helper_handles_huge_integer_without_leaking_overflow(evidence):
    context, records = evidence
    record = replace(records[0], cpu=replace(records[0].cpu, seconds=10**1000))
    with pytest.raises(ValueError):
        validate_source_route_qualification((record, records[1]), expected_context=context)
