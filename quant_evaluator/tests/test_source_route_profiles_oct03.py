"""Contract tests for per-backend-width counterbalanced route profiles v2."""
from dataclasses import replace
import math

import pytest

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement,
    CounterbalancedRouteProfileRecord,
    MetricComparisonReceipt,
    MetricOutputReceipt,
    SourceRouteProfileContext,
    route_profile_execution_config_sha256,
    validate_source_route_profile_qualification,
)


def _h(char):
    return char * 64


@pytest.fixture(params=[4, 2])
def evidence(request):
    gpu_width = request.param
    context = SourceRouteProfileContext(
        request_content_sha256=_h("a"), source_identity_sha256=_h("b"),
        source_content_sha256=_h("c"), executable_source_sha256=_h("d"),
        runtime_fingerprint_sha256=_h("e"), package_fingerprint_sha256=_h("f"),
        thread_fingerprint_sha256=_h("1"), device_fingerprint_sha256=_h("2"),
        common_config_fingerprint_sha256=_h("3"), request_shape=(10, 20, 48),
        metric_ids=("rank_ic", "rank_ic_series"),
        metric_coverage=(("rank_ic", 48), ("rank_ic_series", 480)),
        metric_error_tolerances=(("rank_ic", 1e-10), ("rank_ic_series", 1e-10)),
        expected_coverage_count=528, requested_tile_size=16,
        timing_scope="evaluate_factor_source_batch_wall_v1",
        live_source_admitted_max_tile_size=5,
        gpu_admitted_max_tile_size=gpu_width,
    )
    def profile(backend, width, seconds, digest):
        source_tile_size = 5 if backend == "cpu" else gpu_width
        source_ranges = tuple((start, min(start + source_tile_size, 48))
                              for start in range(0, 48, source_tile_size))
        compute_ranges = (source_ranges if backend == "cpu" else tuple(
            (start, min(start + width, end))
            for source_start, source_end in source_ranges
            for start in range(source_start, source_end, width)
            for end in (source_end,)))
        outputs = (
            MetricOutputReceipt("rank_ic", _h(digest), _h("8"), _h("9"), 48, 48),
            MetricOutputReceipt("rank_ic_series", _h(digest), _h("a"), _h("b"), 480, 480),
        )
        schedule_sha = stable_content_hex(
            tag="RouteProfileExecutionSchedule.v2",
            fields={"scope": "zero_oom_source_equals_compute_v1",
                    "source_ranges": source_ranges, "compute_ranges": compute_ranges},
        )
        return BackendRouteProfileMeasurement(
            backend=backend, seconds=seconds, oom_count=0, actual_tile_size=width,
            source_tile_size=source_tile_size, source_ranges=source_ranges,
            compute_ranges=compute_ranges,
            execution_config_sha256=route_profile_execution_config_sha256(
                context, backend, source_tile_size),
            execution_schedule_sha256=schedule_sha,
            execution_schedule_scope="zero_oom_source_equals_compute_v1",
            correctness_validated=True, coverage_expected=528,
            coverage_observed=528, outputs=outputs,
        )

    comparison = (
        MetricComparisonReceipt(
            metric_id="rank_ic", cpu_values_sha256=_h("4"), cuda_values_sha256=_h("5"),
            cpu_finite_mask_sha256=_h("8"), cuda_finite_mask_sha256=_h("8"),
            cpu_observation_counts_sha256=_h("9"), cuda_observation_counts_sha256=_h("9"),
            compared_value_count=48, finite_mask_equal=True, observation_counts_equal=True,
            max_abs_error=1e-12, max_abs_error_tolerance=1e-10,
            evidence_sha256=_h("6")),
        MetricComparisonReceipt(
            metric_id="rank_ic_series", cpu_values_sha256=_h("4"), cuda_values_sha256=_h("5"),
            cpu_finite_mask_sha256=_h("a"), cuda_finite_mask_sha256=_h("a"),
            cpu_observation_counts_sha256=_h("b"), cuda_observation_counts_sha256=_h("b"),
            compared_value_count=480, finite_mask_equal=True, observation_counts_equal=True,
            max_abs_error=1e-12, max_abs_error_tolerance=1e-10,
            evidence_sha256=_h("7")),
    )
    records = (
        CounterbalancedRouteProfileRecord(
            context, ("cpu", "cuda"), profile("cpu", 5, 4.0, "4"),
            profile("cuda", gpu_width, 5.0, "5"), comparison),
        CounterbalancedRouteProfileRecord(
            context, ("cuda", "cpu"), profile("cpu", 5, 4.2, "4"),
            profile("cuda", gpu_width, 5.1, "5"), comparison),
    )
    return context, records


def test_accepts_cap16_request_with_distinct_cpu5_cuda_gpu_capped_profiles(evidence):
    context, records = evidence
    result = validate_source_route_profile_qualification(records, expected_context=context)
    assert result.winning_backend == "cpu"
    gpu_width = records[0].cuda.source_tile_size
    assert result.source_tile_sizes == (("cpu", 5), ("cuda", gpu_width))
    assert result.actual_tile_sizes == (("cpu", 5), ("cuda", gpu_width))
    assert result.timings_seconds == ((4.0, 5.0), (4.2, 5.1))
    assert result.mean_whole_request_seconds == (("cpu", 4.1), ("cuda", 5.05))


def test_whole_request_mean_is_finite_for_extreme_finite_timings(evidence):
    context, records = evidence
    first = replace(records[0], cpu=replace(records[0].cpu, seconds=1e308),
                    cuda=replace(records[0].cuda, seconds=1.5e308))
    second = replace(records[1], cpu=replace(records[1].cpu, seconds=1e308),
                     cuda=replace(records[1].cuda, seconds=1.5e308))
    result = validate_source_route_profile_qualification((first, second),
                                                         expected_context=context)
    assert result.winning_backend == "cpu"
    assert math.isfinite(dict(result.mean_whole_request_seconds)["cpu"])
    assert math.isfinite(dict(result.mean_whole_request_seconds)["cuda"])
    assert dict(result.mean_whole_request_seconds) == {"cpu": 1e308, "cuda": 1.5e308}


def test_whole_request_mean_remains_positive_at_minimum_subnormal(evidence):
    context, records = evidence
    first = replace(records[0], cpu=replace(records[0].cpu, seconds=5e-324),
                    cuda=replace(records[0].cuda, seconds=1e-323))
    second = replace(records[1], cpu=replace(records[1].cpu, seconds=5e-324),
                     cuda=replace(records[1].cuda, seconds=1e-323))
    result = validate_source_route_profile_qualification((first, second),
                                                         expected_context=context)
    assert result.winning_backend == "cpu"
    assert dict(result.mean_whole_request_seconds) == {"cpu": 5e-324, "cuda": 1e-323}


def test_whole_request_mean_preserves_a_near_tie(evidence):
    context, records = evidence
    slower_cuda = math.nextafter(1.0, 2.0)
    first = replace(records[0], cpu=replace(records[0].cpu, seconds=1.0),
                    cuda=replace(records[0].cuda, seconds=slower_cuda))
    second = replace(records[1], cpu=replace(records[1].cpu, seconds=1.0),
                     cuda=replace(records[1].cuda, seconds=slower_cuda))
    result = validate_source_route_profile_qualification((first, second),
                                                         expected_context=context)
    means = dict(result.mean_whole_request_seconds)
    assert result.winning_backend == "cpu"
    assert means["cpu"] == 1.0
    assert means["cuda"] == slower_cuda


@pytest.mark.parametrize("width", [True, 2.0, 5])
def test_rejects_malformed_or_over_admission_backend_width(evidence, width):
    context, records = evidence
    first = replace(records[0], cuda=replace(records[0].cuda, actual_tile_size=width))
    with pytest.raises(ValueError):
        validate_source_route_profile_qualification((first, records[1]),
                                                    expected_context=context)


def test_rejects_source_profile_above_live_memory_admission(evidence):
    context, records = evidence
    first = replace(records[0], cpu=replace(records[0].cpu, source_tile_size=6))
    with pytest.raises(ValueError, match="live source-admitted ceiling"):
        validate_source_route_profile_qualification((first, records[1]),
                                                    expected_context=context)


def test_rejects_backend_configuration_drift_between_orders(evidence):
    context, records = evidence
    changed = replace(records[1], cuda=replace(records[1].cuda,
                                               execution_config_sha256=_h("0")))
    with pytest.raises(ValueError, match="execution configuration differs"):
        validate_source_route_profile_qualification((records[0], changed),
                                                    expected_context=context)


def test_rejects_compute_schedule_that_crosses_source_tile_boundary(evidence):
    context, records = evidence
    bad_ranges = tuple((start, start + 1) for start in range(48))
    bad_digest = stable_content_hex(
        tag="RouteProfileExecutionSchedule.v2",
        fields={"scope": "zero_oom_source_equals_compute_v1",
                "source_ranges": records[0].cuda.source_ranges,
                "compute_ranges": bad_ranges},
    )
    bad_profile = replace(records[0].cuda, compute_ranges=bad_ranges,
                          execution_schedule_sha256=bad_digest)
    changed = replace(records[0], cuda=bad_profile)
    with pytest.raises(ValueError, match="tile schedule"):
        validate_source_route_profile_qualification((changed, records[1]),
                                                    expected_context=context)


@pytest.mark.parametrize("field,value", [
    ("oom_count", 1), ("oom_count", False),
    ("coverage_observed", 527), ("correctness_validated", False),
])
def test_rejects_backend_run_without_exact_complete_zero_oom_receipt(evidence, field, value):
    context, records = evidence
    changed = replace(records[0], cuda=replace(records[0].cuda, **{field: value}))
    with pytest.raises(ValueError):
        validate_source_route_profile_qualification((changed, records[1]),
                                                    expected_context=context)


@pytest.mark.parametrize("field,value", [
    ("finite_mask_equal", False), ("observation_counts_equal", False),
    ("max_abs_error", 1e-5), ("max_abs_error_tolerance", 1.0),
    ("evidence_sha256", "bad"),
])
def test_rejects_invalid_independent_metric_comparison_evidence(evidence, field, value):
    context, records = evidence
    changed_comparison = (replace(records[0].comparison_evidence[0], **{field: value}),
                          records[0].comparison_evidence[1])
    changed = replace(records[0], comparison_evidence=changed_comparison)
    with pytest.raises(ValueError):
        validate_source_route_profile_qualification((changed, records[1]),
                                                    expected_context=context)


def test_rejects_comparison_receipt_not_bound_to_backend_output_hashes(evidence):
    context, records = evidence
    first_receipt = replace(records[0].comparison_evidence[0], cpu_values_sha256=_h("0"))
    changed = replace(records[0], comparison_evidence=(first_receipt,
                                                       records[0].comparison_evidence[1]))
    with pytest.raises(ValueError, match="comparison differs"):
        validate_source_route_profile_qualification((changed, records[1]),
                                                    expected_context=context)


def test_rejects_same_order_or_winner_flip_between_counterbalanced_runs(evidence):
    context, records = evidence
    same_order = replace(records[1], execution_order=records[0].execution_order)
    with pytest.raises(ValueError):
        validate_source_route_profile_qualification((records[0], same_order),
                                                    expected_context=context)
    slower_cpu = replace(records[1], cpu=replace(records[1].cpu, seconds=6.0))
    with pytest.raises(ValueError, match="winner changes"):
        validate_source_route_profile_qualification((records[0], slower_cpu),
                                                    expected_context=context)


def test_rejects_stale_context_and_incomplete_metric_coverage(evidence):
    context, records = evidence
    stale = replace(records[1], context=replace(context, source_content_sha256=_h("0")))
    with pytest.raises(ValueError):
        validate_source_route_profile_qualification((records[0], stale),
                                                    expected_context=context)
    short_outputs = replace(records[0].cpu, outputs=records[0].cpu.outputs[:1])
    short = replace(records[0], cpu=short_outputs)
    with pytest.raises(ValueError):
        validate_source_route_profile_qualification((short, records[1]),
                                                    expected_context=context)


def test_requires_exact_metric_coverage_map_and_wall_timing_scope(evidence):
    context, records = evidence
    wrong_map = replace(context, metric_coverage=(("rank_ic", 47),
                                                  ("rank_ic_series", 481)))
    wrong_records = tuple(replace(record, context=wrong_map) for record in records)
    with pytest.raises(ValueError, match="per-metric output coverage"):
        validate_source_route_profile_qualification(wrong_records, expected_context=wrong_map)
    wrong_scope = replace(context, timing_scope="gpu_kernel_only")
    with pytest.raises(ValueError, match="timing scope"):
        validate_source_route_profile_qualification(records, expected_context=wrong_scope)
