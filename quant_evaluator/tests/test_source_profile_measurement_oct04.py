"""Focused tests for adapting real benchmark receipts to source-profile v2."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_route_profiles import (
    SourceRouteProfileContext, validate_source_route_profile_qualification,
)
from quant_evaluator.scripts.source_execution_receipt import (
    validate_source_execution_receipt,
)
from quant_evaluator.scripts.source_profile_measurement import (
    build_backend_profile_measurement,
    build_counterbalanced_profile_record,
)


def _h(char: str) -> str:
    return char * 64


@pytest.fixture
def sample():
    # These deterministic context identifiers are test scaffolding only. The
    # production helper accepts a context; it never manufactures one.
    context = SourceRouteProfileContext(
        request_content_sha256=_h("a"), source_identity_sha256=_h("b"),
        source_content_sha256=_h("c"), executable_source_sha256=_h("d"),
        runtime_fingerprint_sha256=_h("e"), package_fingerprint_sha256=_h("f"),
        thread_fingerprint_sha256=_h("1"), device_fingerprint_sha256=_h("2"),
        common_config_fingerprint_sha256=_h("3"), request_shape=(3, 4, 5),
        metric_ids=("rank_ic", "rank_ic_series"),
        expected_coverage_count=20, requested_tile_size=16,
        timing_scope="evaluate_factor_source_batch_wall_v1",
        metric_coverage=(("rank_ic", 5), ("rank_ic_series", 15)),
        metric_error_tolerances=(("rank_ic", 1e-10),
                                 ("rank_ic_series", 1e-10)),
        live_source_admitted_max_tile_size=5, gpu_admitted_max_tile_size=4,
    )
    factor_ids = tuple(f"factor-{i}" for i in range(5))
    scalar_cpu = np.array([0.1, np.nan, 0.4, np.inf, -0.2], dtype=np.float64)
    scalar_cuda = scalar_cpu.copy()
    scalar_cuda[0] += 1e-12
    series_cpu = np.arange(15, dtype=np.float64).reshape(3, 5) / 10
    series_cpu[1, 2] = np.nan
    series_cpu[2, 3] = -np.inf
    series_cuda = series_cpu.copy()
    series_cuda[0, 1] += 1e-12
    counts = {
        "rank_ic": np.array([4, 3, 4, 2, 4], dtype=np.int64),
        "rank_ic_series": np.array([4, 3, 4, 2, 4], dtype=np.int64),
    }

    def bundle(backend, scalar, series):
        return BatchEvaluationBundle(
            factor_ids=factor_ids, label_id="label",
            scalar_metrics={"rank_ic": scalar},
            series_metrics={"rank_ic_series": series},
            metadata={"backend_used": backend,
                      "source_request_fingerprint": context.request_content_sha256},
            observation_counts={k: v.copy() for k, v in counts.items()},
        )

    cpu, cuda = bundle("cpu", scalar_cpu, series_cpu), bundle(
        "cuda", scalar_cuda, series_cuda)

    def ranges(width):
        return [(i, min(i + width, len(factor_ids)))
                for i in range(0, len(factor_ids), width)]

    def run_receipt(bundle_obj, backend, width, seconds):
        schedule = tuple(tuple(item) for item in ranges(width))
        v1_digest = stable_content_hex(
            tag="SourceExecutionSchedule.v1",
            fields={"backend": backend, "factor_count": len(factor_ids),
                    "source_width": width, "source_ranges": schedule,
                    "compute_ranges": schedule, "oom_retries": 0},
        )
        ids_sha = hashlib.sha256(json.dumps(
            list(factor_ids), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        return {
            "backend_used": backend,
            "source_request_fingerprint": context.request_content_sha256,
            "factor_ids_sha256": ids_sha,
            "timing_scope": "evaluate_factor_source_batch_wall_v1",
            "seconds": seconds, "total_wall_seconds": seconds,
            "oom_retries": 0, "actual_source_tile_size": width,
            "actual_gpu_factor_tile_size": width if backend == "cuda" else None,
            "effective_max_tile_size": 5 if backend == "cpu" else 4,
            "execution_schedule_scope": "zero_oom_source_equals_compute_v1",
            "execution_schedule_sha256": v1_digest,
            "factor_tiles_processed": len(schedule),
            "tile_ranges": ranges(width),
            "compute_tile_ranges": schedule,
        }

    cpu_run = run_receipt(cpu, "cpu", 5, 1.0)
    cuda_run = run_receipt(cuda, "cuda", 4, 1.1)

    def bytes_sha(values):
        return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()

    metric_report = {}
    for metric, left, right in (
        ("rank_ic", scalar_cpu, scalar_cuda),
        ("rank_ic_series", series_cpu, series_cuda),
    ):
        finite = np.isfinite(left) & np.isfinite(right)
        error = float(np.max(np.abs(left[finite] - right[finite])))
        metric_report[metric] = {
            "pass": True, "compared_value_count": left.size,
            "artifact_kind": "series" if left.ndim == 2 else "scalar",
            "cpu_shape": list(left.shape), "cuda_shape": list(right.shape),
            "shape_valid": True, "factor_count": len(factor_ids),
            "finite_mask_equal": np.array_equal(np.isfinite(left), np.isfinite(right)),
            "observation_counts_equal": True, "observation_counts_shape_valid": True,
            "cpu_observation_counts_sha256": hashlib.sha256(
                np.asarray(counts[metric], dtype=np.int64).tobytes()).hexdigest(),
            "cuda_observation_counts_sha256": hashlib.sha256(
                np.asarray(counts[metric], dtype=np.int64).tobytes()).hexdigest(),
            "max_abs_error": error,
            "cpu_values_sha256": bytes_sha(left),
            "cuda_values_sha256": bytes_sha(right),
        }
    report = {"pass": True, "metrics": metric_report}
    return context, cpu, cuda, cpu_run, cuda_run, report


def test_builds_real_counterbalanced_record_with_distinct_cpu5_cuda4_schedules(sample):
    context, cpu, cuda, cpu_run, cuda_run, report = sample
    record = build_counterbalanced_profile_record(
        cpu_bundle=cpu, cuda_bundle=cuda, cpu_run_receipt=cpu_run,
        cuda_run_receipt=cuda_run, context=context, comparison_report=report,
        execution_order=("cpu", "cuda"), cpu_correctness_validated=True,
        cuda_correctness_validated=True,
    )
    assert record.cpu.source_tile_size == record.cpu.actual_tile_size == 5
    assert record.cuda.source_tile_size == record.cuda.actual_tile_size == 4
    assert record.cpu.source_ranges == ((0, 5),)
    assert record.cuda.source_ranges == ((0, 4), (4, 5))
    assert record.cpu.compute_ranges == record.cpu.source_ranges
    assert record.cuda.compute_ranges == record.cuda.source_ranges
    assert record.cpu.outputs[0].coverage_observed == 5
    assert record.cuda.outputs[1].coverage_observed == 15
    assert record.cpu.correctness_validated is True
    reverse = replace(record, execution_order=("cuda", "cpu"),
                      cpu=replace(record.cpu, seconds=1.01),
                      cuda=replace(record.cuda, seconds=1.11))
    qualified = validate_source_route_profile_qualification(
        (record, reverse), expected_context=context)
    assert qualified.source_tile_sizes == (("cpu", 5), ("cuda", 4))


def test_hashes_actual_values_masks_and_counts_in_canonical_bytes(sample):
    context, cpu, _, cpu_run, _, _ = sample
    measurement = build_backend_profile_measurement(
        cpu, cpu_run, context, correctness_validated=True)
    scalar = cpu.scalar_metrics["rank_ic"]
    output = measurement.outputs[0]
    assert output.values_sha256 == hashlib.sha256(
        np.ascontiguousarray(scalar).tobytes()).hexdigest()
    assert output.finite_mask_sha256 == hashlib.sha256(
        np.isfinite(scalar).astype(np.uint8).tobytes()).hexdigest()
    assert output.observation_counts_sha256 == hashlib.sha256(
        np.asarray(cpu.observation_counts["rank_ic"], dtype=np.int64).tobytes()).hexdigest()


def test_source_execution_receipt_clips_small_factor_extent_and_rejects_oversize_gpu():
    cpu = validate_source_execution_receipt(
        reads=[(0, 5)], factor_count=5, admitted_cap=16,
        metadata={"backend_used": "cpu", "factor_tiles_processed": 1},
    )
    assert cpu["actual_source_tile_size"] == 5
    assert cpu["compute_tile_ranges"] == ((0, 5),)
    capped_cpu = validate_source_execution_receipt(
        reads=[(0, 5)], factor_count=5, admitted_cap=5,
        metadata={"backend_used": "cpu", "factor_tiles_processed": 1},
    )
    assert capped_cpu["actual_source_tile_size"] == 5

    big_request = validate_source_execution_receipt(
        reads=[(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
               (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24),
               (24, 26), (26, 28), (28, 30), (30, 32), (32, 34), (34, 36),
               (36, 38), (38, 40), (40, 42), (42, 44), (44, 46), (46, 48)],
        factor_count=48, admitted_cap=2,
        metadata={"backend_used": "cpu", "factor_tiles_processed": 24},
    )
    assert big_request["actual_source_tile_size"] == 2
    with pytest.raises(ValueError):
        validate_source_execution_receipt(
            reads=[(0, 16), (16, 32), (32, 48)], factor_count=48,
            admitted_cap=2,
            metadata={"backend_used": "cpu", "factor_tiles_processed": 3},
        )
    with pytest.raises(ValueError, match="outside source admission"):
        validate_source_execution_receipt(
            reads=[(0, 5)], factor_count=5, admitted_cap=16,
            metadata={"backend_used": "cuda", "factor_tile_size": 6,
                      "oom_retries": 0, "factor_tiles_processed": 1},
        )


def test_accepts_cpu_small_factor_extent_with_larger_effective_api_cap(sample):
    context, cpu, _, cpu_run, _, _ = sample
    larger_api_cap = dict(cpu_run, effective_max_tile_size=16)
    measurement = build_backend_profile_measurement(
        cpu, larger_api_cap, context, correctness_validated=True)
    assert measurement.source_tile_size == measurement.actual_tile_size == 5


def test_context_source_cap_below_factor_extent_does_not_accept_unbounded_api_cap(sample):
    context, cpu, _, cpu_run, _, _ = sample
    tighter_context = replace(context, live_source_admitted_max_tile_size=4)
    width4 = dict(cpu_run, actual_source_tile_size=4,
                  factor_tiles_processed=2, tile_ranges=[(0, 4), (4, 5)],
                  compute_tile_ranges=((0, 4), (4, 5)), effective_max_tile_size=16)
    width4["execution_schedule_sha256"] = stable_content_hex(
        tag="SourceExecutionSchedule.v1",
        fields={"backend": "cpu", "factor_count": 5, "source_width": 4,
                "source_ranges": ((0, 4), (4, 5)),
                "compute_ranges": ((0, 4), (4, 5)), "oom_retries": 0},
    )
    with pytest.raises(ValueError, match="effective source cap"):
        build_backend_profile_measurement(
            cpu, width4, tighter_context, correctness_validated=True)


@pytest.mark.parametrize("mutation", [
    "timing_scope", "request_fingerprint", "backend", "seconds_bool",
    "seconds_inf", "oom_bool", "factor_count_float", "source_width_bool",
    "source_ranges", "compute_ranges", "wrong_schedule_scope",
    "wrong_effective_cap",
])
def test_rejects_malformed_or_mismatched_run_metadata(sample, mutation):
    context, cpu, _, cpu_run, _, _ = sample
    changed = dict(cpu_run)
    if mutation == "timing_scope":
        changed["timing_scope"] = "source_read_only"
    elif mutation == "request_fingerprint":
        changed["source_request_fingerprint"] = _h("9")
    elif mutation == "backend":
        changed["backend_used"] = "cuda"
    elif mutation == "seconds_bool":
        changed["seconds"] = True
    elif mutation == "seconds_inf":
        changed["seconds"] = float("inf")
    elif mutation == "oom_bool":
        changed["oom_retries"] = False
    elif mutation == "factor_count_float":
        changed["factor_tiles_processed"] = 1.0
    elif mutation == "source_width_bool":
        changed["actual_source_tile_size"] = True
    elif mutation == "source_ranges":
        changed["tile_ranges"] = [(0, 4), (4, 5)]
    elif mutation == "compute_ranges":
        changed["compute_tile_ranges"] = ((0, 4), (4, 5))
    elif mutation == "wrong_schedule_scope":
        changed["execution_schedule_scope"] = "source_only"
    elif mutation == "wrong_effective_cap":
        changed["effective_max_tile_size"] = 4
    with pytest.raises(ValueError):
        build_backend_profile_measurement(cpu, changed, context,
                                          correctness_validated=True)


@pytest.mark.parametrize("attestation", [False, 1, np.bool_(True)])
def test_requires_explicit_exact_upstream_correctness_attestation(sample, attestation):
    context, cpu, _, cpu_run, _, _ = sample
    with pytest.raises(ValueError, match="correctness_validated"):
        build_backend_profile_measurement(cpu, cpu_run, context,
                                          correctness_validated=attestation)


def test_rejects_missing_metric_or_observation_count(sample):
    context, cpu, _, cpu_run, _, _ = sample
    no_metric = replace(cpu, series_metrics={})
    with pytest.raises(ValueError, match="exactly one"):
        build_backend_profile_measurement(no_metric, cpu_run, context,
                                          correctness_validated=True)
    no_counts = replace(cpu, observation_counts={"rank_ic": cpu.observation_counts["rank_ic"]})
    with pytest.raises(ValueError, match="observation counts missing"):
        build_backend_profile_measurement(no_counts, cpu_run, context,
                                          correctness_validated=True)


def test_rejects_bad_series_cardinality_and_malformed_count_dtype(sample):
    context, cpu, _, cpu_run, _, _ = sample
    bad_series = replace(cpu, series_metrics={
        "rank_ic_series": cpu.series_metrics["rank_ic_series"][:, :4]})
    with pytest.raises(ValueError, match="exact shape"):
        build_backend_profile_measurement(bad_series, cpu_run, context,
                                          correctness_validated=True)
    bad_counts = replace(cpu, observation_counts={
        "rank_ic": np.ones(5, dtype=np.float64),
        "rank_ic_series": cpu.observation_counts["rank_ic_series"],
    })
    with pytest.raises(ValueError, match="nonnegative integers"):
        build_backend_profile_measurement(bad_counts, cpu_run, context,
                                          correctness_validated=True)
    overflow_counts = replace(cpu, observation_counts={
        "rank_ic": np.array([2**63, 0, 0, 0, 0], dtype=np.uint64),
        "rank_ic_series": cpu.observation_counts["rank_ic_series"],
    })
    with pytest.raises(ValueError, match="int64 encoding"):
        build_backend_profile_measurement(overflow_counts, cpu_run, context,
                                          correctness_validated=True)


def test_hashes_unicode_factor_ids_with_benchmark_json_encoding(sample):
    context, cpu, _, cpu_run, _, _ = sample
    ids = ("因子-0", *cpu.factor_ids[1:])
    unicode_bundle = replace(cpu, factor_ids=ids)
    unicode_receipt = dict(cpu_run)
    unicode_receipt["factor_ids_sha256"] = hashlib.sha256(json.dumps(
        list(ids), separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    measurement = build_backend_profile_measurement(
        unicode_bundle, unicode_receipt, context, correctness_validated=True)
    assert measurement.backend == "cpu"


@pytest.mark.parametrize("field,value", [
    ("compared_value_count", True), ("compared_value_count", 5.0),
    ("max_abs_error", True), ("max_abs_error", float("inf")),
])
def test_rejects_malformed_pairwise_numeric_metadata(sample, field, value):
    context, cpu, cuda, cpu_run, cuda_run, report = sample
    changed = {"pass": True, "metrics": {
        key: dict(item) for key, item in report["metrics"].items()}}
    changed["metrics"]["rank_ic"][field] = value
    with pytest.raises(ValueError):
        build_counterbalanced_profile_record(
            cpu_bundle=cpu, cuda_bundle=cuda, cpu_run_receipt=cpu_run,
            cuda_run_receipt=cuda_run, context=context,
            comparison_report=changed, execution_order=("cpu", "cuda"),
            cpu_correctness_validated=True, cuda_correctness_validated=True,
        )


def test_rejects_failed_pairwise_report_without_conflating_correctness(sample):
    context, cpu, cuda, cpu_run, cuda_run, report = sample
    bad_report = {**report, "pass": False}
    with pytest.raises(ValueError, match="pairwise comparison"):
        build_counterbalanced_profile_record(
            cpu_bundle=cpu, cuda_bundle=cuda, cpu_run_receipt=cpu_run,
            cuda_run_receipt=cuda_run, context=context,
            comparison_report=bad_report, execution_order=("cpu", "cuda"),
            cpu_correctness_validated=True, cuda_correctness_validated=True,
        )


def test_rejects_nan_vs_infinity_even_if_forged_report_says_pass(sample):
    context, cpu, cuda, cpu_run, cuda_run, report = sample
    changed_series = cuda.series_metrics["rank_ic_series"].copy()
    changed_series[2, 3] = np.nan  # CPU has -Inf at this cell.
    changed_cuda = replace(cuda, series_metrics={"rank_ic_series": changed_series})
    forged = {"pass": True, "metrics": {key: dict(value)
                                         for key, value in report["metrics"].items()}}
    forged_metric = forged["metrics"]["rank_ic_series"]
    forged_metric["cuda_values_sha256"] = hashlib.sha256(
        np.ascontiguousarray(changed_series).tobytes()).hexdigest()
    with pytest.raises(ValueError, match="NaN/Inf"):
        build_counterbalanced_profile_record(
            cpu_bundle=cpu, cuda_bundle=changed_cuda, cpu_run_receipt=cpu_run,
            cuda_run_receipt=cuda_run, context=context,
            comparison_report=forged, execution_order=("cpu", "cuda"),
            cpu_correctness_validated=True, cuda_correctness_validated=True,
        )
