"""Build v2 source-route receipts from real benchmark outputs.

This converter validates completed BatchEvaluationBundle values and the
benchmark run_backend receipt. It does not capture live source context, time
the API, attest COS provenance, or treat CPU/CUDA agreement as a correctness
oracle. The caller must supply correctness_validated=True from a trusted
upstream oracle for each backend run.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement, CounterbalancedRouteProfileRecord,
    MetricComparisonReceipt, MetricOutputReceipt, SourceRouteProfileContext,
    route_profile_execution_config_sha256,
)
from quant_evaluator.runtime.source_profile_output_identity import (
    hash_array_bytes, hash_finite_mask, normalize_counts_array,
)

_TIMING_SCOPE = "evaluate_factor_source_batch_wall_v1"
_SCHEDULE_SCOPE = "zero_oom_source_equals_compute_v1"
_SERIES_METRICS = frozenset({"rank_ic_series", "pearson_ic_series"})
_HASH_CHUNK = 65_536


def _mapping(value: object, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _hash_array_bytes(values: np.ndarray) -> str:
    """SHA-256 of canonical C-order array bytes, using bounded buffers."""
    return hash_array_bytes(values)


def _finite_mask_hash(values: np.ndarray) -> str:
    return hash_finite_mask(values)


def _counts_array(bundle: BatchEvaluationBundle, metric: str, factor_count: int) -> np.ndarray:
    counts_map = _mapping(bundle.observation_counts, "observation_counts")
    if metric not in counts_map:
        raise ValueError(f"observation counts missing for {metric!r}")
    counts = counts_map[metric]
    if not isinstance(counts, np.ndarray) or counts.shape != (factor_count,):
        raise ValueError(f"observation counts for {metric!r} must have shape (F,)")
    if counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError(f"observation counts for {metric!r} must be nonnegative integers")
    if counts.dtype.kind == "u" and np.any(counts > np.iinfo(np.int64).max):
        raise ValueError(f"observation counts for {metric!r} exceed int64 encoding")
    # Same int64 byte encoding used by benchmark_real_cos_source_batch.compare.
    return normalize_counts_array(counts)


def _metric_array(bundle: BatchEvaluationBundle, metric: str,
                  context: SourceRouteProfileContext) -> tuple[np.ndarray, np.ndarray]:
    scalar = _mapping(bundle.scalar_metrics, "scalar_metrics")
    series = _mapping(bundle.series_metrics, "series_metrics")
    vector = _mapping(bundle.vector_metrics, "vector_metrics")
    if metric in vector:
        raise ValueError(f"vector metric {metric!r} is outside this profile schema")
    in_scalar, in_series = metric in scalar, metric in series
    if in_scalar == in_series:
        raise ValueError(f"metric {metric!r} must occur in exactly one scalar/series map")
    if (metric in _SERIES_METRICS) != in_series:
        raise ValueError(f"metric {metric!r} has the wrong scalar/series artifact kind")
    values = series[metric] if in_series else scalar[metric]
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu":
        raise ValueError(f"metric {metric!r} must be a real numeric ndarray")
    T, _, F = context.request_shape
    expected = (T, F) if in_series else (F,)
    if values.shape != expected:
        raise ValueError(f"metric {metric!r} must have exact shape {expected!r}")
    return values, _counts_array(bundle, metric, F)


def _validate_bundle_identity(bundle: object, context: SourceRouteProfileContext,
                              receipt: Mapping) -> tuple[str, ...]:
    if type(bundle) is not BatchEvaluationBundle:
        raise ValueError("bundle must be an exact BatchEvaluationBundle")
    F = context.request_shape[-1]
    factor_ids = bundle.factor_ids
    if (type(factor_ids) is not tuple or len(factor_ids) != F
            or any(type(item) is not str or not item for item in factor_ids)
            or len(set(factor_ids)) != F):
        raise ValueError("bundle factor axis must contain F unique nonempty IDs")
    meta = _mapping(bundle.metadata, "bundle metadata")
    if (meta.get("source_request_fingerprint") != context.request_content_sha256
            or receipt.get("source_request_fingerprint") != context.request_content_sha256):
        raise ValueError("bundle/run request fingerprint differs from live context")
    ids_sha = hashlib.sha256(json.dumps(
        list(factor_ids), separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if receipt.get("factor_ids_sha256") != ids_sha:
        raise ValueError("run receipt factor identity differs from bundle axis")
    return factor_ids


def _ranges(value: object, *, factor_count: int, width: int, name: str) -> tuple:
    if type(value) not in (tuple, list):
        raise ValueError(f"{name} must be an ordered range sequence")
    expected = tuple((start, min(start + width, factor_count))
                     for start in range(0, factor_count, width))
    normalized = []
    for item in value:
        if (type(item) not in (tuple, list) or len(item) != 2
                or type(item[0]) is not int or type(item[1]) is not int):
            raise ValueError(f"{name} contains malformed range metadata")
        normalized.append((item[0], item[1]))
    result = tuple(normalized)
    if result != expected:
        raise ValueError(f"{name} differs from exact ordered fixed-width coverage")
    return result


def build_backend_profile_measurement(
    bundle: BatchEvaluationBundle, run_receipt: Mapping[str, Any],
    context: SourceRouteProfileContext, *, correctness_validated: bool,
    qualified_execution_cap: int | None = None,
) -> BackendRouteProfileMeasurement:
    """Bind one real backend run to v2 timing, schedule, and output receipts."""
    receipt = _mapping(run_receipt, "run_receipt")
    if type(correctness_validated) is not bool or correctness_validated is not True:
        raise ValueError("trusted upstream correctness_validated=True is required")
    if type(context) is not SourceRouteProfileContext:
        raise ValueError("context must come from live context capture")
    factor_ids = _validate_bundle_identity(bundle, context, receipt)
    bundle_meta = _mapping(bundle.metadata, "bundle metadata")
    backend = receipt.get("backend_used")
    if (type(backend) is not str or backend not in ("cpu", "cuda")
            or bundle_meta.get("backend_used") != backend):
        raise ValueError("actual backend identity is missing or differs from bundle")
    if (receipt.get("timing_scope") != _TIMING_SCOPE
            or context.timing_scope != _TIMING_SCOPE):
        raise ValueError("timing scope is not the complete source API wall time")
    seconds = receipt.get("seconds")
    total_wall = receipt.get("total_wall_seconds")
    if (type(seconds) is not float or not math.isfinite(seconds) or seconds <= 0
            or type(total_wall) is not float or total_wall != seconds):
        raise ValueError("whole-request seconds must be finite, positive, and consistent")
    if type(receipt.get("oom_retries")) is not int or receipt["oom_retries"] != 0:
        raise ValueError("profile requires exact integer zero OOM retries")

    F = len(factor_ids)
    source_width = receipt.get("actual_source_tile_size")
    actual_width = receipt.get("actual_gpu_factor_tile_size")
    if (type(source_width) is not int or not 1 <= source_width
            <= context.live_source_admitted_max_tile_size
            or source_width > context.requested_tile_size
            or source_width > F):
        raise ValueError("actual source width is outside live admission")
    if backend == "cuda":
        if (type(actual_width) is not int or actual_width != source_width
                or actual_width > context.gpu_admitted_max_tile_size
                or actual_width > F):
            raise ValueError("CUDA compute width differs from source/policy admission")
    elif actual_width is not None:
        raise ValueError("CPU run must not claim a CUDA compute width")
    effective_cap = receipt.get("effective_max_tile_size")
    expected_effective_cap = (context.live_source_admitted_max_tile_size
                              if backend == "cpu"
                              else context.gpu_admitted_max_tile_size)
    if qualified_execution_cap is not None:
        # Qualified auto pins the measured execution width, which may be lower
        # than its original admission ceiling. Do not alter that context.
        if (type(qualified_execution_cap) is not int
                or not 1 <= qualified_execution_cap <= expected_effective_cap):
            raise ValueError("qualified execution cap is outside live admission")
        if (receipt.get("backend_requested") != "auto"
                or receipt.get("source_auto_policy") != "qualified_only"
                or bundle_meta.get("source_auto_policy") != "qualified_only"
                or receipt.get("context_before") != context
                or receipt.get("context_after") != context
                or bundle_meta.get("source_qualification_applied") is not True
                or bundle_meta.get("source_qualification_status") != "qualified_current_source"
                or bundle_meta.get("source_qualification_winner") != backend):
            raise ValueError("qualified execution cap requires qualified-only auto")
        if (source_width != qualified_execution_cap
                or (backend == "cuda" and actual_width != qualified_execution_cap)):
            raise ValueError("qualified execution cap differs from actual execution width")
        effective_cap_valid = (
            type(effective_cap) is int and effective_cap == qualified_execution_cap)
    else:
        cap_is_small_factor_extent = expected_effective_cap == F
        effective_cap_valid = (
            type(effective_cap) is int
            and (effective_cap == expected_effective_cap
                 if not cap_is_small_factor_extent
                 else F <= effective_cap <= context.requested_tile_size)
        )
    if not effective_cap_valid or source_width > effective_cap:
        raise ValueError("run effective source cap differs from backend live context")
    if receipt.get("execution_schedule_scope") != _SCHEDULE_SCOPE:
        raise ValueError("run schedule scope is not the zero-OOM exact schedule")
    if type(receipt.get("factor_tiles_processed")) is not int:
        raise ValueError("factor tile count is missing or malformed")
    source_ranges = _ranges(receipt.get("tile_ranges"), factor_count=F,
                            width=source_width, name="source tile schedule")
    compute_ranges = _ranges(receipt.get("compute_tile_ranges"), factor_count=F,
                             width=source_width, name="compute tile schedule")
    if receipt["factor_tiles_processed"] != len(source_ranges):
        raise ValueError("factor tile count differs from source schedule")
    if compute_ranges != source_ranges:
        raise ValueError("zero-OOM receipt requires exact source/compute ranges")
    source_schedule_sha = stable_content_hex(tag="SourceExecutionSchedule.v1", fields={
        "backend": backend, "factor_count": F, "source_width": source_width,
        "source_ranges": source_ranges, "compute_ranges": compute_ranges,
        "oom_retries": 0,
    })
    if receipt.get("execution_schedule_sha256") != source_schedule_sha:
        raise ValueError("benchmark schedule digest differs from actual ranges")

    coverage = dict(context.metric_coverage)
    outputs, total = [], 0
    for metric in context.metric_ids:
        values, counts = _metric_array(bundle, metric, context)
        expected = coverage[metric]
        if values.size != expected:
            raise ValueError(f"metric {metric!r} has incomplete coverage")
        outputs.append(MetricOutputReceipt(
            metric_id=metric, values_sha256=_hash_array_bytes(values),
            finite_mask_sha256=_finite_mask_hash(values),
            observation_counts_sha256=_hash_array_bytes(counts),
            coverage_expected=expected, coverage_observed=int(values.size),
        ))
        total += int(values.size)
    v2_schedule_sha = stable_content_hex(
        tag="RouteProfileExecutionSchedule.v2",
        fields={"scope": _SCHEDULE_SCOPE, "source_ranges": source_ranges,
                "compute_ranges": compute_ranges},
    )
    return BackendRouteProfileMeasurement(
        backend=backend, seconds=float(seconds), oom_count=0,
        source_tile_size=source_width, source_ranges=source_ranges,
        actual_tile_size=source_width, compute_ranges=compute_ranges,
        execution_config_sha256=route_profile_execution_config_sha256(
            context, backend, source_width),
        execution_schedule_sha256=v2_schedule_sha,
        execution_schedule_scope=_SCHEDULE_SCOPE, correctness_validated=True,
        coverage_expected=context.expected_coverage_count,
        coverage_observed=total, outputs=tuple(outputs),
    )


def build_metric_comparison_receipts(
    cpu_bundle: BatchEvaluationBundle, cuda_bundle: BatchEvaluationBundle,
    context: SourceRouteProfileContext, comparison_report: Mapping[str, Any], *,
    cpu_profile: BackendRouteProfileMeasurement,
    cuda_profile: BackendRouteProfileMeasurement,
) -> tuple[MetricComparisonReceipt, ...]:
    """Bind benchmark pairwise evidence without asserting correctness."""
    report = _mapping(comparison_report, "comparison_report")
    if report.get("pass") is not True:
        raise ValueError("benchmark pairwise comparison must explicitly pass")
    if cpu_profile.backend != "cpu" or cuda_profile.backend != "cuda":
        raise ValueError("CPU and CUDA profile measurements are required")
    report_metrics = _mapping(report.get("metrics"), "comparison metrics")
    if set(report_metrics) != set(context.metric_ids):
        raise ValueError("comparison report must cover exact requested metrics")
    cpu_outputs = {item.metric_id: item for item in cpu_profile.outputs}
    cuda_outputs = {item.metric_id: item for item in cuda_profile.outputs}
    tolerances = dict(context.metric_error_tolerances)
    result = []
    for metric in context.metric_ids:
        item = _mapping(report_metrics[metric], f"comparison metric {metric}")
        if item.get("pass") is not True:
            raise ValueError(f"pairwise comparison failed for {metric!r}")
        left, left_counts = _metric_array(cpu_bundle, metric, context)
        right, right_counts = _metric_array(cuda_bundle, metric, context)
        if left.shape != right.shape:
            raise ValueError(f"CPU/CUDA axes differ for {metric!r}")
        expected = dict(context.metric_coverage)[metric]
        if (type(item.get("compared_value_count")) is not int
                or item["compared_value_count"] != expected):
            raise ValueError(f"comparison count is incomplete for {metric!r}")
        factor_count = context.request_shape[-1]
        expected_kind = "series" if metric in _SERIES_METRICS else "scalar"
        expected_shape = ([context.request_shape[0], factor_count]
                          if expected_kind == "series" else [factor_count])
        if (item.get("shape_valid") is not True
                or item.get("artifact_kind") != expected_kind
                or item.get("cpu_shape") != expected_shape
                or item.get("cuda_shape") != expected_shape
                or type(item.get("factor_count")) is not int
                or item["factor_count"] != factor_count
                or item.get("observation_counts_shape_valid") is not True):
            raise ValueError(f"benchmark comparison has incomplete axes for {metric!r}")
        cpu, cuda = cpu_outputs[metric], cuda_outputs[metric]
        if (_hash_array_bytes(left) != cpu.values_sha256
                or _hash_array_bytes(right) != cuda.values_sha256
                or _hash_array_bytes(left_counts) != cpu.observation_counts_sha256
                or _hash_array_bytes(right_counts) != cuda.observation_counts_sha256):
            raise ValueError(f"comparison does not bind measured arrays for {metric!r}")
        if (item.get("cpu_observation_counts_sha256") != cpu.observation_counts_sha256
                or item.get("cuda_observation_counts_sha256") != cuda.observation_counts_sha256):
            raise ValueError(f"benchmark observation-count hashes differ for {metric!r}")
        finite_equal = cpu.finite_mask_sha256 == cuda.finite_mask_sha256
        count_equal = cpu.observation_counts_sha256 == cuda.observation_counts_sha256
        if (item.get("finite_mask_equal") is not True or not finite_equal
                or item.get("observation_counts_equal") is not True or not count_equal):
            raise ValueError(f"finite masks/counts differ for {metric!r}")
        max_error = 0.0
        iterator = np.nditer([left, right],
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"], ["readonly"]], order="C", buffersize=_HASH_CHUNK)
        for a, b in iterator:
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                error = float(np.max(np.abs(a[finite] - b[finite])))
                if not math.isfinite(error):
                    raise ValueError(f"numeric comparison is non-finite for {metric!r}")
                max_error = max(max_error, error)
            # The v2 finite mask intentionally groups NaN and infinities as
            # non-finite. Preserve the benchmark comparator's stricter pairwise
            # semantics by also requiring each non-finite category to match.
            if (not np.array_equal(np.isnan(a), np.isnan(b))
                    or not np.array_equal(np.isposinf(a), np.isposinf(b))
                    or not np.array_equal(np.isneginf(a), np.isneginf(b))):
                raise ValueError(f"NaN/Inf locations or signs differ for {metric!r}")
        tolerance = tolerances[metric]
        reported_error = item.get("max_abs_error")
        if (type(reported_error) is not float or not math.isfinite(reported_error)
                or reported_error < 0 or max_error > tolerance
                or reported_error != max_error):
            raise ValueError(f"reported numeric error differs/exceeds tolerance for {metric!r}")
        if (item.get("cpu_values_sha256") != cpu.values_sha256
                or item.get("cuda_values_sha256") != cuda.values_sha256):
            raise ValueError(f"benchmark value hashes differ for {metric!r}")
        fields = {
            "metric_id": metric, "cpu_values_sha256": cpu.values_sha256,
            "cuda_values_sha256": cuda.values_sha256,
            "cpu_finite_mask_sha256": cpu.finite_mask_sha256,
            "cuda_finite_mask_sha256": cuda.finite_mask_sha256,
            "cpu_observation_counts_sha256": cpu.observation_counts_sha256,
            "cuda_observation_counts_sha256": cuda.observation_counts_sha256,
            "compared_value_count": expected, "finite_mask_equal": True,
            "observation_counts_equal": True, "max_abs_error": max_error,
            "max_abs_error_tolerance": tolerance,
        }
        result.append(MetricComparisonReceipt(
            **fields, evidence_sha256=stable_content_hex(
                tag="SourceRouteMetricComparison.v2", fields=fields)))
    return tuple(result)


def build_counterbalanced_profile_record(
    *, cpu_bundle: BatchEvaluationBundle, cuda_bundle: BatchEvaluationBundle,
    cpu_run_receipt: Mapping[str, Any], cuda_run_receipt: Mapping[str, Any],
    context: SourceRouteProfileContext, comparison_report: Mapping[str, Any],
    execution_order: tuple[str, str], cpu_correctness_validated: bool,
    cuda_correctness_validated: bool,
) -> CounterbalancedRouteProfileRecord:
    """Construct one observed CPU/CUDA pair for a later ABBA qualification."""
    if type(execution_order) is not tuple or execution_order not in (
        ("cpu", "cuda"), ("cuda", "cpu"),
    ):
        raise ValueError("execution_order must identify actual CPU/CUDA order")
    cpu = build_backend_profile_measurement(
        cpu_bundle, cpu_run_receipt, context,
        correctness_validated=cpu_correctness_validated)
    cuda = build_backend_profile_measurement(
        cuda_bundle, cuda_run_receipt, context,
        correctness_validated=cuda_correctness_validated)
    comparisons = build_metric_comparison_receipts(
        cpu_bundle, cuda_bundle, context, comparison_report,
        cpu_profile=cpu, cuda_profile=cuda)
    return CounterbalancedRouteProfileRecord(
        context=context, execution_order=execution_order,
        cpu=cpu, cuda=cuda, comparison_evidence=comparisons)


__all__ = ("build_backend_profile_measurement", "build_metric_comparison_receipts",
           "build_counterbalanced_profile_record")
