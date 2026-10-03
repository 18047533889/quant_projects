"""Caller-trusted v2 qualification for route-specific source tile profiles.

Unlike the v1 single-width context, this schema keeps request/runtime identity
shared while recording distinct source-read and backend-compute schedules and
execution-configuration digests for CPU and CUDA. A width-16 CPU profile can
therefore be compared with a width-4 CUDA profile for the same cap-16 request
without pretending they used the same width or read boundaries.

The validator checks receipt structure and internal consistency only. It does
not authenticate the producer, observe the actual tile schedule, verify loaded
code, independently reproduce numeric comparisons, or attest DataAccess/COS
provenance. Producer-generated execution and evidence digests remain trusted
declarations and require an independently trusted benchmark harness.

Timing uses ``evaluate_factor_source_batch_wall_v1``: elapsed time around the
API invocation, including its validation, source reads, transfers, execution,
and result assembly, but excluding source construction and source close.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import re
from typing import Tuple

from quant_evaluator.contracts._hashutil import stable_content_hex


SCHEMA_VERSION = "source_route_profiles_v2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BACKENDS = ("cpu", "cuda")


def _sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _finite_nonnegative(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def _finite_positive(value: object) -> bool:
    return _finite_nonnegative(value) and value > 0


@dataclass(frozen=True)
class SourceRouteProfileContext:
    """Exact shared request/source/runtime identity and live admitted ceiling."""

    request_content_sha256: str
    source_identity_sha256: str
    source_content_sha256: str
    executable_source_sha256: str
    runtime_fingerprint_sha256: str
    package_fingerprint_sha256: str
    thread_fingerprint_sha256: str
    device_fingerprint_sha256: str
    common_config_fingerprint_sha256: str
    request_shape: Tuple[int, int, int]
    metric_ids: Tuple[str, ...]
    expected_coverage_count: int
    requested_tile_size: int
    timing_scope: str
    metric_coverage: Tuple[Tuple[str, int], ...]
    metric_error_tolerances: Tuple[Tuple[str, float], ...]
    live_source_admitted_max_tile_size: int
    gpu_admitted_max_tile_size: int


@dataclass(frozen=True)
class MetricOutputReceipt:
    """One backend's output identities and complete metric coverage."""

    metric_id: str
    values_sha256: str
    finite_mask_sha256: str
    observation_counts_sha256: str
    coverage_expected: int
    coverage_observed: int


@dataclass(frozen=True)
class MetricComparisonReceipt:
    """Independent CPU/CUDA comparison evidence for one requested metric."""

    metric_id: str
    cpu_values_sha256: str
    cuda_values_sha256: str
    cpu_finite_mask_sha256: str
    cuda_finite_mask_sha256: str
    cpu_observation_counts_sha256: str
    cuda_observation_counts_sha256: str
    compared_value_count: int
    finite_mask_equal: bool
    observation_counts_equal: bool
    max_abs_error: float
    max_abs_error_tolerance: float
    evidence_sha256: str


@dataclass(frozen=True)
class BackendRouteProfileMeasurement:
    """Whole-request timing and source/compute configurations for one backend."""

    backend: str
    seconds: float
    oom_count: int
    source_tile_size: int
    source_ranges: Tuple[Tuple[int, int], ...]
    actual_tile_size: int
    compute_ranges: Tuple[Tuple[int, int], ...]
    execution_config_sha256: str
    execution_schedule_sha256: str
    execution_schedule_scope: str
    correctness_validated: bool
    coverage_expected: int
    coverage_observed: int
    outputs: Tuple[MetricOutputReceipt, ...]


@dataclass(frozen=True)
class CounterbalancedRouteProfileRecord:
    """One complete CPU/CUDA pair executed in the declared order."""

    context: SourceRouteProfileContext
    execution_order: Tuple[str, str]
    cpu: BackendRouteProfileMeasurement
    cuda: BackendRouteProfileMeasurement
    comparison_evidence: Tuple[MetricComparisonReceipt, ...]


@dataclass(frozen=True)
class SourceRouteProfileQualification:
    context: SourceRouteProfileContext
    winning_backend: str
    source_tile_sizes: Tuple[Tuple[str, int], Tuple[str, int]]
    actual_tile_sizes: Tuple[Tuple[str, int], Tuple[str, int]]
    execution_config_sha256: Tuple[Tuple[str, str], Tuple[str, str]]
    run_orders: Tuple[Tuple[str, str], Tuple[str, str]]
    timings_seconds: Tuple[Tuple[float, float], Tuple[float, float]]
    mean_whole_request_seconds: Tuple[Tuple[str, float], Tuple[str, float]]
    correctness_digest_sha256: str


def _validate_context(context: object) -> None:
    if type(context) is not SourceRouteProfileContext:
        raise ValueError("context must be a SourceRouteProfileContext")
    hash_names = (
        "request_content_sha256", "source_identity_sha256", "source_content_sha256",
        "executable_source_sha256", "runtime_fingerprint_sha256",
        "package_fingerprint_sha256", "thread_fingerprint_sha256",
        "device_fingerprint_sha256", "common_config_fingerprint_sha256",
    )
    if any(not _sha256(getattr(context, name)) for name in hash_names):
        raise ValueError("context contains a missing or malformed SHA-256 identity")
    shape = context.request_shape
    if (type(shape) is not tuple or len(shape) != 3
            or any(not _positive_int(value) for value in shape)):
        raise ValueError("request_shape must contain three positive integers")
    metrics = context.metric_ids
    if (type(metrics) is not tuple or not metrics
            or any(type(item) is not str or not item or item.strip() != item
                   for item in metrics)
            or len(set(metrics)) != len(metrics)):
        raise ValueError("metric_ids must be unique nonempty strings")
    if not _positive_int(context.expected_coverage_count):
        raise ValueError("expected_coverage_count must be a positive integer")
    if context.timing_scope != "evaluate_factor_source_batch_wall_v1":
        raise ValueError("timing scope must cover the complete source API call")
    if (not _positive_int(context.requested_tile_size)
            or not _positive_int(context.live_source_admitted_max_tile_size)
            or context.live_source_admitted_max_tile_size > context.requested_tile_size
            or context.live_source_admitted_max_tile_size > shape[-1]
            or not _positive_int(context.gpu_admitted_max_tile_size)
            or context.gpu_admitted_max_tile_size > context.live_source_admitted_max_tile_size):
        raise ValueError("requested, source-admitted, or GPU-admitted tile ceiling is invalid")
    metric_coverage = context.metric_coverage
    if (type(metric_coverage) is not tuple or len(metric_coverage) != len(metrics)
            or any(type(item) is not tuple or len(item) != 2
                   or type(item[0]) is not str or type(item[1]) is not int
                   or item[1] <= 0 for item in metric_coverage)
            or tuple(item[0] for item in metric_coverage) != metrics
            or sum(item[1] for item in metric_coverage) != context.expected_coverage_count):
        raise ValueError("metric coverage map must exactly bind metrics and total coverage")
    tolerances = context.metric_error_tolerances
    if (type(tolerances) is not tuple or len(tolerances) != len(metrics)
            or any(type(item) is not tuple or len(item) != 2
                   or type(item[0]) is not str
                   or not _finite_nonnegative(item[1]) for item in tolerances)
            or tuple(item[0] for item in tolerances) != metrics):
        raise ValueError("metric error tolerance map must exactly bind requested metrics")


def route_profile_execution_config_sha256(
    context: SourceRouteProfileContext, backend: str, source_tile_size: int,
) -> str:
    """Return the only accepted live CPU/GPU execution-profile fingerprint."""
    _validate_context(context)
    if type(backend) is not str or backend not in _BACKENDS:
        raise ValueError("backend must be cpu or cuda")
    if (type(source_tile_size) is not int or source_tile_size <= 0
            or source_tile_size > context.live_source_admitted_max_tile_size
            or (backend == "cuda"
                and source_tile_size > context.gpu_admitted_max_tile_size)):
        raise ValueError("profile width exceeds its live backend admission")
    if backend == "cpu" and source_tile_size > context.requested_tile_size:
        raise ValueError("CPU profile width exceeds the requested source cap")
    return stable_content_hex(
        tag="SourceRouteExecutionConfig.v2",
        fields={
            "common_config_fingerprint_sha256": context.common_config_fingerprint_sha256,
            "backend": backend,
            "requested_tile_size": context.requested_tile_size,
            "live_source_admitted_max_tile_size": context.live_source_admitted_max_tile_size,
            "gpu_admitted_max_tile_size": context.gpu_admitted_max_tile_size,
            "source_tile_size": source_tile_size,
            "actual_tile_size": source_tile_size,
            "execution_schedule_scope": "zero_oom_source_equals_compute_v1",
        },
    )


def _validate_ranges(ranges: object, *, width: int, factor_count: int,
                     expected: tuple | None = None) -> tuple:
    if type(ranges) is not tuple:
        raise ValueError("tile schedule ranges must be an immutable tuple")
    normalized = []
    cursor = 0
    for item in ranges:
        if (type(item) is not tuple or len(item) != 2
                or type(item[0]) is not int or type(item[1]) is not int
                or item[0] != cursor or item[1] <= item[0]
                or item[1] - item[0] > width):
            raise ValueError("tile schedule ranges are malformed, gapped, or exceed width")
        normalized.append(item)
        cursor = item[1]
    if cursor != factor_count or not normalized:
        raise ValueError("tile schedule does not cover the full factor axis")
    result = tuple(normalized)
    if expected is not None and result != expected:
        raise ValueError("tile schedule differs from its declared fixed-width profile")
    return result


def _validate_profile(measurement: object, backend: str,
                      context: SourceRouteProfileContext) -> None:
    if type(measurement) is not BackendRouteProfileMeasurement:
        raise ValueError("backend profile measurement is missing or malformed")
    if type(measurement.backend) is not str or measurement.backend != backend:
        raise ValueError("backend profile identity is inconsistent")
    if not _finite_positive(measurement.seconds):
        raise ValueError("whole-request timing must be finite and positive")
    if type(measurement.oom_count) is not int or measurement.oom_count != 0:
        raise ValueError("backend profile must report exact integer zero OOM retries")
    if (type(measurement.source_tile_size) is not int
            or not 1 <= measurement.source_tile_size <= context.live_source_admitted_max_tile_size
            or measurement.source_tile_size > context.requested_tile_size):
        raise ValueError("actual backend source tile exceeds the live source-admitted ceiling")
    if backend == "cuda" and measurement.source_tile_size > context.gpu_admitted_max_tile_size:
        raise ValueError("CUDA profile exceeds its GPU policy tile ceiling")
    if (type(measurement.actual_tile_size) is not int
            or not 1 <= measurement.actual_tile_size <= measurement.source_tile_size):
        raise ValueError("actual backend compute tile exceeds its source tile")
    if (measurement.actual_tile_size != measurement.source_tile_size
            or measurement.execution_schedule_scope != "zero_oom_source_equals_compute_v1"):
        raise ValueError("execution schedule scope requires actual source and compute tiles to match")
    if (not _sha256(measurement.execution_config_sha256)
            or not _sha256(measurement.execution_schedule_sha256)):
        raise ValueError("backend execution configuration or schedule digest is invalid")
    if measurement.execution_config_sha256 != route_profile_execution_config_sha256(
            context, backend, measurement.source_tile_size):
        raise ValueError("backend execution configuration differs from its live profile")
    factor_count = context.request_shape[-1]
    source_ranges = _validate_ranges(
        measurement.source_ranges, width=measurement.source_tile_size,
        factor_count=factor_count,
        expected=tuple((start, min(start + measurement.source_tile_size, factor_count))
                       for start in range(0, factor_count, measurement.source_tile_size)))
    compute_ranges = _validate_ranges(
        measurement.compute_ranges, width=measurement.actual_tile_size,
        factor_count=factor_count,
        expected=source_ranges)
    if compute_ranges != source_ranges:
        raise ValueError("zero-OOM execution scope requires source and compute ranges to match")
    expected_schedule_sha = stable_content_hex(
        tag="RouteProfileExecutionSchedule.v2",
        fields={"scope": measurement.execution_schedule_scope,
                "source_ranges": source_ranges, "compute_ranges": compute_ranges},
    )
    if measurement.execution_schedule_sha256 != expected_schedule_sha:
        raise ValueError("execution schedule digest does not match actual range receipt")
    if measurement.correctness_validated is not True:
        raise ValueError("backend correctness validation receipt is absent")
    if (type(measurement.coverage_expected) is not int
            or measurement.coverage_expected != context.expected_coverage_count
            or type(measurement.coverage_observed) is not int
            or measurement.coverage_observed != context.expected_coverage_count):
        raise ValueError("backend profile coverage is incomplete or inconsistent")
    outputs = measurement.outputs
    if (type(outputs) is not tuple or len(outputs) != len(context.metric_ids)
            or any(type(item) is not MetricOutputReceipt for item in outputs)
            or tuple(item.metric_id for item in outputs) != context.metric_ids):
        raise ValueError("backend metric output receipts are incomplete or out of order")
    coverage_by_metric = dict(context.metric_coverage)
    for output in outputs:
        if any(not _sha256(getattr(output, name)) for name in (
                "values_sha256", "finite_mask_sha256", "observation_counts_sha256")):
            raise ValueError("backend metric output digest is missing or malformed")
        if (type(output.coverage_expected) is not int
                or type(output.coverage_observed) is not int
                or output.coverage_expected != coverage_by_metric[output.metric_id]
                or output.coverage_observed != output.coverage_expected):
            raise ValueError("per-metric output coverage is incomplete")


def _validate_comparisons(record: CounterbalancedRouteProfileRecord) -> None:
    evidence = record.comparison_evidence
    metric_ids = record.context.metric_ids
    if (type(evidence) is not tuple or len(evidence) != len(metric_ids)
            or any(type(item) is not MetricComparisonReceipt for item in evidence)
            or tuple(item.metric_id for item in evidence) != metric_ids):
        raise ValueError("metric comparison evidence is incomplete or out of order")
    cpu_outputs = {item.metric_id: item for item in record.cpu.outputs}
    cuda_outputs = {item.metric_id: item for item in record.cuda.outputs}
    tolerance_by_metric = dict(record.context.metric_error_tolerances)
    for item in evidence:
        digest_fields = (
            "cpu_values_sha256", "cuda_values_sha256", "cpu_finite_mask_sha256",
            "cuda_finite_mask_sha256", "cpu_observation_counts_sha256",
            "cuda_observation_counts_sha256", "evidence_sha256",
        )
        if any(not _sha256(getattr(item, name)) for name in digest_fields):
            raise ValueError("metric comparison contains a malformed evidence digest")
        cpu = cpu_outputs[item.metric_id]
        cuda = cuda_outputs[item.metric_id]
        if (item.cpu_values_sha256 != cpu.values_sha256
                or item.cuda_values_sha256 != cuda.values_sha256
                or item.cpu_finite_mask_sha256 != cpu.finite_mask_sha256
                or item.cuda_finite_mask_sha256 != cuda.finite_mask_sha256
                or item.cpu_observation_counts_sha256 != cpu.observation_counts_sha256
                or item.cuda_observation_counts_sha256 != cuda.observation_counts_sha256):
            raise ValueError("comparison differs from backend output receipts")
        if (type(item.compared_value_count) is not int
                or item.compared_value_count != cpu.coverage_expected
                or item.compared_value_count != cuda.coverage_expected):
            raise ValueError("metric comparison value coverage is inconsistent")
        if item.finite_mask_equal is not True or (
                item.cpu_finite_mask_sha256 != item.cuda_finite_mask_sha256):
            raise ValueError("CPU/CUDA finite-value masks differ")
        if item.observation_counts_equal is not True or (
                item.cpu_observation_counts_sha256 != item.cuda_observation_counts_sha256):
            raise ValueError("CPU/CUDA observation counts differ")
        if (not _finite_nonnegative(item.max_abs_error)
                or not _finite_nonnegative(item.max_abs_error_tolerance)
                or item.max_abs_error_tolerance != tolerance_by_metric[item.metric_id]
                or item.max_abs_error > item.max_abs_error_tolerance):
            raise ValueError("metric numeric comparison is outside its context-bound tolerance")


def _profile_signature(measurement: BackendRouteProfileMeasurement) -> tuple:
    return (
        measurement.source_tile_size,
        measurement.source_ranges,
        measurement.actual_tile_size,
        measurement.compute_ranges,
        measurement.execution_config_sha256,
        measurement.execution_schedule_sha256,
        measurement.execution_schedule_scope,
        measurement.coverage_expected,
        measurement.coverage_observed,
        measurement.correctness_validated,
        measurement.outputs,
    )


def _comparison_digest(evidence: Tuple[MetricComparisonReceipt, ...]) -> str:
    normalized = tuple(
        tuple(getattr(item, field.name) for field in fields(item))
        for item in evidence
    )
    return stable_content_hex(tag="SourceRouteProfileCorrectness.v2",
                              fields={"metric_evidence": normalized})


def validate_source_route_profile_qualification(
    records: Tuple[CounterbalancedRouteProfileRecord,
                   CounterbalancedRouteProfileRecord],
    *, expected_context: SourceRouteProfileContext,
) -> SourceRouteProfileQualification:
    """Validate two opposite-order pairs with independently sized backend profiles.

    CPU and CUDA actual widths may differ, but each must fit the same live
    request ceiling. Per-backend configuration, width, tile schedule, and
    result digests must remain identical across orders. Both runs require
    complete metric coverage, zero OOM retries, and matching independent
    value/mask/count comparison receipts within declared numeric tolerances.
    """
    _validate_context(expected_context)
    if type(records) is not tuple or len(records) != 2:
        raise ValueError("exactly two counterbalanced profile records are required")
    orders = []
    timings = []
    winners = []
    signatures = {backend: [] for backend in _BACKENDS}
    comparisons = []
    for record in records:
        if type(record) is not CounterbalancedRouteProfileRecord:
            raise ValueError("counterbalanced profile record has an invalid type")
        _validate_context(record.context)
        if record.context != expected_context:
            raise ValueError("profile context is stale or differs from expected context")
        if (type(record.execution_order) is not tuple
                or record.execution_order not in (("cpu", "cuda"), ("cuda", "cpu"))):
            raise ValueError("execution order must contain CPU and CUDA exactly once")
        orders.append(record.execution_order)
        _validate_profile(record.cpu, "cpu", expected_context)
        _validate_profile(record.cuda, "cuda", expected_context)
        _validate_comparisons(record)
        cpu, cuda = record.cpu, record.cuda
        if cpu.seconds == cuda.seconds:
            raise ValueError("whole-request backend timings tie")
        winners.append("cuda" if cuda.seconds < cpu.seconds else "cpu")
        timings.append((cpu.seconds, cuda.seconds))
        signatures["cpu"].append(_profile_signature(cpu))
        signatures["cuda"].append(_profile_signature(cuda))
        comparisons.append(record.comparison_evidence)
    if orders[0] == orders[1]:
        raise ValueError("counterbalanced profiles must use opposite execution orders")
    for backend in _BACKENDS:
        if signatures[backend][0] != signatures[backend][1]:
            raise ValueError(f"{backend} profile changes between execution orders")
    if comparisons[0] != comparisons[1]:
        raise ValueError("independent correctness comparison evidence changes between orders")
    if winners[0] != winners[1]:
        raise ValueError("whole-request winner changes with execution order")
    # This positive-only formula avoids both overflow for very large timers
    # and underflow to zero for the smallest representable positive timers.
    def _mean_positive(left: float, right: float) -> float:
        low, high = (left, right) if left <= right else (right, left)
        return low + (high - low) / 2

    cpu_mean = _mean_positive(timings[0][0], timings[1][0])
    cuda_mean = _mean_positive(timings[0][1], timings[1][1])
    if not _finite_positive(cpu_mean) or not _finite_positive(cuda_mean):
        raise ValueError("mean whole-request timing is not finite and positive")
    if cpu_mean == cuda_mean:
        raise ValueError("mean whole-request timings tie")
    winner = "cuda" if cuda_mean < cpu_mean else "cpu"
    if winner != winners[0]:
        raise ValueError("mean whole-request winner differs from the order-specific winner")
    return SourceRouteProfileQualification(
        context=expected_context,
        winning_backend=winner,
        source_tile_sizes=(
            ("cpu", records[0].cpu.source_tile_size),
            ("cuda", records[0].cuda.source_tile_size),
        ),
        actual_tile_sizes=(
            ("cpu", records[0].cpu.actual_tile_size),
            ("cuda", records[0].cuda.actual_tile_size),
        ),
        execution_config_sha256=(
            ("cpu", records[0].cpu.execution_config_sha256),
            ("cuda", records[0].cuda.execution_config_sha256),
        ),
        run_orders=(orders[0], orders[1]),
        timings_seconds=(timings[0], timings[1]),
        mean_whole_request_seconds=(("cpu", cpu_mean), ("cuda", cuda_mean)),
        correctness_digest_sha256=_comparison_digest(comparisons[0]),
    )


__all__ = (
    "SCHEMA_VERSION", "BackendRouteProfileMeasurement",
    "CounterbalancedRouteProfileRecord", "MetricComparisonReceipt",
    "MetricOutputReceipt", "SourceRouteProfileContext",
    "SourceRouteProfileQualification", "route_profile_execution_config_sha256",
    "validate_source_route_profile_qualification",
)
