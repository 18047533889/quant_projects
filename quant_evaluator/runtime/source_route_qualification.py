"""Fail-closed qualification records for a single source execution route.

This module validates caller-supplied, immutable receipts. It does not attest
that the declared code was loaded, independently prove numeric correctness,
select routes, or run calibration. Consumers must trust the receipt producer
and its numerical oracle before treating ``correctness_validated`` as evidence.

Example (trusted producer receipts; this alone does not change API defaults)::

    digest = "0" * 64
    context = SourceRouteContext(
        request_content_sha256=digest, source_identity_sha256=digest,
        source_content_sha256=digest, executable_source_sha256=digest,
        runtime_fingerprint_sha256=digest, package_fingerprint_sha256=digest,
        thread_fingerprint_sha256=digest, device_fingerprint_sha256=digest,
        config_fingerprint_sha256=digest, request_shape=(10, 20, 2),
        metric_ids=("rank_ic",), expected_coverage_count=20,
        requested_tile_size=2, effective_tile_size=1)
    # Construct cpu_first and cuda_first from the trusted benchmark producer.
    result = validate_source_route_qualification(
        (cpu_first, cuda_first), expected_context=context)
    explicit_backend = "cuda_strict" if result.winning_backend == "cuda" else "cpu"
    # Pass explicit_backend to the existing evaluation API; no default is changed.

Historical receipts without actual thread/runtime identity are insufficient;
this validator is not wired into public routing. Keep backend selection explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Tuple


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BACKENDS = ("cpu", "cuda")


def _sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except (OverflowError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class SourceRouteContext:
    """Exact request, source content and execution environment declared by caller.

    ``source_content_sha256`` must identify the actual DataAccess content or a
    bounded content manifest. A snapshot label/ID alone is not content identity.
    Fingerprints are caller declarations and do not prove loaded-code identity.
    """

    request_content_sha256: str
    source_identity_sha256: str
    source_content_sha256: str
    executable_source_sha256: str
    runtime_fingerprint_sha256: str
    package_fingerprint_sha256: str
    thread_fingerprint_sha256: str
    device_fingerprint_sha256: str
    config_fingerprint_sha256: str
    request_shape: Tuple[int, ...]
    metric_ids: Tuple[str, ...]
    expected_coverage_count: int
    requested_tile_size: int
    effective_tile_size: int


@dataclass(frozen=True)
class BackendMeasurement:
    backend: str
    seconds: float
    oom_count: int
    correctness_digest_sha256: str
    correctness_validated: bool
    coverage_expected: int
    coverage_observed: int


@dataclass(frozen=True)
class CounterbalancedABRecord:
    context: SourceRouteContext
    execution_order: Tuple[str, str]
    cpu: BackendMeasurement
    cuda: BackendMeasurement


@dataclass(frozen=True)
class SourceRouteQualification:
    context: SourceRouteContext
    winning_backend: str
    run_orders: Tuple[Tuple[str, str], Tuple[str, str]]
    timings_seconds: Tuple[Tuple[float, float], Tuple[float, float]]
    correctness_digest_sha256: str


def _validate_context(context: object) -> None:
    """Validate each context without Python's bool/int equality shortcuts."""
    if type(context) is not SourceRouteContext:
        raise ValueError("context must be a SourceRouteContext")
    hashes = (
        "request_content_sha256", "source_identity_sha256",
        "source_content_sha256", "executable_source_sha256",
        "runtime_fingerprint_sha256", "package_fingerprint_sha256",
        "thread_fingerprint_sha256", "device_fingerprint_sha256",
        "config_fingerprint_sha256",
    )
    if any(not _sha256(getattr(context, key)) for key in hashes):
        raise ValueError("context contains a missing or malformed SHA-256 identity")
    if (type(context.request_shape) is not tuple or len(context.request_shape) != 3
            or any(not _positive_int(v) for v in context.request_shape)):
        raise ValueError("request shape must contain exactly three positive integers")
    if (type(context.metric_ids) is not tuple or not context.metric_ids
            or any(type(v) is not str or not v or v.strip() != v for v in context.metric_ids)
            or len(set(context.metric_ids)) != len(context.metric_ids)):
        raise ValueError("metric identifiers must be unique nonempty strings")
    if not _positive_int(context.expected_coverage_count):
        raise ValueError("expected coverage count must be a positive integer")
    if (not _positive_int(context.requested_tile_size)
            or not _positive_int(context.effective_tile_size)
            or context.effective_tile_size > context.requested_tile_size
            or context.effective_tile_size > context.request_shape[-1]):
        raise ValueError("requested/effective tile sizes are invalid")


def validate_source_route_qualification(
    records: Tuple[CounterbalancedABRecord, CounterbalancedABRecord],
    *,
    expected_context: SourceRouteContext,
) -> SourceRouteQualification:
    """Validate exactly two same-context, opposite-order CPU/CUDA receipts.

    Both trials must have validated identical correctness/coverage, zero backend
    OOM retries, finite positive timings, and the same strict winner in both
    execution orders.
    Raises ``ValueError`` for any missing, stale or malformed field.
    """
    _validate_context(expected_context)
    context = expected_context
    if type(records) is not tuple or len(records) != 2:
        raise ValueError("exactly two counterbalanced records are required")
    orders = []
    timings = []
    correctness_digests = []
    coverage_counts = []
    winners = []
    for record in records:
        if type(record) is not CounterbalancedABRecord:
            raise ValueError("counterbalanced record has an invalid type")
        _validate_context(record.context)
        if record.context != context:
            raise ValueError("record context is missing, stale, or differs from expected context")
    for record in records:
        order = record.execution_order
        if type(order) is not tuple or order not in (("cpu", "cuda"), ("cuda", "cpu")):
            raise ValueError("execution order must contain CPU and CUDA exactly once")
        orders.append(order)
        if type(record.cpu) is not BackendMeasurement or type(record.cuda) is not BackendMeasurement:
            raise ValueError("CPU and CUDA measurements are required")
        measurements = (record.cpu, record.cuda)
        for expected_backend, measurement in zip(_BACKENDS, measurements):
            if type(measurement.backend) is not str or measurement.backend != expected_backend:
                raise ValueError("backend measurement identity is inconsistent")
            if not _positive_finite(measurement.seconds):
                raise ValueError("backend timing must be finite and positive")
            if type(measurement.oom_count) is not int or measurement.oom_count < 0:
                raise ValueError("OOM count must be a nonnegative integer")
            if measurement.oom_count != 0:
                raise ValueError("backend receipt reports OOM retries")
            if measurement.correctness_validated is not True:
                raise ValueError("correctness validation receipt is absent")
            if not _sha256(measurement.correctness_digest_sha256):
                raise ValueError("correctness digest is missing or malformed")
            if (not _positive_int(measurement.coverage_expected)
                    or type(measurement.coverage_observed) is not int
                    or measurement.coverage_observed != measurement.coverage_expected
                    or measurement.coverage_expected != context.expected_coverage_count):
                raise ValueError("coverage receipt is incomplete or inconsistent")
        cpu, cuda = measurements
        if (cpu.correctness_digest_sha256 != cuda.correctness_digest_sha256
                or cpu.coverage_expected != cuda.coverage_expected
                or cpu.coverage_observed != cuda.coverage_observed):
            raise ValueError("CPU and CUDA correctness/coverage receipts differ")
        if cuda.seconds == cpu.seconds:
            raise ValueError("backend timings tie; no route winner is established")
        winners.append("cuda" if cuda.seconds < cpu.seconds else "cpu")
        correctness_digests.append(cpu.correctness_digest_sha256)
        coverage_counts.append(cpu.coverage_expected)
        timings.append((cpu.seconds, cuda.seconds))

    if orders[0] == orders[1]:
        raise ValueError("records must use opposite CPU/CUDA execution orders")
    if correctness_digests[0] != correctness_digests[1]:
        raise ValueError("counterbalanced runs do not share one correctness digest")
    if coverage_counts[0] != coverage_counts[1]:
        raise ValueError("counterbalanced runs do not share one coverage count")
    if winners[0] != winners[1]:
        raise ValueError("backend winner changes with execution order")
    return SourceRouteQualification(
        context=context,
        winning_backend=winners[0],
        run_orders=(orders[0], orders[1]),
        timings_seconds=(timings[0], timings[1]),
        correctness_digest_sha256=correctness_digests[0],
    )


__all__ = (
    "BackendMeasurement", "CounterbalancedABRecord", "SourceRouteContext",
    "SourceRouteQualification", "validate_source_route_qualification",
)
