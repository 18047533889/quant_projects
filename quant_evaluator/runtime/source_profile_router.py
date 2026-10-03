"""Live, request-bound routing support for v2 source route profiles.

The v2 receipts in :mod:`source_route_profiles` are caller-trusted benchmark
records. This module binds them to a fresh DataAccess COS manifest, strict QE
source digest, runtime/thread/device identity, exact request, current policy,
and the distinct live source/GPU tile ceilings. It never reads a pilot tile or
performs calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_tile_source import admitted_source_tile_limit
from quant_evaluator.runtime import source_qualification_cache as profile_cache
from quant_evaluator.runtime.source_qualified_router import (
    QUALIFICATION_SCOPE, SOURCE_CONTENT_SCOPE, SourceQualificationError,
    _capture_live_context_state, _policy_fields, _preliminary_cache_key,
)
from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement, CounterbalancedRouteProfileRecord,
    SourceRouteProfileContext, SourceRouteProfileQualification,
    route_profile_execution_config_sha256,
    validate_source_route_profile_qualification,
)
from quant_evaluator.runtime.source_auto_evidence import SOURCE_AUTO_METRICS


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SERIES_METRICS = frozenset({"rank_ic_series", "pearson_ic_series"})
_METRIC_ABS_ERROR_TOLERANCE = 1e-10
PROFILE_QUALIFICATION_SCOPE = "exact_request_bound_cos_runtime_policy_profiles_v2"
PROFILE_CACHE_TAG = "SourceRouteProfileCache.v2"
TIMING_SCOPE = "evaluate_factor_source_batch_wall_v1"
EXECUTION_SCHEDULE_SCOPE = "zero_oom_source_equals_compute_v1"


class SourceProfileQualificationError(ValueError):
    """Safe short reason for declining a v2 current-source profile."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class QualifiedSourceProfileRoute:
    qualification: SourceRouteProfileQualification
    context: SourceRouteProfileContext
    cpu_profile: BackendRouteProfileMeasurement
    cuda_profile: BackendRouteProfileMeasurement
    winning_backend: str
    winner_profile: BackendRouteProfileMeasurement
    evidence_sha256: str
    cache_status: str
    scope: str = PROFILE_QUALIFICATION_SCOPE
    source_content_scope: str = SOURCE_CONTENT_SCOPE


def _live_tile_limits(*, source, metadata, requested_tile_size: int,
                      policy: GPUExecutionPolicy) -> tuple[int, int]:
    if type(requested_tile_size) is not int or requested_tile_size <= 0:
        raise SourceProfileQualificationError("requested_tile_invalid")
    try:
        source_limit = admitted_source_tile_limit(
            metadata.max_tile_size, getattr(source, "admitted_max_tile_size", None))
    except (TypeError, ValueError) as exc:
        raise SourceProfileQualificationError("source_tile_admission_invalid") from exc
    factor_count = len(metadata.factor_ids)
    source_ceiling = min(requested_tile_size, source_limit, factor_count)
    if type(source_ceiling) is not int or source_ceiling <= 0:
        raise SourceProfileQualificationError("source_tile_admission_invalid")
    gpu_ceiling = min(source_ceiling,
                      policy.max_factor_tile_size or source_ceiling)
    return source_ceiling, gpu_ceiling


def _metric_contract(metrics, request_shape) -> tuple[tuple, tuple, tuple]:
    selected = tuple(metrics)
    if (not selected or any(type(item) is not str or not item for item in selected)
            or len(set(selected)) != len(selected)
            or set(selected) - SOURCE_AUTO_METRICS):
        raise SourceProfileQualificationError("profile_metrics_invalid")
    T, _, F = request_shape
    coverage = tuple((metric, T * F if metric in _SERIES_METRICS else F)
                     for metric in selected)
    tolerances = tuple((metric, _METRIC_ABS_ERROR_TOLERANCE) for metric in selected)
    return selected, coverage, tolerances


def capture_source_route_profile_context(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, policy: GPUExecutionPolicy,
) -> SourceRouteProfileContext:
    """Capture the live shared request context and separate source/GPU caps."""
    if not isinstance(policy, GPUExecutionPolicy):
        raise SourceProfileQualificationError("gpu_policy_invalid")
    request_shape = (metadata.time_axis.size, metadata.asset_axis.size,
                     len(metadata.factor_ids))
    selected, metric_coverage, metric_tolerances = _metric_contract(metrics, request_shape)
    source_ceiling, gpu_ceiling = _live_tile_limits(
        source=source, metadata=metadata,
        requested_tile_size=requested_tile_size, policy=policy)
    lookup_key = _preliminary_cache_key(
        source=source, metadata=metadata, metrics=selected,
        request_fingerprint=request_fingerprint,
        requested_tile_size=requested_tile_size,
        maximum_effective_tile_size=source_ceiling, policy=policy,
    )
    try:
        state = _capture_live_context_state(
            source=source, metadata=metadata, metrics=selected,
            request_fingerprint=request_fingerprint,
            requested_tile_size=requested_tile_size,
            maximum_effective_tile_size=source_ceiling,
            policy=policy, lookup_key=lookup_key,
        )
    except SourceQualificationError as exc:
        raise SourceProfileQualificationError(exc.reason) from exc
    except Exception as exc:
        raise SourceProfileQualificationError("profile_live_context_capture_failed") from exc
    common_config = stable_content_hex(
        tag="SourceRouteProfileCommonConfig.v2",
        fields={
            "configuration_base": dict(state.configuration_base),
            "requested_tile_size": requested_tile_size,
            "live_source_admitted_max_tile_size": source_ceiling,
            "gpu_admitted_max_tile_size": gpu_ceiling,
            "metric_coverage": metric_coverage,
            "metric_error_tolerances": metric_tolerances,
            "timing_scope": TIMING_SCOPE,
        },
    )
    base = state.context
    return SourceRouteProfileContext(
        request_content_sha256=base.request_content_sha256,
        source_identity_sha256=base.source_identity_sha256,
        source_content_sha256=base.source_content_sha256,
        executable_source_sha256=base.executable_source_sha256,
        runtime_fingerprint_sha256=base.runtime_fingerprint_sha256,
        package_fingerprint_sha256=base.package_fingerprint_sha256,
        thread_fingerprint_sha256=base.thread_fingerprint_sha256,
        device_fingerprint_sha256=base.device_fingerprint_sha256,
        common_config_fingerprint_sha256=common_config,
        request_shape=request_shape, metric_ids=selected,
        expected_coverage_count=sum(count for _, count in metric_coverage),
        requested_tile_size=requested_tile_size,
        timing_scope=TIMING_SCOPE,
        metric_coverage=metric_coverage,
        metric_error_tolerances=metric_tolerances,
        live_source_admitted_max_tile_size=source_ceiling,
        gpu_admitted_max_tile_size=gpu_ceiling,
    )


def _profile_cache_key(*, source, metadata, metrics, request_fingerprint,
                       requested_tile_size, policy) -> str:
    selected = tuple(metrics)
    source_ceiling, gpu_ceiling = _live_tile_limits(
        source=source, metadata=metadata,
        requested_tile_size=requested_tile_size, policy=policy)
    source_config = {
        "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
        "snapshot_id": metadata.snapshot_id,
        "source_max_tile_size": metadata.max_tile_size,
        "source_admitted_tile_size": getattr(source, "admitted_max_tile_size", None),
        "source_prefetch_mode": getattr(source, "prefetch_mode", None),
        "source_prefetch_workers": getattr(source, "prefetch_workers", None),
        "source_max_source_memory_bytes": getattr(source, "max_source_memory_bytes", None),
        "source_max_prefetch_memory_bytes": getattr(source, "max_prefetch_memory_bytes", None),
        "source_extra_assembly_bytes_per_cell": getattr(source, "extra_assembly_bytes_per_cell", None),
    }
    return stable_content_hex(
        tag=PROFILE_CACHE_TAG,
        fields={
            "request_fingerprint": request_fingerprint,
            "metrics": selected,
            "requested_tile_size": requested_tile_size,
            "live_source_admitted_max_tile_size": source_ceiling,
            "gpu_admitted_max_tile_size": gpu_ceiling,
            "policy": _policy_fields(policy),
            "source": source_config,
        },
    )


def is_source_route_profile_pair(records: object) -> bool:
    return (type(records) is tuple and len(records) == 2
            and all(type(item) is CounterbalancedRouteProfileRecord for item in records))


def get_cached_source_route_profile_records(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, policy: GPUExecutionPolicy,
) -> tuple[CounterbalancedRouteProfileRecord, CounterbalancedRouteProfileRecord] | None:
    """Cheaply look up the v2 namespace before any full live identity capture."""
    try:
        key = _profile_cache_key(
            source=source, metadata=metadata, metrics=metrics,
            request_fingerprint=request_fingerprint,
            requested_tile_size=requested_tile_size, policy=policy)
    except (TypeError, ValueError, SourceProfileQualificationError):
        return None
    records = profile_cache.get_validated_records(key)
    if not is_source_route_profile_pair(records):
        if records is not None:
            profile_cache.discard_validated_records(key)
        return None
    return records


def discard_source_route_profile_cache(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, policy: GPUExecutionPolicy,
) -> None:
    try:
        key = _profile_cache_key(
            source=source, metadata=metadata, metrics=metrics,
            request_fingerprint=request_fingerprint,
            requested_tile_size=requested_tile_size, policy=policy)
    except Exception:
        return
    profile_cache.discard_validated_records(key)


def qualify_source_route_profiles(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, policy: GPUExecutionPolicy,
    records: tuple[CounterbalancedRouteProfileRecord,
                   CounterbalancedRouteProfileRecord] | None,
) -> QualifiedSourceProfileRoute:
    """Bind caller receipts or a cached pair to this exact live route context."""
    if not isinstance(policy, GPUExecutionPolicy):
        raise SourceProfileQualificationError("gpu_policy_invalid")
    try:
        key = _profile_cache_key(
            source=source, metadata=metadata, metrics=metrics,
            request_fingerprint=request_fingerprint,
            requested_tile_size=requested_tile_size, policy=policy)
    except SourceProfileQualificationError:
        raise
    except Exception as exc:
        raise SourceProfileQualificationError("profile_cache_key_invalid") from exc
    cache_status = "supplied"
    if records is None:
        records = profile_cache.get_validated_records(key)
        cache_status = "cache_hit" if records is not None else "cache_miss"
        if records is None:
            raise SourceProfileQualificationError("qualified_profile_cache_miss")
        if not is_source_route_profile_pair(records):
            profile_cache.discard_validated_records(key)
            raise SourceProfileQualificationError("qualified_profile_cache_malformed")
    try:
        expected_context = capture_source_route_profile_context(
            source=source, metadata=metadata, metrics=metrics,
            request_fingerprint=request_fingerprint,
            requested_tile_size=requested_tile_size, policy=policy,
        )
        factor_count = expected_context.request_shape[-1]
        reachable_gpu_widths = {
            min(factor_count, expected_context.requested_tile_size,
                expected_context.live_source_admitted_max_tile_size, candidate)
            for candidate in policy.factor_tile_candidates()
        }
        if any(item.cuda.source_tile_size not in reachable_gpu_widths
               for item in records):
            raise ValueError("CUDA profile width is not reachable under live policy")
        qualification = validate_source_route_profile_qualification(
            records, expected_context=expected_context)
    except SourceQualificationError as exc:
        if cache_status == "cache_hit":
            profile_cache.discard_validated_records(key)
        raise SourceProfileQualificationError(exc.reason) from exc
    except (TypeError, ValueError, OverflowError) as exc:
        if cache_status == "cache_hit":
            profile_cache.discard_validated_records(key)
        raise SourceProfileQualificationError("qualified_profile_receipt_mismatch") from exc
    except Exception as exc:
        if cache_status == "cache_hit":
            profile_cache.discard_validated_records(key)
        raise SourceProfileQualificationError("profile_live_validation_failed") from exc
    if cache_status == "supplied":
        profile_cache.remember_validated_records(key, records)
    winner_profile = (records[0].cpu if qualification.winning_backend == "cpu"
                      else records[0].cuda)
    evidence_sha = stable_content_hex(
        tag="LiveSourceRouteProfileQualification.v2",
        fields={
            "context": tuple(getattr(expected_context, field)
                             for field in expected_context.__dataclass_fields__),
            "winner": qualification.winning_backend,
            "source_tile_sizes": qualification.source_tile_sizes,
            "actual_tile_sizes": qualification.actual_tile_sizes,
            "execution_config_sha256": qualification.execution_config_sha256,
            "timings_seconds": qualification.timings_seconds,
            "correctness_digest_sha256": qualification.correctness_digest_sha256,
            "scope": PROFILE_QUALIFICATION_SCOPE,
            "source_content_scope": SOURCE_CONTENT_SCOPE,
        },
    )
    return QualifiedSourceProfileRoute(
        qualification=qualification, context=expected_context,
        cpu_profile=records[0].cpu, cuda_profile=records[0].cuda,
        winning_backend=qualification.winning_backend,
        winner_profile=winner_profile, evidence_sha256=evidence_sha,
        cache_status=cache_status,
    )


def source_profile_cache_key(**kwargs) -> str:
    """Expose the namespace key for deterministic lifecycle checks."""
    return _profile_cache_key(**kwargs)


__all__ = (
    "EXECUTION_SCHEDULE_SCOPE", "PROFILE_QUALIFICATION_SCOPE", "TIMING_SCOPE",
    "QualifiedSourceProfileRoute", "SourceProfileQualificationError",
    "capture_source_route_profile_context", "discard_source_route_profile_cache",
    "get_cached_source_route_profile_records", "is_source_route_profile_pair",
    "qualify_source_route_profiles", "route_profile_execution_config_sha256",
    "source_profile_cache_key",
)


def validate_source_route_profile_execution(
    *, source, output, profile: BackendRouteProfileMeasurement,
    expected_context: SourceRouteProfileContext, metadata, metrics,
    request_fingerprint: str, requested_tile_size: int,
    policy: GPUExecutionPolicy, backend: str,
) -> str | None:
    """Return a short mismatch reason for actual reads and post-run live state."""
    ranges = getattr(source, "reads", None)
    expected_ranges = profile.source_ranges
    valid_ranges = (
        type(ranges) in (list, tuple)
        and all(type(item) is tuple and len(item) == 2
                and type(item[0]) is int and type(item[1]) is int
                for item in ranges)
        and tuple(ranges) == expected_ranges
    )
    tile_count = output.metadata.get("factor_tiles_processed")
    valid_count = (type(tile_count) is int
                   and tile_count == len(expected_ranges))
    valid_width = True
    if backend == "cuda":
        actual_width = output.metadata.get("factor_tile_size")
        oom_retries = output.metadata.get("oom_retries")
        valid_width = (type(actual_width) is int
                       and actual_width == profile.source_tile_size
                       and type(oom_retries) is int and oom_retries == 0)
    if not valid_ranges or not valid_count or not valid_width:
        return "qualified_profile_execution_receipt_deviated"
    try:
        post_context = capture_source_route_profile_context(
            source=source, metadata=metadata, metrics=metrics,
            request_fingerprint=request_fingerprint,
            requested_tile_size=requested_tile_size, policy=policy)
    except Exception:
        return "qualified_profile_post_context_unavailable"
    if post_context != expected_context:
        return "qualified_profile_live_context_changed"
    return None
