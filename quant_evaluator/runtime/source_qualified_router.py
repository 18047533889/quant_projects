"""Live-context qualification and routing for optional source A/B evidence.

This guard consumes caller-trusted benchmark receipts and injected DataAccess
bound-manifest helper output. It validates
current QE source bytes, the active runtime and thread/device fingerprints, a
bound DataAccess COS manifest, the exact request and current execution policy.
It does not prove that declared code was loaded or independently recompute the
producer's numerical oracle. The producer must trust its injected DataAccess
bound-manifest helpers and call :func:`capture_source_route_context` before and
after its counterbalanced pair, requiring identical contexts. It never
calibrates or reads factor panels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from pathlib import Path
from typing import Any, Mapping

from quant_evaluator.adapters.cos_factor_tile_source import (
    BoundCosFactor, CosFactorTileSource,
)
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.source_identity import ProcessSourceIdentity
from quant_evaluator.runtime.source_dependency_identity import (
    COMPONENT_MODULES as DEPENDENCY_IDENTITY_COMPONENTS,
    SCHEMA_VERSION as DEPENDENCY_IDENTITY_SCHEMA,
    capture_source_dependency_identity,
)
from quant_evaluator.runtime.source_route_qualification import (
    CounterbalancedABRecord, SourceRouteContext, SourceRouteQualification,
    validate_source_route_qualification,
)
from quant_evaluator.runtime.source_runtime_identity import (
    capture_source_runtime_identity, is_qualified_source_runtime_identity,
    source_runtime_identity_digest,
)
from quant_evaluator.runtime.source_qualification_cache import (
    discard_validated_records, get_validated_records,
    remember_validated_records,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_CONTENT_SCOPE = "qe_da_fo_fp_python_dependency_content_v1"
QUALIFICATION_SCOPE = "exact_request_bound_cos_runtime_policy_v1"
_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3


class SourceQualificationError(ValueError):
    """Safe short reason for declining current source qualification."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class QualifiedSourceRoute:
    qualification: SourceRouteQualification
    effective_tile_size: int
    evidence_sha256: str
    cache_status: str
    scope: str = QUALIFICATION_SCOPE
    source_content_scope: str = SOURCE_CONTENT_SCOPE


def _require_sha256(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SourceQualificationError("bound_source_identity_invalid")
    return value


def _cos_source_identities(source: object, metadata) -> tuple[str, str]:
    """Check selected rows against the caller-trusted captured COS manifest.

    This does not authenticate the constructor, callback, DataAccess origin, or
    benchmark producer. Receipts are caller-trusted, not signed or attested.
    """
    if not isinstance(source, CosFactorTileSource):
        raise SourceQualificationError("source_is_not_bound_cos_tile_source")
    if getattr(source, "_closed", True) is not False:
        raise SourceQualificationError("bound_cos_source_closed")
    snapshot = getattr(source, "manifest_snapshot", None)
    manifest_sha = _require_sha256(getattr(source, "manifest_sha256", None))
    if getattr(snapshot, "manifest_sha256", None) != manifest_sha:
        raise SourceQualificationError("bound_cos_manifest_snapshot_mismatch")
    rows = getattr(snapshot, "factors", None)
    source_records = getattr(source, "records", None)
    if not isinstance(rows, Mapping) or type(source_records) is not tuple:
        raise SourceQualificationError("bound_cos_manifest_records_unavailable")
    if (tuple(record.factor_id for record in source_records
               if type(record) is BoundCosFactor) != metadata.factor_ids
            or len(source_records) != len(metadata.factor_ids)):
        raise SourceQualificationError("bound_cos_factor_order_mismatch")

    content_records = []
    for factor_id, record in zip(metadata.factor_ids, source_records):
        if type(record) is not BoundCosFactor:
            raise SourceQualificationError("bound_cos_factor_record_invalid")
        row = rows.get(factor_id)
        if type(row) is not dict:
            raise SourceQualificationError("bound_cos_manifest_row_missing")
        sha = _require_sha256(record.sha256)
        size = record.size_bytes
        uri = record.uri
        if (type(size) is not int or size <= 0 or type(uri) is not str or not uri
                or row.get("uri") != uri or row.get("sha256") != sha
                or type(row.get("bytes")) is not int or row["bytes"] != size):
            raise SourceQualificationError("bound_cos_selected_record_mismatch")
        content_records.append((factor_id, sha, size))

    # Verifies that this captured DataAccess manifest remains current. It does
    # not read or decode any factor panels.
    verify = getattr(source, "verify_manifest", None)
    if not callable(verify):
        raise SourceQualificationError("bound_cos_manifest_verifier_unavailable")
    try:
        verify(snapshot)
    except Exception as exc:
        raise SourceQualificationError("bound_cos_manifest_verification_failed") from exc

    source_identity = stable_content_hex(
        tag="BoundCosSourceIdentity.v1",
        fields={"manifest_sha256": manifest_sha, "snapshot_id": metadata.snapshot_id,
                "factor_ids": metadata.factor_ids, "dtype": metadata.dtype,
                "shape": (metadata.time_axis.size, metadata.asset_axis.size,
                          len(metadata.factor_ids))},
    )
    source_content = stable_content_hex(
        tag="BoundCosSelectedContent.v1",
        fields={"manifest_sha256": manifest_sha, "records": content_records},
    )
    return source_identity, source_content


def _policy_fields(policy: GPUExecutionPolicy) -> dict[str, Any]:
    return {
        "device_ids": tuple(policy.device_ids),
        "max_vram_fraction": policy.max_vram_fraction,
        "pinned_host_memory": policy.pinned_host_memory,
        "async_transfer": policy.async_transfer,
        "double_buffer": policy.double_buffer,
        "precision_policy": policy.precision_policy.value,
        "oom_retile": policy.oom_retile,
        "strict_backend": policy.strict_backend,
        "required_capabilities": tuple(policy.required_capabilities),
        "max_host_result_bytes": policy.max_host_result_bytes,
        "prefer_resident_labels": policy.prefer_resident_labels,
        "max_factor_tile_size": policy.max_factor_tile_size,
    }


@dataclass(frozen=True)
class _LiveContextState:
    context: SourceRouteContext
    lookup_key: str
    maximum_effective_tile_size: int
    configuration_base: Mapping[str, Any]

    def for_effective_width(self, width: int) -> SourceRouteContext:
        if type(width) is not int or not 1 <= width <= self.maximum_effective_tile_size:
            raise SourceQualificationError("effective_tile_outside_live_admission")
        config = dict(self.configuration_base)
        config["effective_tile_size"] = width
        config_hash = stable_content_hex(tag="SourceRouteConfig.v1", fields=config)
        return replace(self.context, effective_tile_size=width,
                       config_fingerprint_sha256=config_hash)


def _capture_live_context_state(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, maximum_effective_tile_size: int,
    policy: GPUExecutionPolicy, lookup_key: str,
) -> _LiveContextState:
    if type(requested_tile_size) is not int or requested_tile_size <= 0:
        raise SourceQualificationError("requested_tile_invalid")
    if (type(maximum_effective_tile_size) is not int
            or not 1 <= maximum_effective_tile_size <= requested_tile_size):
        raise SourceQualificationError("effective_tile_admission_invalid")
    request_sha = _require_sha256(request_fingerprint)
    source_identity_sha, source_content_sha = _cos_source_identities(source, metadata)

    # The scope is deliberately all Python files in quant_evaluator, not the
    # historical benchmark harness directory list.
    source_root = Path(__file__).resolve().parents[1]
    try:
        source_receipt = ProcessSourceIdentity(
            source_root, strict_full_content=True,
            max_records=10000, max_bytes=256 * 1024 * 1024,
        ).identify(strict_full_content=True)
    except Exception as exc:
        raise SourceQualificationError("current_qe_source_identity_unavailable") from exc
    if (source_receipt.full_content_checked is not True
            or source_receipt.strategy != "strict_full_content"
            or source_receipt.drifted is not False
            or type(source_receipt.digest) is not str
            or _SHA256.fullmatch(source_receipt.digest) is None):
        raise SourceQualificationError("current_qe_source_identity_incomplete")

    try:
        dependency_receipt = capture_source_dependency_identity()
    except Exception as exc:
        raise SourceQualificationError("current_dependency_source_identity_unavailable") from exc
    if (dependency_receipt.schema != DEPENDENCY_IDENTITY_SCHEMA
            or type(dependency_receipt.digest) is not str
            or _SHA256.fullmatch(dependency_receipt.digest) is None):
        raise SourceQualificationError("current_dependency_source_identity_incomplete")
    component_digests = getattr(dependency_receipt, "component_digests", None)
    required_components = DEPENDENCY_IDENTITY_COMPONENTS
    if type(component_digests) is not tuple or len(component_digests) != len(required_components):
        raise SourceQualificationError("current_dependency_component_binding_invalid")
    bound_digests = {}
    for component in component_digests:
        if (type(component) is not tuple or len(component) != 2
                or type(component[0]) is not str
                or type(component[1]) is not str
                or _SHA256.fullmatch(component[1]) is None
                or component[0] in bound_digests):
            raise SourceQualificationError("current_dependency_component_binding_invalid")
        bound_digests[component[0]] = component[1]
    if tuple(bound_digests) != required_components:
        raise SourceQualificationError("current_dependency_component_binding_invalid")
    if bound_digests["quant_evaluator"] != source_receipt.digest:
        raise SourceQualificationError("current_qe_dependency_identity_mismatch")
    executable_sha = stable_content_hex(
        tag="SourceRouteExecutableClosure.v1",
        fields={"qe": source_receipt.digest, "dependency_closure": dependency_receipt.digest},
    )

    try:
        runtime = capture_source_runtime_identity()
        if not is_qualified_source_runtime_identity(runtime):
            raise SourceQualificationError("current_runtime_identity_incomplete")
        runtime_sha = source_runtime_identity_digest(runtime)
    except SourceQualificationError:
        raise
    except Exception as exc:
        raise SourceQualificationError("current_runtime_identity_unavailable") from exc
    cuda = runtime.get("cuda")
    device = cuda.get("device") if isinstance(cuda, Mapping) else None
    if (not isinstance(cuda, Mapping) or cuda.get("status") != "captured"
            or cuda.get("context_initialized") is not True
            or not isinstance(device, Mapping)):
        raise SourceQualificationError("current_cuda_device_identity_unavailable")
    if (len(policy.device_ids) != 1 or type(policy.device_ids[0]) is not int
            or device.get("id") != policy.device_ids[0]):
        raise SourceQualificationError("current_cuda_device_differs_from_policy")

    packages_sha = stable_content_hex(
        tag="SourceRoutePackages.v1",
        fields={"python": runtime["python"], "packages": runtime["packages"]},
    )
    thread_sha = stable_content_hex(
        tag="SourceRouteThreads.v1",
        fields={"numba": runtime["numba"], "threadpools": runtime["threadpools"],
                "thread_environment": runtime["thread_environment"]},
    )
    device_sha = stable_content_hex(
        tag="SourceRouteDevice.v1",
        fields={"cuda": runtime["cuda"], "cupy": runtime["packages"]["cupy"]},
    )
    request_shape = (metadata.time_axis.size, metadata.asset_axis.size,
                     len(metadata.factor_ids))
    source_config = {
        "source_max_tile_size": metadata.max_tile_size,
        "source_admitted_tile_size": getattr(source, "admitted_max_tile_size", None),
        "source_prefetch_mode": getattr(source, "prefetch_mode", None),
        "source_prefetch_workers": getattr(source, "prefetch_workers", None),
        "source_max_source_memory_bytes": getattr(source, "max_source_memory_bytes", None),
        "source_max_prefetch_memory_bytes": getattr(source, "max_prefetch_memory_bytes", None),
        "source_extra_assembly_bytes_per_cell": getattr(source, "extra_assembly_bytes_per_cell", None),
    }
    configuration_base = {
        "policy": _policy_fields(policy), "source": source_config,
        "requested_tile_size": requested_tile_size,
    }
    full_config = dict(configuration_base)
    full_config["effective_tile_size"] = maximum_effective_tile_size
    output_coverage = sum(
        (request_shape[0] * request_shape[2]
         if metric in {"rank_ic_series", "pearson_ic_series"}
         else request_shape[2])
        for metric in metrics
    )
    context = SourceRouteContext(
        request_content_sha256=request_sha,
        source_identity_sha256=source_identity_sha,
        source_content_sha256=source_content_sha,
        executable_source_sha256=executable_sha,
        runtime_fingerprint_sha256=runtime_sha,
        package_fingerprint_sha256=packages_sha,
        thread_fingerprint_sha256=thread_sha,
        device_fingerprint_sha256=device_sha,
        config_fingerprint_sha256=stable_content_hex(
            tag="SourceRouteConfig.v1", fields=full_config),
        request_shape=request_shape, metric_ids=tuple(metrics),
        expected_coverage_count=output_coverage,
        requested_tile_size=requested_tile_size,
        effective_tile_size=maximum_effective_tile_size,
    )
    return _LiveContextState(context, lookup_key, maximum_effective_tile_size,
                             configuration_base)


def capture_source_route_context(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, maximum_effective_tile_size: int,
    effective_tile_size: int, policy: GPUExecutionPolicy,
) -> SourceRouteContext:
    """Capture the live context a trusted counterbalanced producer must record.

    ``request_fingerprint`` is the semantic request fingerprint returned by
    the validated source API call. Both CPU and CUDA runs must use the same
    factor tile width represented by ``effective_tile_size``; capture once
    before and once after the counterbalanced pair and require exact context
    equality. Do not use a CPU memory-admitted width and a different CUDA
    working width under one context. A smaller measured width only qualifies
    requests whose live maximum is no larger than that width; it does not
    establish performance for a larger default cap. This scope hashes all
    Python source under ``quant_evaluator`` and is distinct from older
    benchmark harness-only source digests.
    """
    state = _capture_live_context_state(
        source=source, metadata=metadata, metrics=metrics,
        request_fingerprint=request_fingerprint,
        requested_tile_size=requested_tile_size,
        maximum_effective_tile_size=maximum_effective_tile_size,
        policy=policy,
        lookup_key=_preliminary_cache_key(
            source=source, metadata=metadata, metrics=metrics,
            request_fingerprint=request_fingerprint,
            requested_tile_size=requested_tile_size,
            maximum_effective_tile_size=maximum_effective_tile_size,
            policy=policy),
    )
    return state.for_effective_width(effective_tile_size)


def _preliminary_cache_key(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, maximum_effective_tile_size: int,
    policy: GPUExecutionPolicy,
) -> str:
    """Cheap lookup key; a hit is always revalidated against all live hashes."""
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
        tag="SourceRouteValidatedCacheLookup.v1",
        fields={"request_sha256": _require_sha256(request_fingerprint),
                "shape": (metadata.time_axis.size, metadata.asset_axis.size,
                          len(metadata.factor_ids)),
                "metrics": tuple(metrics),
                "requested_tile_size": requested_tile_size,
                "maximum_effective_tile_size": maximum_effective_tile_size,
                "policy": _policy_fields(policy), "source": source_config},
    )


def discard_source_route_cache(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, maximum_effective_tile_size: int,
    policy: GPUExecutionPolicy,
) -> None:
    """Evict cached evidence when the post-execution configuration deviates."""
    key = _preliminary_cache_key(
        source=source, metadata=metadata, metrics=metrics,
        request_fingerprint=request_fingerprint,
        requested_tile_size=requested_tile_size,
        maximum_effective_tile_size=maximum_effective_tile_size,
        policy=policy,
    )
    discard_validated_records(key)


def qualify_source_route(
    *, source, metadata, metrics, request_fingerprint: str,
    requested_tile_size: int, maximum_effective_tile_size: int,
    policy: GPUExecutionPolicy,
    records: tuple[CounterbalancedABRecord, CounterbalancedABRecord] | None,
) -> QualifiedSourceRoute:
    """Validate explicit receipts or retrieve a prior pair for this live request.

    A cache hit is never trusted by itself: it is rechecked against current
    content, executable source, runtime, threads, device and policy context.
    Missing or stale evidence raises a short reason; callers may then preserve
    their existing unqualified envelope route with an honest receipt status.
    """
    if not isinstance(policy, GPUExecutionPolicy):
        raise SourceQualificationError("gpu_policy_invalid")
    lookup_key = _preliminary_cache_key(
        source=source, metadata=metadata, metrics=metrics,
        request_fingerprint=request_fingerprint,
        requested_tile_size=requested_tile_size,
        maximum_effective_tile_size=maximum_effective_tile_size,
        policy=policy,
    )
    cache_status = "supplied"
    if records is None:
        records = get_validated_records(lookup_key)
        cache_status = "cache_hit" if records is not None else "cache_miss"
        if records is None:
            raise SourceQualificationError("qualified_cache_miss")
    try:
        state = _capture_live_context_state(
            source=source, metadata=metadata, metrics=metrics,
            request_fingerprint=request_fingerprint,
            requested_tile_size=requested_tile_size,
            maximum_effective_tile_size=maximum_effective_tile_size,
            policy=policy, lookup_key=lookup_key,
        )
    except SourceQualificationError:
        if cache_status == "cache_hit":
            discard_validated_records(lookup_key)
        raise
    except Exception as exc:
        if cache_status == "cache_hit":
            discard_validated_records(lookup_key)
        raise SourceQualificationError("live_context_capture_failed") from exc
    if type(records) is not tuple or len(records) != 2:
        raise SourceQualificationError("qualified_evidence_not_available")
    if any(type(item) is not CounterbalancedABRecord
           or type(item.context) is not SourceRouteContext for item in records):
        discard_validated_records(state.lookup_key)
        raise SourceQualificationError("qualified_evidence_malformed")
    widths = tuple(item.context.effective_tile_size for item in records)
    if any(type(width) is not int or width < 1 for width in widths):
        discard_validated_records(state.lookup_key)
        raise SourceQualificationError("qualified_effective_tile_mismatch")
    if any(width > maximum_effective_tile_size for width in widths):
        discard_validated_records(state.lookup_key)
        raise SourceQualificationError("effective_tile_outside_live_admission")
    if any(width < maximum_effective_tile_size for width in widths):
        discard_validated_records(state.lookup_key)
        raise SourceQualificationError("qualified_effective_tile_below_live_maximum")
    if widths[0] != widths[1]:
        discard_validated_records(state.lookup_key)
        raise SourceQualificationError("qualified_effective_tile_mismatch")
    expected = state.for_effective_width(widths[0])
    try:
        qualification = validate_source_route_qualification(
            records, expected_context=expected)
    except (ValueError, TypeError, OverflowError) as exc:
        discard_validated_records(state.lookup_key)
        raise SourceQualificationError("qualified_receipt_mismatch") from exc
    if cache_status == "supplied":
        remember_validated_records(state.lookup_key, records)
    evidence_sha = stable_content_hex(
        tag="LiveSourceRouteQualification.v1",
        fields={
            "context": {
                name: getattr(expected, name)
                for name in expected.__dataclass_fields__
            },
            "winner": qualification.winning_backend,
            "orders": qualification.run_orders,
            "timings": qualification.timings_seconds,
            "correctness_digest_sha256": qualification.correctness_digest_sha256,
            "scope": QUALIFICATION_SCOPE,
            "source_content_scope": SOURCE_CONTENT_SCOPE,
        },
    )
    return QualifiedSourceRoute(
        qualification=qualification, effective_tile_size=widths[0],
        evidence_sha256=evidence_sha, cache_status=cache_status,
    )


def cuda_route_rejection(policy: GPUExecutionPolicy) -> str | None:
    """Apply the evaluator's current auto-route resource/device admission."""
    from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection

    return _auto_batch_cuda_rejection(policy, _MIN_EFFECTIVE_VRAM_BYTES)


__all__ = (
    "QUALIFICATION_SCOPE", "SOURCE_CONTENT_SCOPE", "QualifiedSourceRoute",
    "SourceQualificationError", "capture_source_route_context",
    "discard_source_route_cache",
    "cuda_route_rejection", "qualify_source_route",
)
