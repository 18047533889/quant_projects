"""Choose one run-wide DAG only after its control-plane memory is admitted.

Panel values remain governed by the scheduler. This estimate reserves room for
decoded definitions, graph/index objects and IPC copies; it is not a measured
RSS or kernel allocator cap. Large/unusual graphs retain the bounded-wave path.
"""
from __future__ import annotations
import time
from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind


def admit_run_dag(manifest, broker, *, run_id, fallback_items, deadline,
                  check_cancellation=None, descriptor_budget_bytes=None):
    count = len(manifest)
    summary = {
        "schema_version": "factor_engine.run_dag_admission.v1",
        "requested_roots": count,
        "execution_scope": "bounded_waves",
        "cross_wave_value_reuse": False,
        "factor_limit": int(fallback_items),
        "definition_bytes": 0,
        "estimated_metadata_peak_bytes": 0,
        "estimated_descriptor_bytes": 0,
        "estimate_basis": "serialized_definitions_x32_plus_8192_per_root_plus_1MiB",
        "reason": "SMALL_BATCH_ALREADY_FITS",
    }
    if count <= fallback_items:
        return None, summary
    if descriptor_budget_bytes is not None:
        if type(descriptor_budget_bytes) is not int:
            raise ValueError("descriptor_budget_bytes must be an integer or None")
        summary["descriptor_budget_bytes"] = descriptor_budget_bytes
        if descriptor_budget_bytes <= 0:
            summary.update(reason="ARTIFACT_DESCRIPTORS_REQUIRE_BOUNDED_WAVES",
                           scanned_roots=0)
            return None, summary
    try:
        available = max(0, int(broker.execution_budget()))
    except (AttributeError, TypeError, ValueError, OverflowError):
        summary["reason"] = "LIVE_BUDGET_UNAVAILABLE"
        return None, summary
    # Leave at least three quarters of the current pool for read/compute/CSE/
    # output. The broker still checks all existing commitments at acquisition.
    ceiling = available // 4
    estimated = 1024 * 1024
    scanned = 0
    if estimated > ceiling:
        summary.update(reason="METADATA_REQUIRES_BOUNDED_WAVES",
                       estimated_metadata_peak_bytes=estimated,
                       scanned_roots=0)
        return None, summary
    records = iter(manifest.records(start=0, limit=count))
    while scanned < count:
        if scanned % 512 == 0:
            if check_cancellation is not None:
                check_cancellation()
            if time.monotonic() >= deadline:
                summary.update(reason="CATALOG_ADMISSION_DEADLINE",
                               estimated_metadata_peak_bytes=estimated,
                               scanned_roots=scanned)
                return None, summary
        try:
            record = next(records)
        except StopIteration:
            break
        scanned += 1
        size = record.definition_bytes
        if type(size) is not int or size < 0:
            raise ValueError("invalid manifest definition byte count")
        summary["definition_bytes"] += size
        estimated += 8192 + 32 * size
        if descriptor_budget_bytes is not None:
            name = getattr(record, "name", None)
            if not isinstance(name, str):
                raise ValueError("invalid manifest factor name for descriptor estimate")
            summary["estimated_descriptor_bytes"] += 8192 + 4 * len(
                name.encode("utf-8")
            )
            if 2 * summary["estimated_descriptor_bytes"] > descriptor_budget_bytes:
                summary.update(
                    reason="ARTIFACT_DESCRIPTORS_REQUIRE_BOUNDED_WAVES",
                    estimated_metadata_peak_bytes=estimated,
                    scanned_roots=scanned,
                )
                return None, summary
        if estimated > ceiling:
            summary.update(reason="METADATA_REQUIRES_BOUNDED_WAVES",
                           estimated_metadata_peak_bytes=estimated,
                           scanned_roots=scanned)
            return None, summary
    if scanned != count:
        raise ValueError("manifest root count changed during DAG admission")
    summary.update(estimated_metadata_peak_bytes=estimated, scanned_roots=scanned)
    lease = broker.acquire_memory(
        MemoryLeaseKind.MANIFEST_BUFFER, estimated,
        lease_id=f"run-dag-metadata:{run_id}",
    )
    if lease is None:
        summary["reason"] = "METADATA_LEASE_NOT_ADMITTED"
        return None, summary
    summary.update(execution_scope="whole_run_dag", factor_limit=count,
                   reason="WHOLE_RUN_METADATA_ADMITTED")
    return lease, summary
