"""Small, data-free execution evidence extracted from a real batch output."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping


MAX_LEDGER_BYTES = 64 * 1024
_MAX_TEXT = 256
_MAX_ROOT_PATHS = 1_000_000
_MAX_TRANSFER_GROUPS = 32
_MISSING = object()


class ExecutionLedgerError(ValueError):
    pass


def _text(value: Any) -> Any:
    if not isinstance(value, str):
        return _MISSING
    encoded = value.encode("utf-8")
    return value if len(encoded) <= _MAX_TEXT else _MISSING


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool) or type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else _MISSING
    return _text(value)


def _pick(source: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    out = {}
    for key in keys:
        value = _scalar(source.get(key, _MISSING))
        if value is not _MISSING:
            out[key] = value
    return out


_SCHEDULER_KEYS = (
    "mode", "auto_execution_mode", "serial_fused", "real_task_done",
    "virtual_task_done", "wave_scan_done", "virtual_task_ratio",
    "concurrency_limit_final", "future_count", "factor_count",
    "micro_batch_task_count", "micro_batch_root_count", "future_per_factor",
    "scheduler_wait_polling_count", "execution_policy",
)
_PEAK_KEYS = (
    "peak_family_pss", "peak_family_rss", "samples", "sample_count",
    "elapsed_seconds", "duration_seconds", "source",
)
_PHYSICAL_KEYS = (
    "plan_id", "plan_hash", "planned_backend", "actual_backend", "region_id",
    "backend_switch_count", "estimated_peak_memory_bytes", "memory_basis",
    "materialization_count", "resident_reuse_count", "python_to_q_bytes",
    "source_residency",
)
_DECISION_KEYS = (
    "target_concurrency", "target_cpu_tokens", "read_wave_bytes",
    "cse_cache_bytes", "result_queue_bytes", "writer_batch_bytes",
    "execution_memory_bytes", "pressure_stage", "estimate_basis",
)
_TRANSFER_DIMENSIONS = (
    "source_backend", "target_backend", "source_representation",
    "target_representation",
)


def _transfer_key(event: Mapping[str, Any]) -> tuple[str, ...] | None:
    values = tuple(_text(event.get(key)) for key in _TRANSFER_DIMENSIONS)
    return None if any(value is _MISSING for value in values) else values


def summarize_execution_ledger(
    output: Mapping[str, Any], *, run_id: str, evidence_id: str
) -> dict[str, Any]:
    """Extract bounded planned-vs-actual scalar evidence; never copy factor data."""
    if not isinstance(output, Mapping):
        raise TypeError("output must be a mapping")
    for label, value in (("run_id", run_id), ("evidence_id", evidence_id)):
        if _text(value) is _MISSING or not value:
            raise ExecutionLedgerError(f"{label} must be nonempty bounded UTF-8")

    ledger: dict[str, Any] = {
        "schema_version": "factor_engine.execution_ledger.v1",
        "run_id": run_id,
        "evidence_id": evidence_id,
        "evidence_semantics": {
            "planned": "planner estimates and selected routes",
            "actual": "runtime counters and measured transfer/run-peak telemetry",
        },
    }
    executor = _scalar(output.get("executor", _MISSING))
    if executor is not _MISSING:
        ledger["executor"] = executor

    run_stats = output.get("scheduler_stats")
    if isinstance(run_stats, Mapping):
        actual = _pick(run_stats.get("scheduler_stats"), _SCHEDULER_KEYS)
        for key in ("auto_execution_mode", "done"):
            value = _scalar(run_stats.get(key, _MISSING))
            if value is not _MISSING:
                actual[key] = value
        peak = _pick(run_stats.get("run_peak"), _PEAK_KEYS)
        if peak:
            actual["run_peak"] = peak
        decision_source = run_stats.get("scheduler_stats")
        if isinstance(decision_source, Mapping):
            decision = _pick(decision_source.get("resource_decision"), _DECISION_KEYS)
            if decision:
                actual["resource_decision"] = decision
        if actual:
            ledger["actual_scheduler"] = actual

    planned = _pick(output.get("physical_plan"), _PHYSICAL_KEYS)
    if planned:
        ledger["planned_physical"] = planned

    paths = output.get("backend_paths")
    route_counts: dict[tuple[str, str], int] = {}
    transfers: dict[tuple[str, ...], dict[str, Any]] = {}
    other_transfers = 0
    roots_seen = 0
    truncated = False
    if isinstance(paths, Mapping):
        for _root_name, path in paths.items():
            if roots_seen >= _MAX_ROOT_PATHS:
                truncated = True
                break
            roots_seen += 1
            if not isinstance(path, Mapping):
                continue
            physical = path.get("physical_plan")
            if not isinstance(physical, Mapping):
                continue
            planned_backend = _text(physical.get("planned_backend"))
            actual_backend = _text(physical.get("actual_backend"))
            if planned_backend is not _MISSING and actual_backend is not _MISSING:
                pair = (planned_backend, actual_backend)
                route_counts[pair] = route_counts.get(pair, 0) + 1
            events = physical.get("transfers")
            if not isinstance(events, (list, tuple)):
                events = physical.get("transfer_events")
            if not isinstance(events, (list, tuple)):
                continue
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                key = _transfer_key(event)
                if key is None:
                    continue
                if key not in transfers and len(transfers) >= _MAX_TRANSFER_GROUPS:
                    other_transfers += 1
                    continue
                item = transfers.setdefault(key, {
                    **dict(zip(_TRANSFER_DIMENSIONS, key)), "event_count": 0,
                })
                item["event_count"] += 1
                for field in ("actual_bytes", "actual_rows"):
                    value = event.get(field)
                    if type(value) is int and value >= 0:
                        item[field] = item.get(field, 0) + value
                value = event.get("actual_transfer_ms")
                if type(value) in (int, float) and math.isfinite(float(value)) and value >= 0:
                    item["actual_transfer_ms"] = item.get("actual_transfer_ms", 0.0) + float(value)
                for field in ("sort_applied", "repartition_applied", "reshape_applied"):
                    value = event.get(field)
                    if type(value) is bool:
                        count_key = f"{field}_count"
                        item[count_key] = item.get(count_key, 0) + int(value)

    actual_routes: dict[str, Any] = {"root_paths_observed": roots_seen}
    if route_counts:
        actual_routes["route_counts"] = [
            {"planned_backend": pair[0], "actual_backend": pair[1], "root_count": count}
            for pair, count in sorted(route_counts.items())
        ]
    if transfers:
        actual_routes["transfer_groups"] = [transfers[key] for key in sorted(transfers)]
    if other_transfers:
        actual_routes["other_transfer_event_count"] = other_transfers
    if truncated:
        actual_routes["root_paths_truncated"] = True
    if roots_seen or route_counts or transfers:
        ledger["actual_physical"] = actual_routes

    # Keep this byte-for-byte aligned with bounded_pipeline._write_control_receipt;
    # a compact-JSON check can pass while the indented durable artifact exceeds
    # the promised 64 KiB bound.
    encoded = json.dumps(
        ledger, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_LEDGER_BYTES:
        raise ExecutionLedgerError("execution ledger exceeds 64 KiB")
    return ledger
