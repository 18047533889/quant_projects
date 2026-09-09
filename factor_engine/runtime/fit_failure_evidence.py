"""Bounded, detached wire snapshots for rolling-fit failure evidence."""
from __future__ import annotations

import base64
from datetime import date, datetime
import json
from itertools import islice
from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.ts_model._rolling_core import BoundedFitFailureSink, FitScope


SCHEMA_VERSION = "factor_engine.fit_failure_snapshot.v1"
MAX_SNAPSHOT_BYTES = 262_144
MAX_GROUPS = 128
MAX_DETAILS = 64
_MAX_DEPTH = 10
_MAX_NODES = 16_384
_MAX_CONTAINER_ITEMS = 512
_MAX_STRING_CHARS = 4_096
_MAX_INTEGER_BITS = 256
_SCOPE_FIELDS = (
    "canonical", "backend", "profile", "instrument", "window_start", "window_end",
    "output_row", "fit_cutoff", "maturity_cutoff", "execution_id", "run_id",
    "task_id", "factor_id",
)
_TOP_KEYS = frozenset({
    "schema_version", "availability", "coverage", "groups", "details", "dropped_details",
    "overflow_group_count", "unknown_status_count", "truncated", "truncation_reasons",
})
_TRUNCATION_REASONS = frozenset({
    "detail_count_limit", "group_count_limit", "snapshot_bytes_details",
    "snapshot_bytes_groups", "snapshot_structure_details", "sink_detail_eviction",
    "sink_group_overflow",
})


def _tag(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> Any:
    if budget is None:
        budget = [_MAX_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > _MAX_DEPTH:
        raise ValueError("fit-failure value exceeds structural budget")
    if value is None or type(value) is bool:
        return value
    if isinstance(value, np.generic):
        return {"$type": "numpy_scalar", "dtype": str(value.dtype),
                "value": _tag(value.item(), depth=depth + 1, budget=budget)}
    if type(value) is int:
        if value.bit_length() > _MAX_INTEGER_BITS:
            raise ValueError("fit-failure integer exceeds bit bound")
        return value
    if type(value) is float:
        if np.isnan(value):
            return {"$type": "float", "value": "nan"}
        if value == float("inf"):
            return {"$type": "float", "value": "+inf"}
        if value == float("-inf"):
            return {"$type": "float", "value": "-inf"}
        return value
    if type(value) is complex:
        return {"$type": "complex", "real": _tag(float(value.real), depth=depth + 1, budget=budget),
                "imag": _tag(float(value.imag), depth=depth + 1, budget=budget)}
    if type(value) is str:
        if len(value) > _MAX_STRING_CHARS:
            raise ValueError("fit-failure string exceeds character bound")
        return value
    if type(value) is bytes:
        if len(value) > MAX_SNAPSHOT_BYTES:
            raise ValueError("fit-failure bytes exceed byte bound")
        return {"$type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, datetime):
        return {"$type": "datetime", "iso8601": value.isoformat()}
    if isinstance(value, date):
        return {"$type": "date", "iso8601": value.isoformat()}
    if type(value) in (list, tuple):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError("fit-failure sequence exceeds item bound")
        return {"$type": "tuple" if type(value) is tuple else "list",
                "items": [_tag(item, depth=depth + 1, budget=budget) for item in value]}
    if type(value) is dict:
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError("fit-failure mapping exceeds item bound")
        items = [
            [_tag(key, depth=depth + 1, budget=budget),
             _tag(item, depth=depth + 1, budget=budget)]
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return {"$type": "dict", "items": items}
    raise ValueError(f"unsupported fit-failure value type: {type(value).__module__}.{type(value).__qualname__}")


def _detail_wire(receipt: Any, *, budget: list[int]) -> dict[str, Any]:
    status = receipt.status
    scope = receipt.scope
    return {
        "sequence": _tag(receipt.sequence, budget=budget),
        "status": {
            "converged": _tag(status.converged, budget=budget),
            "reason": _tag(status.reason, budget=budget),
            "details": _tag(status.details, budget=budget),
        },
        "scope": {name: _tag(getattr(scope, name), budget=budget) for name in _SCOPE_FIELDS},
        "scope_kind": _tag(receipt.scope_kind, budget=budget),
    }


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _bounded_canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Canonically encode without ever retaining more than the wire limit."""
    encoder = json.JSONEncoder(
        allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    chunks: list[bytes] = []
    total = 0
    for chunk in encoder.iterencode(payload):
        encoded = chunk.encode("utf-8")
        total += len(encoded)
        if total > MAX_SNAPSHOT_BYTES:
            raise ValueError("fit-failure snapshot exceeds byte bound")
        chunks.append(encoded)
    return b"".join(chunks)


def snapshot_fit_failures(sink: BoundedFitFailureSink | None) -> dict[str, Any] | None:
    """Copy a sink atomically, then encode a detached bounded wire snapshot."""
    if sink is None:
        return None
    if not isinstance(sink, BoundedFitFailureSink):
        raise TypeError("sink must be BoundedFitFailureSink or None")
    with sink._lock:
        count_items = list(islice(sink._counts.items(), MAX_GROUPS))
        omitted_group_failures = sum(
            count for _key, count in islice(sink._counts.items(), MAX_GROUPS, None)
        )
        retained_reversed = list(islice(reversed(sink._details), MAX_DETAILS))
        receipts = tuple(reversed(retained_reversed))
        omitted_details = max(0, len(sink._details) - len(receipts))
        dropped = sink._dropped_details
        overflow = sink._overflow_group_count
        unknown = sink._unknown_status_count
        sink_had_eviction = dropped > 0
        sink_had_group_overflow = overflow > 0

    groups = [
        {"canonical": _tag(canonical), "reason": _tag(reason), "count": _tag(count)}
        for (canonical, reason), count in count_items
    ]
    dropped += omitted_details
    overflow += omitted_group_failures
    reasons: list[str] = []
    if sink_had_eviction:
        reasons.append("sink_detail_eviction")
    if sink_had_group_overflow:
        reasons.append("sink_group_overflow")
    if omitted_details:
        reasons.append("detail_count_limit")
    if omitted_group_failures:
        reasons.append("group_count_limit")
    details_reversed = []
    wire_budget = [_MAX_NODES]
    for index, receipt in enumerate(reversed(receipts)):
        trial_budget = [wire_budget[0]]
        try:
            # Encoding and wire validation intentionally have different tree
            # shapes (container tags add ``items`` levels).  Admit a detail
            # only after charging the same budget the final validator uses.
            item = _detail_wire(receipt, budget=[_MAX_NODES])
            _validate_tagged(item["status"]["details"], budget=trial_budget)
            for scope_value in item["scope"].values():
                _validate_tagged(scope_value, budget=trial_budget)
        except ValueError:
            dropped += len(receipts) - index
            reasons.append("snapshot_structure_details")
            break
        details_reversed.append(item)
        wire_budget = trial_budget
    details = list(reversed(details_reversed))

    def body() -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "availability": "observed",
            "coverage": "instrumented_fit_producers_only",
            "groups": groups, "details": details, "dropped_details": dropped,
            "overflow_group_count": overflow, "unknown_status_count": unknown,
            "truncated": bool(reasons), "truncation_reasons": list(reasons),
        }

    while len(_canonical_bytes(body())) > MAX_SNAPSHOT_BYTES and details:
        details.pop(0)
        dropped += 1
        if "snapshot_bytes_details" not in reasons:
            reasons.append("snapshot_bytes_details")
    while len(_canonical_bytes(body())) > MAX_SNAPSHOT_BYTES and groups:
        removed = groups.pop()
        overflow += removed["count"]
        if "snapshot_bytes_groups" not in reasons:
            reasons.append("snapshot_bytes_groups")
    payload = body()
    if len(_canonical_bytes(payload)) > MAX_SNAPSHOT_BYTES:
        raise ValueError("fit-failure snapshot fixed fields exceed byte bound")
    return validate_fit_failure_snapshot(payload)


def _require_exact_dict(value: Any, keys: set[str] | frozenset[str], label: str) -> dict:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} must be an exact-key dict")
    return value


def _validate_tagged(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [_MAX_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > _MAX_DEPTH:
        raise ValueError("tagged value exceeds structural budget")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if value.bit_length() > _MAX_INTEGER_BITS:
            raise ValueError("integer exceeds bit bound")
        return
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("non-finite float")
        return
    if type(value) is str:
        if len(value) > _MAX_STRING_CHARS:
            raise ValueError("string exceeds character bound")
        return
    if type(value) is list:
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError("list exceeds item bound")
        for item in value:
            _validate_tagged(item, depth=depth + 1, budget=budget)
        return
    if type(value) is not dict or "$type" not in value:
        raise ValueError("arbitrary mapping is not a tagged value")
    tag = value["$type"]
    shapes = {
        "float": {"$type", "value"},
        "complex": {"$type", "real", "imag"}, "bytes": {"$type", "base64"},
        "date": {"$type", "iso8601"}, "datetime": {"$type", "iso8601"},
        "numpy_scalar": {"$type", "dtype", "value"},
        "tuple": {"$type", "items"}, "list": {"$type", "items"},
        "dict": {"$type", "items"},
    }
    if tag not in shapes or set(value) != shapes[tag]:
        raise ValueError("invalid tagged value shape")
    if tag == "float":
        if value["value"] not in {"nan", "+inf", "-inf"}:
            raise ValueError("invalid tagged float")
    elif tag == "complex":
        for key in ("real", "imag"):
            component = value[key]
            if type(component) not in (int, float) and not (
                    type(component) is dict and component.get("$type") == "float"):
                raise ValueError("invalid tagged complex component")
            _validate_tagged(component, depth=depth + 1, budget=budget)
    elif tag == "bytes":
        if type(value["base64"]) is not str or len(value["base64"]) > MAX_SNAPSHOT_BYTES * 2:
            raise ValueError("invalid tagged bytes")
        try:
            base64.b64decode(value["base64"], validate=True)
        except Exception as exc:
            raise ValueError("invalid tagged bytes") from exc
    elif tag in {"date", "datetime"}:
        if type(value["iso8601"]) is not str or len(value["iso8601"]) > 64:
            raise ValueError("invalid tagged date")
        try:
            (datetime if tag == "datetime" else date).fromisoformat(value["iso8601"])
        except ValueError as exc:
            raise ValueError("invalid tagged date") from exc
    elif tag == "numpy_scalar":
        if type(value["dtype"]) is not str or len(value["dtype"]) > 64:
            raise ValueError("invalid numpy scalar dtype")
        try:
            dtype = np.dtype(value["dtype"])
        except TypeError as exc:
            raise ValueError("invalid numpy scalar dtype") from exc
        if dtype.hasobject or dtype.subdtype is not None:
            raise ValueError("invalid numpy scalar dtype")
        _validate_tagged(value["value"], depth=depth + 1, budget=budget)
    elif tag in {"tuple", "list"}:
        if type(value["items"]) is not list:
            raise ValueError("invalid tagged sequence items")
        _validate_tagged(value["items"], depth=depth + 1, budget=budget)
    else:
        items = value["items"]
        if type(items) is not list or len(items) > _MAX_CONTAINER_ITEMS:
            raise ValueError("invalid tagged mapping items")
        for pair in items:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("invalid tagged mapping pair")
            _validate_tagged(pair[0], depth=depth + 1, budget=budget)
            _validate_tagged(pair[1], depth=depth + 1, budget=budget)


def _scope_scalar_from_wire(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is not dict:
        raise ValueError("scope field is not scalar")
    tag = value.get("$type")
    if tag == "date" and set(value) == {"$type", "iso8601"}:
        return date.fromisoformat(value["iso8601"])
    if tag == "datetime" and set(value) == {"$type", "iso8601"}:
        # Python datetime truncates pandas' nanoseconds, which could make an
        # invalid owned window appear valid after wire decoding.
        return pd.Timestamp(value["iso8601"])
    if tag == "numpy_scalar" and set(value) == {"$type", "dtype", "value"}:
        inner = _scope_scalar_from_wire(value["value"])
        try:
            # Indexing a zero-dimensional array preserves the NumPy scalar
            # and its datetime unit; ``item()`` may return int or datetime.
            scalar = np.asarray(inner, dtype=np.dtype(value["dtype"])).reshape(())[()]
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid scope numpy scalar") from exc
        return scalar
    raise ValueError("scope field is not a supported scalar")


def validate_fit_failure_snapshot(payload: Any) -> dict[str, Any]:
    """Fail-closed validation returning a detached ordinary JSON-safe dict."""
    body = _require_exact_dict(payload, _TOP_KEYS, "snapshot")
    if (body["schema_version"] != SCHEMA_VERSION or body["availability"] != "observed"
            or body["coverage"] != "instrumented_fit_producers_only"):
        raise ValueError("unsupported fit-failure snapshot schema")
    groups = body["groups"]
    details = body["details"]
    reasons = body["truncation_reasons"]
    if type(groups) is not list or len(groups) > MAX_GROUPS:
        raise ValueError("groups exceed bound")
    if type(details) is not list or len(details) > MAX_DETAILS:
        raise ValueError("details exceed bound")
    if (type(reasons) is not list or len(reasons) > len(_TRUNCATION_REASONS)
            or len(set(reasons)) != len(reasons)
            or any(type(x) is not str or x not in _TRUNCATION_REASONS for x in reasons)):
        raise ValueError("invalid truncation reasons")
    for name in ("dropped_details", "overflow_group_count", "unknown_status_count"):
        value = body[name]
        if type(value) is not int or value < 0 or value.bit_length() > 64:
            raise ValueError(f"invalid {name}")
    if type(body["truncated"]) is not bool or body["truncated"] != bool(reasons):
        raise ValueError("truncated flag/reasons mismatch")
    seen_groups: set[tuple[str, str]] = set()
    for group in groups:
        group = _require_exact_dict(group, {"canonical", "reason", "count"}, "group")
        if (type(group["canonical"]) is not str or not group["canonical"]
                or len(group["canonical"]) > _MAX_STRING_CHARS
                or type(group["reason"]) is not str or not group["reason"]
                or len(group["reason"]) > _MAX_STRING_CHARS
                or type(group["count"]) is not int or group["count"] < 1
                or group["count"].bit_length() > 64):
            raise ValueError("invalid group")
        group_key = (group["canonical"], group["reason"])
        if group_key in seen_groups:
            raise ValueError("duplicate failure group")
        seen_groups.add(group_key)
    budget = [_MAX_NODES]
    previous_sequence = 0
    for detail in details:
        detail = _require_exact_dict(detail, {"sequence", "status", "scope", "scope_kind"}, "detail")
        if (type(detail["sequence"]) is not int or detail["sequence"] < 1
                or detail["sequence"].bit_length() > 64):
            raise ValueError("invalid detail sequence")
        if detail["sequence"] <= previous_sequence:
            raise ValueError("detail sequences must be strictly increasing")
        previous_sequence = detail["sequence"]
        status = _require_exact_dict(detail["status"], {"converged", "reason", "details"}, "status")
        if (status["converged"] is not False or type(status["reason"]) is not str
                or not status["reason"] or len(status["reason"]) > _MAX_STRING_CHARS):
            raise ValueError("invalid status")
        scope = _require_exact_dict(detail["scope"], set(_SCOPE_FIELDS), "scope")
        if detail["scope_kind"] not in {"kernel_only", "factor_window"}:
            raise ValueError("invalid scope kind")
        _validate_tagged(status["details"], budget=budget)
        for item in scope.values():
            _validate_tagged(item, budget=budget)
        decoded_scope = {name: _scope_scalar_from_wire(scope[name]) for name in _SCOPE_FIELDS}
        try:
            computed_kind = FitScope(**decoded_scope).scope_kind
        except Exception as exc:
            raise ValueError("invalid fit scope") from exc
        if computed_kind != detail["scope_kind"]:
            raise ValueError("scope kind does not match owned coordinates")
    encoded = _bounded_canonical_bytes(body)
    return json.loads(encoded.decode("utf-8"))


def encode_fit_failure_snapshot(payload: Any) -> bytes:
    """Validate and canonically encode one snapshot."""
    encoded = _canonical_bytes(validate_fit_failure_snapshot(payload))
    if len(encoded) > MAX_SNAPSHOT_BYTES:
        raise ValueError("fit-failure snapshot exceeds byte bound")
    return encoded
