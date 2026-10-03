"""Small process-local cache for previously validated source A/B records."""
from __future__ import annotations

from collections import OrderedDict
import os
import threading
import time

from quant_evaluator.runtime.source_route_qualification import CounterbalancedABRecord
from quant_evaluator.runtime.source_route_profiles import CounterbalancedRouteProfileRecord


_MAX_ENTRIES = 32
_TTL_SECONDS = 60 * 60
_lock = threading.RLock()
_Record = CounterbalancedABRecord | CounterbalancedRouteProfileRecord
_entries: OrderedDict[str, tuple[float, tuple[_Record, ...]]] = OrderedDict()


def _reset_after_fork() -> None:
    global _lock, _entries
    _lock = threading.RLock()
    _entries = OrderedDict()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def get_validated_records(key: str) -> tuple[_Record, ...] | None:
    """Return an unexpired immutable pair, pruning expired cache entries."""
    if type(key) is not str or not key:
        return None
    now = time.monotonic()
    with _lock:
        for cache_key, (expires_at, _) in tuple(_entries.items()):
            if expires_at <= now:
                del _entries[cache_key]
        entry = _entries.get(key)
        if entry is None:
            return None
        expires_at, records = entry
        if expires_at <= now:
            del _entries[key]
            return None
        _entries.move_to_end(key)
        return records


def remember_validated_records(
    key: str, records: tuple[_Record, _Record],
) -> None:
    """Store only exact immutable record pairs after successful live validation."""
    if (type(key) is not str or not key or type(records) is not tuple
            or len(records) != 2
            or any(type(item) not in (CounterbalancedABRecord,
                                      CounterbalancedRouteProfileRecord)
                   for item in records)
            or type(records[0]) is not type(records[1])):
        raise ValueError("cache accepts only a keyed pair of immutable qualification records")
    with _lock:
        _entries[key] = (time.monotonic() + _TTL_SECONDS, records)
        _entries.move_to_end(key)
        while len(_entries) > _MAX_ENTRIES:
            _entries.popitem(last=False)


def discard_validated_records(key: str) -> None:
    if type(key) is not str:
        return
    with _lock:
        _entries.pop(key, None)


def clear_validated_records() -> None:
    """Clear process-local evidence, primarily for deterministic lifecycle use."""
    with _lock:
        _entries.clear()


__all__ = (
    "clear_validated_records", "discard_validated_records",
    "get_validated_records", "remember_validated_records",
)
