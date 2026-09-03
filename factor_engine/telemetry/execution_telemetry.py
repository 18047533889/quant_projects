"""Execution-pipeline telemetry (P0#7 of the 100k GO battle).

Single-threaded counter sink for CSE / native-fusion / backend-fallback /
run-many canonicalization so that execution is never *silent*:

- ``cse.candidates``  — distinct structural subtrees seen in the forest
- ``cse.shared_nodes`` — profitable subtrees actually materialised as shared
- ``cse.hit_ratio``    — shared / candidates (reuse potential)
- ``cse.reuse_edges``  — total plan_ref consumer edges created (multiples)
- ``cse.released``     — shared nodes freed by refcount lifecycle
- ``fusion.planned``   — fusion groups admitted as single multi-root query
- ``fusion.executed``  — groups that actually succeeded via execute_multi_roots
- ``fusion.binary_split`` — groups that needed a recursive binary split
- ``fusion.fallback`   — groups that honestly fell back to per-root
- ``fusion.roots_total`` — roots consumed by executed fusion groups
- ``fallback.pandas``  — production pandas fallback events observed
- ``fallback.vector``  — vector-kernel bind failures observed (if any)
- ``runmany.factors`   — factors admitted to run_many after canonicalization
- ``runmany.single_run_warn` — agent-facing single-op run() inside loop warned

Every counter is bumped through :func:`record` which logs at INFO the first
time each key fires, so a quiet batch with no telemetry is itself a signal.

Snapshot via :func:`snapshot` (copy, safe for logging / CI assertions) and
:func:`reset` for tests.
"""

from __future__ import annotations

import threading
import time
from collections import Counter

import logging

logger = logging.getLogger("factor_engine.telemetry.execution")

#: Registry-wide counters. All counter names are ``domain.name``; the value is
#: an integer monotonic-within-a-session number of events.
_COUNTERS: "Counter[str]" = Counter()
_LOCK = threading.Lock()
_STARTED_AT = time.time()
_ANNOUNCED: set[str] = set()


def _key(domain: str, name: str) -> str:
    return f"{domain}.{name}"


def record(domain: str, name: str, n: int = 1) -> None:
    """Bump counter ``domain.name`` by ``n`` (thread-safe). Logs first occurrence."""
    k = _key(domain, name)
    with _LOCK:
        _COUNTERS[k] += n
        if k not in _ANNOUNCED:
            _ANNOUNCED.add(k)
            logger.info("execution-telemetry: %s (first fire) -> %d", k, _COUNTERS[k])


def snapshot() -> dict[str, int]:
    """Return a copy of the current counters (int values), safe for assertions."""
    with _LOCK:
        return {k: int(v) for k, v in _COUNTERS.items()}


def reset() -> None:
    """Clear all execution-pipeline counters. Test-only hook."""
    global _STARTED_AT
    with _LOCK:
        _COUNTERS.clear()
        _ANNOUNCED.clear()
        _STARTED_AT = time.time()


def get(domain: str, name: str, default: int = 0) -> int:
    """Read a single counter without taking a copy of the whole registry."""
    with _LOCK:
        return int(_COUNTERS.get(_key(domain, name), default))


def cse_record(delta: dict[str, int]) -> None:
    """Convenience sink for CSE metrics emitted by the planner."""
    for name, value in delta.items():
        record("cse", name, value)


def fusion_record(delta: dict[str, int]) -> None:
    """Convenience sink for native-fusion metrics emitted by the scheduler."""
    for name, value in delta.items():
        record("fusion", name, value)


def fallback_record(domain: str, name: str, n: int = 1) -> None:
    """Explicit fallback event (backend degraded to a slower path)."""
    record(domain, name, n)


__all__ = [
    "record",
    "snapshot",
    "reset",
    "get",
    "cse_record",
    "fusion_record",
    "fallback_record",
]
