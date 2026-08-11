# -*- coding: utf-8 -*-
"""R39-P1-PERF-081: structured fixed-slot performance counters.

The legacy ``runtime_stats`` plain-dict is mutated all over the hot path
(``production_policy.py``, ``adaptive_batch_scheduler.py``, ``batch_service.py``,
``buffer_ref.py``, ``engine.py``, ``resource_telemetry.py``).  Refactoring every
consumer in one pass is invasive and risky, so this module provides a *new*
low-overhead structured counter object and a process-level singleton that the
**new** code paths (PERF-082 certificate validation, the ``PerformanceRunSummary``
builder, and the R39 ``r39_*.json`` evidence generator) wire into.

Design (spec §24 / §28):

- Fixed declared slots only: an unknown name raises ``KeyError``.  The hot path
  never grows a dynamic dict, so there is no allocation pressure.
- ``incr`` / ``add`` are single-int writes into a pre-sized array
  (list of ``int``) — O(1), lock-free (CPython GIL makes a single ``int``
  element read/write atomic; counters are approximate under real concurrency).
- ``snapshot()`` renders to ``dict`` **only at the end** and never mutates the
  counters.
"""

from __future__ import annotations

import threading
from typing import Any

#: Declared fixed KPI slots (R39 §1.2 / §28).  Everything the unified
#: ``PerformanceRunSummary`` and the R39 evidence generators need is here; a
#: counter not declared here is a bug in the caller, not a new slot.
COUNTER_SLOTS: tuple[str, ...] = (
    # --- §1.2 counts ------------------------------------------------------
    "future_count",
    "factor_count",
    "physical_query_count",
    "parquet_file_open_count",
    "parquet_file_write_count",
    "fsync_count",
    "sqlite_transaction_count",
    "full_factor_rescan_count",
    "watchdog_thread_created_count",
    # --- §28 hard-gate KPIs -------------------------------------------------
    "legacy_union_prefetch_count",
    "representation_transition_count",
    "matrix_join_count",
    "micro_batch_task_count",
    "batch_write_transaction_count",
    "scheduled_task_count",
    "read_wave_count",
    # --- PERF-082 certificate / backend observability -----------------------
    "backend_fallback_count",
    "certificate_validation_failure_count",
    "spill_write_count",
    "spill_read_count",
    "cse_cache_hit_count",
    "cse_cache_miss_count",
)

_SLOT_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(COUNTER_SLOTS)}


class PerfCounters:
    """Fixed-slot array-backed integer counters (R39-P1-PERF-081).

    ``incr`` / ``add`` are hot-path safe O(1) writes; ``snapshot`` is the only
    allocation-heavy operation and is meant to run at the end of a batch run.
    """

    __slots__ = ("_values", "_lock", "_slot_names", "_slot_index")

    def __init__(self, slots: tuple[str, ...] = COUNTER_SLOTS) -> None:
        # Validate the caller's slot table up-front so a typo fails fast at
        # construction, not mid-run.
        dup = sorted({name for name in slots if slots.count(name) > 1})
        if dup:
            raise ValueError(f"duplicate perf counter slots: {dup}")
        self._slot_names: tuple[str, ...] = slots  # type: ignore[attr-defined]
        self._slot_index: dict[str, int] = {  # type: ignore[attr-defined]
            name: i for i, name in enumerate(slots)
        }
        self._values: list[int] = [0] * len(slots)
        self._lock = threading.Lock()

    # -- hot path -----------------------------------------------------------
    def incr(self, name: str, delta: int = 1) -> None:
        """Increment ``name`` by ``delta`` (default 1).  Unknown → ``KeyError``."""
        self.add(name, delta)

    def add(self, name: str, delta: int) -> None:
        """Add ``delta`` to ``name``.  Unknown → ``KeyError``."""
        idx = self._slot_index.get(name)  # type: ignore[attr-defined]
        if idx is None:
            raise KeyError(f"undeclared perf counter: {name!r}")
        if delta:
            self._values[idx] += delta

    # -- read path ----------------------------------------------------------
    def snapshot(self) -> dict[str, int]:
        """Render all declared slots to a plain dict.  Never mutates counters."""
        with self._lock:
            names = self._slot_names  # type: ignore[attr-defined]
            return {name: self._values[i] for i, name in enumerate(names)}

    def get(self, name: str) -> int:
        """Read a single slot value (telemetry / assertions).  Unknown → KeyError."""
        idx = self._slot_index.get(name)  # type: ignore[attr-defined]
        if idx is None:
            raise KeyError(f"undeclared perf counter: {name!r}")
        with self._lock:
            return self._values[idx]

    def reset(self) -> None:
        """Zero all counters (start of a new run)."""
        with self._lock:
            for i in range(len(self._values)):
                self._values[i] = 0

    # -- ergonomics ---------------------------------------------------------
    def __getitem__(self, name: str) -> int:
        return self.get(name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PerfCounters({self.snapshot()!r})"


# ---------------------------------------------------------------------------
# process-level singleton
# ---------------------------------------------------------------------------

_global_counters: PerfCounters | None = None
_global_counters_lock = threading.Lock()


def get_global_counters() -> PerfCounters:
    """Return the process-level ``PerfCounters`` singleton.

    Thread-safe lazy init.  All new R39 code paths (certificate validation,
    ``PerformanceRunSummary.from_counters``, evidence generators) share one
    instance so a whole batch run's counters can be aggregated at the end.
    """
    global _global_counters
    if _global_counters is None:
        with _global_counters_lock:
            if _global_counters is None:
                _global_counters = PerfCounters()
    return _global_counters


def reset_global_counters() -> PerfCounters:
    """Replace the process-level singleton with a fresh zeroed instance.

    Test isolation helper: returns the fresh instance so callers can grab it in
    one expression.  The old instance (if any) is simply dropped.
    """
    global _global_counters
    fresh = PerfCounters()
    with _global_counters_lock:
        _global_counters = fresh
    return fresh


__all__ = [
    "COUNTER_SLOTS",
    "PerfCounters",
    "get_global_counters",
    "reset_global_counters",
]
