# -*- coding: utf-8 -*-
"""R39 PERF-044: ``WriteAmplificationTracker`` — write amplification as a
materializer hard KPI.

Tracks per-write:

- ``logical_changed_bytes``   — changed logical output bytes (input frame memory).
- ``physical_new_write_bytes``— physical bytes newly written (delta or rewrite).
- ``historical_rewrite_bytes``— historical (pre-existing) bytes rewritten.  In
  delta mode ordinary 1-day incremental writes report ``0``.
- ``delta_file_count``        — number of delta fragments emitted.
- ``compaction_debt_bytes``   — delta bytes not yet absorbed by a compaction.

The materializer holds one tracker and feeds it from every partition commit, so
the snapshot can be read back (and asserted ``historical_rewrite_bytes == 0`` in
delta mode — R39 Gate-04).
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


@dataclass
class WriteAmplificationSnapshot:
    """Cumulative write-amplification accounting."""

    logical_changed_bytes: int = 0
    physical_new_write_bytes: int = 0
    historical_rewrite_bytes: int = 0
    delta_file_count: int = 0
    compaction_debt_bytes: int = 0
    write_count: int = 0

    @property
    def write_amplification(self) -> float:
        """physical bytes written / changed logical bytes (≥ 1.0)."""
        if not self.physical_new_write_bytes:
            return 0.0
        return self.physical_new_write_bytes / max(int(self.logical_changed_bytes), 1)

    @property
    def rewrite_amplification(self) -> float:
        """historical rewritten bytes / changed logical bytes."""
        if not self.historical_rewrite_bytes:
            return 0.0
        return self.historical_rewrite_bytes / max(int(self.logical_changed_bytes), 1)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["write_amplification"] = self.write_amplification
        out["rewrite_amplification"] = self.rewrite_amplification
        return out


class WriteAmplificationTracker:
    """Thread-safe accumulator of per-write amplification KPIs."""

    def __init__(self, *, delta_mode: bool = False) -> None:
        self.delta_mode = bool(delta_mode)
        self._snap = WriteAmplificationSnapshot()
        self._lock = threading.Lock()

    def record_partition_write(
        self,
        *,
        logical_changed_bytes: int,
        physical_new_write_bytes: int,
        historical_rewrite_bytes: int = 0,
        delta_file_count: int = 0,
    ) -> None:
        """Record one partition commit's amplification contribution."""
        with self._lock:
            s = self._snap
            s.write_count += 1
            s.logical_changed_bytes += int(logical_changed_bytes)
            s.physical_new_write_bytes += int(physical_new_write_bytes)
            s.historical_rewrite_bytes += int(historical_rewrite_bytes)
            s.delta_file_count += int(delta_file_count)

    def record_compaction_debt(self, *, debt_bytes: int) -> None:
        """Set the current compaction debt (delta bytes awaiting compaction)."""
        with self._lock:
            self._snap.compaction_debt_bytes = int(debt_bytes)

    def snapshot(self) -> WriteAmplificationSnapshot:
        """Return a consistent point-in-time snapshot."""
        with self._lock:
            return WriteAmplificationSnapshot(**asdict(self._snap))


__all__ = [
    "WriteAmplificationSnapshot",
    "WriteAmplificationTracker",
]
