"""Bounded, thread-safe telemetry for verified factor-source stages."""
from __future__ import annotations

from contextlib import contextmanager
import math
from numbers import Real
from threading import Lock
import time


class SourceStageTelemetry:
    """Thread-safe fixed-phase counters with detached snapshots.

    Durations are summed across operations and workers, so parallel totals may
    exceed wall time. Only fixed phase names, counts, and durations are kept.
    """
    PHASES = ("context_setup", "bound_factor_read", "arrow_to_pandas_axis",
              "reindex_write", "manifest_verify")

    def __init__(self):
        self._lock = Lock()
        self._values = {phase: {"count": 0, "total_seconds": 0.0}
                        for phase in self.PHASES}

    def _validate_phase(self, phase: str) -> None:
        if phase not in self._values:
            raise ValueError("unknown source telemetry phase")

    def record(self, phase: str, count: int, seconds: float) -> None:
        self._validate_phase(phase)
        if type(count) is not int or count < 0:
            raise ValueError("telemetry count must be a nonnegative integer")
        if (isinstance(seconds, bool) or not isinstance(seconds, Real)
                or not math.isfinite(float(seconds)) or seconds < 0):
            raise ValueError("telemetry duration must be finite and nonnegative")
        with self._lock:
            item = self._values[phase]
            item["count"] += count
            item["total_seconds"] += float(seconds)

    @contextmanager
    def measure(self, phase: str):
        # Reject invalid phases before allowing the caller's operation to run.
        self._validate_phase(phase)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(phase, 1, time.perf_counter() - started)

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            return {phase: dict(values) for phase, values in self._values.items()}
