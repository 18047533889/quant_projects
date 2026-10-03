"""Process-wide serialization for cold CPU/CUDA calibration measurements.

This coordinator intentionally tracks no request keys and does not control
threadpool settings. Its reservation prevents two cold calibration experiments
in this process from timing each other as workload. Ordinary inference/cache
hits should not acquire it.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import numbers
import os
import threading
import time


class _SerialCalibrationCoordinator:
    """One bounded process-wide lock, rebuilt in a forked child."""

    def __init__(self):
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._local = threading.local()

    def _reset_after_fork(self):
        # Only the forking thread survives in the child. Never inherit a lock
        # that may be owned by a vanished parent thread.
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._local = threading.local()

    def _ensure_current_process(self):
        # register_at_fork is the normal path; the PID guard also covers
        # runtimes where at-fork hooks are unavailable or were not installed.
        if self._pid != os.getpid():
            self._reset_after_fork()

    @contextmanager
    def reserve(self, timeout_seconds: float | None = None):
        self._ensure_current_process()
        if getattr(self._local, "active", False):
            raise RuntimeError(
                "serial calibration measurement cannot be nested in the same thread"
            )

        if timeout_seconds is not None:
            if (isinstance(timeout_seconds, bool)
                    or not isinstance(timeout_seconds, numbers.Real)):
                raise ValueError("timeout_seconds must be None or finite and nonnegative")
            timeout_seconds = float(timeout_seconds)
            if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
                raise ValueError("timeout_seconds must be None or finite and nonnegative")
            # The OS-backed lock wait accepts only finite values up to this
            # platform limit, even though Python floats can represent larger
            # finite timeouts. Preserve the caller's effectively-unbounded
            # wait by clamping to the largest supported lock timeout.
            timeout_seconds = min(timeout_seconds, threading.TIMEOUT_MAX)

        lock = self._lock
        wait_started = time.monotonic()
        acquired = (lock.acquire() if timeout_seconds is None else
                    lock.acquire(timeout=timeout_seconds))
        if not acquired:
            raise TimeoutError("timed out waiting for serial calibration measurement reservation")
        waited_seconds = time.monotonic() - wait_started
        self._local.active = True
        try:
            yield waited_seconds
        finally:
            self._local.active = False
            lock.release()


_COORDINATOR = _SerialCalibrationCoordinator()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_COORDINATOR._reset_after_fork)


def serial_calibration_measurement(timeout_seconds: float | None = None):
    """Reserve the sole process-local cold-measurement slot.

    Caller should revalidate identity and recheck its cache after entry, before
    starting measurement timers.
    """
    return _COORDINATOR.reserve(timeout_seconds)


__all__ = ("serial_calibration_measurement",)
