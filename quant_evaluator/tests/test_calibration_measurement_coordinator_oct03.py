"""Concurrency and fork-safety tests for cold calibration measurements."""
from __future__ import annotations

import math
import os
import select
import signal
import threading

import pytest

from quant_evaluator.runtime import calibration_measurement_coordinator as coordinator


def test_concurrent_cold_measurements_are_serial_and_report_queue_wait(monkeypatch):
    first_entered = threading.Event()
    release_first = threading.Event()
    second_acquiring = threading.Event()
    second_entered = threading.Event()
    active_guard = threading.Lock()
    active = 0
    maximum_active = 0
    failures = []

    underlying_lock = threading.Lock()

    class ObservedLock:
        def acquire(self, *args, **kwargs):
            if threading.current_thread().name == "calibration-contender":
                second_acquiring.set()
            return underlying_lock.acquire(*args, **kwargs)

        def release(self):
            underlying_lock.release()

    monkeypatch.setattr(coordinator._COORDINATOR, "_lock", ObservedLock())

    def enter_active():
        nonlocal active, maximum_active
        with active_guard:
            active += 1
            maximum_active = max(maximum_active, active)

    def leave_active():
        nonlocal active
        with active_guard:
            active -= 1

    def first_worker():
        try:
            with coordinator.serial_calibration_measurement() as waited:
                assert math.isfinite(waited) and waited >= 0
                enter_active()
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise AssertionError("test did not release first reservation")
                leave_active()
        except BaseException as exc:
            failures.append(exc)

    def second_worker():
        try:
            if not first_entered.wait(timeout=5):
                raise AssertionError("first reservation did not start")
            with coordinator.serial_calibration_measurement() as waited:
                second_entered.set()
                assert math.isfinite(waited) and waited >= 0
                enter_active()
                leave_active()
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=first_worker, name="calibration-owner")
    second = threading.Thread(target=second_worker, name="calibration-contender")
    first.start()
    second.start()
    try:
        assert second_acquiring.wait(timeout=5)
        assert first_entered.is_set()
        assert not second_entered.is_set()
    finally:
        release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert not failures
    assert maximum_active == 1
    assert active == 0


def test_exception_releases_reservation_for_next_measurement():
    with pytest.raises(ValueError, match="measurement failed"):
        with coordinator.serial_calibration_measurement():
            raise ValueError("measurement failed")

    with coordinator.serial_calibration_measurement() as waited:
        assert math.isfinite(waited) and waited >= 0


def test_same_thread_nested_measurement_fails_without_deadlocking():
    with coordinator.serial_calibration_measurement():
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with coordinator.serial_calibration_measurement():
                pytest.fail("nested measurement unexpectedly entered")

    # The rejected nested reservation must not poison the outer lock.
    with coordinator.serial_calibration_measurement():
        pass


@pytest.mark.skipif(not hasattr(os, "fork") or not hasattr(os, "register_at_fork"),
                    reason="fork hooks are unavailable")
def test_forked_child_rebuilds_reservation_while_parent_holds_lock():
    read_fd, write_fd = os.pipe()
    child_pid = None
    try:
        with coordinator.serial_calibration_measurement():
            child_pid = os.fork()
            if child_pid == 0:
                os.close(read_fd)
                try:
                    with coordinator.serial_calibration_measurement() as waited:
                        payload = f"ok:{waited}".encode("ascii")
                    os.write(write_fd, payload)
                    os._exit(0)
                except BaseException as exc:
                    os.write(write_fd, f"error:{type(exc).__name__}".encode("ascii"))
                    os._exit(1)

            os.close(write_fd)
            ready, _, _ = select.select([read_fd], [], [], 3.0)
            if not ready:
                os.kill(child_pid, signal.SIGKILL)
                os.waitpid(child_pid, 0)
                child_pid = None
                pytest.fail("child inherited a locked calibration reservation")
            payload = os.read(read_fd, 256).decode("ascii")
            _, status = os.waitpid(child_pid, 0)
            child_pid = None
            assert os.waitstatus_to_exitcode(status) == 0
            assert payload.startswith("ok:")
            assert math.isfinite(float(payload.split(":", 1)[1]))
    finally:
        os.close(read_fd)
        if child_pid is not None:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
        try:
            os.close(write_fd)
        except OSError:
            pass



@pytest.mark.parametrize("timeout_seconds", [-1, math.nan, math.inf, True, "1"])
def test_invalid_reservation_timeout_is_rejected(timeout_seconds):
    with pytest.raises(ValueError, match="timeout_seconds"):
        with coordinator.serial_calibration_measurement(timeout_seconds):
            pytest.fail("invalid timeout unexpectedly entered")


def test_zero_timeout_fails_fast_when_reservation_is_busy():
    lock = coordinator._COORDINATOR._lock
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(TimeoutError, match="timed out waiting"):
            with coordinator.serial_calibration_measurement(timeout_seconds=0):
                pytest.fail("busy reservation unexpectedly entered")
    finally:
        lock.release()

    with coordinator.serial_calibration_measurement(timeout_seconds=0) as waited:
        assert math.isfinite(waited) and waited >= 0


def test_oversized_finite_timeout_is_clamped_and_enters_when_uncontended():
    with coordinator.serial_calibration_measurement(timeout_seconds=1e300) as waited:
        assert math.isfinite(waited) and waited >= 0
