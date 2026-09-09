"""Pure, immutable run-deadline records for fail-closed recovery."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RECORD_KEYS = {
    "schema_version", "run_id", "policy_digest", "boot_id", "start_monotonic_ns",
    "start_realtime_ns", "duration_ns",
}
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_POLICY_DIGEST = re.compile(r"[0-9a-f]{64}")
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_MAX_NS = (1 << 63) - 1


class RunDeadlineError(RuntimeError):
    """A persisted deadline cannot safely grant execution time."""


@dataclass(frozen=True)
class RestoredRunDeadline:
    deadline_monotonic_ns: int
    remaining_ns: int
    expired: bool

    @property
    def deadline_monotonic(self) -> float:
        return self.deadline_monotonic_ns / 1_000_000_000

    @property
    def remaining_seconds(self) -> float:
        return self.remaining_ns / 1_000_000_000


def _identity(value: Any, pattern: re.Pattern[str], name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise RunDeadlineError(f"invalid {name}")
    return value


def _boot_identity(value: Any = None) -> str:
    if value is None:
        try:
            value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except (OSError, UnicodeError) as exc:
            raise RunDeadlineError("OS boot identity is unavailable") from exc
    return _identity(value, _BOOT_ID, "boot identity")


def _clock_ns(value: Any, default, name: str) -> int:
    if value is None:
        value = default()
    if type(value) is not int or not 0 <= value <= _MAX_NS:
        raise RunDeadlineError(f"{name} must be a nonnegative integer")
    return value


def _record_clock_ns(value: Any, name: str) -> int:
    """Validate persisted data without treating JSON null as an omitted input."""
    if type(value) is not int or not 0 <= value <= _MAX_NS:
        raise RunDeadlineError(f"{name} must be a nonnegative integer")
    return value


def _duration_seconds(value: Any, name: str) -> int | float:
    if type(value) is int:
        if value <= 0:
            raise RunDeadlineError(f"{name} must be finite and positive")
        return value
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise RunDeadlineError(f"{name} must be finite and positive")
    return value


def create_run_deadline_record(
    *,
    run_id: str,
    policy_digest: str,
    job_unknown_seconds: int | float,
    job_max_seconds: int | float,
    monotonic_ns: int | None = None,
    realtime_ns: int | None = None,
    boot_id: str | None = None,
) -> dict[str, str | int]:
    """Create the seven-field deadline record; callers own durable persistence."""
    run_id = _identity(run_id, _RUN_ID, "run id")
    policy_digest = _identity(policy_digest, _POLICY_DIGEST, "policy digest")
    unknown = _duration_seconds(job_unknown_seconds, "job_unknown_seconds")
    maximum = _duration_seconds(job_max_seconds, "job_max_seconds")
    seconds = min(unknown, maximum)
    if seconds > _MAX_NS / 1_000_000_000:
        raise RunDeadlineError("derived duration exceeds the safe nanosecond range")
    duration_ns = int(seconds * 1_000_000_000)
    if duration_ns <= 0:
        raise RunDeadlineError("derived duration is below one nanosecond")
    start_monotonic = _clock_ns(monotonic_ns, time.monotonic_ns, "monotonic clock")
    if start_monotonic > _MAX_NS - duration_ns:
        raise RunDeadlineError("derived deadline exceeds the safe nanosecond range")
    return {
        "schema_version": "factor_engine.run_deadline.v1",
        "run_id": run_id,
        "policy_digest": policy_digest,
        "boot_id": _boot_identity(boot_id),
        "start_monotonic_ns": start_monotonic,
        "start_realtime_ns": _clock_ns(realtime_ns, time.time_ns, "realtime clock"),
        "duration_ns": duration_ns,
    }


def restore_run_deadline(
    record: Any,
    *,
    expected_run_id: str,
    expected_policy_digest: str,
    monotonic_ns: int | None = None,
    realtime_ns: int | None = None,
    boot_id: str | None = None,
) -> RestoredRunDeadline:
    """Restore only the original same-boot monotonic deadline, never refresh it."""
    expected_run_id = _identity(expected_run_id, _RUN_ID, "expected run id")
    expected_policy_digest = _identity(
        expected_policy_digest, _POLICY_DIGEST, "expected policy digest"
    )
    if type(record) is not dict or set(record) != _RECORD_KEYS:
        raise RunDeadlineError("deadline record requires exact fields")
    if (type(record["schema_version"]) is not str
            or record["schema_version"] != "factor_engine.run_deadline.v1"):
        raise RunDeadlineError("deadline record has an unsupported schema")
    run_id = _identity(record["run_id"], _RUN_ID, "record run id")
    policy_digest = _identity(
        record["policy_digest"], _POLICY_DIGEST, "record policy digest"
    )
    if record["boot_id"] is None:
        raise RunDeadlineError("invalid record boot identity")
    recorded_boot = _boot_identity(record["boot_id"])
    start_monotonic = _record_clock_ns(
        record["start_monotonic_ns"], "record monotonic clock"
    )
    start_realtime = _record_clock_ns(
        record["start_realtime_ns"], "record realtime clock"
    )
    duration = _record_clock_ns(record["duration_ns"], "record duration")
    if duration <= 0:
        raise RunDeadlineError("record duration must be positive")
    if run_id != expected_run_id or policy_digest != expected_policy_digest:
        raise RunDeadlineError("deadline record identity differs from expected run")
    current_boot = _boot_identity(boot_id)
    if current_boot != recorded_boot:
        raise RunDeadlineError("deadline cannot be restored across an OS boot change")
    current_monotonic = _clock_ns(monotonic_ns, time.monotonic_ns, "monotonic clock")
    current_realtime = _clock_ns(realtime_ns, time.time_ns, "realtime clock")
    if current_monotonic < start_monotonic:
        raise RunDeadlineError("monotonic clock moved backwards")
    if current_realtime < start_realtime:
        raise RunDeadlineError("realtime clock moved backwards")
    if start_monotonic > _MAX_NS - duration:
        raise RunDeadlineError("record deadline exceeds the safe nanosecond range")
    deadline = start_monotonic + duration
    remaining = max(0, deadline - current_monotonic)
    return RestoredRunDeadline(deadline, remaining, remaining == 0)
