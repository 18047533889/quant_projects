"""Durable, fail-closed worker process ownership evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_JOURNAL = "worker-ownership.jsonl"
_SCHEMA = "factor_engine.worker_ownership.v1"
_MAX_BYTES = 1024 * 1024
_MAX_RECORDS = 10_000
_MAX_LINE_BYTES = 16 * 1024
_PROCESS_LOCK = threading.RLock()


class WorkerOwnershipError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerInstance:
    instance_id: str
    capability: str


@dataclass(frozen=True)
class RunOwnershipContext:
    """Caller-provided immutable identity, never inferred from the journal."""

    run_id: str
    identity_sha256: str

    def __post_init__(self):
        for value, size in ((self.run_id, 32), (self.identity_sha256, 64)):
            if (type(value) is not str or len(value) != size
                    or any(char not in "0123456789abcdef" for char in value)):
                raise WorkerOwnershipError("invalid run ownership context")


@dataclass(frozen=True)
class WorkerOwnershipValidation:
    all_exited: bool
    reasons: tuple[str, ...]
    instances: tuple[str, ...]


@dataclass(frozen=True)
class WorkerOwnershipInspection:
    """Strict journal completeness plus read-only current process evidence."""

    journal_complete: bool
    all_process_identities_retired_now: bool
    reasons: tuple[str, ...]
    instances: tuple[str, ...]


_EXPECTED_EVENT_KEYS = {
    "STARTING": {"schema", "event", "instance_id", "capability_hash", "role", "recorded_monotonic"},
    "BOUND": {"schema", "event", "instance_id", "pid", "starttime", "boot_id",
              "machine_id", "pid_namespace", "recorded_monotonic"},
    "EXITED": {"schema", "event", "instance_id", "pid", "starttime", "boot_id",
               "machine_id", "pid_namespace", "exit_evidence", "observed_boot_id",
               "recorded_monotonic"},
}


def _valid_recorded_monotonic(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _serialized(function):
    def guarded(*args, **kwargs):
        if not _PROCESS_LOCK.acquire(timeout=1.0):
            raise WorkerOwnershipError("worker ownership process lock timed out")
        try:
            import fcntl
            run_dir = _journal_path(args[0] if args else kwargs["run_dir"]).parent
            lock_path = run_dir / "worker-ownership.lock"
            fd = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            deadline = time.monotonic() + 1.0
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise WorkerOwnershipError("worker ownership lock must be a regular file")
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise WorkerOwnershipError("worker ownership lock timed out")
                        time.sleep(0.01)
                return function(*args, **kwargs)
            finally:
                os.close(fd)
        finally:
            _PROCESS_LOCK.release()
    return guarded


def _finite_flock(fd: int, operation: int) -> None:
    import fcntl
    deadline = time.monotonic() + 1.0
    while True:
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise WorkerOwnershipError("worker ownership journal lock timed out")
            time.sleep(0.01)


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, UnicodeError) as exc:
        raise WorkerOwnershipError("OS boot identity is unavailable") from exc
    if not value:
        raise WorkerOwnershipError("OS boot identity is unavailable")
    return value


def _host_identity() -> tuple[str, str]:
    try:
        machine = Path("/etc/machine-id").read_text().strip()
        ns = os.stat("/proc/self/ns/pid")
    except (OSError, UnicodeError) as exc:
        raise WorkerOwnershipError("host or PID namespace identity is unavailable") from exc
    if not machine:
        raise WorkerOwnershipError("host identity is unavailable")
    return machine, f"{ns.st_dev}:{ns.st_ino}"


def _process_starttime(pid: int) -> tuple[str, str | None]:
    """Return SAME-capable starttime, ABSENT, or UNKNOWN without guessing."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "ABSENT", None
        except (PermissionError, OSError):
            return "UNKNOWN", None
        return "UNKNOWN", None
    except (OSError, UnicodeError):
        return "UNKNOWN", None
    end = raw.rfind(")")
    if end < 0:
        return "UNKNOWN", None
    fields = raw[end + 2 :].split()
    if len(fields) <= 19 or not fields[19].isdigit():
        return "UNKNOWN", None
    return "PRESENT", fields[19]


def _journal_path(run_dir: str | os.PathLike[str]) -> Path:
    root = Path(run_dir)
    if not root.is_dir() or root.is_symlink():
        raise WorkerOwnershipError("worker ownership run directory is unavailable or unsafe")
    return root / _JOURNAL


def _append(run_dir: str | os.PathLike[str], event: dict[str, Any],
            *, context: RunOwnershipContext) -> None:
    import fcntl

    # Public routing happens before the JSON backend's serialized section.
    # Recheck at its mutation boundary so a newly conflicting store cannot be
    # silently adopted between selection and this append.
    if _selected_store(run_dir, context) != "json-v1":
        raise WorkerOwnershipError("JSON mutation conflicts with selected ownership store")
    path = _journal_path(run_dir)
    event = {**event, "run_id": context.run_id,
             "identity_sha256": context.identity_sha256}
    encoded = (json.dumps(event, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    if len(encoded) > _MAX_LINE_BYTES:
        raise WorkerOwnershipError("worker ownership record exceeds size bound")
    flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        created = not path.exists()
        fd = os.open(path, flags, 0o600)
        try:
            _finite_flock(fd, fcntl.LOCK_EX)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size + len(encoded) > _MAX_BYTES:
                raise WorkerOwnershipError("worker ownership journal exceeds safety bound")
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise WorkerOwnershipError("worker ownership journal write was incomplete")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except OSError as exc:
        raise WorkerOwnershipError("worker ownership journal write failed") from exc


def _capability_hash(capability: str) -> str:
    return hashlib.sha256(capability.encode("ascii")).hexdigest()


def _verify_context(run_dir, context):
    if type(context) is not RunOwnershipContext:
        raise WorkerOwnershipError("explicit run ownership context is required")
    path = _journal_path(run_dir).parent / "identity.json"
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
                raise WorkerOwnershipError("run identity file exceeds safety bound")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                payload = stream.read(_MAX_BYTES + 1)
        finally:
            os.close(fd)
    except OSError as exc:
        raise WorkerOwnershipError("run identity is unavailable") from exc
    if (len(payload) > _MAX_BYTES
            or hashlib.sha256(payload).hexdigest() != context.identity_sha256):
        raise WorkerOwnershipError("run identity digest mismatch")
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate identity key")
                result[key] = value
            return result
        identity = json.loads(payload, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise WorkerOwnershipError("run identity is corrupt") from exc
    if (type(identity) is not dict or type(identity.get("run_id")) is not str
            or identity["run_id"] != context.run_id):
        raise WorkerOwnershipError("run identity differs from expected run")
    return identity


def _selected_store(run_dir, context) -> str:
    identity = _verify_context(run_dir, context)
    selected = identity.get("ownership_store", "json-v1")
    if type(selected) is not str or selected not in {"json-v1", "sqlite-v1"}:
        raise WorkerOwnershipError("run identity selects an unknown ownership store")
    incompatible = (
        "worker-ownership.sqlite3" if selected == "json-v1" else _JOURNAL
    )
    if os.path.lexists(_journal_path(run_dir).parent / incompatible):
        raise WorkerOwnershipError("incompatible worker ownership stores coexist")
    return selected


def start_worker(run_dir: str | os.PathLike[str], role: str,
                 *, context: RunOwnershipContext) -> WorkerInstance:
    """Dispatch lifecycle storage only from the exact durable run identity."""
    if _selected_store(run_dir, context) == "sqlite-v1":
        from factor_engine.runtime.worker_ownership_sqlite import start_worker_sqlite
        return start_worker_sqlite(run_dir, role, context=context)
    return _start_worker_json(run_dir, role, context=context)


def bind_worker(run_dir: str | os.PathLike[str], instance: WorkerInstance,
                pid: int, *, context: RunOwnershipContext) -> None:
    if _selected_store(run_dir, context) == "sqlite-v1":
        from factor_engine.runtime.worker_ownership_sqlite import bind_worker_sqlite
        return bind_worker_sqlite(run_dir, instance, pid, context=context)
    return _bind_worker_json(run_dir, instance, pid, context=context)


def mark_worker_exited(run_dir: str | os.PathLike[str], instance: WorkerInstance,
                       *, context: RunOwnershipContext) -> None:
    if _selected_store(run_dir, context) == "sqlite-v1":
        from factor_engine.runtime.worker_ownership_sqlite import mark_worker_exited_sqlite
        return mark_worker_exited_sqlite(run_dir, instance, context=context)
    return _mark_worker_exited_json(run_dir, instance, context=context)


def _bound_events(run_dir, context):
    _verify_context(run_dir, context)
    events = _events(run_dir)
    header = {"schema": _SCHEMA, "event": "HEADER",
              "run_id": context.run_id, "identity_sha256": context.identity_sha256}
    if events[0] != header:
        raise WorkerOwnershipError("worker journal header/run binding mismatch")
    result = []
    for event in events[1:]:
        if (event.get("run_id") != context.run_id
                or event.get("identity_sha256") != context.identity_sha256):
            raise WorkerOwnershipError("worker event/run binding mismatch")
        result.append({key: value for key, value in event.items()
                       if key not in {"run_id", "identity_sha256"}})
    return result


@_serialized
def _start_worker_json(run_dir: str | os.PathLike[str], role: str,
                 *, context: RunOwnershipContext) -> WorkerInstance:
    """Persist STARTING before a caller creates the worker process."""
    if type(role) is not str or not role:
        raise ValueError("worker role must be a nonempty string")
    _verify_context(run_dir, context)
    if _journal_path(run_dir).exists():
        _bound_events(run_dir, context)
    else:
        _append(run_dir, {"schema": _SCHEMA, "event": "HEADER"}, context=context)
    instance = WorkerInstance(uuid.uuid4().hex, uuid.uuid4().hex)
    _append(run_dir, {
        "schema": _SCHEMA, "event": "STARTING", "instance_id": instance.instance_id,
        "capability_hash": _capability_hash(instance.capability), "role": role,
        "recorded_monotonic": time.monotonic(),
    }, context=context)
    return instance


def _events(run_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    import fcntl

    path = _journal_path(run_dir)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        raise WorkerOwnershipError("worker ownership journal is missing") from None
    except OSError as exc:
        raise WorkerOwnershipError("worker ownership journal is unreadable") from exc
    out = []
    try:
        _finite_flock(fd, fcntl.LOCK_SH)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            raise WorkerOwnershipError("worker ownership journal exceeds safety bound")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for index, line in enumerate(stream, start=1):
                if index > _MAX_RECORDS or len(line) > _MAX_LINE_BYTES or not line.endswith(b"\n"):
                    raise WorkerOwnershipError("worker ownership journal is truncated or oversized")
                try:
                    def no_duplicates(pairs):
                        value = {}
                        for key, item in pairs:
                            if key in value:
                                raise ValueError("duplicate JSON key")
                            value[key] = item
                        return value
                    value = json.loads(line, object_pairs_hook=no_duplicates)
                except (UnicodeError, ValueError):
                    raise WorkerOwnershipError("worker ownership journal is corrupt") from None
                if type(value) is not dict or value.get("schema") != _SCHEMA:
                    raise WorkerOwnershipError("worker ownership journal has invalid schema")
                out.append(value)
    finally:
        os.close(fd)
    if not out:
        raise WorkerOwnershipError("worker ownership journal is empty")
    return out


def _history(run_dir: str | os.PathLike[str], instance: WorkerInstance,
             context: RunOwnershipContext) -> list[dict[str, Any]]:
    history = [e for e in _bound_events(run_dir, context)
               if e.get("instance_id") == instance.instance_id]
    if not history or history[0].get("capability_hash") != _capability_hash(instance.capability):
        raise WorkerOwnershipError("worker instance capability mismatch")
    return history


@_serialized
def _bind_worker_json(
    run_dir: str | os.PathLike[str], instance: WorkerInstance, pid: int,
    *, context: RunOwnershipContext,
) -> None:
    """Bind STARTING to a parent-observed live PID and kernel starttime."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("worker pid must be a positive integer")
    history = _history(run_dir, instance, context)
    if [event.get("event") for event in history] != ["STARTING"]:
        raise WorkerOwnershipError("worker instance is not uniquely STARTING")
    status, starttime = _process_starttime(pid)
    if status != "PRESENT" or starttime is None:
        raise WorkerOwnershipError("worker process identity cannot be bound")
    machine_id, pid_namespace = _host_identity()
    _append(run_dir, {
        "schema": _SCHEMA, "event": "BOUND", "instance_id": instance.instance_id,
        "pid": pid, "starttime": starttime, "boot_id": _boot_id(),
        "machine_id": machine_id, "pid_namespace": pid_namespace,
        "recorded_monotonic": time.monotonic(),
    }, context=context)


@_serialized
def _mark_worker_exited_json(
    run_dir: str | os.PathLike[str], instance: WorkerInstance,
    *, context: RunOwnershipContext,
) -> None:
    """Persist EXITED only after explicit OS identity evidence."""
    history = _history(run_dir, instance, context)
    if [event.get("event") for event in history] != ["STARTING", "BOUND"]:
        raise WorkerOwnershipError("worker instance is not uniquely bound and active")
    bound = history[1]
    current_boot = _boot_id()
    machine_id, pid_namespace = _host_identity()
    if (bound.get("machine_id"), bound.get("pid_namespace")) != (
        machine_id, pid_namespace
    ):
        raise WorkerOwnershipError("worker belongs to another host or PID namespace")
    if bound.get("boot_id") != current_boot:
        evidence = "BOOT_CHANGED"
    else:
        status, observed = _process_starttime(bound.get("pid"))
        if status == "UNKNOWN":
            raise WorkerOwnershipError("worker exit identity is unknown")
        if status == "PRESENT" and observed == bound.get("starttime"):
            raise WorkerOwnershipError("worker process is still alive")
        evidence = "PID_ABSENT" if status == "ABSENT" else "PID_REUSED"
    _append(run_dir, {
        "schema": _SCHEMA, "event": "EXITED", "instance_id": instance.instance_id,
        "pid": bound["pid"], "starttime": bound["starttime"],
        "boot_id": bound["boot_id"], "exit_evidence": evidence,
        "machine_id": bound["machine_id"],
        "pid_namespace": bound["pid_namespace"],
        "observed_boot_id": current_boot, "recorded_monotonic": time.monotonic(),
    }, context=context)


def _validate_bound_event_snapshot(
    events: list[dict[str, Any]],
) -> WorkerOwnershipValidation:
    """Validate one already run-bound immutable event snapshot."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    reasons = []
    expected_keys = _EXPECTED_EVENT_KEYS
    for event in events:
        instance_id = event.get("instance_id")
        kind = event.get("event")
        timestamp = event.get("recorded_monotonic")
        if (
            type(instance_id) is not str or len(instance_id) != 32
            or not set(instance_id).issubset(set("0123456789abcdef"))
            or type(kind) is not str
            or kind not in expected_keys or set(event) != expected_keys.get(kind)
            or not _valid_recorded_monotonic(timestamp)
        ):
            reasons.append("invalid worker instance identity")
            continue
        grouped.setdefault(instance_id, []).append(event)
    identities = set()
    try:
        current_machine, current_namespace = _host_identity()
        current_boot = _boot_id()
    except WorkerOwnershipError as exc:
        reasons.append(str(exc))
        current_machine = current_namespace = current_boot = None
    for instance_id, history in grouped.items():
        transitions = [event.get("event") for event in history]
        if transitions != ["STARTING", "BOUND", "EXITED"]:
            reasons.append(f"worker {instance_id} has incomplete or duplicate transitions")
            continue
        starting = history[0]
        if (
            type(starting.get("role")) is not str or not starting["role"]
            or type(starting.get("capability_hash")) is not str
            or len(starting["capability_hash"]) != 64
            or not set(starting["capability_hash"]).issubset(set("0123456789abcdef"))
        ):
            reasons.append(f"worker {instance_id} has invalid STARTING authority")
        bound, exited = history[1], history[2]
        timestamps = [event["recorded_monotonic"] for event in history]
        if (timestamps[0] > timestamps[1]
                or (exited.get("exit_evidence") != "BOOT_CHANGED"
                    and timestamps[1] > timestamps[2])):
            reasons.append(f"worker {instance_id} timestamps moved backwards")
        if (
            type(bound.get("pid")) is not int or bound["pid"] <= 0
            or type(bound.get("starttime")) is not str or not bound["starttime"].isdigit()
            or any(type(bound.get(key)) is not str or not bound.get(key)
                   for key in ("boot_id", "machine_id", "pid_namespace"))
        ):
            reasons.append(f"worker {instance_id} has invalid bound identity")
            continue
        identity = (
            bound.get("machine_id"), bound.get("pid_namespace"),
            bound.get("boot_id"), bound.get("pid"), bound.get("starttime"),
        )
        if identity in identities:
            reasons.append("duplicate worker process identity")
        identities.add(identity)
        if any(exited.get(key) != bound.get(key) for key in (
            "machine_id", "pid_namespace", "boot_id", "pid", "starttime"
        )):
            reasons.append(f"worker {instance_id} exit identity mismatch")
        if (
            type(exited.get("pid")) is not int
            or type(exited.get("starttime")) is not str
            or any(type(exited.get(key)) is not str or not exited.get(key) for key in (
                "boot_id", "machine_id", "pid_namespace", "observed_boot_id",
                "exit_evidence",
            ))
        ):
            reasons.append(f"worker {instance_id} has invalid EXITED evidence types")
            continue
        if exited.get("exit_evidence") not in {"PID_ABSENT", "PID_REUSED", "BOOT_CHANGED"}:
            reasons.append(f"worker {instance_id} lacks explicit exit evidence")
        if ((exited["observed_boot_id"] != bound["boot_id"])
                != (exited["exit_evidence"] == "BOOT_CHANGED")):
            reasons.append(f"worker {instance_id} has inconsistent boot-change evidence")
        if (bound.get("machine_id"), bound.get("pid_namespace")) != (
            current_machine, current_namespace
        ):
            reasons.append(f"worker {instance_id} belongs to another host or PID namespace")
        elif bound.get("boot_id") == current_boot:
            status, observed = _process_starttime(bound["pid"])
            if status == "UNKNOWN" or (status == "PRESENT" and observed == bound["starttime"]):
                reasons.append(f"worker {instance_id} exit cannot be confirmed by current OS")
    return WorkerOwnershipValidation(
        not reasons and bool(grouped), tuple(reasons), tuple(sorted(grouped))
    )


def validate_all_workers_exited(
    run_dir: str | os.PathLike[str], *, context: RunOwnershipContext,
) -> WorkerOwnershipValidation:
    """Read-only structural validation; every ambiguity returns fail closed."""
    try:
        events = _bound_events(run_dir, context)
    except WorkerOwnershipError as exc:
        return WorkerOwnershipValidation(False, (str(exc),), ())
    return _validate_bound_event_snapshot(events)


def inspect_worker_retirement_now(
    run_dir: str | os.PathLike[str], *, context: RunOwnershipContext,
) -> WorkerOwnershipInspection:
    """Read current OS evidence without completing or repairing the journal."""
    try:
        events = _bound_events(run_dir, context)
    except WorkerOwnershipError as exc:
        return WorkerOwnershipInspection(False, False, (str(exc),), ())
    strict = _validate_bound_event_snapshot(events)
    grouped: dict[str, list[dict[str, Any]]] = {}
    reasons = []
    for event in events:
        instance_id = event.get("instance_id")
        kind = event.get("event")
        timestamp = event.get("recorded_monotonic")
        if (
            type(instance_id) is not str or len(instance_id) != 32
            or not set(instance_id).issubset(set("0123456789abcdef"))
            or type(kind) is not str or kind not in _EXPECTED_EVENT_KEYS
            or set(event) != _EXPECTED_EVENT_KEYS.get(kind)
            or not _valid_recorded_monotonic(timestamp)
        ):
            return WorkerOwnershipInspection(
                False, False, ("invalid worker instance identity",), ()
            )
        grouped.setdefault(instance_id, []).append(event)
    if not grouped:
        return WorkerOwnershipInspection(False, False, ("worker journal has no instances",), ())
    try:
        current_machine, current_namespace = _host_identity()
        current_boot = _boot_id()
    except WorkerOwnershipError as exc:
        return WorkerOwnershipInspection(
            False, False, (str(exc),), tuple(sorted(grouped))
        )
    identities = set()
    all_retired = True
    for instance_id, history in grouped.items():
        transitions = [event.get("event") for event in history]
        if transitions not in (["STARTING"], ["STARTING", "BOUND"],
                                ["STARTING", "BOUND", "EXITED"]):
            return WorkerOwnershipInspection(
                False, False,
                (f"worker {instance_id} has invalid or duplicate transitions",),
                tuple(sorted(grouped)),
            )
        starting = history[0]
        if (
            type(starting.get("role")) is not str or not starting["role"]
            or type(starting.get("capability_hash")) is not str
            or len(starting["capability_hash"]) != 64
            or not set(starting["capability_hash"]).issubset(
                set("0123456789abcdef")
            )
        ):
            return WorkerOwnershipInspection(
                False, False,
                (f"worker {instance_id} has invalid STARTING authority",),
                tuple(sorted(grouped)),
            )
        if transitions == ["STARTING"]:
            all_retired = False
            reasons.append(
                f"worker {instance_id} journal history is not closed; "
                "current process identity is unknown"
            )
            continue
        bound = history[1]
        if (
            type(bound.get("pid")) is not int or bound["pid"] <= 0
            or type(bound.get("starttime")) is not str
            or not bound["starttime"].isdigit()
            or any(type(bound.get(key)) is not str or not bound.get(key)
                   for key in ("boot_id", "machine_id", "pid_namespace"))
            or history[0]["recorded_monotonic"] > bound["recorded_monotonic"]
        ):
            return WorkerOwnershipInspection(
                False, False, (f"worker {instance_id} has invalid bound identity",),
                tuple(sorted(grouped)),
            )
        identity = (
            bound["machine_id"], bound["pid_namespace"], bound["boot_id"],
            bound["pid"], bound["starttime"],
        )
        if identity in identities:
            return WorkerOwnershipInspection(
                False, False, ("duplicate worker process identity",),
                tuple(sorted(grouped)),
            )
        identities.add(identity)
        if (bound["machine_id"], bound["pid_namespace"]) != (
            current_machine, current_namespace
        ):
            retired = False
            evidence = "belongs to another host or PID namespace"
        elif bound["boot_id"] != current_boot:
            retired = True
            evidence = "is retired after an OS boot change"
        else:
            status, observed = _process_starttime(bound["pid"])
            retired = status == "ABSENT" or (
                status == "PRESENT" and observed != bound["starttime"]
            )
            evidence = (
                "is retired by current OS identity evidence" if retired
                else "cannot be proven retired by the current OS"
            )
        all_retired = all_retired and retired
        if len(history) == 2:
            reasons.append(
                f"worker {instance_id} journal history is not closed; "
                f"current process identity {evidence}"
            )
            continue
        exited = history[2]
        if (
            type(exited.get("pid")) is not int or exited["pid"] <= 0
            or type(exited.get("starttime")) is not str
            or not exited["starttime"].isdigit()
            or any(type(exited.get(key)) is not str or not exited.get(key)
                   for key in ("boot_id", "machine_id", "pid_namespace"))
            or any(exited.get(key) != bound.get(key) for key in (
                "machine_id", "pid_namespace", "boot_id", "pid", "starttime"
            ))
            or any(type(exited.get(key)) is not str or not exited.get(key)
                   for key in ("observed_boot_id", "exit_evidence"))
            or exited.get("exit_evidence") not in {
                "PID_ABSENT", "PID_REUSED", "BOOT_CHANGED"
            }
            or ((exited["observed_boot_id"] != bound["boot_id"])
                != (exited["exit_evidence"] == "BOOT_CHANGED"))
            or (exited.get("exit_evidence") != "BOOT_CHANGED"
                and bound["recorded_monotonic"] > exited["recorded_monotonic"])
        ):
            return WorkerOwnershipInspection(
                False, False, (f"worker {instance_id} has invalid EXITED evidence",),
                tuple(sorted(grouped)),
            )
        if not retired:
            reasons.append(
                f"worker {instance_id} complete journal conflicts with current OS evidence: "
                f"{evidence}"
            )
    return WorkerOwnershipInspection(
        strict.all_exited,
        all_retired,
        tuple(reasons if reasons else strict.reasons),
        tuple(sorted(grouped)),
    )
