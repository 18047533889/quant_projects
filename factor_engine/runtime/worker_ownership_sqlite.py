"""Indexed durable worker ownership store for explicitly selected new runs."""
from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from factor_engine.runtime.worker_ownership import (
    RunOwnershipContext,
    WorkerInstance,
    WorkerOwnershipError,
    _EXPECTED_EVENT_KEYS,
    _boot_id,
    _capability_hash,
    _host_identity,
    _process_starttime,
    _valid_recorded_monotonic,
    _verify_context,
    _validate_bound_event_snapshot,
)


STORE_NAME = "worker-ownership.sqlite3"
STORE_SCHEMA = "factor_engine.worker_ownership.sqlite.v1"
LEGACY_NAME = "worker-ownership.jsonl"
MAX_INSTANCES = 1_000_000
MAX_EVENTS = 3_000_000
_BUSY_MS = 1_000
_MAX_DB_BYTES = 1024 * 1024 * 1024
_SQLITE_MAX_VALUE_BYTES = 64 * 1024


@dataclass(frozen=True)
class SQLiteWorkerOwnershipValidation:
    all_exited: bool
    reasons: tuple[str, ...]
    instance_count: int


@dataclass(frozen=True)
class SQLiteWorkerOwnershipInspection:
    journal_complete: bool
    all_process_identities_retired_now: bool
    reasons: tuple[str, ...]
    instance_count: int
    example_instances: tuple[str, ...]
    truncated: bool


_INSPECTION_EXAMPLE_LIMIT = 32
_INSPECTION_REASON_LIMIT = 64
_INSPECTION_PAGE_SIZE = 512


def _store_path(run_dir: str | os.PathLike[str]) -> Path:
    root = Path(run_dir)
    if not root.is_dir() or root.is_symlink():
        raise WorkerOwnershipError("worker ownership run directory is unavailable or unsafe")
    return root / STORE_NAME


def _reject_legacy_coexistence(path: Path) -> None:
    if os.path.lexists(path.parent / LEGACY_NAME):
        raise WorkerOwnershipError("legacy ownership journal cannot coexist with SQLite")


def _verify_sqlite_identity(run_dir, context) -> None:
    identity = _verify_context(run_dir, context)
    if identity.get("ownership_store") != "sqlite-v1":
        raise WorkerOwnershipError("run identity does not select SQLite ownership")


def _regular_identity(path: Path) -> tuple[int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise WorkerOwnershipError("worker ownership database is unavailable") from exc
    if (path.is_symlink() or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid() or info.st_mode & 0o077):
        raise WorkerOwnershipError("worker ownership database must be a regular file")
    return info.st_dev, info.st_ino


def _configure(connection: sqlite3.Connection) -> None:
    connection.execute(f"PRAGMA busy_timeout={_BUSY_MS}")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=FULL")


def _apply_sqlite_limits(connection: sqlite3.Connection) -> None:
    """Bound native SQLite allocations before executing untrusted SQL."""
    try:
        category = sqlite3.SQLITE_LIMIT_LENGTH
        connection.setlimit(category, _SQLITE_MAX_VALUE_BYTES)
        if connection.getlimit(category) > _SQLITE_MAX_VALUE_BYTES:
            raise WorkerOwnershipError("worker ownership SQLite length limit is unavailable")
    except (AttributeError, sqlite3.Error) as exc:
        raise WorkerOwnershipError("worker ownership SQLite length limit is unavailable") from exc


def _finite_flock(fd: int, operation: int) -> None:
    import fcntl
    deadline = time.monotonic() + 1.0
    while True:
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise WorkerOwnershipError("worker ownership descriptor lock timed out")
            time.sleep(0.01)


def _connect_existing(path: Path) -> tuple[sqlite3.Connection, int]:
    """Open mode=rw while holding and checking the exact pre-open inode."""
    import fcntl

    _reject_legacy_coexistence(path)
    before = _regular_identity(path)
    fd = None
    connection = None
    succeeded = False
    try:
        fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != before:
            raise WorkerOwnershipError("worker ownership database descriptor mismatch")
        _finite_flock(fd, fcntl.LOCK_EX)
        uri = "file:" + urllib.parse.quote(str(path), safe="/") + "?mode=rw"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        _apply_sqlite_limits(connection)
        _configure(connection)
        if _regular_identity(path) != before:
            raise WorkerOwnershipError("worker ownership database path changed during open")
        if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete":
            raise WorkerOwnershipError("worker ownership database has an unsafe journal mode")
        succeeded = True
        return connection, fd
    except (sqlite3.Error, OSError) as exc:
        raise WorkerOwnershipError("worker ownership database open failed") from exc
    finally:
        if not succeeded:
            if connection is not None:
                connection.close()
            if fd is not None:
                os.close(fd)


def _safe_path_identity(path: Path) -> tuple[int, int] | None:
    try:
        return _regular_identity(path)
    except WorkerOwnershipError:
        return None


def _close_write_connection(connection: sqlite3.Connection, fd: int) -> None:
    connection.close()
    os.close(fd)


def _readonly_snapshot_connection(path: Path) -> tuple[sqlite3.Connection, int]:
    """Bind SQLite to an O_NOFOLLOW fd and refuse any hot-journal recovery."""
    import fcntl

    _reject_legacy_coexistence(path)
    if os.path.lexists(str(path) + "-journal"):
        raise WorkerOwnershipError("worker ownership database has a hot journal")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise WorkerOwnershipError("worker ownership database is unavailable") from exc
    connection = None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_DB_BYTES:
            raise WorkerOwnershipError("worker ownership database exceeds safety bound")
        _finite_flock(fd, fcntl.LOCK_SH)
        uri = f"file:/proc/self/fd/{fd}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        _apply_sqlite_limits(connection)
        _configure(connection)
        connection.execute("PRAGMA query_only=ON")
        if os.path.lexists(str(path) + "-journal"):
            raise WorkerOwnershipError("worker ownership database has a hot journal")
        return connection, fd
    except BaseException:
        if connection is not None:
            connection.close()
        os.close(fd)
        raise


def _verify_binding(connection: sqlite3.Connection, context: RunOwnershipContext) -> None:
    try:
        raw_rows = connection.execute(
            "SELECT key,value FROM ownership_meta LIMIT 4"
        ).fetchall()
    except sqlite3.Error as exc:
        raise WorkerOwnershipError("worker ownership database schema is unavailable") from exc
    if len(raw_rows) != 3 or len({row[0] for row in raw_rows}) != 3:
        raise WorkerOwnershipError("worker ownership database run binding mismatch")
    rows = dict(raw_rows)
    expected = {
        "schema": STORE_SCHEMA,
        "run_id": context.run_id,
        "identity_sha256": context.identity_sha256,
    }
    if rows != expected:
        raise WorkerOwnershipError("worker ownership database run binding mismatch")


def _index_signature(connection: sqlite3.Connection, table: str) -> dict[str, tuple[bool, bool, tuple[str, ...]]]:
    """Return unique/partial/column metadata without reading untrusted rows."""
    signatures = {}
    for row in connection.execute(f"PRAGMA index_list('{table}')"):
        name, unique, partial = row[1], row[2], row[4]
        columns = tuple(item[2] for item in connection.execute(
            "PRAGMA index_info(" + json.dumps(name) + ")"
        ))
        signatures[name] = (unique == 1, partial == 1, columns)
    return signatures


def _verify_schema_indexes(connection: sqlite3.Connection) -> None:
    """Reject altered cardinality/index invariants before reading payloads."""
    tables = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_schema WHERE type='table'"
    )}
    required = {"ownership_meta", "store_counters", "worker_instances", "worker_events"}
    if not required.issubset(tables):
        raise WorkerOwnershipError("worker ownership database schema is unavailable")
    instance_indexes = _index_signature(connection, "worker_instances")
    process = instance_indexes.get("worker_process_idx")
    if process != (True, True, (
        "machine_id", "pid_namespace", "boot_id", "pid", "starttime",
    )):
        raise WorkerOwnershipError("worker ownership process identity index is unavailable")
    process_sql_row = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type='index' AND name='worker_process_idx'"
    ).fetchone()
    process_sql = "" if process_sql_row is None or type(process_sql_row[0]) is not str else "".join(
        process_sql_row[0].lower().split()
    )
    if process_sql != (
        "createuniqueindexworker_process_idxonworker_instances("
        "machine_id,pid_namespace,boot_id,pid,starttime)wherepidisnotnull"
    ):
        raise WorkerOwnershipError("worker ownership process identity predicate is unavailable")
    event_indexes = _index_signature(connection, "worker_events")
    if not any(unique and not partial and columns == ("instance_id", "event")
               for unique, partial, columns in event_indexes.values()):
        raise WorkerOwnershipError("worker ownership event uniqueness is unavailable")
    if event_indexes.get("worker_event_instance_idx") != (
        False, False, ("instance_id", "seq"),
    ):
        raise WorkerOwnershipError("worker ownership event lookup index is unavailable")


def _read_counters(connection: sqlite3.Connection) -> tuple[int, int]:
    rows = connection.execute(
        "SELECT instance_count,event_count FROM store_counters WHERE singleton=1 LIMIT 2"
    ).fetchall()
    if len(rows) != 1:
        raise WorkerOwnershipError("worker ownership counters are unavailable")
    instance_count, event_count = rows[0]
    if (type(instance_count) is not int or type(event_count) is not int
            or instance_count < 0 or event_count < 0
            or instance_count > MAX_INSTANCES or event_count > MAX_EVENTS
            or event_count > instance_count * 3):
        raise WorkerOwnershipError("worker ownership database exceeds safety bound")
    return instance_count, event_count


@contextmanager
def _write_transaction(run_dir, context) -> Iterator[sqlite3.Connection]:
    _verify_sqlite_identity(run_dir, context)
    connection, fd = _connect_existing(_store_path(run_dir))
    try:
        _verify_binding(connection, context)
        connection.execute("BEGIN IMMEDIATE")
        _reject_legacy_coexistence(_store_path(run_dir))
        yield connection
        connection.commit()
    except sqlite3.OperationalError as exc:
        connection.rollback()
        raise WorkerOwnershipError("worker ownership database lock timed out") from exc
    except BaseException:
        connection.rollback()
        raise
    finally:
        _close_write_connection(connection, fd)


def initialize_worker_store(
    run_dir: str | os.PathLike[str], *, context: RunOwnershipContext,
) -> None:
    """Create a new SQLite authority; never migrate or coexist with JSON."""
    _verify_sqlite_identity(run_dir, context)
    path = _store_path(run_dir)
    _reject_legacy_coexistence(path)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
        created_identity = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        uri = "file:" + urllib.parse.quote(str(path), safe="/") + "?mode=rw"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        _apply_sqlite_limits(connection)
        if _regular_identity(path) != created_identity:
            raise WorkerOwnershipError("worker ownership database changed during initialization")
        _configure(connection)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript("""
            CREATE TABLE ownership_meta(
              key TEXT PRIMARY KEY, value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE store_counters(
              singleton INTEGER PRIMARY KEY CHECK(singleton=1),
              instance_count INTEGER NOT NULL CHECK(instance_count>=0),
              event_count INTEGER NOT NULL CHECK(event_count>=0)
            );
            CREATE TABLE worker_instances(
              instance_id TEXT PRIMARY KEY,
              capability_hash TEXT NOT NULL,
              role TEXT NOT NULL,
              state TEXT NOT NULL CHECK(state IN ('STARTING','BOUND','EXITED')),
              pid INTEGER, starttime TEXT, boot_id TEXT,
              machine_id TEXT, pid_namespace TEXT,
              exit_evidence TEXT, observed_boot_id TEXT,
              starting_monotonic REAL NOT NULL,
              bound_monotonic REAL, exited_monotonic REAL
            ) WITHOUT ROWID;
            CREATE TABLE worker_events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              instance_id TEXT NOT NULL REFERENCES worker_instances(instance_id),
              event TEXT NOT NULL CHECK(event IN ('STARTING','BOUND','EXITED')),
              payload TEXT NOT NULL,
              UNIQUE(instance_id,event)
            );
            CREATE INDEX worker_state_idx ON worker_instances(state);
            CREATE UNIQUE INDEX worker_process_idx ON worker_instances(
              machine_id,pid_namespace,boot_id,pid,starttime
            ) WHERE pid IS NOT NULL;
            CREATE INDEX worker_event_instance_idx ON worker_events(instance_id,seq);
        """)
        with connection:
            connection.executemany(
                "INSERT INTO ownership_meta(key,value) VALUES(?,?)",
                (("schema", STORE_SCHEMA), ("run_id", context.run_id),
                 ("identity_sha256", context.identity_sha256)),
            )
            connection.execute("INSERT INTO store_counters VALUES(1,0,0)")
        connection.close()
        os.close(fd)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if "connection" in locals() and connection is not None:
            connection.close()
        if "fd" in locals():
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _event_payload(event: dict[str, Any], context: RunOwnershipContext) -> str:
    return json.dumps(
        {**event, "run_id": context.run_id,
         "identity_sha256": context.identity_sha256},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _decode_payload(payload: Any) -> dict[str, Any]:
    if type(payload) is not str or len(payload.encode("utf-8")) > 16 * 1024:
        raise WorkerOwnershipError("worker ownership event payload is invalid")
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate event key")
                result[key] = value
            return result
        event = json.loads(payload, object_pairs_hook=unique)
    except (TypeError, ValueError, UnicodeError):
        raise WorkerOwnershipError("worker ownership event payload is corrupt") from None
    if type(event) is not dict:
        raise WorkerOwnershipError("worker ownership event payload is invalid")
    return event


def start_worker_sqlite(run_dir, role: str, *, context: RunOwnershipContext) -> WorkerInstance:
    if type(role) is not str or not role or len(role.encode("utf-8")) > 1024:
        raise ValueError("worker role must be a bounded nonempty string")
    instance = WorkerInstance(uuid.uuid4().hex, uuid.uuid4().hex)
    now = time.monotonic()
    event = {
        "schema": "factor_engine.worker_ownership.v1", "event": "STARTING",
        "instance_id": instance.instance_id,
        "capability_hash": _capability_hash(instance.capability), "role": role,
        "recorded_monotonic": now,
    }
    with _write_transaction(run_dir, context) as connection:
        count = connection.execute(
            "SELECT instance_count FROM store_counters WHERE singleton=1"
        ).fetchone()[0]
        if count >= MAX_INSTANCES:
            raise WorkerOwnershipError("worker ownership instance safety bound exceeded")
        connection.execute(
            "INSERT INTO worker_instances(instance_id,capability_hash,role,state,starting_monotonic) VALUES(?,?,?,'STARTING',?)",
            (instance.instance_id, event["capability_hash"], role, now),
        )
        connection.execute(
            "INSERT INTO worker_events(instance_id,event,payload) VALUES(?,'STARTING',?)",
            (instance.instance_id, _event_payload(event, context)),
        )
        connection.execute(
            "UPDATE store_counters SET instance_count=instance_count+1,event_count=event_count+1 WHERE singleton=1"
        )
    return instance


def _instance_row(connection, instance, expected_state):
    row = connection.execute(
        "SELECT capability_hash,state FROM worker_instances WHERE instance_id=?",
        (instance.instance_id,),
    ).fetchone()
    if (row is None or row[0] != _capability_hash(instance.capability)
            or row[1] != expected_state):
        raise WorkerOwnershipError("worker instance capability or state mismatch")


def bind_worker_sqlite(run_dir, instance: WorkerInstance, pid: int, *, context) -> None:
    if type(pid) is not int or pid <= 0:
        raise ValueError("worker pid must be a positive integer")
    status, starttime = _process_starttime(pid)
    if status != "PRESENT" or starttime is None:
        raise WorkerOwnershipError("worker process identity cannot be bound")
    machine, namespace = _host_identity()
    boot = _boot_id()
    now = time.monotonic()
    event = {
        "schema": "factor_engine.worker_ownership.v1", "event": "BOUND",
        "instance_id": instance.instance_id, "pid": pid, "starttime": starttime,
        "boot_id": boot, "machine_id": machine, "pid_namespace": namespace,
        "recorded_monotonic": now,
    }
    with _write_transaction(run_dir, context) as connection:
        _instance_row(connection, instance, "STARTING")
        duplicate = connection.execute(
            "SELECT 1 FROM worker_instances WHERE machine_id=? AND pid_namespace=? AND boot_id=? AND pid=? AND starttime=?",
            (machine, namespace, boot, pid, starttime),
        ).fetchone()
        if duplicate:
            raise WorkerOwnershipError("duplicate worker process identity")
        connection.execute(
            "UPDATE worker_instances SET state='BOUND',pid=?,starttime=?,boot_id=?,machine_id=?,pid_namespace=?,bound_monotonic=? WHERE instance_id=?",
            (pid, starttime, boot, machine, namespace, now, instance.instance_id),
        )
        connection.execute(
            "INSERT INTO worker_events(instance_id,event,payload) VALUES(?,'BOUND',?)",
            (instance.instance_id, _event_payload(event, context)),
        )
        connection.execute(
            "UPDATE store_counters SET event_count=event_count+1 WHERE singleton=1"
        )


def mark_worker_exited_sqlite(run_dir, instance: WorkerInstance, *, context) -> None:
    with _write_transaction(run_dir, context) as connection:
        _instance_row(connection, instance, "BOUND")
        row = connection.execute(
            "SELECT pid,starttime,boot_id,machine_id,pid_namespace,bound_monotonic FROM worker_instances WHERE instance_id=?",
            (instance.instance_id,),
        ).fetchone()
        pid, starttime, boot, machine, namespace, bound_time = row
        current_machine, current_namespace = _host_identity()
        current_boot = _boot_id()
        if (machine, namespace) != (current_machine, current_namespace):
            raise WorkerOwnershipError("worker belongs to another host or PID namespace")
        if boot != current_boot:
            evidence = "BOOT_CHANGED"
        else:
            status, observed = _process_starttime(pid)
            if status == "UNKNOWN":
                raise WorkerOwnershipError("worker exit identity is unknown")
            if status == "PRESENT" and observed == starttime:
                raise WorkerOwnershipError("worker process is still alive")
            evidence = "PID_ABSENT" if status == "ABSENT" else "PID_REUSED"
        now = time.monotonic()
        event = {
            "schema": "factor_engine.worker_ownership.v1", "event": "EXITED",
            "instance_id": instance.instance_id, "pid": pid, "starttime": starttime,
            "boot_id": boot, "exit_evidence": evidence, "machine_id": machine,
            "pid_namespace": namespace, "observed_boot_id": current_boot,
            "recorded_monotonic": now,
        }
        connection.execute(
            "UPDATE worker_instances SET state='EXITED',exit_evidence=?,observed_boot_id=?,exited_monotonic=? WHERE instance_id=?",
            (evidence, current_boot, now, instance.instance_id),
        )
        connection.execute(
            "INSERT INTO worker_events(instance_id,event,payload) VALUES(?,'EXITED',?)",
            (instance.instance_id, _event_payload(event, context)),
        )
        connection.execute(
            "UPDATE store_counters SET event_count=event_count+1 WHERE singleton=1"
        )


def read_worker_event_snapshot_sqlite(run_dir, *, context) -> Iterator[dict[str, Any]]:
    """Yield canonical events under cooperative descriptor-lock protection."""
    _verify_sqlite_identity(run_dir, context)
    connection, fd = _readonly_snapshot_connection(_store_path(run_dir))
    try:
        connection.execute("BEGIN")
        _verify_binding(connection, context)
        _verify_schema_indexes(connection)
        instance_count, event_count = _read_counters(connection)
        actual_instances = connection.execute("SELECT count(*) FROM worker_instances").fetchone()[0]
        actual_events = connection.execute("SELECT count(*) FROM worker_events").fetchone()[0]
        if (instance_count, event_count) != (actual_instances, actual_events):
            raise WorkerOwnershipError("worker ownership counters differ from stored rows")
        cursor = connection.execute("SELECT payload FROM worker_events ORDER BY seq")
        seen = 0
        while True:
            rows = cursor.fetchmany(_INSPECTION_PAGE_SIZE)
            if not rows:
                break
            for (payload,) in rows:
                seen += 1
                try:
                    event = _decode_payload(payload)
                except (TypeError, ValueError, UnicodeError):
                    raise WorkerOwnershipError("worker ownership event payload is corrupt") from None
                if (type(event) is not dict
                        or event.get("run_id") != context.run_id
                        or event.get("identity_sha256") != context.identity_sha256):
                    raise WorkerOwnershipError("worker ownership event binding mismatch")
                yield {key: value for key, value in event.items()
                       if key not in {"run_id", "identity_sha256"}}
        if seen != event_count:
            raise WorkerOwnershipError("worker ownership event count changed during read")
    except sqlite3.Error as exc:
        raise WorkerOwnershipError("worker ownership database read failed") from exc
    finally:
        connection.close()
        os.close(fd)


def validate_all_workers_exited_sqlite(
    run_dir, *, context: RunOwnershipContext,
) -> SQLiteWorkerOwnershipValidation:
    """Read-only strict validation through the shared canonical event authority."""
    try:
        _verify_sqlite_identity(run_dir, context)
        connection, fd = _readonly_snapshot_connection(_store_path(run_dir))
        try:
            connection.execute("BEGIN")
            _verify_binding(connection, context)
            _verify_schema_indexes(connection)
            instance_count, event_count = _read_counters(connection)
            if instance_count == 0:
                return SQLiteWorkerOwnershipValidation(
                    False, ("worker ownership snapshot has no instances",), 0
                )
            actual = connection.execute(
                "SELECT (SELECT count(*) FROM worker_instances),(SELECT count(*) FROM worker_events)"
            ).fetchone()
            if actual != (instance_count, event_count):
                raise WorkerOwnershipError("worker ownership counters differ from stored rows")
            duplicate = connection.execute(
                "SELECT 1 FROM worker_instances WHERE pid IS NOT NULL GROUP BY machine_id,pid_namespace,boot_id,pid,starttime HAVING count(*)>1 LIMIT 1"
            ).fetchone()
            if duplicate:
                raise WorkerOwnershipError("duplicate worker process identity")
            cursor = connection.execute(
                "SELECT instance_id,state,capability_hash,role,pid,starttime,boot_id,machine_id,pid_namespace,exit_evidence,observed_boot_id FROM worker_instances ORDER BY instance_id"
            )
            seen = 0
            while True:
                rows = cursor.fetchmany(512)
                if not rows:
                    break
                identifiers = [row[0] for row in rows]
                placeholders = ",".join("?" for _ in identifiers)
                event_rows = connection.execute(
                    f"SELECT payload FROM worker_events WHERE instance_id IN ({placeholders}) ORDER BY instance_id,seq LIMIT ?",
                    (*identifiers, len(identifiers) * 3 + 1),
                ).fetchall()
                if len(event_rows) > len(identifiers) * 3:
                    raise WorkerOwnershipError("worker ownership event cardinality exceeds safety bound")
                events = []
                for (payload,) in event_rows:
                    event = _decode_payload(payload)
                    if (event.get("run_id") != context.run_id
                            or event.get("identity_sha256") != context.identity_sha256):
                        raise WorkerOwnershipError("worker ownership event binding mismatch")
                    events.append({key: value for key, value in event.items()
                                   if key not in {"run_id", "identity_sha256"}})
                validation = _validate_bound_event_snapshot(events)
                if not validation.all_exited:
                    return SQLiteWorkerOwnershipValidation(
                        False, validation.reasons, instance_count
                    )
                by_instance: dict[str, list[dict[str, Any]]] = {}
                for event in events:
                    by_instance.setdefault(event["instance_id"], []).append(event)
                for row in rows:
                    (instance_id, state, capability_hash, role, pid, starttime, boot,
                     machine, namespace, evidence, observed_boot) = row
                    history = by_instance.get(instance_id)
                    if history is None:
                        raise WorkerOwnershipError("worker ownership snapshot history mismatch")
                    starting, bound, exited = history
                    if (
                        state != "EXITED" or capability_hash != starting["capability_hash"]
                        or role != starting["role"] or pid != bound["pid"]
                        or starttime != bound["starttime"] or boot != bound["boot_id"]
                        or machine != bound["machine_id"]
                        or namespace != bound["pid_namespace"]
                        or evidence != exited["exit_evidence"]
                        or observed_boot != exited["observed_boot_id"]
                    ):
                        raise WorkerOwnershipError("worker ownership snapshot differs from events")
                seen += len(rows)
            if seen != instance_count:
                raise WorkerOwnershipError("worker ownership snapshot row count mismatch")
        finally:
            connection.close()
            os.close(fd)
        return SQLiteWorkerOwnershipValidation(True, (), instance_count)
    except (WorkerOwnershipError, sqlite3.Error) as exc:
        return SQLiteWorkerOwnershipValidation(False, (str(exc),), 0)


def inspect_worker_retirement_now_sqlite(
    run_dir, *, context: RunOwnershipContext,
) -> SQLiteWorkerOwnershipInspection:
    """Read current OS retirement evidence without completing SQLite history."""
    examples: list[str] = []
    reasons: list[str] = []
    truncated = False
    journal_complete = True
    all_retired = True
    connection = None
    fd = None
    try:
        _verify_sqlite_identity(run_dir, context)
        connection, fd = _readonly_snapshot_connection(_store_path(run_dir))
        connection.execute("BEGIN")
        _verify_binding(connection, context)
        _verify_schema_indexes(connection)
        instance_count, event_count = _read_counters(connection)
        if instance_count == 0:
            return SQLiteWorkerOwnershipInspection(
                False, False, ("worker ownership snapshot has no instances",),
                0, (), False,
            )
        actual = connection.execute(
            "SELECT (SELECT count(*) FROM worker_instances),"
            "(SELECT count(*) FROM worker_events)"
        ).fetchone()
        if actual != (instance_count, event_count):
            raise WorkerOwnershipError(
                "worker ownership counters differ from stored rows"
            )
        duplicate = connection.execute(
            "SELECT 1 FROM worker_instances WHERE pid IS NOT NULL "
            "GROUP BY machine_id,pid_namespace,boot_id,pid,starttime "
            "HAVING count(*)>1 LIMIT 1"
        ).fetchone()
        if duplicate:
            raise WorkerOwnershipError("duplicate worker process identity")
        current_machine, current_namespace = _host_identity()
        current_boot = _boot_id()
        cursor = connection.execute(
            "SELECT instance_id,state,capability_hash,role,pid,starttime,"
            "boot_id,machine_id,pid_namespace,exit_evidence,observed_boot_id,"
            "starting_monotonic,bound_monotonic,exited_monotonic "
            "FROM worker_instances ORDER BY instance_id"
        )
        seen = events_seen = 0
        while True:
            rows = cursor.fetchmany(512)
            if not rows:
                break
            identifiers = [row[0] for row in rows]
            remaining = _INSPECTION_EXAMPLE_LIMIT - len(examples)
            if remaining > 0:
                examples.extend(identifiers[:remaining])
            if len(identifiers) > remaining:
                truncated = True
            placeholders = ",".join("?" for _ in identifiers)
            event_rows = connection.execute(
                f"SELECT instance_id,payload FROM worker_events "
                f"WHERE instance_id IN ({placeholders}) "
                "ORDER BY instance_id,seq LIMIT ?",
                (*identifiers, len(identifiers) * 3 + 1),
            ).fetchall()
            if len(event_rows) > len(identifiers) * 3:
                raise WorkerOwnershipError(
                    "worker ownership event cardinality exceeds safety bound"
                )
            events_seen += len(event_rows)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for stored_instance, payload in event_rows:
                event = _decode_payload(payload)
                if (event.get("run_id") != context.run_id
                        or event.get("identity_sha256")
                        != context.identity_sha256):
                    raise WorkerOwnershipError(
                        "worker ownership event binding mismatch"
                    )
                event = {key: value for key, value in event.items()
                         if key not in {"run_id", "identity_sha256"}}
                instance_id = event.get("instance_id")
                kind = event.get("event")
                if (instance_id != stored_instance
                        or type(instance_id) is not str
                        or len(instance_id) != 32
                        or any(char not in "0123456789abcdef"
                               for char in instance_id)
                        or event.get("schema")
                        != "factor_engine.worker_ownership.v1"
                        or type(kind) is not str
                        or kind not in _EXPECTED_EVENT_KEYS
                        or set(event) != _EXPECTED_EVENT_KEYS[kind]
                        or not _valid_recorded_monotonic(
                            event.get("recorded_monotonic"))):
                    raise WorkerOwnershipError("invalid worker instance identity")
                grouped.setdefault(instance_id, []).append(event)
            for row in rows:
                (instance_id, state, capability_hash, role, pid, starttime,
                 boot, machine, namespace, exit_evidence, observed_boot,
                 starting_time, bound_time, exited_time) = row
                history = grouped.get(instance_id)
                if history is None:
                    raise WorkerOwnershipError(
                        "worker ownership snapshot history mismatch"
                    )
                transitions = [event["event"] for event in history]
                if transitions not in (
                    ["STARTING"], ["STARTING", "BOUND"],
                    ["STARTING", "BOUND", "EXITED"],
                ):
                    raise WorkerOwnershipError(
                        f"worker {instance_id} has invalid or duplicate transitions"
                    )
                starting = history[0]
                if (type(starting.get("role")) is not str
                        or not starting["role"]
                        or type(starting.get("capability_hash")) is not str
                        or len(starting["capability_hash"]) != 64
                        or any(char not in "0123456789abcdef"
                               for char in starting["capability_hash"])
                        or capability_hash != starting["capability_hash"]
                        or role != starting["role"]
                        or starting_time != starting["recorded_monotonic"]):
                    raise WorkerOwnershipError(
                        f"worker {instance_id} has invalid STARTING authority"
                    )
                if transitions == ["STARTING"]:
                    if (state != "STARTING" or any(value is not None for value in (
                        pid, starttime, boot, machine, namespace, exit_evidence,
                        observed_boot, bound_time, exited_time,
                    ))):
                        raise WorkerOwnershipError(
                            "worker ownership snapshot differs from events"
                        )
                    journal_complete = all_retired = False
                    if len(reasons) < _INSPECTION_REASON_LIMIT:
                        reasons.append(
                            f"worker {instance_id} history is not closed; "
                            "current process identity is unknown"
                        )
                    else:
                        truncated = True
                    continue
                bound = history[1]
                if (type(bound.get("pid")) is not int or bound["pid"] <= 0
                        or type(bound.get("starttime")) is not str
                        or not bound["starttime"].isdigit()
                        or any(type(bound.get(key)) is not str or not bound[key]
                               for key in ("boot_id", "machine_id", "pid_namespace"))
                        or starting_time > bound["recorded_monotonic"]
                        or (pid, starttime, boot, machine, namespace, bound_time)
                        != (bound["pid"], bound["starttime"], bound["boot_id"],
                            bound["machine_id"], bound["pid_namespace"],
                            bound["recorded_monotonic"])):
                    raise WorkerOwnershipError(
                        f"worker {instance_id} has invalid bound identity"
                    )
                if (machine, namespace) != (current_machine, current_namespace):
                    retired = False
                    evidence = "belongs to another host or PID namespace"
                elif boot != current_boot:
                    retired = True
                    evidence = "is retired after an OS boot change"
                else:
                    status, observed = _process_starttime(pid)
                    retired = (
                        (status == "ABSENT" and observed is None)
                        or (status == "PRESENT"
                            and type(observed) is str and observed.isdigit()
                            and observed != starttime)
                    )
                    evidence = (
                        "is retired by current OS identity evidence" if retired
                        else "cannot be proven retired by the current OS"
                    )
                all_retired = all_retired and retired
                if transitions == ["STARTING", "BOUND"]:
                    if (state != "BOUND" or exit_evidence is not None
                            or observed_boot is not None or exited_time is not None):
                        raise WorkerOwnershipError(
                            "worker ownership snapshot differs from events"
                        )
                    journal_complete = False
                    if len(reasons) < _INSPECTION_REASON_LIMIT:
                        reasons.append(
                            f"worker {instance_id} history is not closed; "
                            f"current process identity {evidence}"
                        )
                    else:
                        truncated = True
                    continue
                exited = history[2]
                if (state != "EXITED"
                        or exited.get("schema")
                        != "factor_engine.worker_ownership.v1"
                        or type(exited.get("pid")) is not int
                        or exited["pid"] <= 0
                        or type(exited.get("starttime")) is not str
                        or not exited["starttime"].isdigit()
                        or any(exited.get(key) != bound.get(key) for key in (
                            "machine_id", "pid_namespace", "boot_id", "pid", "starttime"
                        ))
                        or any(type(exited.get(key)) is not str or not exited[key]
                               for key in ("observed_boot_id", "exit_evidence"))
                        or exited["exit_evidence"] not in {
                            "PID_ABSENT", "PID_REUSED", "BOOT_CHANGED"
                        }
                        or ((exited["observed_boot_id"] != bound["boot_id"])
                            != (exited["exit_evidence"] == "BOOT_CHANGED"))
                        or (exited["exit_evidence"] != "BOOT_CHANGED"
                            and bound_time > exited["recorded_monotonic"])
                        or (exit_evidence, observed_boot, exited_time)
                        != (exited["exit_evidence"], exited["observed_boot_id"],
                            exited["recorded_monotonic"])):
                    raise WorkerOwnershipError(
                        f"worker {instance_id} has invalid EXITED evidence"
                    )
                if not retired:
                    journal_complete = False
                    if len(reasons) < _INSPECTION_REASON_LIMIT:
                        reasons.append(
                            f"worker {instance_id} complete history conflicts with "
                            f"current OS evidence: {evidence}"
                        )
                    else:
                        truncated = True
            seen += len(rows)
        if seen != instance_count or events_seen != event_count:
            raise WorkerOwnershipError(
                "worker ownership snapshot row count mismatch"
            )
    except (WorkerOwnershipError, sqlite3.Error) as exc:
        return SQLiteWorkerOwnershipInspection(
            False, False, (str(exc),), 0, (), False
        )
    finally:
        if connection is not None:
            connection.close()
        if fd is not None:
            os.close(fd)
    return SQLiteWorkerOwnershipInspection(
        journal_complete, all_retired, tuple(reasons), instance_count,
        tuple(examples), truncated,
    )
