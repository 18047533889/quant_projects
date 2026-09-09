"""Disk-backed, finite factor manifests for bounded batch execution."""
from __future__ import annotations

import hashlib
import math
import os
import pickle
import sqlite3
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from factor_engine.runtime.manifest_metadata_limits import (
    MAX_DEPENDENCIES_PER_FACTOR,
    MAX_DEPENDENCY_NAME_UTF8,
    MAX_ERROR_CODE_CHARS,
    MAX_FACTOR_NAME_UTF8,
    MAX_MANIFEST_ROW_METADATA_BYTES,
    bounded_utf8_size,
)


class ManifestError(ValueError):
    """Base class for finite-manifest validation failures."""


class ManifestInputTimeout(ManifestError, TimeoutError):
    reason_code = "INPUT_INGESTION_TIMEOUT"


class ManifestLimitExceeded(ManifestError):
    pass


class ManifestNameConflict(ManifestError):
    pass


class _PickleLimitExceeded(Exception):
    pass


class _BoundedPickleWriter:
    """Collect at most limit bytes plus one byte of oversize evidence."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.payload = bytearray()

    def write(self, chunk: bytes) -> int:
        remaining = self.limit + 1 - len(self.payload)
        if remaining > 0:
            self.payload.extend(chunk[:remaining])
        if len(chunk) > remaining or len(self.payload) > self.limit:
            raise _PickleLimitExceeded
        return len(chunk)


def _bounded_pickle(value: Any, limit: int) -> tuple[bytes | None, int, bool]:
    """Serialize through a capped sink; an oversize size is only a lower bound."""
    writer = _BoundedPickleWriter(limit)
    try:
        pickle.Pickler(writer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    except _PickleLimitExceeded:
        return None, limit + 1, True
    return bytes(writer.payload), len(writer.payload), False


_PICKLE_FRAME_ALLOWANCE = 64 * 1024


def controlled_serialization_buffer_bytes(max_definition_bytes: int) -> int:
    """Reservation for capped output, its final copy, and one pickle frame.

    This deliberately excludes the inherited definition, pickle memo, COW,
    and arbitrary allocations or side effects performed by ``__reduce__``.
    """
    return 2 * (max_definition_bytes + 1) + _PICKLE_FRAME_ALLOWANCE


@dataclass(frozen=True)
class ManifestRecord:
    ordinal: int
    name: str
    definition_bytes: int
    definition_digest: str
    valid: bool
    error_code: str | None


class FiniteFactorManifest:
    """SQLite manifest retaining definitions and indexes outside Python RAM.

    The input is consumed once under count/definition/deadline bounds. Duplicate
    names are rejected even when their definitions are identical: v2 outcomes
    are ordinal-scoped and must never silently overwrite a same-name result.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS factors(
              ordinal INTEGER PRIMARY KEY, name TEXT NOT NULL,
              definition BLOB, definition_bytes INTEGER NOT NULL,
              definition_digest TEXT NOT NULL, valid INTEGER NOT NULL,
              error_code TEXT);
            CREATE INDEX IF NOT EXISTS factor_name_idx ON factors(name, ordinal);
            CREATE TABLE IF NOT EXISTS dependencies(
              ordinal INTEGER NOT NULL, dependency_name TEXT NOT NULL,
              PRIMARY KEY(ordinal, dependency_name));
            CREATE INDEX IF NOT EXISTS dependency_name_idx
              ON dependencies(dependency_name, ordinal);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        self.input_complete = self._meta("input_complete", "1") == "1"
        self.input_error = self._meta("input_error", "") or None

    def _meta(self, key: str, default: str = "") -> str:
        row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def _set_read_length_limit(self, maximum: int) -> int:
        """Apply a native SQLite row/cell bound before reading untrusted data."""
        try:
            previous = self._db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, maximum)
            if self._db.getlimit(sqlite3.SQLITE_LIMIT_LENGTH) > maximum:
                raise ManifestError("SQLite manifest length bound is unavailable")
            return previous
        except (AttributeError, OverflowError, sqlite3.Error) as exc:
            raise ManifestError("SQLite manifest length bound is unavailable") from exc

    @classmethod
    def ingest_supervised(cls, factors: Iterable[Any], path: str | Path, *,
                          process_context: str = "fork",
                          serialization_broker: Any = None,
                          deadline_monotonic: float | None = None,
                          ownership_run_dir: str | Path | None = None,
                          ownership_context: Any = None,
                          ownership_role: str = "manifest-ingestion",
                          **limits: Any) -> "FiniteFactorManifest":
        """Ingest in a killable process so a blocking ``__next__`` is finite."""
        deadline = limits.get("deadline_seconds", 300.0)
        if (type(deadline) not in (int, float) or isinstance(deadline, bool)
                or not math.isfinite(deadline)
                or deadline <= 0):
            raise ValueError("input ingestion deadline must be positive and finite")
        factor_limit = limits.get("max_factors", 1_000_000)
        if type(factor_limit) is not int or factor_limit < 1:
            raise ValueError("max_factors must be a positive integer")
        if deadline_monotonic is None:
            absolute_deadline = time.monotonic() + float(deadline)
        elif (type(deadline_monotonic) not in (int, float)
                or not math.isfinite(deadline_monotonic)):
            raise ValueError("input ingestion absolute deadline must be finite")
        else:
            absolute_deadline = float(deadline_monotonic)
        definition_limit = limits.get("max_definition_bytes", 131_072)
        if type(definition_limit) is not int or definition_limit < 1:
            raise ValueError("max_definition_bytes must be a positive integer")
        context = mp.get_context(process_context)
        parent, child = context.Pipe(duplex=False)
        ownership_enabled = ownership_run_dir is not None or ownership_context is not None
        if (ownership_run_dir is None) != (ownership_context is None):
            parent.close()
            child.close()
            raise ValueError("ingestion ownership run directory and context must be paired")
        gate_child = gate_parent = None
        ownership_instance = None
        ownership_bound = ownership_exited = False
        retain_authority = False
        serialization_lease = None
        if ownership_enabled:
            try:
                gate_child, gate_parent = context.Pipe(duplex=False)
            except BaseException:
                parent.close()
                child.close()
                raise
        process = None
        primary = None

        def quarantine(message, *, cause=None):
            nonlocal retain_authority
            from factor_engine.runtime.supervised_worker import WorkerQuarantined
            retain_authority = True
            error = WorkerQuarantined(
                message, pid=getattr(process, "pid", None),
            )
            error.ingestion_process = process
            error.ingestion_parent = parent
            error.ingestion_child = child
            error.ownership_gate_parent = gate_parent
            error.ownership_gate_child = gate_child
            error.ownership_instance = ownership_instance
            error.ownership_context = ownership_context
            error.serialization_buffer_lease = serialization_lease
            if cause is None:
                raise error
            raise error from cause

        def record_exit():
            nonlocal ownership_exited
            if ownership_bound and not ownership_exited:
                from factor_engine.runtime.worker_ownership import mark_worker_exited
                try:
                    mark_worker_exited(
                        ownership_run_dir, ownership_instance,
                        context=ownership_context,
                    )
                except BaseException as exc:
                    quarantine("input ingestion exit journal could not be completed", cause=exc)
                ownership_exited = True

        def retire():
            if process is None or process.pid is None:
                return
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(1.0)
                if process.is_alive():
                    process.kill()
                    process.join(1.0)
            except BaseException as exc:
                quarantine(
                    "input ingestion worker retirement failed", cause=exc,
                )
            if process.is_alive() or process.exitcode is None:
                quarantine(
                    "input ingestion worker exit could not be observed",
                )
            record_exit()

        def incomplete(detail):
            # This is legal only after the writer's OS exit was observed.
            manifest = cls(path)
            with manifest._db:
                count = manifest._db.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('count',?)", (str(count),))
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('input_complete','0')")
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('input_error',?)", (detail,))
            manifest.input_complete, manifest.input_error = False, detail
            return manifest

        try:
            if serialization_broker is not None:
                from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
                requested_bytes = controlled_serialization_buffer_bytes(
                    definition_limit
                )
                manifest_scope = hashlib.sha256(
                    os.fsencode(Path(path).resolve(strict=False))
                ).hexdigest()
                serialization_lease = serialization_broker.acquire_memory(
                    MemoryLeaseKind.MANIFEST_BUFFER,
                    requested_bytes,
                    lease_id=f"manifest-ingestion:{manifest_scope}",
                )
                if serialization_lease is None:
                    raise ManifestError(
                        "controlled serialization buffer admission was denied"
                    )
            remaining = absolute_deadline - time.monotonic()
            if remaining <= 0:
                raise ManifestInputTimeout("factor input ingestion deadline exceeded")
            limits = dict(limits)
            limits["deadline_seconds"] = remaining
            if ownership_enabled:
                from factor_engine.runtime.worker_ownership import start_worker
                ownership_instance = start_worker(
                    ownership_run_dir, ownership_role, context=ownership_context,
                )
            process = context.Process(
                target=_ingest_child,
                args=(
                    factors, str(path), limits, child, gate_child,
                    ownership_instance.capability if ownership_instance is not None else None,
                ),
                daemon=True,
            )
            try:
                process.start()
            except BaseException as exc:
                if getattr(process, "_popen", None) is None or process.pid is None:
                    quarantine(
                        "input ingestion start crossed OS-spawn boundary; exit is unknown",
                        cause=exc,
                    )
                retire()
                return incomplete(
                    f"input ingestion worker could not start: {type(exc).__name__}: {str(exc)[:4096]}"
                )
            if ownership_enabled:
                from factor_engine.runtime.worker_ownership import bind_worker
                try:
                    bind_worker(
                        ownership_run_dir, ownership_instance, process.pid,
                        context=ownership_context,
                    )
                    ownership_bound = True
                except BaseException as exc:
                    try:
                        retire()
                    except BaseException as retire_exc:
                        exc.cleanup_errors = [retire_exc]
                    quarantine("input ingestion ownership binding failed", cause=exc)
            from factor_engine.runtime.resource_broker import (
                NoActiveHeavyRunGuard, register_heavy_worker,
            )
            try:
                register_heavy_worker(process.pid)
            except NoActiveHeavyRunGuard:
                pass
            if ownership_enabled:
                try:
                    gate_parent.send_bytes(
                        b"BOUND:" + ownership_instance.capability.encode("ascii")
                    )
                except BaseException as exc:
                    retire()
                    raise
                gate_parent.close()
                gate_parent = None
            child.close()
            process.join(max(0.0, absolute_deadline - time.monotonic()))
            if process.is_alive():
                retire()
                return incomplete("factor input ingestion deadline exceeded")
            record_exit()
            if not parent.poll():
                raise ManifestError(f"manifest ingestion worker exited with code {process.exitcode}")
            try:
                ok, kind, detail = parent.recv()
            except EOFError as exc:
                raise ManifestError("manifest ingestion worker closed without a result") from exc
            return cls(path) if ok else incomplete(detail)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup_primary = None
            try:
                if retain_authority and not ownership_enabled:
                    # Preserve the established legacy endpoint-close contract;
                    # unknown process authority and its buffer lease remain.
                    parent.close()
                    child.close()
                elif not retain_authority:
                    try:
                        retire()
                    finally:
                        if not retain_authority:
                            parent.close()
                            child.close()
                            if gate_parent is not None:
                                gate_parent.close()
                            if gate_child is not None:
                                gate_child.close()
                            if process is not None and process.exitcode is not None:
                                process.close()
            except BaseException as exc:
                cleanup_primary = exc
            if not retain_authority and serialization_lease is not None:
                try:
                    serialization_lease.release()
                except BaseException as release_exc:
                    retain_authority = True
                    release_exc.cleanup_pending = True
                    release_exc.serialization_buffer_lease = serialization_lease
                    if primary is None and cleanup_primary is None:
                        raise
                    target = primary if primary is not None else cleanup_primary
                    target.cleanup_pending = True
                    target.serialization_buffer_lease = serialization_lease
                    target.cleanup_errors = list(
                        getattr(target, "cleanup_errors", [])
                    ) + ([cleanup_primary] if primary is not None and cleanup_primary is not None else []) + [release_exc]
            if cleanup_primary is not None:
                if primary is None:
                    raise cleanup_primary
                if cleanup_primary not in getattr(primary, "cleanup_errors", []):
                    primary.cleanup_errors = list(
                        getattr(primary, "cleanup_errors", [])
                    ) + [cleanup_primary]
                if getattr(cleanup_primary, "cleanup_pending", False):
                    primary.cleanup_pending = True
                    if hasattr(cleanup_primary, "serialization_buffer_lease"):
                        primary.serialization_buffer_lease = cleanup_primary.serialization_buffer_lease

    @classmethod
    def ingest(
        cls, factors: Iterable[Any], path: str | Path, *,
        max_factors: int = 1_000_000, max_definition_bytes: int = 131_072,
        deadline_seconds: float = 300.0,
    ) -> "FiniteFactorManifest":
        if type(max_factors) is not int or max_factors < 1:
            raise ValueError("max_factors must be a positive integer")
        if type(max_definition_bytes) is not int or max_definition_bytes < 1:
            raise ValueError("max_definition_bytes must be a positive integer")
        if (type(deadline_seconds) not in (int, float)
                or isinstance(deadline_seconds, bool)
                or not math.isfinite(deadline_seconds)
                or deadline_seconds <= 0):
            raise ValueError("deadline_seconds must be positive and finite")
        manifest = cls(path)
        started = time.monotonic()
        iterator = None
        primary = None
        try:
            iterator = iter(factors)
            with manifest._db:
                manifest._db.execute("DELETE FROM dependencies")
                manifest._db.execute("DELETE FROM factors")
                manifest._db.execute("DELETE FROM meta")
                for ordinal, factor in enumerate(iterator):
                    if time.monotonic() - started > deadline_seconds:
                        raise ManifestInputTimeout("factor input ingestion deadline exceeded")
                    if ordinal >= max_factors:
                        raise ManifestLimitExceeded(f"factor count exceeds {max_factors}")
                    valid, error = True, None
                    try:
                        name = getattr(factor, "name", None)
                        if name is None and isinstance(factor, dict):
                            name = factor.get("name")
                    except Exception:
                        name, valid, error = (
                            f"<invalid:{ordinal}>", False, "INVALID_FACTOR_NAME"
                        )
                    if valid and (not isinstance(name, str) or not name):
                        name, valid, error = f"<invalid:{ordinal}>", False, "INVALID_FACTOR_NAME"
                    elif valid and bounded_utf8_size(name, MAX_FACTOR_NAME_UTF8) is None:
                        name, valid, error = (
                            f"<invalid:{ordinal}>", False, "FACTOR_NAME_TOO_LARGE"
                        )
                    payload, size, oversize = None, 0, False
                    if valid:
                        try:
                            payload, size, oversize = _bounded_pickle(
                                factor, max_definition_bytes
                            )
                        except Exception:
                            valid, error = False, "INVALID_FACTOR_DEFINITION"
                        if oversize:
                            # Exact total size is intentionally not computed: size is
                            # the observed lower bound proving the configured limit.
                            valid, error = False, "DEFINITION_TOO_LARGE"
                    dependency_names = []
                    if valid:
                        dependency_iterator = None
                        dependency_timeout = None
                        try:
                            deps = getattr(factor, "dependencies", ()) or ()
                            dependency_iterator = iter(deps)
                            for dependency_index in range(
                                MAX_DEPENDENCIES_PER_FACTOR + 1
                            ):
                                if time.monotonic() - started > deadline_seconds:
                                    raise ManifestInputTimeout(
                                        "factor input ingestion deadline exceeded"
                                    )
                                try:
                                    dep = next(dependency_iterator)
                                except StopIteration:
                                    break
                                if dependency_index >= MAX_DEPENDENCIES_PER_FACTOR:
                                    valid, error = False, "TOO_MANY_DEPENDENCIES"
                                    dependency_names = []
                                    break
                                dep_name = getattr(dep, "name", dep)
                                if isinstance(dep_name, str) and dep_name:
                                    if bounded_utf8_size(
                                        dep_name, MAX_DEPENDENCY_NAME_UTF8
                                    ) is None:
                                        valid, error = (
                                            False, "DEPENDENCY_NAME_TOO_LARGE"
                                        )
                                        dependency_names = []
                                        break
                                    dependency_names.append(dep_name)
                        except ManifestInputTimeout as exc:
                            dependency_timeout = exc
                        except Exception:
                            valid, error = False, "INVALID_FACTOR_DEPENDENCIES"
                            dependency_names = []
                        finally:
                            try:
                                close_dependencies = getattr(
                                    dependency_iterator, "close", None
                                )
                                if callable(close_dependencies):
                                    close_dependencies()
                            except Exception:
                                if dependency_timeout is None:
                                    valid, error = (
                                        False, "INVALID_FACTOR_DEPENDENCIES"
                                    )
                                    dependency_names = []
                        if dependency_timeout is not None:
                            raise dependency_timeout
                        if not valid:
                            payload, size = None, 0
                    if manifest._db.execute(
                        "SELECT 1 FROM factors WHERE name=? LIMIT 1", (name,)
                    ).fetchone() is not None:
                        valid, error = False, "DUPLICATE_FACTOR_NAME"
                        manifest._db.execute(
                            "UPDATE factors SET valid=0,error_code='DUPLICATE_FACTOR_NAME' WHERE name=?",
                            (name,),
                        )
                    digest = hashlib.sha256(payload).hexdigest() if payload is not None else ""
                    manifest._db.execute(
                        "INSERT INTO factors VALUES(?,?,?,?,?,?,?)",
                        (ordinal, name, payload, size, digest, int(valid), error),
                    )
                    # Each accepted ordinal is durable independently so a later
                    # blocked/invalid iterator can still produce a partial receipt.
                    manifest._db.commit()
                    if valid:
                        manifest._db.executemany(
                            "INSERT OR IGNORE INTO dependencies VALUES(?,?)",
                            ((ordinal, dep_name) for dep_name in dependency_names),
                        )
                count = manifest._db.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
                manifest._db.execute("INSERT INTO meta VALUES('count', ?)", (str(count),))
                manifest._db.execute("INSERT INTO meta VALUES('sealed', '1')")
                manifest._db.execute("INSERT INTO meta VALUES('input_complete', '1')")
        except BaseException as exc:
            primary = exc
            try:
                manifest.close()
            except BaseException as cleanup_exc:
                exc.cleanup_errors = list(getattr(exc, "cleanup_errors", [])) + [cleanup_exc]
            raise
        finally:
            try:
                close = getattr(iterator, "close", None) if iterator is not None else None
                if callable(close):
                    close()
            except BaseException as cleanup_exc:
                if primary is not None:
                    primary.cleanup_errors = list(
                        getattr(primary, "cleanup_errors", [])
                    ) + [cleanup_exc]
                else:
                    try:
                        manifest.close()
                    except BaseException as manifest_cleanup_exc:
                        cleanup_exc.cleanup_errors = list(
                            getattr(cleanup_exc, "cleanup_errors", [])
                        ) + [manifest_cleanup_exc]
                    raise
        return manifest

    def __len__(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key='count'").fetchone()
        return int(row[0]) if row else 0

    def records(self, *, start: int = 0, limit: int = 512) -> Iterator[ManifestRecord]:
        try:
            rows = self._db.execute(
                "SELECT ordinal,CASE WHEN length(CAST(name AS BLOB))<=? THEN name END,definition_bytes,"
                "CASE WHEN length(CAST(definition_digest AS BLOB))<=64 THEN definition_digest END,valid,"
                "CASE WHEN error_code IS NULL OR length(CAST(error_code AS BLOB))<=? THEN error_code END "
                "FROM factors WHERE ordinal>=? ORDER BY ordinal LIMIT ?",
                (MAX_FACTOR_NAME_UTF8, MAX_ERROR_CODE_CHARS, start, limit),
            )
            for row in rows:
                if (bounded_utf8_size(row[1], MAX_FACTOR_NAME_UTF8) is None
                        or row[3] is None
                        or (row[5] is None and not bool(row[4]))
                        or (row[5] is not None and bounded_utf8_size(
                            row[5], MAX_ERROR_CODE_CHARS
                        ) is None)):
                    raise ManifestError("manifest record metadata exceeds safety bound")
                yield ManifestRecord(
                    row[0], row[1], row[2], row[3], bool(row[4]), row[5]
                )
        except sqlite3.Error as exc:
            # This projection bounds values crossing into Python. It cannot use
            # SQLITE_LIMIT_LENGTH because that limit also charges an unselected,
            # legitimately large definition BLOB stored in the same table row.
            raise ManifestError("manifest record metadata read failed") from exc

    def load_wave(self, *, start: int, limit: int) -> list[tuple[int, Any]]:
        rows = self._db.execute(
            "SELECT ordinal,definition,valid,"
            "CASE WHEN error_code IS NULL OR length(CAST(error_code AS BLOB))<=? THEN error_code END,"
            "CASE WHEN length(CAST(name AS BLOB))<=? THEN name END FROM factors WHERE ordinal>=? "
            "ORDER BY ordinal LIMIT ?",
            (MAX_ERROR_CODE_CHARS, MAX_FACTOR_NAME_UTF8, start, limit),
        )
        output = []
        for ordinal, payload, valid, error, name in rows:
            if (bounded_utf8_size(name, MAX_FACTOR_NAME_UTF8) is None
                    or (error is None and not bool(valid))
                    or (error is not None and bounded_utf8_size(
                        error, MAX_ERROR_CODE_CHARS
                    ) is None)):
                raise ManifestError("manifest load metadata exceeds safety bound")
            output.append((int(ordinal), pickle.loads(payload) if valid else None, error, name))
        return output

    def load_wave_bounded(self, *, start: int, max_items: int,
                          max_definition_bytes: int) -> list[tuple[int, Any, str | None, str]]:
        """Load definitions under the original byte budget plus fixed metadata.

        Callers can reserve ``controlled_wave_buffer_bytes(max_items, budget)``;
        metadata does not silently reduce the established definition budget.
        """
        previous = self._set_read_length_limit(
            max_definition_bytes + MAX_MANIFEST_ROW_METADATA_BYTES + 4096
        )
        try:
            rows = self._db.execute(
                "SELECT ordinal,substr(definition,1,?),valid,"
                "CASE WHEN error_code IS NULL OR length(CAST(error_code AS BLOB))<=? THEN error_code END,"
                "CASE WHEN length(CAST(name AS BLOB))<=? THEN name END,definition_bytes FROM factors "
                "WHERE ordinal>=? ORDER BY ordinal LIMIT ?",
                (max_definition_bytes + 1, MAX_ERROR_CODE_CHARS,
                 MAX_FACTOR_NAME_UTF8, start, max_items),
            )
            output, charged = [], 0
            for ordinal, payload, valid, error, name, size in rows:
                if (bounded_utf8_size(name, MAX_FACTOR_NAME_UTF8) is None
                        or type(size) is not int or size < 0
                        or (error is None and not bool(valid))
                        or (error is not None and bounded_utf8_size(
                            error, MAX_ERROR_CODE_CHARS
                        ) is None)
                        or (valid and (payload is None or len(payload) != size
                                       or size > max_definition_bytes))):
                    raise ManifestError("manifest wave row exceeds safety bound")
                if output and charged + size > max_definition_bytes:
                    break
                output.append((
                    int(ordinal), pickle.loads(payload) if valid else None,
                    error, name,
                ))
                charged += size
            return output
        except sqlite3.Error as exc:
            raise ManifestError("manifest wave exceeds native SQLite bound") from exc
        finally:
            self._db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, previous)

    def dependent_ordinals(self, name: str) -> tuple[int, ...]:
        if bounded_utf8_size(name, MAX_DEPENDENCY_NAME_UTF8) is None:
            raise ManifestError("dependency name exceeds safety bound")
        return tuple(row[0] for row in self._db.execute(
            "SELECT ordinal FROM dependencies WHERE dependency_name=? ORDER BY ordinal", (name,)
        ))

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "FiniteFactorManifest":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _ingest_child(factors: Iterable[Any], path: str, limits: dict[str, Any], connection: Any,
                  ownership_gate: Any = None, ownership_capability: str | None = None) -> None:
    if ownership_gate is not None:
        try:
            permitted = ownership_gate.recv_bytes(128)
            if permitted != b"BOUND:" + ownership_capability.encode("ascii"):
                raise RuntimeError("invalid ingestion ownership admission")
        except BaseException:
            connection.close()
            raise
        finally:
            ownership_gate.close()
    try:
        manifest = FiniteFactorManifest.ingest(factors, path, **limits)
        manifest.close()
        connection.send((True, "", ""))
    except Exception as exc:
        # Parent joins before receiving: keep the control response below pipe
        # capacity so an enormous exception cannot deadlock the child send.
        detail = str(exc).encode("utf-8", errors="replace")[:2048].decode("utf-8", errors="ignore")
        connection.send((False, type(exc).__name__[:128], detail))
    finally:
        connection.close()
