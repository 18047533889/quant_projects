"""Disk-backed, finite factor manifests for bounded batch execution."""
from __future__ import annotations

import hashlib
import pickle
import sqlite3
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


class ManifestError(ValueError):
    """Base class for finite-manifest validation failures."""


class ManifestInputTimeout(ManifestError):
    pass


class ManifestLimitExceeded(ManifestError):
    pass


class ManifestNameConflict(ManifestError):
    pass


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

    @classmethod
    def ingest_supervised(cls, factors: Iterable[Any], path: str | Path, *,
                          process_context: str = "fork", **limits: Any) -> "FiniteFactorManifest":
        """Ingest in a killable process so a blocking ``__next__`` is finite."""
        deadline = float(limits.get("deadline_seconds", 300.0))
        context = mp.get_context(process_context)
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_ingest_child, args=(factors, str(path), limits, child), daemon=True
        )
        try:
            process.start()
        except Exception as exc:
            manifest = cls(path)
            detail = f"input ingestion worker could not start: {type(exc).__name__}: {exc}"
            with manifest._db:
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('count','0')")
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('input_complete','0')")
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('input_error',?)", (detail,))
            manifest.input_complete, manifest.input_error = False, detail
            return manifest
        try:
            from factor_engine.runtime.resource_broker import (
                NoActiveHeavyRunGuard, register_heavy_worker,
            )
            register_heavy_worker(process.pid)
        except NoActiveHeavyRunGuard:
            pass
        child.close()
        process.join(deadline)
        if process.is_alive():
            process.terminate()
            process.join(10.0)
            if process.is_alive():
                detail = "input iterator blocked; worker cleanup pending"
            else:
                detail = "factor input ingestion deadline exceeded"
            manifest = cls(path)
            with manifest._db:
                count = manifest._db.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('count',?)", (str(count),))
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('input_complete','0')")
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('input_error',?)", (detail,))
            manifest.input_complete, manifest.input_error = False, detail
            return manifest
        if not parent.poll():
            raise ManifestError(f"manifest ingestion worker exited with code {process.exitcode}")
        ok, kind, detail = parent.recv()
        parent.close()
        manifest = cls(path)
        if not ok:
            with manifest._db:
                count = manifest._db.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('count',?)", (str(count),))
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('input_complete','0')")
                manifest._db.execute("INSERT OR REPLACE INTO meta VALUES('input_error',?)", (detail,))
            manifest.input_complete, manifest.input_error = False, detail
        return manifest

    @classmethod
    def ingest(
        cls, factors: Iterable[Any], path: str | Path, *,
        max_factors: int = 1_000_000, max_definition_bytes: int = 131_072,
        deadline_seconds: float = 300.0,
    ) -> "FiniteFactorManifest":
        if isinstance(max_factors, bool) or max_factors < 1:
            raise ValueError("max_factors must be a positive integer")
        if isinstance(max_definition_bytes, bool) or max_definition_bytes < 1:
            raise ValueError("max_definition_bytes must be a positive integer")
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        manifest = cls(path)
        started = time.monotonic()
        iterator = iter(factors)
        try:
            with manifest._db:
                manifest._db.execute("DELETE FROM dependencies")
                manifest._db.execute("DELETE FROM factors")
                manifest._db.execute("DELETE FROM meta")
                for ordinal, factor in enumerate(iterator):
                    if time.monotonic() - started > deadline_seconds:
                        raise ManifestInputTimeout("factor input ingestion deadline exceeded")
                    if ordinal >= max_factors:
                        raise ManifestLimitExceeded(f"factor count exceeds {max_factors}")
                    name = getattr(factor, "name", None)
                    if name is None and isinstance(factor, dict):
                        name = factor.get("name")
                    valid, error = True, None
                    if not isinstance(name, str) or not name:
                        name, valid, error = f"<invalid:{ordinal}>", False, "INVALID_FACTOR_NAME"
                    try:
                        payload = pickle.dumps(factor, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception:
                        payload, valid, error = None, False, "INVALID_FACTOR_DEFINITION"
                    size = len(payload) if payload is not None else 0
                    if size > max_definition_bytes:
                        payload, size, valid, error = None, size, False, "DEFINITION_TOO_LARGE"
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
                    deps = getattr(factor, "dependencies", ()) or ()
                    for dep in deps:
                        dep_name = getattr(dep, "name", dep)
                        if isinstance(dep_name, str) and dep_name:
                            manifest._db.execute(
                                "INSERT OR IGNORE INTO dependencies VALUES(?,?)",
                                (ordinal, dep_name),
                            )
                count = manifest._db.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
                manifest._db.execute("INSERT INTO meta VALUES('count', ?)", (str(count),))
                manifest._db.execute("INSERT INTO meta VALUES('sealed', '1')")
                manifest._db.execute("INSERT INTO meta VALUES('input_complete', '1')")
        except BaseException:
            manifest.close()
            raise
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        return manifest

    def __len__(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key='count'").fetchone()
        return int(row[0]) if row else 0

    def records(self, *, start: int = 0, limit: int = 512) -> Iterator[ManifestRecord]:
        rows = self._db.execute(
            "SELECT ordinal,name,definition_bytes,definition_digest,valid,error_code FROM factors "
            "WHERE ordinal>=? ORDER BY ordinal LIMIT ?", (start, limit),
        )
        for row in rows:
            yield ManifestRecord(row[0], row[1], row[2], row[3], bool(row[4]), row[5])

    def load_wave(self, *, start: int, limit: int) -> list[tuple[int, Any]]:
        rows = self._db.execute(
            "SELECT ordinal,definition,valid,error_code,name FROM factors WHERE ordinal>=? "
            "ORDER BY ordinal LIMIT ?", (start, limit),
        )
        return [
            (int(ordinal), pickle.loads(payload) if valid else None, error, name)
            for ordinal, payload, valid, error, name in rows
        ]

    def load_wave_bounded(self, *, start: int, max_items: int,
                          max_definition_bytes: int) -> list[tuple[int, Any, str | None, str]]:
        """Load a byte-bounded lookahead; always admits one stored definition."""
        rows = self._db.execute(
            "SELECT ordinal,definition,valid,error_code,name,definition_bytes FROM factors "
            "WHERE ordinal>=? ORDER BY ordinal LIMIT ?", (start, max_items),
        )
        output, charged = [], 0
        for ordinal, payload, valid, error, name, size in rows:
            if output and charged + int(size) > max_definition_bytes:
                break
            output.append((int(ordinal), pickle.loads(payload) if valid else None, error, name))
            charged += int(size)
        return output

    def dependent_ordinals(self, name: str) -> tuple[int, ...]:
        return tuple(row[0] for row in self._db.execute(
            "SELECT ordinal FROM dependencies WHERE dependency_name=? ORDER BY ordinal", (name,)
        ))

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "FiniteFactorManifest":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _ingest_child(factors: Iterable[Any], path: str, limits: dict[str, Any], connection: Any) -> None:
    try:
        manifest = FiniteFactorManifest.ingest(factors, path, **limits)
        manifest.close()
        connection.send((True, "", ""))
    except Exception as exc:
        connection.send((False, type(exc).__name__, str(exc)))
    finally:
        connection.close()
