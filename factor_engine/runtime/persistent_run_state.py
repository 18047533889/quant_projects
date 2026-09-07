"""Durable ordinal outcomes and a shared, bounded retry budget."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TERMINAL_STATES = frozenset({
    "SUCCEEDED", "REUSED", "REJECTED", "FAILED", "BLOCKED", "CANCELLED"
})


@dataclass(frozen=True)
class OrdinalOutcome:
    ordinal: int
    name: str
    state: str
    attempts: int
    error_code: str | None
    retryable: bool
    commit_state: str = "NOT_STARTED"
    error_detail: str | None = None


class PersistentRunState:
    """SQLite state is the authority shared by source/executor/sink retries."""

    def __init__(self, path: str | Path, *, max_attempts: int = 3) -> None:
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be an integer in [1, 3]")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._lock = threading.RLock()
        self.max_attempts = max_attempts
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS outcomes(
              ordinal INTEGER PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0, error_code TEXT,
              error_detail TEXT, retryable INTEGER NOT NULL DEFAULT 0,
              generation INTEGER NOT NULL DEFAULT 1,
              artifact_json TEXT, commit_state TEXT NOT NULL DEFAULT 'NOT_STARTED');
            CREATE TABLE IF NOT EXISTS events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, ordinal INTEGER,
              stage TEXT NOT NULL, event TEXT NOT NULL, detail TEXT);
            CREATE TABLE IF NOT EXISTS state_policy(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(outcomes)")}
        if "artifact_generation" not in columns:
            self._db.execute("ALTER TABLE outcomes ADD COLUMN artifact_generation TEXT")
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO state_policy VALUES('max_attempts',?)",
                             (str(max_attempts),))
        stored = self._db.execute("SELECT value FROM state_policy WHERE key='max_attempts'").fetchone()[0]
        if stored != str(max_attempts):
            self._db.close()
            raise ValueError("persisted retry policy differs; reopening cannot reset the attempt budget")

    @staticmethod
    def _validate_ordinal(ordinal: int) -> None:
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")

    def register(self, ordinal: int, name: str) -> None:
        self._validate_ordinal(ordinal)
        if not isinstance(name, str) or not name:
            raise ValueError("factor name must be a nonempty string")
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO outcomes(ordinal,name,state) VALUES(?,?,'ACCEPTED')",
                (ordinal, name),
            )
            stored = self._db.execute("SELECT name FROM outcomes WHERE ordinal=?", (ordinal,)).fetchone()[0]
            if stored != name:
                raise ValueError("ordinal is already bound to a different factor name")

    def consume_attempt(self, ordinal: int, stage: str) -> int | None:
        """Atomically consume one of at most three attempts across every layer."""
        self._validate_ordinal(ordinal)
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT attempts,state FROM outcomes WHERE ordinal=?", (ordinal,)
            ).fetchone()
            if row is None:
                raise KeyError(ordinal)
            attempts, state = int(row[0]), row[1]
            if state in TERMINAL_STATES or attempts >= self.max_attempts:
                return None
            attempts += 1
            self._db.execute(
                "UPDATE outcomes SET attempts=?,state='RUNNING' WHERE ordinal=?",
                (attempts, ordinal),
            )
            self._db.execute(
                "INSERT INTO events(ordinal,stage,event,detail) VALUES(?,?,?,?)",
                (ordinal, stage, "ATTEMPT", str(attempts)),
            )
            return attempts

    def terminal(self, ordinal: int, state: str, *, error_code: str | None = None,
                 detail: str | None = None, retryable: bool = False,
                 artifact: dict[str, Any] | None = None,
                 commit_state: str = "NOT_STARTED") -> None:
        self._validate_ordinal(ordinal)
        if state not in TERMINAL_STATES:
            raise ValueError(f"not a terminal state: {state}")
        if state in {"SUCCEEDED", "REUSED"} and (
                not isinstance(artifact, dict) or artifact.get("committed") is not True or
                artifact.get("verified") is not True or commit_state != "VERIFIED"):
            raise ValueError("successful ordinal requires a verified committed artifact")
        with self._lock, self._db:
            current = self._db.execute(
                "SELECT state,artifact_generation FROM outcomes WHERE ordinal=?", (ordinal,)
            ).fetchone()
            if current is None:
                raise KeyError(ordinal)
            if current[0] in TERMINAL_STATES:
                raise RuntimeError(f"ordinal {ordinal} is already terminal: {current[0]}")
            if (state in {"SUCCEEDED", "REUSED"} and current[1] is not None
                    and artifact.get("generation") != current[1]):
                raise ValueError("artifact generation differs from persisted commit intent")
            self._db.execute(
                "UPDATE outcomes SET state=?,error_code=?,error_detail=?,retryable=?,artifact_json=?,commit_state=? "
                "WHERE ordinal=?",
                (state, error_code, detail, int(retryable),
                 json.dumps(artifact, sort_keys=True) if artifact is not None else None,
                 commit_state, ordinal),
            )

    def mark_unknown_commit(self, ordinal: int, detail: str) -> None:
        """Persist uncertainty; callers must reconcile, never blindly rewrite."""
        self.terminal(ordinal, "FAILED", error_code="UNKNOWN_COMMIT", detail=detail,
                      commit_state="UNKNOWN")

    def record_commit_intent(self, ordinal: int, generation: str) -> None:
        """Durably bind exactly one output generation before starting its writer."""
        self._validate_ordinal(ordinal)
        if not isinstance(generation, str) or re.fullmatch(r"[0-9a-f]{32}", generation) is None:
            raise ValueError("invalid artifact generation")
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT state,artifact_generation FROM outcomes WHERE ordinal=?", (ordinal,)).fetchone()
            if row is None:
                raise KeyError(ordinal)
            if row[0] in TERMINAL_STATES:
                raise RuntimeError("cannot change a terminal ordinal commit intent")
            if row[1] is not None:
                if row[1] != generation:
                    raise RuntimeError("commit intent already binds a different generation")
                return
            self._db.execute(
                "UPDATE outcomes SET artifact_generation=?,commit_state='INTENT' WHERE ordinal=?",
                (generation, ordinal))
            self._db.execute(
                "INSERT INTO events(ordinal,stage,event,detail) VALUES(?,'sink','COMMIT_INTENT',?)",
                (ordinal, generation))

    def get_commit_intent(self, ordinal: int) -> dict[str, str] | None:
        self._validate_ordinal(ordinal)
        with self._lock:
            row = self._db.execute(
                "SELECT artifact_generation,commit_state FROM outcomes WHERE ordinal=?", (ordinal,)).fetchone()
            if row is None:
                raise KeyError(ordinal)
            return None if row[0] is None else {"generation": row[0], "commit_state": row[1]}

    def outcomes(self) -> list[OrdinalOutcome]:
        rows = self._db.execute(
            "SELECT ordinal,name,state,attempts,error_code,retryable,commit_state,error_detail "
            "FROM outcomes ORDER BY ordinal"
        )
        return [OrdinalOutcome(r[0], r[1], r[2], r[3], r[4], bool(r[5]), r[6], r[7]) for r in rows]

    def outcomes_page(self, *, start: int = 0, limit: int = 512) -> list[OrdinalOutcome]:
        """Read one bounded ordinal page; the durable index remains authoritative."""
        self._validate_ordinal(start)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("outcome page limit must be an integer in [1, 1000]")
        with self._lock:
            rows = self._db.execute(
                "SELECT ordinal,name,state,attempts,error_code,retryable,commit_state,error_detail "
                "FROM outcomes WHERE ordinal>=? ORDER BY ordinal LIMIT ?", (start, limit))
            return [OrdinalOutcome(r[0], r[1], r[2], r[3], r[4], bool(r[5]), r[6], r[7]) for r in rows]

    def counts(self) -> dict[str, int]:
        return {state: count for state, count in self._db.execute(
            "SELECT state,COUNT(*) FROM outcomes GROUP BY state"
        )}

    def close(self) -> None:
        self._db.close()
