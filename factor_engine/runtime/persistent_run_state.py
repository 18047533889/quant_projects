"""Durable ordinal outcomes and a shared, bounded retry budget."""
from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
            CREATE TABLE IF NOT EXISTS fit_failure_evidence(
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              evidence_id TEXT NOT NULL UNIQUE,
              scope TEXT NOT NULL,
              assignment_count INTEGER NOT NULL,
              assignments_sha256 TEXT NOT NULL,
              availability TEXT NOT NULL,
              payload_json TEXT,
              evidence_sha256 TEXT NOT NULL,
              payload_bytes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS fit_failure_evidence_assignments(
              evidence_id TEXT NOT NULL,
              ordinal INTEGER NOT NULL,
              PRIMARY KEY(evidence_id,ordinal));
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

    def _register_in_transaction(self, ordinal: int, name: str) -> None:
        self._validate_ordinal(ordinal)
        if not isinstance(name, str) or not name:
            raise ValueError("factor name must be a nonempty string")
        self._db.execute(
            "INSERT OR IGNORE INTO outcomes(ordinal,name,state) VALUES(?,?,'ACCEPTED')",
            (ordinal, name),
        )
        stored = self._db.execute("SELECT name FROM outcomes WHERE ordinal=?", (ordinal,)).fetchone()[0]
        if stored != name:
            raise ValueError("ordinal is already bound to a different factor name")

    def register(self, ordinal: int, name: str) -> None:
        with self._lock, self._db:
            self._register_in_transaction(ordinal, name)

    def register_many(self, records, *, batch_size: int = 512) -> None:
        """Stream bounded transactions without collecting all input records.

        Earlier completed pages remain durable if a later page fails. A failed
        page rolls back atomically. Existing names, attempts and outcomes retain
        the exact same protections as single-record registration.
        """
        if type(batch_size) is not int or not 1 <= batch_size <= 512:
            raise ValueError("registration batch size must be an integer in [1, 512]")
        iterator = iter(records)
        while True:
            with self._lock, self._db:
                for _ in range(batch_size):
                    try:
                        ordinal, name = next(iterator)
                    except StopIteration:
                        return
                    self._register_in_transaction(ordinal, name)

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

    def record_fit_failure_evidence(
        self, evidence_id: str, assignments: Iterable[dict[str, Any]],
        snapshot: dict[str, Any] | None, *, availability: str = "OBSERVED",
    ) -> int:
        """Persist one bounded wave sample idempotently before terminal outcomes."""
        marker = self._db.execute(
            "SELECT value FROM state_policy WHERE key='fit_failure_evidence_schema'"
        ).fetchone()
        if marker != ("v1",):
            raise RuntimeError("fit failure evidence schema is not enabled for this run")
        if (type(evidence_id) is not str
                or re.fullmatch(r"[0-9a-f]{32}", evidence_id) is None):
            raise ValueError("invalid fit failure evidence id")
        if availability not in {"OBSERVED", "UNAVAILABLE"}:
            raise ValueError("invalid fit failure evidence availability")
        try:
            assignment_iterator = iter(assignments)
        except TypeError:
            raise ValueError("invalid fit failure evidence assignments") from None
        if availability == "OBSERVED":
            if type(snapshot) is not dict:
                raise ValueError("observed fit failure evidence requires a snapshot")
            from factor_engine.runtime.fit_failure_evidence import (
                encode_fit_failure_snapshot, validate_fit_failure_snapshot,
            )
            snapshot = validate_fit_failure_snapshot(snapshot)
            payload = encode_fit_failure_snapshot(snapshot)
            payload_json = payload.decode("utf-8")
            payload_bytes = len(payload)
        else:
            if snapshot is not None:
                raise ValueError("unavailable fit failure evidence cannot contain counts")
            payload_json = None
            payload_bytes = 0
        with self._lock, self._db:
            assignment_digest = hashlib.sha256(
                b"factor_engine.fit_failure_assignments.v1\0")
            assignment_count = 0
            previous_ordinal = -1
            for item in assignment_iterator:
                if (type(item) is not dict or set(item) != {"ordinal", "name"}
                        or type(item["ordinal"]) is not int
                        or not 0 <= item["ordinal"] <= 9223372036854775807
                        or type(item["name"]) is not str or not item["name"]):
                    raise ValueError("invalid fit failure evidence assignment")
                ordinal, name = item["ordinal"], item["name"]
                if ordinal <= previous_ordinal:
                    raise ValueError("fit failure evidence ordinals must be unique and ordered")
                previous_ordinal = ordinal
                if self._db.execute(
                    "SELECT name FROM outcomes WHERE ordinal=?", (ordinal,)
                ).fetchone() != (name,):
                    raise ValueError("fit failure evidence assignment is not registered")
                encoded_name = name.encode("utf-8")
                assignment_digest.update(ordinal.to_bytes(8, "big"))
                assignment_digest.update(len(encoded_name).to_bytes(8, "big"))
                assignment_digest.update(encoded_name)
                self._db.execute(
                    "INSERT OR IGNORE INTO fit_failure_evidence_assignments"
                    "(evidence_id,ordinal) VALUES(?,?)", (evidence_id, ordinal))
                assignment_count += 1
            if assignment_count == 0:
                raise ValueError("fit failure evidence assignments are empty")
            assignments_sha256 = assignment_digest.hexdigest()
            evidence_digest_payload = json.dumps({
                "evidence_id": evidence_id, "scope": "wave_sample",
                "assignment_count": assignment_count,
                "assignments_sha256": assignments_sha256,
                "availability": availability, "snapshot": snapshot,
            }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            evidence_sha256 = hashlib.sha256(evidence_digest_payload).hexdigest()
            row = (evidence_id, "wave_sample", assignment_count, assignments_sha256,
                   availability, payload_json, evidence_sha256, payload_bytes)
            self._db.execute(
                "INSERT OR IGNORE INTO fit_failure_evidence"
                "(evidence_id,scope,assignment_count,assignments_sha256,availability,"
                "payload_json,evidence_sha256,payload_bytes) VALUES(?,?,?,?,?,?,?,?)", row)
            stored = self._db.execute(
                "SELECT evidence_id,scope,assignment_count,assignments_sha256,"
                "availability,payload_json,"
                "evidence_sha256,payload_bytes FROM fit_failure_evidence WHERE evidence_id=?",
                (evidence_id,),
            ).fetchone()
            if stored != row:
                raise ValueError("fit failure evidence id conflicts with persisted evidence")
            linked = self._db.execute(
                "SELECT COUNT(*) FROM fit_failure_evidence_assignments WHERE evidence_id=?",
                (evidence_id,),
            ).fetchone()[0]
            if linked != assignment_count:
                raise ValueError("fit failure evidence assignments conflict")
            return int(self._db.execute(
                "SELECT seq FROM fit_failure_evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()[0])

    def enable_fit_failure_evidence(self) -> None:
        """Mark a newly-created run as requiring the v1 evidence table forever."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO state_policy VALUES('fit_failure_evidence_schema','v1')"
            )
            if self._db.execute(
                "SELECT value FROM state_policy WHERE key='fit_failure_evidence_schema'"
            ).fetchone() != ("v1",):
                raise RuntimeError("persisted fit failure evidence schema differs")

    def fit_failure_evidence_page(self, *, after_seq: int = 0, limit: int = 100):
        if type(after_seq) is not int or after_seq < 0:
            raise ValueError("after_seq must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 512:
            raise ValueError("evidence page limit must be in [1, 512]")
        page = []
        used_bytes = 0
        maximum_page_bytes = 1024 * 1024
        with self._lock:
            rows = self._db.execute(
                "SELECT seq,evidence_id,scope,assignment_count,assignments_sha256,"
                "availability,evidence_sha256,payload_bytes,"
                "length(CAST(payload_json AS BLOB)) FROM fit_failure_evidence "
                "WHERE seq>? ORDER BY seq LIMIT ?", (after_seq, limit)
            )
            for row in rows:
                stored_bytes = row[8]
                if row[5] == "OBSERVED":
                    if (type(stored_bytes) is not int or not 1 <= stored_bytes <= 262144
                            or row[7] != stored_bytes):
                        raise ValueError("persisted fit failure evidence payload is invalid")
                    charge = stored_bytes + 512
                elif row[5] == "UNAVAILABLE":
                    if stored_bytes is not None or row[7] != 0:
                        raise ValueError("unavailable fit failure evidence contains payload")
                    charge = 512
                else:
                    raise ValueError("persisted fit failure evidence availability is invalid")
                if page and used_bytes + charge > maximum_page_bytes:
                    break
                payload_json = self._db.execute(
                    "SELECT payload_json FROM fit_failure_evidence WHERE seq=?", (row[0],)
                ).fetchone()[0]
                page.append(dict(
                    seq=row[0], evidence_id=row[1], scope=row[2],
                    assignment_count=row[3], assignments_sha256=row[4],
                    availability=row[5],
                    snapshot=None if payload_json is None else json.loads(payload_json),
                    evidence_sha256=row[6], payload_bytes=row[7],
                ))
                used_bytes += charge
        return page

    def fit_failure_evidence_summary(self) -> dict[str, int]:
        with self._lock:
            counts = dict(self._db.execute(
                "SELECT availability,COUNT(*) FROM fit_failure_evidence GROUP BY availability"
            ))
            truncated = self._db.execute(
                "SELECT COUNT(*) FROM fit_failure_evidence WHERE availability='OBSERVED' "
                "AND json_extract(payload_json,'$.truncated')=1"
            ).fetchone()
            last = self._db.execute("SELECT COALESCE(MAX(seq),0) FROM fit_failure_evidence").fetchone()
        return {"observed_waves": int(counts.get("OBSERVED", 0)),
                "unavailable_waves": int(counts.get("UNAVAILABLE", 0)),
                "truncated_waves": int(truncated[0]),
                "wave_count": sum(int(value) for value in counts.values()),
                "last_seq": int(last[0])}

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
