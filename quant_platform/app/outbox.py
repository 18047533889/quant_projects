"""Transactional outbox/inbox — same-transaction write + background publish.

QRP-P1. Implements the transactional outbox pattern (spec §11.1): an event row
is inserted in the SAME transaction as the state update it announces, so the
platform never double-writes events and state. A background publisher claims
pending events with at-least-once delivery and marks them sent; an
inbox consumer dedupes by idempotency key and drives a durable state machine.

Outbox row lifecycle (R55 #91):
    pending -> claimed(worker_id, claim_token, claimed_at) -> sent
              `-> pending (delivery failure, retry_count++) -> dead_letter

``claim()`` is the concurrency guard: it atomically flips a *specific* pending
row to ``claimed`` (single ``UPDATE ... WHERE status='pending'`` — row-locked in
PostgreSQL, ``BEGIN IMMEDIATE`` serialized in SQLite), so two workers can never
hold the same claim. ``publish_pending`` claims each eligible row in its own
transaction, then publishes and marks ``sent``; a publish failure returns the
row to ``pending`` with ``retry_count`` incremented.

Inbox row lifecycle (R55 #90):
    received -> processing -> done         (handler succeeded)
                           `-> failed (retry_count++, next_attempt_at, last_error)
                                     `-> dead_letter (after max attempts)

The inbox ack is applied in the SAME transaction as the handler's DB side
effects: ``process`` runs ``handler`` and the ``UPDATE ... done`` inside one
``db.transaction()``, so a handler exception rolls the ack back and the message
is NEVER silently burned as "processed". It is instead recorded ``failed`` with
``retry_count``/``last_error`` and becomes retryable (or dead-lettered past the
max). ``process_many`` is the out-of-transaction convenience loop for message
broker consumption.

The actual message-bus publish is a pluggable ``Publisher`` Protocol. An
in-process ``InMemoryPublisher`` is provided for tests.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Publisher",
    "InMemoryPublisher",
    "Outbox",
    "OutboxClaimError",
    "Inbox",
    "InboxEventNotFound",
    "INBOX_RECEIVED",
    "INBOX_PROCESSING",
    "INBOX_DONE",
    "INBOX_FAILED",
    "INBOX_STATUSES",
    "INBOX_MAX_RETRIES",
]


@runtime_checkable
class Publisher(Protocol):
    """Message-bus publish seam (spec §11.1)."""

    def publish(self, event: dict[str, Any]) -> None:
        """Deliver a single outbox event to the bus."""
        ...


class InMemoryPublisher:
    """In-process publisher for tests; records delivered events."""

    def __init__(self) -> None:
        self.delivered: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.delivered.append(event)


class OutboxClaimError(RuntimeError):
    """Raised by claim operations when the targeted row is not claimable."""


class Outbox:
    """Transactional outbox writer + background claim/publish.

    ``db`` must implement the ``Db`` Protocol. ``publisher`` is a ``Publisher``.
    """

    def __init__(self, db: Any, publisher: Publisher) -> None:
        self._db = db
        self._publisher = publisher

    def emit(
        self,
        *,
        event_type: str,
        aggregate_type: str,
        aggregate_id: str,
        correlation_id: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        causation_id: str | None = None,
        trace_id: str | None = None,
        actor_principal_id: str | None = None,
    ) -> int:
        """Insert an outbox row in the caller's transaction.

        IMPORTANT: this method must be called INSIDE a ``db.transaction()`` block
        so the event row commits atomically with the state update it announces.
        Returns the new outbox row id.
        """
        payload_json = json.dumps(payload or {})
        # P0-PLAT-008: ``emit`` runs inside ``db.transaction()``.  SQLite's raw
        # connection cursor exposes ``lastrowid``; the PostgreSQL backend exposes
        # ``execute_returning`` (``RETURNING id``) for the same scalar.  Route
        # through ``db.execute_returning`` when available so the SAME insert
        # works on both backends.
        returning = getattr(self._db, "execute_returning", None)
        if returning is not None:
            row_id = returning(
                "INSERT INTO outbox_events "
                "(event_type, aggregate_type, aggregate_id, correlation_id, causation_id, "
                " trace_id, actor_principal_id, idempotency_key, payload_json, occurred_at, status, attempts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0) RETURNING id",
                (
                    event_type,
                    aggregate_type,
                    aggregate_id,
                    correlation_id,
                    causation_id,
                    trace_id,
                    actor_principal_id,
                    idempotency_key,
                    payload_json,
                    int(time.time()),
                ),
            )
            return int(row_id)
        cur = self._db.execute(
            "INSERT INTO outbox_events "
            "(event_type, aggregate_type, aggregate_id, correlation_id, causation_id, "
            " trace_id, actor_principal_id, idempotency_key, payload_json, occurred_at, status, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0)",
            (
                event_type,
                aggregate_type,
                aggregate_id,
                correlation_id,
                causation_id,
                trace_id,
                actor_principal_id,
                idempotency_key,
                payload_json,
                int(time.time()),
            ),
        )
        return int(cur.lastrowid)

    # ---- claim / state transition helpers (concurrency boundary) ----
    def claim(self, event_id: int, *, worker_id: str, claim_token: str, now: float) -> bool:
        """Atomically claim one specific pending row.

        Only an eligible pending row can be flipped to claimed. Reusing a worker
        ID does not permit stealing an existing claim. The guarded UPDATE is the
        row lock: PostgreSQL acquires a row-level lock per row (and a live PG
        backend should use ``SELECT ... FOR UPDATE SKIP LOCKED`` for the
        ``publish_pending`` scan — see its docstring), SQLite relies on the
        ``BEGIN IMMEDIATE`` serialization already in ``transaction()``.

        No races need to be handled in caller code: exactly one worker can ever
        observe this row as pending-and-claimable for itself.
        """
        cursor = self._db.execute(
            "UPDATE outbox_events SET status = 'claimed', worker_id = ?, "
            "claim_token = ?, claimed_at = ?, retry_count = retry_count + 1 "
            "WHERE id = ? AND status = 'pending' "
            "AND (next_attempt_at IS NULL OR next_attempt_at <= ?)",
            (worker_id, claim_token, now, event_id, now),
        )
        return cursor.rowcount == 1

    def finish(self, event_id: int, *, claim_token: str) -> bool:
        """Mark a claimed row ``sent``. No-op (False) unless the caller still
        holds the claim (guarded by claim_token)."""
        cursor = self._db.execute(
            "UPDATE outbox_events SET status = 'sent', claimed_at = NULL "
            "WHERE id = ? AND status = 'claimed' AND claim_token = ?",
            (event_id, claim_token),
        )
        return cursor.rowcount == 1

    def fail(
        self,
        event_id: int,
        *,
        claim_token: str,
        error: str,
        now: float,
        next_attempt_at: float,
        dead_letter: bool = False,
    ) -> None:
        """Return a claimed row to ``pending`` (or dead-letter it) with retry
        accounting. No-op unless the caller holds the claim_token."""
        if dead_letter:
            self._db.execute(
                "UPDATE outbox_events SET status = 'dead_letter', "
                "last_error = ?, dead_letter_reason = ?, claimed_at = NULL, "
                "next_attempt_at = NULL "
                "WHERE id = ? AND status = 'claimed' AND claim_token = ?",
                (error, error, event_id, claim_token),
            )
        else:
            self._db.execute(
                "UPDATE outbox_events SET status = 'pending', worker_id = NULL, "
                "claim_token = NULL, claimed_at = NULL, last_error = ?, "
                "next_attempt_at = ? "
                "WHERE id = ? AND status = 'claimed' AND claim_token = ?",
                (error, next_attempt_at, event_id, claim_token),
            )

    def recover_expired_claims(self, *, now: float, lease_seconds: float,
                               idempotency_key: str | None = None) -> int:
        """Fence expired acknowledgements and make interrupted sends retryable.

        The bus may already have accepted an expired send. Consumers therefore
        must deduplicate by the stable event idempotency key. Expiry cannot
        cancel a slow external sender or guarantee exactly-once network effects.
        """
        if not math.isfinite(now) or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("claim recovery requires finite time and positive lease_seconds")
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not idempotency_key):
            raise ValueError("idempotency_key must be a nonempty string")
        scope = " AND idempotency_key = ?" if idempotency_key is not None else ""
        params = (now, now - lease_seconds)
        if idempotency_key is not None:
            params += (idempotency_key,)
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE outbox_events SET status = 'pending', worker_id = NULL, "
                "claim_token = NULL, claimed_at = NULL, next_attempt_at = ?, "
                "last_error = 'claim lease expired; delivery outcome unknown' "
                "WHERE status = 'claimed' AND claimed_at IS NOT NULL AND claimed_at <= ?" + scope,
                params,
            )
            return cursor.rowcount

    def publish_pending(self, limit: int = 100, *, claim_lease_seconds: float = 300.0,
                        idempotency_key: str | None = None) -> int:
        """Claim + publish pending events and mark them sent.

        Each event is claimed in its own transaction (concurrency-safe across
        workers), delivered via the publisher, then marked ``sent`` in a second
        transaction. Failures retry after their backoff; expired claims recover
        on a later drain. External delivery is at-least-once, not exactly-once.

        PostgreSQL note: a live PG backend should select candidate rows with
        ``SELECT ... FOR UPDATE SKIP LOCKED`` (row-claim semantics) so that two
        workers scanning concurrently never both pick the same row. The claim
        here is row-lock-safe even without it — the guarded ``UPDATE``/verify
        round-trip flips only a ``pending`` row and the second worker's update
        simply matches 0 rows. The ``SKIP LOCKED`` optimization is documented in
        ``postgres_backend.py`` as the deployment-time tuning knob.

        An exact idempotency_key scopes both lease recovery and publication;
        shadow workflows must supply it to avoid touching unrelated events.
        Returns count published.
        """
        scan_time = float(time.time())
        self.recover_expired_claims(now=scan_time, lease_seconds=claim_lease_seconds,
                                    idempotency_key=idempotency_key)
        scope = " AND idempotency_key = ?" if idempotency_key is not None else ""
        params = (scan_time, limit) if idempotency_key is None else (scan_time, idempotency_key, limit)
        rows = self._db.query(
            "SELECT * FROM outbox_events WHERE status = 'pending' "
            "AND (next_attempt_at IS NULL OR next_attempt_at <= ?)" + scope + " ORDER BY id LIMIT ?",
            params,
        )
        published = 0
        for row in rows:
            event_id = int(row["id"])
            worker_id = f"w-{uuid.uuid4().hex[:8]}"
            claim_token = uuid.uuid4().hex
            now = float(time.time())
            claimed = False
            with self._db.transaction() as conn:
                # Single-statement atomic claim: the row is serialized by the
                # UPDATE's row lock (PG) / BEGIN IMMEDIATE (SQLite).
                conn.execute(
                    "UPDATE outbox_events SET status = 'claimed', worker_id = ?, "
                    "claim_token = ?, claimed_at = ?, retry_count = retry_count + 1 "
                    "WHERE id = ? AND status = 'pending' "
                    "AND (next_attempt_at IS NULL OR next_attempt_at <= ?)",
                    (worker_id, claim_token, now, event_id, now),
                )
                claimed = conn.execute(
                    "SELECT 1 FROM outbox_events WHERE id = ? AND status = 'claimed' "
                    "AND claim_token = ?",
                    (event_id, claim_token),
                ).fetchone() is not None
            if not claimed:
                continue  # another worker raced and won this row
            event = dict(row)
            event["payload"] = json.loads(row["payload_json"] or "{}")
            try:
                self._publisher.publish(event)
            except Exception as exc:
                with self._db.transaction() as conn:
                    conn.execute(
                        "UPDATE outbox_events SET status = 'pending', worker_id = NULL, "
                        "claim_token = NULL, claimed_at = NULL, last_error = ?, "
                        "attempts = attempts + 1, next_attempt_at = ? "
                        "WHERE id = ? AND status = 'claimed' AND claim_token = ?",
                        (str(exc)[:4000], now + 1.0, event_id, claim_token),
                    )
                continue
            with self._db.transaction() as conn:
                conn.execute(
                    "UPDATE outbox_events SET status = 'sent', claimed_at = NULL "
                    "WHERE id = ? AND status = 'claimed' AND claim_token = ?",
                    (event_id, claim_token),
                )
                sent = conn.execute(
                    "SELECT 1 FROM outbox_events WHERE id = ? AND status = 'sent' "
                    "AND claim_token = ?",
                    (event_id, claim_token),
                ).fetchone() is not None
            if sent:
                published += 1
        return published


class InboxEventNotFound(RuntimeError):
    """Raised when an inbox event is missing or already dead-lettered."""


INBOX_RECEIVED = "RECEIVED"
INBOX_PROCESSING = "PROCESSING"
INBOX_DONE = "DONE"
INBOX_FAILED = "FAILED"
INBOX_STATUSES = (INBOX_RECEIVED, INBOX_PROCESSING, INBOX_DONE, INBOX_FAILED)
INBOX_MAX_RETRIES = 3


class Inbox:
    """Idempotent event consumer with a durable four-state machine.

    ``db`` must implement the ``Db`` Protocol. ``handler`` is a callable
    ``(event: dict) -> None`` invoked once per unique idempotency key.

    State machine (spec §11.1 / R55 #90)::

        RECEIVED --process--> PROCESSING --handler OK--> DONE
                                    `----handler raises--> FAILED
                      FAILED --retry--> PROCESSING (retry_count++ / next_attempt_at)
                      past max retries --> dead_letter=True (FAILED + dead_letter)

    ``process`` runs the handler and the ``DONE`` ack in ONE transaction, so a
    handler exception rolls the ack back and the message is NEVER burned as
    processed: it is recorded ``FAILED`` with retry bookkeeping instead.
    """

    def __init__(
        self,
        db: Any,
        handler: Any,
        *,
        max_retries: int = INBOX_MAX_RETRIES,
        replica: bool = False,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._db = db
        self._handler = handler
        self.max_retries = max_retries
        self.replica = replica  # True: this consumer is a read-only snapshot

    # ---- queries ----
    def get(self, idempotency_key: str) -> dict[str, Any] | None:
        rows = self._db.query(
            "SELECT * FROM inbox_events WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        return rows[0] if rows else None

    def status_of(self, idempotency_key: str) -> str:
        row = self.get(idempotency_key)
        return row["status"] if row else "UNKNOWN"

    # ---- processing ----
    def mark_failed(
        self,
        event: dict[str, Any],
        *,
        error: str,
        now: float,
        dead_letter: bool = False,
    ) -> None:
        """Record a FAILED inbox row (retry_count++ / next_attempt_at / error).

        If ``dead_letter`` is True the row is terminal: it will not be retried
        and the consumer raises ``InboxEventNotFound`` on later attempts. The
        idempotency_key is kept so a re-delivery is still deduped — but it is
        surfaced as dead-lettered rather than silently reprocessed.
        """
        key = event.get("idempotency_key")
        prev = self.get(key) if key else None
        prev_count = int(prev["retry_count"]) if prev else 0
        next_attempt_at = None if dead_letter else now + 1.0
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE inbox_events SET status = 'FAILED', "
                "retry_count = ?, next_attempt_at = ?, last_error = ?, "
                "dead_letter = ?, dead_letter_reason = ?, "
                "processed_at = ? "
                "WHERE idempotency_key = ?",
                (
                    prev_count + 1,
                    next_attempt_at,
                    error[:4000],
                    bool(dead_letter),
                    error[:4000] if dead_letter else None,
                    now,
                    key,
                ),
            )

    def process(self, event: dict[str, Any]) -> bool:
        """Atomically claim, handle and acknowledge database-local work.

        The claim is never committed separately: a killed process rolls back
        both its claim and handler writes. A savepoint retains failure metadata
        without retaining failed handler writes. Handlers must use this DB
        connection and must not commit it or open a nested transaction. Network
        effects require their own idempotency key or a transactional outbox;
        this method does not promise exactly-once external delivery.

        Legacy durable PROCESSING rows are deliberately not reclaimed without
        operator evidence that their old worker is stopped.
        """
        key = event.get("idempotency_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("inbox event requires idempotency_key")
        if self.replica:
            # A read-only replica must never acquire a durable processing claim.
            row = self.get(key)
            if row and (row["dead_letter"] or row["status"] in (INBOX_DONE, INBOX_PROCESSING)):
                return False
            self._handler(event)
            return True
        now = float(time.time())
        failure: Exception | None = None
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO inbox_events "
                "(event_id, idempotency_key, status, retry_count, dead_letter, processed_at) "
                "VALUES (?, ?, 'RECEIVED', 0, FALSE, ?)",
                (event.get("event_id") or key, key, now),
            )
            # Lock existing rows too. On PostgreSQL a concurrent first insert
            # waits for the other transaction; this update then locks its result.
            conn.execute(
                "UPDATE inbox_events SET status = status WHERE idempotency_key = ?",
                (key,),
            )
            row = conn.execute(
                "SELECT status, dead_letter, retry_count, last_error FROM inbox_events "
                "WHERE idempotency_key = ?", (key,),
            ).fetchone()
            if row["dead_letter"]:
                raise InboxEventNotFound(f"inbox event for key {key!r} is dead-lettered")
            if row["status"] in (INBOX_DONE, INBOX_PROCESSING):
                return False
            if row["status"] not in (INBOX_RECEIVED, INBOX_FAILED):
                raise ValueError(f"unsupported inbox state: {row['status']!r}")
            retries = int(row["retry_count"])
            if row["status"] == INBOX_FAILED and retries >= self.max_retries:
                conn.execute(
                    "UPDATE inbox_events SET status = 'FAILED', retry_count = ?, "
                    "dead_letter = TRUE, dead_letter_reason = ?, last_error = ?, "
                    "next_attempt_at = NULL, processed_at = ? WHERE idempotency_key = ?",
                    (retries + 1, "max retries exceeded", "max retries exceeded", now, key),
                )
                failure = InboxEventNotFound(f"inbox event for key {key!r} exceeded {self.max_retries} retries")
            else:
                conn.execute(
                    "UPDATE inbox_events SET status = 'PROCESSING', processed_at = ? "
                    "WHERE idempotency_key = ?", (now, key),
                )
                savepoint = "inbox_handler_" + uuid.uuid4().hex
                conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    self._handler(event)
                    conn.execute(
                        "UPDATE inbox_events SET status = 'DONE', processed_at = ?, "
                        "next_attempt_at = NULL, last_error = NULL WHERE idempotency_key = ?",
                        (float(time.time()), key),
                    )
                except Exception as exc:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    conn.execute(
                        "UPDATE inbox_events SET status = 'FAILED', retry_count = ?, "
                        "next_attempt_at = ?, last_error = ?, processed_at = ? "
                        "WHERE idempotency_key = ?",
                        (retries + 1, now + 1.0, str(exc)[:4000], now, key),
                    )
                    failure = exc
                finally:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        if failure is not None:
            raise failure
        return True

    def process_many(self, events: list[dict[str, Any]]) -> tuple[int, int]:
        """Process a batch outside any caller transaction.

        Returns ``(handled, duplicates)`` where ``handled`` counts rows that ran
        the handler (including failures recorded as FAILED, which are retryable
        later). Dead-lettered / missing / skipped rows are counted as duplicates
        (not handled).
        """
        handled = 0
        duplicates = 0
        for event in events:
            try:
                if self.process(event):
                    handled += 1
                else:
                    duplicates += 1
            except InboxEventNotFound:
                duplicates += 1
            except ValueError:
                duplicates += 1
            except Exception:
                # Handler failure for a message that is NOT dead-lettered: the
                # message is durably recorded FAILED (retryable) — counted as
                # handled so the broker/consumer does not drop it.
                handled += 1
        return handled, duplicates

    # ---- production ops ----
    def dead_lettered(self) -> list[dict[str, Any]]:
        """Rows in the terminal dead-letter state (production ops visibility).

        P0-PLAT-005: the inbox stores a ``dead_letter`` flag but had no
        read-side export for operations to drain/review.  This returns every
        inbox row with ``dead_letter = 1``, newest last_error first.  Each dict
        is a raw row (``idempotency_key`` / ``status`` / ``retry_count`` /
        ``last_error`` / ``dead_letter_reason`` / ``processed_at``) — an honest
        snapshot, never an auto-retry.
        """
        return list(
            self._db.query(
                "SELECT idempotency_key, status, retry_count, last_error, "
                "dead_letter_reason, processed_at FROM inbox_events "
                "WHERE dead_letter = TRUE ORDER BY processed_at DESC"
            )
        )
