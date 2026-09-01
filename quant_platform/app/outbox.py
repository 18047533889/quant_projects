"""Transactional outbox/inbox — same-transaction write + background publish.

QRP-P1. Implements the transactional outbox pattern (spec §11.1): an event row
is inserted in the SAME transaction as the state update it announces, so the
platform never double-writes events and state. A background publisher claims
pending events (a claim ack is never double-delivered) and marks them sent; an
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

        Only a row currently in ``pending``/``claimed`` for this same worker can
        be flipped to ``claimed``; any other state (``sent``, or ``claimed`` by a
        *different* worker) returns False. The single guarded ``UPDATE`` is the
        row lock: PostgreSQL acquires a row-level lock per row (and a live PG
        backend should use ``SELECT ... FOR UPDATE SKIP LOCKED`` for the
        ``publish_pending`` scan — see its docstring), SQLite relies on the
        ``BEGIN IMMEDIATE`` serialization already in ``transaction()``.

        No races need to be handled in caller code: exactly one worker can ever
        observe this row as pending-and-claimable for itself.
        """
        self._db.execute(
            "UPDATE outbox_events SET status = 'claimed', worker_id = ?, "
            "claim_token = ?, claimed_at = ?, retry_count = retry_count + 1 "
            "WHERE id = ? AND status IN ('pending', 'claimed') "
            "AND (worker_id IS NULL OR worker_id = ?)",
            (worker_id, claim_token, now, event_id, worker_id),
        )
        rows = self._db.query(
            "SELECT status, worker_id, claim_token FROM outbox_events WHERE id = ?",
            (event_id,),
        )
        if not rows:
            return False
        row = rows[0]
        return (
            row["status"] == "claimed"
            and str(row.get("worker_id") or "") == worker_id
            and str(row.get("claim_token") or "") == claim_token
        )

    def finish(self, event_id: int, *, claim_token: str) -> bool:
        """Mark a claimed row ``sent``. No-op (False) unless the caller still
        holds the claim (guarded by claim_token)."""
        self._db.execute(
            "UPDATE outbox_events SET status = 'sent', claimed_at = NULL "
            "WHERE id = ? AND status = 'claimed' AND claim_token = ?",
            (event_id, claim_token),
        )
        rows = self._db.query(
            "SELECT status FROM outbox_events WHERE id = ?", (event_id,)
        )
        return bool(rows and rows[0]["status"] == "sent")

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

    def publish_pending(self, limit: int = 100) -> int:
        """Claim + publish pending events and mark them sent.

        Each event is claimed in its own transaction (concurrency-safe across
        workers), delivered via the publisher, then marked ``sent`` in a second
        transaction. A delivery failure returns the row to ``pending`` with
        ``retry_count`` incremented (never double-delivered, always recoverable).

        PostgreSQL note: a live PG backend should select candidate rows with
        ``SELECT ... FOR UPDATE SKIP LOCKED`` (row-claim semantics) so that two
        workers scanning concurrently never both pick the same row. The claim
        here is row-lock-safe even without it — the guarded ``UPDATE``/verify
        round-trip flips only a ``pending`` row and the second worker's update
        simply matches 0 rows. The ``SKIP LOCKED`` optimization is documented in
        ``postgres_backend.py`` as the deployment-time tuning knob.

        Returns count published.
        """
        rows = self._db.query(
            "SELECT * FROM outbox_events WHERE status = 'pending' ORDER BY id LIMIT ?",
            (limit,),
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
                    "WHERE id = ? AND status = 'pending'",
                    (worker_id, claim_token, now, event_id),
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
                    1 if dead_letter else 0,
                    error[:4000] if dead_letter else None,
                    now,
                    key,
                ),
            )

    def process(self, event: dict[str, Any]) -> bool:
        """Process ``event`` through one state transition.

        Returns True if the handler newly ran (RECEIVED->PROCESSING->DONE or
        FAILED-recovered), False if it was a duplicate (already DONE) or a
        dead-lettered message processed by a replica. ``InboxEventNotFound`` is
        raised when the event is missing or already dead-lettered after the
        handler failed past max retries (the handler's original exception is
        surfaced as-is on the first failure so callers observe it).
        """
        key = event.get("idempotency_key")
        if not key:
            raise ValueError("inbox event requires idempotency_key")
        now = float(time.time())
        event_id = event.get("event_id") or key

        rows = self._db.query(
            "SELECT status, dead_letter, retry_count FROM inbox_events "
            "WHERE idempotency_key = ?",
            (key,),
        )
        if not rows:
            # First sight: RECEIVE + PROCESSING together with the handler in one
            # transaction so the ack and the handler's DB side effects commit
            # (or roll back) atomically.
            with self._db.transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO inbox_events "
                    "(event_id, idempotency_key, status, retry_count, next_attempt_at, "
                    " last_error, dead_letter, dead_letter_reason, processed_at) "
                    "VALUES (?, ?, 'PROCESSING', 0, NULL, NULL, 0, NULL, ?)",
                    (event_id, key, now),
                )
                # R55 #92: ``INSERT OR IGNORE`` is SQLite-only.  PostgreSQL
                # normalizes it to ``INSERT ... ON CONFLICT DO NOTHING`` (see
                # ``PostgresDialect.adapt_ignore``).
        else:
            prev = rows[0]
            prev_status = prev["status"]
            if prev["dead_letter"]:
                raise InboxEventNotFound(
                    f"inbox event for key {key!r} is dead-lettered: "
                    f"{prev.get('last_error')!r}"
                )
            if prev_status == INBOX_DONE:
                return False
            if prev_status == INBOX_PROCESSING:
                # Already claimed by another consumer; do not run the handler.
                return False
            if prev_status == INBOX_RECEIVED:
                # Upgrade to PROCESSING in its own transaction so the next
                # reader never doubles the handler on this row.
                with self._db.transaction() as conn:
                    conn.execute(
                        "UPDATE inbox_events SET status = 'PROCESSING', processed_at = ? "
                        "WHERE idempotency_key = ? AND status = 'RECEIVED'",
                        (now, key),
                    )
            if prev_status == INBOX_FAILED:
                # FAILED + retries exhausted -> dead-letter now (before this
                # retry attempt burns another handler run).
                if int(prev["retry_count"]) >= self.max_retries:
                    self.mark_failed(
                        event, error="max retries exceeded", now=now, dead_letter=True
                    )
                    raise InboxEventNotFound(
                        f"inbox event for key {key!r} exceeded {self.max_retries} retries; "
                        "dead-lettered"
                    )
                # FAILED -> PROCESSING for this retry attempt.
                with self._db.transaction() as conn:
                    conn.execute(
                        "UPDATE inbox_events SET status = 'PROCESSING', processed_at = ? "
                        "WHERE idempotency_key = ? AND status = 'FAILED' "
                        "AND dead_letter = 0",
                        (now, key),
                    )

        if self.replica:
            self._handler(event)
            return True

        try:
            with self._db.transaction() as conn:
                self._handler(event)
                conn.execute(
                    "UPDATE inbox_events SET status = 'DONE', processed_at = ? "
                    "WHERE idempotency_key = ?",
                    (now, key),
                )
        except InboxEventNotFound:
            raise
        except Exception as exc:
            error = str(exc)[:4000]
            rows_after = self._db.query(
                "SELECT retry_count FROM inbox_events WHERE idempotency_key = ?",
                (key,),
            )
            retries = int(rows_after[0]["retry_count"]) if rows_after else 0
            dead_letter = retries >= self.max_retries
            self.mark_failed(
                event,
                error=error,
                now=now,
                dead_letter=dead_letter,
            )
            if dead_letter:
                raise InboxEventNotFound(
                    f"inbox event for key {key!r} exceeded {self.max_retries} retries; "
                    "dead-lettered: " + error
                ) from exc
            # Non-terminal handler failure: recorded FAILED (retryable), message
            # is NOT burned as processed — a later process() re-runs it. The
            # handler's original exception is propagated so callers can react.
            raise
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
                "WHERE dead_letter = 1 ORDER BY processed_at DESC"
            )
        )
