"""Transactional outbox — same-transaction write + background publish.

QRP-P1. Implements the transactional outbox pattern (spec §11.1): an event row
is inserted in the SAME transaction as the state update it announces, so the
platform never double-writes events and state. A background publisher marks
pending events as sent; an inbox consumer dedupes by idempotency key.

The actual message-bus publish is a pluggable ``Publisher`` Protocol. An
in-process ``InMemoryPublisher`` is provided for tests.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Publisher",
    "InMemoryPublisher",
    "Outbox",
    "Inbox",
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


class Outbox:
    """Transactional outbox writer + background mark-sent.

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

    def publish_pending(self, limit: int = 100) -> int:
        """Publish pending events and mark them sent. Returns count published.

        Runs in its own transaction: reads pending rows, delivers each via the
        publisher, then marks them ``sent``. A delivery failure leaves the row
        ``pending`` (attempts incremented) for a later retry.
        """
        rows = self._db.query(
            "SELECT * FROM outbox_events WHERE status = 'pending' ORDER BY id LIMIT ?",
            (limit,),
        )
        published = 0
        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(row["payload_json"] or "{}")
            try:
                self._publisher.publish(event)
            except Exception:
                with self._db.transaction() as conn:
                    conn.execute(
                        "UPDATE outbox_events SET attempts = attempts + 1 WHERE id = ?",
                        (row["id"],),
                    )
                continue
            with self._db.transaction() as conn:
                conn.execute(
                    "UPDATE outbox_events SET status = 'sent' WHERE id = ?",
                    (row["id"],),
                )
            published += 1
        return published


class Inbox:
    """Idempotent event consumer.

    ``db`` must implement the ``Db`` Protocol. ``handler`` is a callable
    ``(event: dict) -> None`` invoked once per unique idempotency key.
    """

    def __init__(self, db: Any, handler: Any) -> None:
        self._db = db
        self._handler = handler

    def process(self, event: dict[str, Any]) -> bool:
        """Process ``event`` exactly once (by idempotency key). Returns True if
        newly processed, False if it was a duplicate."""
        key = event.get("idempotency_key")
        if not key:
            raise ValueError("inbox event requires idempotency_key")
        existing = self._db.query(
            "SELECT event_id FROM inbox_events WHERE idempotency_key = ?",
            (key,),
        )
        if existing:
            return False
        event_id = event.get("event_id") or key
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO inbox_events (event_id, idempotency_key, processed_at) "
                "VALUES (?, ?, ?)",
                (event_id, key, int(time.time())),
            )
        self._handler(event)
        return True
