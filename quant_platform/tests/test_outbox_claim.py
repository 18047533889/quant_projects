"""R55 #90/#91 — inbox state machine + outbox claim-once.

Covers:
- inbox: handler failure is recorded FAILED (never burned as DONE), retry
  recovers, dead-letter at max retries, handler side-effect + ack in one
  transaction, duplicate DONE is a no-op.
- outbox: SQL claim-once across two workers (exactly one claimer), the new
  claim/retry columns, and failure leaves a recoverable pending row.
"""

from __future__ import annotations

import pytest

from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.outbox import (
    InMemoryPublisher,
    Inbox,
    InboxEventNotFound,
    Outbox,
)
from quant_platform.app.db.schema import create_schema


def _emit(db: Outbox | None = None):
    """Emit one event in its own transaction; returns (outbox, event_id)."""
    box = Outbox(db, InMemoryPublisher())
    with db.transaction() as conn:
        event_id = box.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key="ik-1",
            payload={"candidate_id": "cand-1"},
        )
    return box, event_id


def _source_event(key: str = "ik-1") -> dict:
    return {
        "event_id": "e1",
        "idempotency_key": key,
        "event_type": "FactorCandidateDiscovered",
        "aggregate_type": "factor_candidate",
        "aggregate_id": "cand-1",
    }


# ---------------------------------------------------------------------------
# #90 Inbox state machine
# ---------------------------------------------------------------------------


def test_handler_failure_recorded_failed_not_burned():
    db = SqliteDb(":memory:")

    def boom(event):
        raise RuntimeError("handler exploded")

    inbox = Inbox(db, boom)
    with pytest.raises(RuntimeError):
        inbox.process(_source_event())
    row = inbox.get("ik-1")
    assert row["status"] == "FAILED"
    assert row["retry_count"] == 1
    assert "handler exploded" in (row["last_error"] or "")
    assert row["dead_letter"] == 0
    # A later reprocess with a working handler recovers the SAME message.
    handled = []
    inbox2 = Inbox(db, lambda ev: handled.append(ev))
    assert inbox2.process(_source_event()) is True
    assert inbox2.get("ik-1")["status"] == "DONE"
    assert len(handled) == 1


def test_duplicate_done_is_noop():
    db = SqliteDb(":memory:")
    handled = []
    inbox = Inbox(db, lambda ev: handled.append(ev))
    ev = _source_event()
    assert inbox.process(ev) is True
    assert inbox.process(ev) is False
    assert len(handled) == 1
    assert inbox.get("ik-1")["status"] == "DONE"


def test_dead_letter_after_max_retries():
    db = SqliteDb(":memory:")

    def boom(event):
        raise RuntimeError("always fails")

    inbox = Inbox(db, boom, max_retries=2)
    # first failure -> FAILED (retry_count=1), still retryable
    with pytest.raises(RuntimeError):
        inbox.process(_source_event())
    assert inbox.get("ik-1")["status"] == "FAILED"
    assert inbox.get("ik-1")["retry_count"] == 1
    # second failure -> FAILED (retry_count=2); the NEXT attempt (already at
    # max_retries) is refused up-front and dead-lettered instead of re-running
    # the handler a third time.
    with pytest.raises(RuntimeError):
        inbox.process(_source_event())
    assert inbox.get("ik-1")["retry_count"] == 2
    assert inbox.get("ik-1")["dead_letter"] == 0
    with pytest.raises(InboxEventNotFound):
        inbox.process(_source_event())
    row = inbox.get("ik-1")
    assert row["status"] == "FAILED"
    assert row["dead_letter"] == 1
    assert row["retry_count"] == 3
    # a later attempt raises InboxEventNotFound (dead-lettered, not reprocessed)
    handled = []
    inbox3 = Inbox(db, lambda ev: handled.append(ev))
    with pytest.raises(InboxEventNotFound):
        inbox3.process(_source_event())
    assert len(handled) == 0


def test_handler_side_effect_and_ack_same_transaction():
    db = SqliteDb(":memory:")

    class Handler:
        def __init__(self, db, boom: bool = False):
            self.boom = boom

        def __call__(self, event):
            # side effect on a separate table
            db.execute(
                "INSERT INTO teams (team_id, name) VALUES ('t-auto', 'auto')",
            )
            if self.boom:
                raise RuntimeError("boom after side effect")

    inbox = Inbox(db, Handler(db, boom=True))
    with pytest.raises(RuntimeError):
        inbox.process(_source_event())
    # handler side effect rolled back WITH the ack -> not burned, no ghost row
    rows = db.query("SELECT * FROM teams")
    assert rows == []
    assert inbox.get("ik-1")["status"] == "FAILED"
    assert inbox.get("ik-1")["retry_count"] == 1


def test_inbox_requires_idempotency_key():
    db = SqliteDb(":memory:")
    inbox = Inbox(db, lambda ev: None)
    with pytest.raises(ValueError):
        inbox.process({"event_id": "e1"})


def test_process_many_counts_failures_as_handled_not_lost():
    db = SqliteDb(":memory:")
    inbox = Inbox(db, lambda ev: (_ for _ in ()).throw(RuntimeError("nope")), max_retries=3)
    handled, dups = inbox.process_many([_source_event("ik-a")])
    assert handled == 1 and dups == 0
    assert inbox.get("ik-a")["status"] == "FAILED"
    assert inbox.get("ik-a")["retry_count"] == 1


# ---------------------------------------------------------------------------
# #91 Outbox claim-once
# ---------------------------------------------------------------------------


def test_schema_exposes_claim_and_retry_columns():
    db = SqliteDb(":memory:")
    cols = {r["name"] for r in db.query("PRAGMA table_info(outbox_events)")}
    for col in (
        "worker_id",
        "claim_token",
        "claimed_at",
        "retry_count",
        "next_attempt_at",
        "last_error",
        "dead_letter_reason",
    ):
        assert col in cols, f"outbox_events lacks column {col}"
    icols = {r["name"] for r in db.query("PRAGMA table_info(inbox_events)")}
    for col in ("status", "retry_count", "next_attempt_at", "last_error", "dead_letter"):
        assert col in icols, f"inbox_events lacks column {col}"


def test_two_workers_only_one_can_claim_same_sql_row():
    db = SqliteDb(":memory:")
    outbox = Outbox(db, InMemoryPublisher())
    with db.transaction() as conn:
        event_id = outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key="ik-1",
        )
    now = 1000.0
    # Worker A claims (atomically flips pending->claimed).
    assert outbox.claim(event_id, worker_id="worker-A", claim_token="A", now=now) is True
    row = db.query("SELECT status, worker_id, retry_count FROM outbox_events")[0]
    assert row["status"] == "claimed"
    assert row["worker_id"] == "worker-A"
    assert row["retry_count"] == 1
    # Worker B tries the same row while A holds it -> cannot claim.
    assert outbox.claim(event_id, worker_id="worker-B", claim_token="B", now=now + 1) is False
    row = db.query("SELECT status, worker_id, claim_token FROM outbox_events")[0]
    assert row["status"] == "claimed"
    assert row["worker_id"] == "worker-A"
    assert row["claim_token"] == "A"


def test_publish_pending_never_double_delivers_and_recovers_failures():
    db = SqliteDb(":memory:")
    delivered = []

    class P:
        fail_first: bool = True

        def publish(self, event):
            if self.fail_first:
                self.fail_first = False
                raise RuntimeError("bus down")
            delivered.append(event)

    outbox = Outbox(db, P())
    with db.transaction() as conn:
        event_id = outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key="ik-1",
        )
    # first drain: delivery fails -> pending + attempts/retry_count bumped
    assert outbox.publish_pending() == 0
    row = db.query(
        "SELECT status, attempts, retry_count, last_error FROM outbox_events"
    )[0]
    assert row["status"] == "pending"
    assert row["attempts"] == 1
    assert row["retry_count"] == 1
    assert row["last_error"] == "bus down"
    # second drain succeeds -> sent exactly once
    assert outbox.publish_pending() == 1
    assert len(delivered) == 1
    assert db.query("SELECT status FROM outbox_events")[0]["status"] == "sent"


def test_publish_failure_keeps_pending_and_increments_attempts_legacy_alignment():
    db = SqliteDb(":memory:")
    failing = InMemoryPublisher()
    failing.publish = lambda ev: (_ for _ in ()).throw(RuntimeError("bus down"))
    outbox = Outbox(db, failing)
    with db.transaction() as conn:
        outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key="ik-1",
        )
    assert outbox.publish_pending() == 0
    row = db.query("SELECT status, attempts FROM outbox_events")[0]
    assert row["status"] == "pending"
    assert row["attempts"] == 1