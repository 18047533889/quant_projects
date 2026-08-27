"""Outbox tests — same-transaction write, publish, inbox idempotency."""

from __future__ import annotations

import pytest

from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.outbox import InMemoryPublisher, Inbox, Outbox


def test_emit_inside_transaction_commits():
    db = SqliteDb(":memory:")
    outbox = Outbox(db, InMemoryPublisher())
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO principals (principal_id, principal_type, display_name) VALUES (?, ?, ?)",
            ("p1", "HUMAN", "Alice"),
        )
        outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key="ik-1",
            payload={"candidate_id": "cand-1"},
        )
    assert len(db.query("SELECT * FROM outbox_events")) == 1


def test_emit_rolls_back_with_transaction():
    db = SqliteDb(":memory:")
    outbox = Outbox(db, InMemoryPublisher())
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            outbox.emit(
                event_type="FactorCandidateDiscovered",
                aggregate_type="factor_candidate",
                aggregate_id="cand-1",
                correlation_id="corr-1",
                idempotency_key="ik-1",
            )
            raise RuntimeError("boom")
    assert db.query("SELECT * FROM outbox_events") == []


def test_publish_pending_delivers_and_marks_sent():
    db = SqliteDb(":memory:")
    publisher = InMemoryPublisher()
    outbox = Outbox(db, publisher)
    with db.transaction() as conn:
        outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key="ik-1",
        )
    assert outbox.publish_pending() == 1
    assert len(publisher.delivered) == 1
    assert db.query("SELECT status FROM outbox_events")[0]["status"] == "sent"


def test_publish_failure_keeps_pending_and_increments_attempts():
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


def test_inbox_dedupes_by_idempotency_key():
    db = SqliteDb(":memory:")
    handled = []
    inbox = Inbox(db, lambda ev: handled.append(ev))
    ev = {"event_id": "e1", "idempotency_key": "ik-1", "event_type": "X"}
    assert inbox.process(ev) is True
    assert inbox.process(ev) is False
    assert len(handled) == 1


def test_inbox_requires_idempotency_key():
    db = SqliteDb(":memory:")
    inbox = Inbox(db, lambda ev: None)
    with pytest.raises(ValueError):
        inbox.process({"event_id": "e1"})
