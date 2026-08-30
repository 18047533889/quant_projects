"""R55 #91 — in-memory WorkerLoop claim-once across two workers.

Two workers racing the same in-memory outbox row must result in exactly one
delivery (mark_claiming is the atomic guard; the losing worker cannot publish).
"""

from __future__ import annotations

from quant_platform.app.worker.publish import (
    InMemoryOutbox,
    PumpReport,
    WorkerLoop,
)

from quant_platform.app.outbox import InMemoryPublisher as SQLInMemoryPublisher


def _envelope(eid: str, key: str):
    from datetime import datetime, timezone

    from quant_platform.app.contracts import EventEnvelope

    return EventEnvelope(
        event_id=eid,
        event_type="FactorCandidateDiscovered",
        schema_version="1.0",
        occurred_at=datetime.now(timezone.utc),
        producer="platform",
        aggregate_type="factor_candidate",
        aggregate_id="cand-1",
        correlation_id="corr-1",
        idempotency_key=key,
    )


def test_two_workers_only_one_claims_same_inmemory_row():
    outbox = InMemoryOutbox()
    outbox.append(_envelope("e1", "ik-1"))

    pubB = SQLInMemoryPublisher()
    loopB = WorkerLoop(outbox, pubB, in_flight_timeout_s=10.0)

    # Worker A claims the pending row first.
    assert outbox.mark_claiming("e1", "worker-A") is True
    assert outbox.row("e1").status == "claimed"
    assert outbox.row("e1").claimed_by == "worker-A"

    # Worker B attempts the same row -> claim fails (cannot win).
    assert outbox.mark_claiming("e1", "worker-B") is False
    assert outbox.row("e1").claimed_by == "worker-A"

    # Worker B's loop sees nothing publishable on its behalf.
    repB = loopB.publish_and_claim(worker_id="worker-B")
    assert repB == PumpReport()
    assert not pubB.delivered

    # Worker A updates and can settle the row (claimed->sent exactly once).
    repA = loopB.publish_and_claim(worker_id="worker-A")
    assert repA.published == 1
    assert outbox.row("e1").status == "sent"
    assert len(pubB.delivered) == 1


def test_same_worker_claim_is_idempotent_not_double_publish():
    outbox = InMemoryOutbox()
    outbox.append(_envelope("e1", "ik-1"))
    pub = SQLInMemoryPublisher()
    loop = WorkerLoop(outbox, pub, in_flight_timeout_s=10.0)
    pub.publish = lambda ev: pub.delivered.append(ev)

    assert loop.publish_and_claim(worker_id="w1").published == 1
    # a second full loop must not re-deliver the same event
    rep = loop.publish_and_claim(worker_id="w1")
    assert rep.published == 0
    assert len(pub.delivered) == 1


def test_claim_and_publish_with_failure_records_error_and_recovers():
    outbox = InMemoryOutbox()
    outbox.append(_envelope("e1", "ik-1"))
    pub = SQLInMemoryPublisher()
    loop = WorkerLoop(outbox, pub, in_flight_timeout_s=10.0)
    state = {"left": 1}

    def flaky(ev):
        if state["left"] > 0:
            state["left"] -= 1
            raise RuntimeError("bus down")
        pub.delivered.append(ev)

    pub.publish = flaky
    rep = loop.publish_and_claim(worker_id="w1")
    assert rep.failed == 1 and rep.published == 0
    assert outbox.row("e1").status == "failed"
    assert outbox.row("e1").last_error == "bus down"
    # row stays failed (not re-pulled), then recovers on the next claim cycle
    assert loop.publish_and_claim(worker_id="w1") == PumpReport()
    outbox.reject("e1")
    rep2 = loop.publish_and_claim(worker_id="w1")
    assert rep2.published == 1
    assert outbox.row("e1").status == "sent"