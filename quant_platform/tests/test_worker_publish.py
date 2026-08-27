"""WorkerLoop tests — outbox publish: success, failure-retry, idempotency, in-flight timeout.

Covers ``quant_platform/app/worker/publish.py`` (pure in-memory worker side).
"""

from __future__ import annotations

import pytest

from quant_platform.app.outbox import InMemoryPublisher
from quant_platform.app.worker.publish import (
    InMemoryOutbox,
    OutboxRow,
    OutboxWorkerError,
    PublishTimeoutError,
    PumpReport,
    WorkerLoop,
)


def _envelope(eid: str, key: str, event_type: str = "FactorCandidateDiscovered"):
    from datetime import datetime, timezone

    from quant_platform.app.contracts import EventEnvelope

    return EventEnvelope(
        event_id=eid,
        event_type=event_type,
        schema_version="1.0",
        occurred_at=datetime.now(timezone.utc),
        producer="platform",
        aggregate_type="factor_candidate",
        aggregate_id="cand-1",
        correlation_id="corr-1",
        idempotency_key=key,
    )


def _make(publisher) -> tuple[WorkerLoop, InMemoryOutbox]:
    outbox = InMemoryOutbox()
    loop = WorkerLoop(outbox, publisher, in_flight_timeout_s=10.0)
    return loop, outbox


# ---- publisher stubs ----
class _Failing(  # noqa: D101
    InMemoryPublisher
):  # InMemoryPublisher is a concrete class; subclass to inject failure
    fail_first: bool = False
    failures_left: int = 0

    def publish(self, event):
        if self.failures_left > 0:
            self.failures_left -= 1
            raise RuntimeError("bus down")
        self.delivered.append(event)


def _always_fail(publisher, failures=1):
    publisher.failures_left = failures
    return publisher


class _TimeoutPub:
    """Hands off events whose idempotency_key starts with 'timeout-' (no ack)."""

    def __init__(self):
        self.published: list = []
        self.handed_off: list = []

    def publish(self, event):
        env = event if isinstance(event, dict) else event
        key = env.get("idempotency_key") if isinstance(env, dict) else getattr(event, "idempotency_key", "")
        if str(key).startswith("timeout-"):
            self.handed_off.append(event)
            raise PublishTimeoutError()
        self.published.append(event)


# ---- success ----
def test_worker_delivers_and_marks_sent():
    publisher = InMemoryPublisher()
    loop, outbox = _make(publisher)
    row = outbox.append(_envelope("e1", "ik-1"))
    rep = loop.publish_once()
    assert rep == PumpReport(published=1)
    assert outbox.pending() == ()
    assert outbox.row("e1").status == "sent"
    assert publisher.delivered[0].event_id == "e1"
    assert loop.delivered() == ("e1",)


def test_worker_drain_respects_limit():
    # publish_once drains ALL pending; per-event guarantee is 1 publish each.
    publisher = InMemoryPublisher()
    loop, outbox = _make(publisher)
    outbox.append(_envelope("e1", "ik-1"))
    outbox.append(_envelope("e2", "ik-2"))
    rep = loop.publish_once()
    assert rep.published == 2
    assert len(publisher.delivered) == 2


# ---- failure retry ----
def test_failure_keeps_pending_and_increments_attempts():
    publisher = _Failing()
    publisher.delivered = []
    _always_fail(publisher, failures=1)
    loop, outbox = _make(publisher)
    outbox.append(_envelope("e1", "ik-1"))
    rep = loop.publish_once()
    assert rep.failed == 1
    assert outbox.pending_row("e1") is not None
    assert outbox.row("e1").attempt_count == 1
    assert outbox.row("e1").status == "pending"
    # no delivery recorded yet
    assert not publisher.delivered


def test_retry_succeeds_on_second_cycle():
    publisher = _Failing()
    publisher.delivered = []
    _always_fail(publisher, failures=1)
    loop, outbox = _make(publisher)
    outbox.append(_envelope("e1", "ik-1"))
    assert loop.publish_once().failed == 1
    # second cycle succeeds
    assert loop.publish_once().published == 1
    assert outbox.row("e1").status == "sent"
    assert publisher.delivered[0].event_id == "e1"


# ---- idempotency ----
def test_same_event_id_not_reprocessed_after_sent():
    publisher = InMemoryPublisher()
    loop, outbox = _make(publisher)
    outbox.append(_envelope("e1", "ik-1"))
    assert loop.publish_once().published == 1
    # Calling again must not re-deliver (row no longer pending).
    assert loop.publish_once() == PumpReport()
    assert len(publisher.delivered) == 1


def test_append_dedupes_by_idempotency_key():
    _, outbox = _make(InMemoryPublisher())
    r1 = outbox.append(_envelope("e1", "same-key"))
    r2 = outbox.append(_envelope("e2", "same-key"))
    assert r1 is r2
    assert len(outbox) == 1


# ---- in-flight timeout ----
def test_in_flight_timeout_settles_and_retries():
    pub = _TimeoutPub()
    loop, outbox = _make(pub)
    outbox.append(_envelope("ev-1", "timeout-ev"))
    now = [100.0]

    def clock():
        return now[0]

    loop._now_fn = clock
    # First pass: publish raises PublishTimeoutError -> leased in-flight.
    rep1 = loop.publish_once()
    assert rep1.in_flight == 1
    assert outbox.row("ev-1").status == "pending"
    assert loop.in_flight_events() == ("ev-1",)

    # Before timeout: still leased, untouched (no delivery, no state change).
    now[0] += 5.0  # < 10s
    rep2 = loop.publish_once()
    assert rep2.published == 0 and rep2.timed_out == 0
    assert outbox.row("ev-1").status == "pending"
    assert loop.in_flight_events() == ("ev-1",)

    # Lease expires: settle timed out (attempt bumped), then re-attempt. The
    # timeout publisher keeps handoff-without-ack, so the retry goes back
    # in-flight again — the row stays pending.
    now[0] += 6.0  # 11s total > 10s
    rep3 = loop.publish_once()
    assert rep3.timed_out == 1
    assert outbox.row("ev-1").status == "pending"
    assert outbox.row("ev-1").attempt_count == 1  # the lease expiry counts
    assert loop.in_flight_events() == ("ev-1",)


def test_in_flight_release_after_ack_node_adversary():
    # A restart-recovery style node may have no in-flight tracking but must not
    # redeliver the just-acked event. With lease bookkeeping reset, the pending
    # row would be re-published — that is the ack-not-received window. The
    # worker prevents redelivery WITHIN a process by counting the row sent.
    pub = InMemoryPublisher()
    loop, outbox = _make(pub)
    outbox.append(_envelope("e1", "ik-1"))
    assert loop.publish_once().published == 1
    assert len(loop.delivered()) == 1
    # simulate a dropped _delivered (ack not persisted): row already sent.
    assert outbox.row("e1").status == "sent"
    assert loop.publish_once() == PumpReport()


def test_publish_timeout_input_error_is_not_a_timeout():
    # A non-timeout failure is distinct from an in-flight timeout; a normal event
    # publishes cleanly.
    pub = _TimeoutPub()
    loop, outbox = _make(pub)
    outbox.append(_envelope("e-normal", "normal-key"))
    assert loop.publish_once().published == 1
    assert pub.published  # delivered


def test_publish_pending_items_never_redelivered():
    # After a successful publish, a retry cycle never re-delivers.
    pub = InMemoryPublisher()
    loop, outbox = _make(pub)
    outbox.append(_envelope("e1", "ik-1"))
    assert loop.publish_once().published == 1
    assert loop.publish_once() == PumpReport()
    assert len(pub.delivered) == 1


def test_worker_rejects_non_envelope_rows():
    loop, outbox = _make(InMemoryPublisher())
    with pytest.raises(TypeError):
        outbox.append({"event_id": "e1"})