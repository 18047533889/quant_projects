"""In-memory outbox worker loop — Transactional Outbox worker side.

Complements ``quant_platform/app/outbox.py``, which is the *writer* side (durable
SQL rows + ``Publisher`` seam). This module is the *reader/publisher* worker
side, pure in-memory (dict + dataclass) for P4.

Pipeline (spec §11.1):

    Producer --append()--> InMemoryOutbox --WorkerLoop.publish_once()--> Publisher

- ``InMemoryOutbox`` queues pending events (deduped by idempotency_key; keyed by
  event_id). Rows move ``pending ``->``sent`` only after the publisher delivers.
- ``WorkerLoop`` pulls pending rows and drives ``Publisher.publish(event)``:
  success marks the row ``sent`` (published at most once per event_id); a
  delivery exception leaves it ``pending`` and increments attempts for a later
  retry.
- In-flight timeout: ``Publisher.publish`` may raise ``PublishTimeoutError`` to
  signal "handed to the bus, ack pending" (no confirmation). The worker leases
  the row as in-flight and, if the ack never settles inside
  ``in_flight_timeout_s``, settles it: increments attempts and re-attempts.
  Because ``sent`` rows are never re-pulled, the same event_id is delivered at
  most once per loop until it is marked sent, and the bus-dedup guarantee holds
  even across loop restarts (rows that were in-flight remain ``pending``).

The publisher must satisfy the ``outbox.Publisher`` Protocol
(``publish(event) -> None``); in addition it may raise ``PublishTimeoutError``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from ..contracts.event_envelope import EVENT_TYPES, EventEnvelope

__all__ = [
    "OutboxWorkerError",
    "PublishTimeoutError",
    "OutboxRow",
    "InMemoryOutbox",
    "WorkerLoop",
    "PumpReport",
    # Re-export for worker infra convenience.
    "EventEnvelope",
]


class OutboxWorkerError(Exception):
    """Base error raised by the in-memory outbox worker."""


class PublishTimeoutError(OutboxWorkerError):
    """Raised by a publisher when a publish attempt never confirmed.

    Signals to the worker that the event was handed to the bus but no ack
    followed (in-flight / ack-pending). The worker keeps the row pending and
    leases it as in-flight until the in-flight timeout settles a retry.
    """


@dataclass(frozen=True)
class OutboxRow:
    """One outbox row (event_id keyed). ``status`` is ``pending`` or ``sent``."""

    event: EventEnvelope
    status: str = "pending"
    attempt_count: int = 0
    created_at: float = field(default_factory=time.time)
    sent_at: float | None = None

    @property
    def event_id(self) -> str:
        return self.event.event_id


class InMemoryOutbox:
    """Queue-backed outbox, pure in-memory.

    ``append`` is the write (subscript, deduped by idempotency_key);
    ``pending`` is the readable snapshot the worker pulls from.
    """

    def __init__(self) -> None:
        self._rows: dict[str, OutboxRow] = {}
        self._by_key: dict[str, str] = {}
        self._order: list[str] = []

    def append(
        self,
        event: EventEnvelope,
        *,
        idempotency_key: str | None = None,
        event_id: str | None = None,
    ) -> OutboxRow:
        """Queue a pending event. Idempotent by idempotency_key: a repeat with
        the same key returns the existing row (never a duplicate)."""
        _validate_event(event)
        key = idempotency_key or event.idempotency_key
        if not key:
            raise ValueError("outbox event requires idempotency_key")
        if existing := self._by_key.get(key):
            return self._rows[existing]
        eid = event_id or event.event_id
        row = OutboxRow(event=event, status="pending", attempt_count=0)
        self._rows[eid] = row
        self._by_key[key] = eid
        self._order.append(eid)
        return row

    def pending(self) -> tuple[OutboxRow, ...]:
        """Pending (unsent) rows in FIFO order, oldest first."""
        return tuple(
            self._rows[eid] for eid in self._order if self._rows[eid].status == "pending"
        )

    def pending_row(self, event_id: str) -> OutboxRow | None:
        """Latest row for ``event_id`` if it is still pending, else None."""
        row = self._rows.get(event_id)
        return row if row is not None and row.status == "pending" else None

    def row(self, event_id: str) -> OutboxRow:
        return self._rows[event_id]

    def rows(self) -> tuple[OutboxRow, ...]:
        """All rows (pending + sent) in FIFO order."""
        return tuple(self._rows[eid] for eid in self._order)

    def mark_sent(self, event_id: str, *, at: float | None = None) -> None:
        row = self._rows.get(event_id)
        if row is None:
            raise KeyError(f"no outbox row for event_id {event_id!r}")
        if row.status != "pending":
            raise OutboxWorkerError(f"event {event_id!r} not pending (status={row.status!r})")
        self._rows[event_id] = replace(
            row,
            status="sent",
            sent_at=at if at is not None else time.time(),
        )

    def mark_attempt(self, event_id: str) -> int:
        """Increment a pending row's attempt count; returns the new count."""
        row = self._rows.get(event_id)
        if row is None:
            raise KeyError(f"no outbox row for event_id {event_id!r}")
        if row.status != "pending":
            return row.attempt_count
        self._rows[event_id] = replace(row, attempt_count=row.attempt_count + 1)
        return self._rows[event_id].attempt_count

    def __len__(self) -> int:
        return len(self._order)


@dataclass(frozen=True)
class PumpReport:
    """Per-cycle worker accounting."""

    published: int = 0  # rows marked sent this cycle
    failed: int = 0  # publish raised (non-timeout); attempt incremented
    in_flight: int = 0  # publish raised PublishTimeoutError (ack pending)
    timed_out: int = 0  # in-flight lease expired; attempt incremented + retried

    def __add__(self, other: "PumpReport") -> "PumpReport":
        return PumpReport(
            published=self.published + other.published,
            failed=self.failed + other.failed,
            in_flight=self.in_flight + other.in_flight,
            timed_out=self.timed_out + other.timed_out,
        )


class WorkerLoop:
    """Background loop draining pending events to a ``Publisher``.

    Parameters
    ----------
    outbox : ``InMemoryOutbox`` the worker pulls pending rows from.
    publisher : the ``outbox.Publisher`` seam delivering each event. May raise
        ``PublishTimeoutError`` for ack-pending publishes, or any ``Exception``
        for a definitive failure.
    in_flight_timeout_s : in-flight lease. A pending event whose publish handed
        off without an ack is considered timed out (attempt incremented and
        retried) once its in-flight time exceeds this window.
    now_fn : test seam for a controllable clock ``() -> float``.
    """

    def __init__(
        self,
        outbox: InMemoryOutbox,
        publisher: Any,
        *,
        in_flight_timeout_s: float = 1.0,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.outbox = outbox
        self.publisher = publisher
        self.in_flight_timeout_s = (
            in_flight_timeout_s if in_flight_timeout_s > 0 else 1.0
        )
        self._now_fn = now_fn if now_fn is not None else time.time
        self._in_flight: dict[str, float] = {}  # event_id -> dispatch time (lease)
        self._delivered: list[str] = []  # event_ids marked sent by this loop

    # ---- query ----
    def in_flight_events(self) -> tuple[str, ...]:
        return tuple(sorted(self._in_flight))

    def delivered(self) -> tuple[str, ...]:
        return tuple(self._delivered)

    # ---- main pulls ----
    def publish_once(self, *, now: float | None = None) -> PumpReport:
        """One drain pass over ``outbox.pending()``. Returns per-cycle counts.

        A row is pulled while ``status == 'pending'``; once ``mark_sent`` is
        applied it is never re-pulled, so the same event_id is processed at most
        once (idempotent within the loop).
        """
        now = self._now_fn() if now is None else now
        report = PumpReport()
        for row in self.outbox.pending():
            eid = row.event_id
            # In-flight lease check: handed off, ack pending.
            dispatched_at = self._in_flight.get(eid)
            if dispatched_at is not None:
                if now - dispatched_at < self.in_flight_timeout_s:
                    report = report + PumpReport(in_flight=1)
                    continue  # lease active: leave untouched, zero-count
                # Lease expired with no ack: settle the handoff, keep the row
                # pending (attempt bumped) and let this same cycle retry the
                # publish below. At-least-once on retry; the consumer dedupes on
                # idempotency_key.
                self.outbox.mark_attempt(eid)
                del self._in_flight[eid]
                report = report + PumpReport(timed_out=1)
            current = self.outbox.pending_row(eid)
            if current is None:
                continue  # raced / already sent
            try:
                self.publisher.publish(current.event)
                self.outbox.mark_sent(eid, at=now)
                self._in_flight.pop(eid, None)
                self._delivered.append(eid)
                report = report + PumpReport(published=1)
            except PublishTimeoutError:
                # Handed to the bus but no ack: lease it as in-flight.
                self._in_flight[eid] = now
                report = report + PumpReport(in_flight=1)
                continue
            except Exception:
                # Definitive delivery failure: keep pending for a later retry.
                self.outbox.mark_attempt(eid)
                self._in_flight.pop(eid, None)
                report = report + PumpReport(failed=1)
                continue
        return report

    def run_until_quiesce(self, *, max_cycles: int = 100) -> PumpReport:
        """Keep draining until a cycle makes no progress. Returns accumulated
        counts across all cycles."""
        total = PumpReport()
        for _ in range(max_cycles):
            rep = self.publish_once()
            total = total + rep
            if rep == PumpReport():
                break
        return total


def _validate_event(event: Any) -> EventEnvelope:
    if not isinstance(event, EventEnvelope):
        raise TypeError("outbox rows must be EventEnvelope instances")
    if event.event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event_type: {event.event_type!r}")
    if not event.idempotency_key:
        raise ValueError("outbox event requires idempotency_key")
    return event