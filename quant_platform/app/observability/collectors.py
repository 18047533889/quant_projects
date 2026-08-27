"""Collectors — feed platform layer read-only state into a ``MetricsRegistry``.

Pure in-memory, stdlib only. Each collector reads an existing layer's immutable
snapshot surface (``JobExecutionRecord`` snapshots, ``PumpReport``/``InMemoryOutbox``,
``ArtifactRegistry``) and derives a small set of gauges/counters onto a
``MetricsRegistry`` (which it returns — pass one in to accumulate, or rely on a
fresh registry). Collectors never mutate the inspected object.

Kind semantics follow what each value *is*:

* snapshot state (rows currently pending/sent/attempted, registered entries,
  per-type distributions, per-status job counts, resolution hits/misses) ->
  ``set_value`` gauges — re-collection is idempotent;
* historical totals / aggregates (summed durations, pump-cycle counts,
  reconcile batch counts) -> counters via ``incr`` / boxes via ``observe``.

Metric names (label dimensions in ``{braces}``):

``jobs_total{status}``         gauge  — records currently in each JobStatus
``jobs_retried_total``         gauge  — records whose attempt_count > 1
``jobs_duration_seconds``      gauge  — min/max/sum/count histogram over per-record
                                        (finished - started) durations
``outbox_pending``             gauge  — rows still pending
``outbox_sent_total``          gauge  — rows marked sent
``outbox_attempts_total``      gauge  — summed attempt_count across all rows
``outbox_pump_published_total`` counter — PumpReport.published
``outbox_pump_failed_total``   counter — PumpReport.failed
``outbox_pump_inflight_total`` counter — PumpReport.in_flight
``outbox_pump_timedout_total`` counter — PumpReport.timed_out
``registry_entries``           gauge  — total registered entries
``registry_hits_total``        gauge  — registered content hashes that resolve
``registry_misses_total``      gauge  — content hashes probed that did not resolve
``registry_by_type{art_type}`` gauge  — per-artifact-type entry counts
``reconcile_new_total``        counter — NEW candidates
``reconcile_duplicate_total``  counter — DUPLICATE_EXACT candidates
``reconcile_conflict_total``   counter — CONFLICT_SEMANTIC_TO_HASH + CONFLICT_HASH_TO_SEMANTIC
``reconcile_batch_total``      counter — total candidates analyzed
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from ..storage.registry import ArtifactRegistry
from ..worker.jobs import JobExecutionRecord
from ..worker.publish import InMemoryOutbox, PumpReport
from .metrics import MetricsRegistry

__all__ = [
    "collect_job_metrics",
    "collect_outbox_metrics",
    "collect_registry_metrics",
    "collect_reconcile_metrics",
]


def _seconds_delta(end: Any, start: Any | None) -> float:
    """Total seconds between two datetimes; 0.0 when unavailable/mixed."""
    if start is None or end is None:
        return 0.0
    try:
        return (end - start).total_seconds()
    except (TypeError, OverflowError):
        return 0.0


def collect_job_metrics(
    records: Iterable[JobExecutionRecord],
    registry: MetricsRegistry | None = None,
) -> MetricsRegistry:
    """Fill job-layer metrics from ``JobExecutionRecord`` snapshots.

    ``records`` is consumed via ``tuple()`` so generators are safe. Per-status
    counts and the retried count are gauges (current snapshot state); duration
    observations accumulate a min/max/sum/count box (using ``started_at`` ->
    ``finished_at`` so no clock is required).
    """
    reg = registry if registry is not None else MetricsRegistry()
    recs = tuple(records)
    by_status: Counter[str] = Counter()
    retried = 0
    for rec in recs:
        by_status[rec.status.value] += 1
        if rec.attempt_count > 1:
            retried += 1
        reg.observe(
            "jobs_duration_seconds",
            _seconds_delta(rec.finished_at, rec.started_at),
        )
    for status, n in sorted(by_status.items()):
        reg.set_value("jobs_total", n, labels={"status": status})
    reg.set_value("jobs_retried_total", retried)
    return reg


def collect_outbox_metrics(
    outbox: InMemoryOutbox,
    report: PumpReport | None = None,
    registry: MetricsRegistry | None = None,
) -> MetricsRegistry:
    """Fill outbox-layer metrics from an ``InMemoryOutbox`` (+ optional report).

    Rows are read via the immutable ``rows()`` surface. Pump-cycle counts are
    counters (each cycle is an event) — omitted entirely when ``report`` is None
    so re-collection of a bare outbox is idempotent.
    """
    reg = registry if registry is not None else MetricsRegistry()
    rows = tuple(outbox.rows())
    pending = sum(1 for r in rows if r.status == "pending")
    sent = sum(1 for r in rows if r.status == "sent")
    attempts = sum(int(r.attempt_count) for r in rows)
    reg.set_value("outbox_pending", pending)
    reg.set_value("outbox_sent_total", sent)
    reg.set_value("outbox_attempts_total", attempts)
    if report is not None:
        reg.incr("outbox_pump_published_total", int(report.published))
        reg.incr("outbox_pump_failed_total", int(report.failed))
        reg.incr("outbox_pump_inflight_total", int(report.in_flight))
        reg.incr("outbox_pump_timedout_total", int(report.timed_out))
    return reg


#: read-only probe over the underlying sqlite connection (memory or file mode).
_SQL_TOTAL = "SELECT COUNT(*) AS n FROM artifacts"
_SQL_BY_TYPE = (
    "SELECT artifact_type AS t, COUNT(*) AS n FROM artifacts GROUP BY artifact_type"
)
_SQL_DISTINCT_HASHES = "SELECT DISTINCT content_hash AS h FROM artifacts"


def collect_registry_metrics(
    registry: ArtifactRegistry,
    metrics: MetricsRegistry | None = None,
) -> MetricsRegistry:
    """Fill registry-layer metrics from ``ArtifactRegistry`` (read-only).

    The registry is a sqlite-backed store; we read totals / per-type counts /
    distinct hashes through the connection and probe each hash with the public
    ``contains`` surface (hits = resolved registrations, misses = none by
    construction). Any access error degrades to zeroed gauges rather than
    raising into the collector pipeline.
    """
    reg = metrics if metrics is not None else MetricsRegistry()
    total = 0
    by_type: dict[str, int] = {}
    if registry is not None and hasattr(registry, "_conn"):
        try:
            conn = registry._conn
            total = int(conn.execute(_SQL_TOTAL).fetchone()["n"])
            for row in conn.execute(_SQL_BY_TYPE):
                by_type[str(row["t"])] = int(row["n"])
        except Exception:
            total = 0
            by_type = {}
    reg.set_value("registry_entries", total)
    for art_type, n in sorted(by_type.items()):
        reg.set_value("registry_by_type", n, labels={"art_type": art_type})
    hits = misses = 0
    if registry is not None and hasattr(registry, "_conn"):
        try:
            hashes = [str(row["h"]) for row in registry._conn.execute(_SQL_DISTINCT_HASHES)]
            hits = sum(1 for h in hashes if registry.contains(h))
            misses = len(hashes) - hits
        except Exception:
            hits = misses = 0
    reg.set_value("registry_hits_total", hits)
    reg.set_value("registry_misses_total", misses)
    return reg


def collect_reconcile_metrics(
    report: Any,
    registry: MetricsRegistry | None = None,
) -> MetricsRegistry:
    """Fill reconcile-layer counters from a ``ReconcileReport``.

    ``NEW`` / ``DUPLICATE_EXACT`` are counted directly; both ``CONFLICT_*``
    reasons fold into ``reconcile_conflict_total``; ``reconcile_batch_total``
    is the sum of all classes.
    """
    reg = registry if registry is not None else MetricsRegistry()
    try:
        counts: dict[str, int] = {}
        for reason, n in dict(report.counts).items():
            counts[str(reason)] = int(n)
    except Exception:
        counts = {}
    new = counts.get("NEW", 0)
    dup = counts.get("DUPLICATE_EXACT", 0)
    conflict = counts.get("CONFLICT_SEMANTIC_TO_HASH", 0) + counts.get(
        "CONFLICT_HASH_TO_SEMANTIC", 0
    )
    reg.incr("reconcile_new_total", new)
    reg.incr("reconcile_duplicate_total", dup)
    reg.incr("reconcile_conflict_total", conflict)
    reg.incr("reconcile_batch_total", int(sum(counts.values())))
    return reg