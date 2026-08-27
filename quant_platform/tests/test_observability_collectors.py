"""Tests for quant_platform.app.observability.collectors.

Covers each collector wiring a layer's read-only state into a MetricsRegistry:
positive cases (state fully reflected) and negative cases (empty / degraded
inputs produce well-defined zeroed or empty metrics).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quant_platform.app.contracts import (
    ARTIFACT_TYPE_FACTOR_DEFINITION,
    ArtifactRef,
    ErrorClass,
    JobResult,
    JobSpec,
    JobStatus,
)
from quant_platform.app.candidate.ingest import ReconcileReport, reconcile_candidates
from quant_platform.app.storage.registry import ArtifactRegistry
from quant_platform.app.worker.jobs import JobRunner
from quant_platform.app.worker.publish import InMemoryOutbox, PumpReport
from quant_platform.app.observability import (
    MetricsRegistry,
    collect_job_metrics,
    collect_outbox_metrics,
    collect_registry_metrics,
    collect_reconcile_metrics,
)


def _tz(secs: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=secs)


# ===================================================================== #
# collect_job_metrics
# ===================================================================== #
def _job_records() -> tuple:
    runner = JobRunner(worker_id="w1")
    runner.execute(JobSpec(job_type="materialize", idempotency_key="a"), handler=lambda s: JobResult())
    runner.execute(JobSpec(job_type="materialize", idempotency_key="b"), handler=lambda s: JobResult())

    def boom(spec):
        raise RuntimeError("boom")

    runner.execute(JobSpec(job_type="qe", idempotency_key="c"), handler=boom)
    return runner.records()


def test_collect_job_metrics_positive():
    reg = collect_job_metrics(_job_records())
    assert reg.get("jobs_retried_total").value == 0
    # three records all SUCCEEDED/FAILED_TERMINAL -> 2 succeeded + 1 failed
    assert reg.get("jobs_total", labels={"status": "SUCCEEDED"}).value == 2
    assert reg.get("jobs_total", labels={"status": "FAILED_TERMINAL"}).value == 1
    # every record has started+finished -> duration box has 3 observations
    dur = reg.get("jobs_duration_seconds")
    assert dur.count == 3
    assert dur.sum >= 0.0


def test_collect_job_metrics_retry_and_duration():
    runner = JobRunner(worker_id="w1")

    def flaky(spec):
        if spec.idempotency_key == "r1":
            raise __import__("quant_platform.app.worker.jobs", fromlist=["JobError"]).JobError(
                ErrorClass.RETRYABLE_DATABASE
            )
        return JobResult()

    # first attempt fails retryable, second (fresh key) succeeds: r1 has 1 attempt
    runner.execute(JobSpec(job_type="qe", idempotency_key="r1"), handler=flaky)
    runner.execute(JobSpec(job_type="qe", idempotency_key="r2"), handler=flaky)
    reg = collect_job_metrics(runner.records())
    assert reg.get("jobs_retried_total").value == 0  # attempt_count == 1 for all
    assert reg.get("jobs_total", labels={"status": "FAILED_RETRYABLE"}).value == 1
    assert reg.get("jobs_total", labels={"status": "SUCCEEDED"}).value == 1


def test_collect_job_metrics_negative_empty():
    reg = collect_job_metrics([])
    assert reg.get("jobs_retried_total").value == 0
    assert reg.get("jobs_duration_seconds") is None  # no observations recorded


def test_collect_job_metrics_uses_provided_registry():
    base = MetricsRegistry()
    base.incr("pre", 1)
    reg = collect_job_metrics(_job_records(), registry=base)
    assert reg is base
    assert base.get("pre").value == 1.0
    assert base.get("jobs_total", labels={"status": "SUCCEEDED"}).value == 2


# ===================================================================== #
# collect_outbox_metrics
# ===================================================================== #
def _envelope(eid: str, key: str):
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


def _outbox(rows: list[str] = ("e1", "e2", "e3")) -> InMemoryOutbox:
    outbox = InMemoryOutbox()
    for i, eid in enumerate(rows):
        outbox.append(_envelope(eid, f"ik-{i}"))
    return outbox


def test_collect_outbox_metrics_positive():
    outbox = _outbox()
    outbox.mark_sent("e1")
    outbox.mark_sent("e2")
    reg = collect_outbox_metrics(outbox)
    assert reg.get("outbox_pending").value == 1
    assert reg.get("outbox_sent_total").value == 2
    assert reg.get("outbox_attempts_total").value == 0
    # no pump report passed -> no pump counters
    assert reg.get("outbox_pump_published_total") is None


def test_collect_outbox_metrics_attempts_and_report():
    outbox = _outbox(rows=("e1", "e2"))
    outbox.mark_attempt("e1")
    outbox.mark_attempt("e1")
    reg = collect_outbox_metrics(
        outbox, report=PumpReport(published=1, failed=1, in_flight=1, timed_out=1)
    )
    assert reg.get("outbox_attempts_total").value == 2
    assert reg.get("outbox_pump_published_total").value == 1
    assert reg.get("outbox_pump_failed_total").value == 1
    assert reg.get("outbox_pump_inflight_total").value == 1
    assert reg.get("outbox_pump_timedout_total").value == 1


def test_collect_outbox_metrics_negative_empty():
    reg = collect_outbox_metrics(InMemoryOutbox())
    assert reg.get("outbox_pending").value == 0
    assert reg.get("outbox_sent_total").value == 0
    assert reg.get("outbox_attempts_total").value == 0


# ===================================================================== #
# collect_registry_metrics
# ===================================================================== #
def _artifact(i: int) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"aid-{i}",
        artifact_type=ARTIFACT_TYPE_FACTOR_DEFINITION,
        schema_version="1.0",
        content_hash=f"{i:064x}",
        storage_uri="cos://bucket/x",
        size_bytes=10,
        created_at=datetime.now(timezone.utc),
        producer_type="miner",
        producer_version="1.0",
    )


def test_collect_registry_metrics_positive():
    with ArtifactRegistry() as reg_ar:
        reg_ar.register(_artifact(1))
        reg_ar.register(_artifact(2))
        reg = collect_registry_metrics(reg_ar)
        assert reg.get("registry_entries").value == 2
        assert reg.get("registry_by_type", labels={"art_type": ARTIFACT_TYPE_FACTOR_DEFINITION}).value == 2
        assert reg.get("registry_hits_total").value == 2  # every registered hash resolves
        assert reg.get("registry_misses_total").value == 0


def test_collect_registry_metrics_type_distribution():
    from quant_platform.app.contracts import ARTIFACT_TYPE_FACTOR_CANDIDATE, ARTIFACT_TYPE_FEATURE_SET

    with ArtifactRegistry() as reg_ar:
        reg_ar.register(_artifact(1))
        cand = _artifact(2)
        cand = ArtifactRef(
            artifact_id="aid-2",
            artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE,
            schema_version="1.0",
            content_hash=f"{2:064x}",
            storage_uri="cos://bucket/y",
            size_bytes=5,
            created_at=datetime.now(timezone.utc),
            producer_type="miner",
            producer_version="1.0",
        )
        reg_ar.register(cand)
        feat = _artifact(3)
        feat = ArtifactRef(
            artifact_id="aid-3",
            artifact_type=ARTIFACT_TYPE_FEATURE_SET,
            schema_version="1.0",
            content_hash=f"{3:064x}",
            storage_uri="cos://bucket/z",
            size_bytes=5,
            created_at=datetime.now(timezone.utc),
            producer_type="pipeline",
            producer_version="1.0",
        )
        reg_ar.register(feat)
        reg = collect_registry_metrics(reg_ar)
        assert reg.get("registry_entries").value == 3
        assert reg.get("registry_by_type", labels={"art_type": ARTIFACT_TYPE_FACTOR_DEFINITION}).value == 1
        assert reg.get("registry_by_type", labels={"art_type": ARTIFACT_TYPE_FACTOR_CANDIDATE}).value == 1
        assert reg.get("registry_by_type", labels={"art_type": ARTIFACT_TYPE_FEATURE_SET}).value == 1


def test_collect_registry_metrics_negative_empty():
    with ArtifactRegistry() as reg_ar:
        reg = collect_registry_metrics(reg_ar)
        assert reg.get("registry_entries").value == 0
        assert reg.get("registry_hits_total").value == 0
        assert reg.get("registry_misses_total").value == 0


# ===================================================================== #
# collect_reconcile_metrics
# ===================================================================== #
def _manifest(i: int, content: str | None = None, semantic: str | None = None):
    from quant_platform.app.contracts import FactorCandidateManifest

    return FactorCandidateManifest(
        schema_version="1.0",
        candidate_id=f"fc-{i}",
        submitted_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        submitted_by="miner",
        generator_type="mine",
        generator_version="1.0",
        market="CN",
        frequency="daily",
        formula_language="recipe",
        factor_spec_uri="cos://bucket/x",
        factor_spec_sha256=content or f"{i:064x}",
        semantic_family_hint=semantic or f"sem-{i}",
    )


def _report() -> ReconcileReport:
    # one NEW, one DUPLICATE_EXACT, one CONFLICT_SEMANTIC_TO_HASH
    known = [
        {"content_hash": f"{2:064x}", "semantic_hash": "sem-2"},
        {"content_hash": f"{3:064x}", "semantic_hash": "sem-x"},  # different semantic, same content below
    ]
    candidates = [
        _manifest(1),  # NEW
        _manifest(2),  # DUPLICATE_EXACT
        _manifest(4, content=f"{2:064x}", semantic="sem-4"),  # same content as known, diff semantic
    ]
    return reconcile_candidates(candidates, known_registry=known)


def test_collect_reconcile_metrics_positive():
    report = _report()
    reg = collect_reconcile_metrics(report)
    assert reg.get("reconcile_new_total").value == 1
    assert reg.get("reconcile_duplicate_total").value == 1
    assert reg.get("reconcile_conflict_total").value == 1
    assert reg.get("reconcile_batch_total").value == 3


def test_collect_reconcile_metrics_negative_all_new():
    from quant_platform.app.candidate.ingest import reconcile_candidates as rc

    report = rc([_manifest(1), _manifest(2)], known_registry=[])
    reg = collect_reconcile_metrics(report)
    assert reg.get("reconcile_new_total").value == 2
    assert reg.get("reconcile_duplicate_total").value == 0
    assert reg.get("reconcile_conflict_total").value == 0
    assert reg.get("reconcile_batch_total").value == 2


def test_collect_reconcile_metrics_negative_bare_dict():
    # a plain dict-shaped "report" still yields defined zeros, never an exception
    reg = collect_reconcile_metrics({"counts": {}, "conflicts": (), "batch_fingerprint": ""})
    assert reg.get("reconcile_new_total").value == 0
    assert reg.get("reconcile_duplicate_total").value == 0
    assert reg.get("reconcile_conflict_total").value == 0
    assert reg.get("reconcile_batch_total").value == 0