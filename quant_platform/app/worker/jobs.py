"""In-memory JobRunner — executes JobSpec handlers and records JobRecord.

Pure in-memory (dict + dataclass). Implements the worker-side job execution
bookkeeping of ``quant_platform/app/contracts/jobs.py`` without a scheduler:

- ``JobRunner.execute(spec, handler=None)`` runs ``handler(spec)`` (the callable
  worker) and returns a ``JobRecord`` whose status accrues to terminal state.
- Success -> ``SUCCEEDED``; a raised exception -> ``FAILED_RETRYABLE`` when an
  ``ErrorClass.retryable`` is attached (via ``JobError``) or
  ``FAILED_TERMINAL`` otherwise; ``attempt_count`` grows per attempt.
- Idempotency: submitting the same ``spec.idempotency_key`` again returns the
  *existing* terminal record and never re-runs the handler.

Contract alignment: ``JobRunner`` stores results by ``idempotency_key`` (which is
what spec §42 dedupes on). ``WorkflowBackend`` (contracts/workflow.py) remains a
pure Protocol — the runner implements job execution *for* a backend.

Timing fields use ``datetime.now(timezone.utc)`` (aware) to satisfy contenthash
strictness. Attempts are recorded as ``JobAttempt(worker_id)``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from ..contracts.jobs import (
    ErrorClass,
    JobAttempt,
    JobRecord,
    JobResult,
    JobSpec,
    JobStatus,
)

__all__ = [
    "JobRunnerError",
    "JobError",
    "JobRunner",
    "JobExecutionRecord",
    "JobStatus",
    "JobResult",
    "ErrorClass",
    "JobAttempt",
]


class JobRunnerError(Exception):
    """Base error raised by ``JobRunner``."""


class JobError(Exception):
    """Handler exception carrying a spec §43 ``ErrorClass``.

    ``JobRunner`` maps ``ErrorClass.retryable`` to ``FAILED_RETRYABLE`` and
    everything else to ``FAILED_TERMINAL``.
    """

    def __init__(self, error_class: ErrorClass, message: str | None = None) -> None:
        super().__init__(message or error_class.value)
        self.error_class = error_class


@dataclass(frozen=True)
class JobExecutionRecord:
    """One job's runtime bookkeeping as accrued by ``JobRunner``.

    Deserialized on demand from the internal model; mirrors ``JobRecord`` but
    carries the ``job_type`` and ``idempotency_key`` strings for lookup.
    """

    job_type: str
    idempotency_key: str
    status: JobStatus
    current_stage: str | None = None
    progress: float = 0.0
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt_count: int = 0
    attempts: tuple[JobAttempt, ...] = ()
    error_class: ErrorClass | None = None
    result: JobResult | None = None

    def __post_init__(self) -> None:
        if not self.job_type:
            raise ValueError("job_type is required")
        if not self.idempotency_key:
            raise ValueError("idempotency_key is required")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be in [0.0, 1.0]")

    def to_job_record(self) -> JobRecord:
        """Project onto the public ``JobRecord`` contract (frozen dataclass)."""
        return JobRecord(
            status=self.status,
            current_stage=self.current_stage,
            progress=self.progress,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            attempt_count=self.attempt_count,
            attempts=self.attempts,
            error_class=self.error_class,
        )


class _InternalJob:
    """Mutable internal job state; exposed as frozen ``JobExecutionRecord``."""

    __slots__ = (
        "job_type",
        "idempotency_key",
        "status",
        "current_stage",
        "progress",
        "created_at",
        "started_at",
        "finished_at",
        "attempts",
        "error_class",
        "result",
        "summary",
    )

    def __init__(self, spec: JobSpec) -> None:
        now = datetime.now(timezone.utc)
        self.job_type = spec.job_type
        self.idempotency_key = spec.idempotency_key
        self.status = JobStatus.PENDING
        self.current_stage = None
        self.progress = 0.0
        self.created_at = now
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.attempts: list[JobAttempt] = []
        self.error_class: ErrorClass | None = None
        self.result: JobResult | None = None
        self.summary = ""

    def freeze(self) -> JobExecutionRecord:
        return JobExecutionRecord(
            job_type=self.job_type,
            idempotency_key=self.idempotency_key,
            status=self.status,
            current_stage=self.current_stage,
            progress=self.progress,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            attempt_count=len(self.attempts),
            attempts=tuple(self.attempts),
            error_class=self.error_class,
            result=self.result,
        )


class JobRunner:
    """In-memory job executor with attempt + idempotency bookkeeping.

    Registries are plain dicts; ``JobRecord`` is exposed via the public frozen
    dataclass from ``contracts/jobs.py`` (no new contract symbols).
    """

    def __init__(self, worker_id: str = "qrpp4-worker") -> None:
        if not worker_id:
            raise ValueError("worker_id is required")
        self.worker_id = worker_id
        self._jobs: dict[str, _InternalJob] = {}  # by idempotency_key
        self._records: dict[str, JobExecutionRecord] = {}  # frozen snapshot
        self._executions: list[JobExecutionRecord] = []  # append-only log
        self._default_handler: Callable[[JobSpec], JobResult] | None = None

    # ---- queries ----
    def get(self, idempotency_key: str) -> JobExecutionRecord | None:
        return self._records.get(idempotency_key)

    def records(self) -> tuple[JobExecutionRecord, ...]:
        return tuple(self._records.values())

    def executions(self) -> tuple[JobExecutionRecord, ...]:
        return tuple(self._executions)

    def __len__(self) -> int:
        return len(self._jobs)

    # ---- execution ----
    def execute(
        self,
        spec: JobSpec,
        handler: Callable[[JobSpec], JobResult] | None = None,
    ) -> JobExecutionRecord:
        """Execute ``spec`` via ``handler`` and return the resulting record.

        - Handler is optional; ``None`` uses the registry-wide default.
        - Idempotent by ``spec.idempotency_key``: a repeat submit returns the
          existing record without re-running the handler.
        - The active default handler may be changed at runtime.
        """
        if spec is None or not isinstance(spec, JobSpec):
            raise JobRunnerError("execute requires a JobSpec")
        job = self._jobs.get(spec.idempotency_key)
        if job is None:
            job = _InternalJob(spec)
            self._jobs[spec.idempotency_key] = job
        elif job.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED_RETRYABLE,
            JobStatus.FAILED_TERMINAL,
            JobStatus.TIMED_OUT,
            JobStatus.CANCELLED,
        }:
            # Idempotent resubmission: return the existing terminal record
            # without re-running the handler.
            existing = self._records.get(spec.idempotency_key)
            if existing is not None:
                return existing
            snapshot = job.freeze()
            self._records[spec.idempotency_key] = snapshot
            return snapshot
        else:
            # In-flight (RUNNING/PENDING): refuse to double-execute.
            raise JobRunnerError(
                f"job {spec.idempotency_key!r} already RUNNING/PENDING — "
                "in-memory runner executes one attempt at a time"
            )

        handler = handler or self._read_default_handler()
        if handler is None:
            self._fail(job, ErrorClass.CAPABILITY, "no handler registered")
            return self._snapshot(job)

        now = datetime.now(timezone.utc)
        job.status = JobStatus.RUNNING
        if job.started_at is None:
            job.started_at = now
        job.current_stage = "handling"

        attempt = JobAttempt(worker_id=self.worker_id, heartbeat=now)
        job.attempts.append(attempt)
        # Publish the RUNNING snapshot immediately so an idempotency-key repeat
        # submit during execution observes the in-flight record (not PENDING).
        self._records[job.idempotency_key] = job.freeze()
        self._executions.append(job.freeze())

        try:
            raw = handler(spec)
            result = raw if isinstance(raw, JobResult) else JobResult()
            job.result = result
            job.status = JobStatus.SUCCEEDED
            job.progress = 1.0
            job.finished_at = datetime.now(timezone.utc)
            job.current_stage = None
            job.summary = result.summary
        except JobError as exc:
            self._fail(job, exc.error_class)
        except Exception as exc:  # unknown error -> terminal failure
            self._fail(job, ErrorClass.CAPABILITY, str(exc))

        return self._snapshot(job)

    # ---- registry hooks ----
    def register(self, handler: Callable[[JobSpec], JobResult]) -> None:
        """Register the default handler for jobs without an explicit handler."""
        if not callable(handler):
            raise JobRunnerError("handler must be callable")
        self._default_handler = handler

    # ---- internals ----
    def _read_default_handler(self) -> Callable[[JobSpec], JobResult] | None:
        return self._default_handler

    def _fail(self, job: _InternalJob, error_class: ErrorClass, message: str | None = None) -> None:
        job.status = (
            JobStatus.FAILED_RETRYABLE if error_class.retryable else JobStatus.FAILED_TERMINAL
        )
        job.error_class = error_class
        if message:
            job.summary = message
        job.finished_at = datetime.now(timezone.utc)
        job.current_stage = None

    def _snapshot(self, job: _InternalJob) -> JobExecutionRecord:
        rec = job.freeze()
        self._records[job.idempotency_key] = rec
        self._executions.append(rec)
        return self._records[job.idempotency_key]


# ``_default_handler`` is an ordinary instance attribute (set in ``__init__``).