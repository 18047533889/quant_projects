"""Worker package — pure in-memory worker side of the platform infrastructure
(Transactional Outbox worker loop, JobRunner, Workflow lifecycle).

This package is the worker complement to the durable write-side
``quant_platform.app.outbox``. All state is plain dicts + dataclasses (pure
in-memory). Existing contract symbols (``outbox.Publisher``,
``contracts.workflow.WorkflowBackend``, ``contracts.jobs`` dataclasses) are
reused as-is; no contract export is modified.
"""

from .jobs import (
    ErrorClass,
    JobAttempt,
    JobError,
    JobExecutionRecord,
    JobResult,
    JobRunner,
    JobRunnerError,
    JobSpec,
    JobStatus,
)
from .publish import (
    EventEnvelope,
    InMemoryOutbox,
    OutboxRow,
    OutboxWorkerError,
    PumpReport,
    PublishTimeoutError,
    WorkerLoop,
)
from .workflow import (
    IllegalTransitionError,
    WorkflowLifecycle,
    WorkflowLifecycleError,
    WorkflowRunState,
    WorkflowStatus,
)

__all__ = [
    # publish
    "InMemoryOutbox",
    "OutboxRow",
    "OutboxWorkerError",
    "PublishTimeoutError",
    "PumpReport",
    "WorkerLoop",
    "EventEnvelope",
    # jobs
    "JobRunner",
    "JobRunnerError",
    "JobExecutionRecord",
    "JobError",
    "JobSpec",
    "JobRecord",
    "JobResult",
    "JobStatus",
    "JobAttempt",
    "ErrorClass",
    # workflow
    "WorkflowLifecycle",
    "WorkflowLifecycleError",
    "WorkflowRunState",
    "IllegalTransitionError",
    "WorkflowStatus",
    # re-export for convenient imports
    "JobRecord",
]