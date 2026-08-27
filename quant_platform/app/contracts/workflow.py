"""Workflow / Job split — WorkflowBackend Protocol + Workflow types.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.11 (spec §12).
PURE stdlib ``typing.Protocol`` + frozen dataclasses. The platform isolates the
workflow runtime (Temporal) behind this protocol; business modules never see
Temporal-specific decorators.

Workflow vs Job are two layers (§5.4):
- A Factor pipeline **Workflow** contains multiple Jobs/Activities.
- Each Job is one activity, declared via ``JobSpec`` and tracked via
  ``JobRecord``/``JobResult`` (see ``jobs.py``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from .jobs import JobRecord, JobSpec, JobStatus

__all__ = [
    "WorkflowBackend",
    "WorkflowSignal",
    "WorkflowSpec",
    "WorkflowRun",
    "WorkflowStatus",
]


class WorkflowStatus(enum.Enum):
    """Execution status of a whole workflow run (distinct from JobStatus)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    TIMED_OUT = "TIMED_OUT"


@dataclass(frozen=True)
class WorkflowSpec:
    """Immutable declaration of a Factor pipeline workflow.

    A workflow contains multiple jobs/activities. The spec is the *start*
    declaration; it carries no outputs (outputs accrue on per-job ``JobResult``).
    """

    workflow_type: str
    idempotency_key: str
    jobs: tuple[JobSpec, ...] = ()
    inputs: dict[str, Any] = field(default_factory=dict)
    policy_ref: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.workflow_type:
            raise ValueError("workflow_type is required")
        if not self.idempotency_key:
            raise ValueError("idempotency_key is required")


@dataclass(frozen=True)
class WorkflowRun:
    """Runtime facts of a workflow run (per-workflow bookkeeping)."""

    workflow_id: str
    status: WorkflowStatus
    job_records: tuple[JobRecord, ...] = ()
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.workflow_id:
            raise ValueError("workflow_id is required")


@runtime_checkable
class WorkflowSignal(Protocol):
    """A named signal with a payload (spec §12)."""

    name: str
    payload: dict[str, Any]


@runtime_checkable
class WorkflowBackend(Protocol):
    """Workflow runtime abstraction (spec §12)."""

    def start(self, spec: WorkflowSpec) -> str:
        """Start a workflow for ``spec``; return the workflow id."""
        ...

    def signal(self, workflow_id: str, signal_name: str, payload: dict[str, Any]) -> None:
        """Send a named signal to a running workflow."""
        ...

    def cancel(self, workflow_id: str) -> None:
        """Request cancellation of a workflow."""
        ...

    def status(self, workflow_id: str) -> WorkflowStatus:
        """Return the current execution status of a workflow."""
        ...


# Re-export for backward-compat convenience.
__all__ += ["JobSpec", "JobStatus"]
