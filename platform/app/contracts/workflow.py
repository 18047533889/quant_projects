"""WorkflowBackend Protocol + job/status types.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.11 (spec §12).
PURE stdlib ``typing.Protocol``. The platform isolates the workflow runtime
(Temporal) behind this protocol; business modules never see Temporal-specific
decorators.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .jobs import JobSpec, JobStatus

__all__ = ["WorkflowBackend", "WorkflowSignal"]


@runtime_checkable
class WorkflowSignal(Protocol):
    """A named signal with a payload (spec §12)."""

    name: str
    payload: dict[str, Any]


@runtime_checkable
class WorkflowBackend(Protocol):
    """Workflow runtime abstraction (spec §12)."""

    def start(self, job_spec: JobSpec) -> str:
        """Start a workflow for ``job_spec``; return the workflow/job id."""
        ...

    def signal(self, job_id: str, signal_name: str, payload: dict[str, Any]) -> None:
        """Send a named signal to a running workflow."""
        ...

    def cancel(self, job_id: str) -> None:
        """Request cancellation of a workflow."""
        ...

    def status(self, job_id: str) -> JobStatus:
        """Return the current execution status of a workflow."""
        ...
