"""Workflow run lifecycle — enforces WorkflowStatus legal transitions.

Pure in-memory lifecycle state machine over ``contracts.workflow.WorkflowStatus``
(spec §12). ``PENDING -> RUNNING -> SUCCEEDED/FAILED/CANCELED/TIMED_OUT`` with
strict transition guards: any illegal jump (e.g. PENDING -> SUCCEEDED, or a
second RUNNING) raises ``IllegalTransitionError``. ``WorkflowBackend`` (the
contracts Protocol) is left untouched so existing export symbols remain intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..contracts.workflow import WorkflowRun, WorkflowStatus

__all__ = [
    "WorkflowLifecycleError",
    "IllegalTransitionError",
    "WorkflowRunState",
    "WorkflowLifecycle",
    "WorkflowStatus",
]


class WorkflowLifecycleError(Exception):
    """Base error raised by the workflow lifecycle state machine."""


class IllegalTransitionError(WorkflowLifecycleError):
    """A ``WorkflowStatus`` transition violated the allowed state machine."""


_TERMINAL = frozenset(
    {
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELED,
        WorkflowStatus.TIMED_OUT,
    }
)


def _datetime_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class WorkflowRunState:
    """Immutable snapshot of a workflow run's in-memory state."""

    workflow_id: str
    status: WorkflowStatus
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.workflow_id:
            raise ValueError("workflow_id is required")

    def to_workflow_run(self) -> WorkflowRun:
        """Project onto the public ``WorkflowRun`` contract dataclass."""
        return WorkflowRun(
            workflow_id=self.workflow_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


class WorkflowLifecycle:
    """The workflow run state machine, keyed by ``workflow_id``."""

    # Legal transitions: start -> running -> terminal (and running -> cancel_requested).
    _ALLOWED = {
        WorkflowStatus.PENDING: frozenset({WorkflowStatus.RUNNING, WorkflowStatus.FAILED}),
        WorkflowStatus.RUNNING: frozenset(
            {
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELED,
                WorkflowStatus.TIMED_OUT,
            }
        ),
    }

    def __init__(self) -> None:
        self._runs: dict[str, WorkflowRunState] = {}

    # ---- queries ----
    def get(self, workflow_id: str) -> WorkflowRunState | None:
        return self._runs.get(workflow_id)

    def runs(self) -> tuple[WorkflowRunState, ...]:
        return tuple(self._runs.values())

    def __len__(self) -> int:
        return len(self._runs)

    # ---- mutations ----
    def start(self, workflow_id: str) -> WorkflowRunState:
        """Register a new run in ``PENDING``."""
        if not workflow_id:
            raise ValueError("workflow_id is required")
        if workflow_id in self._runs:
            raise IllegalTransitionError(
                f"workflow {workflow_id!r} already exists — cannot start twice"
            )
        now = _datetime_now()
        state = WorkflowRunState(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            created_at=now,
        )
        self._runs[workflow_id] = state
        return state

    def start_run(self, workflow_id: str) -> WorkflowRunState:
        """Move a pending run to ``RUNNING``."""
        return self._transition(workflow_id, WorkflowStatus.RUNNING)

    def succeed(self, workflow_id: str) -> WorkflowRunState:
        return self._transition(workflow_id, WorkflowStatus.SUCCEEDED)

    def fail(self, workflow_id: str) -> WorkflowRunState:
        return self._transition(workflow_id, WorkflowStatus.FAILED)

    def cancel(self, workflow_id: str) -> WorkflowRunState:
        return self._transition(workflow_id, WorkflowStatus.CANCELED)

    def time_out(self, workflow_id: str) -> WorkflowRunState:
        return self._transition(workflow_id, WorkflowStatus.TIMED_OUT)

    # ---- internals ----
    def _transition(self, workflow_id: str, target: WorkflowStatus) -> WorkflowRunState:
        current = self._runs.get(workflow_id)
        if current is None:
            raise IllegalTransitionError(
                f"workflow {workflow_id!r} has no run — must start() first"
            )
        allowed = self._ALLOWED.get(current.status)
        if allowed is None or target not in allowed:
            raise IllegalTransitionError(
                f"illegal transition {current.status.value} -> {target.value} "
                f"for workflow {workflow_id!r}"
            )
        now = _datetime_now()
        finished_at = now if target in _TERMINAL else None
        started_at = (
            current.started_at
            if current.started_at is not None
            else (now if target is WorkflowStatus.RUNNING else None)
        )
        state = WorkflowRunState(
            workflow_id=workflow_id,
            status=target,
            created_at=current.created_at,
            started_at=started_at,
            finished_at=finished_at,
        )
        self._runs[workflow_id] = state
        return state