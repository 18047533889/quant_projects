"""Health assessment — ``HealthReport`` + ``assess``.

Pure-in-memory, stdlib only. A ``HealthReport`` is an immutable snapshot of one
health evaluation: a top-level ``status`` (HEALTHY / DEGRADED / UNHEALTHY), the
ordered ``checks`` that produced it, and a plain-text ``summary``. ``assess``
composes per-layer checks from three booleans (``job_errors``,
``outbox_pending``, ``registry_ok``) and applies the threshold rule:

* 0 failing checks  -> HEALTHY;
* 1 failing check   -> DEGRADED;
* ``degraded_threshold`` or more failing checks (default 2) -> UNHEALTHY.

Both the threshold and the per-check messages are injectable so callers can tune
the classification without touching the dataclass. ``HealthReport`` freezes
every field on construction and exposes read-only properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "HealthStatus",
    "HealthCheck",
    "HealthReport",
    "assess",
]


class HealthStatus:
    """Enum-like status constants.

    Values are plain ``str`` subtypes so ``report.status == "HEALTHY"`` works
    and downstream string formatting is trivial.
    """

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


_STATUSES = frozenset({"HEALTHY", "DEGRADED", "UNHEALTHY"})


def _coerce_status(value: str) -> str:
    if not isinstance(value, str) or value not in _STATUSES:
        raise ValueError(f"invalid health status: {value!r}")
    return value


@dataclass(frozen=True)
class HealthCheck:
    """One named check's outcome."""

    name: str
    ok: bool
    severity: str = "high"
    message: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", "unnamed")
        if self.severity not in {"low", "medium", "high"}:
            object.__setattr__(self, "severity", "high")


@dataclass(frozen=True)
class HealthReport:
    """Immutable snapshot of one health evaluation.

    ``checks`` is stored as a tuple; ``status`` and ``summary`` are validated /
    auto-filled on construction. Any attempt to mutate a field raises
    ``FrozenInstanceError``.
    """

    status: str
    checks: tuple[HealthCheck, ...]
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_status(self.status))
        object.__setattr__(self, "checks", tuple(self.checks))
        if not self.summary:
            object.__setattr__(self, "summary", _summary(self.status, self.checks))

    # ---- read-only view helpers ----
    @property
    def ok(self) -> bool:
        return self.status == "HEALTHY"

    @property
    def healthy(self) -> bool:
        return self.ok

    @property
    def degraded(self) -> bool:
        return self.status == "DEGRADED"

    @property
    def unhealthy(self) -> bool:
        return self.status == "UNHEALTHY"

    @property
    def failing_checks(self) -> tuple[HealthCheck, ...]:
        return tuple(c for c in self.checks if not c.ok)

    @property
    def check_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks)

    def with_override(self, *, status: str | None = None, summary: str | None = None) -> "HealthReport":
        """Return a new report with ``status`` / ``summary`` overridden."""
        return HealthReport(
            status=status if status is not None else self.status,
            checks=self.checks,
            summary=summary if summary is not None else self.summary,
        )


def assess(
    job_errors: bool = False,
    outbox_pending: bool = False,
    registry_ok: bool = True,
    *,
    degraded_threshold: int = 2,
) -> HealthReport:
    """Assess platform health from three per-layer booleans.

    ``job_errors`` — True when worker jobs are failing/retrying beyond budget.
    ``outbox_pending`` — True when the outbox has backed-up pending events.
    ``registry_ok`` — True when the artifact registry is healthy.

    Failing checks = ``(not registry_ok)`` + ``job_errors`` + ``outbox_pending``.
    ``degraded_threshold`` (default 2) is the number of failing checks that
    flips the status from DEGRADED to UNHEALTHY.
    """
    if degraded_threshold < 1:
        raise ValueError("degraded_threshold must be >= 1")
    checks = (
        HealthCheck(
            name="jobs",
            ok=not job_errors,
            severity="high",
            message="job errors detected" if job_errors else "jobs ok",
        ),
        HealthCheck(
            name="outbox",
            ok=not outbox_pending,
            severity="medium",
            message="outbox pending backlog" if outbox_pending else "outbox ok",
        ),
        HealthCheck(
            name="registry",
            ok=registry_ok,
            severity="high",
            message="registry ok" if registry_ok else "registry degraded",
        ),
    )
    failing = sum(1 for c in checks if not c.ok)
    if failing == 0:
        status = "HEALTHY"
    elif failing < degraded_threshold:
        status = "DEGRADED"
    else:
        status = "UNHEALTHY"
    return HealthReport(status=status, checks=checks, summary=_summary(status, checks))


def _summary(status: str, checks: Iterable[HealthCheck]) -> str:
    names: list[str] = []
    total = 0
    for c in checks:
        total += 1
        if not c.ok:
            names.append(c.name)
    if not names:
        return "all platform checks ok"
    return f"{len(names)} of {total} checks failing: {', '.join(names)}; status={status}"