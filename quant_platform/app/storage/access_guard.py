"""Artifact access gate + formula-endpoint authorization guard (spec §23).

PURE, testable implementations of the ``ArtifactAccessGate`` and
``FormulaAccessGuard`` ports declared in
:mod:`quant_platform.app.contracts.storage`.

Design
------
- :class:`ClassificationAccessGate` implements ``ArtifactAccessGate``. It is a
  minimal access-check port: ``authorize(principal, permission, classification,
  resource)`` returns True iff the principal holds ``permission`` AND the
  resource's ``classification`` clears the permission's minimum classification
  (``PERMISSION_MIN_CLASSIFICATION``). The RBAC agent wires this to the full
  platform permission/principal model by supplying a ``permission_check``
  callable (e.g. ``RBAC.has_permission``); when none is supplied it falls back
  to the pure role→permission mapping ``ROLE_PERMISSIONS``.
- :class:`FormulaAccessGuardImpl` implements ``FormulaAccessGuard``. It enforces
  ``GET /factors/{id}/formula``: requires ``Permission.FACTOR_READ_FORMULA`` at
  ``SecurityClassification.CONFIDENTIAL_ALPHA`` (the permission's minimum). A
  denied decision maps to HTTP 403 — never 200 + null. Every invocation
  (allow or deny) records an ``AuditEvent`` carrying ``security_digest`` and
  ``credential_scope_id``.

Only ``quant_platform.app.contracts`` is imported (PURE-DTO boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

from quant_platform.app.contracts.rbac import (
    PERMISSION_MIN_CLASSIFICATION,
    ROLE_PERMISSIONS,
    Permission,
    Role,
    SecurityClassification,
)
from quant_platform.app.contracts.storage import AuditEvent

__all__ = [
    "ClassificationAccessGate",
    "FormulaAccessGuardImpl",
    "AuditLogStore",
    "FORMULA_PERMISSION",
    "FORMULA_MIN_CLASSIFICATION",
]


@runtime_checkable
class AuditLogStore(Protocol):
    """Durable audit-log seam (spec §24.2). Append-only; never mutates."""

    def append(self, event: AuditEvent) -> None:
        """Record a single audit event."""
        ...


def _insert_audit_sql(db: Any, event: AuditEvent) -> None:
    """Persist an ``AuditEvent`` to the ``audit_logs`` table via the ``Db``.

    The audit row is written inside the caller's transaction (spec §24.2:
    authoritative tone) so the guard records the decision atomically with any
    surrounding state change. Cols mirror :func:`_audit_row`.
    """
    row = _audit_row(event)
    cols = tuple(row.keys())
    sql = (
        f"INSERT INTO audit_logs ({','.join(cols)}) VALUES "
        f"({','.join('?' for _ in cols)})"
    )
    db.execute(sql, tuple(row.values()))


def _audit_row(event: AuditEvent) -> dict[str, Any]:
    """Map an ``AuditEvent`` onto the ``audit_logs`` columns."""

    iso = event.timestamp.isoformat()
    if isinstance(event.timestamp, datetime) and event.timestamp.tzinfo is not None:
        iso = event.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "actor_principal_id": event.principal_id,
        "action": event.action,
        "resource_type": "factor",
        "resource_id": _factor_id_from_resource(event.resource),
        "detail_json": _audit_detail(event),
        "occurred_at": iso,
    }


def _factor_id_from_resource(resource: str) -> str | None:
    """Extract a factor id from ``factor:fc-1`` — never from the request."""

    return resource[len("factor:") :] if resource.startswith("factor:") else resource or ""


def _audit_detail(event: AuditEvent) -> str:
    """JSON detail string; every non-timestamp field is projected (Null never
    appears — a None column is emitted with the None key, distinguishing
    "unknown" from "unset")."""

    import json

    payload = {
        "permission": event.permission,
        "resource": event.resource,
        "result": event.result,
        "team": event.team,
        "role": event.role,
        "principal_id": event.principal_id,
        "request_id": event.request_id,
        "security_digest": event.security_digest,
        "credential_scope_id": event.credential_scope_id,
        "action": event.action,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)

# The permission + minimum classification required to read a factor formula.
FORMULA_PERMISSION = Permission.FACTOR_READ_FORMULA
FORMULA_MIN_CLASSIFICATION = PERMISSION_MIN_CLASSIFICATION[FORMULA_PERMISSION]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ClassificationAccessGate:
    """Minimal artifact access gate (implements ``ArtifactAccessGate``).

    ``permission_check`` is an optional callable ``(principal, permission,
    classification) -> bool`` supplied by the RBAC agent to wire the full
    platform permission/principal model. When None, falls back to the pure
    role→permission mapping: a principal carrying a ``role`` attribute is
    authorized iff ``permission in ROLE_PERMISSIONS[role]`` and the resource
    classification clears the permission's minimum.
    """

    permission_check: Callable[[Any, Permission, SecurityClassification], bool] | None = None

    def authorize(
        self,
        principal: Any,
        permission: Permission,
        classification: SecurityClassification,
        resource: Any = None,
    ) -> bool:
        """True iff ``principal`` may exercise ``permission`` at ``classification``."""
        if self.permission_check is not None:
            return bool(self.permission_check(principal, permission, classification))

        # Pure fallback: role-based.
        role = getattr(principal, "role", None)
        if role is None or not isinstance(role, Role):
            return False
        if permission not in ROLE_PERMISSIONS.get(role, frozenset()):
            return False
        minimum = PERMISSION_MIN_CLASSIFICATION.get(permission)
        if minimum is not None and classification.rank < minimum.rank:
            return False
        return True


class FormulaAccessGuardImpl:
    """Enforces ``GET /factors/{id}/formula`` (implements ``FormulaAccessGuard``).

    Requires ``factor:read_formula`` at ``CONFIDENTIAL_ALPHA``. Denied → 403
    (never 200 + null). Records an ``AuditEvent`` on every invocation.
    """

    def __init__(
        self,
        gate: ClassificationAccessGate | None = None,
        *,
        audit: Any = None,
        db: Any = None,
    ) -> None:
        """``audit`` must implement ``audit_port.append`` (spec §24.2). When not
        supplied, falls back to persisting the record to ``db`` (``Db`` Protocol,
        ``audit_logs`` table) when ``db`` is given. ``db`` is optional: the guard
        still enforces authorization and returns the event without persistence."""
        self.gate = gate or ClassificationAccessGate()
        self.audit = audit
        self.db = db

    def guard(
        self,
        *,
        principal: Any,
        factor_id: str,
        request_id: str | None = None,
        security_digest: str | None = None,
        credential_scope_id: str | None = None,
    ) -> AuditEvent:
        """Authorize formula read; returns the audit event for the decision."""
        allowed = self.gate.authorize(
            principal,
            FORMULA_PERMISSION,
            FORMULA_MIN_CLASSIFICATION,
            resource=f"factor:{factor_id}",
        )
        event = AuditEvent(
            principal_id=str(getattr(principal, "principal_id", "unknown")),
            team=getattr(principal, "team", None),
            role=getattr(principal, "role", None),
            permission=FORMULA_PERMISSION.value,
            resource=f"factor:{factor_id}",
            request_id=request_id,
            security_digest=security_digest,
            credential_scope_id=credential_scope_id,
            action="factor:read_formula",
            result="ALLOW" if allowed else "DENY",
            timestamp=datetime.now(timezone.utc),
        )
        self._persist(event)
        return event

    # ------------------------------------------------------------------
    # audit persistence (spec §24.2): append-only, never fail-open
    # ------------------------------------------------------------------

    def _persist(self, event: AuditEvent) -> None:
        """Record the decision. ``audit`` (``AuditLogStore``) wins; else ``db``
        (``audit_logs`` table); else no-op. Persistence failure is recorded and
        surfaced, but never changes the authorization result."""

        if self.audit is not None:
            try:
                self.audit.append(event)
                return
            except Exception as exc:
                raise type(exc)(f"audit store append failed: {exc}") from exc
        if self.db is not None:
            try:
                _insert_audit_sql(self.db, event)
                return
            except Exception as exc:
                raise type(exc)(f"audit_logs insert failed: {exc}") from exc

    def is_allowed(self, event: AuditEvent) -> bool:
        """Convenience: True iff the audit event records an ALLOW."""
        return event.result == "ALLOW"
