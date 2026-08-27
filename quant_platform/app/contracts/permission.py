"""Permission vocabulary + resource-scoped grants + pure authorization.

DRAFT. PURE stdlib (frozen dataclasses + enums). A **grant** is a triple
``(principal, permission, resource_scope)``. Access is granted **iff** a grant
covers the requested ``(permission, resource)``. There is NO bare
"RESEARCHER can read all formulas" rule — every grant is scoped to a resource
(team, resource type/id, or COS prefix).

The authorization functions here are pure/stateless (no DB dependency) so the
storage / COS layer can consume them directly.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .security import Role, SecurityClassification, Team

__all__ = [
    "Permission",
    "Role",
    "Team",
    "SecurityClassification",
    "ResourceScope",
    "Grant",
    "authorize",
    "grants_for",
    "WORKLOAD_PREFIX_MATRIX",
    "WORKLOAD_LEAST_PRIVILEGE",
]


class Permission(enum.Enum):
    """Capability-based permission vocabulary (spec §24.1)."""

    FACTOR_READ_SUMMARY = "factor:read_summary"
    FACTOR_READ_EVIDENCE = "factor:read_evidence"
    FACTOR_READ_FORMULA = "factor:read_formula"
    FACTOR_READ_VALUES = "factor:read_raw_values"
    FACTOR_READ_TREATED_VALUES = "factor:read_treated_values"
    FACTOR_DOWNLOAD_VALUES = "factor:download_values"
    FACTOR_SUBMIT = "factor:submit"
    FACTOR_REPROCESS = "factor:reprocess"
    CLUSTER_READ = "cluster:read"
    LIBRARY_READ = "library:read"
    LIBRARY_CREATE_CANDIDATE = "library:create_candidate"
    LIBRARY_APPROVE = "library:approve"
    LIBRARY_PROMOTE = "library:promote"
    LIBRARY_ROLLBACK = "library:rollback"
    FEATURE_SET_READ = "feature_set:read"
    FEATURE_SET_DOWNLOAD = "feature_set:download"
    JOB_READ = "job:read"
    JOB_RETRY = "job:retry"
    JOB_CANCEL = "job:cancel"
    ARTIFACT_READ = "artifact:read"
    ARTIFACT_DOWNLOAD = "artifact:download"
    AUDIT_READ = "audit:read"
    STANDARDS_READ = "standards:read"
    STANDARDS_EDIT = "standards:edit"
    USER_MANAGE = "user:manage"
    PERMISSION_MANAGE = "permission:manage"


@dataclass(frozen=True)
class ResourceScope:
    """Which resources a grant applies to.

    A scope is a conjunction of optional constraints. ``None`` / empty means
    "unconstrained on that axis". ``covers(...)`` is fail-closed: if a scope
    constrains an axis and the requested resource does not satisfy it (or the
    caller did not supply the axis), the scope does NOT cover the resource.

    Wave1-F (action granularity): ``allowed_actions`` carries the object-storage
    actions (``object:read/list/write/delete`` + ``multipart:*`` +
    ``metadata:commit``) this grant is permitted to perform on the scoped
    resources.  ``authorize()`` now requires BOTH the ``action`` AND the path
    scope coverage — a read-only grant on ``factors/`` can read but never write
    within that prefix.
    """

    team: "Team | None" = None
    product: str | None = None
    branch: str | None = None
    # Empty tuple = all resource ids.
    resource_ids: tuple[str, ...] = ()
    # Resource-kind scoping, e.g. "factor" | "library" | "feature_set" | "job" | "artifact".
    resource_type: str | None = None
    # A single specific resource id (e.g. a factor id, library name, feature-set id).
    resource_id: str | None = None
    # COS object prefix (e.g. "factors/", "production/"). A request whose
    # cos_prefix starts with this prefix is covered.
    cos_prefix: str | None = None
    # Wave1-F: object actions permitted on this scope. Empty tuple = NOT
    # constrained on the action axis (legacy behavior); a non-empty tuple means
    # the action MUST be among these.
    allowed_actions: tuple[str, ...] = ()

    def covers(
        self,
        resource_type: str | None = None,
        resource_id: str | None = None,
        cos_prefix: str | None = None,
        team: "Team | None" = None,
        action: str | None = None,
    ) -> bool:
        """True iff this scope covers the requested resource on every axis it
        constrains. Fail-closed: an un-supplied axis that a scope constrains
        yields False."""
        if self.resource_type is not None:
            if resource_type is None or self.resource_type != resource_type:
                return False
        if self.resource_id is not None:
            if resource_id is None or self.resource_id != resource_id:
                return False
        if self.resource_ids:
            if resource_id is None or resource_id not in self.resource_ids:
                return False
        if self.cos_prefix is not None:
            if cos_prefix is None or not cos_prefix.startswith(self.cos_prefix):
                return False
        if self.team is not None:
            if team is None or self.team != team:
                return False
        # Wave1-F action granularity: a scope that constrains actions covers only
        # those actions. A caller-supplied action that is not in the scope's
        # allowed set is NOT covered (fail-closed). (Exactly one of (defined or
        # requested action) being present does NOT indicate an unconstrained
        # action axis — keep semantics explicit.)
        if self.allowed_actions:
            if action is None or action not in self.allowed_actions:
                return False
        return True


@dataclass(frozen=True)
class Grant:
    """A single (principal, permission, resource_scope) grant."""

    principal_id: str
    permission: Permission
    scope: ResourceScope = field(default_factory=ResourceScope)


# ---------------------------------------------------------------------------
# Workload least-privilege model
# ---------------------------------------------------------------------------

# service name -> allowed COS prefixes (for the storage layer to enforce).
WORKLOAD_PREFIX_MATRIX: dict[str, tuple[str, ...]] = {
    "service.factor_ingestion": ("inbox/",),
    "service.factor_engine": ("factors/", "raw/"),
    "service.quant_evaluator": ("evaluations/",),
    "service.factor_preprocess": ("treated/",),
    "service.factor_optimizer": ("optimized/",),
    "service.factor_assets": ("library/",),
    "service.feature_snapshot": ("feature_sets/",),
    "service.model_training": ("model_datasets/",),
    "service.production": ("production/",),
    "service.platform_api": ("",),
}

# service name -> (least-privilege permission set, resource scope).
# Each workload principal carries ONLY the permissions it needs, scoped to the
# COS prefix it is allowed to touch.
WORKLOAD_LEAST_PRIVILEGE: dict[str, tuple[frozenset[Permission], ResourceScope]] = {
    "service.factor_ingestion": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_SUBMIT,
                Permission.JOB_READ,
                Permission.JOB_RETRY,
            }
        ),
        ResourceScope(cos_prefix="inbox/"),
    ),
    "service.factor_engine": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FACTOR_READ_FORMULA,
                Permission.FACTOR_READ_VALUES,
                Permission.FACTOR_SUBMIT,
                Permission.FACTOR_REPROCESS,
                Permission.CLUSTER_READ,
                Permission.LIBRARY_READ,
                Permission.JOB_READ,
                Permission.JOB_RETRY,
                Permission.ARTIFACT_READ,
            }
        ),
        ResourceScope(cos_prefix="factors/"),
    ),
    "service.quant_evaluator": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FACTOR_READ_TREATED_VALUES,
                Permission.FEATURE_SET_READ,
                Permission.JOB_READ,
                Permission.JOB_RETRY,
                Permission.ARTIFACT_READ,
            }
        ),
        ResourceScope(cos_prefix="evaluations/"),
    ),
    "service.factor_preprocess": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FACTOR_READ_VALUES,
                Permission.FACTOR_READ_TREATED_VALUES,
                Permission.FACTOR_SUBMIT,
                Permission.FACTOR_REPROCESS,
                Permission.JOB_READ,
                Permission.JOB_RETRY,
                Permission.ARTIFACT_READ,
            }
        ),
        ResourceScope(cos_prefix="treated/"),
    ),
    "service.factor_optimizer": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FACTOR_READ_FORMULA,
                Permission.FACTOR_READ_TREATED_VALUES,
                Permission.FACTOR_SUBMIT,
                Permission.FACTOR_REPROCESS,
                Permission.JOB_READ,
                Permission.JOB_RETRY,
                Permission.ARTIFACT_READ,
            }
        ),
        ResourceScope(cos_prefix="optimized/"),
    ),
    "service.factor_assets": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FACTOR_READ_FORMULA,
                Permission.FACTOR_READ_TREATED_VALUES,
                Permission.CLUSTER_READ,
                Permission.LIBRARY_READ,
                Permission.LIBRARY_CREATE_CANDIDATE,
                Permission.LIBRARY_APPROVE,
                Permission.LIBRARY_PROMOTE,
                Permission.LIBRARY_ROLLBACK,
                Permission.FEATURE_SET_READ,
                Permission.JOB_READ,
                Permission.ARTIFACT_READ,
            }
        ),
        ResourceScope(cos_prefix="library/"),
    ),
    "service.feature_snapshot": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FEATURE_SET_READ,
                Permission.FEATURE_SET_DOWNLOAD,
                Permission.JOB_READ,
                Permission.ARTIFACT_READ,
            }
        ),
        ResourceScope(cos_prefix="feature_sets/"),
    ),
    "service.model_training": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FEATURE_SET_READ,
                Permission.FEATURE_SET_DOWNLOAD,
                Permission.JOB_READ,
                Permission.JOB_RETRY,
                Permission.ARTIFACT_READ,
                Permission.ARTIFACT_DOWNLOAD,
            }
        ),
        ResourceScope(cos_prefix="model_datasets/"),
    ),
    "service.production": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FACTOR_READ_TREATED_VALUES,
                Permission.FEATURE_SET_READ,
                Permission.JOB_READ,
                Permission.JOB_RETRY,
                Permission.JOB_CANCEL,
                Permission.ARTIFACT_READ,
                Permission.ARTIFACT_DOWNLOAD,
                Permission.AUDIT_READ,
            }
        ),
        ResourceScope(cos_prefix="production/"),
    ),
    "service.platform_api": (
        frozenset(
            {
                Permission.FACTOR_READ_SUMMARY,
                Permission.FACTOR_READ_EVIDENCE,
                Permission.FACTOR_READ_FORMULA,
                Permission.FACTOR_READ_TREATED_VALUES,
                Permission.CLUSTER_READ,
                Permission.LIBRARY_READ,
                Permission.FEATURE_SET_READ,
                Permission.JOB_READ,
                Permission.ARTIFACT_READ,
                Permission.STANDARDS_READ,
            }
        ),
        ResourceScope(cos_prefix=""),
    ),
}


# ---------------------------------------------------------------------------
# Pure authorization
# ---------------------------------------------------------------------------

def _workload_grants(principal) -> list[Grant]:
    perms, scope = WORKLOAD_LEAST_PRIVILEGE.get(
        principal.service_name, (frozenset(), ResourceScope())
    )
    return [Grant(principal.principal_id, perm, scope) for perm in perms]


def _human_grants(principal) -> list[Grant]:
    # Role permissions, each scoped to the principal's resource_scope, plus any
    # explicit grants. There is no unscoped "role can read everything" grant.
    from .rbac import ROLE_PERMISSIONS  # deferred to avoid import cycle

    grants: list[Grant] = [
        Grant(principal.principal_id, perm, principal.resource_scope)
        for perm in ROLE_PERMISSIONS.get(principal.role, frozenset())
    ]
    grants.extend(principal.explicit_grants)
    return grants


def grants_for(principal) -> list[Grant]:
    """All grants held by a principal (pure, stateless).

    - WorkloadPrincipal -> least-privilege grants for its service.
    - HumanPrincipal   -> role permissions scoped to its resource_scope + explicit grants.
    """
    from .principal import HumanPrincipal, WorkloadPrincipal  # deferred import

    if isinstance(principal, WorkloadPrincipal):
        return _workload_grants(principal)
    if isinstance(principal, HumanPrincipal):
        return _human_grants(principal)
    return []


def authorize(
    principal,
    permission: Permission,
    resource_type: str | None = None,
    resource_id: str | None = None,
    cos_prefix: str | None = None,
    team: "Team | None" = None,
    action: str | None = None,
) -> bool:
    """True iff ``principal`` holds a grant covering ``(permission, resource)``
    AND (Wave1-F) the object-storage ``action`` on that resource.

    Fail-closed: a HUMAN/TEAM request must have an explicit scoped grant; it
    NEVER falls back to a service credential. Resource-scope ``allowed_actions``
    (when non-empty) must contain the requested ``action``; a scope that
    constrains actions but does not list it does NOT cover the request. If no
    grant covers the requested resource, this returns False.

    Args:
        principal: HumanPrincipal | WorkloadPrincipal.
        permission: Permission being requested.
        resource_type/resource_id/cos_prefix/team: resource-matching axes.
        action: object-storage action string (e.g. ``"object:read"``). When the
            grant's scope constrains ``allowed_actions``, this must be one of
            them.
    """
    for grant in grants_for(principal):
        if grant.permission == permission and grant.scope.covers(
            resource_type, resource_id, cos_prefix, team, action
        ):
            return True
    return False
