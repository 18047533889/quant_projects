"""RBAC — principals, teams, roles, permissions, security classification.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.10 (spec
§24.1). PURE stdlib enums + frozen dataclasses. Permissions are capability-based,
not page-based.

Final authorization is the product of four dimensions:

    Authorization = Role Permission × Team × ResourceScope × SecurityClassification

Each ``Permission`` maps to a minimum ``SecurityClassification``; a principal may
exercise a permission only if their effective classification clears the
permission's requirement.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime

__all__ = [
    "Permission",
    "Role",
    "Team",
    "SecurityClassification",
    "HumanPrincipal",
    "WorkloadPrincipal",
    "ResourceScope",
    "ROLE_PERMISSIONS",
    "PERMISSION_MIN_CLASSIFICATION",
    "ROLE_DEFAULT_CLASSIFICATION",
]


class SecurityClassification(enum.Enum):
    """Sensitivity tiers for factor assets (§5.8)."""

    PUBLIC_METADATA = "PUBLIC_METADATA"
    INTERNAL_RESEARCH = "INTERNAL_RESEARCH"
    CONFIDENTIAL_ALPHA = "CONFIDENTIAL_ALPHA"
    RESTRICTED_RAW_VALUES = "RESTRICTED_RAW_VALUES"
    PRODUCTION_ONLY = "PRODUCTION_ONLY"

    # Ordered by ascending sensitivity — higher rank = more restricted.
    @property
    def rank(self) -> int:
        return {
            SecurityClassification.PUBLIC_METADATA: 0,
            SecurityClassification.INTERNAL_RESEARCH: 1,
            SecurityClassification.CONFIDENTIAL_ALPHA: 2,
            SecurityClassification.RESTRICTED_RAW_VALUES: 3,
            SecurityClassification.PRODUCTION_ONLY: 4,
        }[self]


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


class Role(enum.Enum):
    """Roles (spec §24.1, §5.8)."""

    MEMBER = "MEMBER"
    LEAD = "LEAD"
    CORE = "CORE"
    ADMIN = "ADMIN"
    SERVICE = "SERVICE"


class Team(enum.Enum):
    """Teams (§5.8)."""

    FACTOR_TEAM = "FACTOR_TEAM"
    MODEL_TEAM = "MODEL_TEAM"
    PRODUCTION_TEAM = "PRODUCTION_TEAM"
    EXECUTIVE = "EXECUTIVE"
    PLATFORM_ADMIN = "PLATFORM_ADMIN"


# Minimum security classification required to exercise each permission.
# ``factor:read_formula`` / ``factor:read_raw_values`` require >= CONFIDENTIAL_ALPHA /
# RESTRICTED_RAW_VALUES respectively.
PERMISSION_MIN_CLASSIFICATION: dict[Permission, SecurityClassification] = {
    Permission.FACTOR_READ_SUMMARY: SecurityClassification.PUBLIC_METADATA,
    Permission.FACTOR_READ_EVIDENCE: SecurityClassification.INTERNAL_RESEARCH,
    Permission.FACTOR_READ_FORMULA: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.FACTOR_READ_VALUES: SecurityClassification.RESTRICTED_RAW_VALUES,
    Permission.FACTOR_READ_TREATED_VALUES: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.FACTOR_DOWNLOAD_VALUES: SecurityClassification.RESTRICTED_RAW_VALUES,
    Permission.FACTOR_SUBMIT: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.FACTOR_REPROCESS: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.CLUSTER_READ: SecurityClassification.INTERNAL_RESEARCH,
    Permission.LIBRARY_READ: SecurityClassification.INTERNAL_RESEARCH,
    Permission.LIBRARY_CREATE_CANDIDATE: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.LIBRARY_APPROVE: SecurityClassification.PRODUCTION_ONLY,
    Permission.LIBRARY_PROMOTE: SecurityClassification.PRODUCTION_ONLY,
    Permission.LIBRARY_ROLLBACK: SecurityClassification.PRODUCTION_ONLY,
    Permission.FEATURE_SET_READ: SecurityClassification.INTERNAL_RESEARCH,
    Permission.FEATURE_SET_DOWNLOAD: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.JOB_READ: SecurityClassification.INTERNAL_RESEARCH,
    Permission.JOB_RETRY: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.JOB_CANCEL: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.ARTIFACT_READ: SecurityClassification.INTERNAL_RESEARCH,
    Permission.ARTIFACT_DOWNLOAD: SecurityClassification.CONFIDENTIAL_ALPHA,
    Permission.AUDIT_READ: SecurityClassification.PRODUCTION_ONLY,
    Permission.STANDARDS_READ: SecurityClassification.PUBLIC_METADATA,
    Permission.STANDARDS_EDIT: SecurityClassification.PRODUCTION_ONLY,
    Permission.USER_MANAGE: SecurityClassification.PRODUCTION_ONLY,
    Permission.PERMISSION_MANAGE: SecurityClassification.PRODUCTION_ONLY,
}


# Role -> permission mapping (spec §24.1, §5.8).
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.MEMBER: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.CLUSTER_READ,
            Permission.LIBRARY_READ,
            Permission.FEATURE_SET_READ,
            Permission.JOB_READ,
            Permission.ARTIFACT_READ,
            Permission.STANDARDS_READ,
        }
    ),
    Role.LEAD: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.FACTOR_READ_FORMULA,
            Permission.FACTOR_READ_TREATED_VALUES,
            Permission.CLUSTER_READ,
            Permission.LIBRARY_READ,
            Permission.LIBRARY_CREATE_CANDIDATE,
            Permission.FEATURE_SET_READ,
            Permission.JOB_READ,
            Permission.JOB_RETRY,
            Permission.ARTIFACT_READ,
            Permission.STANDARDS_READ,
        }
    ),
    Role.CORE: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.FACTOR_READ_FORMULA,
            Permission.FACTOR_READ_VALUES,
            Permission.FACTOR_READ_TREATED_VALUES,
            Permission.FACTOR_DOWNLOAD_VALUES,
            Permission.FACTOR_SUBMIT,
            Permission.FACTOR_REPROCESS,
            Permission.CLUSTER_READ,
            Permission.LIBRARY_READ,
            Permission.LIBRARY_CREATE_CANDIDATE,
            Permission.FEATURE_SET_READ,
            Permission.FEATURE_SET_DOWNLOAD,
            Permission.JOB_READ,
            Permission.JOB_RETRY,
            Permission.JOB_CANCEL,
            Permission.ARTIFACT_READ,
            Permission.ARTIFACT_DOWNLOAD,
            Permission.STANDARDS_READ,
        }
    ),
    Role.ADMIN: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.FACTOR_READ_FORMULA,
            Permission.FACTOR_READ_VALUES,
            Permission.FACTOR_READ_TREATED_VALUES,
            Permission.FACTOR_DOWNLOAD_VALUES,
            Permission.FACTOR_SUBMIT,
            Permission.FACTOR_REPROCESS,
            Permission.CLUSTER_READ,
            Permission.LIBRARY_READ,
            Permission.LIBRARY_CREATE_CANDIDATE,
            Permission.LIBRARY_APPROVE,
            Permission.LIBRARY_PROMOTE,
            Permission.LIBRARY_ROLLBACK,
            Permission.FEATURE_SET_READ,
            Permission.FEATURE_SET_DOWNLOAD,
            Permission.JOB_READ,
            Permission.JOB_RETRY,
            Permission.JOB_CANCEL,
            Permission.ARTIFACT_READ,
            Permission.ARTIFACT_DOWNLOAD,
            Permission.AUDIT_READ,
            Permission.STANDARDS_READ,
            Permission.STANDARDS_EDIT,
            Permission.USER_MANAGE,
        }
    ),
    Role.SERVICE: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.FACTOR_READ_FORMULA,
            Permission.FACTOR_READ_VALUES,
            Permission.FACTOR_READ_TREATED_VALUES,
            Permission.FACTOR_DOWNLOAD_VALUES,
            Permission.FACTOR_SUBMIT,
            Permission.FACTOR_REPROCESS,
            Permission.CLUSTER_READ,
            Permission.LIBRARY_READ,
            Permission.LIBRARY_CREATE_CANDIDATE,
            Permission.LIBRARY_APPROVE,
            Permission.LIBRARY_PROMOTE,
            Permission.LIBRARY_ROLLBACK,
            Permission.FEATURE_SET_READ,
            Permission.FEATURE_SET_DOWNLOAD,
            Permission.JOB_READ,
            Permission.JOB_RETRY,
            Permission.JOB_CANCEL,
            Permission.ARTIFACT_READ,
            Permission.ARTIFACT_DOWNLOAD,
            Permission.STANDARDS_READ,
            Permission.STANDARDS_EDIT,
        }
    ),
}


# Default maximum classification a role is trusted with (an upper bound on what
# the principal's effective classification may be for role-based decisions).
ROLE_DEFAULT_CLASSIFICATION: dict[Role, SecurityClassification] = {
    Role.MEMBER: SecurityClassification.INTERNAL_RESEARCH,
    Role.LEAD: SecurityClassification.CONFIDENTIAL_ALPHA,
    Role.CORE: SecurityClassification.RESTRICTED_RAW_VALUES,
    Role.ADMIN: SecurityClassification.PRODUCTION_ONLY,
    Role.SERVICE: SecurityClassification.RESTRICTED_RAW_VALUES,
}


@dataclass(frozen=True)
class ResourceScope:
    """Which resources a permission applies to (product/team/branch scoping)."""

    team: Team | None = None
    product: str | None = None
    branch: str | None = None
    # Empty tuple = all.
    resource_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class HumanPrincipal:
    """A human actor (user id + name + team + role)."""

    principal_id: str
    display_name: str
    team: Team
    role: Role
    resource_scope: ResourceScope = field(default_factory=ResourceScope)
    classification: SecurityClassification = SecurityClassification.INTERNAL_RESEARCH

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("principal_id is required")


@dataclass(frozen=True)
class WorkloadPrincipal:
    """A service / workload actor (OAuth service identity)."""

    principal_id: str
    service_name: str
    team: Team = Team.PLATFORM_ADMIN
    role: Role = Role.SERVICE
    resource_scope: ResourceScope = field(default_factory=ResourceScope)
    classification: SecurityClassification = SecurityClassification.CONFIDENTIAL_ALPHA

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("principal_id is required")
