"""RBAC — teams, roles, security classification, and the grant-based model.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.10 (spec
§24.1). PURE stdlib enums + frozen dataclasses.

Authorization model (upgraded to User/Team/Role/Permission/ResourceScope):

    Authorization = grant(principal, permission, resource_scope) covers (permission, resource)

A **grant** is a triple ``(principal, permission, resource_scope)``. Access is
granted **iff** a grant covers the requested ``(permission, resource)``. There
is NO bare "RESEARCHER can read all formulas" rule — every grant is scoped.

This module keeps the legacy role/classification vocabulary (``Team``, ``Role``,
``SecurityClassification``, ``ROLE_PERMISSIONS``,
``PERMISSION_MIN_CLASSIFICATION``, ``ROLE_DEFAULT_CLASSIFICATION``) and
re-exports the grant-based model from ``permission.py`` / ``principal.py`` so
existing callers keep working.
"""

from __future__ import annotations

from .permission import (
    WORKLOAD_LEAST_PRIVILEGE,
    WORKLOAD_PREFIX_MATRIX,
    Grant,
    Permission,
    ResourceScope,
    authorize,
    grants_for,
)
from .principal import (
    PRINCIPAL_TYPE_HUMAN,
    PRINCIPAL_TYPE_SERVICE,
    WORKLOAD_SERVICE_NAMES,
    HumanPrincipal,
    WorkloadPrincipal,
)
from .security import Role, SecurityClassification, Team

__all__ = [
    "Permission",
    "Role",
    "Team",
    "SecurityClassification",
    "HumanPrincipal",
    "WorkloadPrincipal",
    "ResourceScope",
    "Grant",
    "ROLE_PERMISSIONS",
    "PERMISSION_MIN_CLASSIFICATION",
    "ROLE_DEFAULT_CLASSIFICATION",
    "authorize",
    "grants_for",
    "WORKLOAD_PREFIX_MATRIX",
    "WORKLOAD_LEAST_PRIVILEGE",
    "WORKLOAD_SERVICE_NAMES",
    "PRINCIPAL_TYPE_HUMAN",
    "PRINCIPAL_TYPE_SERVICE",
]


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


# Role -> permission mapping (spec §24.1, §5.8). These are the DEFAULT
# permissions a role holds; each is still scoped to the principal's
# ``resource_scope`` when turned into a grant (see ``permission.grants_for``).
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
