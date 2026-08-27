"""RBAC permission vocabulary + roles.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.10 (spec
§24.1). PURE stdlib enums. Permissions are capability-based, not page-based.
"""

from __future__ import annotations

import enum

__all__ = ["Permission", "Role", "ROLE_PERMISSIONS"]


class Permission(enum.Enum):
    """Capability-based permission vocabulary (spec §24.1)."""

    FACTOR_READ_SUMMARY = "factor:read_summary"
    FACTOR_READ_EVIDENCE = "factor:read_evidence"
    FACTOR_READ_FORMULA = "factor:read_formula"
    FACTOR_READ_VALUES = "factor:read_values"
    FACTOR_DOWNLOAD_VALUES = "factor:download_values"
    FACTOR_REPROCESS = "factor:reprocess"
    CLUSTER_READ = "cluster:read"
    LIBRARY_PROMOTE = "library:promote"
    JOB_RETRY = "job:retry"
    USER_MANAGE = "user:manage"
    STANDARDS_EDIT = "standards:edit"


class Role(enum.Enum):
    """Roles (spec §24.1)."""

    VIEWER = "VIEWER"
    RESEARCHER = "RESEARCHER"
    CORE_RESEARCHER = "CORE_RESEARCHER"
    OPS = "OPS"
    ADMIN = "ADMIN"


# Suggested role -> permission mapping (spec §24.1). DRAFT — to be confirmed.
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.CLUSTER_READ,
        }
    ),
    Role.RESEARCHER: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.FACTOR_READ_FORMULA,
            Permission.FACTOR_READ_VALUES,
            Permission.CLUSTER_READ,
        }
    ),
    Role.CORE_RESEARCHER: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.FACTOR_READ_FORMULA,
            Permission.FACTOR_READ_VALUES,
            Permission.FACTOR_DOWNLOAD_VALUES,
            Permission.FACTOR_REPROCESS,
            Permission.CLUSTER_READ,
            Permission.LIBRARY_PROMOTE,
        }
    ),
    Role.OPS: frozenset(
        {
            Permission.FACTOR_READ_SUMMARY,
            Permission.FACTOR_READ_EVIDENCE,
            Permission.CLUSTER_READ,
            Permission.JOB_RETRY,
        }
    ),
    Role.ADMIN: frozenset(Permission),
}
