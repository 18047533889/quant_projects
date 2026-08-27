"""RBAC — role→permission resolution + min-classification enforcement.

QRP-P1. Resolves a principal's effective permissions from the contracts
``ROLE_PERMISSIONS`` + ``PERMISSION_MIN_CLASSIFICATION`` mappings, plus per-user
override rows in the ``permissions`` table. Enforces ``require(permission,
resource_security_classification)``.

Authorization model (contracts §5.8):

    Authorization = Role Permission × Team × ResourceScope × SecurityClassification

A principal may exercise a permission only if:
1. the permission is in their effective set (role permissions ∪ per-user
   overrides), AND
2. the resource's security classification clears the permission's minimum
   classification (``PERMISSION_MIN_CLASSIFICATION[permission]``).
"""

from __future__ import annotations

from typing import Any, Iterable

from quant_platform.app.contracts.rbac import (
    PERMISSION_MIN_CLASSIFICATION,
    ROLE_PERMISSIONS,
    Permission,
    Role,
    SecurityClassification,
)

__all__ = [
    "AuthorizationError",
    "PermissionDenied",
    "RBAC",
    "effective_permissions",
    "resolve_principal_permissions",
]


class AuthorizationError(Exception):
    """Base for authorization failures."""


class PermissionDenied(AuthorizationError):
    """Raised when a principal lacks a required permission / classification."""


def _role_permissions(role: Role) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(role, frozenset())


def resolve_principal_permissions(
    role: Role,
    override_permissions: Iterable[str] = (),
) -> frozenset[Permission]:
    """Effective permission set = role permissions ∪ per-user overrides.

    ``override_permissions`` are raw permission strings from the ``permissions``
    table (e.g. ``"factor:read_formula"``). Unknown strings are ignored.
    """
    result = set(_role_permissions(role))
    for raw in override_permissions:
        try:
            result.add(Permission(raw))
        except ValueError:
            # Unknown permission string in the override table — ignore.
            continue
    return frozenset(result)


def effective_permissions(
    role: Role,
    override_permissions: Iterable[str] = (),
) -> frozenset[Permission]:
    """Alias for ``resolve_principal_permissions``."""
    return resolve_principal_permissions(role, override_permissions)


class RBAC:
    """Enforcement engine backed by the metadata DB.

    ``db`` must implement the ``Db`` Protocol. Per-user override rows are read
    from the ``permissions`` table (``principal_id``, ``permission``).
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    def _overrides(self, principal_id: str) -> list[str]:
        rows = self._db.query(
            "SELECT permission FROM permissions WHERE principal_id = ?",
            (principal_id,),
        )
        return [r["permission"] for r in rows]

    def _roles(self, principal_id: str) -> list[Role]:
        rows = self._db.query(
            "SELECT role FROM roles WHERE principal_id = ?",
            (principal_id,),
        )
        roles: list[Role] = []
        for r in rows:
            try:
                roles.append(Role(r["role"]))
            except ValueError:
                continue
        return roles

    def permissions_for(self, principal_id: str) -> frozenset[Permission]:
        """Effective permission set for a principal (union across roles + overrides)."""
        result: set[Permission] = set()
        for role in self._roles(principal_id):
            result |= set(_role_permissions(role))
        result |= set(resolve_principal_permissions(Role.MEMBER, self._overrides(principal_id)))
        return frozenset(result)

    def has_permission(
        self,
        principal_id: str,
        permission: Permission,
        resource_classification: SecurityClassification | None = None,
    ) -> bool:
        """True iff the principal holds ``permission`` and (if a resource
        classification is given) clears its minimum classification."""
        if permission not in self.permissions_for(principal_id):
            return False
        if resource_classification is not None:
            minimum = PERMISSION_MIN_CLASSIFICATION.get(permission)
            # The permission requires the resource to be AT LEAST as sensitive as
            # its minimum classification (e.g. factor:read_formula requires >=
            # CONFIDENTIAL_ALPHA). A resource below that floor is denied.
            if minimum is not None and resource_classification.rank < minimum.rank:
                return False
        return True

    def require(
        self,
        principal_id: str,
        permission: Permission,
        resource_classification: SecurityClassification | None = None,
    ) -> None:
        """Raise ``PermissionDenied`` unless the principal is authorized."""
        if not self.has_permission(principal_id, permission, resource_classification):
            raise PermissionDenied(
                f"principal {principal_id!r} lacks permission {permission.value!r}"
                + (
                    f" at classification {resource_classification.value!r}"
                    if resource_classification is not None
                    else ""
                )
            )
