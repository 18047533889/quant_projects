"""RBAC tests — role permission resolution, min-classification, per-user override."""

from __future__ import annotations

import pytest

from quant_platform.app.contracts.rbac import (
    PERMISSION_MIN_CLASSIFICATION,
    Permission,
    Role,
    SecurityClassification,
)
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.security.rbac import (
    PermissionDenied,
    RBAC,
    effective_permissions,
    resolve_principal_permissions,
)


def _seed_principal(db, principal_id, role, overrides=()):
    db.execute(
        "INSERT INTO principals (principal_id, principal_type, display_name) VALUES (?, ?, ?)",
        (principal_id, "HUMAN", principal_id),
    )
    db.execute(
        "INSERT INTO roles (principal_id, role) VALUES (?, ?)",
        (principal_id, role.value),
    )
    for perm in overrides:
        db.execute(
            "INSERT INTO permissions (principal_id, permission) VALUES (?, ?)",
            (principal_id, perm.value),
        )


def test_role_permission_resolution():
    perms = resolve_principal_permissions(Role.MEMBER)
    assert Permission.FACTOR_READ_SUMMARY in perms
    assert Permission.FACTOR_READ_FORMULA not in perms  # MEMBER lacks formula

    core = resolve_principal_permissions(Role.CORE)
    assert Permission.FACTOR_READ_FORMULA in core
    assert Permission.LIBRARY_APPROVE not in core  # CORE lacks approve


def test_effective_permissions_alias():
    assert effective_permissions(Role.ADMIN) == resolve_principal_permissions(Role.ADMIN)


def test_per_user_override_grants_permission():
    db = SqliteDb(":memory:")
    _seed_principal(db, "p1", Role.MEMBER, overrides=[Permission.FACTOR_READ_FORMULA])
    rbac = RBAC(db)
    assert Permission.FACTOR_READ_FORMULA in rbac.permissions_for("p1")


def test_min_classification_enforcement():
    db = SqliteDb(":memory:")
    _seed_principal(db, "p1", Role.CORE)
    rbac = RBAC(db)

    # CORE holds factor:read_formula (min CONFIDENTIAL_ALPHA).
    assert rbac.has_permission(
        "p1", Permission.FACTOR_READ_FORMULA, SecurityClassification.CONFIDENTIAL_ALPHA
    )
    # A resource at or above the floor is readable.
    assert rbac.has_permission(
        "p1", Permission.FACTOR_READ_FORMULA, SecurityClassification.RESTRICTED_RAW_VALUES
    )
    # A resource BELOW the floor (PUBLIC_METADATA) is denied.
    assert not rbac.has_permission(
        "p1", Permission.FACTOR_READ_FORMULA, SecurityClassification.PUBLIC_METADATA
    )


def test_min_classification_denied_when_below_minimum():
    db = SqliteDb(":memory:")
    _seed_principal(db, "p1", Role.MEMBER)
    rbac = RBAC(db)

    # MEMBER holds factor:read_evidence (min INTERNAL_RESEARCH).
    assert rbac.has_permission(
        "p1", Permission.FACTOR_READ_EVIDENCE, SecurityClassification.INTERNAL_RESEARCH
    )
    # A resource at or above the floor is readable.
    assert rbac.has_permission(
        "p1", Permission.FACTOR_READ_EVIDENCE, SecurityClassification.CONFIDENTIAL_ALPHA
    )
    # A resource BELOW the floor (PUBLIC_METADATA) is denied.
    assert not rbac.has_permission(
        "p1", Permission.FACTOR_READ_EVIDENCE, SecurityClassification.PUBLIC_METADATA
    )


def test_require_raises_permission_denied():
    db = SqliteDb(":memory:")
    _seed_principal(db, "p1", Role.MEMBER)
    rbac = RBAC(db)
    with pytest.raises(PermissionDenied):
        rbac.require("p1", Permission.LIBRARY_APPROVE)
    # No exception when authorized.
    rbac.require("p1", Permission.FACTOR_READ_SUMMARY)


def test_unknown_override_ignored():
    db = SqliteDb(":memory:")
    db.execute(
        "INSERT INTO principals (principal_id, principal_type, display_name) VALUES (?, ?, ?)",
        ("p1", "HUMAN", "p1"),
    )
    db.execute(
        "INSERT INTO roles (principal_id, role) VALUES (?, ?)",
        ("p1", Role.MEMBER.value),
    )
    db.execute(
        "INSERT INTO permissions (principal_id, permission) VALUES (?, ?)",
        ("p1", "not:a:real:permission"),
    )
    rbac = RBAC(db)
    # Unknown override string is ignored, not fatal.
    assert Permission.FACTOR_READ_SUMMARY in rbac.permissions_for("p1")
