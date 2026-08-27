# -*- coding: utf-8 -*-
"""Wave1-F Part B tests — action-granular object-storage security (non-vacuous).

Covers:
  (a) read-only scope cannot write (permission.py ResourceScope.allowed_actions +
      authorize(action), and ObjectStorageAuthorizationBoundary per-action grants).
  (b) prefix-A scope cannot touch prefix B (path scoping still enforced).
  (c) production resolver with no target raises (no placeholder default).
  (d) every object action (read/list/write/delete/multipart:* /metadata:commit) is
      gateable per-scope.
"""
from __future__ import annotations

import pytest

from data_access.core.exceptions import AccessDeniedError, ValidationError
from data_access.write.object_storage_boundary import (
    ACTION_METADATA_COMMIT,
    ACTION_MULTIPART_COMPLETE,
    ACTION_MULTIPART_CREATE,
    ACTION_MULTIPART_UPLOAD,
    ACTION_OBJECT_DELETE,
    ACTION_OBJECT_LIST,
    ACTION_OBJECT_READ,
    ACTION_OBJECT_WRITE,
    ALL_OBJECT_ACTIONS,
    ObjectResource,
    ObjectResourceScope,
    ObjectStorageAuthorizationBoundary,
    ProductionTargetResolver,
)

from quant_platform.app.contracts.permission import (
    Permission,
    ResourceScope,
    authorize,
)
from quant_platform.app.contracts.principal import HumanPrincipal
from quant_platform.app.contracts.security import Role, Team


def _human(scope: ResourceScope):
    return HumanPrincipal(
        principal_id="u1",
        display_name="u1",
        team=Team.MODEL_TEAM,
        role=Role.CORE,  # CORE carries FACTOR_SUBMIT / FACTOR_READ_VALUES
        resource_scope=scope,
    )


# ---- (a) read-only scope cannot write ---------------------------------------

def test_read_only_scope_cannot_write_permission_layer():
    user = _human(
        ResourceScope(cos_prefix="factors/", allowed_actions=("object:read", "object:list"))
    )
    # read is authorized
    assert authorize(
        user, Permission.FACTOR_READ_VALUES,
        cos_prefix="factors/active/f1", action="object:read",
    )
    # write is NOT
    assert not authorize(
        user, Permission.FACTOR_SUBMIT,
        cos_prefix="factors/active/f1", action="object:write",
    )
    # delete is NOT
    assert not authorize(
        user, Permission.FACTOR_SUBMIT,
        cos_prefix="factors/active/f1", action="object:delete",
    )


def test_read_only_boundary_cannot_write():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(
                bucket="factor-lake", key_prefix="factors/*",
                allowed_actions=frozenset({ACTION_OBJECT_READ, ACTION_OBJECT_LIST}),
            )
        ],
    )
    res = ObjectResource(bucket="factor-lake", key_prefix="factors/f1/data.parquet")
    # read/list authorized
    boundary.authorize_object(res, ACTION_OBJECT_READ)
    boundary.authorize_object(res, ACTION_OBJECT_LIST)
    # write action denied
    for action in (ACTION_OBJECT_WRITE, ACTION_OBJECT_DELETE,
                   ACTION_MULTIPART_CREATE, ACTION_MULTIPART_UPLOAD,
                   ACTION_MULTIPART_COMPLETE, ACTION_METADATA_COMMIT):
        with pytest.raises(AccessDeniedError):
            boundary.authorize_object(res, action)
    with pytest.raises(AccessDeniedError):
        boundary.authorize_factor_write("f1")


def test_legacy_scope_unconstrained_action_still_works():
    # A scope with NO allowed_actions keeps the legacy semantic: path coverage is
    # sufficient (backwards compatible with all existing workloads/tests).
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*"),
        ],
    )
    res = ObjectResource(bucket="factor-lake", key_prefix="factors/f1/data.parquet")
    boundary.authorize_object(res, ACTION_OBJECT_WRITE)
    boundary.authorize_object(res, ACTION_OBJECT_READ)


def test_mixed_grants_require_both_path_and_action():
    # Two scopes: one read-only on prefix A, one write-only on prefix B.
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(
                bucket="factor-lake", key_prefix="factors/",
                allowed_actions=frozenset({ACTION_OBJECT_READ}),
            ),
            ObjectResourceScope(
                bucket="factor-lake", key_prefix="treated/",
                allowed_actions=frozenset({ACTION_OBJECT_WRITE}),
            ),
        ],
    )
    read_a = ObjectResource(bucket="factor-lake", key_prefix="factors/f1/x")
    boundary.authorize_object(read_a, ACTION_OBJECT_READ)
    with pytest.raises(AccessDeniedError):
        boundary.authorize_object(read_a, ACTION_OBJECT_WRITE)  # path A not write

    write_b = ObjectResource(bucket="factor-lake", key_prefix="treated/f1/x")
    boundary.authorize_object(write_b, ACTION_OBJECT_WRITE)
    with pytest.raises(AccessDeniedError):
        boundary.authorize_object(write_b, ACTION_OBJECT_READ)  # path B not read


# ---- (b) prefix-A scope cannot touch prefix B --------------------------------

def test_prefix_a_scope_cannot_touch_prefix_b():
    user = _human(
        ResourceScope(cos_prefix="inbox/", allowed_actions=("object:read",))
    )
    assert authorize(
        user, Permission.FACTOR_READ_VALUES, cos_prefix="inbox/f1", action="object:read"
    )
    assert not authorize(
        user, Permission.FACTOR_READ_VALUES, cos_prefix="factors/f1", action="object:read"
    )
    assert not authorize(
        user, Permission.FACTOR_READ_VALUES, cos_prefix="inbox2/f1", action="object:read"
    )
    # boundary: same prefix restriction on write.
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(
                bucket="factor-lake", key_prefix="inbox/",
                allowed_actions=frozenset({ACTION_OBJECT_WRITE}),
            )
        ],
    )
    boundary.authorize_object(
        ObjectResource(bucket="factor-lake", key_prefix="inbox/f1"), ACTION_OBJECT_WRITE
    )
    with pytest.raises(AccessDeniedError):
        boundary.authorize_object(
            ObjectResource(bucket="factor-lake", key_prefix="other/f1"), ACTION_OBJECT_WRITE
        )


def test_require_any_action_still_path_gated():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(
                bucket="factor-lake", key_prefix="factors/*",
                allowed_actions=frozenset(ALL_OBJECT_ACTIONS),
            )
        ],
    )
    boundary.require_any_object_write_action(
        ObjectResource(bucket="factor-lake", key_prefix="factors/f1")
    )
    with pytest.raises(AccessDeniedError):
        boundary.require_any_object_write_action(
            ObjectResource(bucket="other", key_prefix="factors/f1")
        )


# ---- (c) production resolver with no target raises ---------------------------

def test_production_boundary_construction_requires_resolver():
    with pytest.raises(ValidationError):
        ObjectStorageAuthorizationBoundary(
            allowed_scopes=[ObjectResourceScope(bucket="factor-lake")],
            environment="production",
        )


def test_production_resolver_no_placeholder_default_for_unknown():
    resolver = ProductionTargetResolver(  # explicit layout authority
        layout_map={"f1": ("factor-lake", "factors/f1")},
    )
    assert resolver.resolve("f1") == ("factor-lake", "factors/f1")
    # factor NOT in the layout map -> fail, no fallback to factor-lake/factors/x
    with pytest.raises(ValidationError):
        resolver.resolve("f_unknown")


def test_production_resolver_incomplete_target_fails():
    resolver = ProductionTargetResolver(
        layout_map={"f1": ("", "factors/f1")}  # empty bucket = placeholder
    )
    with pytest.raises(ValidationError):
        resolver.resolve("f1")


def test_production_boundary_uses_provided_resolver_and_fails_when_unresolvable():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*"),
        ],
        factor_target_resolver=ProductionTargetResolver(
            layout_map={"f1": ("factor-lake", "factors/f1")}
        ),
        environment="production",
    )
    # known factor resolves + authorizes
    resource = boundary.authorize_factor_write("f1")
    assert resource.bucket == "factor-lake"
    # unknown factor -> ValidationError (no placeholder default)
    with pytest.raises(ValidationError):
        boundary.resolve_factor_write_target("f_unknown")


def test_research_resolver_still_uses_default_for_backward_compat():
    # research mode keeps legacy default resolver (explicit, backward-compat).
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*"),
        ],
        environment="research",
    )
    resource = boundary.authorize_factor_write("f2")
    assert resource.bucket == "factor-lake"
    assert resource.key_prefix == "factors/f2"


# ---- (d) every action is gateable per-scope ----------------------------------

def test_every_known_action_is_denied_outside_allowed_actions():
    read_only = frozenset({ACTION_OBJECT_READ})
    for action in sorted(ALL_OBJECT_ACTIONS - read_only):
        boundary = ObjectStorageAuthorizationBoundary(
            allowed_scopes=[
                ObjectResourceScope(
                    bucket="factor-lake", key_prefix="factors/*",
                    allowed_actions=read_only,
                )
            ],
        )
        with pytest.raises(AccessDeniedError):
            boundary.authorize_object(
                ObjectResource(bucket="factor-lake", key_prefix="factors/f1"),
                action,
            )


def test_permission_layer_action_must_be_in_scope_allowed_actions():
    write_only = ResourceScope(cos_prefix="out/", allowed_actions=("object:write",))
    user = _human(write_only)
    assert not authorize(
        user, Permission.FACTOR_SUBMIT, cos_prefix="out/f1", action="object:read"
    )
    assert authorize(
        user, Permission.FACTOR_SUBMIT, cos_prefix="out/f1", action="object:write"
    )
    # a principal whose scope does not constrain actions is unconstrained.
    user_legacy = _human(ResourceScope(cos_prefix="out/"))
    assert authorize(
        user_legacy, Permission.FACTOR_SUBMIT, cos_prefix="out/f1", action="object:read"
    )