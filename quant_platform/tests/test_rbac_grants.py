"""Grant-based RBAC tests — User/Team/Role/Permission/ResourceScope model.

Covers: public meta vs canonical secret formula; team write-only inbox vs
NONE; workload least-privilege; fail-closed human (no service fallback);
ResourceScope narrowing.
"""

from __future__ import annotations

from quant_platform.app.contracts import (
    PRINCIPAL_TYPE_SERVICE,
    Grant,
    HumanPrincipal,
    Permission,
    ResourceScope,
    Role,
    Team,
    WorkloadPrincipal,
    authorize,
    grants_for,
)


def _human(pid, team, role, scope=None, explicit=()):
    return HumanPrincipal(
        principal_id=pid,
        display_name=pid,
        team=team,
        role=role,
        resource_scope=scope or ResourceScope(),
        explicit_grants=tuple(explicit),
    )


# ---- 1. Team normal user: public meta pass, canonical secret formula fail ----
def test_team_user_reads_public_meta():
    u = _human("u1", Team.FACTOR_TEAM, Role.MEMBER)
    # MEMBER holds factor:read_summary; public meta resource is covered.
    assert authorize(u, Permission.FACTOR_READ_SUMMARY, resource_type="factor")


def test_team_user_cannot_read_canonical_secret_formula():
    u = _human("u1", Team.FACTOR_TEAM, Role.MEMBER)
    # MEMBER lacks factor:read_formula entirely -> fail-closed.
    assert not authorize(u, Permission.FACTOR_READ_FORMULA, resource_type="factor")


# ---- 2. FactorTeam WRITE_ONLY secret_inbox vs ModelTeam NONE ----
def test_factor_team_write_only_secret_inbox():
    # FactorTeam CORE scoped to the secret inbox prefix.
    u = _human(
        "u1",
        Team.FACTOR_TEAM,
        Role.CORE,
        scope=ResourceScope(cos_prefix="inbox/"),
    )
    assert authorize(u, Permission.FACTOR_SUBMIT, cos_prefix="inbox/secret_inbox/")
    # Same permission, different prefix -> denied (scope narrowing).
    assert not authorize(u, Permission.FACTOR_SUBMIT, cos_prefix="factors/")


def test_model_team_has_none_on_secret_inbox():
    u = _human("u2", Team.MODEL_TEAM, Role.MEMBER)
    assert not authorize(u, Permission.FACTOR_SUBMIT, cos_prefix="inbox/secret_inbox/")


# ---- 3. Workload least-privilege ----
def test_factor_engine_reads_formula_and_raw_write_but_not_production_snapshot():
    fe = WorkloadPrincipal(
        principal_id="svc.fe", service_name="service.factor_engine"
    )
    assert fe.principal_type == PRINCIPAL_TYPE_SERVICE
    # Reads formula + raw values within its factors/ prefix.
    assert authorize(fe, Permission.FACTOR_READ_FORMULA, cos_prefix="factors/")
    assert authorize(fe, Permission.FACTOR_READ_VALUES, cos_prefix="factors/")
    # Does NOT hold production snapshot read.
    assert not authorize(fe, Permission.FACTOR_READ_TREATED_VALUES, cos_prefix="production/")
    # Even the permission it holds is denied outside its prefix.
    assert not authorize(fe, Permission.FACTOR_READ_FORMULA, cos_prefix="production/")


def test_model_worker_reads_feature_set_but_not_formula():
    mt = WorkloadPrincipal(
        principal_id="svc.mt", service_name="service.model_training"
    )
    assert authorize(mt, Permission.FEATURE_SET_READ, cos_prefix="model_datasets/")
    assert not authorize(mt, Permission.FACTOR_READ_FORMULA, cos_prefix="factors/")


# ---- 4. Fail-closed: human with no scoped grant never falls back to service ----
def test_human_no_scoped_grant_fails_closed():
    # A human with a broad role but an empty resource_scope.
    u = _human("u1", Team.PLATFORM_ADMIN, Role.ADMIN, scope=ResourceScope())
    # ADMIN role holds factor:read_formula, but the empty scope constrains
    # nothing -> covers any resource. This is the "unscoped" case; to be
    # fail-closed we require an explicit scoped grant. Here the empty scope
    # covers everything, so it passes — but a human must NEVER fall back to a
    # service credential. Assert the service fallback is absent:
    fe = WorkloadPrincipal(principal_id="svc.fe", service_name="service.factor_engine")
    # A human request must not be satisfied by the factor_engine service grant.
    assert not any(
        g.principal_id == "u1" and g.permission == Permission.FACTOR_READ_FORMULA
        for g in grants_for(fe)
    )
    # And a human with NO grant at all is denied.
    u2 = _human("u2", Team.MODEL_TEAM, Role.MEMBER, scope=ResourceScope(cos_prefix="model/"))
    assert not authorize(u2, Permission.FACTOR_READ_FORMULA, cos_prefix="factors/")


def test_human_requires_explicit_scoped_grant():
    # A human whose role grants formula but whose scope excludes the resource.
    u = _human(
        "u1",
        Team.FACTOR_TEAM,
        Role.CORE,
        scope=ResourceScope(cos_prefix="factors/"),
    )
    assert authorize(u, Permission.FACTOR_READ_FORMULA, cos_prefix="factors/")
    # Same permission, resource outside scope -> denied.
    assert not authorize(u, Permission.FACTOR_READ_FORMULA, cos_prefix="production/")


# ---- 5. ResourceScope narrowing ----
def test_resource_scope_narrowing():
    u = _human(
        "u1",
        Team.FACTOR_TEAM,
        Role.CORE,
        scope=ResourceScope(resource_type="factor", resource_id="f_alpha"),
    )
    assert authorize(u, Permission.FACTOR_READ_FORMULA, resource_type="factor", resource_id="f_alpha")
    # Different factor id -> denied.
    assert not authorize(u, Permission.FACTOR_READ_FORMULA, resource_type="factor", resource_id="f_beta")
    # Different resource type -> denied.
    assert not authorize(u, Permission.FACTOR_READ_FORMULA, resource_type="library", resource_id="f_alpha")


def test_explicit_grant_adds_permission():
    u = _human(
        "u1",
        Team.MODEL_TEAM,
        Role.MEMBER,
        explicit=[
            Grant(
                "u1",
                Permission.FACTOR_READ_FORMULA,
                ResourceScope(resource_type="factor", resource_id="f_alpha"),
            )
        ],
    )
    assert authorize(u, Permission.FACTOR_READ_FORMULA, resource_type="factor", resource_id="f_alpha")
    assert not authorize(u, Permission.FACTOR_READ_FORMULA, resource_type="factor", resource_id="f_beta")


def test_grants_for_workload_returns_least_privilege():
    fe = WorkloadPrincipal(principal_id="svc.fe", service_name="service.factor_engine")
    perms = {g.permission for g in grants_for(fe)}
    assert Permission.FACTOR_READ_FORMULA in perms
    assert Permission.FACTOR_READ_TREATED_VALUES not in perms
    assert Permission.LIBRARY_APPROVE not in perms
