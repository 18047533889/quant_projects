"""Principal model — HumanPrincipal vs WorkloadPrincipal.

DRAFT. PURE stdlib (frozen dataclasses). A **HumanPrincipal** is a person with
teams/roles/identity; a **WorkloadPrincipal** is a service identity
(``service.factor_ingestion``, ``service.factor_engine``, ...) carrying a
least-privilege permission set + resource scope + ``principal_type="SERVICE"``.

Authorization is grant-based (see ``permission.py``): a principal is authorized
for ``(permission, resource)`` iff a grant covers it. There is NO bare
"role can read all formulas" rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .permission import Grant, ResourceScope, Role, SecurityClassification, Team

__all__ = [
    "HumanPrincipal",
    "WorkloadPrincipal",
    "PRINCIPAL_TYPE_HUMAN",
    "PRINCIPAL_TYPE_SERVICE",
    "WORKLOAD_SERVICE_NAMES",
]

PRINCIPAL_TYPE_HUMAN = "HUMAN"
PRINCIPAL_TYPE_SERVICE = "SERVICE"

# Canonical service identities.
WORKLOAD_SERVICE_NAMES: tuple[str, ...] = (
    "service.factor_ingestion",
    "service.factor_engine",
    "service.quant_evaluator",
    "service.factor_preprocess",
    "service.factor_optimizer",
    "service.factor_assets",
    "service.feature_snapshot",
    "service.model_training",
    "service.production",
    "service.platform_api",
)


@dataclass(frozen=True)
class HumanPrincipal:
    """A human actor (user id + name + team + role + scoped grants)."""

    principal_id: str
    display_name: str
    team: Team
    role: Role
    resource_scope: ResourceScope = field(default_factory=ResourceScope)
    classification: SecurityClassification = SecurityClassification.INTERNAL_RESEARCH
    # Explicit, resource-scoped grants beyond the role's default permissions.
    explicit_grants: tuple[Grant, ...] = ()
    principal_type: str = PRINCIPAL_TYPE_HUMAN

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("principal_id is required")
        if self.principal_type != PRINCIPAL_TYPE_HUMAN:
            raise ValueError("HumanPrincipal.principal_type must be HUMAN")


@dataclass(frozen=True)
class WorkloadPrincipal:
    """A service / workload actor (OAuth service identity).

    Least-privilege grants are derived from ``WORKLOAD_LEAST_PRIVILEGE`` by
    ``permission.grants_for``; the principal itself carries only its identity.
    """

    principal_id: str
    service_name: str
    team: Team = Team.PLATFORM_ADMIN
    role: Role = Role.SERVICE
    resource_scope: ResourceScope = field(default_factory=ResourceScope)
    classification: SecurityClassification = SecurityClassification.CONFIDENTIAL_ALPHA
    principal_type: str = PRINCIPAL_TYPE_SERVICE

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("principal_id is required")
        if self.principal_type != PRINCIPAL_TYPE_SERVICE:
            raise ValueError("WorkloadPrincipal.principal_type must be SERVICE")
