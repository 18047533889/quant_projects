"""R21-001..005: endpoint execution policy — the non-downgradable floor.

Sits in ``runtime`` (not ``service``) so ``runtime/engine.py`` can enforce it
without importing the service package (which imports the engine).  The service
layer re-exports these symbols for the HTTP layer.
"""

from __future__ import annotations

import enum
from typing import Any


class EndpointExecutionPolicy(str, enum.Enum):
    """Per-endpoint execution policy.  ``PRODUCTION`` is a non-downgradable floor."""

    RESEARCH = "research"
    PRODUCTION = "production"

    @classmethod
    def parse(cls, value: Any) -> "EndpointExecutionPolicy":
        text = str(value or "").strip().lower()
        if text not in {"research", "production"}:
            raise ValueError(f"invalid run mode: {value!r}")
        return cls(text)

    @property
    def is_production(self) -> bool:
        return self is EndpointExecutionPolicy.PRODUCTION


class ProductionPolicyConflictError(RuntimeError):
    """R21-002: production endpoint + config that lowers the policy.

    ``reason`` is a stable code consumed by the service error taxonomy.
    """

    def __init__(self, detail: str, *, reason: str = "PRODUCTION_ENDPOINT_CONFIG_POLICY_CONFLICT"):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def collect_config_policy_conflicts(config: Any) -> list[str]:
    """Return config declarations that would lower a production endpoint policy.

    Called by ``FactorEngine.from_loaded_config`` when an endpoint forces
    ``execution_policy="production"`` (R21-001/002/005).
    """
    conflicts: list[str] = []
    if str(getattr(config.run, "mode", "research") or "research").lower() != "production":
        conflicts.append(f"run.mode={getattr(config.run, 'mode', 'research')!r} != production")
    pit = getattr(config, "pit", None)
    if pit is None or not bool(getattr(pit, "enforce", False)):
        conflicts.append("pit.enforce=false")
    dq = getattr(config, "dq", None)
    if dq is None or not bool(getattr(dq, "strict", False)):
        conflicts.append("dq.strict=false")
    mat = getattr(config, "materialization", None)
    if mat is not None:
        target = str(getattr(mat, "target", "local") or "local").lower()
        if target == "local":
            conflicts.append("materialization.target=local (direct-local write)")
    return conflicts
