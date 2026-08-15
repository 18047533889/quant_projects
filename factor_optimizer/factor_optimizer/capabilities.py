"""Truthful runtime capability metadata for factor_optimizer."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class ExecutionMode(str, Enum):
    """Supported execution intent."""

    RESEARCH_ONLY = "research_only"
    PRODUCTION = "production"


class CapabilityStatus(str, Enum):
    """Availability state for a capability."""

    SUPPORTED = "supported"
    RESEARCH_ONLY = "research_only"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ProductionCapability:
    """Machine-readable package production readiness."""

    status: CapabilityStatus
    supported: bool
    blockers: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status.value,
            "supported": self.supported,
            "blockers": list(self.blockers),
        }


PRODUCTION_CAPABILITY = ProductionCapability(
    status=CapabilityStatus.RESEARCH_ONLY,
    supported=False,
    blockers=(
        "split_plan_contract_stub",
        "sealed_test_handle_stub",
        "split_aware_qe_integration_missing",
        "production_legality_chain_missing",
    ),
)


def require_production_capability() -> None:
    """Fail closed until all upstream production contracts are implemented."""
    if not PRODUCTION_CAPABILITY.supported:
        from factor_optimizer.errors import CapabilityError

        blockers = ", ".join(PRODUCTION_CAPABILITY.blockers)
        raise CapabilityError(
            "factor_optimizer production execution is unsupported; "
            f"package status is {PRODUCTION_CAPABILITY.status.value}. "
            f"Unimplemented capabilities: {blockers}"
        )


__all__ = [
    "ExecutionMode",
    "CapabilityStatus",
    "ProductionCapability",
    "PRODUCTION_CAPABILITY",
    "require_production_capability",
]
