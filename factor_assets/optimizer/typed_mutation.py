"""
Typed mutation representation for factor asset optimization.

Bridges factor_optimizer MutationSpec with factor_assets identity and lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MutationStatus(Enum):
    """Status of a mutation operation."""

    PENDING = "pending"
    VALIDATED = "validated"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TypedMutation:
    """
    Typed mutation with FA identity binding.

    Attributes:
        mutation_id: Unique mutation identifier
        parent_asset_id: Parent factor asset being mutated
        mutation_type: Type from factor_optimizer grammar
        parameters: Mutation parameters (validated against MutationSpec)
        context: Mutation context from diagnosis/search
        created_at: Creation timestamp
        version: Grammar version
    """

    mutation_id: str
    parent_asset_id: str
    mutation_type: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    version: str = "0.1.0"

    def __post_init__(self):
        if not self.mutation_id:
            raise ValueError("mutation_id is required")
        if not self.parent_asset_id:
            raise ValueError("parent_asset_id is required")
        if not self.mutation_type:
            raise ValueError("mutation_type is required")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mutation_id": self.mutation_id,
            "parent_asset_id": self.parent_asset_id,
            "mutation_type": self.mutation_type,
            "parameters": dict(self.parameters),
            "context": dict(self.context) if self.context else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypedMutation":
        """Deserialize from dictionary."""
        data = dict(data)
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class MutationContext:
    """
    Context for mutation execution.

    Attributes:
        campaign_id: Campaign this mutation belongs to
        trial_id: Trial identifier
        diagnosis: Optional diagnosis that triggered this mutation
        search_metadata: Search algorithm metadata
        budget_remaining: Remaining evaluation budget
        fidelity_tier: Target evaluation fidelity (L0-L4)
    """

    campaign_id: str
    trial_id: str
    diagnosis: Optional[Dict[str, Any]] = None
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    budget_remaining: Optional[float] = None
    fidelity_tier: str = "L2"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "trial_id": self.trial_id,
            "diagnosis": dict(self.diagnosis) if self.diagnosis else None,
            "search_metadata": dict(self.search_metadata),
            "budget_remaining": self.budget_remaining,
            "fidelity_tier": self.fidelity_tier,
        }


@dataclass
class MutationResult:
    """
    Result of mutation execution.

    Attributes:
        mutation_id: Mutation that was executed
        status: Execution status
        child_asset_id: Generated child asset ID (if successful)
        error_message: Error message (if failed)
        execution_time_ms: Execution duration
        validation_errors: Validation errors from FE/FO
        metadata: Additional execution metadata
    """

    mutation_id: str
    status: MutationStatus
    child_asset_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if mutation succeeded."""
        return self.status == MutationStatus.EXECUTED and self.child_asset_id is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mutation_id": self.mutation_id,
            "status": self.status.value,
            "child_asset_id": self.child_asset_id,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "validation_errors": list(self.validation_errors),
            "metadata": dict(self.metadata),
        }
