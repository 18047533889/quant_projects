"""CandidateMutation: typed mutation proposal with provenance and complexity estimate."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from factor_optimizer.errors import MissingInputError, InvalidContractError


@dataclass(frozen=True)
class CandidateMutation:
    """
    A proposed mutation to a parent factor.

    Attributes:
        mutation_id: Unique identifier for this mutation proposal
        mutation_spec_version: Version of the mutation grammar used
        parent_factor_ids: Factor IDs being mutated (usually one, sometimes multiple for composition)
        mutation_type: Type of mutation (e.g., "parameter_tune", "operator_swap", "composition")
        parameters: Mutation-specific parameters with typed values
        mechanism_hypothesis: Optional human/LLM hypothesis about why this might improve
        expected_signatures: Expected metric improvements or behavior changes
        complexity_estimate: Estimated compute/memory cost reference
        lineage_ref: Reference to parent lineage chain
        trial_ref: Reference to search trial/campaign context
        created_at: Creation timestamp
        producer: System/agent that proposed this mutation
        producer_version: Version of the proposing system
    """

    mutation_id: str
    mutation_spec_version: str
    parent_factor_ids: List[str]
    mutation_type: str
    parameters: Dict[str, Any]

    # Optional metadata
    mechanism_hypothesis: Optional[str] = None
    expected_signatures: List[str] = field(default_factory=list)
    complexity_estimate: Optional[Dict[str, Any]] = None
    lineage_ref: Optional[str] = None
    trial_ref: Optional[str] = None

    # Provenance
    created_at: Optional[datetime] = None
    producer: str = "factor_optimizer"
    producer_version: str = "0.1.0"

    def __post_init__(self):
        """Validate required fields."""
        if not self.mutation_id:
            raise MissingInputError("mutation_id is required")
        if not self.mutation_spec_version:
            raise MissingInputError("mutation_spec_version is required")
        if not self.parent_factor_ids:
            raise MissingInputError("At least one parent_factor_id is required")
        if not self.mutation_type:
            raise MissingInputError("mutation_type is required")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "mutation_id": self.mutation_id,
            "mutation_spec_version": self.mutation_spec_version,
            "parent_factor_ids": list(self.parent_factor_ids),
            "mutation_type": self.mutation_type,
            "parameters": dict(self.parameters),
            "mechanism_hypothesis": self.mechanism_hypothesis,
            "expected_signatures": list(self.expected_signatures),
            "complexity_estimate": self.complexity_estimate,
            "lineage_ref": self.lineage_ref,
            "trial_ref": self.trial_ref,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "producer": self.producer,
            "producer_version": self.producer_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateMutation":
        """Deserialize from dictionary."""
        data = dict(data)
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
