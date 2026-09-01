"""Trial: record of a single mutation attempt with status and results."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


"""Trial: record of a single mutation attempt with status and results."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TrialStatus(Enum):
    """Status of a mutation trial.

    FO-P0-04: the trial lifecycle is recorded with a single unified status
    vocabulary.  The historical members are retained for backward
    compatibility; the hardening adds explicit outcomes for every proposal
    attempt so a burned-budget proposal is never silently dropped or mislabelled
    as a duplicate.
    """

    PROPOSED = "proposed"  # Mutation created, not yet validated
    VALIDATING = "validating"  # Checking legality/grammar
    ILLEGAL = "illegal"  # Failed grammar/legality check
    LEGAL = "legal"  # Passed validation, ready for evaluation
    EVALUATING = "evaluating"  # Being evaluated by QE
    EVALUATED = "evaluated"  # Evaluation complete, has evidence
    FAILED = "failed"  # Evaluation failed (error, timeout, etc.)
    DUPLICATE = "duplicate"  # Seen cache hit
    # FO-P0-04: proposal-attempt outcomes distinct from a duplicate.
    PROPOSAL_FAILED = "proposal_failed"  # proposal_fn raised; budget burned
    INVALID_PROPOSAL = "invalid_proposal"  # proposal_fn returned a non-Trial
    EVALUATION_FAILED = "evaluation_failed"  # evaluation errored/illegal result
    PRUNED = "pruned"  # multi-fidelity: dropped without promotion
    SELECTED = "selected"  # EVALUATED and selected as incumbent


@dataclass
class Trial:
    """
    Record of a single mutation trial.

    Attributes:
        trial_id: Unique identifier for this trial
        mutation_id: Reference to the CandidateMutation
        status: Current trial status
        parent_factor_ids: Parent factor IDs (denormalized for quick access)
        created_at: Trial creation timestamp
        updated_at: Last status update timestamp
        legality_check: Legality validation result (if status >= LEGAL)
        evaluation_ref: Reference to QE EvaluationBundle (if status == EVALUATED)
        failure_reason: Error message (if status == FAILED)
        metadata: Additional trial metadata
    """

    trial_id: str
    mutation_id: str
    status: TrialStatus
    parent_factor_ids: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Results
    legality_check: Optional[Dict[str, Any]] = None
    evaluation_ref: Optional[str] = None
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_status(self, new_status: TrialStatus, **kwargs) -> None:
        """Update trial status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)

        # Update optional fields from kwargs
        if "legality_check" in kwargs:
            self.legality_check = kwargs["legality_check"]
        if "evaluation_ref" in kwargs:
            self.evaluation_ref = kwargs["evaluation_ref"]
        if "failure_reason" in kwargs:
            self.failure_reason = kwargs["failure_reason"]
        if "metadata" in kwargs:
            self.metadata.update(kwargs["metadata"])

    def is_terminal(self) -> bool:
        """Check if trial has reached a terminal status."""
        return self.status in {
            TrialStatus.ILLEGAL,
            TrialStatus.EVALUATED,
            TrialStatus.FAILED,
            TrialStatus.DUPLICATE,
            TrialStatus.PROPOSAL_FAILED,
            TrialStatus.INVALID_PROPOSAL,
            TrialStatus.EVALUATION_FAILED,
            TrialStatus.PRUNED,
            TrialStatus.SELECTED,
        }

    def is_successful(self) -> bool:
        """Check if trial succeeded (has evaluation)."""
        return self.status == TrialStatus.EVALUATED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trial_id": self.trial_id,
            "mutation_id": self.mutation_id,
            "status": self.status.value,
            "parent_factor_ids": list(self.parent_factor_ids),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "legality_check": self.legality_check,
            "evaluation_ref": self.evaluation_ref,
            "failure_reason": self.failure_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trial":
        """Deserialize from dictionary."""
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            data["status"] = TrialStatus(data["status"])
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)
