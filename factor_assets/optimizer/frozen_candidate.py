"""
Frozen candidate with hash verification and test contamination tracking.

State machine ensures candidates cannot be modified after evaluation/test assignment.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class FrozenState(Enum):
    """States in the frozen candidate lifecycle."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    EVALUATION_READY = "evaluation_ready"
    EVALUATED = "evaluated"
    SPLIT_ASSIGNED = "split_assigned"
    TEST_SEALED = "test_sealed"
    TEST_CONTAMINATED = "test_contaminated"
    ADMITTED = "admitted"
    REJECTED = "rejected"


class ContaminationKind(Enum):
    """Type of test contamination."""

    PARAMETER_TUNING = "parameter_tuning"
    TRAIN_SPLIT_REUSE = "train_split_reuse"
    VALIDATION_SPLIT_REUSE = "validation_split_reuse"
    TEST_SPLIT_REUSE = "test_split_reuse"
    EARLY_STOPPING = "early_stopping"
    FEATURE_SELECTION = "feature_selection"


# Legal state transitions
LEGAL_TRANSITIONS: Dict[FrozenState, Set[FrozenState]] = {
    FrozenState.PROPOSED: {FrozenState.VALIDATED, FrozenState.REJECTED},
    FrozenState.VALIDATED: {FrozenState.EVALUATION_READY, FrozenState.REJECTED},
    FrozenState.EVALUATION_READY: {FrozenState.EVALUATED, FrozenState.REJECTED},
    FrozenState.EVALUATED: {FrozenState.SPLIT_ASSIGNED, FrozenState.REJECTED},
    FrozenState.SPLIT_ASSIGNED: {FrozenState.TEST_SEALED, FrozenState.REJECTED},
    FrozenState.TEST_SEALED: {
        FrozenState.TEST_CONTAMINATED,
        FrozenState.ADMITTED,
        FrozenState.REJECTED,
    },
    FrozenState.TEST_CONTAMINATED: {FrozenState.REJECTED},
    FrozenState.ADMITTED: set(),
    FrozenState.REJECTED: set(),
}


@dataclass(frozen=True)
class FrozenCandidate:
    """
    Immutable candidate with cryptographic hash verification.

    Attributes:
        candidate_id: Unique candidate identifier
        asset_id: Factor asset ID
        parameters: Frozen parameters
        mutation_id: Optional parent mutation
        state: Current lifecycle state
        content_hash: SHA256 hash of (asset_id + parameters)
        created_at: Creation timestamp
        state_history: History of state transitions
        contamination_flags: Set of contamination kinds detected
        metadata: Additional frozen metadata
    """

    candidate_id: str
    asset_id: str
    parameters: Dict[str, Any]
    mutation_id: Optional[str] = None
    state: FrozenState = FrozenState.PROPOSED
    content_hash: str = ""
    created_at: Optional[datetime] = None
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    contamination_flags: Set[ContaminationKind] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Compute content hash if not provided
        if not self.content_hash:
            computed_hash = self._compute_hash()
            object.__setattr__(self, "content_hash", computed_hash)

        # Initialize created_at if not provided
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.utcnow())

        # Ensure state_history is a list
        if not isinstance(self.state_history, list):
            object.__setattr__(self, "state_history", list(self.state_history))

        # Ensure contamination_flags is a set
        if not isinstance(self.contamination_flags, set):
            object.__setattr__(
                self, "contamination_flags", set(self.contamination_flags)
            )

    def _compute_hash(self) -> str:
        """Compute SHA256 hash of candidate content."""
        content = {
            "asset_id": self.asset_id,
            "parameters": self.parameters,
            "mutation_id": self.mutation_id,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Verify content hash matches current content."""
        return self.content_hash == self._compute_hash()

    def is_sealed(self) -> bool:
        """Check if candidate is sealed (cannot be modified)."""
        return self.state in {
            FrozenState.TEST_SEALED,
            FrozenState.TEST_CONTAMINATED,
            FrozenState.ADMITTED,
            FrozenState.REJECTED,
        }

    def is_contaminated(self) -> bool:
        """Check if candidate has any contamination flags."""
        return len(self.contamination_flags) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "asset_id": self.asset_id,
            "parameters": dict(self.parameters),
            "mutation_id": self.mutation_id,
            "state": self.state.value,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "state_history": list(self.state_history),
            "contamination_flags": [f.value for f in self.contamination_flags],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrozenCandidate":
        """Deserialize from dictionary."""
        data = dict(data)
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("state"):
            data["state"] = FrozenState(data["state"])
        if data.get("contamination_flags"):
            data["contamination_flags"] = {
                ContaminationKind(f) for f in data["contamination_flags"]
            }
        return cls(**data)


class FrozenCandidateStateMachine:
    """
    State machine for frozen candidate lifecycle.

    Enforces legal transitions and tracks contamination.
    """

    def __init__(self):
        self.candidates: Dict[str, FrozenCandidate] = {}

    def register(self, candidate: FrozenCandidate) -> None:
        """Register a new candidate."""
        if candidate.candidate_id in self.candidates:
            raise ValueError(f"Candidate {candidate.candidate_id} already registered")

        if not candidate.verify_hash():
            raise ValueError(
                f"Candidate {candidate.candidate_id} hash verification failed"
            )

        self.candidates[candidate.candidate_id] = candidate

    def transition(
        self,
        candidate_id: str,
        new_state: FrozenState,
        reason: str = "",
    ) -> FrozenCandidate:
        """
        Transition candidate to new state.

        Args:
            candidate_id: Candidate to transition
            new_state: Target state
            reason: Reason for transition

        Returns:
            New frozen candidate with updated state

        Raises:
            ValueError: If transition is illegal or candidate not found
        """
        if candidate_id not in self.candidates:
            raise ValueError(f"Candidate {candidate_id} not found")

        old_candidate = self.candidates[candidate_id]

        # Check if transition is legal
        if new_state not in LEGAL_TRANSITIONS.get(old_candidate.state, set()):
            raise ValueError(
                f"Illegal transition from {old_candidate.state.value} to {new_state.value}"
            )

        # Create transition record
        transition_record = {
            "from_state": old_candidate.state.value,
            "to_state": new_state.value,
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
        }

        # Create new candidate with updated state
        new_history = list(old_candidate.state_history) + [transition_record]
        new_candidate = FrozenCandidate(
            candidate_id=old_candidate.candidate_id,
            asset_id=old_candidate.asset_id,
            parameters=dict(old_candidate.parameters),
            mutation_id=old_candidate.mutation_id,
            state=new_state,
            content_hash=old_candidate.content_hash,
            created_at=old_candidate.created_at,
            state_history=new_history,
            contamination_flags=set(old_candidate.contamination_flags),
            metadata=dict(old_candidate.metadata),
        )

        # Update registry
        self.candidates[candidate_id] = new_candidate

        return new_candidate

    def flag_contamination(
        self,
        candidate_id: str,
        contamination_kind: ContaminationKind,
        reason: str = "",
    ) -> FrozenCandidate:
        """
        Flag candidate as contaminated and transition to TEST_CONTAMINATED.

        Args:
            candidate_id: Candidate to flag
            contamination_kind: Type of contamination
            reason: Reason for contamination flag

        Returns:
            New frozen candidate with contamination flag

        Raises:
            ValueError: If candidate not found or not in TEST_SEALED state
        """
        if candidate_id not in self.candidates:
            raise ValueError(f"Candidate {candidate_id} not found")

        old_candidate = self.candidates[candidate_id]

        # Can only flag contamination from TEST_SEALED state
        if old_candidate.state != FrozenState.TEST_SEALED:
            raise ValueError(
                f"Can only flag contamination from TEST_SEALED state, "
                f"current state is {old_candidate.state.value}"
            )

        # Add contamination flag
        new_flags = set(old_candidate.contamination_flags)
        new_flags.add(contamination_kind)

        # Create new candidate with contamination flag
        new_candidate = FrozenCandidate(
            candidate_id=old_candidate.candidate_id,
            asset_id=old_candidate.asset_id,
            parameters=dict(old_candidate.parameters),
            mutation_id=old_candidate.mutation_id,
            state=old_candidate.state,
            content_hash=old_candidate.content_hash,
            created_at=old_candidate.created_at,
            state_history=list(old_candidate.state_history),
            contamination_flags=new_flags,
            metadata=dict(old_candidate.metadata),
        )

        # Update registry
        self.candidates[candidate_id] = new_candidate

        # Transition to TEST_CONTAMINATED
        return self.transition(
            candidate_id,
            FrozenState.TEST_CONTAMINATED,
            f"{contamination_kind.value}: {reason}",
        )

    def get_candidate(self, candidate_id: str) -> Optional[FrozenCandidate]:
        """Get candidate by ID."""
        return self.candidates.get(candidate_id)

    def get_candidates_by_state(self, state: FrozenState) -> List[FrozenCandidate]:
        """Get all candidates in a given state."""
        return [c for c in self.candidates.values() if c.state == state]

    def get_contaminated_candidates(self) -> List[FrozenCandidate]:
        """Get all contaminated candidates."""
        return [c for c in self.candidates.values() if c.is_contaminated()]
