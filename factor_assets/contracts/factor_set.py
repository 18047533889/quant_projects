"""
FactorSet and selection specifications.

Groups of factors selected for downstream use (e.g., preprocessing, modeling).
References factors by ID only — no raw values stored.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FactorMembership:
    """
    Per-member provenance record inside an assembled FactorSet.

    Captures why and how a factor entered the set: the admission decision
    reference, evidence and novelty references, orientation relative to the
    set, and the role assigned during assembly.
    """

    factor_id: str
    role: str = "member"                       # "member" | "representative" | "shadow"
    orientation: Optional[int] = None          # +1 / -1 relative to set convention
    family_id: Optional[str] = None
    cluster_id: Optional[int] = None
    selection_decision_ref: Optional[str] = None
    evidence_ref: Optional[str] = None
    novelty_ref: Optional[str] = None
    reason: Optional[str] = None               # SelectionReason value

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if self.orientation is not None and self.orientation not in (-1, 1):
            raise ValueError("orientation must be -1, 1, or None")


@dataclass(frozen=True)
class FactorSetSpec:
    """
    Specification for factor set selection.

    Defines criteria and constraints for assembling a FactorSet.
    """
    set_id: str
    name: str
    selection_policy: str  # "family_robust", "pareto_front", "manual", etc.
    universe_ref: Optional[str] = None
    frequency: Optional[str] = None
    max_factors: Optional[int] = None
    min_evidence_date: Optional[str] = None
    required_domains: tuple[str, ...] = ()
    excluded_domains: tuple[str, ...] = ()
    min_lifecycle_state: Optional[str] = None
    family_constraints: Optional[str] = None
    split_ref: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if not self.set_id:
            raise ValueError("set_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.selection_policy:
            raise ValueError("selection_policy is required")


@dataclass(frozen=True)
class FactorSetArtifact:
    """
    Complete, hash-addressed assembly output.

    Wraps the factor ID list with full per-member provenance and the content
    hashes needed to verify reproducibility: what policy produced it
    (policy_hash), on what data snapshot (snapshot_ref), under which split
    (split_ref), and the content hash of the assembly itself (assembly_hash).
    """

    set_id: str
    name: str
    members: tuple[FactorMembership, ...]
    created_at: str  # ISO 8601
    policy_hash: str
    assembly_hash: str
    universe_ref: Optional[str] = None
    snapshot_ref: Optional[str] = None
    split_ref: Optional[str] = None
    spec: Optional[FactorSetSpec] = None

    def __post_init__(self):
        if not self.set_id:
            raise ValueError("set_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.members:
            raise ValueError("members cannot be empty")
        if not self.policy_hash:
            raise ValueError("policy_hash is required")
        if not self.assembly_hash:
            raise ValueError("assembly_hash is required")
        if not self.created_at:
            raise ValueError("created_at is required")
        member_ids = [m.factor_id for m in self.members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("duplicate factor_id in members")

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """Factor IDs of all members, in membership order."""
        return tuple(m.factor_id for m in self.members)

    def contains(self, factor_id: str) -> bool:
        """Check if factor_id is a member of this artifact."""
        return any(m.factor_id == factor_id for m in self.members)


@dataclass(frozen=True)
class FactorSet:
    """
    A selected set of factor assets.

    Contains factor IDs and references only — no raw factor values.
    Used to pass selected factors to FactorPreprocess or other consumers.
    """
    set_id: str
    name: str
    factor_ids: tuple[str, ...]
    created_at: str  # ISO 8601
    spec: Optional[FactorSetSpec] = None
    universe_ref: Optional[str] = None
    frequency: Optional[str] = None
    # Selection metadata
    selection_run_id: Optional[str] = None
    parent_set_id: Optional[str] = None
    # Aggregation info (if factors are grouped)
    families: tuple[str, ...] = ()
    representatives: tuple[str, ...] = ()
    # Per-member provenance (empty for legacy assemblies)
    memberships: tuple[FactorMembership, ...] = ()
    assembly_hash: Optional[str] = None
    policy_hash: Optional[str] = None
    snapshot_ref: Optional[str] = None
    split_ref: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if not self.set_id:
            raise ValueError("set_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.factor_ids:
            raise ValueError("factor_ids cannot be empty")
        if not self.created_at:
            raise ValueError("created_at is required")

    @property
    def size(self) -> int:
        """Get number of factors in the set."""
        return len(self.factor_ids)

    @property
    def has_families(self) -> bool:
        """Check if factors are organized into families."""
        return len(self.families) > 0

    @property
    def has_representatives(self) -> bool:
        """Check if representative factors are designated."""
        return len(self.representatives) > 0

    def contains(self, factor_id: str) -> bool:
        """Check if factor_id is in this set."""
        return factor_id in self.factor_ids
