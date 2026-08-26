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
    set, and the role assigned during assembly.  Every member is traceable
    back to its factor version, family/cluster, representative status, and
    the similarity/health/novelty evidence that justified admission.
    """

    factor_id: str
    role: str = "member"                       # "member" | "representative" | "shadow"
    orientation: Optional[int] = None          # +1 / -1 relative to set convention
    family_id: Optional[str] = None
    cluster_id: Optional[int] = None
    factor_version: Optional[str] = None       # version of the factor definition
    representative_of: Optional[str] = None    # family/cluster this member represents
    selection_decision_ref: Optional[str] = None
    evidence_ref: Optional[str] = None
    novelty_ref: Optional[str] = None
    similarity_ref: Optional[str] = None       # similarity evidence that justified admission
    health_state_ref: Optional[str] = None     # health evidence at assembly time
    # Auto-treatment optimizer references.  FA is CONSUME-only for these: it
    # never recomputes or fabricates the selected treatment — it only carries
    # the reference produced upstream by the treatment-selection stage.  In
    # production mode the assembler fails closed if a treatment selection ref
    # is required but missing.
    treatment_selection_ref: Optional[str] = None   # selection artifact that picked the treatment
    preprocess_policy_ref: Optional[str] = None     # preprocess policy that consumes this member
    preprocess_state_ref: Optional[str] = None      # preprocess state snapshot the ref was computed on
    assembly_score: Optional[float] = None     # policy score that ranked this member
    selection_rank: Optional[int] = None       # 0-based rank within the assembled set
    reason: Optional[str] = None               # SelectionReason value

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if self.orientation is not None and self.orientation not in (-1, 1):
            raise ValueError("orientation must be -1, 1, or None")
        if self.assembly_score is not None and self.assembly_score != self.assembly_score:
            raise ValueError("assembly_score must be finite")
        for _name, _ref in (
            ("treatment_selection_ref", self.treatment_selection_ref),
            ("preprocess_policy_ref", self.preprocess_policy_ref),
            ("preprocess_state_ref", self.preprocess_state_ref),
        ):
            if _ref is not None and not isinstance(_ref, str):
                raise TypeError(f"{_name} must be a str or None")
            if _ref is not None and not _ref:
                raise ValueError(f"{_name} must be a non-empty string or None")


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
    # Provenance of the data snapshot the set was assembled against.  This is
    # REQUIRED: the universe must NOT impersonate the snapshot.  A set is only
    # reproducible when it records the exact data snapshot it was built on.
    data_snapshot_ref: Optional[str] = None
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
        if not self.data_snapshot_ref:
            raise ValueError(
                "data_snapshot_ref is required — the universe must not "
                "impersonate the data snapshot"
            )


@dataclass(frozen=True)
class FactorSetArtifact:
    """
    Complete, hash-addressed assembly output — the CANONICAL authority.

    Wraps the factor ID list with full per-member provenance and the content
    hashes needed to verify reproducibility: what policy produced it
    (policy_hash), on what data snapshot (snapshot_ref), under which split
    (split_ref), and the content hash of the assembly itself (assembly_hash).

    ``snapshot_ref``, ``universe_ref`` and ``split_ref`` are REQUIRED — a
    reproducible artifact must record the exact data snapshot, universe and
    split it was assembled against.  ``versions`` records the factor-version
    of each member and ``evidence_refs`` the evidence that justified the
    assembly, so every member is traceable.
    """

    set_id: str
    name: str
    members: tuple[FactorMembership, ...]
    created_at: str  # ISO 8601
    policy_hash: str
    assembly_hash: str
    snapshot_ref: str
    universe_ref: str
    split_ref: str
    versions: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
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
        if not self.snapshot_ref:
            raise ValueError("snapshot_ref is required")
        if not self.universe_ref:
            raise ValueError("universe_ref is required")
        if not self.split_ref:
            raise ValueError("split_ref is required")
        member_ids = [m.factor_id for m in self.members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("duplicate factor_id in members")

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """Factor IDs of all members, in membership order."""
        return tuple(m.factor_id for m in self.members)

    @property
    def memberships(self) -> tuple[FactorMembership, ...]:
        """Alias for ``members`` (legacy consumers read ``memberships``)."""
        return self.members

    @property
    def size(self) -> int:
        """Number of members in the artifact."""
        return len(self.members)

    @property
    def families(self) -> tuple[str, ...]:
        """Distinct family IDs among members, sorted."""
        return tuple(sorted({m.family_id for m in self.members if m.family_id is not None}))

    def contains(self, factor_id: str) -> bool:
        """Check if factor_id is a member of this artifact."""
        return any(m.factor_id == factor_id for m in self.members)

    def to_legacy_view(self) -> "FactorSet":
        """
        Produce the deprecated read-only :class:`FactorSet` compatibility view.

        The legacy ``FactorSet`` is NOT a second authority — it is a
        projection of this canonical artifact for consumers that predate the
        artifact contract.  New consumers should use the artifact directly.
        """
        return FactorSet(
            set_id=self.set_id,
            name=self.name,
            factor_ids=self.factor_ids,
            created_at=self.created_at,
            spec=self.spec,
            universe_ref=self.universe_ref,
            snapshot_ref=self.snapshot_ref,
            split_ref=self.split_ref,
            memberships=self.members,
            assembly_hash=self.assembly_hash,
            policy_hash=self.policy_hash,
            families=tuple(sorted({m.family_id for m in self.members if m.family_id is not None})),
            representatives=tuple(
                m.factor_id for m in self.members if m.representative_of is not None
            ),
        )


@dataclass(frozen=True)
class FactorSet:
    """
    DEPRECATED — read-only compatibility view of a :class:`FactorSetArtifact`.

    The canonical assembly authority is :class:`FactorSetArtifact`.  This
    legacy type is retained only for consumers that predate the artifact
    contract and is produced via ``FactorSetArtifact.to_legacy_view()``.  It
    is NOT a second authority and should not be constructed directly for new
    assemblies.

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
