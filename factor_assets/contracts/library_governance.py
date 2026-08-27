"""Factor Library governance contracts for factor_assets (DLIB-FA-012).

A Factor Library is a use-case-driven collection of factors (one library spans
many clusters; one factor can be in many libraries).  Library count is
use-case driven, not cluster-count driven.  A new stable family leads to a
:class:`NewLibraryProposal` awaiting policy/approval.

The canonical artifacts here are deep-immutable and hash-addressed:

- :class:`FactorLibraryDefinition` — the logical library identity and its
  governance policies.
- :class:`FactorLibraryMembership` — per-factor membership with full provenance.
- :class:`FactorLibraryVersionArtifact` — an immutable, versioned snapshot of a
  library (one per promotion).  Promotion changes only the active pointer;
  rollback only changes the pointer; the previous version (e.g. v102) is
  preserved.
- :class:`NewLibraryProposal` — a proposal for a new library awaiting
  policy/approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional

from factor_assets.contracts._canonical import canonical_digest

__all__ = [
    "FactorLibraryStatus",
    "FactorLibraryDefinition",
    "FactorLibraryMembership",
    "FactorLibraryVersionArtifact",
    "NewLibraryProposal",
]


def _canonical_optimization_ref(value: object) -> object:
    """Canonicalize a treatment-optimization reference into its transport form.

    Accepts ``None``, a bare ``str`` (content-hash shorthand), or a ``dict``
    (FO ``LibrarySnapshotRef.to_dict()`` PURE-DTO).  Returns the canonical
    string for any other scalar — FA carries the reference cross-package as an
    opaque string (matching FO ``library_snapshot_ref.to_dict()`` semantics at
    the FO boundary) and never imports the FO contract.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise ValueError(
                "treatment_optimization_ref must be a non-empty string, a dict, or None"
            )
        return value
    if isinstance(value, Mapping):
        # Full ref mapping (FO LibrarySnapshotRef.to_dict() form).  Normalize a
        # bare content_hash shorthand into its string transport form.
        ref = value.get("content_hash")
        if isinstance(ref, str):
            if not ref:
                raise ValueError(
                    "treatment_optimization_ref dict content_hash must be a "
                    "non-empty string"
                )
            return ref
        return value
    if isinstance(value, tuple) and len(value) == 1:
        return _canonical_optimization_ref(value[0])
    raise TypeError(
        "treatment_optimization_ref must be a non-empty string, a dict, or None; "
        f"got {type(value).__name__}"
    )


class FactorLibraryStatus(Enum):
    """Lifecycle status of a FactorLibraryVersionArtifact."""

    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class FactorLibraryDefinition:
    """Canonical definition of a factor library (DLIB-FA-012).

    ``logical_library_id`` is the stable library identity.  The governance
    policies (selection / cluster-budget / capacity / health / promotion) are
    referenced, not embedded, so a policy change is a new policy ref, not a
    mutation of this definition.
    """

    logical_library_id: str
    name: str
    purpose: str
    consumer_profiles: tuple[str, ...] = ()
    selection_policy_ref: Optional[str] = None
    cluster_budget_policy_ref: Optional[str] = None
    capacity_policy_ref: Optional[str] = None
    health_policy_ref: Optional[str] = None
    promotion_policy_ref: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.logical_library_id:
            raise ValueError("logical_library_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.purpose:
            raise ValueError("purpose is required")
        object.__setattr__(self, "consumer_profiles", tuple(self.consumer_profiles))

    def to_dict(self) -> dict:
        return {
            "logical_library_id": self.logical_library_id,
            "name": self.name,
            "purpose": self.purpose,
            "consumer_profiles": list(self.consumer_profiles),
            "selection_policy_ref": self.selection_policy_ref,
            "cluster_budget_policy_ref": self.cluster_budget_policy_ref,
            "capacity_policy_ref": self.capacity_policy_ref,
            "health_policy_ref": self.health_policy_ref,
            "promotion_policy_ref": self.promotion_policy_ref,
            "description": self.description,
        }


@dataclass(frozen=True)
class FactorLibraryMembership:
    """Per-factor membership in a FactorLibraryVersion (DLIB-FA-012).

    Carries the factor definition ref, the selected treatment ref, the
    orientation, the logical cluster id, the cluster-set-version ref, and the
    admission / evaluation / novelty / similarity / health refs, plus the
    assembly score and selection rank.  ``assembly_score`` must be finite
    (not just non-NaN).
    """

    factor_definition_ref: str
    selected_treatment_ref: Optional[str] = None
    # Reference to the FO treatment-optimization run that produced this
    # member's winning treatment (DLIB-FO-008 / INT-2 closure).  FA is
    # CONSUME-only: it carries the reference produced upstream by the
    # optimizer (carried as a bare str / dict PURE-DTO, matching FO's
    # ``library_snapshot_ref.to_dict()`` form) and never recomputes or
    # fabricates it.  Included in the content hash so the member is
    # attributable to the exact optimization result.
    treatment_optimization_ref: Optional[str | dict] = None
    orientation: Optional[int] = None
    logical_cluster_id: Optional[str] = None
    cluster_set_version_ref: Optional[str] = None
    admission_ref: Optional[str] = None
    evaluation_ref: Optional[str] = None
    novelty_ref: Optional[str] = None
    similarity_ref: Optional[str] = None
    health_ref: Optional[str] = None
    assembly_score: Optional[float] = None
    selection_rank: Optional[int] = None
    role: str = "member"

    def __post_init__(self) -> None:
        if not self.factor_definition_ref:
            raise ValueError("factor_definition_ref is required")
        # Canonicalize the treatment-optimization reference into its transport
        # form (bare str shorthand or full ref mapping PURE-DTO).
        object.__setattr__(
            self,
            "treatment_optimization_ref",
            _canonical_optimization_ref(self.treatment_optimization_ref),
        )
        if self.orientation is not None and self.orientation not in (-1, 1):
            raise ValueError("orientation must be -1, 1, or None")
        if self.assembly_score is not None:
            if isinstance(self.assembly_score, bool) or not isinstance(
                self.assembly_score, (int, float)
            ):
                raise TypeError("assembly_score must be a non-boolean number or None")
            score = float(self.assembly_score)
            if score != score or score in (float("inf"), float("-inf")):
                raise ValueError("assembly_score must be finite")
            object.__setattr__(self, "assembly_score", score)

    def to_dict(self) -> dict:
        return {
            "factor_definition_ref": self.factor_definition_ref,
            "selected_treatment_ref": self.selected_treatment_ref,
            "treatment_optimization_ref": self.treatment_optimization_ref,
            "orientation": self.orientation,
            "logical_cluster_id": self.logical_cluster_id,
            "cluster_set_version_ref": self.cluster_set_version_ref,
            "admission_ref": self.admission_ref,
            "evaluation_ref": self.evaluation_ref,
            "novelty_ref": self.novelty_ref,
            "similarity_ref": self.similarity_ref,
            "health_ref": self.health_ref,
            "assembly_score": self.assembly_score,
            "selection_rank": self.selection_rank,
            "role": self.role,
        }


@dataclass(frozen=True)
class FactorLibraryVersionArtifact:
    """Immutable, versioned snapshot of a factor library (DLIB-FA-012).

    ``cluster_set_version_ref`` is REQUIRED (a library spans many clusters, so
    it references the ClusterSetVersion, NOT a single cluster version).
    Promotion changes only the active pointer; rollback only changes the
    pointer; the previous version (e.g. v102) is preserved.  ``content_hash``
    is derived-only.
    """

    library_version_id: str
    logical_library_id: str
    members: tuple[FactorLibraryMembership, ...]
    cluster_set_version_ref: str
    selection_policy_ref: str
    evidence_snapshot_ref: str
    snapshot_ref: str
    universe_ref: str
    created_at: str
    status: FactorLibraryStatus = FactorLibraryStatus.CANDIDATE
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.library_version_id:
            raise ValueError("library_version_id is required")
        if not self.logical_library_id:
            raise ValueError("logical_library_id is required")
        if not self.members:
            raise ValueError("members cannot be empty")
        if not self.cluster_set_version_ref:
            raise ValueError(
                "cluster_set_version_ref is REQUIRED — a library spans many "
                "clusters, so it references the ClusterSetVersion, not a single "
                "cluster version"
            )
        if not self.selection_policy_ref:
            raise ValueError("selection_policy_ref is required")
        if not self.evidence_snapshot_ref:
            raise ValueError("evidence_snapshot_ref is required")
        if not self.snapshot_ref:
            raise ValueError("snapshot_ref is required")
        if not self.universe_ref:
            raise ValueError("universe_ref is required")
        if not self.created_at:
            raise ValueError("created_at is required")
        if not isinstance(self.status, FactorLibraryStatus):
            raise TypeError("status must be a FactorLibraryStatus")
        object.__setattr__(self, "members", tuple(self.members))
        member_refs = [m.factor_definition_ref for m in self.members]
        if len(set(member_refs)) != len(member_refs):
            raise ValueError("duplicate factor_definition_ref in members")
        computed = canonical_digest(
            self.library_version_id,
            self.logical_library_id,
            tuple(m.to_dict() for m in self.members),
            self.cluster_set_version_ref,
            self.selection_policy_ref,
            self.evidence_snapshot_ref,
            self.snapshot_ref,
            self.universe_ref,
            self.status.value,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed factor-library "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )

    @property
    def factor_definition_refs(self) -> tuple[str, ...]:
        """Factor definition refs of all members, in membership order."""
        return tuple(m.factor_definition_ref for m in self.members)

    def to_dict(self) -> dict:
        return {
            "library_version_id": self.library_version_id,
            "logical_library_id": self.logical_library_id,
            "members": [m.to_dict() for m in self.members],
            "cluster_set_version_ref": self.cluster_set_version_ref,
            "selection_policy_ref": self.selection_policy_ref,
            "evidence_snapshot_ref": self.evidence_snapshot_ref,
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "created_at": self.created_at,
            "status": self.status.value,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class NewLibraryProposal:
    """Proposal for a new factor library (DLIB-FA-012).

    A new stable family leads to a NewLibraryProposal awaiting policy /
    approval.  Library count is use-case driven, not cluster-count driven.
    """

    proposal_id: str
    logical_library_id: str
    name: str
    purpose: str
    rationale: str
    proposed_by: Optional[str] = None
    created_at: Optional[str] = None
    status: str = "PENDING"

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ValueError("proposal_id is required")
        if not self.logical_library_id:
            raise ValueError("logical_library_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.purpose:
            raise ValueError("purpose is required")
        if not self.rationale:
            raise ValueError("rationale is required")
        if not self.created_at:
            object.__setattr__(
                self, "created_at", datetime.now(timezone.utc).isoformat()
            )

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "logical_library_id": self.logical_library_id,
            "name": self.name,
            "purpose": self.purpose,
            "rationale": self.rationale,
            "proposed_by": self.proposed_by,
            "created_at": self.created_at,
            "status": self.status,
        }
