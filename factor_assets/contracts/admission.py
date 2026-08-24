"""
Factor admission artifact contract.

A complete, immutable record of *everything* that justified a factor's
admission during the admission (selection) stage: the quality evidence,
health state, similarity and novelty references, cluster/orientation
assignment, the admission decision itself, the gate results and the policy
that produced them.  It is the single authoritative object that
:class:`factor_assets.assembly.engine.FactorSetAssembler` consumes (via
``admission_artifacts``) to populate the production-mandatory membership
fields — so the assembler never silently invents health/cluster/orientation.

The object is intentionally self-describing: ``content_hash`` is a sha256
over every semantic field, so two artifacts with identical decision content
share a hash and any drift in the frozen inputs is detectable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Optional

from factor_assets.contracts.similarity import SimilarityArtifact

__all__ = [
    "AdmissionDecision",
    "FactorAdmissionArtifact",
    "ADMISSION_DECISION_APPROVED",
    "ADMISSION_DECISION_REJECTED",
    "ADMISSION_DECISION_SHADOWED",
]


class AdmissionDecision(Enum):
    """Final decision of the admission stage for a factor."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SHADOWED = "SHADOWED"


def _length_prefixed(digest: "hashlib._Hash", field_value: object) -> None:
    """Hash a field with length-prefixing so delimiters cannot collide."""
    encoded = str(field_value).encode("utf-8")
    digest.update(str(len(encoded)).encode("ascii"))
    digest.update(b":")
    digest.update(encoded)


def _content_hash(
    factor_id: str,
    factor_version: Optional[str],
    quality: Optional[float],
    health_state_ref: Optional[str],
    similarity_ref: Optional[str],
    novelty_ref: Optional[str],
    cluster_id: Optional[int],
    orientation: Optional[int],
    decision: AdmissionDecision,
    reason: Optional[str],
    evidence_refs: tuple[str, ...],
    gate_results: tuple[str, ...],
    policy_ref: Optional[str],
) -> str:
    """sha256 over every semantic field of the admission artifact.

    Includes the decision enum, the quality score, the gate results and every
    reference — so two artifacts that differ in *any* admission-relevant
    detail produce different content hashes.  Producer/created_at are
    intentionally excluded: they are provenance, not decision content.
    """
    digest = hashlib.sha256()
    for item in (
        factor_id,
        factor_version or "",
        quality,
        health_state_ref or "",
        similarity_ref or "",
        novelty_ref or "",
        cluster_id,
        orientation,
        decision.value,
        reason or "",
        "#",
        *evidence_refs,
        "#",
        *gate_results,
        policy_ref or "",
    ):
        _length_prefixed(digest, item)
    return digest.hexdigest()


@dataclass(frozen=True)
class FactorAdmissionArtifact:
    """Immutable record of a factor's full admission-stage decision basis.

    Fields:
        factor_id: The factor being admitted.
        factor_version: Version of the factor definition that was evaluated.
        quality: Composite quality score (finite float) that justified the
            decision; ``None`` when the policy did not score it.
        health_state_ref: Reference to the health evidence at admission time.
        similarity_ref: Reference to the similarity evidence (e.g. a
            :class:`SimilarityArtifact.similarity_spec_hash` or legacy id).
        novelty_ref: Reference to the novelty assessment evidence.
        cluster_id: Cluster/family the factor was assigned to during
            admission.  Optional outside production; mandatory in
            production mode.
        orientation: +1 / -1 orientation relative to the set convention.
        decision: Approved / Rejected / Shadowed.
        reason: Human-readable reason (or SelectionReason value).
        evidence_refs: Evidence bundle references that supported admission.
        gate_results: Gate evaluation result identifiers.
        policy_ref: Identifier of the policy that produced this decision.
        created_at: ISO 8601 creation timestamp.
        content_hash: sha256 over every semantic field (auto-computed when
            not supplied).  Two artifacts with identical admission content
            share a hash; any drift is detectable.
    """

    factor_id: str
    decision: AdmissionDecision
    quality: Optional[float] = None
    factor_version: Optional[str] = None
    health_state_ref: Optional[str] = None
    similarity_ref: Optional[str] = None
    novelty_ref: Optional[str] = None
    cluster_id: Optional[int] = None
    orientation: Optional[int] = None
    reason: Optional[str] = None
    evidence_refs: tuple[str, ...] = ()
    gate_results: tuple[str, ...] = ()
    policy_ref: Optional[str] = None
    created_at: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not isinstance(self.decision, AdmissionDecision):
            raise TypeError("decision must be an AdmissionDecision")
        if self.quality is not None:
            if isinstance(self.quality, bool) or not isinstance(
                self.quality, (int, float)
            ):
                raise TypeError("quality must be a non-boolean number or None")
            quality = float(self.quality)
            if quality != quality or quality in (float("inf"), float("-inf")):
                raise ValueError("quality must be finite")
            object.__setattr__(self, "quality", quality)
        if self.orientation is not None and self.orientation not in (-1, 1):
            raise ValueError("orientation must be -1, 1, or None")
        if self.cluster_id is not None and (
            isinstance(self.cluster_id, bool)
            or not isinstance(self.cluster_id, int)
        ):
            raise TypeError("cluster_id must be an int or None")
        if self.decision is AdmissionDecision.APPROVED:
            if not self.evidence_refs:
                raise ValueError("APPROVED admission requires evidence_refs")
            if not self.gate_results:
                raise ValueError("APPROVED admission requires gate_results")

        if not self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                _content_hash(
                    self.factor_id,
                    self.factor_version,
                    self.quality,
                    self.health_state_ref,
                    self.similarity_ref,
                    self.novelty_ref,
                    self.cluster_id,
                    self.orientation,
                    self.decision,
                    self.reason,
                    self.evidence_refs,
                    self.gate_results,
                    self.policy_ref,
                ),
            )
        if not self.created_at:
            object.__setattr__(
                self, "created_at", datetime.now(timezone.utc).isoformat()
            )

    @classmethod
    def from_selection_decision(
        cls,
        decision: object,
        *,
        factor_version: Optional[str] = None,
        quality: Optional[float] = None,
        health_state_ref: Optional[str] = None,
        cluster_id: Optional[int] = None,
        orientation: Optional[int] = None,
        policy_ref: Optional[str] = None,
    ) -> "FactorAdmissionArtifact":
        """Adapt a legacy :class:`SelectionDecision` into an admission artifact.

        Maps ``approved`` + ``reason`` onto the canonical
        :class:`AdmissionDecision` enum:
          - approved=True                    -> APPROVED
          - approved=False, reason=SHADOW    -> SHADOWED
          - otherwise                        -> REJECTED

        Legacy decisions do not carry cluster/orientation/health, so those
        fields are ``None`` unless supplied.  ``similarity_ref`` /
        ``novelty_ref`` are lifted from the decision's ref tuples when
        present.
        """
        factor_id = getattr(decision, "factor_id", None)
        if factor_id is None:
            raise TypeError("decision must expose factor_id (e.g. SelectionDecision)")
        approved = getattr(decision, "approved", None)
        reason = getattr(decision, "reason", None)
        reason_value = getattr(reason, "value", reason)

        if approved:
            admission_decision = AdmissionDecision.APPROVED
        elif reason_value == "SHADOW":
            admission_decision = AdmissionDecision.SHADOWED
        else:
            admission_decision = AdmissionDecision.REJECTED

        similarity_refs = tuple(getattr(decision, "similarity_refs", ()) or ())
        novelty_refs = tuple(getattr(decision, "novelty_refs", ()) or ())

        return cls(
            factor_id=factor_id,
            decision=admission_decision,
            factor_version=factor_version,
            quality=quality,
            health_state_ref=health_state_ref,
            similarity_ref=similarity_refs[0] if similarity_refs else None,
            novelty_ref=novelty_refs[0] if novelty_refs else None,
            cluster_id=cluster_id,
            orientation=orientation,
            reason=reason_value,
            evidence_refs=tuple(getattr(decision, "evidence_refs", ()) or ()),
            gate_results=tuple(getattr(decision, "gate_results", ()) or ()),
            policy_ref=policy_ref or getattr(decision, "policy_version", None),
            created_at=getattr(decision, "timestamp", None),
        )

    @property
    def is_approved(self) -> bool:
        """Whether the admission stage approved this factor."""
        return self.decision is AdmissionDecision.APPROVED

    @property
    def is_rejected(self) -> bool:
        """Whether the admission stage rejected this factor."""
        return self.decision is AdmissionDecision.REJECTED

    @property
    def is_shadowed(self) -> bool:
        """Whether this factor was shadowed (recorded, not admitted)."""
        return self.decision is AdmissionDecision.SHADOWED

    def to_dict(self) -> dict:
        """Serializable dict form (content_hash + created_at for audit)."""
        return {
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "quality": self.quality,
            "health_state_ref": self.health_state_ref,
            "similarity_ref": self.similarity_ref,
            "novelty_ref": self.novelty_ref,
            "cluster_id": self.cluster_id,
            "orientation": self.orientation,
            "decision": self.decision.value,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "gate_results": list(self.gate_results),
            "policy_ref": self.policy_ref,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }


ADMISSION_DECISION_APPROVED = AdmissionDecision.APPROVED
ADMISSION_DECISION_REJECTED = AdmissionDecision.REJECTED
ADMISSION_DECISION_SHADOWED = AdmissionDecision.SHADOWED
