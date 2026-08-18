"""
Selection decision records and policies.

Immutable records of factor selection decisions with full provenance.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional
from datetime import datetime, timezone
import math


class SelectionReason(Enum):
    """Reason for selection decision."""
    APPROVED = "APPROVED"                    # Passed all gates
    REJECTED_GATE_FAILURE = "REJECTED_GATE_FAILURE"  # Failed admission gates
    REJECTED_SIMILARITY = "REJECTED_SIMILARITY"      # Too similar to existing
    REJECTED_EVIDENCE = "REJECTED_EVIDENCE"          # Insufficient evidence
    REJECTED_QUALITY = "REJECTED_QUALITY"            # Quality concerns
    PENDING_EVALUATION = "PENDING_EVALUATION"        # Awaiting evaluation
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"              # Manual decision


@dataclass(frozen=True)
class SelectionDecision:
    """
    Immutable record of a factor selection decision.

    Complete provenance: gates evaluated, evidence considered, final decision.
    Append-only record for audit trail.
    """
    decision_id: str
    factor_id: str
    approved: bool
    reason: SelectionReason
    timestamp: str  # ISO 8601
    policy_version: str
    evidence_refs: tuple[str, ...]
    gate_results: tuple[str, ...]  # Gate evaluation IDs
    similarity_refs: tuple[str, ...] = ()
    actor: Optional[str] = None
    notes: Optional[str] = None
    metadata: Mapping[str, str] = None

    def __post_init__(self):
        if not self.decision_id:
            raise ValueError("decision_id is required")
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.timestamp:
            raise ValueError("timestamp is required")
        if not self.policy_version:
            raise ValueError("policy_version is required")
        # Snapshot caller-owned metadata and expose it through an immutable view.
        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def is_approved(self) -> bool:
        """Check if decision approved the factor."""
        return self.approved

    @property
    def is_rejected(self) -> bool:
        """Check if decision rejected the factor."""
        return not self.approved

    @property
    def has_evidence(self) -> bool:
        """Check if decision has evidence references."""
        return len(self.evidence_refs) > 0

    @property
    def has_gate_results(self) -> bool:
        """Check if decision has gate evaluation results."""
        return len(self.gate_results) > 0


class SelectionPolicy:
    """
    Policy for making factor selection decisions.

    Evaluates gates, checks similarity, and produces selection decisions.
    """

    def __init__(
        self,
        policy_name: str,
        policy_version: str = "1.0",
        similarity_threshold: float = 0.7,
        require_evidence: bool = True,
    ):
        """
        Initialize selection policy.

        Args:
            policy_name: Policy identifier
            policy_version: Policy version
            similarity_threshold: Maximum allowed similarity to existing factors
            require_evidence: If True, require evidence for approval
        """
        if not policy_name:
            raise ValueError("policy_name is required")
        if not policy_version:
            raise ValueError("policy_version is required")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")

        self._policy_name = policy_name
        self._policy_version = policy_version
        self._similarity_threshold = similarity_threshold
        self._require_evidence = require_evidence
        self._decisions: list[SelectionDecision] = []

    @property
    def policy_name(self) -> str:
        """Get policy name."""
        return self._policy_name

    @property
    def policy_version(self) -> str:
        """Get policy version."""
        return self._policy_version

    def make_decision(
        self,
        factor_id: str,
        gate_evaluations: list,  # List of GateEvaluation
        evidence_refs: tuple[str, ...] = (),
        similarity_refs: tuple[str, ...] = (),
        max_similarity: Optional[float] = None,
        actor: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> SelectionDecision:
        """
        Make a selection decision based on gates and evidence.

        Args:
            factor_id: Factor identifier
            gate_evaluations: List of GateEvaluation results
            evidence_refs: Evidence reference IDs
            similarity_refs: Similarity check reference IDs
            max_similarity: Maximum similarity to existing factors (if computed)
            actor: Decision actor identifier
            notes: Optional decision notes

        Returns:
            SelectionDecision record
        """
        now = datetime.now(timezone.utc).isoformat()
        decision_id = f"SD_{factor_id}_{now.replace(':', '').replace('-', '')[:15]}"

        gate_result_ids = tuple(
            f"{ev.gate_name}_{ev.timestamp[:10]}"
            for ev in gate_evaluations
        )

        # Check evidence requirement
        if self._require_evidence and not evidence_refs:
            decision = SelectionDecision(
                decision_id=decision_id,
                factor_id=factor_id,
                approved=False,
                reason=SelectionReason.REJECTED_EVIDENCE,
                timestamp=now,
                policy_version=self._policy_version,
                evidence_refs=evidence_refs,
                gate_results=gate_result_ids,
                similarity_refs=similarity_refs,
                actor=actor,
                notes=notes or "Insufficient evidence",
            )
            self._decisions.append(decision)
            return decision

        # Gate evaluation is mandatory for admission.  Keep this fail-closed:
        # all([]) is True, but an approval without any evaluated gate is invalid.
        all_gates_passed = bool(gate_evaluations) and all(
            hasattr(ev, 'passed') and ev.passed
            for ev in gate_evaluations
        )

        if not all_gates_passed:
            failed_gates = [
                ev.gate_name
                for ev in gate_evaluations
                if hasattr(ev, 'passed') and not ev.passed
            ]
            gate_note = (
                "No gate evaluations provided"
                if not gate_evaluations
                else f"Failed gates: {', '.join(failed_gates)}"
            )
            decision = SelectionDecision(
                decision_id=decision_id,
                factor_id=factor_id,
                approved=False,
                reason=SelectionReason.REJECTED_GATE_FAILURE,
                timestamp=now,
                policy_version=self._policy_version,
                evidence_refs=evidence_refs,
                gate_results=gate_result_ids,
                similarity_refs=similarity_refs,
                actor=actor,
                notes=notes or gate_note,
            )
            self._decisions.append(decision)
            return decision

        # Similarity scores use canonical absolute-score semantics at the policy
        # boundary.  Nonfinite values are never admissible.
        normalized_similarity = None
        if max_similarity is not None:
            if not math.isfinite(max_similarity):
                decision = SelectionDecision(
                    decision_id=decision_id,
                    factor_id=factor_id,
                    approved=False,
                    reason=SelectionReason.REJECTED_SIMILARITY,
                    timestamp=now,
                    policy_version=self._policy_version,
                    evidence_refs=evidence_refs,
                    gate_results=gate_result_ids,
                    similarity_refs=similarity_refs,
                    actor=actor,
                    notes=notes or f"Nonfinite similarity score {max_similarity!r}",
                )
                self._decisions.append(decision)
                return decision
            normalized_similarity = abs(max_similarity)

        if normalized_similarity is not None and normalized_similarity > self._similarity_threshold:
            decision = SelectionDecision(
                decision_id=decision_id,
                factor_id=factor_id,
                approved=False,
                reason=SelectionReason.REJECTED_SIMILARITY,
                timestamp=now,
                policy_version=self._policy_version,
                evidence_refs=evidence_refs,
                gate_results=gate_result_ids,
                similarity_refs=similarity_refs,
                actor=actor,
                notes=notes or f"Similarity {normalized_similarity:.3f} exceeds threshold {self._similarity_threshold:.3f}",
            )
            self._decisions.append(decision)
            return decision

        # All checks passed - approve
        decision = SelectionDecision(
            decision_id=decision_id,
            factor_id=factor_id,
            approved=True,
            reason=SelectionReason.APPROVED,
            timestamp=now,
            policy_version=self._policy_version,
            evidence_refs=evidence_refs,
            gate_results=gate_result_ids,
            similarity_refs=similarity_refs,
            actor=actor,
            notes=notes or "Passed all admission criteria",
        )
        self._decisions.append(decision)
        return decision

    def get_decision(self, factor_id: str) -> Optional[SelectionDecision]:
        """
        Get most recent decision for a factor.

        Args:
            factor_id: Factor identifier

        Returns:
            Most recent SelectionDecision if found, None otherwise
        """
        factor_decisions = [
            d for d in self._decisions
            if d.factor_id == factor_id
        ]

        if not factor_decisions:
            return None

        return sorted(factor_decisions, key=lambda d: d.timestamp, reverse=True)[0]

    def get_all_decisions(self, factor_id: Optional[str] = None) -> list[SelectionDecision]:
        """
        Get all decisions, optionally filtered by factor.

        Args:
            factor_id: Optional factor identifier to filter by

        Returns:
            List of SelectionDecision records
        """
        if factor_id:
            return [d for d in self._decisions if d.factor_id == factor_id]
        return list(self._decisions)

    def count_approved(self) -> int:
        """Get count of approved factors."""
        return sum(1 for d in self._decisions if d.approved)

    def count_rejected(self) -> int:
        """Get count of rejected factors."""
        return sum(1 for d in self._decisions if not d.approved)

    def count_total(self) -> int:
        """Get total number of decisions."""
        return len(self._decisions)
