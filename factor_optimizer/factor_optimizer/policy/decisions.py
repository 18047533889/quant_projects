"""Admission decision records for mutation proposals."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AdmissionVerdict(Enum):
    """Admission decision outcome."""

    ADMITTED = "admitted"  # Approved for evaluation
    REJECTED = "rejected"  # Rejected, will not evaluate
    DEFERRED = "deferred"  # Deferred pending resource/budget
    CONDITIONAL = "conditional"  # Admitted with constraints


class RejectionReason(Enum):
    """Reason for rejection."""

    BUDGET_EXCEEDED = "budget_exceeded"  # Exceeds resource budget
    DUPLICATE = "duplicate"  # Semantic duplicate of known factor
    ILLEGAL = "illegal"  # Violates grammar/legality rules
    DOMAIN_VIOLATION = "domain_violation"  # Wrong domain/market
    TIMING_VIOLATION = "timing_violation"  # Look-ahead or PIT violation
    LOW_EXPECTED_VALUE = "low_expected_value"  # Expected value below threshold
    COMPLEXITY_LIMIT = "complexity_limit"  # Too complex
    PARENT_QUALITY = "parent_quality"  # Parent factor quality insufficient
    POLICY_VIOLATION = "policy_violation"  # Violates organizational policy


@dataclass
class AdmissionCriteria:
    """
    Criteria used for admission decision.

    Attributes:
        max_complexity_cost: Maximum allowed complexity cost
        min_expected_value: Minimum expected information value
        max_lookback_periods: Maximum lookback window
        allowed_domains: Allowed data domains (empty = all)
        allowed_sources: Allowed data sources (empty = all)
        require_parent_evidence: Whether parent must have evaluation evidence
        min_parent_quality: Minimum parent quality score [0.0, 1.0]
        budget_constraints: Additional budget constraints
    """

    max_complexity_cost: float = float("inf")
    min_expected_value: float = 0.0
    max_lookback_periods: int = 252
    allowed_domains: List[str] = field(default_factory=list)
    allowed_sources: List[str] = field(default_factory=list)
    require_parent_evidence: bool = False
    min_parent_quality: float = 0.0
    budget_constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_complexity_cost": self.max_complexity_cost,
            "min_expected_value": self.min_expected_value,
            "max_lookback_periods": self.max_lookback_periods,
            "allowed_domains": list(self.allowed_domains),
            "allowed_sources": list(self.allowed_sources),
            "require_parent_evidence": self.require_parent_evidence,
            "min_parent_quality": self.min_parent_quality,
            "budget_constraints": dict(self.budget_constraints),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdmissionCriteria":
        """Deserialize from dictionary (fail-closed on unknown fields)."""
        known = {
            "max_complexity_cost", "min_expected_value", "max_lookback_periods",
            "allowed_domains", "allowed_sources", "require_parent_evidence",
            "min_parent_quality", "budget_constraints",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown AdmissionCriteria fields: {sorted(unknown)}"
            )
        # JSON round-trips inf as null; a null complexity bound means the
        # serialized side had "no limit" — restore it, never treat null
        # as a comparison operand (TypeError) or as 0 (reject everything).
        values = dict(data)
        if values.get("max_complexity_cost") is None:
            values["max_complexity_cost"] = float("inf")
        return cls(**values)


@dataclass
class AdmissionDecision:
    """
    Record of an admission decision for a mutation proposal.

    Attributes:
        decision_id: Unique decision identifier
        mutation_id: Mutation being decided on
        trial_id: Trial identifier
        verdict: Admission verdict
        decided_at: Decision timestamp
        criteria_used: Criteria applied for decision
        rejection_reasons: Reasons if rejected (primary and secondary)
        conditions: Conditions if admitted conditionally
        expected_value: Expected information value estimate
        complexity_cost: Complexity cost estimate
        decision_metadata: Additional decision context
        decided_by: Decision maker identifier (system/human)
    """

    decision_id: str
    mutation_id: str
    trial_id: str
    verdict: AdmissionVerdict
    decided_at: datetime = field(default_factory=datetime.now)
    criteria_used: Optional[AdmissionCriteria] = None
    rejection_reasons: List[RejectionReason] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    expected_value: Optional[float] = None
    complexity_cost: Optional[float] = None
    decision_metadata: Dict[str, Any] = field(default_factory=dict)
    decided_by: str = "factor_optimizer"

    def is_approved(self) -> bool:
        """Check if mutation is approved for evaluation."""
        return self.verdict in {AdmissionVerdict.ADMITTED, AdmissionVerdict.CONDITIONAL}

    def is_rejected(self) -> bool:
        """Check if mutation is rejected."""
        return self.verdict == AdmissionVerdict.REJECTED

    def is_deferred(self) -> bool:
        """Check if mutation is deferred."""
        return self.verdict == AdmissionVerdict.DEFERRED

    def primary_rejection_reason(self) -> Optional[RejectionReason]:
        """Get primary rejection reason."""
        return self.rejection_reasons[0] if self.rejection_reasons else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "decision_id": self.decision_id,
            "mutation_id": self.mutation_id,
            "trial_id": self.trial_id,
            "verdict": self.verdict.value,
            "decided_at": self.decided_at.isoformat(),
            "criteria_used": self.criteria_used.to_dict() if self.criteria_used else None,
            "rejection_reasons": [r.value for r in self.rejection_reasons],
            "conditions": list(self.conditions),
            "expected_value": self.expected_value,
            "complexity_cost": self.complexity_cost,
            "decision_metadata": dict(self.decision_metadata),
            "decided_by": self.decided_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdmissionDecision":
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("verdict"), str):
            data["verdict"] = AdmissionVerdict(data["verdict"])
        if isinstance(data.get("decided_at"), str):
            data["decided_at"] = datetime.fromisoformat(data["decided_at"])
        if "rejection_reasons" in data:
            data["rejection_reasons"] = [
                RejectionReason(r) if isinstance(r, str) else r for r in data["rejection_reasons"]
            ]
        if data.get("criteria_used"):
            data["criteria_used"] = AdmissionCriteria.from_dict(data["criteria_used"])
        return cls(**data)


class AdmissionPolicy:
    """
    Policy engine for mutation admission decisions.

    Evaluates mutation proposals against admission criteria and produces decisions.
    """

    def __init__(self, criteria: Optional[AdmissionCriteria] = None):
        """
        Initialize admission policy.

        Args:
            criteria: Default admission criteria
        """
        self.criteria = criteria or AdmissionCriteria()
        self._decision_history: Dict[str, AdmissionDecision] = {}

    def decide(
        self,
        mutation_id: str,
        trial_id: str,
        complexity_cost: float,
        expected_value: float,
        parent_quality: Optional[float] = None,
        domains: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        lookback_periods: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdmissionDecision:
        """
        Make admission decision for a mutation proposal.

        Args:
            mutation_id: Mutation identifier
            trial_id: Trial identifier
            complexity_cost: Estimated complexity cost
            expected_value: Expected information value
            parent_quality: Parent factor quality score
            domains: Data domains required
            sources: Data sources required
            lookback_periods: Lookback window length
            metadata: Additional decision context

        Returns:
            AdmissionDecision
        """
        # decision_id is derived from trial_id, so a second decision for
        # the same trial would silently overwrite the first in history.
        # Use a monotonically increasing suffix instead.
        decision_id = f"decision_{trial_id}"
        if decision_id in self._decision_history:
            seq = 2
            while f"decision_{trial_id}_{seq}" in self._decision_history:
                seq += 1
            decision_id = f"decision_{trial_id}_{seq}"
        rejection_reasons = []

        # Check complexity
        if complexity_cost > self.criteria.max_complexity_cost:
            rejection_reasons.append(RejectionReason.COMPLEXITY_LIMIT)

        # Check expected value
        if expected_value < self.criteria.min_expected_value:
            rejection_reasons.append(RejectionReason.LOW_EXPECTED_VALUE)

        # Check lookback
        if lookback_periods is not None and lookback_periods > self.criteria.max_lookback_periods:
            rejection_reasons.append(RejectionReason.COMPLEXITY_LIMIT)

        # Check domains
        if self.criteria.allowed_domains and domains:
            if not any(d in self.criteria.allowed_domains for d in domains):
                rejection_reasons.append(RejectionReason.DOMAIN_VIOLATION)

        # Check sources
        if self.criteria.allowed_sources and sources:
            if not any(s in self.criteria.allowed_sources for s in sources):
                rejection_reasons.append(RejectionReason.DOMAIN_VIOLATION)

        # Check parent quality
        if parent_quality is not None and parent_quality < self.criteria.min_parent_quality:
            rejection_reasons.append(RejectionReason.PARENT_QUALITY)

        # Determine verdict
        if rejection_reasons:
            verdict = AdmissionVerdict.REJECTED
        else:
            verdict = AdmissionVerdict.ADMITTED

        decision = AdmissionDecision(
            decision_id=decision_id,
            mutation_id=mutation_id,
            trial_id=trial_id,
            verdict=verdict,
            criteria_used=self.criteria,
            rejection_reasons=rejection_reasons,
            expected_value=expected_value,
            complexity_cost=complexity_cost,
            decision_metadata=metadata or {},
        )

        self._decision_history[decision_id] = decision
        return decision

    def get_decision(self, decision_id: str) -> Optional[AdmissionDecision]:
        """Retrieve decision by ID."""
        return self._decision_history.get(decision_id)

    def get_decisions_for_trial(self, trial_id: str) -> List[AdmissionDecision]:
        """Get all decisions for a trial."""
        return [d for d in self._decision_history.values() if d.trial_id == trial_id]

    def update_criteria(self, criteria: AdmissionCriteria) -> None:
        """Update default admission criteria."""
        self.criteria = criteria

    def get_admission_stats(self) -> Dict[str, Any]:
        """Get admission decision statistics."""
        total = len(self._decision_history)
        if total == 0:
            return {"total": 0}

        admitted = sum(1 for d in self._decision_history.values() if d.verdict == AdmissionVerdict.ADMITTED)
        rejected = sum(1 for d in self._decision_history.values() if d.verdict == AdmissionVerdict.REJECTED)
        deferred = sum(1 for d in self._decision_history.values() if d.verdict == AdmissionVerdict.DEFERRED)
        conditional = sum(1 for d in self._decision_history.values() if d.verdict == AdmissionVerdict.CONDITIONAL)

        # Count rejection reasons
        rejection_counts: Dict[str, int] = {}
        for decision in self._decision_history.values():
            for reason in decision.rejection_reasons:
                rejection_counts[reason.value] = rejection_counts.get(reason.value, 0) + 1

        return {
            "total": total,
            "admitted": admitted,
            "rejected": rejected,
            "deferred": deferred,
            "conditional": conditional,
            "admission_rate": admitted / total if total > 0 else 0.0,
            "rejection_reasons": rejection_counts,
        }

    def clear_history(self) -> None:
        """Clear decision history."""
        self._decision_history.clear()
