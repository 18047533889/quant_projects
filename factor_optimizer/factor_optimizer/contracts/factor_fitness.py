"""FactorFitnessSpec + CandidateFitnessArtifact (R61-FI-031 / plan §21).

The search may internally use scalar utilities (:mod:`factor_optimizer.search.
winner_selector` RobustBalancedUtility and friends) to *rank* candidates, but
**admission and winner authority must pass through this contract** (plan §21 /
matrix E1): a FactorFitnessSpec names which policies are in force (health
dimension floors, raw-relative deltas, uncertainty, multiplicity, complexity,
winner) and a CandidateFitnessArtifact is the immutable per-candidate record
the decision pipeline produces and the treatment-result artifact consumes.

Contract fields
---------------
``FactorFitnessSpec`` is frozen and carries the policy ids of every step in
the 8-step decision pipeline, plus the set of health dimensions whose floors
are enforced and the minimum evidence tier a candidate must reach to be a
valid winner:

- ``evidence_profile_id``        QE evidence-profile binding (what metrics /
                                 statuses exist for a candidate).
- ``required_health_dimensions`` the FA 14-dim health dimensions the policy
                                 enforces (subset of
                                 ``HealthDimension.all()``).
- ``hard_gate_policy_id``        integrity / hard-gate policy in force.
- ``dimension_floor_policy_id``  floor policy for the required health dims.
- ``raw_relative_policy_id``     RAW-relative delta policy (RAW = baseline).
- ``uncertainty_policy_id``      uncertainty config policy.
- ``multiplicity_policy_id``     multiple-testing policy.
- ``complexity_policy_id``       complexity-preference policy.
- ``winner_policy_id``           winner-selection policy.
- ``minimum_evidence_tier``      weakest :class:`EvidenceTier` a winner may
                                 carry.

``CandidateFitnessArtifact`` is the frozen per-candidate record:

- ``trial_id``                   candidate identity.
- ``evaluation_ref``             QE evaluation bundle ref.
- ``health_card_ref``            FA health-card ref.
- ``raw_relative_deltas``        mapping dimension/metric -> delta, each delta
                                 NORMALIZED so positive = better-than-RAW
                                 (lower-better raw metrics such as turnover /
                                 exposure / drawdown have their delta sign
                                 flipped).  A ``None`` delta means the RAW
                                 baseline or the treatment value was missing
                                 (missing != 0).  Immutable — deltas are
                                 evidence, not just a log line.
- ``evidence_tier``              the candidate's achieved tier (the weakest of
                                 its load-bearing evidence).
- ``status``                     :class:`EvidenceStatus` of the candidate's
                                 evidence bundle.
- ``dimension_scores``           optional per-health-dimension desirabilities
                                 (0..1 higher-better), absent dims are simply
                                 not present.

No grading rule or numeric threshold lives in this contract; all floors and
anchors are owned by the referenced policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from factor_optimizer.contracts.evidence_value import (
    EvidenceStatus,
    EvidenceTier,
)
from factor_optimizer.ports.factor_intelligence import HealthDimension

__all__ = [
    "FactorFitnessSpec",
    "CandidateFitnessArtifact",
    "MIN_EVIDENCE_TIER_LEVELS",
]

#: canonical health dimension ids this contract understands (mirror of the FA
#: authority; FA extends, never FO — an unknown id fails loudly at validation).
_HEALTH_DIMENSION_IDS = frozenset(HealthDimension.all())

#: Recognized evidence tier level names (weakest -> strongest).  Kept in sync
#: with :class:`EvidenceTier`.
MIN_EVIDENCE_TIER_LEVELS = (
    "POINT_ESTIMATE_ONLY",
    "VALIDATION_SERIES",
    "BOOTSTRAP_CONFIDENCE",
    "MULTIPLE_TESTING_ADJUSTED",
    "SEALED_TEST_CONFIRMED",
)


@dataclass(frozen=True)
class FactorFitnessSpec:
    """Frozen contract naming every policy id the decision pipeline applies.

    Attributes:
        evidence_profile_id: QE evidence-profile ref binding the metric set
            and status vocabulary a candidate's evidence is read against.
        required_health_dimensions: subset of the 14 FA health dimensions
            whose floors the policy enforces.  Empty tuple = no health floor
            is enforced (research mode).
        hard_gate_policy_id: integrity hard-gate policy ref.
        dimension_floor_policy_id: dimension-floor policy ref.
        raw_relative_policy_id: RAW-relative delta policy ref.
        uncertainty_policy_id: uncertainty (bootstrap / CI) policy ref.
        multiplicity_policy_id: multiple-testing policy ref.
        complexity_policy_id: complexity-preference policy ref.
        winner_policy_id: winner-selection policy ref.
        minimum_evidence_tier: weakest EvidenceTier a valid winner may carry.
    """

    evidence_profile_id: str = "QE_EVIDENCE_PROFILE"
    required_health_dimensions: tuple[str, ...] = ()
    hard_gate_policy_id: str = ""
    dimension_floor_policy_id: str = ""
    raw_relative_policy_id: str = ""
    uncertainty_policy_id: str = ""
    multiplicity_policy_id: str = ""
    complexity_policy_id: str = ""
    winner_policy_id: str = ""
    minimum_evidence_tier: EvidenceTier = EvidenceTier.POINT_ESTIMATE_ONLY

    def __post_init__(self) -> None:
        for name in (
            "evidence_profile_id",
            "hard_gate_policy_id",
            "dimension_floor_policy_id",
            "raw_relative_policy_id",
            "uncertainty_policy_id",
            "multiplicity_policy_id",
            "complexity_policy_id",
            "winner_policy_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} must be a non-empty policy/ref string"
                )
        dims = tuple(self.required_health_dimensions)
        unknown = set(dims) - _HEALTH_DIMENSION_IDS
        if unknown:
            raise ValueError(
                f"unknown required health dimension(s): {sorted(unknown)}; "
                f"valid dims are the 14 FA HealthDimension ids"
            )
        object.__setattr__(self, "required_health_dimensions", dims)
        object.__setattr__(
            self,
            "minimum_evidence_tier",
            EvidenceTier.from_value(self.minimum_evidence_tier),
        )

    def requires_evidence_tier(self, tier: EvidenceTier) -> bool:
        """True when ``tier`` meets this spec's minimum evidence tier."""
        return EvidenceTier.from_value(tier).meets(self.minimum_evidence_tier)

    def to_dict(self) -> dict:
        return {
            "evidence_profile_id": self.evidence_profile_id,
            "required_health_dimensions": list(self.required_health_dimensions),
            "hard_gate_policy_id": self.hard_gate_policy_id,
            "dimension_floor_policy_id": self.dimension_floor_policy_id,
            "raw_relative_policy_id": self.raw_relative_policy_id,
            "uncertainty_policy_id": self.uncertainty_policy_id,
            "multiplicity_policy_id": self.multiplicity_policy_id,
            "complexity_policy_id": self.complexity_policy_id,
            "winner_policy_id": self.winner_policy_id,
            "minimum_evidence_tier": self.minimum_evidence_tier.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FactorFitnessSpec":
        return cls(
            evidence_profile_id=data.get("evidence_profile_id", "QE_EVIDENCE_PROFILE"),
            required_health_dimensions=tuple(data.get("required_health_dimensions", ())),
            hard_gate_policy_id=data.get("hard_gate_policy_id", ""),
            dimension_floor_policy_id=data.get("dimension_floor_policy_id", ""),
            raw_relative_policy_id=data.get("raw_relative_policy_id", ""),
            uncertainty_policy_id=data.get("uncertainty_policy_id", ""),
            multiplicity_policy_id=data.get("multiplicity_policy_id", ""),
            complexity_policy_id=data.get("complexity_policy_id", ""),
            winner_policy_id=data.get("winner_policy_id", ""),
            minimum_evidence_tier=data.get(
                "minimum_evidence_tier", "POINT_ESTIMATE_ONLY"
            ),
        )


@dataclass(frozen=True)
class CandidateFitnessArtifact:
    """Immutable per-candidate fitness record the pipeline consumes/produces.

    Attributes:
        trial_id: Candidate identity (e.g. ``RAW``, ``EWMA``).
        evaluation_ref: QE evaluation bundle ref (required, may be a bare id).
        health_card_ref: FA health-card ref (required, may be a bare id).
        raw_relative_deltas: mapping dimension/metric -> delta normalized so
            positive = better-than-RAW; ``None`` = a side was missing
            (missing != 0).
        evidence_tier: the candidate's achieved :class:`EvidenceTier`.
        status: :class:`EvidenceStatus` of the candidate's evidence bundle.
        dimension_scores: optional per-health-dimension desirabilities.
    """

    trial_id: str
    evaluation_ref: str
    health_card_ref: str
    raw_relative_deltas: Mapping[str, Optional[float]] = field(default_factory=dict)
    evidence_tier: EvidenceTier = EvidenceTier.POINT_ESTIMATE_ONLY
    status: EvidenceStatus = EvidenceStatus.COMPUTED
    dimension_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string")
        for name in ("evaluation_ref", "health_card_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty ref string")
        deltas = dict(self.raw_relative_deltas)
        for key, value in deltas.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("raw_relative_deltas keys must be non-empty strings")
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        f"raw_relative_delta {key!r} must be numeric or None"
                    )
                v = float(value)
                if v != v or v in (float("inf"), float("-inf")):
                    raise ValueError(
                        f"raw_relative_delta {key!r} must be finite (missing "
                        "deltas are None, never NaN)"
                    )
                deltas[key] = v
        dims = dict(self.dimension_scores)
        for key, value in dims.items():
            if key not in _HEALTH_DIMENSION_IDS:
                raise ValueError(f"unknown health dimension in scores: {key!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"dimension score {key!r} must be numeric")
            v = float(value)
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError(f"dimension score {key!r} must be finite")
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"dimension score {key!r} must be in [0, 1]")
            dims[key] = v
        object.__setattr__(self, "raw_relative_deltas", deltas)
        object.__setattr__(self, "dimension_scores", dims)
        object.__setattr__(
            self, "evidence_tier", EvidenceTier.from_value(self.evidence_tier)
        )
        object.__setattr__(self, "status", EvidenceStatus.from_value(self.status))
        if self.status is EvidenceStatus.COMPUTED:
            # A COMPUTED status is the only legal carrier of the "winner may
            # be chosen on this evidence" semantics; nothing further to check.
            pass

    def delta(self, name: str) -> Optional[float]:
        """Raw-relative delta for one dimension/metric (None when missing)."""
        return self.raw_relative_deltas.get(name)

    def meets_evidence_tier(self, minimum: EvidenceTier) -> bool:
        """True when this artifact's tier meets ``minimum``."""
        return self.evidence_tier.meets(minimum)

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "evaluation_ref": self.evaluation_ref,
            "health_card_ref": self.health_card_ref,
            "raw_relative_deltas": dict(self.raw_relative_deltas),
            "evidence_tier": self.evidence_tier.value,
            "status": self.status.value,
            "dimension_scores": dict(self.dimension_scores),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CandidateFitnessArtifact":
        return cls(
            trial_id=data["trial_id"],
            evaluation_ref=data.get("evaluation_ref", ""),
            health_card_ref=data.get("health_card_ref", ""),
            raw_relative_deltas=dict(data.get("raw_relative_deltas", {})),
            evidence_tier=data.get("evidence_tier", "POINT_ESTIMATE_ONLY"),
            status=data.get("status", "computed"),
            dimension_scores=dict(data.get("dimension_scores", {})),
        )
