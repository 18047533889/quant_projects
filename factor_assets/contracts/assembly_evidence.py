"""Typed assembly evidence contracts for factor_assets (DLIB-FA-013).

The legacy :class:`~factor_assets.selection.policy.SelectionDecision.metadata`
is stringly-typed (``Mapping[str, str]``) but holds floats and nested values.
These typed evidence artifacts replace that ad-hoc metadata for production
assembly:

- :class:`AssemblyEvidence` — the typed per-factor evidence that justifies a
  library/assembly membership.
- :class:`LibraryCandidateEvidence` — the typed evidence for a library
  candidate, with the full multi-dimensional score set.

Production assembly must NOT fall back to recency when quality is missing —
it must be UNKNOWN / NOT_ELIGIBLE.  Research may keep recency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional

from factor_assets.contracts._canonical import canonical_digest

__all__ = [
    "EvidenceMaturity",
    "AssemblyEvidence",
    "LibraryCandidateEvidence",
    "AssemblyPolicy",
]


class EvidenceMaturity(Enum):
    """Maturity of the evidence backing an assembly decision (DLIB-FA-013).

    Production assembly must NOT fall back to recency when quality is missing
    — it must be UNKNOWN / NOT_ELIGIBLE.  Research may keep recency.
    """

    MATURE = "MATURE"
    IMMATURE = "IMMATURE"
    UNKNOWN = "UNKNOWN"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a non-boolean number")
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return v


@dataclass(frozen=True)
class AssemblyEvidence:
    """Typed per-factor evidence for an assembly decision (DLIB-FA-013).

    Replaces the stringly-typed ``SelectionDecision.metadata`` for production
    assembly.  ``quality_score`` is required and must be finite; a missing
    quality is NOT_ELIGIBLE, never a recency fallback.
    """

    factor_id: str
    quality_score: float
    predictive_score: Optional[float] = None
    stability_score: Optional[float] = None
    robustness_score: Optional[float] = None
    residual_novelty: Optional[float] = None
    turnover_score: Optional[float] = None
    cost_score: Optional[float] = None
    capacity_score: Optional[float] = None
    health_score: Optional[float] = None
    worst_slice_score: Optional[float] = None
    cluster_redundancy: Optional[float] = None
    evidence_refs: tuple[str, ...] = ()
    maturity: EvidenceMaturity = EvidenceMaturity.MATURE
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        object.__setattr__(self, "quality_score", _finite_float(self.quality_score, "quality_score"))
        for name in (
            "predictive_score", "stability_score", "robustness_score",
            "residual_novelty", "turnover_score", "cost_score", "capacity_score",
            "health_score", "worst_slice_score", "cluster_redundancy",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_float(value, name))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if not isinstance(self.maturity, EvidenceMaturity):
            raise TypeError("maturity must be an EvidenceMaturity")
        computed = canonical_digest(
            self.factor_id,
            self.quality_score,
            self.predictive_score,
            self.stability_score,
            self.robustness_score,
            self.residual_novelty,
            self.turnover_score,
            self.cost_score,
            self.capacity_score,
            self.health_score,
            self.worst_slice_score,
            self.cluster_redundancy,
            self.evidence_refs,
            self.maturity.value,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed assembly-evidence "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "quality_score": self.quality_score,
            "predictive_score": self.predictive_score,
            "stability_score": self.stability_score,
            "robustness_score": self.robustness_score,
            "residual_novelty": self.residual_novelty,
            "turnover_score": self.turnover_score,
            "cost_score": self.cost_score,
            "capacity_score": self.capacity_score,
            "health_score": self.health_score,
            "worst_slice_score": self.worst_slice_score,
            "cluster_redundancy": self.cluster_redundancy,
            "evidence_refs": list(self.evidence_refs),
            "maturity": self.maturity.value,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class LibraryCandidateEvidence:
    """Typed evidence for a library candidate (DLIB-FA-013).

    The full multi-dimensional score set used by production assembly.  A
    missing quality is NOT_ELIGIBLE, never a recency fallback.
    """

    factor_definition_ref: str
    quality_score: float
    predictive_score: Optional[float] = None
    stability_score: Optional[float] = None
    robustness_score: Optional[float] = None
    residual_novelty: Optional[float] = None
    turnover_score: Optional[float] = None
    cost_score: Optional[float] = None
    capacity_score: Optional[float] = None
    health_score: Optional[float] = None
    worst_slice_score: Optional[float] = None
    cluster_redundancy: Optional[float] = None
    evidence_refs: tuple[str, ...] = ()
    maturity: EvidenceMaturity = EvidenceMaturity.MATURE
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.factor_definition_ref:
            raise ValueError("factor_definition_ref is required")
        object.__setattr__(self, "quality_score", _finite_float(self.quality_score, "quality_score"))
        for name in (
            "predictive_score", "stability_score", "robustness_score",
            "residual_novelty", "turnover_score", "cost_score", "capacity_score",
            "health_score", "worst_slice_score", "cluster_redundancy",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_float(value, name))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if not isinstance(self.maturity, EvidenceMaturity):
            raise TypeError("maturity must be an EvidenceMaturity")
        computed = canonical_digest(
            self.factor_definition_ref,
            self.quality_score,
            self.predictive_score,
            self.stability_score,
            self.robustness_score,
            self.residual_novelty,
            self.turnover_score,
            self.cost_score,
            self.capacity_score,
            self.health_score,
            self.worst_slice_score,
            self.cluster_redundancy,
            self.evidence_refs,
            self.maturity.value,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed library-candidate "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "factor_definition_ref": self.factor_definition_ref,
            "quality_score": self.quality_score,
            "predictive_score": self.predictive_score,
            "stability_score": self.stability_score,
            "robustness_score": self.robustness_score,
            "residual_novelty": self.residual_novelty,
            "turnover_score": self.turnover_score,
            "cost_score": self.cost_score,
            "capacity_score": self.capacity_score,
            "health_score": self.health_score,
            "worst_slice_score": self.worst_slice_score,
            "cluster_redundancy": self.cluster_redundancy,
            "evidence_refs": list(self.evidence_refs),
            "maturity": self.maturity.value,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class AssemblyPolicy:
    """Versioned assembly policy (DLIB-FA-013).

    Replaces the magic ``lambda_weight=0.5`` in the MMR diverse policy with a
    versioned policy carrying quality_weight, redundancy_weight,
    max_per_microcluster, max_per_macrocluster, capacity_budget,
    turnover_budget, and health_floor.
    """

    policy_id: str
    version: str
    quality_weight: float = 0.5
    redundancy_weight: float = 0.5
    max_per_microcluster: int = 1
    max_per_macrocluster: int = 5
    capacity_budget: Optional[int] = None
    turnover_budget: Optional[float] = None
    health_floor: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if not self.version:
            raise ValueError("version is required")
        for name in ("quality_weight", "redundancy_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a non-boolean number")
            v = float(value)
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, v)
        if self.max_per_microcluster < 1:
            raise ValueError("max_per_microcluster must be >= 1")
        if self.max_per_macrocluster < 1:
            raise ValueError("max_per_macrocluster must be >= 1")

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "quality_weight": self.quality_weight,
            "redundancy_weight": self.redundancy_weight,
            "max_per_microcluster": self.max_per_microcluster,
            "max_per_macrocluster": self.max_per_macrocluster,
            "capacity_budget": self.capacity_budget,
            "turnover_budget": self.turnover_budget,
            "health_floor": self.health_floor,
        }
