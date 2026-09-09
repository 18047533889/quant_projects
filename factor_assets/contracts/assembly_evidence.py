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
    "AssemblyClusterMembership",
    "GreedySelectionStep",
    "AssemblySelectionEvidence",
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
    quality_definition: str = "legacy.unspecified"
    quality_unit: str = "legacy.unspecified"
    quality_policy_ref: str = "legacy.unspecified"
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
        for name in ("quality_definition", "quality_unit", "quality_policy_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
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
            self.quality_definition,
            self.quality_unit,
            self.quality_policy_ref,
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
            "quality_definition": self.quality_definition,
            "quality_unit": self.quality_unit,
            "quality_policy_ref": self.quality_policy_ref,
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
    required_objectives: tuple[str, ...] = ()
    optional_objectives: tuple[str, ...] = ()
    selection_algorithm_version: str = "constrained_mmr.v2"
    quality_definition: str = "legacy.unspecified"
    quality_unit: str = "legacy.unspecified"
    quality_policy_ref: str = "legacy.unspecified"
    similarity_view: str = "rank_corr"

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
        if isinstance(self.max_per_microcluster, bool) or not isinstance(
            self.max_per_microcluster, int
        ):
            raise TypeError("max_per_microcluster must be an int")
        if isinstance(self.max_per_macrocluster, bool) or not isinstance(
            self.max_per_macrocluster, int
        ):
            raise TypeError("max_per_macrocluster must be an int")
        if self.max_per_microcluster < 1:
            raise ValueError("max_per_microcluster must be >= 1")
        if self.max_per_macrocluster < 1:
            raise ValueError("max_per_macrocluster must be >= 1")
        if self.max_per_microcluster > self.max_per_macrocluster:
            raise ValueError(
                "max_per_microcluster must be <= max_per_macrocluster — a "
                "microcluster is a subset of a macrocluster, so per-micro "
                "budget cannot exceed the per-macro budget"
            )
        if self.capacity_budget is not None:
            if isinstance(self.capacity_budget, bool) or not isinstance(
                self.capacity_budget, int
            ):
                raise TypeError("capacity_budget must be an int or None")
            if self.capacity_budget < 1:
                raise ValueError("capacity_budget must be >= 1 or None")
        if self.turnover_budget is not None:
            if isinstance(self.turnover_budget, bool) or not isinstance(
                self.turnover_budget, (int, float)
            ):
                raise TypeError("turnover_budget must be a non-boolean number or None")
            tb = float(self.turnover_budget)
            if tb != tb or tb in (float("inf"), float("-inf")):
                raise ValueError("turnover_budget must be finite")
            if tb < 0.0:
                raise ValueError("turnover_budget must be non-negative or None")
            object.__setattr__(self, "turnover_budget", tb)
        if self.health_floor is not None:
            if isinstance(self.health_floor, bool) or not isinstance(
                self.health_floor, (int, float)
            ):
                raise TypeError("health_floor must be a non-boolean number or None")
            hf = float(self.health_floor)
            if hf != hf or hf in (float("inf"), float("-inf")):
                raise ValueError("health_floor must be finite")
            if not 0.0 <= hf <= 1.0:
                raise ValueError("health_floor must be in [0, 1] or None")
            object.__setattr__(self, "health_floor", hf)
        required = tuple(dict.fromkeys(self.required_objectives))
        optional = tuple(dict.fromkeys(self.optional_objectives))
        if set(required) & set(optional) or any(not x for x in required + optional):
            raise ValueError("required/optional objectives must be unique non-empty names")
        object.__setattr__(self, "required_objectives", required)
        object.__setattr__(self, "optional_objectives", optional)
        for name in (
            "selection_algorithm_version", "quality_definition", "quality_unit",
            "quality_policy_ref", "similarity_view",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

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
            "required_objectives": list(self.required_objectives),
            "optional_objectives": list(self.optional_objectives),
            "selection_algorithm_version": self.selection_algorithm_version,
            "quality_definition": self.quality_definition,
            "quality_unit": self.quality_unit,
            "quality_policy_ref": self.quality_policy_ref,
            "similarity_view": self.similarity_view,
        }

    @property
    def content_hash(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True)
class AssemblyClusterMembership:
    """Actual information/risk grouping used by assembly, separate from lineage family."""

    factor_id: str
    microcluster_id: Optional[str]
    macrocluster_id: Optional[str]
    cluster_set_version_ref: str
    representative_factor_id: Optional[str] = None
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        if not self.factor_id or not self.cluster_set_version_ref:
            raise ValueError("factor_id and cluster_set_version_ref are required")
        if self.microcluster_id is None and self.macrocluster_id is None and not self.evidence_ref:
            raise ValueError("unknown grouping requires an explicit evidence/reason ref")


@dataclass(frozen=True)
class GreedySelectionStep:
    """One committed constrained-MMR choice; rejected candidates never appear."""

    rank: int
    factor_id: str
    quality: float
    max_redundancy: float
    objective: float
    constraint_headroom: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        if self.rank < 0 or not self.factor_id:
            raise ValueError("rank must be non-negative and factor_id is required")
        for name in ("quality", "max_redundancy", "objective"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        object.__setattr__(self, "constraint_headroom", MappingProxyType(dict(self.constraint_headroom)))


@dataclass(frozen=True)
class AssemblySelectionEvidence:
    """Replayable evidence from the feasibility-aware greedy selector."""

    algorithm_version: str
    policy_ref: str
    steps: tuple[GreedySelectionStep, ...]
    rejections: Mapping[str, str]
    pair_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.algorithm_version or not self.policy_ref:
            raise ValueError("algorithm_version and policy_ref are required")
        object.__setattr__(self, "steps", tuple(self.steps))
        if any(not isinstance(step, GreedySelectionStep) for step in self.steps):
            raise TypeError("steps must contain GreedySelectionStep values")
        rejected = dict(self.rejections)
        if any(not factor_id or not reason for factor_id, reason in rejected.items()):
            raise ValueError("rejection factor IDs and reasons must be non-empty")
        object.__setattr__(self, "rejections", MappingProxyType(rejected))
        object.__setattr__(self, "pair_evidence_refs", tuple(self.pair_evidence_refs))
