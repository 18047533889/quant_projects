"""Factor health-card artifact + deterministic admission-relevant structure
(R61-FI-025, plan §8-§11).

:class:`FactorHealthCardArtifact` bundles the 14 dimension grade artifacts,
typed integrity-gate results, a **display-only** overall grade and an
admission-relevant summary structure.

Display vs admission semantics (plan §9.1)
------------------------------------------
The health card keeps the two concerns **explicitly separated** in the field
naming:

- ``display_overall_grade`` (a letter) is for management UI only.  It is
  derived from the mean of graded dimension scores under the policy's display
  score bands and is **never** an admission authority.
- ``admission_relevant`` carries the structures that *actually* gate
  admission: typed integrity-gate results (:class:`IntegrityGateResult`),
  per-dimension floor violations, and the hard-gate dimension failures.
  Admission is hard gates + dimension floors + Pareto (plan §9.1); no single
  weighted-average grade ever decides admission.

Missing evidence is explicit and fail-closed: an integrity gate with missing /
failed evidence is ``passed=False`` (never a silent pass); an ungraded
hard-gate dimension fails the card regardless of other dimensions; a dimension
with no evidence is marked ``evidence_tier=MISSING`` and is *not* given a zero
score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from factor_assets.profiling.dimensions import (
    EVIDENCE_TIER_MISSING,
    HEALTH_DIMENSIONS,
    DimensionGradeArtifact,
)
from factor_assets.profiling.policies import (
    INTEGRITY_GATE_IDS,
    HealthGradeVocabulary,
    get_health_policy,
)

__all__ = [
    "INTEGRITY_GATE_IDS",
    "IntegrityGateResult",
    "AdmissionSummary",
    "FactorHealthCardArtifact",
    "build_health_card",
]


# ---------------------------------------------------------------------------
# Integrity gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegrityGateResult:
    """Result of one typed integrity gate (plan §13.1).

    ``gate_id`` is one of the typed cross-package gates (``pit_valid``,
    ``label_maturity_ok``, ``snapshot_identity_ok``, ``universe_identity_ok``,
    ``finite_shape_schema_ok``, ``unit_scale_ok``, ``return_basis_ok``).
    ``passed`` is a tri-state: ``True`` (explicit pass), ``False`` (explicit
    fail) or ``None`` (no evidence either way — treated as a fail by the card,
    never a silent pass).  ``evidence_ref`` pins the cross-package integrity
    evidence that produced the result.
    """

    gate_id: str
    passed: bool | None
    evidence_ref: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.gate_id not in INTEGRITY_GATE_IDS:
            raise ValueError(
                f"unknown integrity gate {self.gate_id!r}; expected one of "
                f"{INTEGRITY_GATE_IDS}"
            )
        if self.passed is not None and not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool, None, or absent")
        if not isinstance(self.evidence_ref, str):
            raise TypeError("evidence_ref must be a string")
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a string")

    @property
    def is_hard_fail(self) -> bool:
        """True iff the gate did not explicitly pass (fail or missing)."""
        return self.passed is not True


@dataclass(frozen=True)
class AdmissionSummary:
    """Admission-relevant summary (plan §9.1) — *not* a weighted grade.

    ``hard_gates_passed`` is True only when every typed integrity gate
    explicitly passed.  ``dimension_floor_violations`` lists the dimension ids
    whose grade is below their policy floor (or whose evidence is missing /
    ungraded when the policy requires all dimensions graded).
    ``hard_gate_dimension_failures`` lists the hard-gate dimensions that are
    ungraded / below floor.  ``admissible`` is the fail-closed conjunction:
    all integrity gates passed AND no hard-gate dimension failed AND no floor
    violation.  It is a *summary* flag for callers that want one, not a
    replacement for the FO Pareto / multi-dimensional decision.
    """

    hard_gates_passed: bool
    dimension_floor_violations: tuple[str, ...]
    hard_gate_dimension_failures: tuple[str, ...]
    ungraded_dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dimension_floor_violations", tuple(self.dimension_floor_violations)
        )
        object.__setattr__(
            self,
            "hard_gate_dimension_failures",
            tuple(self.hard_gate_dimension_failures),
        )
        object.__setattr__(self, "ungraded_dimensions", tuple(self.ungraded_dimensions))
        if not isinstance(self.hard_gates_passed, bool):
            raise TypeError("hard_gates_passed must be a bool")

    @property
    def admissible(self) -> bool:
        return (
            self.hard_gates_passed
            and not self.hard_gate_dimension_failures
            and not self.dimension_floor_violations
        )


# ---------------------------------------------------------------------------
# Health card artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorHealthCardArtifact:
    """Frozen factor health card (plan §8/§9).

    Fields
    ------
    factor_definition_id / evaluation_ref:
        Identities.
    dimension_grades:
        The 14 :class:`DimensionGradeArtifact` records (canonical order).
    integrity_gates:
        Typed integrity-gate results (plan §13.1).
    display_overall_grade:
        **Display-only** letter for management UI.  Derived from the mean of
        the graded dimension scores under the policy's display score bands.
        Never an admission authority (plan §9.1).
    display_overall_score:
        The mean graded-dimension score (0..100) behind the display grade
        (``None`` when no dimension was graded).
    admission_relevant:
        The :class:`AdmissionSummary` carrying gates / floor violations /
        hard-gate failures — the actual admission-facing structure.
    health_policy_id / health_policy_version:
        The policy this card was built under.
    """

    factor_definition_id: str
    evaluation_ref: str
    dimension_grades: tuple[DimensionGradeArtifact, ...]
    integrity_gates: tuple[IntegrityGateResult, ...]
    display_overall_grade: str
    display_overall_score: float | None
    admission_relevant: AdmissionSummary
    health_policy_id: str
    health_policy_version: str

    def __post_init__(self) -> None:
        if not self.factor_definition_id:
            raise ValueError("factor_definition_id is required")
        if not self.evaluation_ref:
            raise ValueError("evaluation_ref is required")
        object.__setattr__(self, "dimension_grades", tuple(self.dimension_grades))
        object.__setattr__(self, "integrity_gates", tuple(self.integrity_gates))
        if len(self.dimension_grades) != len(HEALTH_DIMENSIONS):
            raise ValueError(
                "health card requires exactly one dimension artifact per health "
                f"dimension (got {len(self.dimension_grades)} for "
                f"{len(HEALTH_DIMENSIONS)})"
            )
        seen_dims = [d.dimension_id for d in self.dimension_grades]
        if seen_dims != list(HEALTH_DIMENSIONS):
            raise ValueError(
                "dimension_grades must be in canonical dimension order "
                f"({HEALTH_DIMENSIONS[0]}..{HEALTH_DIMENSIONS[-1]}), got {seen_dims}"
            )
        for gate in self.integrity_gates:
            if not isinstance(gate, IntegrityGateResult):
                raise TypeError("integrity_gates must contain IntegrityGateResult")
        if self.display_overall_grade not in HealthGradeVocabulary.all():
            raise ValueError(
                f"unknown display overall grade {self.display_overall_grade!r}"
            )
        if self.display_overall_score is not None:
            s = float(self.display_overall_score)
            if math.isnan(s) or math.isinf(s) or not 0.0 <= s <= 100.0:
                raise ValueError("display_overall_score must be in [0, 100] or None")
        if not isinstance(self.admission_relevant, AdmissionSummary):
            raise TypeError("admission_relevant must be an AdmissionSummary")
        if not self.health_policy_id or not self.health_policy_version:
            raise ValueError("health policy id/version are required")

    def dimension(self, dimension_id: str) -> DimensionGradeArtifact | None:
        for d in self.dimension_grades:
            if d.dimension_id == dimension_id:
                return d
        return None

    def gate(self, gate_id: str) -> IntegrityGateResult | None:
        for g in self.integrity_gates:
            if g.gate_id == gate_id:
                return g
        return None

    @property
    def display_grade_is_display_only(self) -> bool:
        """Semantic marker: the display overall grade is never an admission authority."""
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_definition_id": self.factor_definition_id,
            "evaluation_ref": self.evaluation_ref,
            "dimension_grades": [d.to_dict() for d in self.dimension_grades],
            "integrity_gates": [
                {
                    "gate_id": g.gate_id,
                    "passed": g.passed,
                    "evidence_ref": g.evidence_ref,
                    "detail": g.detail,
                }
                for g in self.integrity_gates
            ],
            "display_overall_grade": self.display_overall_grade,
            "display_overall_score": self.display_overall_score,
            "admission_relevant": {
                "hard_gates_passed": self.admission_relevant.hard_gates_passed,
                "dimension_floor_violations": list(
                    self.admission_relevant.dimension_floor_violations
                ),
                "hard_gate_dimension_failures": list(
                    self.admission_relevant.hard_gate_dimension_failures
                ),
                "ungraded_dimensions": list(
                    self.admission_relevant.ungraded_dimensions
                ),
                "admissible": self.admission_relevant.admissible,
            },
            "health_policy_id": self.health_policy_id,
            "health_policy_version": self.health_policy_version,
        }


# ---------------------------------------------------------------------------
# Deterministic health-card builder
# ---------------------------------------------------------------------------


def _integrity_results(
    gate_results: Mapping[str, bool | None] | Sequence[IntegrityGateResult],
    gate_details: Mapping[str, str] | None = None,
) -> tuple[IntegrityGateResult, ...]:
    details = dict(gate_details or {})
    if isinstance(gate_results, Sequence) and not isinstance(gate_results, (str, bytes)):
        return tuple(gate_results)
    out: list[IntegrityGateResult] = []
    for gate_id in INTEGRITY_GATE_IDS:
        passed = gate_results.get(gate_id) if isinstance(gate_results, Mapping) else None
        out.append(
            IntegrityGateResult(
                gate_id=gate_id,
                passed=passed,
                evidence_ref=f"integrity:{gate_id}",
                detail=details.get(gate_id, ""),
            )
        )
    return tuple(out)


def build_health_card(
    *,
    factor_definition_id: str,
    evaluation_ref: str,
    dimension_grades: Sequence[DimensionGradeArtifact],
    integrity_gates: Mapping[str, bool | None]
    | Sequence[IntegrityGateResult] = (),
    policy=None,
    gate_details: Mapping[str, str] | None = None,
) -> FactorHealthCardArtifact:
    """Deterministically assemble a frozen health card under a health policy.

    Missing / failed integrity gates fail closed (``passed is not True`` ->
    card not hard-gates-passed).  Hard-gate dimensions (policy
    ``admission_floors.hard_gate_dimensions``) that are ungraded or below
    their floor fail the admission-relevant summary.  When the policy requires
    all dimensions graded, ungraded dimensions are floor violations.  The
    display overall grade is a mean-of-graded-scores display letter only.
    """
    if policy is None:
        policy = get_health_policy()
    dims = tuple(dimension_grades)
    gates = _integrity_results(integrity_gates, gate_details)

    # -- display overall (mean of graded dimension scores; never admission) --
    graded_scores = [d.score for d in dims if d.score is not None]
    if graded_scores:
        display_score = float(sum(graded_scores) / len(graded_scores))
        display_grade = policy.display_grade_for_score(display_score)
    else:
        display_score = None
        display_grade = HealthGradeVocabulary.NONE

    # -- admission-relevant structure ---------------------------------------
    ungraded: list[str] = []
    floor_violations: list[str] = []
    hard_gate_failures: list[str] = []
    floors = policy.admission_floors
    for dim in dims:
        if dim.evidence_tier == EVIDENCE_TIER_MISSING or dim.grade is None:
            ungraded.append(dim.dimension_id)
            if policy.is_hard_gate_dimension(dim.dimension_id):
                hard_gate_failures.append(dim.dimension_id)
            elif floors.require_all_dimensions_graded:
                floor_violations.append(dim.dimension_id)
            continue
        floor = policy.dimension_floor(dim.dimension_id)
        if floor is not None and dim.grade not in HealthGradeVocabulary.RANKED:
            # ungraded handled above; this is defensive
            floor_violations.append(dim.dimension_id)
            continue
        if floor is not None:
            if HealthGradeVocabulary.rank(dim.grade) > HealthGradeVocabulary.rank(floor):
                floor_violations.append(dim.dimension_id)
                if policy.is_hard_gate_dimension(dim.dimension_id):
                    hard_gate_failures.append(dim.dimension_id)
        elif policy.is_hard_gate_dimension(dim.dimension_id):
            # hard-gate dimension with no explicit floor: must be >= B to pass
            if HealthGradeVocabulary.rank(dim.grade) > HealthGradeVocabulary.rank("B"):
                hard_gate_failures.append(dim.dimension_id)

    hard_gates_passed = all(g.passed is True for g in gates)
    summary = AdmissionSummary(
        hard_gates_passed=hard_gates_passed,
        dimension_floor_violations=tuple(floor_violations),
        hard_gate_dimension_failures=tuple(hard_gate_failures),
        ungraded_dimensions=tuple(ungraded),
    )

    return FactorHealthCardArtifact(
        factor_definition_id=factor_definition_id,
        evaluation_ref=evaluation_ref,
        dimension_grades=dims,
        integrity_gates=gates,
        display_overall_grade=display_grade,
        display_overall_score=display_score,
        admission_relevant=summary,
        health_policy_id=policy.policy_id,
        health_policy_version=policy.policy_version,
    )
