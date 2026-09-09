"""Factor Intelligence provider port (R61-FI-014 / plan §20 E6, matrix E6).

The narrow, **consumer-side** boundary through which FO reads factor
intelligence (taxonomy / health / diagnosis) from the canonical authority.

Ownership & authority
---------------------
- FA (``factor_assets``) is the canonical owner of factor taxonomy and health:
  the deterministic taxonomy classifier lives in ``factor_assets.profiling``
  (``FactorTaxonomyArtifact`` / ``DomainTagSet`` / ``TagEvidence`` /
  ``TaxonomyPolicy``) and the versioned health/diagnosis engines are FA-side.
  This module defines **views over FA artifacts**, never parallel grading
  rules.
- FO never writes grading rules, thresholds or magic confidence numbers here.
  Views are opaque, frozen snapshots consumed by FO policies (e.g. the
  treatment decision policy in Wave 3).  `get_taxonomy` only projects the
  narrow fields FO needs; the FA artifact stays authoritative.
- Unknown factors fail explicitly: the provider either returns an explicit
  ``UNKNOWN`` view (``is_unknown``) or raises the typed
  :class:`FactorIntelligenceUnknownFactorError`; it must never silently return
  ``None`` (the treatment decision pipeline treats missing-evidence and
  unknown as distinct, non-defaulted states).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

# ---------------------------------------------------------------------------
# Health vocabulary
# ---------------------------------------------------------------------------


class HealthGrade:
    """Grade letters with S+ best, D worst (plan §7 C5 / matrix C5).

    The 14 health dimensions share a single grade alphabet.  ``NONE`` marks an
    explicitly absent/unknown grade (a grade must never be invented on the FO
    side).
    """

    S_PLUS = "S+"
    S = "S"
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"
    D = "D"
    NONE = "NONE"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (cls.S_PLUS, cls.S, cls.A_PLUS, cls.A, cls.B_PLUS, cls.B, cls.C, cls.D, cls.NONE)


class HealthDimension:
    """Canonical 14 health dimensions (plan §7/§8, matrix C5 '14 dims').

    These are the dimension ids FO's treatment decision policy may consume in
    Wave 3.  The list is **closed** on the consumer boundary — FA (not FO)
    extends the canonical dimension set; an unknown id fails loudly when the
    view is validated.
    """

    PREDICTIVE_POWER = "predictive_power"
    STABILITY = "stability"
    ROBUSTNESS = "robustness"
    TURNOVER = "turnover"
    CAPACITY = "capacity"
    COST_DRAG = "cost_drag"
    DRAWDOWN = "drawdown"
    TAIL_RISK = "tail_risk"
    DATA_COVERAGE = "data_coverage"
    FRESHNESS = "freshness"
    COMPLEXITY = "complexity"
    ECONOMIC_SENSE = "economic_sense"
    SHAPE_QUALITY = "shape_quality"
    REGIME_SENSITIVITY = "regime_sensitivity"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (
            cls.PREDICTIVE_POWER,
            cls.STABILITY,
            cls.ROBUSTNESS,
            cls.TURNOVER,
            cls.CAPACITY,
            cls.COST_DRAG,
            cls.DRAWDOWN,
            cls.TAIL_RISK,
            cls.DATA_COVERAGE,
            cls.FRESHNESS,
            cls.COMPLEXITY,
            cls.ECONOMIC_SENSE,
            cls.SHAPE_QUALITY,
            cls.REGIME_SENSITIVITY,
        )


class DiagnosisSeverity:
    """Diagnosis severity bands (plan §9 C7)."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (cls.CRITICAL, cls.HIGH, cls.MEDIUM, cls.LOW, cls.INFO, cls.UNKNOWN)


class DiagnosisRepairability:
    """Whether/how a diagnosis can be repaired (plan §9 / FO repair vocabulary)."""

    REPAIRABLE = "REPAIRABLE"
    PARTIALLY_REPAIRABLE = "PARTIALLY_REPAIRABLE"
    UNREPAIRABLE = "UNREPAIRABLE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (cls.REPAIRABLE, cls.PARTIALLY_REPAIRABLE, cls.UNREPAIRABLE, cls.UNKNOWN)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorTaxonomyView:
    """Consumer view of a factor's taxonomy (narrow projection of the FA artifact).

    Only the three fields FO's policy layer actually consumes are projected
    (plan §7 C2/C3): the data-domain set, the derived display family, and the
    deterministic mechanism tags.  The full FA ``FactorTaxonomyArtifact``
    (structure tags, frequency tags, field/operator usage refs, policy version,
    content hash) stays FA-side; this view is a lossless *projection* of those
    three, not a parallel taxonomy.

    ``mechanism_tags`` are ``(tag, source, confidence, evidence_refs)`` tuples
    mirroring the FA ``TagEvidence`` records.  ``is_unknown`` is ``True`` only
    for an explicit unknown-factor view (never a default).
    """

    factor_definition_id: str
    data_domains: tuple[str, ...] = ()
    display_family: str = "UNKNOWN"
    mechanism_tags: tuple[tuple[str, str, float, tuple[str, ...]], ...] = ()
    is_unknown: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.factor_definition_id, str) or not self.factor_definition_id:
            raise ValueError("factor_definition_id must be a non-empty string")
        object.__setattr__(self, "data_domains", tuple(self.data_domains))
        object.__setattr__(self, "mechanism_tags", tuple(self.mechanism_tags))
        for tag in self.mechanism_tags:
            name, source, confidence, refs = tag
            if not isinstance(name, str) or not name:
                raise ValueError("mechanism tag name must be a non-empty string")
            if not isinstance(source, str) or not source:
                raise ValueError("mechanism tag source must be a non-empty string")
            conf = float(confidence)
            if conf != conf or conf in (float("inf"), float("-inf")):
                raise ValueError("mechanism tag confidence must be finite")
            if not 0.0 <= conf <= 1.0:
                raise ValueError("mechanism tag confidence must be in [0, 1]")
            if not isinstance(refs, (tuple, list)):
                raise ValueError("mechanism tag evidence_refs must be a sequence of strings")

    @property
    def mechanism_tag_names(self) -> tuple[str, ...]:
        """Tag names only (no evidence) for quick checks."""
        return tuple(tag[0] for tag in self.mechanism_tags)


@dataclass(frozen=True)
class FactorHealthView:
    """Consumer view of a factor's health card (plan §7/§8, matrix C5).

    ``dimension_grades`` maps each of the 14 canonical dimensions to a
    :class:`HealthGrade`.  ``overall_grade`` is the FA-assigned overall grade
    (never recomputed by FO).  ``is_unknown`` marks an explicit unknown-factor
    health view.
    """

    factor_definition_id: str
    evaluation_ref: str = ""
    dimension_grades: dict[str, str] = None  # type: ignore[assignment]
    overall_grade: str = HealthGrade.NONE
    is_unknown: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.factor_definition_id, str) or not self.factor_definition_id:
            raise ValueError("factor_definition_id must be a non-empty string")
        grades = dict(self.dimension_grades or {})
        unknown = {name: HealthGrade.NONE for name in HealthDimension.all()}
        if grades:
            unknown_keys = set(grades) - set(HealthDimension.all())
            if unknown_keys:
                raise ValueError(
                    f"unknown health dimension(s): {sorted(unknown_keys)}"
                )
            merged = dict(unknown)
            merged.update(grades)
            grades = merged
        else:
            grades = unknown
        object.__setattr__(self, "dimension_grades", grades)
        overall = self.overall_grade or HealthGrade.NONE
        if overall not in HealthGrade.all():
            raise ValueError(f"unknown overall grade: {overall!r}")
        object.__setattr__(self, "overall_grade", overall)

    def grade_of(self, dimension: str) -> str:
        """Grade of one dimension (``HealthGrade.NONE`` when absent/unknown)."""
        return self.dimension_grades.get(dimension, HealthGrade.NONE)

    @property
    def all_dimensions_graded(self) -> bool:
        return all(grade != HealthGrade.NONE for grade in self.dimension_grades.values())


@dataclass(frozen=True)
class DiagnosisView:
    """Consumer view of one diagnosis (plan §9 C7).

    ``tag`` is the canonical diagnosis kind (mirrors the FA diagnosis taxonomy
    and the FO ``policy/repair.py`` ``DiagnosisKind`` vocabulary — e.g.
    ``high_turnover``, ``low_signal``, ``poor_coverage``, ``high_variance``,
    ``overfitting``, ``timing_violation`` ...).  ``severity``/``confidence``/
    ``repairability`` are FA-assigned; FO never recomputes them.  ``details``
    carries the FA evidence refs / repair hints verbatim.
    """

    factor_definition_id: str
    health_ref: str = ""
    tag: str = ""
    severity: str = DiagnosisSeverity.UNKNOWN
    confidence: float = 0.0
    repairability: str = DiagnosisRepairability.UNKNOWN
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.factor_definition_id, str) or not self.factor_definition_id:
            raise ValueError("factor_definition_id must be a non-empty string")
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError("diagnosis tag must be a non-empty string")
        conf = float(self.confidence)
        if conf != conf or conf in (float("inf"), float("-inf")):
            raise ValueError("diagnosis confidence must be finite")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("diagnosis confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", conf)
        if self.severity not in DiagnosisSeverity.all():
            raise ValueError(f"unknown diagnosis severity: {self.severity!r}")
        if self.repairability not in DiagnosisRepairability.all():
            raise ValueError(f"unknown diagnosis repairability: {self.repairability!r}")
        object.__setattr__(self, "details", tuple(self.details))


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FactorIntelligenceProvider(Protocol):
    """Narrow consumer boundary FO binds to FA factor intelligence.

    Implementations must satisfy the "unknown is explicit" contract: an
    unknown ``factor_definition_id`` either returns a view with
    ``is_unknown=True`` or raises :class:`FactorIntelligenceUnknownFactorError`;
    it may **never** return ``None`` or a silent default.  FO only consumes
    these three calls — FA publishes, FO consumes references.
    """

    def get_taxonomy(self, factor_definition_id: str) -> FactorTaxonomyView:
        """Taxonomy view for a factor definition.

        Returns an explicit ``UNKNOWN`` view or raises
        :class:`FactorIntelligenceUnknownFactorError` for unknown ids.
        """
        ...

    def get_health_card(
        self, factor_definition_id: str, evaluation_ref: str
    ) -> FactorHealthView:
        """Health-card view for a factor definition + evaluation."""
        ...

    def get_diagnoses(
        self, factor_definition_id: str, health_ref: str
    ) -> Sequence[DiagnosisView]:
        """Diagnoses for a factor definition + health-card ref."""
        ...


@runtime_checkable
class SelectionDecisionRequestView(Protocol):
    """Structural FO view of FA's authoritative decision request."""

    request_id: str
    policy_id: str
    policy_content_hash: str
    comparison_context_hash: str
    purpose: str
    decision_level: str
    baseline_ref: Optional[str]
    candidates: Sequence[Any]
    required_final_fidelity: str
    hypothesis_family_ref: str

    @property
    def content_hash(self) -> str: ...

    @property
    def candidate_set_hash(self) -> str: ...


@runtime_checkable
class SelectionDecisionReceiptView(Protocol):
    """Structural FO view of the immutable FA decision receipt."""

    request_id: str
    request_hash: str
    policy_id: str
    policy_content_hash: str
    comparison_context_hash: str
    candidate_set_hash: str
    decision_id: str
    content_hash: str
    status: Any
    eligibility: Mapping[str, bool]
    gate_receipts: Sequence[Any]
    point_utility: Mapping[str, Optional[float]]
    conservative_utility: Mapping[str, Optional[float]]
    relationship: Mapping[str, Any]
    effect_refs: Mapping[str, Optional[str]]
    qualification_scope: Mapping[str, Optional[str]]
    reasons: Sequence[str]
    winner_id: Optional[str]
    retained_ids: Sequence[str]
    final_fidelity: str


@runtime_checkable
class DecisionProvider(Protocol):
    """Sole selection authority consumed by FO; implementations live in FA."""

    def decide(
        self, request: SelectionDecisionRequestView
    ) -> SelectionDecisionReceiptView: ...


def require_bound_decision_receipt(
    request: SelectionDecisionRequestView,
    receipt: SelectionDecisionReceiptView,
) -> SelectionDecisionReceiptView:
    """Fail closed unless an FA receipt is bound to the exact FO request."""
    bindings = {
        "request_id": request.request_id,
        "request_hash": request.content_hash,
        "policy_id": request.policy_id,
        "policy_content_hash": request.policy_content_hash,
        "comparison_context_hash": request.comparison_context_hash,
        "candidate_set_hash": request.candidate_set_hash,
        "final_fidelity": request.required_final_fidelity,
    }
    for name, expected in bindings.items():
        if getattr(receipt, name, None) != expected:
            raise ValueError(f"decision receipt {name} binding mismatch")
    if not isinstance(receipt.decision_id, str) or not receipt.decision_id:
        raise ValueError("decision receipt requires decision_id")
    if not isinstance(receipt.content_hash, str) or not receipt.content_hash:
        raise ValueError("decision receipt requires content_hash")
    status = getattr(receipt.status, "value", receipt.status)
    if status in {"WAIT", "INCOMPARABLE"}:
        if any(value is not None for value in receipt.point_utility.values()):
            raise ValueError("unready decision must not publish point utility")
        if any(value is not None for value in receipt.conservative_utility.values()):
            raise ValueError("unready decision must not publish conservative utility")
    return receipt

# ---------------------------------------------------------------------------
# Unknown-factor sentinel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorIntelligenceView:
    """Union-of-views sentinel: one unknown id maps to one explicit view.

    Used by providers that prefer returning an explicit unknown view over
    raising (each provider implementation chooses its own convention as long as
    it is never ``None``).
    """

    taxonomy: FactorTaxonomyView
    health: FactorHealthView
    diagnoses: tuple[DiagnosisView, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnoses", tuple(self.diagnoses))
        if not isinstance(self.taxonomy, FactorTaxonomyView):
            raise TypeError("taxonomy must be a FactorTaxonomyView")
        if not isinstance(self.health, FactorHealthView):
            raise TypeError("health must be a FactorHealthView")


class FactorIntelligenceUnknownFactorError(Exception):
    """Typed error for an unknown factor definition id.

    Raised by providers that fail loudly (rather than returning an explicit
    unknown view).  Consumers must treat this as distinct from
    missing/invalid evidence.
    """

    def __init__(self, factor_definition_id: str):
        self.factor_definition_id = factor_definition_id
        super().__init__(f"unknown factor definition id: {factor_definition_id!r}")


def unknown_factor_view(factor_definition_id: str) -> FactorIntelligenceView:
    """Explicit unknown-factor view set for providers that don't raise.

    Every view is flagged ``is_unknown=True`` and carries no fabricated
    grades/tags.  This is the one sanctioned place FO constructs an unknown
    view; normal providers must only ever call it for genuinely unknown ids.
    """
    return FactorIntelligenceView(
        taxonomy=FactorTaxonomyView(
            factor_definition_id=factor_definition_id,
            data_domains=(),
            display_family="UNKNOWN",
            mechanism_tags=(),
            is_unknown=True,
        ),
        health=FactorHealthView(
            factor_definition_id=factor_definition_id,
            evaluation_ref="",
            overall_grade=HealthGrade.NONE,
            is_unknown=True,
        ),
        diagnoses=(),
    )


__all__ = [
    "HealthGrade",
    "HealthDimension",
    "DiagnosisSeverity",
    "DiagnosisRepairability",
    "FactorTaxonomyView",
    "FactorHealthView",
    "DiagnosisView",
    "FactorIntelligenceProvider",
    "FactorIntelligenceView",
    "FactorIntelligenceUnknownFactorError",
    "unknown_factor_view",
    "SelectionDecisionRequestView",
    "SelectionDecisionReceiptView",
    "DecisionProvider",
    "require_bound_decision_receipt",
]
