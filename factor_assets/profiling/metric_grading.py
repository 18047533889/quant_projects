"""Metric-grade artifact — one graded metric per evaluation (R61-FI-025, plan §8).

This module is a **pure contract + deterministic grading helper**: given one
evaluation evidence value and its QE ``EvidenceStatus``, it produces a frozen
:class:`MetricGradeArtifact` stamped with the versioned
:class:`~factor_assets.profiling.policies.FactorHealthPolicy` it ran under.

The plan §8 "missing evidence ≠ 0" iron law is enforced *in the artifact's
``__post_init__``* — not as a caller convention:

- When ``evidence_status`` is any of ``FAILED / INVALID / STALE / UNKNOWN /
  NOT_COMPUTED_STAGE / UNAVAILABLE_INPUT / NOT_APPLICABLE`` the artifact makes
  ``grade=None`` and ``desirability=None`` and refuses to accept a numeric
  ``desirability`` on a non-computed status (a caller passing a bogus numeric
  grade/desirability together with a non-computed status gets a
  :class:`ValueError`, fail-closed).
- ``COMPUTED`` evidence may carry a grade + desirability; the grade it carries
  must be a member of the grade alphabet bound to the grading policy, and the
  desirability must lie in ``[0, 1]`` (finite).  Absolute grade and cohort
  percentile are both stored and are **never** derived from one another
  (plan §10.3).

Status vocabulary (reference-only strings — this package may not import
``quant_evaluator``; FA already mirrors the same 8-token QE vocabulary in
``library/promotion_gate.py``).  The plan §8 uppercase tokens are the *canonical
grade-engine* tokens; consumers from QE pass the lowercase QE status values
which map onto them 1:1 (``not_computed``/``label_not_mature`` ->
``NOT_COMPUTED_STAGE``, ``invalid_evidence`` -> ``INVALID``,
``failed`` -> ``FAILED``, ``unavailable`` -> ``UNAVAILABLE_INPUT``,
``unsupported`` -> ``NOT_APPLICABLE``).

Grading itself (value -> grade / desirability mapping) is policy data living in
:mod:`~factor_assets.profiling.policies`; this module only stamps/validates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from factor_assets.profiling.policies import (
    HealthGradeVocabulary,
    get_health_policy,
)

__all__ = [
    "EvidenceStatusVocabulary",
    "BAD_EVIDENCE_STATUSES",
    "NON_GRADE_STATUSES",
    "MetricGradeArtifact",
    "grade_metric_evidence",
    "resolve_evidence_status",
]


# ---------------------------------------------------------------------------
# Evidence-status vocabulary (plan §8; reference mirror of QE EvidenceStatus)
# ---------------------------------------------------------------------------


class EvidenceStatusVocabulary:
    """Canonical grade-engine evidence-status tokens (plan §8, matrix C5).

    The seven non-computed tokens are the *exact* set that forbid a grade; the
    eighth token ``COMPUTED`` is the only one that may carry a numeric grade /
    desirability.  ``NONE`` marks an explicitly missing status token (the
    artifact-level fail-closed option when the caller refuses to declare one).
    """

    COMPUTED = "COMPUTED"
    NOT_COMPUTED_STAGE = "NOT_COMPUTED_STAGE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE_INPUT = "UNAVAILABLE_INPUT"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"
    INVALID = "INVALID"
    NONE = "NONE"

    #: QE lowercase status value -> canonical grade-engine token.
    #: Reference mapping only (do not import quant_evaluator).
    QE_ALIASES: Mapping[str, str] = {
        "computed": "COMPUTED",
        "not_computed": "NOT_COMPUTED_STAGE",
        "label_not_mature": "NOT_COMPUTED_STAGE",
        "insufficient_data": "NOT_COMPUTED_STAGE",
        "unavailable": "UNAVAILABLE_INPUT",
        "unsupported": "NOT_APPLICABLE",
        "invalid_evidence": "INVALID",
        "failed": "FAILED",
    }

    @classmethod
    def non_computed(cls) -> tuple[str, ...]:
        return (
            cls.NOT_COMPUTED_STAGE,
            cls.NOT_APPLICABLE,
            cls.UNAVAILABLE_INPUT,
            cls.FAILED,
            cls.UNKNOWN,
            cls.STALE,
            cls.INVALID,
        )

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (cls.COMPUTED,) + cls.non_computed()


EVIDENCE_LEVELS = ("E0", "E1", "E2", "E3")
APPLICABILITY_STATES = ("APPLICABLE", "NOT_APPLICABLE", "NOT_RUN_BUDGET")


#: The iron-law set: any of these statuses -> no grade, no desirability.
BAD_EVIDENCE_STATUSES: tuple[str, ...] = EvidenceStatusVocabulary.non_computed()
#: Alias for readability in call sites.
NON_GRADE_STATUSES: tuple[str, ...] = BAD_EVIDENCE_STATUSES


def resolve_evidence_status(value: object) -> str:
    """Resolve a str / QE-lowercase / None status to a canonical token.

    Unknown strings fail closed (``ValueError``) — an unrecognised status is
    never silently treated as computed.  ``None`` resolves to ``NONE`` so the
    caller can set an explicit non-computed grade by declaring
    ``None`` status; callers that would rather fail than default pass the
    ``TRULY_NONE_AS_ERROR`` path themselves.
    """
    if value is None:
        return EvidenceStatusVocabulary.NONE
    if not isinstance(value, str):
        raise TypeError(
            f"evidence_status must be a str, got {type(value).__name__}"
        )
    if value in EvidenceStatusVocabulary.all():
        return value
    mapped = EvidenceStatusVocabulary.QE_ALIASES.get(value)
    if mapped is not None:
        return mapped
    raise ValueError(
        f"unknown evidence_status {value!r}; expected one of "
        f"{[s for s in EvidenceStatusVocabulary.all()]} (or a QE lowercase "
        f"status alias: {sorted(EvidenceStatusVocabulary.QE_ALIASES)})"
    )


# ---------------------------------------------------------------------------
# MetricGradeArtifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricGradeArtifact:
    """Frozen grade artifact for one metric evidence value (plan §8).

    Fields exactly follow the plan §8 target shape.  ``grade`` and
    ``desirability`` are ``None`` whenever ``evidence_status`` is any
    non-computed token — missing evidence is never numeric zero and never
    grade-carrying.  ``cohort_percentile`` is stored independently from (and
    never recomputed into) ``grade`` — absolute grade is policy-stable, cohort
    percentile is relative to the library population.

    ``grading_policy_id`` / ``grading_policy_version`` stamp the policy that
    produced the grade; ``evaluation_ref`` pins the QE evaluation this grade
    came from.
    """

    #: Canonical metric id in the QE ``MetricRegistry`` authority
    #: (e.g. ``rank_ic``, ``tie_ratio``, ``max_drawdown``).
    metric_id: str
    #: Metric version / registry revision the value was computed under (or the
    #: empty string when unknown).
    metric_version: str = ""
    #: The raw evidence value (finite, or ``None`` when not computed).
    value: float | None = None
    #: Canonical evidence-status token (plan §8 vocabulary).
    evidence_status: str = EvidenceStatusVocabulary.NOT_COMPUTED_STAGE
    #: Grade letter (alphabet of the stamped grading policy), or ``None``.
    grade: str | None = None
    #: Desirability in ``[0, 1]`` (higher is better), or ``None``.
    desirability: float | None = None
    #: Cohort percentile in ``[0, 100]`` (higher is better), or ``None``.
    cohort_percentile: float | None = None
    #: Two-sided confidence interval ``(lo, hi)``, or ``None``.
    confidence_interval: tuple[float, float] | None = None
    #: Statistical confidence (e.g. a t-stat-derived strength) in ``[0, 1]``.
    statistical_confidence: float | None = None
    #: Raw relative delta vs the RAW baseline (value - value_raw)/abs(value_raw)
    #: — stored, never used as the grade authority.
    raw_relative_delta: float | None = None
    #: Grading policy the grade/desirability were produced under.
    grading_policy_id: str = ""
    #: Grading policy version the grade/desirability were produced under.
    grading_policy_version: str = ""
    #: Reference to the QE evaluation that produced the evidence.
    evaluation_ref: str = ""
    factor_definition_id: str = ""
    factor_value_ref: str = ""
    factor_axis_ref: str = ""
    config_hash: str = ""
    created_from_refs: tuple[str, ...] = ()
    use_case: str = "GENERIC"
    applicability: str = "APPLICABLE"
    applicability_reason: str = ""
    evidence_level: str = "E0"
    calibration_ref: str = ""
    metric_instance: str = ""
    horizon: int | None = None
    universe_ref: str = ""
    data_as_of: str = ""
    recipe_ref: str = ""
    representation_ref: str = ""
    portfolio_spec_ref: str = ""
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metric_id, str) or not self.metric_id:
            raise ValueError("metric_id must be a non-empty string")
        if not isinstance(self.metric_version, str):
            raise ValueError("metric_version must be a string")
        if self.metric_version and not self.metric_version[0].isdigit():
            raise ValueError(
                "metric_version must be a version (or empty), got "
                f"{self.metric_version!r}"
            )
        object.__setattr__(self, "created_from_refs", tuple(self.created_from_refs))
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(self.reason_codes)))
        if self.applicability not in APPLICABILITY_STATES:
            raise ValueError("unknown applicability state")
        if self.applicability != "APPLICABLE" and not self.applicability_reason:
            raise ValueError("non-applicable/not-run metrics require an applicability reason")
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError("evidence_level must be E0..E3")
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("horizon must be positive")
        for name in ("factor_definition_id", "factor_value_ref", "factor_axis_ref", "config_hash", "evaluation_ref"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")

        status = resolve_evidence_status(self.evidence_status)
        object.__setattr__(self, "evidence_status", status)
        if self.evidence_level != "E0" and status != EvidenceStatusVocabulary.COMPUTED:
            raise ValueError("non-computed evidence cannot claim E1-E3")

        # -- value / status consistency ----------------------------------
        value = self._coerce_optional_fraction(self.value, "value")
        if value is not None and status != EvidenceStatusVocabulary.COMPUTED:
            # Degraded partial evals may still carry a value (QE allows a
            # non-COMPUTED status with a non-None artifact); the grade engine
            # simply refuses to *grade* it.  Value itself is allowed.
            pass
        object.__setattr__(self, "value", value)

        # -- statistical confidence / interval ---------------------------
        stat_conf = self._coerce_optional_fraction(
            self.statistical_confidence, "statistical_confidence"
        )
        if stat_conf is not None and not 0.0 <= stat_conf <= 1.0:
            raise ValueError("statistical_confidence must be in [0, 1]")
        object.__setattr__(self, "statistical_confidence", stat_conf)
        ci = self.confidence_interval
        if ci is not None:
            if not isinstance(ci, Sequence) or len(ci) != 2:
                raise ValueError(
                    "confidence_interval must be a (lo, hi) pair or None"
                )
            lo = self._coerce_finite(ci[0], "confidence_interval[0]")
            hi = self._coerce_finite(ci[1], "confidence_interval[1]")
            if lo > hi:
                raise ValueError(
                    f"confidence_interval lo {lo!r} must be <= hi {hi!r}"
                )
            object.__setattr__(self, "confidence_interval", (lo, hi))

        raw_delta = self.raw_relative_delta
        if raw_delta is not None:
            raw_delta = self._coerce_finite(raw_delta, "raw_relative_delta")
        object.__setattr__(self, "raw_relative_delta", raw_delta)

        # -- Iron law (plan §8): no grade / desirability on bad evidence ----
        if status in BAD_EVIDENCE_STATUSES:
            if self.grade is not None or self.desirability is not None:
                raise ValueError(
                    "non-computed evidence status must not carry grade/desirability "
                    "(missing evidence is never zero, never a grade); "
                    f"status={status} grade={self.grade!r} "
                    f"desirability={self.desirability!r}"
                )
            object.__setattr__(self, "grade", None)
            object.__setattr__(self, "desirability", None)

        if self.grade is not None:
            if status != EvidenceStatusVocabulary.COMPUTED:
                raise ValueError(
                    "grade requires COMPUTED evidence status "
                    f"(got {status!r})"
                )
            if not isinstance(self.grade, str) or not self.grade:
                raise ValueError("grade must be a non-empty string or None")
        if self.desirability is not None:
            if status != EvidenceStatusVocabulary.COMPUTED:
                raise ValueError(
                    "desirability requires COMPUTED evidence status "
                    f"(got {status!r})"
                )
            d = self._coerce_finite(self.desirability, "desirability")
            if not 0.0 <= d <= 1.0:
                raise ValueError("desirability must be in [0, 1]")
            object.__setattr__(self, "desirability", d)

        cohort = self.cohort_percentile
        if cohort is not None:
            cohort = self._coerce_finite(cohort, "cohort_percentile")
            if not 0.0 <= cohort <= 100.0:
                raise ValueError("cohort_percentile must be in [0, 100]")
        object.__setattr__(self, "cohort_percentile", cohort)

        if self.grading_policy_id and not isinstance(self.grading_policy_id, str):
            raise ValueError("grading_policy_id must be a string")
        if self.grading_policy_version and not isinstance(
            self.grading_policy_version, str
        ):
            raise ValueError("grading_policy_version must be a string")
        if not isinstance(self.evaluation_ref, str):
            raise ValueError("evaluation_ref must be a string")

    # -- validation helpers ------------------------------------------------

    @staticmethod
    def _coerce_finite(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be a non-boolean number")
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise ValueError(f"{label} must be finite")
        return number

    @classmethod
    def _coerce_optional_fraction(cls, value: object, label: str) -> float | None:
        if value is None:
            return None
        return cls._coerce_finite(value, label)

    @staticmethod
    def _coerce_fraction(value: object, label: str) -> float:
        try:
            number = MetricGradeArtifact._coerce_finite(value, label)
        except (TypeError, ValueError):
            raise
        return number

    # -- conveniences -------------------------------------------------------

    @property
    def production_provenance_complete(self) -> bool:
        return all(
            (
                self.factor_definition_id,
                self.factor_value_ref,
                self.factor_axis_ref,
                self.config_hash,
                self.evaluation_ref,
                self.metric_version,
            )
        ) and bool(self.created_from_refs)

    @property
    def has_grade(self) -> bool:
        """True iff a grade is bound (implies COMPUTED evidence)."""
        return self.grade is not None

    @property
    def has_desirability(self) -> bool:
        """True iff a desirability is bound (implies COMPUTED evidence)."""
        return self.desirability is not None

    @property
    def effect_grade(self) -> str | None:
        """V5 name for the point-effect grade; evidence_level is independent."""
        return self.grade

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "value": self.value,
            "evidence_status": self.evidence_status,
            "grade": self.grade,
            "desirability": self.desirability,
            "cohort_percentile": self.cohort_percentile,
            "confidence_interval": (
                list(self.confidence_interval) if self.confidence_interval else None
            ),
            "statistical_confidence": self.statistical_confidence,
            "raw_relative_delta": self.raw_relative_delta,
            "grading_policy_id": self.grading_policy_id,
            "grading_policy_version": self.grading_policy_version,
            "evaluation_ref": self.evaluation_ref,
            "factor_definition_id": self.factor_definition_id,
            "factor_value_ref": self.factor_value_ref,
            "factor_axis_ref": self.factor_axis_ref,
            "config_hash": self.config_hash,
            "created_from_refs": list(self.created_from_refs),
            "use_case": self.use_case,
            "applicability": self.applicability,
            "applicability_reason": self.applicability_reason,
            "effect_grade": self.effect_grade,
            "evidence_level": self.evidence_level,
            "calibration_ref": self.calibration_ref,
            "metric_instance": self.metric_instance,
            "horizon": self.horizon,
            "universe_ref": self.universe_ref,
            "data_as_of": self.data_as_of,
            "recipe_ref": self.recipe_ref,
            "representation_ref": self.representation_ref,
            "portfolio_spec_ref": self.portfolio_spec_ref,
            "reason_codes": list(self.reason_codes),
        }


# ---------------------------------------------------------------------------
# Deterministic grading entry point
# ---------------------------------------------------------------------------


def grade_metric_evidence(
    *,
    metric_id: str,
    value: float | None,
    evidence_status: str,
    policy=None,
    evaluation_ref: str = "",
    metric_version: str = "",
    cohort_percentile: float | None = None,
    confidence_interval: tuple[float, float] | None = None,
    statistical_confidence: float | None = None,
    raw_relative_delta: float | None = None,
    factor_definition_id: str = "",
    factor_value_ref: str = "",
    factor_axis_ref: str = "",
    config_hash: str = "",
    created_from_refs: Sequence[str] = (),
    use_case: str = "GENERIC",
    applicability: str = "APPLICABLE",
    applicability_reason: str = "",
    evidence_level: str | None = None,
    calibration_ref: str | None = None,
    metric_instance: str = "",
    horizon: int | None = None,
    universe_ref: str = "",
    data_as_of: str = "",
    recipe_ref: str = "",
    representation_ref: str = "",
    portfolio_spec_ref: str = "",
    reason_codes: Sequence[str] = (),
) -> MetricGradeArtifact:
    """Grade one metric evidence value under a versioned FactorHealthPolicy.

    The value -> grade / desirability mapping is the policy's
    ``metric_grade_rules[metric_id]`` (an absolute, policy-stable anchor band
    table with symmetric grade boundaries and a desirability band map); this
    function only *applies* policy data — no magic numbers live here.

    Non-computed evidence statuses short-circuit: the artifact is stamped with
    ``grade=None`` / ``desirability=None`` regardless of the numeric value
    passed (missing evidence is never numeric zero).
    """
    if policy is None:
        policy = get_health_policy()
    status = resolve_evidence_status(evidence_status)
    canonical_metric_id = policy.canonical_metric_id(metric_id)

    # A point estimate can establish only descriptive E1. E2/E3 must be
    # explicitly supplied by a validation workflow carrying its own evidence.
    resolved_level = evidence_level or (
        "E1" if status == EvidenceStatusVocabulary.COMPUTED else "E0"
    )

    grade: str | None = None
    desirability: float | None = None
    if status == EvidenceStatusVocabulary.COMPUTED:
        grade, desirability = policy.grade_and_desirability(canonical_metric_id, value)

    return MetricGradeArtifact(
        metric_id=canonical_metric_id,
        metric_version=metric_version,
        value=value,
        evidence_status=status,
        grade=grade,
        desirability=desirability,
        cohort_percentile=cohort_percentile,
        confidence_interval=confidence_interval,
        statistical_confidence=statistical_confidence,
        raw_relative_delta=raw_relative_delta,
        grading_policy_id=policy.policy_id,
        grading_policy_version=policy.policy_version,
        evaluation_ref=evaluation_ref,
        factor_definition_id=factor_definition_id,
        factor_value_ref=factor_value_ref,
        factor_axis_ref=factor_axis_ref,
        config_hash=config_hash,
        created_from_refs=tuple(created_from_refs),
        use_case=use_case,
        applicability=applicability,
        applicability_reason=applicability_reason,
        evidence_level=resolved_level,
        calibration_ref=policy.calibration_ref if calibration_ref is None else calibration_ref,
        metric_instance=metric_instance,
        horizon=horizon,
        universe_ref=universe_ref,
        data_as_of=data_as_of,
        recipe_ref=recipe_ref,
        representation_ref=representation_ref,
        portfolio_spec_ref=portfolio_spec_ref,
        reason_codes=tuple(reason_codes),
    )


def _grade_alphabet_from_policy(policy) -> tuple[str, ...]:
    """Extract the grade alphabet bound to a stamped policy (helper for tests)."""
    return HealthGradeVocabulary.of_policy(policy)
