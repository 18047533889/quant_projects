"""Versioned policy registries for factor profiling (R61-FI-013/025/026).

This module owns every *policy constant* the deterministic FA profiling
engines run under:

- :class:`TaxonomyPolicy` (R61-FI-013) — mechanism/structure tag vocab + tag
  confidence weights + display-family tokens.
- :class:`FactorHealthPolicy` (R61-FI-025) — grade alphabet, display score
  bands, RankIC/ICIR/retention anchor tables, per-metric grade rules,
  per-dimension aggregation rules (plan §9.1 weights), admission floors and
  typed integrity-gate ids.  No magic number lives in grading / dimension /
  health-card rule code.
- :class:`DiagnosisPolicy` (R61-FI-026) — deterministic diagnosis-rule
  severity / confidence / repairability and the numeric thresholds that gate
  each tag.

All registries are **read-only frozen catalogs**: policies are declared at
import time and never mutated.  ``get_*_policy`` resolves a policy by
``policy_id`` (latest version when version omitted); unknown ids fail closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence
import hashlib
import json
from types import MappingProxyType

__all__ = [
    "TagSource",
    "TagEvidence",
    "TaxonomyPolicy",
    "TAXONOMY_POLICY_CURRENT_ID",
    "TAXONOMY_POLICY_CURRENT_VERSION",
    "TAXONOMY_POLICIES",
    "get_taxonomy_policy",
    "HealthGradeVocabulary",
    "GradeAnchor",
    "MetricGradeRule",
    "DimensionRule",
    "AdmissionFloors",
    "FactorHealthPolicy",
    "CalibrationArtifact",
    "FACTOR_HEALTH_POLICY_CURRENT_ID",
    "FACTOR_HEALTH_POLICY_CURRENT_VERSION",
    "FACTOR_HEALTH_POLICIES",
    "get_health_policy",
    "DiagnosisPolicy",
    "DIAGNOSIS_POLICY_CURRENT_ID",
    "DIAGNOSIS_POLICY_CURRENT_VERSION",
    "DIAGNOSIS_POLICIES",
    "get_diagnosis_policy",
]

# ===========================================================================
# Part 1 — Taxonomy (R61-FI-013)
# ===========================================================================


class TagSource:
    """Enum-like constants for TagEvidence.source.

    The classifier only ever emits ``DETERMINISTIC_RULE``; the other values are
    reserved for the future agent-suggestion / human-audit promotion chain
    (plan §7 C3) and must not be produced by the deterministic rule engine.
    """

    DETERMINISTIC_RULE = "deterministic_rule"
    AGENT_SUGGESTION = "agent_suggestion"  # reserved — provisional, never emitted here
    HUMAN_AUDIT = "human_audit"            # reserved — audited override, never emitted here


@dataclass(frozen=True)
class TagEvidence:
    """Evidence for a single mechanism tag.

    ``tag`` is the mechanism token this evidence backs (e.g. ``MOMENTUM``).
    ``source`` identifies the provenance channel (always ``deterministic_rule``
    for R61-FI-013).  ``confidence`` is a policy-versioned weight in
    ``[0, 1]``.  ``evidence_refs`` are the exact inputs the rule consumed.
    """

    tag: str
    source: str
    confidence: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError("tag is required")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source is required")
        conf = float(self.confidence)
        if conf != conf or conf in (float("inf"), float("-inf")):
            raise ValueError("confidence must be finite")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", conf)
        refs = tuple(str(r) for r in self.evidence_refs)
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class TaxonomyPolicy:
    """An immutable, versioned taxonomy policy."""

    policy_id: str
    policy_version: str
    description: str
    mechanism_tag_vocab: tuple[str, ...] = (
        "MOMENTUM", "REVERSAL", "BREAKOUT", "TREND", "MEAN_REVERSION",
        "VOLATILITY", "LIQUIDITY", "CROWDING", "VALUE", "QUALITY",
        "PROFITABILITY", "GROWTH", "INVESTMENT", "LEVERAGE", "CASHFLOW",
        "EVENT_DECAY", "TAIL", "INTERACTION", "SEASONALITY",
        "MICROSTRUCTURE", "UNKNOWN_MECHANISM",
    )
    structure_tag_vocab: tuple[str, ...] = (
        "TIME_SERIES", "CROSS_SECTIONAL", "GROUPED", "RANKED", "ZSCORED",
        "WINSORIZED", "SMOOTHED", "NEUTRALIZED", "NEUTRALIZED_SIZE",
        "NEUTRALIZED_INDUSTRY", "SPARSE", "EVENT_DRIVEN", "BINARY",
        "DISCRETE", "MIXED_DOMAIN", "HIGH_TURNOVER", "LOW_FREQUENCY",
    )
    domain_vocab: tuple[str, ...] = (
        "PRICE", "VOLUME", "LIQUIDITY", "VALUATION", "FUNDAMENTAL",
        "FUNDAMENTAL.VALUE", "FUNDAMENTAL.QUALITY", "FUNDAMENTAL.GROWTH",
        "FUNDAMENTAL.INVESTMENT", "FUNDAMENTAL.CASHFLOW",
        "FUNDAMENTAL.LEVERAGE", "SIZE", "EVENT", "FLOW_SENTIMENT",
        "MICROSTRUCTURE", "RISK", "CALENDAR", "ALTERNATIVE", "UNKNOWN",
    )
    mechanism_confidence: float = 0.9
    structure_confidence: float = 1.0
    display_family_tokens: tuple[str, ...] = (
        "PRICE", "VOLUME", "LIQUIDITY", "VALUATION", "FUNDAMENTAL", "SIZE",
    )
    display_family_label_groups: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "PRICE": ("PRICE",),
            "VOLUME": ("VOLUME",),
            "LIQUIDITY": ("LIQUIDITY",),
            "VALUATION": ("VALUATION",),
            "FUNDAMENTAL": (
                "FUNDAMENTAL",
                "FUNDAMENTAL.VALUE",
                "FUNDAMENTAL.QUALITY",
                "FUNDAMENTAL.GROWTH",
                "FUNDAMENTAL.INVESTMENT",
                "FUNDAMENTAL.CASHFLOW",
                "FUNDAMENTAL.LEVERAGE",
            ),
            "SIZE": ("SIZE",),
        }
    )

    # Append-only to retain the public dataclass's historical positional arguments.
    mechanism_sources: str = "all_fields_legacy"

    def __post_init__(self) -> None:
        if self.mechanism_sources not in {"all_fields_legacy", "alpha_only"}:
            raise ValueError("unknown mechanism_sources policy")
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if not self.policy_version:
            raise ValueError("policy_version is required")
        for conf in (self.mechanism_confidence, self.structure_confidence):
            c = float(conf)
            if not 0.0 <= c <= 1.0:
                raise ValueError("confidence weights must be in [0, 1]")
        object.__setattr__(
            self, "mechanism_tag_vocab", tuple(dict.fromkeys(self.mechanism_tag_vocab))
        )
        object.__setattr__(
            self, "structure_tag_vocab", tuple(dict.fromkeys(self.structure_tag_vocab))
        )
        object.__setattr__(self, "domain_vocab", tuple(dict.fromkeys(self.domain_vocab)))
        if not set(self.display_family_label_groups).issubset(
            set(self.display_family_tokens)
        ):
            raise ValueError(
                "display_family_label_groups keys must be in display_family_tokens"
            )
        for label, group in self.display_family_label_groups.items():
            if not group:
                raise ValueError(f"display family label group {label} is empty")

    @property
    def display_family_labels(self) -> tuple[str, ...]:
        return tuple(
            t for t in self.display_family_tokens if t in self.display_family_label_groups
        )


TAXONOMY_POLICY_CURRENT_ID = "CN_A_SHARE_DAILY_TAXONOMY_V1"
TAXONOMY_POLICY_CURRENT_VERSION = "2.0.0"

TAXONOMY_POLICIES: Mapping[str, tuple[TaxonomyPolicy, ...]] = {
    TAXONOMY_POLICY_CURRENT_ID: (
        TaxonomyPolicy(
            policy_id=TAXONOMY_POLICY_CURRENT_ID,
            policy_version="1.0.0",
            description="R61-FI-013 initial deterministic taxonomy policy for "
            "A-share daily factors.",
        ),
        TaxonomyPolicy(
            policy_id=TAXONOMY_POLICY_CURRENT_ID,
            policy_version="2.0.0",
            description="Explicit usage-role taxonomy: economic mechanisms derive from alpha source domains only.",
            mechanism_sources="alpha_only",
        ),
    ),
}


def get_taxonomy_policy(
    policy_id: str = TAXONOMY_POLICY_CURRENT_ID,
    policy_version: str | None = None,
) -> TaxonomyPolicy:
    versions = TAXONOMY_POLICIES.get(policy_id)
    if not versions:
        raise KeyError(f"unknown taxonomy policy_id: {policy_id}")
    if policy_version is None:
        return versions[-1]
    for policy in versions:
        if policy.policy_version == policy_version:
            return policy
    raise KeyError(
        f"unknown taxonomy policy version {policy_version!r} "
        f"for policy_id {policy_id!r}"
    )


# ===========================================================================
# Part 2 — Health grading (R61-FI-025, plan §8-§10)
# ===========================================================================

#: Canonical 14 health-dimension ids — **aligned with the FO consumer view**
#: (``factor_optimizer.ports.factor_intelligence.HealthDimension``).  FA is the
#: authority; FO only projects these ids verbatim.
HEALTH_DIMENSIONS: tuple[str, ...] = (
    "predictive_power", "stability", "robustness", "turnover", "capacity",
    "cost_drag", "drawdown", "tail_risk", "data_coverage", "freshness",
    "complexity", "economic_sense", "shape_quality", "regime_sensitivity",
)

#: Typed card-level integrity gates (plan §9 / §13.1).  These are *not* the
#: 14 scoring dimensions — they are cross-package typed gates (PIT, label
#: maturity, snapshot/universe identity, finite shape/schema, return basis).
INTEGRITY_GATE_IDS: tuple[str, ...] = (
    "pit_valid",
    "label_maturity_ok",
    "snapshot_identity_ok",
    "universe_identity_ok",
    "finite_shape_schema_ok",
    "unit_scale_ok",
    "return_basis_ok",
)


class HealthGradeVocabulary:
    """Grade letters S+ (best) .. D (worst); ``NONE`` = explicitly absent.

    Mirrors the FO consumer ``HealthGrade`` so FA grades project verbatim.
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

    RANKED: tuple[str, ...] = (S_PLUS, S, A_PLUS, A, B_PLUS, B, C, D)

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return cls.RANKED + (cls.NONE,)

    @classmethod
    def of_policy(cls, policy: "FactorHealthPolicy") -> tuple[str, ...]:
        return cls.RANKED + (cls.NONE,)

    @classmethod
    def rank(cls, grade: str) -> int:
        """0 == S+ (best) .. 7 == D; raises KeyError for NONE/unknown."""
        try:
            return cls.RANKED.index(grade)
        except ValueError:
            raise KeyError(f"unknown ranked grade: {grade!r}") from None


@dataclass(frozen=True)
class GradeAnchor:
    """One absolute anchor band of a metric grading table.

    Band semantics depend on the owning :class:`MetricGradeRule` direction:

    - ``higher_is_better``: the rule's grades are thresholds ``value >= ge``.
      Anchors are ordered best->worst with strictly *decreasing* ``ge`` and
      only the final anchor may carry ``ge=None`` (catches everything below
      the previous threshold).
    - ``lower_is_better``: grades are ceilings ``value <= ge``.  Anchors are
      ordered best->worst with strictly *increasing* ``ge`` and only the final
      anchor may carry ``ge=None`` (catches everything above the previous
      ceiling).

    ``desirability`` is the fixed desirability in ``[0, 1]`` awarded when a
    value lands in this band.  Absolute anchors are policy-stable and never
    derived from cohort percentile (plan §10.3).
    """

    grade: str
    ge: float | None
    desirability: float

    def __post_init__(self) -> None:
        if self.grade not in HealthGradeVocabulary.RANKED:
            raise ValueError(f"grade must be a ranked grade, got {self.grade!r}")
        d = float(self.desirability)
        if not 0.0 <= d <= 1.0:
            raise ValueError("desirability must be in [0, 1]")
        if self.ge is not None:
            ge = float(self.ge)
            if math.isnan(ge) or math.isinf(ge):
                raise ValueError("ge must be finite or None")
        object.__setattr__(self, "desirability", d)


@dataclass(frozen=True)
class MetricGradeRule:
    """Absolute grade/desirability rule for one metric under a policy.

    ``anchors`` ordered best->worst.  ``higher_is_better`` selects whether the
    band match is ``value >= ge`` (higher better) or ``value <= ge`` (lower
    better).  ``missing_desirability`` is the desirability used when the metric
    evidence is missing — **0.0 today** (missing evidence never earns partial
    credit; the field exists so the missing policy is versioned data).
    """

    metric_id: str
    anchors: tuple[GradeAnchor, ...]
    higher_is_better: bool = True
    missing_desirability: float = 0.0
    evaluation_role: str = "PERFORMANCE_HIGHER"
    applicability: tuple[str, ...] = ("*",)
    unit: str = ""
    runtime_metric_id: str = ""

    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("metric_id is required")
        if not self.anchors:
            raise ValueError("anchors cannot be empty")
        object.__setattr__(self, "anchors", tuple(self.anchors))
        grades = [a.grade for a in self.anchors]
        if len(set(grades)) != len(grades):
            raise ValueError("anchor grades must be unique")
        if tuple(grades) != tuple(HealthGradeVocabulary.RANKED[: len(grades)]):
            raise ValueError(
                "anchor grades must be the leading ranked-grade sequence "
                f"S+..{grades[-1]}, got {grades}"
            )
        ges: list[float | None] = [a.ge for a in self.anchors]
        non_none = [g for g in ges[:-1]]
        if any(g is None for g in non_none):
            raise ValueError("only the final anchor may have ge=None")
        if not self.higher_is_better:
            # best->worst ceilings must strictly increase; final is None
            for i in range(len(ges) - 2):
                if not (ges[i] < ges[i + 1]):
                    raise ValueError(
                        "lower-is-better anchor ceilings must strictly increase "
                        f"best->worst, got {ges[:-1]}"
                    )
        else:
            for i in range(len(ges) - 2):
                if not (ges[i] > ges[i + 1]):
                    raise ValueError(
                        "higher-is-better anchor thresholds must strictly "
                        f"decrease best->worst, got {ges[:-1]}"
                    )
        md = float(self.missing_desirability)
        if not 0.0 <= md <= 1.0:
            raise ValueError("missing_desirability must be in [0, 1]")
        object.__setattr__(self, "missing_desirability", md)
        if self.evaluation_role not in {
            "PERFORMANCE_HIGHER", "RISK_LOWER", "TARGET_RANGE",
            "DIAGNOSTIC_ONLY", "INTEGRITY_BOOLEAN", "SAMPLE_EVIDENCE",
        }:
            raise ValueError("unknown evaluation_role")
        object.__setattr__(self, "applicability", tuple(dict.fromkeys(self.applicability)))
        if not self.applicability:
            raise ValueError("applicability cannot be empty")
        if not isinstance(self.runtime_metric_id, str):
            raise TypeError("runtime_metric_id must be a string")

    def grade_for(self, value: float | None) -> str | None:
        if value is None:
            return None
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        if self.higher_is_better:
            for anchor in self.anchors:
                if anchor.ge is None or v >= anchor.ge:
                    return anchor.grade
            return self.anchors[-1].grade  # pragma: no cover - guard
        for anchor in self.anchors:
            if anchor.ge is None:
                continue
            if v <= anchor.ge:
                return anchor.grade
        return self.anchors[-1].grade

    def desirability_for(self, value: float | None) -> float | None:
        grade = self.grade_for(value)
        if grade is None:
            return None
        for anchor in self.anchors:
            if anchor.grade == grade:
                return anchor.desirability
        return None  # pragma: no cover - guard


@dataclass(frozen=True)
class DimensionRule:
    """Aggregation rule for one health dimension (plan §9.1).

    ``metric_ids`` lists the metrics aggregated into this dimension (canonical
    QE metric ids).  ``min_weight``/``geo_weight`` version the plan §9.1
    aggregation formula::

        dimension_score = min_weight * min(metric desirabilities)
                        + geo_weight * geometric_mean(metric desirabilities)

    Weights sum to 1.  ``missing_desirability`` is substituted for a metric
    whose evidence is missing/not computed (0.0 today — missing is never a
    partial credit).  ``repairable`` records whether a failing score on this
    dimension is repairable in principle (policy data).
    """

    dimension_id: str
    metric_ids: tuple[str, ...]
    min_weight: float = 0.4
    geo_weight: float = 0.6
    missing_desirability: float = 0.0
    repairable: bool = True
    required_metric_ids: tuple[str, ...] = ()
    optional_metric_ids: tuple[str, ...] = ()
    alternative_metric_groups: tuple[tuple[str, ...], ...] = ()
    applicable_use_cases: tuple[str, ...] = ("*",)

    def __post_init__(self) -> None:
        if not self.dimension_id:
            raise ValueError("dimension_id is required")
        if not self.metric_ids:
            raise ValueError("metric_ids cannot be empty")
        object.__setattr__(self, "metric_ids", tuple(dict.fromkeys(self.metric_ids)))
        required = tuple(dict.fromkeys(self.required_metric_ids or self.metric_ids))
        optional = tuple(dict.fromkeys(self.optional_metric_ids))
        if not set(required).issubset(self.metric_ids) or not set(optional).issubset(self.metric_ids):
            raise ValueError("required/optional metric ids must belong to metric_ids")
        if set(required) & set(optional):
            raise ValueError("a metric cannot be both required and optional")
        object.__setattr__(self, "required_metric_ids", required)
        object.__setattr__(self, "optional_metric_ids", optional)
        groups = tuple(tuple(dict.fromkeys(g)) for g in self.alternative_metric_groups)
        if any(not g or not set(g).issubset(self.metric_ids) for g in groups):
            raise ValueError("alternative metric groups must be non-empty subsets of metric_ids")
        object.__setattr__(self, "alternative_metric_groups", groups)
        object.__setattr__(self, "applicable_use_cases", tuple(dict.fromkeys(self.applicable_use_cases)))
        mw = float(self.min_weight)
        gw = float(self.geo_weight)
        if not 0.0 <= mw <= 1.0 or not 0.0 <= gw <= 1.0:
            raise ValueError("min_weight and geo_weight must be in [0, 1]")
        if abs((mw + gw) - 1.0) > 1e-9:
            raise ValueError("min_weight + geo_weight must sum to 1.0")
        md = float(self.missing_desirability)
        if not 0.0 <= md <= 1.0:
            raise ValueError("missing_desirability must be in [0, 1]")
        if not isinstance(self.repairable, bool):
            raise TypeError("repairable must be a bool")
        object.__setattr__(self, "min_weight", mw)
        object.__setattr__(self, "geo_weight", gw)
        object.__setattr__(self, "missing_desirability", md)

    def score_from(
        self, desirabilities: Mapping[str, float | None]
    ) -> tuple[float, list[str]]:
        """Plan §9.1 dimension score (0..100) + bottleneck metric ids.

        ``desirabilities`` maps metric id -> desirability (or None for
        missing).  Only metric ids in ``self.metric_ids`` are consumed;
        metrics not present are treated as missing with
        ``missing_desirability``.  Bottlenecks are the metric id(s) attaining
        the minimum desirability.
        """
        nums: list[float] = []
        sources: list[str] = []
        for metric_id in self.metric_ids:
            d = desirabilities.get(metric_id)
            if d is None:
                d = self.missing_desirability
            else:
                fd = float(d)
                if not 0.0 <= fd <= 1.0:
                    raise ValueError(
                        f"desirability for {metric_id!r} must be in [0, 1] or None"
                    )
                d = fd
            nums.append(d)
            sources.append(metric_id)
        if not nums:
            raise ValueError("metric_ids cannot be empty")
        geomean = float(math.prod(nums) ** (1.0 / len(nums)))
        min_d = min(nums)
        score = 100.0 * (self.min_weight * min_d + self.geo_weight * geomean)
        bottlenecks = [mid for mid, d in zip(sources, nums) if d == min_d]
        return score, bottlenecks


@dataclass(frozen=True)
class AdmissionFloors:
    """Admission floors + typed gate vocabulary (plan §9.1).

    ``dimension_floors`` maps (a subset of) the 14 health-dimension ids to a
    minimum *grade* floor — a dimension whose grade is below its floor fails
    that dimension for admission.  ``hard_gate_dimensions`` lists dimension ids
    that are hard floor gates: an ungraded/missing hard-gate dimension fails
    the card regardless of other dimensions (plan §9.1 "an S+ RankIC cannot
    compensate for D DataQuality").  ``integrity_gate_ids`` enumerates the
    typed card-level integrity gates (plan §13.1) evaluated by the health-card
    engine against cross-package integrity evidence.
    """

    dimension_floors: Mapping[str, str] = field(default_factory=dict)
    hard_gate_dimensions: tuple[str, ...] = ()
    integrity_gate_ids: tuple[str, ...] = INTEGRITY_GATE_IDS
    require_all_dimensions_graded: bool = True
    use_case: str = "GENERIC"
    required_dimension_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension_floors", MappingProxyType(dict(self.dimension_floors)))
        for dim, floor in self.dimension_floors.items():
            if dim not in HEALTH_DIMENSIONS:
                raise ValueError(
                    f"unknown health dimension in floors: {dim!r}; expected one of "
                    f"{HEALTH_DIMENSIONS}"
                )
            if floor not in HealthGradeVocabulary.RANKED:
                raise ValueError(
                    f"dimension floor must be a ranked grade, got {floor!r}"
                )
        object.__setattr__(self, "hard_gate_dimensions", tuple(self.hard_gate_dimensions))
        for dim in self.hard_gate_dimensions:
            if dim not in HEALTH_DIMENSIONS:
                raise ValueError(
                    f"unknown health dimension in hard gates: {dim!r}; expected "
                    f"one of {HEALTH_DIMENSIONS}"
                )
        object.__setattr__(self, "integrity_gate_ids", tuple(self.integrity_gate_ids))
        for gate in self.integrity_gate_ids:
            if not gate:
                raise ValueError("integrity gate ids must be non-empty strings")
        if not isinstance(self.require_all_dimensions_graded, bool):
            raise TypeError("require_all_dimensions_graded must be a bool")
        required = tuple(dict.fromkeys(self.required_dimension_ids))
        if not set(required).issubset(HEALTH_DIMENSIONS):
            raise ValueError("required_dimension_ids must be canonical health dimensions")
        object.__setattr__(self, "required_dimension_ids", required)


@dataclass(frozen=True)
class CalibrationArtifact:
    """Immutable development-only, family-aware calibration provenance."""

    calibration_id: str
    calibration_version: str
    policy_id: str
    cutoff_as_of: str
    method: str
    family_scope: str
    reference_candidate_hash: str
    sample_count: int
    split_role: str = "DEVELOPMENT"
    sealed_test_used: bool = False

    def __post_init__(self) -> None:
        if not all((self.calibration_id, self.calibration_version, self.policy_id,
                    self.cutoff_as_of, self.method, self.family_scope,
                    self.reference_candidate_hash)):
            raise ValueError("calibration provenance fields are required")
        if self.split_role not in {"TRAIN", "DEVELOPMENT"} or self.sealed_test_used:
            raise ValueError("calibration must be train/development-only; sealed test is forbidden")
        if self.sample_count < 1:
            raise ValueError("calibration sample_count must be positive")


def _anchor_table(
    ge_values: Sequence[float],
    *,
    desirabilities: Sequence[float],
    higher_is_better: bool,
) -> tuple[GradeAnchor, ...]:
    """Build a best->worst GradeAnchor table from thresholds + desirabilities."""
    grades = HealthGradeVocabulary.RANKED
    if len(ge_values) != len(grades) - 1:
        raise ValueError(
            f"expected {len(grades) - 1} thresholds for grades S+..C, "
            f"got {len(ge_values)}"
        )
    if len(desirabilities) != len(grades):
        raise ValueError(
            f"expected {len(grades)} desirabilities, got {len(desirabilities)}"
        )
    seq: list[float | None] = [float(g) for g in ge_values] + [None]
    if higher_is_better:
        # thresholds best->worst strictly decreasing
        if any(a <= b for a, b in zip(seq[:-1], seq[1:-1])):
            raise ValueError("higher-is-better thresholds must strictly decrease")
    else:
        # ceilings best->worst strictly increasing
        if any(a >= b for a, b in zip(seq[:-1], seq[1:-1])):
            raise ValueError("lower-is-better ceilings must strictly increase")
    return tuple(
        GradeAnchor(grade=grades[i], ge=seq[i], desirability=float(desirabilities[i]))
        for i in range(len(grades))
    )


@dataclass(frozen=True)
class FactorHealthPolicy:
    """Immutable, versioned health grading policy (plan §10).

    Binds every numeric anchor used by the FA health engine: the display-grade
    score bands, the RankIC anchor table, the ICIR anchor table, the retention
    anchor table, per-metric grade rules, per-dimension aggregation rules,
    admission floors / hard gates / integrity-gate ids and the grade alphabet.
    No scattered constants live in grading/dimension/health-card code.
    """

    policy_id: str
    policy_version: str
    market: str = "CN"
    frequency: str = "1d"
    target_id: str = "TargetVwapReturnH10"
    universe_class: str = "cn_a_share_daily_full_float"
    factor_family_scope: str = "*"
    description: str = ""
    grade_alphabet: tuple[str, ...] = HealthGradeVocabulary.RANKED

    #: Display score bands (100-point scale), ordered best->worst, strictly
    #: decreasing min_score, last band floor 0 (plan §10):
    #: S+ 90-100 / S 82-89.9 / A+ 76-81.9 / A 70-75.9 / B+ 63-69.9 /
    #: B 55-62.9 / C 45-54.9 / D <45.
    display_score_bands: tuple[tuple[str, float], ...] = (
        ("S+", 90.0),
        ("S", 82.0),
        ("A+", 76.0),
        ("A", 70.0),
        ("B+", 63.0),
        ("B", 55.0),
        ("C", 45.0),
        ("D", 0.0),
    )

    #: Absolute RankIC anchors (plan §10.1), best->worst.
    rank_ic_anchors: tuple[GradeAnchor, ...] = field(
        default_factory=lambda: _anchor_table(
            (0.050, 0.040, 0.030, 0.022, 0.017, 0.012, 0.005),
            desirabilities=(1.00, 0.94, 0.87, 0.80, 0.70, 0.58, 0.40, 0.10),
            higher_is_better=True,
        )
    )
    #: Absolute ICIR anchors (plan §10.2), best->worst.
    icir_anchors: tuple[GradeAnchor, ...] = field(
        default_factory=lambda: _anchor_table(
            (1.50, 1.10, 0.85, 0.65, 0.45, 0.30, 0.15),
            desirabilities=(1.00, 0.94, 0.87, 0.80, 0.70, 0.58, 0.40, 0.10),
            higher_is_better=True,
        )
    )
    #: Absolute validation-retention anchors (plan §13.6), best->worst.
    retention_anchors: tuple[GradeAnchor, ...] = field(
        default_factory=lambda: _anchor_table(
            (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.25),
            desirabilities=(1.00, 0.94, 0.87, 0.80, 0.70, 0.58, 0.40, 0.10),
            higher_is_better=True,
        )
    )

    metric_grade_rules: Mapping[str, MetricGradeRule] = field(default_factory=dict)
    dimension_rules: Mapping[str, DimensionRule] = field(default_factory=dict)
    admission_floors: AdmissionFloors = field(default_factory=AdmissionFloors)
    use_case_admission_floors: Mapping[str, AdmissionFloors] = field(default_factory=dict)
    calibration_ref: str = ""
    policy_status: str = "CALIBRATION_REQUIRED"
    metric_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if not self.policy_version:
            raise ValueError("policy_version is required")
        if not self.grade_alphabet:
            raise ValueError("grade_alphabet must be non-empty")
        mins = [float(b[1]) for b in self.display_score_bands]
        if len(mins) != len(HealthGradeVocabulary.RANKED):
            raise ValueError(
                "display_score_bands must have exactly one row per ranked grade"
            )
        for (g, _), ranked in zip(
            self.display_score_bands, HealthGradeVocabulary.RANKED
        ):
            if g != ranked:
                raise ValueError(
                    "display_score_bands must be ordered S+..D, got "
                    f"{[b[0] for b in self.display_score_bands]}"
                )
        for i in range(len(mins) - 1):
            if not (mins[i] > mins[i + 1]):
                raise ValueError(
                    "display_score_bands min scores must strictly decrease "
                    f"best->worst, got {mins}"
                )
        object.__setattr__(self, "metric_grade_rules", MappingProxyType(dict(self.metric_grade_rules)))
        object.__setattr__(self, "dimension_rules", MappingProxyType(dict(self.dimension_rules)))
        object.__setattr__(self, "use_case_admission_floors", MappingProxyType(dict(self.use_case_admission_floors)))
        object.__setattr__(self, "metric_aliases", MappingProxyType(dict(self.metric_aliases)))
        for rule in self.metric_grade_rules.values():
            if not isinstance(rule, MetricGradeRule):
                raise TypeError("metric_grade_rules values must be MetricGradeRule")
        for rule in self.dimension_rules.values():
            if not isinstance(rule, DimensionRule):
                raise TypeError("dimension_rules values must be DimensionRule")
        if not isinstance(self.admission_floors, AdmissionFloors):
            raise TypeError("admission_floors must be an AdmissionFloors")
        if any(not isinstance(v, AdmissionFloors) for v in self.use_case_admission_floors.values()):
            raise TypeError("use_case_admission_floors values must be AdmissionFloors")

    # -- resolution helpers -------------------------------------------------

    def display_grade_for_score(self, score: float) -> str:
        """Display grade for a 0..100 aggregate score (plan §10 bands)."""
        s = float(score)
        if math.isnan(s) or math.isinf(s):
            return HealthGradeVocabulary.NONE
        for grade, min_score in self.display_score_bands:
            if s >= min_score:
                return grade
        return self.display_score_bands[-1][0]

    def metric_rule(self, metric_id: str) -> MetricGradeRule:
        metric_id = self.metric_aliases.get(metric_id, metric_id)
        try:
            return self.metric_grade_rules[metric_id]
        except KeyError:
            raise KeyError(
                f"no metric_grade_rule for {metric_id!r} under policy "
                f"{self.policy_id}/{self.policy_version}"
            ) from None

    def has_metric_rule(self, metric_id: str) -> bool:
        return self.metric_aliases.get(metric_id, metric_id) in self.metric_grade_rules

    def canonical_metric_id(self, metric_id: str) -> str:
        return self.metric_aliases.get(metric_id, metric_id)

    def admission_for(self, use_case: str | None = None) -> AdmissionFloors:
        if use_case is None:
            return self.admission_floors
        try:
            return self.use_case_admission_floors[use_case]
        except KeyError:
            raise KeyError(f"unknown use_case {use_case!r} for policy {self.policy_id}") from None

    @property
    def policy_hash(self) -> str:
        payload = {
            "id": self.policy_id, "version": self.policy_version,
            "scope": [self.market, self.frequency, self.target_id, self.universe_class],
            "calibration_ref": self.calibration_ref,
            "aliases": dict(self.metric_aliases),
            "use_cases": {k: {"floors": dict(v.dimension_floors), "required": v.required_dimension_ids,
                               "gates": v.integrity_gate_ids} for k, v in self.use_case_admission_floors.items()},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def grade_and_desirability(
        self, metric_id: str, value: float | None
    ) -> tuple[str | None, float | None]:
        rule = self.metric_rule(metric_id)
        return rule.grade_for(value), rule.desirability_for(value)

    def dimension_rule(self, dimension_id: str) -> DimensionRule:
        try:
            return self.dimension_rules[dimension_id]
        except KeyError:
            raise KeyError(
                f"no dimension_rule for {dimension_id!r} under policy "
                f"{self.policy_id}/{self.policy_version}"
            ) from None

    def has_dimension_rule(self, dimension_id: str) -> bool:
        return dimension_id in self.dimension_rules

    def dimension_floor(self, dimension_id: str) -> str | None:
        return self.admission_floors.dimension_floors.get(dimension_id)

    def is_hard_gate_dimension(self, dimension_id: str) -> bool:
        return dimension_id in self.admission_floors.hard_gate_dimensions


# ---------------------------------------------------------------------------
# Default health policy content
# ---------------------------------------------------------------------------

_DEFAULT_RANK_IC_RULE = None
_DEFAULT_ICIR_RULE = None
_DEFAULT_RETENTION_RULE = None


def _default_metric_grade_rules() -> dict[str, MetricGradeRule]:
    """Seed metric grade rules for the anchor-table flagship metrics.

    The shared RankIC / ICIR / retention tables declared on the policy are
    bound under their canonical QE metric ids so the deterministic grader can
    apply them.  Additional metric families gain grade rules in later policy
    versions as QE registers the corresponding ids; an ungraded metric can
    still flow into a dimension as a caller-built
    ``MetricGradeArtifact`` carrying its own desirability.
    """
    from copy import deepcopy

    rank_ic = deepcopy(
        FactorHealthPolicy(policy_id="tmp", policy_version="0").rank_ic_anchors
    )
    icir = deepcopy(
        FactorHealthPolicy(policy_id="tmp", policy_version="0").icir_anchors
    )
    retention = deepcopy(
        FactorHealthPolicy(policy_id="tmp", policy_version="0").retention_anchors
    )

    def lower_is_better(
        metric_id: str,
        *,
        best: float,
        worst: float,
    ) -> MetricGradeRule:
        """Lower-is-better absolute anchor table (best value -> S+, worst -> D).

        Thresholds are *ceilings* (value <= ge) increasing best->worst; the
        final D row catches everything above ``worst``.
        """
        span = worst - best
        return MetricGradeRule(
            metric_id=metric_id,
            higher_is_better=False,
            missing_desirability=0.0,
            anchors=(
                GradeAnchor("S+", best, 1.00),
                GradeAnchor("S", best + 0.15 * span, 0.92),
                GradeAnchor("A+", best + 0.35 * span, 0.84),
                GradeAnchor("A", best + 0.55 * span, 0.74),
                GradeAnchor("B+", best + 0.70 * span, 0.63),
                GradeAnchor("B", best + 0.85 * span, 0.50),
                GradeAnchor("C", worst, 0.32),
                GradeAnchor("D", None, 0.10),
            ),
        )

    def higher_is_better(
        metric_id: str,
        *,
        good: float,
        bad: float,
    ) -> MetricGradeRule:
        """Higher-is-better absolute anchor table (>= good -> S+ .. < bad -> D)."""
        span = good - bad
        return MetricGradeRule(
            metric_id=metric_id,
            higher_is_better=True,
            missing_desirability=0.0,
            anchors=(
                GradeAnchor("S+", good, 1.00),
                GradeAnchor("S", good - 0.15 * span, 0.92),
                GradeAnchor("A+", good - 0.35 * span, 0.84),
                GradeAnchor("A", good - 0.55 * span, 0.74),
                GradeAnchor("B+", good - 0.70 * span, 0.63),
                GradeAnchor("B", good - 0.85 * span, 0.50),
                GradeAnchor("C", bad, 0.32),
                GradeAnchor("D", None, 0.10),
            ),
        )

    return {
        "rank_ic": MetricGradeRule(
            metric_id="rank_ic", anchors=rank_ic, higher_is_better=True,
            missing_desirability=0.0,
        ),
        "rank_ic_ir": MetricGradeRule(
            metric_id="rank_ic_ir", anchors=icir, higher_is_better=True,
            missing_desirability=0.0,
        ),
        "icir": MetricGradeRule(
            metric_id="icir", anchors=icir, higher_is_better=True,
            missing_desirability=0.0,
        ),
        "validation_retention": MetricGradeRule(
            metric_id="validation_retention", anchors=retention, higher_is_better=True,
            missing_desirability=0.0,
        ),
        # --- additional directional families bound by the initial policy ----
        "coverage": higher_is_better("coverage", good=0.95, bad=0.50),
        "missing_ratio": lower_is_better("missing_ratio", best=0.0, worst=0.50),
        "tie_ratio": lower_is_better("tie_ratio", best=0.0, worst=0.30),
        "factor_turnover_rate": lower_is_better(
            "factor_turnover_rate", best=0.05, worst=0.60
        ),
        "cost_drag": lower_is_better("cost_drag", best=0.0, worst=0.15),
        # Canonical QE drawdown is a positive loss magnitude: lower is better.
        "max_drawdown": lower_is_better("max_drawdown", best=0.10, worst=0.45),
        "max_drawdown_duration": lower_is_better(
            "max_drawdown_duration", best=30.0, worst=300.0
        ),
        "staleness": lower_is_better("staleness", best=0.0, worst=30.0),
    }


def _default_dimension_rules() -> dict[str, DimensionRule]:
    """Initial dimension binding over plan §13 evidence families.

    Each of the 14 dimensions aggregates the subset of metric ids that the
    plan §13 family lists.  Metric membership / weights are policy data —
    future waves change them by bumping the policy version.
    """
    return {
        "predictive_power": DimensionRule(
            "predictive_power", ("rank_ic", "rank_ic_ir"), repairable=False,
        ),
        "stability": DimensionRule(
            "stability",
            ("rolling_rank_ic_mean", "rolling_ic_volatility", "ic_sign_consistency"),
        ),
        "robustness": DimensionRule(
            "robustness",
            ("regime_worst_ic", "regime_conditional_ic", "subsample_stability"),
        ),
        "turnover": DimensionRule(
            "turnover", ("factor_turnover_rate",), repairable=True,
        ),
        "capacity": DimensionRule(
            "capacity", ("tradable_coverage", "est_capacity"),
        ),
        "cost_drag": DimensionRule(
            "cost_drag", ("cost_drag",), repairable=True,
        ),
        "drawdown": DimensionRule(
            "drawdown", ("max_drawdown", "max_drawdown_duration"),
        ),
        "tail_risk": DimensionRule(
            "tail_risk", ("cvar_95", "worst_month"),
        ),
        "data_coverage": DimensionRule(
            "data_coverage",
            ("coverage", "missing_ratio", "effective_n", "tie_ratio"),
        ),
        "freshness": DimensionRule(
            "freshness", ("staleness", "label_maturity"),
        ),
        "complexity": DimensionRule(
            "complexity", ("complexity_score", "lookback_days"), repairable=True,
        ),
        "economic_sense": DimensionRule(
            "economic_sense", ("mechanism_support_score",), repairable=False,
        ),
        "shape_quality": DimensionRule(
            "shape_quality",
            ("quantile_monotonicity", "u_shape_score", "top_tail_cliff"),
            required_metric_ids=("top_tail_cliff",),
            optional_metric_ids=("quantile_monotonicity", "u_shape_score"),
        ),
        "regime_sensitivity": DimensionRule(
            "regime_sensitivity", ("regime_dispersion", "regime_sign_consistency"),
        ),
    }


FACTOR_HEALTH_POLICY_CURRENT_ID = "CN_A_SHARE_DAILY_H10_V1"
FACTOR_HEALTH_POLICY_CURRENT_VERSION = "2.0.0"


def _v5_metric_grade_rules() -> dict[str, MetricGradeRule]:
    rules = _default_metric_grade_rules()
    effect_desirability = (1.00, 0.94, 0.87, 0.80, 0.70, 0.58, 0.40, 0.10)
    rank_ic = _anchor_table(
        (0.040, 0.030, 0.022, 0.016, 0.010, 0.005, 0.0),
        desirabilities=effect_desirability, higher_is_better=True,
    )
    raw_icir = _anchor_table(
        (0.50, 0.35, 0.25, 0.18, 0.12, 0.06, 0.0),
        desirabilities=effect_desirability, higher_is_better=True,
    )
    rules["rank_ic"] = MetricGradeRule("rank_ic", rank_ic, unit="correlation", runtime_metric_id="rank_ic")
    rules["rank_icir_raw"] = MetricGradeRule("rank_icir_raw", raw_icir, unit="mean_over_sample_std", runtime_metric_id="ic_ir")
    old_cost = rules["cost_drag"]
    rules["cost_drag"] = MetricGradeRule(
        "cost_drag", old_cost.anchors, higher_is_better=old_cost.higher_is_better,
        evaluation_role="RISK_LOWER", runtime_metric_id="turnover_cost",
    )
    # Risk/resource utilization u=observed/budget. C remains a report grade;
    # admission floors independently reject u>1.
    utilization = _anchor_table(
        (0.35, 0.50, 0.65, 0.80, 0.90, 1.00, 1.20),
        desirabilities=effect_desirability, higher_is_better=False,
    )
    for metric_id in ("drawdown_budget_utilization", "underwater_budget_utilization",
                      "tail_budget_utilization", "cost_budget_utilization"):
        rules[metric_id] = MetricGradeRule(
            metric_id, utilization, higher_is_better=False,
            evaluation_role="RISK_LOWER", unit="ratio",
            runtime_metric_id=("max_drawdown" if metric_id == "drawdown_budget_utilization"
                               else "drawdown_duration" if metric_id == "underwater_budget_utilization"
                               else "cvar_95" if metric_id == "tail_budget_utilization"
                               else "turnover_cost"),
        )
    return rules


def _v5_dimension_rules() -> dict[str, DimensionRule]:
    rules = _default_dimension_rules()
    rules["predictive_power"] = DimensionRule(
        "predictive_power", ("rank_ic", "rank_icir_raw"),
        required_metric_ids=("rank_ic",), optional_metric_ids=("rank_icir_raw",),
        repairable=False,
    )
    rules["drawdown"] = DimensionRule(
        "drawdown", ("drawdown_budget_utilization", "underwater_budget_utilization"),
        required_metric_ids=("drawdown_budget_utilization",),
        optional_metric_ids=("underwater_budget_utilization",),
    )
    # Ties and generic 30-day staleness are diagnostics, not universal grades.
    rules["data_coverage"] = DimensionRule(
        "data_coverage", ("coverage", "missing_ratio", "effective_n"),
        required_metric_ids=("coverage",), optional_metric_ids=("missing_ratio", "effective_n"),
    )
    rules["freshness"] = DimensionRule(
        "freshness", ("label_maturity",), required_metric_ids=("label_maturity",),
    )
    return rules

FACTOR_HEALTH_POLICIES: Mapping[str, tuple[FactorHealthPolicy, ...]] = {
    FACTOR_HEALTH_POLICY_CURRENT_ID: (
        FactorHealthPolicy(
            policy_id=FACTOR_HEALTH_POLICY_CURRENT_ID,
            policy_version="1.0.0",
            market="CN",
            frequency="1d",
            target_id="TargetVwapReturnH10",
            universe_class="cn_a_share_daily_full_float",
            factor_family_scope="*",
            description="R61-FI-025 initial health grading policy for CN A-share "
            "daily H10-vwap factors (plan §8-§10).",
            metric_grade_rules=_default_metric_grade_rules(),
            dimension_rules=_default_dimension_rules(),
            admission_floors=AdmissionFloors(
                dimension_floors={
                    "predictive_power": "B+",
                    "stability": "B",
                    "turnover": "B",
                    "data_coverage": "B",
                    "drawdown": "B",
                    "tail_risk": "C",
                },
                hard_gate_dimensions=("data_coverage",),
                require_all_dimensions_graded=True,
            ),
        ),
        FactorHealthPolicy(
            policy_id=FACTOR_HEALTH_POLICY_CURRENT_ID,
            policy_version="2.0.0",
            market="CN", frequency="1d", target_id="TargetVwapReturnH10",
            universe_class="cn_a_share_daily_full_float", factor_family_scope="*",
            description="V5 H10 cold-start, use-case scoped multidimensional grading policy.",
            rank_ic_anchors=_anchor_table(
                (0.040, 0.030, 0.022, 0.016, 0.010, 0.005, 0.0),
                desirabilities=(1.00, .94, .87, .80, .70, .58, .40, .10),
                higher_is_better=True,
            ),
            icir_anchors=_anchor_table(
                (0.50, 0.35, 0.25, 0.18, 0.12, 0.06, 0.0),
                desirabilities=(1.00, .94, .87, .80, .70, .58, .40, .10),
                higher_is_better=True,
            ),
            metric_grade_rules=_v5_metric_grade_rules(),
            dimension_rules=_v5_dimension_rules(),
            admission_floors=AdmissionFloors(
                dimension_floors={"predictive_power": "B+", "data_coverage": "B"},
                hard_gate_dimensions=("data_coverage",),
                require_all_dimensions_graded=False,
            ),
            use_case_admission_floors={
                "LONG_ONLY_RESEARCH": AdmissionFloors(
                    use_case="LONG_ONLY_RESEARCH",
                    dimension_floors={"data_coverage": "B", "drawdown": "B"},
                    hard_gate_dimensions=("data_coverage", "drawdown"),
                    required_dimension_ids=("predictive_power", "data_coverage", "drawdown", "cost_drag"),
                    require_all_dimensions_graded=False,
                ),
                "LONG_SHORT_RESEARCH": AdmissionFloors(
                    use_case="LONG_SHORT_RESEARCH",
                    dimension_floors={"predictive_power": "B", "data_coverage": "B", "drawdown": "B"},
                    hard_gate_dimensions=("data_coverage", "drawdown"),
                    required_dimension_ids=("predictive_power", "data_coverage", "turnover", "cost_drag", "drawdown", "tail_risk", "capacity"),
                    require_all_dimensions_graded=False,
                ),
                "MODEL_FEATURE": AdmissionFloors(
                    use_case="MODEL_FEATURE",
                    dimension_floors={"data_coverage": "C"},
                    hard_gate_dimensions=("data_coverage",),
                    required_dimension_ids=("robustness", "data_coverage", "complexity"),
                    require_all_dimensions_graded=False,
                ),
            },
            metric_aliases={
                "rank_ic_ir": "rank_icir_raw", "icir": "rank_icir_raw",
                "rankicir": "rank_icir_raw", "rank_icir": "rank_icir_raw",
            },
            calibration_ref="coldstart:CN_A_DAILY_H10:20260907",
            policy_status="CALIBRATION_REQUIRED",
        ),
    ),
}


def get_health_policy(
    policy_id: str = FACTOR_HEALTH_POLICY_CURRENT_ID,
    policy_version: str | None = None,
) -> FactorHealthPolicy:
    versions = FACTOR_HEALTH_POLICIES.get(policy_id)
    if not versions:
        raise KeyError(f"unknown factor-health policy_id: {policy_id}")
    if policy_version is None:
        return versions[-1]
    for policy in versions:
        if policy.policy_version == policy_version:
            return policy
    raise KeyError(
        f"unknown factor-health policy version {policy_version!r} "
        f"for policy_id {policy_id!r}"
    )


# ===========================================================================
# Part 3 — Diagnosis policy (R61-FI-026, plan §9/§13/§15-§17)
# ===========================================================================


@dataclass(frozen=True)
class DiagnosisThresholds:
    """Versioned numeric thresholds for the deterministic diagnosis rules.

    Every threshold the :mod:`~factor_assets.profiling.diagnosis` engine reads
    lives here (policy data), never in rule code.
    """

    # integrity / data quality
    coverage_min: float = 0.5
    missing_ratio_max: float = 0.5
    tie_ratio_max: float = 0.3
    outlier_ratio_max: float = 0.2
    stale_days_max: float = 30.0
    nonfinite_ratio_max: float = 0.05
    min_effective_n: float = 200.0

    # predictive / stability / generalization
    rank_ic_low_abs: float = 0.005
    ic_vol_max: float = 0.05
    recent_degradation_delta_max: float = -0.005
    retention_min: float = 0.5
    train_validation_icir_delta_max: float = 1.0
    hac_tstat_min: float = 2.0
    ic_positive_ratio_min: float = 0.52

    # shape
    u_shape_score_min: float = 0.6
    inverted_u_shape_score_min: float = 0.6
    top_tail_cliff_min: float = 0.5
    bottom_tail_cliff_min: float = 0.5
    shape_stability_min: float = 0.5

    # portfolio economics / drawdown
    turnover_max: float = 0.5
    cost_drag_max: float = 0.1
    max_drawdown_max: float = -0.3
    underwater_days_max: float = 250.0
    cvar_95_max: float = -0.1
    regime_dispersion_max: float = 0.5
    regime_sign_consistency_min: float = 0.5

    # exposures
    max_abs_style_exposure_max: float = 0.4
    industry_exposure_max: float = 0.5
    size_exposure_max: float = 0.5

    # novelty / complexity
    duplicate_corr_min: float = 0.98
    near_duplicate_corr_min: float = 0.90
    novelty_min: float = 0.05
    complexity_max: float = 0.8

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"threshold {name} must be numeric")
            fv = float(value)
            if math.isnan(fv) or math.isinf(fv):
                raise ValueError(f"threshold {name} must be finite")


@dataclass(frozen=True)
class DiagnosisPolicy:
    """Versioned deterministic diagnosis policy (R61-FI-026).

    Binds the severity / repairability / base-confidence of every diagnosis
    tag the engine may emit, plus the numeric thresholds that gate them.
    The engine emits a tag only when its inputs satisfy the threshold under
    this policy; unknown/missing evidence never fabricates a tag.
    """

    policy_id: str
    policy_version: str
    description: str
    severity_by_tag: Mapping[str, str] = field(default_factory=dict)
    repairability_by_tag: Mapping[str, str] = field(default_factory=dict)
    confidence_by_tag: Mapping[str, float] = field(default_factory=dict)
    thresholds: DiagnosisThresholds = field(default_factory=DiagnosisThresholds)

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if not self.policy_version:
            raise ValueError("policy_version is required")
        object.__setattr__(self, "severity_by_tag", dict(self.severity_by_tag))
        object.__setattr__(
            self, "repairability_by_tag", dict(self.repairability_by_tag)
        )
        object.__setattr__(self, "confidence_by_tag", dict(self.confidence_by_tag))
        for tag, severity in self.severity_by_tag.items():
            if severity not in (
                "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO", "UNKNOWN",
            ):
                raise ValueError(
                    f"unknown severity {severity!r} for tag {tag!r}"
                )
        for tag, repairability in self.repairability_by_tag.items():
            if repairability not in (
                "REPAIRABLE", "PARTIALLY_REPAIRABLE", "UNREPAIRABLE", "UNKNOWN",
            ):
                raise ValueError(
                    f"unknown repairability {repairability!r} for tag {tag!r}"
                )
        for tag, confidence in self.confidence_by_tag.items():
            conf = float(confidence)
            if not 0.0 <= conf <= 1.0:
                raise ValueError(
                    f"confidence for tag {tag!r} must be in [0, 1]"
                )
            object.__setattr__(self, "confidence_by_tag", self.confidence_by_tag)

    # -- lookup helpers ------------------------------------------------------

    def severity(self, tag: str, default: str = "UNKNOWN") -> str:
        return self.severity_by_tag.get(tag, default)

    def repairability(self, tag: str, default: str = "UNKNOWN") -> str:
        return self.repairability_by_tag.get(tag, default)

    def confidence(self, tag: str, default: float = 0.5) -> float:
        return float(self.confidence_by_tag.get(tag, default))


DIAGNOSIS_POLICY_CURRENT_ID = "CN_A_SHARE_DAILY_DIAGNOSIS_V1"
DIAGNOSIS_POLICY_CURRENT_VERSION = "1.0.0"


def _default_diagnosis_policy() -> DiagnosisPolicy:
    """Initial deterministic diagnosis policy for CN A-share daily factors.

    Severity / repairability are plan-aligned: integrity & data-quality
    failures are CRITICAL/REPAIRABLE or UNREPAIRABLE; statistical overfit /
    drawdown failures HIGH; shape/exposure failures MEDIUM/LOW; novelty tags
    LOW.  Base confidence is the engine's stated certainty per tag (threshold
    distance can further raise it in the engine — but never above 1.0).
    """
    return DiagnosisPolicy(
        policy_id=DIAGNOSIS_POLICY_CURRENT_ID,
        policy_version="1.0.0",
        description="R61-FI-026 initial deterministic diagnosis policy for CN "
        "A-share daily factors.",
        severity_by_tag={
            "INTEGRITY_FAILURE": "CRITICAL",
            "DATA_QUALITY_FAILURE": "CRITICAL",
            "POOR_COVERAGE": "HIGH",
            "HIGH_TIE_RATIO": "MEDIUM",
            "SPARSE_FACTOR": "MEDIUM",
            "NUMERICAL_INSTABILITY": "CRITICAL",
            "STALE_DATA": "HIGH",
            "LOW_PREDICTIVE": "HIGH",
            "UNSTABLE_IC": "HIGH",
            "RECENT_DEGRADATION": "HIGH",
            "OVERFIT_GENERALIZATION": "HIGH",
            "LOW_STATISTICAL_CONFIDENCE": "MEDIUM",
            "U_SHAPE": "LOW",
            "INVERTED_U": "LOW",
            "TOP_TAIL_COLLAPSE": "MEDIUM",
            "BOTTOM_TAIL_COLLAPSE": "MEDIUM",
            "NONSTATIONARY_SHAPE": "MEDIUM",
            "HIGH_TURNOVER": "MEDIUM",
            "HIGH_COST_DRAG": "HIGH",
            "HIGH_DRAWDOWN": "HIGH",
            "LONG_UNDERWATER": "MEDIUM",
            "REGIME_DEPENDENT": "LOW",
            "SIZE_EXPOSURE": "LOW",
            "INDUSTRY_EXPOSURE": "LOW",
            "MULTI_STYLE_EXPOSURE": "MEDIUM",
            "SEMANTIC_DUPLICATE": "HIGH",
            "VALUE_NEAR_DUPLICATE": "MEDIUM",
            "LOW_NOVELTY": "LOW",
            "HIGH_COMPLEXITY": "LOW",
        },
        repairability_by_tag={
            "INTEGRITY_FAILURE": "UNREPAIRABLE",
            "DATA_QUALITY_FAILURE": "PARTIALLY_REPAIRABLE",
            "POOR_COVERAGE": "REPAIRABLE",
            "HIGH_TIE_RATIO": "PARTIALLY_REPAIRABLE",
            "SPARSE_FACTOR": "REPAIRABLE",
            "NUMERICAL_INSTABILITY": "REPAIRABLE",
            "STALE_DATA": "REPAIRABLE",
            "LOW_PREDICTIVE": "REPAIRABLE",
            "UNSTABLE_IC": "PARTIALLY_REPAIRABLE",
            "RECENT_DEGRADATION": "PARTIALLY_REPAIRABLE",
            "OVERFIT_GENERALIZATION": "REPAIRABLE",
            "LOW_STATISTICAL_CONFIDENCE": "PARTIALLY_REPAIRABLE",
            "U_SHAPE": "REPAIRABLE",
            "INVERTED_U": "REPAIRABLE",
            "TOP_TAIL_COLLAPSE": "PARTIALLY_REPAIRABLE",
            "BOTTOM_TAIL_COLLAPSE": "PARTIALLY_REPAIRABLE",
            "NONSTATIONARY_SHAPE": "UNKNOWN",
            "HIGH_TURNOVER": "REPAIRABLE",
            "HIGH_COST_DRAG": "REPAIRABLE",
            "HIGH_DRAWDOWN": "PARTIALLY_REPAIRABLE",
            "LONG_UNDERWATER": "UNKNOWN",
            "REGIME_DEPENDENT": "UNKNOWN",
            "SIZE_EXPOSURE": "REPAIRABLE",
            "INDUSTRY_EXPOSURE": "REPAIRABLE",
            "MULTI_STYLE_EXPOSURE": "REPAIRABLE",
            "SEMANTIC_DUPLICATE": "UNREPAIRABLE",
            "VALUE_NEAR_DUPLICATE": "REPAIRABLE",
            "LOW_NOVELTY": "UNREPAIRABLE",
            "HIGH_COMPLEXITY": "REPAIRABLE",
        },
        confidence_by_tag={
            "INTEGRITY_FAILURE": 0.95,
            "DATA_QUALITY_FAILURE": 0.9,
            "POOR_COVERAGE": 0.85,
            "HIGH_TIE_RATIO": 0.8,
            "SPARSE_FACTOR": 0.75,
            "NUMERICAL_INSTABILITY": 0.9,
            "STALE_DATA": 0.85,
            "LOW_PREDICTIVE": 0.8,
            "UNSTABLE_IC": 0.8,
            "RECENT_DEGRADATION": 0.7,
            "OVERFIT_GENERALIZATION": 0.75,
            "LOW_STATISTICAL_CONFIDENCE": 0.8,
            "U_SHAPE": 0.6,
            "INVERTED_U": 0.6,
            "TOP_TAIL_COLLAPSE": 0.65,
            "BOTTOM_TAIL_COLLAPSE": 0.65,
            "NONSTATIONARY_SHAPE": 0.6,
            "HIGH_TURNOVER": 0.85,
            "HIGH_COST_DRAG": 0.8,
            "HIGH_DRAWDOWN": 0.85,
            "LONG_UNDERWATER": 0.75,
            "REGIME_DEPENDENT": 0.6,
            "SIZE_EXPOSURE": 0.7,
            "INDUSTRY_EXPOSURE": 0.7,
            "MULTI_STYLE_EXPOSURE": 0.65,
            "SEMANTIC_DUPLICATE": 0.9,
            "VALUE_NEAR_DUPLICATE": 0.75,
            "LOW_NOVELTY": 0.6,
            "HIGH_COMPLEXITY": 0.7,
        },
    )


DIAGNOSIS_POLICIES: Mapping[str, tuple[DiagnosisPolicy, ...]] = {
    DIAGNOSIS_POLICY_CURRENT_ID: (
        _default_diagnosis_policy(),
    ),
}


def get_diagnosis_policy(
    policy_id: str = DIAGNOSIS_POLICY_CURRENT_ID,
    policy_version: str | None = None,
) -> DiagnosisPolicy:
    versions = DIAGNOSIS_POLICIES.get(policy_id)
    if not versions:
        raise KeyError(f"unknown diagnosis policy_id: {policy_id}")
    if policy_version is None:
        return versions[-1]
    for policy in versions:
        if policy.policy_version == policy_version:
            return policy
    raise KeyError(
        f"unknown diagnosis policy version {policy_version!r} "
        f"for policy_id {policy_id!r}"
    )
