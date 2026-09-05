"""14 health-dimension grade artifacts + deterministic aggregation (R61-FI-025, plan §9).

A :class:`DimensionGradeArtifact` aggregates the graded metrics of one of the
14 canonical health dimensions under a versioned
:class:`~factor_assets.profiling.policies.FactorHealthPolicy`.

Dimension ids are aligned 1:1 with the FO consumer view
(``factor_optimizer.ports.factor_intelligence.HealthDimension``) so FO's
projection is verbatim:

``predictive_power / stability / robustness / turnover / capacity / cost_drag /
drawdown / tail_risk / data_coverage / freshness / complexity / economic_sense /
shape_quality / regime_sensitivity``

The plan §9.1 aggregation is versioned in the policy's per-dimension
:class:`~factor_assets.profiling.policies.DimensionRule`::

    dimension_score = 0.40 * min(metric desirabilities)
                    + 0.60 * geometric_mean(metric desirabilities)

A dimension whose evidence is entirely missing is **not** scored as zero by
this module: :func:`build_dimension_grades` returns the dimension with
``score=None`` / ``grade=None`` / ``evidence_tier=...`` and flags the missing
metric refs on the artifact, so the health card can apply its fail-closed
"all-missing dimension = gate, not 0" rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from factor_assets.profiling.metric_grading import MetricGradeArtifact
from factor_assets.profiling.policies import (
    HEALTH_DIMENSIONS,
    HealthGradeVocabulary,
    get_health_policy,
)

__all__ = [
    "HEALTH_DIMENSIONS",
    "DimensionGradeArtifact",
    "build_dimension_grade",
    "build_dimension_grades",
    "aggregate_dimension_score",
]

#: Evidence tier tokens describing how much of the dimension's metric evidence
#: was COMPUTED and carried a desirability.
EVIDENCE_TIER_COMPLETE = "COMPLETE"
EVIDENCE_TIER_PARTIAL = "PARTIAL"
EVIDENCE_TIER_MISSING = "MISSING"


@dataclass(frozen=True)
class DimensionGradeArtifact:
    """Frozen grade artifact for one of the 14 health dimensions.

    ``metric_grade_refs`` are the :class:`MetricGradeArtifact` records that
    were aggregated (ordered by the policy's metric order).  ``score`` is the
    plan §9.1 aggregate in ``[0, 100]`` (``None`` when no metric desirability
    was available); ``grade`` is the display grade letter for the score band
    under the policy (``None`` when ``score`` is ``None``).  ``bottlenecks``
    lists the metric ids attaining the minimum desirability.  ``repairable``
    and ``diagnosis_tags`` come from the policy's dimension rule / caller.

    ``evidence_tier`` is ``COMPLETE`` when every metric in the dimension rule
    carried a desirability, ``PARTIAL`` when some did, ``MISSING`` when none
    did.  ``missing_metric_refs`` explicitly lists the ids with no grade.
    """

    factor_definition_id: str
    evaluation_ref: str
    dimension_id: str
    metric_grade_refs: tuple[MetricGradeArtifact, ...]
    score: float | None
    grade: str | None
    bottlenecks: tuple[str, ...]
    repairable: bool
    diagnosis_tags: tuple[str, ...]
    evidence_tier: str
    missing_metric_refs: tuple[str, ...]
    dimension_policy_id: str
    dimension_policy_version: str

    def __post_init__(self) -> None:
        if not self.factor_definition_id:
            raise ValueError("factor_definition_id is required")
        if not self.evaluation_ref:
            raise ValueError("evaluation_ref is required")
        if self.dimension_id not in HEALTH_DIMENSIONS:
            raise ValueError(
                f"unknown health dimension {self.dimension_id!r}; expected one "
                f"of {HEALTH_DIMENSIONS}"
            )
        object.__setattr__(self, "metric_grade_refs", tuple(self.metric_grade_refs))
        object.__setattr__(self, "bottlenecks", tuple(self.bottlenecks))
        object.__setattr__(self, "diagnosis_tags", tuple(self.diagnosis_tags))
        object.__setattr__(self, "missing_metric_refs", tuple(self.missing_metric_refs))
        if self.score is not None:
            s = float(self.score)
            if math.isnan(s) or math.isinf(s) or not 0.0 <= s <= 100.0:
                raise ValueError("score must be in [0, 100] or None")
            object.__setattr__(self, "score", s)
        if self.grade is not None and self.grade not in HealthGradeVocabulary.all():
            raise ValueError(f"unknown grade {self.grade!r}")
        if self.evidence_tier not in (
            EVIDENCE_TIER_COMPLETE,
            EVIDENCE_TIER_PARTIAL,
            EVIDENCE_TIER_MISSING,
        ):
            raise ValueError(f"unknown evidence_tier {self.evidence_tier!r}")
        if not isinstance(self.repairable, bool):
            raise TypeError("repairable must be a bool")
        if not self.dimension_policy_id or not self.dimension_policy_version:
            raise ValueError("dimension policy id/version are required")

    @property
    def has_score(self) -> bool:
        return self.score is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_definition_id": self.factor_definition_id,
            "evaluation_ref": self.evaluation_ref,
            "dimension_id": self.dimension_id,
            "metric_grade_refs": [g.to_dict() for g in self.metric_grade_refs],
            "score": self.score,
            "grade": self.grade,
            "bottlenecks": list(self.bottlenecks),
            "repairable": self.repairable,
            "diagnosis_tags": list(self.diagnosis_tags),
            "evidence_tier": self.evidence_tier,
            "missing_metric_refs": list(self.missing_metric_refs),
            "dimension_policy_id": self.dimension_policy_id,
            "dimension_policy_version": self.dimension_policy_version,
        }


def aggregate_dimension_score(
    desirabilities: Sequence[float],
    *,
    min_weight: float = 0.4,
    geo_weight: float = 0.6,
) -> float:
    """Plan §9.1 aggregate: ``0.40*min + 0.60*geomean`` over desirabilities.

    Accepts only numeric desirabilities (callers substitute the policy's
    ``missing_desirability`` first).  Returns a ``[0, 1]`` desirability-scale
    value; multiply by 100 for the 0..100 dimension score.
    """
    nums = [float(d) for d in desirabilities]
    if not nums:
        raise ValueError("desirabilities cannot be empty")
    for d in nums:
        if not 0.0 <= d <= 1.0:
            raise ValueError("desirabilities must be in [0, 1]")
    geomean = float(math.prod(nums) ** (1.0 / len(nums)))
    return min_weight * min(nums) + geo_weight * geomean


def _dimension_score_from_rule(
    rule, desirabilities: Mapping[str, float]
) -> tuple[float, list[str]]:
    """Apply a DimensionRule to numeric desirabilities -> (0..100 score, bottlenecks)."""
    nums: list[float] = []
    present_ids: list[str] = []
    for metric_id in rule.metric_ids:
        d = desirabilities.get(metric_id)
        if d is None:
            d = rule.missing_desirability
        else:
            fd = float(d)
            if not 0.0 <= fd <= 1.0:
                raise ValueError(
                    f"desirability for {metric_id!r} must be in [0, 1]"
                )
            d = fd
        nums.append(d)
        present_ids.append(metric_id)
    score01 = aggregate_dimension_score(
        nums, min_weight=rule.min_weight, geo_weight=rule.geo_weight
    )
    min_d = min(nums)
    bottlenecks = [mid for mid, d in zip(present_ids, nums) if d == min_d]
    return 100.0 * score01, bottlenecks


def build_dimension_grade(
    *,
    factor_definition_id: str,
    evaluation_ref: str,
    dimension_id: str,
    metric_grade_refs: Sequence[MetricGradeArtifact],
    policy=None,
    diagnosis_tags: Sequence[str] = (),
) -> DimensionGradeArtifact:
    """Aggregate graded metrics of one dimension into a frozen dimension artifact.

    Metrics are mapped by their ``metric_id`` onto the policy's
    :class:`~factor_assets.profiling.policies.DimensionRule` for this
    dimension.  A metric in the rule that has **no** grade / desirability
    (bad evidence status) contributes the rule's ``missing_desirability`` and
    is listed in ``missing_metric_refs``.  When **no** metric carried a
    desirability the dimension is returned with ``score=None`` / ``grade=None``
    and ``evidence_tier=MISSING`` (the caller decides gate semantics; this
    module never turns missing evidence into a zero score).
    """
    if policy is None:
        policy = get_health_policy()
    rule = policy.dimension_rule(dimension_id)

    refs = tuple(metric_grade_refs)
    by_id: dict[str, MetricGradeArtifact] = {}
    for ref in refs:
        if not isinstance(ref, MetricGradeArtifact):
            raise TypeError("metric_grade_refs must contain MetricGradeArtifact")
        if ref.metric_id in by_id:
            raise ValueError(f"duplicate metric grade ref for {ref.metric_id!r}")
        by_id[ref.metric_id] = ref

    desirabilities: dict[str, float] = {}
    missing: list[str] = []
    tier = EVIDENCE_TIER_COMPLETE
    for metric_id in rule.metric_ids:
        ref = by_id.get(metric_id)
        d = None if ref is None else ref.desirability
        if ref is not None and ref.evidence_status == "COMPUTED" and d is not None:
            desirabilities[metric_id] = d
        else:
            missing.append(metric_id)

    score: float | None
    grade: str | None
    bottlenecks: tuple[str, ...]
    if not desirabilities:
        score = None
        grade = None
        bottlenecks = ()
        tier = EVIDENCE_TIER_MISSING
    else:
        score01, bottlenecks_list = _dimension_score_from_rule(rule, desirabilities)
        score = score01
        grade = policy.display_grade_for_score(score)
        bottlenecks = tuple(bottlenecks_list)
        if missing:
            tier = EVIDENCE_TIER_PARTIAL
        else:
            tier = EVIDENCE_TIER_COMPLETE

    return DimensionGradeArtifact(
        factor_definition_id=factor_definition_id,
        evaluation_ref=evaluation_ref,
        dimension_id=dimension_id,
        metric_grade_refs=refs,
        score=score,
        grade=grade,
        bottlenecks=bottlenecks,
        repairable=rule.repairable,
        diagnosis_tags=tuple(diagnosis_tags),
        evidence_tier=tier,
        missing_metric_refs=tuple(missing),
        dimension_policy_id=policy.policy_id,
        dimension_policy_version=policy.policy_version,
    )


def _default_desirabilities_for_dimension(
    metric_grade_refs: Sequence[MetricGradeArtifact],
) -> dict[str, float]:
    """Desirability map over computed metric refs (helper for tests/callers).

    Only refs whose evidence_status is COMPUTED and whose desirability is not
    None contribute; anything else is left out (missing).
    """
    out: dict[str, float] = {}
    for ref in metric_grade_refs:
        if ref.evidence_status == "COMPUTED" and ref.desirability is not None:
            out[ref.metric_id] = ref.desirability
    return out


def build_dimension_grades(
    *,
    factor_definition_id: str,
    evaluation_ref: str,
    metric_grade_refs: Mapping[str, Sequence[MetricGradeArtifact]],
    policy=None,
    diagnosis_tags: Mapping[str, Sequence[str]] | None = None,
) -> tuple[DimensionGradeArtifact, ...]:
    """Aggregate all 14 dimensions in canonical order.

    ``metric_grade_refs`` maps metric id -> the graded refs produced for that
    metric (each ref carries the metric id itself; the mapping is a
    convenience index).  Dimensions whose rule metrics are absent entirely
    come back with ``score=None`` / ``evidence_tier=MISSING`` — never a zero.
    """
    if policy is None:
        policy = get_health_policy()
    tags = dict(diagnosis_tags or {})
    all_refs: list[MetricGradeArtifact] = []
    for refs in metric_grade_refs.values():
        all_refs.extend(refs)
    return tuple(
        build_dimension_grade(
            factor_definition_id=factor_definition_id,
            evaluation_ref=evaluation_ref,
            dimension_id=dim_id,
            metric_grade_refs=[r for r in all_refs if r.metric_id in {
                mid for mid in policy.dimension_rules.get(dim_id).metric_ids
            }] if policy.has_dimension_rule(dim_id) else (),
            policy=policy,
            diagnosis_tags=tags.get(dim_id, ()),
        )
        for dim_id in HEALTH_DIMENSIONS
    )
