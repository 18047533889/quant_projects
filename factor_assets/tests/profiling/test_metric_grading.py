# -*- coding: utf-8 -*-
"""Tests for factor_assets/profiling — metric grading artifact (R61-FI-025, plan §8).

Coverage:
1. frozen MetricGradeArtifact with the exact plan §8 field surface;
2. iron law: bad evidence statuses (FAILED/INVALID/STALE/UNKNOWN/
   NOT_COMPUTED_STAGE/UNAVAILABLE_INPUT/NOT_APPLICABLE) force grade=None and
   desirability=None and *refuse* caller-supplied numeric grade/desirability
   (missing evidence is never numeric zero, never a grade);
3. COMPUTED evidence may carry grade + desirability via policy anchor tables;
4. absolute grade and cohort percentile both stored, never derived from one
   another;
5. QE lowercase status aliases map onto the canonical tokens;
6. deterministic grading entry ``grade_metric_evidence`` is idempotent.
"""

import math
from dataclasses import FrozenInstanceError

import pytest

from factor_assets.profiling.metric_grading import (
    BAD_EVIDENCE_STATUSES,
    MetricGradeArtifact,
    grade_metric_evidence,
    resolve_evidence_status,
)
from factor_assets.profiling.policies import (
    HealthGradeVocabulary,
    get_health_policy,
)

POLICY = get_health_policy()


class TestArtifactSurface:
    def test_frozen(self):
        a = MetricGradeArtifact(metric_id="rank_ic", evidence_status="COMPUTED")
        with pytest.raises(FrozenInstanceError):
            a.grade = "A"  # type: ignore[misc]

    def test_plan8_fields_present(self):
        a = MetricGradeArtifact(
            metric_id="rank_ic",
            metric_version="1",
            value=0.04,
            evidence_status="COMPUTED",
            grade="S",
            desirability=0.94,
            cohort_percentile=72.0,
            confidence_interval=(0.02, 0.06),
            statistical_confidence=0.9,
            raw_relative_delta=0.05,
            grading_policy_id=POLICY.policy_id,
            grading_policy_version=POLICY.policy_version,
            evaluation_ref="eval-1",
        )
        assert a.metric_id == "rank_ic"
        assert a.value == 0.04
        assert a.grade == "S"
        assert a.desirability == 0.94
        assert a.cohort_percentile == 72.0
        assert a.confidence_interval == (0.02, 0.06)
        assert a.statistical_confidence == 0.9
        assert a.raw_relative_delta == 0.05
        assert a.grading_policy_id == POLICY.policy_id
        assert a.grading_policy_version == POLICY.policy_version
        assert a.evaluation_ref == "eval-1"


class TestIronLawNoGradeOnBadEvidence:
    @pytest.mark.parametrize("status", list(BAD_EVIDENCE_STATUSES))
    def test_bad_status_forces_none_grade_and_desirability(self, status):
        a = MetricGradeArtifact(
            metric_id="rank_ic", value=0.06, evidence_status=status
        )
        assert a.grade is None
        assert a.desirability is None
        assert a.has_grade is False
        assert a.has_desirability is False

    @pytest.mark.parametrize(
        "status", ["FAILED", "INVALID", "STALE", "UNKNOWN", "NOT_APPLICABLE"]
    )
    def test_constructor_refuses_numeric_grade_on_bad_evidence(self, status):
        with pytest.raises(ValueError):
            MetricGradeArtifact(
                metric_id="rank_ic",
                value=0.06,
                evidence_status=status,
                grade="A",
                desirability=0.8,
            )

    def test_constructor_refuses_desirability_only_on_bad_evidence(self):
        with pytest.raises(ValueError):
            MetricGradeArtifact(
                metric_id="rank_ic",
                value=0.06,
                evidence_status="FAILED",
                desirability=0.8,
            )

    def test_grading_entry_bad_evidence_never_grades(self):
        for status in ("FAILED", "INVALID", "STALE", "UNKNOWN",
                       "NOT_COMPUTED_STAGE", "UNAVAILABLE_INPUT", "NOT_APPLICABLE"):
            a = grade_metric_evidence(
                metric_id="rank_ic", value=0.09, evidence_status=status,
                evaluation_ref="ev",
            )
            assert a.grade is None
            assert a.desirability is None
            assert a.evidence_status == status

    def test_unknown_status_fails_closed(self):
        with pytest.raises(ValueError):
            grade_metric_evidence(
                metric_id="rank_ic", value=0.04, evidence_status="MADE_UP"
            )


class TestComputedGrading:
    def test_rank_ic_high_scores_s_plus(self):
        a = grade_metric_evidence(
            metric_id="rank_ic", value=0.06, evidence_status="COMPUTED"
        )
        assert a.grade == "S+"
        assert a.desirability == 1.0
        assert a.value == 0.06

    def test_rank_ic_weak_nonnegative_scores_c(self):
        a = grade_metric_evidence(
            metric_id="rank_ic", value=0.001, evidence_status="COMPUTED"
        )
        assert a.grade == "C"
        assert a.desirability == 0.4

    def test_icir_anchor_policy_bound(self):
        a = grade_metric_evidence(
            metric_id="rank_ic_ir", value=1.6, evidence_status="COMPUTED"
        )
        assert a.grade == "S+"
        assert a.desirability == 1.0

    def test_validation_retention_policy_bound(self):
        a = grade_metric_evidence(
            metric_id="validation_retention",
            value=0.95, evidence_status="COMPUTED",
        )
        assert a.grade == "S+"
        b = grade_metric_evidence(
            metric_id="validation_retention",
            value=0.2, evidence_status="COMPUTED",
        )
        assert b.grade == "D"

    def test_ungraded_metric_raises_keyerror(self):
        with pytest.raises(KeyError):
            grade_metric_evidence(
                metric_id="not_a_registered_metric",
                value=0.5, evidence_status="COMPUTED",
            )

    def test_grade_in_ranked_alphabet(self):
        for v in (0.06, 0.045, 0.035, 0.026, 0.019, 0.014, 0.008, 0.001):
            a = grade_metric_evidence(
                metric_id="rank_ic", value=v, evidence_status="COMPUTED"
            )
            assert a.grade in HealthGradeVocabulary.RANKED


class TestAbsoluteVsCohortNeverDerived:
    def test_cohort_percentile_is_not_recomputed_from_grade(self):
        a = MetricGradeArtifact(
            metric_id="rank_ic",
            value=0.04,
            evidence_status="COMPUTED",
            grade="S",
            cohort_percentile=42.0,  # S grade with a low cohort percentile is legal
        )
        assert a.grade == "S"
        assert a.cohort_percentile == 42.0

    def test_grade_does_not_imply_cohort_percentile(self):
        a = grade_metric_evidence(
            metric_id="rank_ic", value=0.04, evidence_status="COMPUTED"
        )
        assert a.grade == "S+"
        assert a.cohort_percentile is None

    def test_cohort_percentile_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            MetricGradeArtifact(
                metric_id="rank_ic",
                evidence_status="COMPUTED",
                grade="S",
                cohort_percentile=101.0,
            )


class TestStatusResolution:
    def test_qe_lowercase_aliases_map(self):
        assert resolve_evidence_status("not_computed") == "NOT_COMPUTED_STAGE"
        assert resolve_evidence_status("label_not_mature") == "NOT_COMPUTED_STAGE"
        assert resolve_evidence_status("invalid_evidence") == "INVALID"
        assert resolve_evidence_status("failed") == "FAILED"
        assert resolve_evidence_status("unavailable") == "UNAVAILABLE_INPUT"
        assert resolve_evidence_status("unsupported") == "NOT_APPLICABLE"
        assert resolve_evidence_status("computed") == "COMPUTED"

    def test_grading_accepts_qe_lowercase(self):
        a = grade_metric_evidence(
            metric_id="rank_ic", value=0.06, evidence_status="computed"
        )
        assert a.evidence_status == "COMPUTED"
        assert a.grade == "S+"

    def test_missing_status_resolves_to_none_token(self):
        assert resolve_evidence_status(None) == "NONE"


class TestValidation:
    def test_nonfinite_value_rejected(self):
        with pytest.raises(ValueError):
            MetricGradeArtifact(
                metric_id="rank_ic", value=float("nan"), evidence_status="COMPUTED"
            )

    def test_interval_lo_le_hi(self):
        with pytest.raises(ValueError):
            MetricGradeArtifact(
                metric_id="rank_ic",
                evidence_status="COMPUTED",
                confidence_interval=(0.3, 0.1),
            )

    def test_desirability_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            MetricGradeArtifact(
                metric_id="rank_ic",
                value=0.04,
                evidence_status="COMPUTED",
                desirability=1.5,
            )

    def test_statistical_confidence_range_checked(self):
        with pytest.raises(ValueError):
            MetricGradeArtifact(
                metric_id="rank_ic",
                evidence_status="COMPUTED",
                statistical_confidence=-0.1,
            )

    def test_metric_version_must_be_version_or_empty(self):
        with pytest.raises(ValueError):
            MetricGradeArtifact(metric_id="rank_ic", metric_version="not-a-version")

    def test_grade_requires_computed(self):
        with pytest.raises(ValueError):
            MetricGradeArtifact(
                metric_id="rank_ic", evidence_status="NOT_COMPUTED_STAGE", grade="A"
            )

    def test_idempotent_grading(self):
        a1 = grade_metric_evidence(
            metric_id="rank_ic", value=0.04, evidence_status="COMPUTED",
            evaluation_ref="ev-1",
        )
        a2 = grade_metric_evidence(
            metric_id="rank_ic", value=0.04, evidence_status="COMPUTED",
            evaluation_ref="ev-1",
        )
        assert a1 == a2
        assert a1.to_dict() == a2.to_dict()
