# -*- coding: utf-8 -*-
"""Tests for factor_assets/profiling — versioned policy registries
(R61-FI-025/026, plan §8-§10).

Coverage:
1. FactorHealthPolicyRegistry resolves a versioned policy
   (``CN_A_SHARE_DAILY_H10_V1`` / 1.0.0); unknown id/version fail closed;
2. grade alphabet, display score bands, RankIC / ICIR / retention anchors are
   all **inside the policy object** — no scattered constants, and the anchor
   thresholds match the plan §10 tables verbatim;
3. absolute grade anchors and cohort percentile are independent (a grade is
   never defined as a library percentile; policy version bump changes anchors
   without touching code);
4. DimensionRule weights (plan §9.1 0.4/0.6) + missing desirability are
   versioned; sum-to-1 validated;
5. DiagnosisPolicy registry resolves tags with severity/repairability/confidence
   bound; the tag vocabulary covers the R61-FI-026 first batch.
"""

import pytest

from factor_assets.profiling.policies import (
    DIAGNOSIS_POLICIES,
    DIAGNOSIS_POLICY_CURRENT_ID,
    DIAGNOSIS_POLICY_CURRENT_VERSION,
    FACTOR_HEALTH_POLICIES,
    FACTOR_HEALTH_POLICY_CURRENT_ID,
    FACTOR_HEALTH_POLICY_CURRENT_VERSION,
    AdmissionFloors,
    DiagnosisPolicy,
    DimensionRule,
    FactorHealthPolicy,
    GradeAnchor,
    HEALTH_DIMENSIONS,
    HealthGradeVocabulary,
    MetricGradeRule,
    get_diagnosis_policy,
    get_health_policy,
)


def test_health_policy_rule_mappings_are_deeply_immutable():
    policy = get_health_policy()
    with pytest.raises(TypeError):
        policy.metric_grade_rules["rank_ic"] = policy.metric_grade_rules["rank_ic"]
    with pytest.raises(TypeError):
        policy.dimension_rules["predictive_power"] = policy.dimension_rules["predictive_power"]


class TestHealthPolicyRegistry:
    def test_resolve_current(self):
        p = get_health_policy()
        assert p.policy_id == "CN_A_SHARE_DAILY_H10_V1"
        assert p.policy_version == "2.0.0"
        assert p.market == "CN"
        assert p.frequency == "1d"
        assert p.target_id == "TargetVwapReturnH10"

    def test_registry_constants_match(self):
        assert FACTOR_HEALTH_POLICY_CURRENT_ID == "CN_A_SHARE_DAILY_H10_V1"
        assert FACTOR_HEALTH_POLICY_CURRENT_VERSION == "2.0.0"
        assert FACTOR_HEALTH_POLICIES[FACTOR_HEALTH_POLICY_CURRENT_ID][0].policy_id == \
            FACTOR_HEALTH_POLICY_CURRENT_ID

    def test_unknown_id_fails_closed(self):
        with pytest.raises(KeyError):
            get_health_policy(policy_id="CN_DAILY_H10_MADE_UP")

    def test_unknown_version_fails_closed(self):
        with pytest.raises(KeyError):
            get_health_policy(policy_version="9.9.9")

    def test_resolve_explicit_version(self):
        p = get_health_policy(policy_version="1.0.0")
        assert p.policy_version == "1.0.0"


class TestAnchorsVersionedInPolicy:
    def test_grade_alphabet_policy_bound(self):
        p = get_health_policy()
        assert p.grade_alphabet == ("S+", "S", "A+", "A", "B+", "B", "C", "D")

    def test_display_score_bands_match_plan(self):
        p = get_health_policy()
        bands = dict(p.display_score_bands)
        # plan §10: S+ 90-100 / S 82-89.9 / A+ 76-81.9 / A 70-75.9 /
        #           B+ 63-69.9 / B 55-62.9 / C 45-54.9 / D <45
        assert bands["S+"] == 90.0
        assert bands["S"] == 82.0
        assert bands["A+"] == 76.0
        assert bands["A"] == 70.0
        assert bands["B+"] == 63.0
        assert bands["B"] == 55.0
        assert bands["C"] == 45.0
        assert bands["D"] == 0.0

    def test_rank_ic_anchors_match_plan_101(self):
        p = get_health_policy()
        expected = {
            "S+": 0.040, "S": 0.030, "A+": 0.022, "A": 0.016,
            "B+": 0.010, "B": 0.005, "C": 0.000, "D": None,
        }
        assert {a.grade: a.ge for a in p.rank_ic_anchors} == expected

    def test_icir_anchors_match_plan_102(self):
        p = get_health_policy()
        expected = {
            "S+": 0.50, "S": 0.35, "A+": 0.25, "A": 0.18,
            "B+": 0.12, "B": 0.06, "C": 0.00, "D": None,
        }
        assert {a.grade: a.ge for a in p.icir_anchors} == expected

    def test_retention_anchors_match_plan_136(self):
        p = get_health_policy()
        expected = {
            "S+": 0.90, "S": 0.80, "A+": 0.70, "A": 0.60,
            "B+": 0.50, "B": 0.40, "C": 0.25, "D": None,
        }
        assert {a.grade: a.ge for a in p.retention_anchors} == expected

    def test_metric_grade_rule_binds_rank_ic_anchor(self):
        p = get_health_policy()
        rule = p.metric_rule("rank_ic")
        assert isinstance(rule, MetricGradeRule)
        assert [a.grade for a in rule.anchors] == list(HealthGradeVocabulary.RANKED)
        # same anchor table as the policy-level rank_ic_anchors
        assert {a.grade: a.ge for a in rule.anchors} == {
            a.grade: a.ge for a in p.rank_ic_anchors
        }

    def test_all_anchors_ordered_best_to_worst(self):
        p = get_health_policy()
        for table in (p.rank_ic_anchors, p.icir_anchors, p.retention_anchors):
            ges = [a.ge for a in table]
            non_none = [g for g in ges if g is not None]
            assert non_none == sorted(non_none, reverse=True)

    def test_grade_anchor_validation(self):
        with pytest.raises(ValueError):
            GradeAnchor(grade="X", ge=0.05, desirability=1.0)
        with pytest.raises(ValueError):
            GradeAnchor(grade="S+", ge=0.05, desirability=1.5)


class TestAbsoluteVsCohortPolicySemantics:
    def test_policy_defines_no_cohort_percentile(self):
        # plan §10.3: absolute grade is policy-stable and must never be defined
        # as the current library's percentile.  The policy object carries no
        # percentile-derived grade table.
        p = get_health_policy()
        assert not hasattr(p, "cohort_grade_table")
        assert not hasattr(p, "percentile_to_grade")

    def test_policy_versioning_changes_anchors_without_code(self):
        # A bumped policy version with a different anchor must be a *new*
        # policy object — anchors are data, not constants in code.
        strict = FactorHealthPolicy(
            policy_id="CN_A_SHARE_DAILY_H10_STRICT",
            policy_version="1.0.0",
            rank_ic_anchors=(
                GradeAnchor("S+", 0.06, 1.0),
                GradeAnchor("S", 0.05, 0.9),
                GradeAnchor("A+", 0.04, 0.8),
                GradeAnchor("A", 0.03, 0.7),
                GradeAnchor("B+", 0.02, 0.6),
                GradeAnchor("B", 0.015, 0.5),
                GradeAnchor("C", 0.008, 0.4),
                GradeAnchor("D", None, 0.1),
            ),
        )
        assert strict.rank_ic_anchors[0].ge == 0.06
        current = get_health_policy()
        assert current.rank_ic_anchors[0].ge == 0.040
        assert strict.rank_ic_anchors[0].ge == 0.06


class TestDimensionRulesVersioned:
    def test_plan91_weights_in_policy(self):
        p = get_health_policy()
        rule = p.dimension_rule("predictive_power")
        assert rule.min_weight == pytest.approx(0.4)
        assert rule.geo_weight == pytest.approx(0.6)

    def test_weight_sum_validated(self):
        with pytest.raises(ValueError):
            DimensionRule("x", ("a",), min_weight=0.7, geo_weight=0.7)

    def test_missing_desirability_zero_default(self):
        p = get_health_policy()
        for dim in HEALTH_DIMENSIONS:
            assert p.dimension_rule(dim).missing_desirability == 0.0

    def test_unknown_dimension_floor_rejected(self):
        with pytest.raises(ValueError):
            AdmissionFloors(dimension_floors={"not_a_dim": "B"})

    def test_floor_grade_must_be_ranked(self):
        with pytest.raises(ValueError):
            AdmissionFloors(dimension_floors={"predictive_power": "NONE"})

    def test_hard_gate_must_be_known_dimension(self):
        with pytest.raises(ValueError):
            AdmissionFloors(hard_gate_dimensions=("not_a_dim",))


class TestDiagnosisPolicyRegistry:
    def test_resolve_current(self):
        p = get_diagnosis_policy()
        assert p.policy_id == DIAGNOSIS_POLICY_CURRENT_ID
        assert p.policy_version == DIAGNOSIS_POLICY_CURRENT_VERSION

    def test_registry_has_one_version(self):
        assert len(DIAGNOSIS_POLICIES[DIAGNOSIS_POLICY_CURRENT_ID]) == 1

    def test_unknown_diagnosis_id_fails_closed(self):
        with pytest.raises(KeyError):
            get_diagnosis_policy(policy_id="DIAG_MADE_UP")

    def test_first_batch_tags_bound(self):
        p = get_diagnosis_policy()
        expected_batch = {
            "INTEGRITY_FAILURE", "DATA_QUALITY_FAILURE", "POOR_COVERAGE",
            "HIGH_TIE_RATIO", "SPARSE_FACTOR", "NUMERICAL_INSTABILITY",
            "STALE_DATA", "LOW_PREDICTIVE", "UNSTABLE_IC",
            "RECENT_DEGRADATION", "OVERFIT_GENERALIZATION",
            "LOW_STATISTICAL_CONFIDENCE", "U_SHAPE", "INVERTED_U",
            "TOP_TAIL_COLLAPSE", "BOTTOM_TAIL_COLLAPSE",
            "NONSTATIONARY_SHAPE", "HIGH_TURNOVER", "HIGH_COST_DRAG",
            "HIGH_DRAWDOWN", "LONG_UNDERWATER", "REGIME_DEPENDENT",
            "SIZE_EXPOSURE", "INDUSTRY_EXPOSURE", "MULTI_STYLE_EXPOSURE",
            "SEMANTIC_DUPLICATE", "VALUE_NEAR_DUPLICATE", "LOW_NOVELTY",
            "HIGH_COMPLEXITY",
        }
        assert set(p.severity_by_tag) == expected_batch
        assert set(p.repairability_by_tag) == expected_batch
        assert set(p.confidence_by_tag) == expected_batch

    def test_policy_severity_lookup(self):
        p = get_diagnosis_policy()
        assert p.severity("INTEGRITY_FAILURE") == "CRITICAL"
        assert p.repairability("SEMANTIC_DUPLICATE") == "UNREPAIRABLE"
        assert 0.0 <= p.confidence("POOR_COVERAGE") <= 1.0

    def test_thresholds_finite_and_bound(self):
        p = get_diagnosis_policy()
        for name in p.thresholds.__dataclass_fields__:
            value = getattr(p.thresholds, name)
            assert isinstance(value, (int, float))
            assert value == value  # not NaN

    def test_diagnosis_policy_validation(self):
        with pytest.raises(ValueError):
            DiagnosisPolicy(
                policy_id="X", policy_version="1.0.0", description="d",
                severity_by_tag={"LOW_PREDICTIVE": "NOT_A_SEVERITY"},
            )
