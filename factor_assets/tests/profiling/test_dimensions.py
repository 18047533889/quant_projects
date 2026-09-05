# -*- coding: utf-8 -*-
"""Tests for factor_assets/profiling — 14 health dimensions + plan §9.1
aggregation (R61-FI-025).

Coverage:
1. the 14 dimension ids align exactly with the FO consumer view
   (``factor_optimizer.ports.factor_intelligence.HealthDimension``) — FA is
   the authority, FO projects verbatim;
2. plan §9.1 aggregation ``0.40*min + 0.60*geometric_mean`` hand-computed
   cross-check on both the module helper and the policy-bound
   ``DimensionRule.score_from``;
3. min-weight term dominates a single bad (low) desirability;
4. geometric-mean term: all-equal desirabilities collapse to that value;
5. a dimension whose metrics are entirely missing comes back with
   ``score=None`` / ``grade=None`` / ``evidence_tier=MISSING`` (missing
   evidence is never a zero score);
6. PARTIAL evidence tier when some metrics carry desirability;
7. bottlenecks = metric ids attaining the min desirability;
8. unknown dimension id fails closed.
"""

import math

import pytest

from factor_assets.profiling.dimensions import (
    HEALTH_DIMENSIONS,
    DimensionGradeArtifact,
    aggregate_dimension_score,
    build_dimension_grade,
    build_dimension_grades,
)
from factor_assets.profiling.metric_grading import grade_metric_evidence
from factor_assets.profiling.policies import (
    FactorHealthPolicy,
    MetricGradeRule,
    get_health_policy,
)

POLICY = get_health_policy()

# The exact FO consumer dimension order (mirrors HealthDimension.all()) —
# this test pins the 14-dim alignment contract.
FO_DIMENSION_IDS = (
    "predictive_power", "stability", "robustness", "turnover", "capacity",
    "cost_drag", "drawdown", "tail_risk", "data_coverage", "freshness",
    "complexity", "economic_sense", "shape_quality", "regime_sensitivity",
)


def _grade(metric_id, value, status="COMPUTED"):
    return grade_metric_evidence(metric_id=metric_id, value=value, evidence_status=status)


class TestFourteenDimensionAlignment:
    def test_dimension_ids_align_with_fo_consumer_view(self):
        assert HEALTH_DIMENSIONS == FO_DIMENSION_IDS

    def test_policy_dimension_rules_cover_all_14(self):
        assert sorted(POLICY.dimension_rules) == sorted(HEALTH_DIMENSIONS)

    def test_every_policy_dimension_rule_references_known_metric_order(self):
        for dim in HEALTH_DIMENSIONS:
            rule = POLICY.dimension_rule(dim)
            assert rule.dimension_id == dim
            assert len(rule.metric_ids) >= 1


class TestAggregationFormula:
    def test_plan91_helper_hand_computed(self):
        # 0.40 * min + 0.60 * geomean over [1.0, 0.5]
        d = [1.0, 0.5]
        expected = 0.4 * 0.5 + 0.6 * math.sqrt(0.5)
        assert aggregate_dimension_score(d) == pytest.approx(expected)

    def test_plan91_helper_three_values(self):
        d = [0.9, 0.6, 0.3]
        expected = 0.4 * 0.3 + 0.6 * (0.9 * 0.6 * 0.3) ** (1 / 3)
        assert aggregate_dimension_score(d) == pytest.approx(expected)

    def test_geometric_mean_of_equality_collapses(self):
        d = [0.8, 0.8, 0.8]
        assert aggregate_dimension_score(d) == pytest.approx(0.8)

    def test_min_weight_term_dominates_bad_metric(self):
        # [1.0, 1.0, 0.05]: the geometric mean alone would be (0.05)^(1/3)
        # ~= 0.37; the 0.40 min weight drags the aggregate under the pure
        # geomean of the *two good* metrics (sqrt(1.0) = 1.0) as well.
        score = aggregate_dimension_score([1.0, 1.0, 0.05])
        assert score < (0.05) ** (1 / 3)
        # .. and the arithmetic-mean of all three (0.683) is far higher
        assert score < 0.4

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            aggregate_dimension_score([])

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            aggregate_dimension_score([0.5, 1.5])


class TestDimensionGradeArtifact:
    def _pred_power(self, rank_ic=0.06, icir=1.6, status_ic="COMPUTED",
                    status_ir="COMPUTED"):
        refs = [
            grade_metric_evidence(
                metric_id="rank_ic", value=rank_ic, evidence_status=status_ic
            ),
            grade_metric_evidence(
                metric_id="rank_ic_ir", value=icir, evidence_status=status_ir
            ),
        ]
        return build_dimension_grade(
            factor_definition_id="F1",
            evaluation_ref="E1",
            dimension_id="predictive_power",
            metric_grade_refs=refs,
        )

    def test_all_s_plus_dimension_is_100(self):
        d = self._pred_power()
        assert d.score == pytest.approx(100.0)
        assert d.grade == "S+"
        assert d.evidence_tier == "COMPLETE"
        # every metric shares the same (maximum) desirability so all are
        # "bottlenecks" in the min sense — the aggregation is a pure S+
        assert d.bottlenecks == ("rank_ic", "rank_ic_ir")
        assert d.missing_metric_refs == ()
        assert d.has_score

    def test_mixed_desirability_hand_computed(self):
        # rank_ic = 0.003 -> D desirability 0.10; icir = 1.6 -> S+ desirability 1.0
        d = self._pred_power(rank_ic=0.003, icir=1.6)
        expected01 = 0.4 * 0.10 + 0.6 * math.sqrt(0.10)
        assert d.score == pytest.approx(100.0 * expected01)
        assert d.bottlenecks == ("rank_ic",)

    def test_single_metric_dimension_min_equals_geomean(self):
        # turnover aggregates a single metric: 0.4*min + 0.6*geomean == desirability
        ref = grade_metric_evidence(
            metric_id="factor_turnover_rate", value=0.9, evidence_status="COMPUTED"
        )
        assert ref.grade == "D"  # 0.9 turnover is far above the worst ceiling
        d = build_dimension_grade(
            factor_definition_id="F1",
            evaluation_ref="E1",
            dimension_id="turnover",
            metric_grade_refs=[ref],
        )
        assert ref.grade == d.grade
        assert ref.desirability * 100 == pytest.approx(d.score)

    def test_missing_evidence_never_zero_score(self):
        ref = grade_metric_evidence(
            metric_id="rank_ic", value=0.06, evidence_status="FAILED"
        )
        d = build_dimension_grade(
            factor_definition_id="F1",
            evaluation_ref="E1",
            dimension_id="predictive_power",
            metric_grade_refs=[ref],
        )
        assert d.score is None
        assert d.grade is None
        assert d.evidence_tier == "MISSING"
        assert set(d.missing_metric_refs) == {"rank_ic", "rank_ic_ir"}

    def test_partial_evidence_tier(self):
        # one COMPUTED, one missing (not in the passed refs)
        ref = grade_metric_evidence(
            metric_id="rank_ic", value=0.06, evidence_status="COMPUTED"
        )
        d = build_dimension_grade(
            factor_definition_id="F1",
            evaluation_ref="E1",
            dimension_id="predictive_power",
            metric_grade_refs=[ref],
        )
        assert d.evidence_tier == "PARTIAL"
        assert d.score is not None
        assert "rank_ic_ir" in d.missing_metric_refs

    def test_dimension_grade_artifact_frozen(self):
        d = self._pred_power()
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            d.score = 50.0  # type: ignore[misc]

    def test_unknown_dimension_fails_closed(self):
        ref = grade_metric_evidence(
            metric_id="rank_ic", value=0.06, evidence_status="COMPUTED"
        )
        with pytest.raises(KeyError):
            build_dimension_grade(
                factor_definition_id="F1",
                evaluation_ref="E1",
                dimension_id="not_a_dimension",
                metric_grade_refs=[ref],
            )

    def test_unknown_metric_in_rule_raises_keyerror(self):
        # complexity aggregates metrics with no policy grade rule -> a caller
        # cannot grade them through the policy entry (KeyError fail-closed)
        with pytest.raises(KeyError):
            grade_metric_evidence(
                metric_id="complexity_score", value=0.9, evidence_status="COMPUTED"
            )

    def test_policy_missing_dimension_rule_fails_closed(self):
        custom = FactorHealthPolicy(
            policy_id="CN_A_SHARE_DAILY_H10_CUSTOM",
            policy_version="1.0.0",
            metric_grade_rules={
                "rank_ic": POLICY.metric_rule("rank_ic"),
            },
            dimension_rules={},
        )
        ref = grade_metric_evidence(
            metric_id="rank_ic", value=0.06, evidence_status="COMPUTED"
        )
        with pytest.raises(KeyError):
            build_dimension_grade(
                factor_definition_id="F1",
                evaluation_ref="E1",
                dimension_id="predictive_power",
                metric_grade_refs=[ref],
                policy=custom,
            )


class TestBuildAllDimensions:
    def test_build_all_returns_14_in_canonical_order(self):
        refs = {"rank_ic": [_grade("rank_ic", 0.06)]}
        dims = build_dimension_grades(
            factor_definition_id="F1", evaluation_ref="E1", metric_grade_refs=refs
        )
        assert len(dims) == 14
        assert [d.dimension_id for d in dims] == list(HEALTH_DIMENSIONS)

    def test_only_graded_dimension_scores_others_missing(self):
        refs = {"rank_ic": [_grade("rank_ic", 0.06)]}
        dims = build_dimension_grades(
            factor_definition_id="F1", evaluation_ref="E1", metric_grade_refs=refs
        )
        by = {d.dimension_id: d for d in dims}
        assert by["predictive_power"].evidence_tier == "PARTIAL"
        assert by["predictive_power"].score is not None
        assert by["turnover"].evidence_tier == "MISSING"
        assert by["turnover"].score is None
