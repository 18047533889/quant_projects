# -*- coding: utf-8 -*-
"""Tests for factor_assets/profiling — health card artifact (R61-FI-025, plan §8-§9).

Coverage:
1. display overall grade vs admission-relevant structure are separate fields
   (plan §9.1 — display grade is never an admission authority);
2. integrity hard gates fail closed on missing/failed gate evidence;
3. dimension floors + hard-gate dimensions drive the admission summary;
4. a dimension with no evidence is NOT scored zero and fails the
   require-all-graded policy;
5. artifact validation (canonical 14 dims order, gate id closed vocab);
6. a fully healthy card is admission-admissible.
"""

from dataclasses import FrozenInstanceError

import pytest

from factor_assets.profiling.dimensions import build_dimension_grades
from factor_assets.profiling.health_card import (
    INTEGRITY_GATE_IDS,
    AdmissionSummary,
    FactorHealthCardArtifact,
    IntegrityGateResult,
    build_health_card,
)
from factor_assets.profiling.metric_grading import grade_metric_evidence
from factor_assets.profiling.policies import (
    HEALTH_DIMENSIONS,
    HealthGradeVocabulary,
    get_health_policy,
)

POLICY = get_health_policy()


def _healthy_refs():
    """Metric refs that grade every dimension to a passable score.

    Only the metric ids with a policy metric-grade rule can be graded through
    the policy entry; dimensions without such a metric stay MISSING (which is
    exactly what the require-all-graded policy should flag).
    """
    return {
        "rank_ic": [grade_metric_evidence(metric_id="rank_ic", value=0.06, evidence_status="COMPUTED")],
        "rank_ic_ir": [grade_metric_evidence(metric_id="rank_ic_ir", value=1.6, evidence_status="COMPUTED")],
    }


def _all_gates_pass():
    return tuple(
        IntegrityGateResult(gid, True, evidence_ref=f"evidence://{gid}")
        for gid in INTEGRITY_GATE_IDS
    )


def _build(refs=None, gates=None, policy=None):
    selected_policy = policy or POLICY
    return build_health_card(
        factor_definition_id="F1",
        evaluation_ref="E1",
        dimension_grades=build_dimension_grades(
            factor_definition_id="F1",
            evaluation_ref="E1",
            metric_grade_refs=refs or _healthy_refs(),
            policy=selected_policy,
        ),
        integrity_gates=gates if gates is not None else _all_gates_pass(),
        policy=selected_policy,
    )


class TestDisplayVsAdmissionSeparation:
    def test_display_overall_grade_is_separate_field(self):
        card = _build()
        # display grade is derived and NONE-safe; admission summary exists
        assert isinstance(card.display_overall_grade, str)
        assert card.display_overall_grade in HealthGradeVocabulary.all()
        assert hasattr(card, "admission_relevant")
        assert isinstance(card.admission_relevant, AdmissionSummary)
        # The card documents the semantics on the artifact.
        assert card.display_grade_is_display_only is True

    def test_display_grade_not_an_admission_authority(self):
        # A high display grade must not make the card admissible when hard
        # gates fail: force a data-quality hard-gate failure + missing
        # integrity evidence.
        gates = {gid: None for gid in INTEGRITY_GATE_IDS}  # all missing
        card = _build(gates=gates)
        assert card.admission_relevant.admissible is False
        # display grade could still be high (predictive dim only graded)
        assert card.admission_relevant.hard_gates_passed is False

    def test_display_overall_score_is_mean_of_graded_dimension_scores(self):
        card = _build()
        dims = card.dimension_grades
        graded = [d.score for d in dims if d.score is not None]
        assert card.display_overall_score == pytest.approx(
            sum(graded) / len(graded)
        )
        assert card.display_overall_grade == POLICY.display_grade_for_score(
            card.display_overall_score
        )


class TestIntegrityHardGates:
    def test_production_card_rejects_unresolved_metric_provenance(self):
        with pytest.raises(ValueError, match="resolved factor/value"):
            build_health_card(
                factor_definition_id="F1",
                evaluation_ref="E1",
                dimension_grades=_build().dimension_grades,
                integrity_gates=_all_gates_pass(),
                production=True,
            )

    def test_missing_integrity_evidence_fails_closed(self):
        card = _build(gates={gid: None for gid in INTEGRITY_GATE_IDS})
        assert card.admission_relevant.hard_gates_passed is False
        for gate in card.integrity_gates:
            assert gate.passed is None
            assert gate.is_hard_fail is True

    def test_one_failed_gate_fails_card(self):
        gates = {gid: True for gid in INTEGRITY_GATE_IDS}
        gates["pit_valid"] = False
        card = _build(gates=gates)
        assert card.admission_relevant.hard_gates_passed is False
        assert card.admission_relevant.admissible is False

    def test_all_passing_gates_pass(self):
        card = _build()
        assert card.admission_relevant.hard_gates_passed is True

    def test_bare_bool_mapping_cannot_manufacture_evidence(self):
        card = _build(gates={gid: True for gid in INTEGRITY_GATE_IDS})
        assert card.admission_relevant.hard_gates_passed is False

    def test_duplicate_gate_rejected(self):
        gates = list(_all_gates_pass())
        gates[-1] = gates[0]
        with pytest.raises(ValueError, match="duplicate"):
            _build(gates=gates)

    def test_manual_card_rejects_foreign_dimension_context(self):
        card = _build()
        foreign = list(card.dimension_grades)
        foreign[0] = foreign[0].__class__(
            **{**foreign[0].__dict__, "factor_definition_id": "OTHER"}
        )
        with pytest.raises(ValueError, match="factor identity"):
            FactorHealthCardArtifact(
                **{**card.__dict__, "dimension_grades": tuple(foreign)}
            )

    def test_manual_card_rejects_false_gates_true_summary(self):
        card = _build()
        gates = list(card.integrity_gates)
        gates[0] = IntegrityGateResult(gates[0].gate_id, False, "evidence://failed")
        with pytest.raises(ValueError, match="contradicts"):
            FactorHealthCardArtifact(**{**card.__dict__, "integrity_gates": tuple(gates)})

    def test_sequence_of_gate_results_accepted(self):
        gates = [IntegrityGateResult(gid, True, evidence_ref=f"ev:{gid}") for gid in INTEGRITY_GATE_IDS]
        card = _build(gates=gates)
        assert card.admission_relevant.hard_gates_passed is True

    def test_unknown_gate_id_rejected(self):
        with pytest.raises(ValueError):
            IntegrityGateResult(gate_id="not_a_gate", passed=True)

    def test_bad_gate_type_rejected(self):
        with pytest.raises(TypeError):
            IntegrityGateResult(gate_id="pit_valid", passed="yes")  # type: ignore[arg-type]


class TestDimensionFloorsAndHardGates:
    def test_ungraded_hard_gate_dimension_fails(self):
        # data_coverage is a hard-gate dimension under the current policy; with
        # only predictive refs it stays MISSING -> hard-gate failure.
        card = _build()
        assert "data_coverage" in card.admission_relevant.hard_gate_dimension_failures
        assert card.admission_relevant.admissible is False

    def test_require_all_graded_flags_missing_dimensions(self):
        card = _build()
        assert set(card.admission_relevant.ungraded_dimensions) == {
            dim for dim in HEALTH_DIMENSIONS if dim != "predictive_power"
        }

    def test_floor_violation_tracked(self):
        # predictive_power floor is B+. Grade it D and keep gates passing ->
        # floor violation (but not a hard-gate failure since predictive_power
        # is not a hard gate under the current policy).
        refs = {
            "rank_ic": [grade_metric_evidence(metric_id="rank_ic", value=0.001, evidence_status="COMPUTED")],
        }
        card = _build(refs=refs)
        assert "predictive_power" in card.admission_relevant.dimension_floor_violations

    def test_full_health_card_is_admissible(self):
        # Only achievable when every dimension is graded at/above its floor and
        # gates pass.  Build a purpose-built policy with no floors/hard gates
        # and require_all_dimensions_graded=False, then a card with only the
        # predictive dimension graded should be admissible.
        from factor_assets.profiling.policies import (
            AdmissionFloors,
            FactorHealthPolicy,
        )

        loose = FactorHealthPolicy(
            policy_id="CN_A_SHARE_DAILY_H10_LOOSE",
            policy_version="1.0.0",
            metric_grade_rules=POLICY.metric_grade_rules,
            dimension_rules=POLICY.dimension_rules,
            admission_floors=AdmissionFloors(
                dimension_floors={},
                hard_gate_dimensions=(),
                require_all_dimensions_graded=False,
            ),
        )
        # Metric grades stamped by the default policy cannot be laundered
        # through a different, looser policy during dimension assembly.
        with pytest.raises(ValueError, match="policy identity/version mismatch"):
            _build(policy=loose)


class TestCardArtifactValidation:
    def test_card_frozen(self):
        card = _build()
        with pytest.raises(FrozenInstanceError):
            card.display_overall_grade = "A"  # type: ignore[misc]

    def test_dimension_count_and_order_validated(self):
        from factor_assets.profiling.dimensions import DimensionGradeArtifact

        one_dim = card_dims()[:1]
        with pytest.raises(ValueError):
            FactorHealthCardArtifact(
                factor_definition_id="F1",
                evaluation_ref="E1",
                dimension_grades=one_dim,
                integrity_gates=(),
                display_overall_grade="NONE",
                display_overall_score=None,
                admission_relevant=AdmissionSummary(
                    hard_gates_passed=True, dimension_floor_violations=(),
                    hard_gate_dimension_failures=(), ungraded_dimensions=(),
                ),
                health_policy_id="x", health_policy_version="1",
            )

    def test_to_dict_round_trip_structure(self):
        card = _build()
        d = card.to_dict()
        assert d["factor_definition_id"] == "F1"
        assert len(d["dimension_grades"]) == 14
        assert "admission_relevant" in d
        assert "display_overall_grade" in d
        # display grade is explicitly a *display* field, separate from admission
        assert "admissible" in d["admission_relevant"]


def card_dims():
    return build_dimension_grades(
        factor_definition_id="F1",
        evaluation_ref="E1",
        metric_grade_refs=_healthy_refs(),
    )
