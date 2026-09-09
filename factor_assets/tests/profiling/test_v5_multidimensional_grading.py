from dataclasses import replace

import pytest

from factor_assets.profiling.dimensions import (
    EVIDENCE_TIER_COMPLETE,
    EVIDENCE_TIER_MISSING,
    DimensionGradeArtifact,
    HEALTH_DIMENSIONS,
)
from factor_assets.profiling.health_card import IntegrityGateResult, build_health_card
from factor_assets.profiling.metric_grading import grade_metric_evidence
from factor_assets.profiling.policies import (
    AdmissionFloors,
    CalibrationArtifact,
    FactorHealthPolicy,
    get_health_policy,
)


def _dimensions(policy, *, present):
    out = []
    for dimension_id in HEALTH_DIMENSIONS:
        is_present = dimension_id in present
        out.append(DimensionGradeArtifact(
            factor_definition_id="factor-A", evaluation_ref="eval-A",
            dimension_id=dimension_id, metric_grade_refs=(),
            score=70.0 if is_present else None,
            grade="A" if is_present else None, bottlenecks=(), repairable=True,
            diagnosis_tags=(),
            evidence_tier=EVIDENCE_TIER_COMPLETE if is_present else EVIDENCE_TIER_MISSING,
            missing_metric_refs=(), dimension_policy_id=policy.policy_id,
            dimension_policy_version=policy.policy_version,
        ))
    return out


def _gates(policy, use_case):
    return [IntegrityGateResult(g, True, f"evidence://factor-A/recipe-A/split-dev/snapshot-A/{g}")
            for g in policy.admission_for(use_case).integrity_gate_ids]


def test_t09_raw_rankicir_alias_is_point_two_five_and_h10_scoped():
    policy = get_health_policy()
    artifact = grade_metric_evidence(
        metric_id="rank_ic_ir", value=.02 / .08, evidence_status="computed",
        policy=policy, horizon=10,
    )
    assert artifact.metric_id == "rank_icir_raw"
    assert artifact.value == pytest.approx(.25)
    assert artifact.effect_grade == "A+"
    assert artifact.evidence_level == "E1"  # point score alone cannot claim E2/E3
    assert policy.target_id.endswith("H10")


def test_t15_empty_wrong_or_missing_policy_gates_fail_closed():
    policy = get_health_policy()
    dims = _dimensions(policy, present=set(HEALTH_DIMENSIONS))
    card = build_health_card(
        factor_definition_id="factor-A", evaluation_ref="eval-A",
        dimension_grades=dims, integrity_gates={}, policy=policy,
        use_case="LONG_ONLY_RESEARCH",
    )
    assert not card.admission_relevant.admissible
    with pytest.raises(ValueError, match="exactly cover"):
        build_health_card(
            factor_definition_id="factor-A", evaluation_ref="eval-A",
            dimension_grades=dims, integrity_gates=_gates(policy, "LONG_ONLY_RESEARCH")[:-1],
            policy=policy, use_case="LONG_ONLY_RESEARCH",
        )


def test_t16_positive_drawdown_magnitude_is_not_best_grade():
    artifact = grade_metric_evidence(
        metric_id="max_drawdown", value=.40, evidence_status="COMPUTED",
        policy=get_health_policy(),
    )
    assert artifact.grade not in {"S+", "S"}


def test_t17_t18_ties_staleness_and_wrong_shape_are_not_universal_requirements():
    policy = get_health_policy()
    assert "tie_ratio" not in policy.dimension_rule("data_coverage").metric_ids
    assert "staleness" not in policy.dimension_rule("freshness").metric_ids
    shape = policy.dimension_rule("shape_quality")
    assert "quantile_monotonicity" in shape.optional_metric_ids
    assert "u_shape_score" in shape.optional_metric_ids


def test_actual_14_vector_and_use_case_specific_lo_ls_admission():
    policy = get_health_policy()
    lo_required = set(policy.admission_for("LONG_ONLY_RESEARCH").required_dimension_ids)
    dims = _dimensions(policy, present=lo_required)
    lo = build_health_card(
        factor_definition_id="factor-A", evaluation_ref="eval-A",
        dimension_grades=dims, integrity_gates=_gates(policy, "LONG_ONLY_RESEARCH"),
        policy=policy, use_case="LONG_ONLY_RESEARCH",
    )
    ls = build_health_card(
        factor_definition_id="factor-A", evaluation_ref="eval-A",
        dimension_grades=dims, integrity_gates=_gates(policy, "LONG_SHORT_RESEARCH"),
        policy=policy, use_case="LONG_SHORT_RESEARCH",
    )
    assert len(lo.dimension_grades) == 14
    assert lo.admission_relevant.admissible
    assert not ls.admission_relevant.admissible


def test_policy_deep_freeze_and_calibration_is_development_only():
    policy = get_health_policy()
    with pytest.raises(TypeError):
        policy.metric_grade_rules["new"] = object()
    with pytest.raises(ValueError, match="sealed test"):
        CalibrationArtifact(
            "cal-1", "1", policy.policy_id, "2026-01-01", "family-weighted",
            "continuous", "sha256:abc", 25, split_role="DEVELOPMENT",
            sealed_test_used=True,
        )


def test_policy_only_rescore_reuses_raw_value():
    current = get_health_policy()
    old = get_health_policy(policy_version="1.0.0")
    raw = .25
    old_grade = grade_metric_evidence(
        metric_id="icir", value=raw, evidence_status="computed", policy=old,
        evaluation_ref="same-raw-evaluation",
    )
    new_grade = grade_metric_evidence(
        metric_id="icir", value=raw, evidence_status="computed", policy=current,
        evaluation_ref="same-raw-evaluation",
    )
    assert old_grade.value == new_grade.value == raw
    assert old_grade.evaluation_ref == new_grade.evaluation_ref
    assert old_grade.grade != new_grade.grade
