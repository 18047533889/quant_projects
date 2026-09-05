"""Tests for the Stage0-5 multi-fidelity profile binding + cost telemetry (R61-FI-036)."""

import pytest

from factor_optimizer.policy.repair import DiagnosisKind
from factor_optimizer.policy.repair_registry import (
    RepairBudgetConfig,
    diagnosis_search_budget,
)
from factor_optimizer.search.multifidelity import (
    DEFAULT_STAGE_PROFILES,
    EvaluationStage,
    FidelityTier,
    MultiStageCostTelemetry,
    StageCostTelemetry,
    StageEvaluationProfile,
    Stage2DiagnosisBudget,
    StageProfileRegistry,
    build_stage2_budget,
)


def test_stage_vocabulary_is_six_canonical_stages():
    assert [s.name for s in EvaluationStage] == [
        "STAGE0", "STAGE1", "STAGE2", "STAGE3", "STAGE4", "STAGE5",
    ]


def test_stage_fidelity_mapping():
    # Stage1 = GPU cheap screen maps to L0 runner fidelity; Stage3 full
    # validation = L2; Stage4 = L3; Stage5 sealed = L4.
    assert EvaluationStage.STAGE1.fidelity_tier is FidelityTier.L0
    assert EvaluationStage.STAGE2.fidelity_tier is FidelityTier.L1
    assert EvaluationStage.STAGE3.fidelity_tier is FidelityTier.L2
    assert EvaluationStage.STAGE4.fidelity_tier is FidelityTier.L3
    assert EvaluationStage.STAGE5.fidelity_tier is FidelityTier.L4


def test_default_stage_profile_bindings_are_named_qe_profiles():
    registry = StageProfileRegistry.default()
    ids = registry.evidence_profiles()
    assert ids["STAGE1"] == "CHEAP_SCREEN_CN_1D"
    assert ids["STAGE2"] == "SHAPE_DIAGNOSTIC_CN_1D"
    assert ids["STAGE3"] == "FULL_VALIDATION_CN_1D"
    assert ids["STAGE4"] == "EXPENSIVE_STATISTICAL_CN_1D"
    assert ids["STAGE5"] == "MODEL_FEATURE_DIAGNOSTIC_CN_1D"


def test_default_stage_registry_is_versioned():
    registry = StageProfileRegistry.default()
    assert registry.policy_id == "FO_EVALUATION_STAGE"
    assert registry.policy_version == "1.0.0"
    assert len(registry) == 6
    assert len(DEFAULT_STAGE_PROFILES) == 6


def test_stage_profile_validation_fails_closed():
    # Non-empty upper-case QE EvidenceProfile id required (no implicit default).
    with pytest.raises(ValueError, match="non-empty"):
        StageEvaluationProfile(stage=EvaluationStage.STAGE1, evidence_profile="")
    with pytest.raises(ValueError, match="upper-case"):
        StageEvaluationProfile(stage=EvaluationStage.STAGE1, evidence_profile="cheap_screen")
    with pytest.raises(ValueError):
        StageEvaluationProfile(stage=EvaluationStage.STAGE1, evidence_profile="")
    assert StageEvaluationProfile(
        stage=EvaluationStage.STAGE4,
        evidence_profile="EXPENSIVE_STATISTICAL_CN_1D",
    ).evidence_profile == "EXPENSIVE_STATISTICAL_CN_1D"


def test_stage_registry_resolves_stage_by_member_and_int():
    registry = StageProfileRegistry.default()
    assert registry.profile_for(EvaluationStage.STAGE3).evidence_profile == "FULL_VALIDATION_CN_1D"
    assert registry.profile_for(3).evidence_profile == "FULL_VALIDATION_CN_1D"
    assert registry.profile_for(5).stage is EvaluationStage.STAGE5
    with pytest.raises(ValueError, match="unknown evaluation stage"):
        registry.profile_for(99)


def test_stage_registry_rejects_policy_mismatch():
    with pytest.raises(ValueError, match="policy_id"):
        StageProfileRegistry(
            [
                StageEvaluationProfile(
                    stage=EvaluationStage.STAGE1,
                    evidence_profile="CHEAP_SCREEN_CN_1D",
                    policy_id="OTHER",
                )
            ]
        )


def test_stage_registry_dict_round_trip():
    registry = StageProfileRegistry.default()
    data = registry.to_dict()
    assert data["policy_id"] == "FO_EVALUATION_STAGE"
    assert len(data["profiles"]) == 6
    assert data["profiles"][0]["stage"] == "STAGE0"


def test_no_implicit_coverage_default_in_stage_bindings():
    """Plan §31/D1: no FO stage may silently default to metrics=('coverage',)."""
    registry = StageProfileRegistry.default()
    for stage in registry.stages:
        profile = registry.profile_for(stage)
        # The profile binding must always be an explicit upper-case QE profile
        # id (Stage0's static entry is also explicit, never an empty string).
        assert profile.evidence_profile
        assert profile.evidence_profile.isupper()


# ---------------------------------------------------------------------------
# Stage 2 diagnosis-directed budget
# ---------------------------------------------------------------------------


def test_stage2_budget_binds_profile_and_plan():
    budget = build_stage2_budget(
        [DiagnosisKind.HIGH_TURNOVER, DiagnosisKind.OVERFIT_GENERALIZATION]
    )
    assert budget.stage_profile.stage is EvaluationStage.STAGE2
    assert budget.stage_profile.evidence_profile == "SHAPE_DIAGNOSTIC_CN_1D"
    assert budget.plan.primary_diagnoses
    assert budget.candidate_ceiling <= 12


def test_stage2_budget_bounded_6_12():
    budget = build_stage2_budget(
        [DiagnosisKind.HIGH_TURNOVER, DiagnosisKind.U_SHAPE, DiagnosisKind.OVERFIT_GENERALIZATION]
    )
    assert 6 <= budget.candidate_ceiling <= 12


def test_stage2_budget_uses_custom_config():
    config = RepairBudgetConfig(max_primary_diagnoses=1, max_repair_families_per_diagnosis=1)
    budget = build_stage2_budget(
        [DiagnosisKind.HIGH_TURNOVER, DiagnosisKind.U_SHAPE],
        budget_config=config,
    )
    assert len(budget.plan.primary_diagnoses) == 1


def test_stage2_budget_requires_stage2_profile():
    with pytest.raises(ValueError, match="STAGE2"):
        Stage2DiagnosisBudget(
            stage_profile=StageEvaluationProfile(
                stage=EvaluationStage.STAGE3,
                evidence_profile="FULL_VALIDATION_CN_1D",
            ),
            plan=diagnosis_search_budget([DiagnosisKind.HIGH_TURNOVER]),
        )


# ---------------------------------------------------------------------------
# Cost telemetry
# ---------------------------------------------------------------------------


def test_stage_cost_telemetry_records_and_accumulates():
    t = StageCostTelemetry(
        stage=EvaluationStage.STAGE1,
        evidence_profile="CHEAP_SCREEN_CN_1D",
    )
    t.record_evaluation(candidate_count=1, metric_count=23, wall_time_s=0.5,
                        gpu_time_s=0.2, h2d_d2h_bytes=1024, peak_vram_bytes=512,
                        bytes_read=100)
    t.record_evaluation(candidate_count=1, metric_count=23, wall_time_s=0.7,
                        gpu_time_s=0.3, h2d_d2h_bytes=2048, peak_vram_bytes=4096,
                        bytes_read=50)
    assert t.candidate_count == 2
    assert t.metric_count == 46
    assert t.wall_time_s == 1.2
    assert t.gpu_time_s == 0.5
    assert t.h2d_d2h_bytes == 3072
    assert t.peak_vram_bytes == 4096  # peak = max, not sum
    assert t.bytes_read == 150


def test_stage_cost_telemetry_negative_rejected():
    t = StageCostTelemetry(stage=EvaluationStage.STAGE1, evidence_profile="CHEAP_SCREEN_CN_1D")
    with pytest.raises(ValueError, match="non-negative"):
        t.record_evaluation(wall_time_s=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        StageCostTelemetry(stage=EvaluationStage.STAGE1, evidence_profile="CHEAP_SCREEN_CN_1D",
                           wall_time_s=-1.0)


def test_stage0_telemetry_marks_profile_static():
    t = StageCostTelemetry(stage=EvaluationStage.STAGE0)
    assert t.evidence_profile == "static"
    # Stage0 is near-zero factor-value cost: no QE profile id is bound.
    assert not t.evidence_profile.isupper() or t.evidence_profile == "static"


def test_multistage_telemetry_accumulates_per_stage():
    mt = MultiStageCostTelemetry()
    mt.record_stage(EvaluationStage.STAGE1, candidate_count=50, metric_count=23, wall_time_s=1.0)
    mt.record_stage(EvaluationStage.STAGE1, candidate_count=10, metric_count=23, wall_time_s=2.0)
    mt.record_stage(EvaluationStage.STAGE3, candidate_count=3, metric_count=39, wall_time_s=9.0)
    report = mt.report()
    assert set(report) == {"STAGE1", "STAGE3"}
    assert report["STAGE1"]["candidate_count"] == 60
    assert report["STAGE1"]["wall_time_s"] == 3.0
    assert report["STAGE3"]["candidate_count"] == 3
    # Stage telemetry carries the versioned stage-profile policy.
    assert report["STAGE1"]["policy_id"] == "FO_EVALUATION_STAGE"
    assert report["STAGE3"]["evidence_profile"] == "FULL_VALIDATION_CN_1D"


def test_multistage_telemetry_serializes():
    mt = MultiStageCostTelemetry()
    mt.record_stage(EvaluationStage.STAGE2, candidate_count=8, metric_count=8, wall_time_s=0.4)
    data = mt.to_dict()
    assert "per_stage" in data
    assert "STAGE2" in data["per_stage"]
    assert data["stage_profiles"]["policy_id"] == "FO_EVALUATION_STAGE"
    # The telemetry dict round-trips into an identical record.
    from factor_optimizer.search.multifidelity import StageCostTelemetry as SCT
    restored = SCT.from_dict(data["per_stage"]["STAGE2"])
    assert restored.candidate_count == 8
    assert restored.evidence_profile == "SHAPE_DIAGNOSTIC_CN_1D"


def test_telemetry_records_different_stages_share_profile_binding():
    mt = MultiStageCostTelemetry()
    s1 = mt.telemetry_for(EvaluationStage.STAGE1)
    s3 = mt.telemetry_for(EvaluationStage.STAGE3)
    assert s1.evidence_profile == "CHEAP_SCREEN_CN_1D"
    assert s3.evidence_profile == "FULL_VALIDATION_CN_1D"
    assert s1.stage is EvaluationStage.STAGE1
    assert s3.stage is EvaluationStage.STAGE3


def test_stage_profiles_reference_strings_not_imports():
    """FO must not import QE — the profile id is a string reference only."""
    import factor_optimizer.search.multifidelity as mf
    import inspect

    source = inspect.getsource(mf)
    assert "import quant_evaluator" not in source
    assert "from quant_evaluator" not in source
