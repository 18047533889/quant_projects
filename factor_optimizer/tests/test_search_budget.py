"""Tests for the E5 diagnosis-driven search-budget generator (R61-FI-034)."""

import pytest

from factor_optimizer.policy.repair import DiagnosisKind
from factor_optimizer.policy.repair_registry import (
    DiagnosisSearchBudget,
    RepairBudgetConfig,
    RepairCandidateSlot,
    diagnosis_search_budget,
)


def test_e5_defaults_match_plan():
    config = RepairBudgetConfig()
    assert config.max_primary_diagnoses == 3
    assert config.max_repair_families_per_diagnosis == 2
    assert config.min_candidates_per_family == 2
    assert config.max_candidates_per_family == 4
    assert config.min_total_candidates == 6
    assert config.max_total_candidates == 12


def test_budget_primary_diagnoses_bounded_at_three():
    budget = diagnosis_search_budget(
        [
            DiagnosisKind.HIGH_TURNOVER,
            DiagnosisKind.U_SHAPE,
            DiagnosisKind.OVERFIT_GENERALIZATION,
            DiagnosisKind.SIZE_EXPOSURE,
            DiagnosisKind.INDUSTRY_EXPOSURE,
            DiagnosisKind.LOW_NOVELTY,
        ]
    )
    assert len(budget.primary_diagnoses) <= 3
    assert len(budget.primary_diagnoses) == len(set(budget.primary_diagnoses))


def test_budget_families_per_diagnosis_capped_at_two():
    budget = diagnosis_search_budget([DiagnosisKind.TOP_TAIL_COLLAPSE])
    by_diag = [s for s in budget.slots if s.diagnosis == "TOP_TAIL_COLLAPSE"]
    assert len(by_diag) <= 2


def test_budget_candidates_per_family_2_4():
    budget = diagnosis_search_budget([DiagnosisKind.HIGH_TURNOVER])
    for slot in budget.slots:
        assert 2 <= slot.max_candidates <= 4


def test_budget_soft_total_6_12():
    budget = diagnosis_search_budget(
        [DiagnosisKind.HIGH_TURNOVER, DiagnosisKind.U_SHAPE, DiagnosisKind.OVERFIT_GENERALIZATION]
    )
    ceiling = budget.total_candidates_ceiling
    assert 6 <= ceiling <= 12
    assert budget.within_soft_budget()


def test_budget_soft_total_respected_when_families_would_exceed():
    # Each primary diagnosis maps to 2 families x up to 4 candidates; a
    # config with 2 candidates per family keeps the ceiling inside the cap.
    config = RepairBudgetConfig(max_candidates_per_family=4)
    budget = diagnosis_search_budget(
        [DiagnosisKind.HIGH_TURNOVER, DiagnosisKind.U_SHAPE, DiagnosisKind.OVERFIT_GENERALIZATION],
        config=config,
    )
    assert budget.total_candidates_ceiling <= budget.max_total_candidates


def test_budget_custom_family_ceiling_overrides_declaration_max():
    # U_SHAPE_REPAIR declares max 3; config max 2 should win.
    config = RepairBudgetConfig(max_candidates_per_family=2, min_candidates_per_family=1)
    budget = diagnosis_search_budget([DiagnosisKind.U_SHAPE], config=config)
    for slot in budget.slots:
        assert slot.max_candidates <= 2


def test_budget_respects_family_declared_maximum_candidates():
    # LOW_PREDICTIVE routes to SIGN_ORIENTATION (max 2) + OPERATOR_SWAP (max 3).
    budget = diagnosis_search_budget([DiagnosisKind.LOW_PREDICTIVE])
    by_family = {slot.family: slot.max_candidates for slot in budget.slots}
    assert by_family["SIGN_ORIENTATION"] <= 2
    assert by_family["OPERATOR_SWAP"] <= 3


def test_budget_configurable_primary_diagnoses():
    config = RepairBudgetConfig(max_primary_diagnoses=1)
    budget = diagnosis_search_budget(
        [DiagnosisKind.HIGH_TURNOVER, DiagnosisKind.U_SHAPE],
        config=config,
    )
    assert len(budget.primary_diagnoses) == 1


def test_budget_evidence_conditioned_rules_apply():
    # SIZE_EXPOSURE + residual alpha -> SIZE_NEUTRALIZATION only (1 family).
    budget = diagnosis_search_budget(
        [DiagnosisKind.SIZE_EXPOSURE],
        evidence={"residual_rank_ic": 0.02},
    )
    assert [s.family for s in budget.slots] == ["SIZE_NEUTRALIZATION"]


def test_budget_slots_carry_diagnosis_and_family():
    budget = diagnosis_search_budget([DiagnosisKind.HIGH_TURNOVER])
    for slot in budget.slots:
        assert slot.diagnosis == "HIGH_TURNOVER"
        assert slot.family in {"CAUSAL_SMOOTHING", "DECAY_REFINEMENT", "WINDOW_REFINEMENT"}


def test_budget_versioned_and_serializable():
    budget = diagnosis_search_budget([DiagnosisKind.U_SHAPE])
    data = budget.to_dict()
    assert data["policy_id"] == "FO_DIAGNOSIS_REPAIR"
    assert data["policy_version"] == "1.0.0"
    assert data["primary_diagnoses"] == ["U_SHAPE"]
    assert len(data["slots"]) >= 1
    assert data["max_total_candidates"] == 12
    assert data["min_total_candidates"] == 6


def test_empty_budget_still_valid():
    budget = diagnosis_search_budget([])
    assert budget.primary_diagnoses == ()
    assert budget.slots == ()
    assert budget.total_candidates_ceiling == 0
    # Zero candidates is not a budget violation (nothing invented).
    assert budget.within_soft_budget()


def test_unrepairable_only_diagnoses_yield_empty_plan():
    budget = diagnosis_search_budget([DiagnosisKind.PIT_VIOLATION])
    assert budget.primary_diagnoses == ()
    assert budget.slots == ()


def test_budget_plan_dataclass_validation():
    with pytest.raises(ValueError, match="unknown diagnosis"):
        RepairCandidateSlot(diagnosis="NOPE", family="CAUSAL_SMOOTHING")
    with pytest.raises(ValueError):
        RepairCandidateSlot(diagnosis="HIGH_TURNOVER", family="CAUSAL_SMOOTHING", max_candidates=0)


def test_diagnosis_search_budget_preserves_input_order():
    budget = diagnosis_search_budget(
        [DiagnosisKind.U_SHAPE, DiagnosisKind.HIGH_TURNOVER, DiagnosisKind.OVERFIT_GENERALIZATION]
    )
    assert budget.primary_diagnoses[0] == "U_SHAPE"
    assert budget.primary_diagnoses[1] == "HIGH_TURNOVER"
    assert budget.primary_diagnoses[2] == "OVERFIT_GENERALIZATION"
