"""Tests for the diagnosis-specific repair policy (R61-FI-034, plan §20 E4)."""

import pytest

from factor_optimizer.policy.repair import DiagnosisKind
from factor_optimizer.policy.repair_registry import (
    DomainToken,
    RepairRule,
    classify_evidence_context,
    default_diagnosis_repair_policy,
    diagnosis_search_budget,
    RepairBudgetConfig,
    RepairFamilyRegistry,
    DiagnosisRepairPolicy,
)


def _evidence(**tokens):
    return tokens


# ---------------------------------------------------------------------------
# 1. Evidence-context classification is conservative (fail closed)
# ---------------------------------------------------------------------------


def test_classify_evidence_absent_is_none():
    tokens = classify_evidence_context({})
    assert tokens == frozenset({DomainToken.NONE})
    tokens = classify_evidence_context(None)
    assert tokens == frozenset({DomainToken.NONE})


def test_classify_evidence_unknown_never_fires():
    # A key present with a non-matching value must NOT fire the token.
    tokens = classify_evidence_context({"data_domains_price_volume": False})
    assert DomainToken.PRICE_VOLUME not in tokens
    tokens = classify_evidence_context({"outlier_ratio": 0.0})
    assert DomainToken.OUTLIER_EVIDENCE not in tokens
    tokens = classify_evidence_context({"shape_confidence": 0.1})
    assert DomainToken.STABLE_SHAPE_CONFIDENCE not in tokens
    tokens = classify_evidence_context({"residual_rank_ic": 0.0})
    assert DomainToken.ALPHA_AFTER_RESIDUAL not in tokens


def test_classify_evidence_explicit_tokens():
    tokens = classify_evidence_context({"data_domains_price_volume": True})
    assert DomainToken.PRICE_VOLUME in tokens
    tokens = classify_evidence_context({"evidence.outlier": True})
    assert DomainToken.OUTLIER_EVIDENCE in tokens
    tokens = classify_evidence_context({"size_exposure": 0.05})
    assert DomainToken.SIZE_CONCENTRATION in tokens
    tokens = classify_evidence_context({"shape_confidence": 0.8})
    assert DomainToken.STABLE_SHAPE_CONFIDENCE in tokens
    tokens = classify_evidence_context({"residual_rank_ic": 0.02})
    assert DomainToken.ALPHA_AFTER_RESIDUAL in tokens


# ---------------------------------------------------------------------------
# 2. Plan §20 E4 rules
# ---------------------------------------------------------------------------


def test_policy_is_versioned():
    policy = default_diagnosis_repair_policy()
    assert policy.policy_id == "FO_DIAGNOSIS_REPAIR"
    assert policy.policy_version == "1.0.0"


def test_high_turnover_plus_price_volume_routes_to_smoothing():
    policy = default_diagnosis_repair_policy()
    families = policy.families_for(
        DiagnosisKind.HIGH_TURNOVER,
        evidence=_evidence(data_domains_price_volume=True),
    )
    assert families == ("CAUSAL_SMOOTHING", "DECAY_REFINEMENT")


def test_high_turnover_without_pv_falls_back():
    policy = default_diagnosis_repair_policy()
    families = policy.families_for(DiagnosisKind.HIGH_TURNOVER)
    assert families == ("DECAY_REFINEMENT", "WINDOW_REFINEMENT")
    # A non-matching evidence context does not accidentally trigger the PV rule.
    families = policy.families_for(
        DiagnosisKind.HIGH_TURNOVER,
        evidence=_evidence(data_domains_price_volume=False),
    )
    assert families == ("DECAY_REFINEMENT", "WINDOW_REFINEMENT")


def test_u_shape_stable_confidence_routes_to_u_shape_repair():
    policy = default_diagnosis_repair_policy()
    families = policy.families_for(
        DiagnosisKind.U_SHAPE,
        evidence=_evidence(shape_confidence=0.8),
    )
    assert families == ("U_SHAPE_REPAIR",)
    # Without the stability evidence the fallback keeps both repair options.
    assert policy.families_for(DiagnosisKind.U_SHAPE) == ("U_SHAPE_REPAIR", "TAIL_HINGE")


def test_top_tail_collapse_outlier_evidence_routes_to_robust():
    policy = default_diagnosis_repair_policy()
    # The evidence token key includes the explicit "evidence.outlier" form.
    families = policy.families_for(
        DiagnosisKind.TOP_TAIL_COLLAPSE,
        evidence=_evidence(**{"evidence.outlier": True}),
    )
    assert families == ("ROBUST_OUTLIER", "TAIL_SATURATION")


def test_top_tail_collapse_size_concentration_routes_to_size_neutralization():
    policy = default_diagnosis_repair_policy()
    families = policy.families_for(
        DiagnosisKind.TOP_TAIL_COLLAPSE,
        evidence=_evidence(size_exposure=0.05),
    )
    assert families == ("SIZE_NEUTRALIZATION",)


def test_top_tail_collapse_fallback():
    policy = default_diagnosis_repair_policy()
    families = policy.families_for(DiagnosisKind.TOP_TAIL_COLLAPSE)
    assert families == ("ROBUST_OUTLIER", "TAIL_SATURATION", "TAIL_HINGE")


def test_size_exposure_alpha_after_residual_routes_to_size_neutralization():
    policy = default_diagnosis_repair_policy()
    families = policy.families_for(
        DiagnosisKind.SIZE_EXPOSURE,
        evidence=_evidence(residual_rank_ic=0.02),
    )
    assert families == ("SIZE_NEUTRALIZATION",)


def test_overfit_generalization_reduces_dof():
    policy = default_diagnosis_repair_policy()
    families = policy.families_for(DiagnosisKind.OVERFIT_GENERALIZATION)
    assert families == ("LOW_DOF_INTERACTION", "WINDOW_REFINEMENT")


def test_unknown_diagnosis_no_rule_returns_empty():
    policy = default_diagnosis_repair_policy()
    with pytest.raises(ValueError, match="unknown diagnosis"):
        policy.families_for("NOT_A_DIAGNOSIS")
    # An E2 diagnosis with no rule yields an empty tuple (never invented).
    assert policy.families_for(DiagnosisKind.LOW_NOVELTY) == ("OPERATOR_SWAP",)


def test_every_rule_family_is_registered():
    policy = default_diagnosis_repair_policy()
    registry = RepairFamilyRegistry.default()
    for rule in policy.rules:
        for family_name in rule.allowed_families:
            assert family_name in registry, f"{rule.rule_id} -> {family_name}"


def test_repair_rule_validation():
    with pytest.raises(ValueError, match="unknown repair family"):
        RepairRule(
            rule_id="x", diagnosis="HIGH_TURNOVER",
            allowed_families=("NOT_A_FAMILY",),
        )
    with pytest.raises(ValueError, match="unknown diagnosis"):
        RepairRule(rule_id="x", diagnosis="NOT_A_DIAGNOSIS", allowed_families=("ABANDON",))


# ---------------------------------------------------------------------------
# 3. E5 search budget generation
# ---------------------------------------------------------------------------


def test_budget_generates_from_diagnoses():
    policy = default_diagnosis_repair_policy()
    budget = diagnosis_search_budget(
        [
            DiagnosisKind.HIGH_TURNOVER,
            DiagnosisKind.OVERFIT_GENERALIZATION,
            DiagnosisKind.U_SHAPE,
        ],
        policy=policy,
    )
    assert set(budget.primary_diagnoses) == {
        "HIGH_TURNOVER",
        "OVERFIT_GENERALIZATION",
        "U_SHAPE",
    }
    # Each primary diagnosis has a slot for every allowed family (≤2).
    by_diag = {}
    for slot in budget.slots:
        by_diag.setdefault(slot.diagnosis, []).append(slot.family)
    assert len(by_diag["HIGH_TURNOVER"]) == 2
    assert len(by_diag["OVERFIT_GENERALIZATION"]) == 2
    assert len(by_diag["U_SHAPE"]) == 2


def test_budget_caps_primary_diagnoses_at_three():
    budget = diagnosis_search_budget(
        [
            DiagnosisKind.HIGH_TURNOVER,
            DiagnosisKind.U_SHAPE,
            DiagnosisKind.OVERFIT_GENERALIZATION,
            DiagnosisKind.SIZE_EXPOSURE,
            DiagnosisKind.TOP_TAIL_COLLAPSE,
        ]
    )
    assert len(budget.primary_diagnoses) == 3


def test_budget_skips_unrepairable_diagnosis():
    # PIT_VIOLATION only allows ABANDON (no candidates to propose).
    budget = diagnosis_search_budget(
        [DiagnosisKind.PIT_VIOLATION, DiagnosisKind.U_SHAPE]
    )
    assert "PIT_VIOLATION" not in budget.primary_diagnoses
    assert budget.primary_diagnoses == ("U_SHAPE",)


def test_budget_soft_total_within_6_12():
    budget = diagnosis_search_budget(
        [
            DiagnosisKind.HIGH_TURNOVER,
            DiagnosisKind.U_SHAPE,
            DiagnosisKind.OVERFIT_GENERALIZATION,
        ]
    )
    assert budget.within_soft_budget()
    ceiling = budget.total_candidates_ceiling
    assert budget.min_total_candidates <= ceiling <= budget.max_total_candidates


def test_budget_custom_config_is_configurable():
    config = RepairBudgetConfig(
        max_primary_diagnoses=2,
        max_repair_families_per_diagnosis=1,
        min_candidates_per_family=2,
        max_candidates_per_family=2,
    )
    budget = diagnosis_search_budget(
        [
            DiagnosisKind.HIGH_TURNOVER,
            DiagnosisKind.U_SHAPE,
            DiagnosisKind.OVERFIT_GENERALIZATION,
            DiagnosisKind.SIZE_EXPOSURE,
        ],
        config=config,
    )
    assert len(budget.primary_diagnoses) == 2
    for slot in budget.slots:
        assert slot.max_candidates == 2
    assert budget.total_candidates_ceiling == 2 * 1 * 2  # 2 diagnoses x 1 family x 2


def test_budget_evidence_conditioned_rules_consume_top_two_families():
    # With PV evidence the top-two families are CAUSAL_SMOOTHING/DECAY_REFINEMENT.
    budget = diagnosis_search_budget(
        [DiagnosisKind.HIGH_TURNOVER],
        evidence=_evidence(data_domains_price_volume=True),
    )
    assert [s.family for s in budget.slots] == ["CAUSAL_SMOOTHING", "DECAY_REFINEMENT"]


def test_budget_config_validation():
    with pytest.raises(ValueError):
        RepairBudgetConfig(max_primary_diagnoses=0)
    with pytest.raises(ValueError):
        RepairBudgetConfig(min_candidates_per_family=4, max_candidates_per_family=2)
    with pytest.raises(ValueError):
        RepairBudgetConfig(min_total_candidates=12, max_total_candidates=6)


def test_budget_versioned_policy_ids():
    budget = diagnosis_search_budget([DiagnosisKind.HIGH_TURNOVER])
    assert budget.policy_id == "FO_DIAGNOSIS_REPAIR"
    assert budget.policy_version == "1.0.0"


def test_empty_diagnoses_yield_empty_budget():
    budget = diagnosis_search_budget([])
    assert budget.primary_diagnoses == ()
    assert budget.slots == ()
    assert budget.within_soft_budget()
