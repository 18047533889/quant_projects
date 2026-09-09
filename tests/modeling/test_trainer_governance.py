# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from modeling.contracts import DecisionClock
from modeling.hyperparams import validate_search_grid
from modeling.selection import ConsumerNonInferiorityContract, candidate_identity, select_best_validation
from modeling.trainer import _assert_declared_feature_availability
from modeling.trainer_governance import (
    CandidateExposureLedger,
    ComparisonRecord,
    ComparisonStatus,
    GovernanceContract,
    apply_versioned_transform,
    common_oos_cohort,
    enforce_coverage,
    incremental_value,
    validate_pit_categorical_vocabulary,
)


def test_approved_search_space_is_a_hard_boundary():
    assert validate_search_grid("pcr", [{"n_components": 2}]) == [{"n_components": 2}]
    with pytest.raises(ValueError, match="outside approved"):
        validate_search_grid("pcr", [{"n_components": 4}])
    with pytest.raises(ValueError, match="outside approved"):
        validate_search_grid("pcr", [{"unknown": 2}])
    with pytest.raises(ValueError, match="no approved"):
        validate_search_grid("unknown", [{"x": 1}])


def test_candidate_identity_and_tie_break_are_deterministic():
    a = {"alpha": 0.1, "l1_ratio": 0.5}
    b = {"l1_ratio": 0.5, "alpha": 0.1}
    assert candidate_identity(a) == candidate_identity(b)
    scores = [
        {"hyperparams": {"n_components": 3}, "rank_ic": 0.1},
        {"hyperparams": {"n_components": 2}, "rank_ic": 0.1},
    ]
    best1, diag1 = select_best_validation(scores)
    best2, diag2 = select_best_validation(list(reversed(scores)))
    assert best1 == best2 == {"n_components": 2}
    assert diag1["best_candidate_id"] == diag2["best_candidate_id"]


def test_exposure_budget_and_ledger_are_hard_and_stable():
    ledger = CandidateExposureLedger(2)
    ledger.expose("a")
    ledger.expose("b")
    assert ledger.identities == ("a", "b")
    assert ledger.digest() == CandidateExposureLedger(2, ["a", "b"]).digest()
    with pytest.raises(ValueError, match="budget"):
        ledger.expose("c")


def test_selection_applies_consumer_objective_noninferiority_and_risk_gate():
    contract = ConsumerNonInferiorityContract(
        "strict-portfolio", "net_utility", epsilon=.01,
        min_risk_improvement=.02, max_risk_budget=.10,
    )
    common = dict(consumer_profile="strict-portfolio", baseline_score=0.0,
                  delta_ci_high=.02, effect_size=.01)
    scores = [
        {"hyperparams": {"n_components": 2}, "net_utility": .20,
         **common, "delta_ci_low": -.02, "risk_improvement": .05, "risk_budget": .05},
        {"hyperparams": {"n_components": 3}, "net_utility": .10,
         **common, "delta_ci_low": -.005, "risk_improvement": .03, "risk_budget": .05},
        {"hyperparams": {"n_components": 5}, "net_utility": .30,
         **common, "delta_ci_low": 0.0, "risk_improvement": .03, "risk_budget": .20},
    ]
    best, diagnostics = select_best_validation(
        scores, objective="net_utility", noninferiority=contract
    )
    assert best == {"n_components": 3}
    assert diagnostics["noninferiority"][candidate_identity({"n_components": 2})]["passed"] is False
    assert diagnostics["noninferiority"][candidate_identity({"n_components": 5})]["passed"] is False


def test_comparison_status_keeps_not_computable_separate_from_code_error():
    missing = ComparisonRecord(ComparisonStatus.NOT_COMPUTABLE, reason="one-date cohort")
    baseline = ComparisonRecord(ComparisonStatus.COMPUTED, 0.2)
    assert incremental_value(missing, baseline).status == ComparisonStatus.NOT_COMPUTABLE
    broken = ComparisonRecord(ComparisonStatus.CODE_ERROR, reason="bug")
    with pytest.raises(RuntimeError, match="bug"):
        incremental_value(broken, baseline)
    delta = incremental_value(ComparisonRecord(ComparisonStatus.COMPUTED, 0.3), baseline)
    assert delta.value == pytest.approx(0.1)


def test_feature_availability_gate_is_explicit_and_fail_closed():
    clock = DecisionClock(feature_available_at={"known": "t_close"})
    _assert_declared_feature_availability(clock, ["known"])
    with pytest.raises(ValueError, match="undeclared"):
        _assert_declared_feature_availability(clock, ["known", "unknown"])


def test_coverage_common_cohort_transforms_and_pit_vocabulary():
    assert enforce_coverage(observed=9, expected=10, minimum=0.9) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="below hard minimum"):
        enforce_coverage(observed=8, expected=10, minimum=0.9)
    assert common_oos_cohort([[3, 2, 1], [2, 3, 4]]) == (2, 3)
    with pytest.raises(ValueError, match="empty intersection"):
        common_oos_cohort([[1], [2]])
    assert np.allclose(apply_versioned_transform(np.array([1.0]), "negate-v1"), [-1.0])
    with pytest.raises(ValueError, match="unknown transform"):
        GovernanceContract(version="v1", objective_transform_version="latest")
    assert validate_pit_categorical_vocabulary(
        ["A", "B"], approved=["A", "B", "C"], version="industry-v1"
    ) == ("A", "B")
    with pytest.raises(ValueError, match="unknown PIT categorical"):
        validate_pit_categorical_vocabulary(
            ["A", "NEW"], approved=["A"], version="industry-v1"
        )
