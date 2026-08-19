"""Adversarial regression tests for search correctness boundaries."""

from datetime import datetime

import numpy as np
import pytest

from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import (
    EvaluationProtocol,
    SealedTestHandle,
    SealedTestResult,
    SplitPlan,
    create_split_aware_evaluation_fn,
    validate_split_plan,
)
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.plateau import PlateauConfig, PlateauDetector
from factor_optimizer.search.runner import SearchConfig, SearchRunner


def _trial(trial_id="trial", mutation_id="mutation"):
    return Trial(trial_id=trial_id, mutation_id=mutation_id, status=TrialStatus.PROPOSED)


def test_malformed_trial_is_rejected_before_evaluation():
    evaluated = []
    runner = SearchRunner(
        SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
            enable_multifidelity=False,
        ),
        lambda: _trial(trial_id=""),
        lambda trial, fidelity: evaluated.append(trial) or {"score": 1.0, "cost": 1.0},
    )

    session = runner.run("malformed")

    assert evaluated == []
    assert session.trials[0].status is TrialStatus.ILLEGAL
    assert session.trials[0].legality_check["is_legal"] is False
    assert "trial_id is required" in session.trials[0].legality_check["errors"]


def test_malformed_validator_response_fails_closed():
    runner = SearchRunner(
        SearchConfig(budget=SearchBudget(max_trials=1, max_evaluations=1)),
        _trial,
        lambda trial, fidelity: pytest.fail("illegal trial reached evaluation"),
        trial_validator=lambda trial: {"checks_passed": ["claimed"]},
    )

    session = runner.run("bad-validator")

    assert session.trials[0].status is TrialStatus.ILLEGAL
    assert "boolean is_legal" in session.trials[0].legality_check["errors"][0]


def test_search_runner_persists_published_evidence_reference():
    def evaluate(trial, fidelity):
        return {
            "evaluation_id": "eval-1",
            "evidence_ref": "store://evidence-1",
            "score": 1.0,
            "cost": 1.0,
        }

    session = SearchRunner(
        SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
            enable_multifidelity=False,
            evaluation_cost_units=1.0,
        ),
        _trial,
        evaluate,
    ).run("evidence-reference")

    assert session.trials[0].evaluation_ref == "store://evidence-1"


def test_search_runner_rejects_evaluation_without_evidence_reference():
    session = SearchRunner(
        SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
            enable_multifidelity=False,
            evaluation_cost_units=1.0,
        ),
        _trial,
        lambda trial, fidelity: {"score": 1.0, "cost": 1.0},
    ).run("missing-evidence-reference")

    trial = session.trials[0]
    assert trial.status is TrialStatus.FAILED
    assert trial.evaluation_ref is None
    assert "non-empty evidence_ref or evaluation_id" in trial.failure_reason
    assert session.successful_trials() == []
    assert session.best_trial_id is None


def test_multifidelity_does_not_promote_on_score_alone():
    fidelities = []

    def evaluate(trial, fidelity):
        fidelities.append(fidelity)
        return {"evaluation_id": "eval", "score": 999.0, "cost": 1.0}

    session = SearchRunner(
        SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=5, max_cost_units=10.0),
            enable_multifidelity=True,
            evaluation_cost_units=1.0,
        ),
        _trial,
        evaluate,
    ).run("no-false-promotion")

    assert fidelities == [0]
    assert session.trials[0].metadata["fidelity"] == 0


def test_multifidelity_promotes_only_with_explicit_eligible_evidence():
    fidelities = []

    def evaluate(trial, fidelity):
        fidelities.append(fidelity)
        return {
            "evaluation_id": f"eval-{fidelity}",
            "score": 1.0,
            "cost": 1.0,
            "promote": fidelity == 0,
            "rank": 0,
            "total": 2,
        }

    session = SearchRunner(
        SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=5, max_cost_units=10.0),
            enable_multifidelity=True,
            evaluation_cost_units=1.0,
        ),
        _trial,
        evaluate,
    ).run("explicit-promotion")

    assert fidelities == [0, 1]
    assert session.trials[0].metadata["fidelity"] == 1


def test_safe_search_rejects_unrestricted_callback():
    with pytest.raises(TypeError, match="EvaluationProtocol"):
        SearchRunner(
            SearchConfig(
                budget=SearchBudget(max_trials=1, max_evaluations=1),
                require_evaluation_protocol=True,
            ),
            _trial,
            lambda trial, fidelity: {"score": 1.0, "cost": 1.0},
        )


def test_safe_search_accepts_validated_protocol():
    protocol = EvaluationProtocol(
        SplitPlan("split", [True], [False], [False], {}),
        lambda trial, fidelity: {
            "evaluation_id": "eval-safe", "score": 1.0, "cost": 1.0
        },
    )
    session = SearchRunner(
        SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
            enable_multifidelity=False,
            require_evaluation_protocol=True,
        ),
        _trial,
        protocol,
    ).run("safe")
    assert session.successful_trials()[0].evaluation_ref == "eval-safe"


def test_recovery_to_old_best_is_plateau_not_progress():
    detector = PlateauDetector(
        PlateauConfig(window_size=3, min_relative_improvement=0.001)
    )
    assert detector.is_plateau([1.0, 0.5, 1.0])


def test_split_plan_validates_boundaries_and_is_usable():
    plan = SplitPlan("split", [True, False], [False, True], [False, False], {})
    assert validate_split_plan(plan)["validated"] is True


def test_split_plan_rejects_integer_masks():
    with pytest.raises(ValueError, match="boolean boundaries"):
        SplitPlan("split", [1, 0], [0, 1], [0, 0], {})


def test_split_plan_rejects_non_builtin_bool_masks():
    with pytest.raises(ValueError, match="boolean boundaries"):
        SplitPlan("split", [np.bool_(True)], [False], [False], {})


@pytest.mark.parametrize(
    "factory, error",
    [
        (lambda: SplitPlan("", [True], [False], [False], {}), ValueError),
        (lambda: SplitPlan("split", [], [False], [False], {}), ValueError),
        (lambda: SplitPlan("split", [True], [False, True], [False], {}), ValueError),
        (lambda: SplitPlan("split", [True], [True], [False], {}), ValueError),
        (lambda: validate_split_plan(object()), TypeError),
        (lambda: create_split_aware_evaluation_fn(object(), object()), TypeError),
    ],
)
def test_invalid_split_contracts_fail_closed(factory, error):
    with pytest.raises(error):
        factory()


def test_sealed_test_stubs_still_fail_closed():
    with pytest.raises(NotImplementedError):
        SealedTestHandle("session", "trial", datetime.now())
    with pytest.raises(NotImplementedError):
        SealedTestResult("trial", {"rank_ic": 1.0}, datetime.now(), "session")
