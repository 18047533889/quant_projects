"""Adversarial regression tests for search correctness boundaries."""

from datetime import datetime

import pytest

from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import (
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


def test_recovery_to_old_best_is_plateau_not_progress():
    detector = PlateauDetector(
        PlateauConfig(window_size=3, min_relative_improvement=0.001)
    )
    assert detector.is_plateau([1.0, 0.5, 1.0])


@pytest.mark.parametrize(
    "operation",
    [
        lambda: SplitPlan("split", [1], [2], [3], {}),
        lambda: SealedTestHandle("session", "trial", datetime.now()),
        lambda: SealedTestResult("trial", {"rank_ic": 1.0}, datetime.now(), "session"),
        lambda: validate_split_plan(object()),
        lambda: create_split_aware_evaluation_fn(object(), object()),
    ],
)
def test_split_and_sealed_test_stubs_fail_closed(operation):
    with pytest.raises(NotImplementedError):
        operation()
