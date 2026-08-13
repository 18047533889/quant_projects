"""Tests for SearchRunner."""

import pytest
from datetime import datetime

from factor_optimizer.search.runner import SearchRunner, SearchConfig, SearchSession
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.trial import Trial, TrialStatus


@pytest.fixture
def basic_budget():
    return SearchBudget(max_trials=50, max_evaluations=30, max_cost_units=500.0)


@pytest.fixture
def search_config(basic_budget):
    return SearchConfig(
        budget=basic_budget,
        plateau_window=10,
        plateau_threshold=0.01,
    )


def test_search_config_validation():
    budget = SearchBudget(max_trials=10, max_evaluations=5)

    with pytest.raises(ValueError, match="plateau_window"):
        SearchConfig(budget=budget, plateau_window=0)

    with pytest.raises(ValueError, match="plateau_threshold"):
        SearchConfig(budget=budget, plateau_threshold=-0.1)

    with pytest.raises(ValueError, match="max_concurrency"):
        SearchConfig(budget=budget, max_concurrency=0)


def test_search_session_initialization(search_config):
    from factor_optimizer.contracts.search_budget import BudgetTracker

    session = SearchSession(
        session_id="test-session",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
    )

    assert session.session_id == "test-session"
    assert session.best_score is None
    assert session.best_trial_id is None
    assert len(session.trials) == 0
    assert not session.is_finished()


def test_search_session_add_trial(search_config):
    from factor_optimizer.contracts.search_budget import BudgetTracker

    session = SearchSession(
        session_id="test",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
    )

    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.PROPOSED)
    session.add_trial(trial)

    assert len(session.trials) == 1
    assert session.trials[0].trial_id == "t1"


def test_search_session_update_best(search_config):
    from factor_optimizer.contracts.search_budget import BudgetTracker

    session = SearchSession(
        session_id="test",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
    )

    # First update
    improved = session.update_best("t1", 0.5)
    assert improved
    assert session.best_score == 0.5
    assert session.best_trial_id == "t1"

    # Better score
    improved = session.update_best("t2", 0.7)
    assert improved
    assert session.best_score == 0.7
    assert session.best_trial_id == "t2"

    # Worse score
    improved = session.update_best("t3", 0.6)
    assert not improved
    assert session.best_score == 0.7
    assert session.best_trial_id == "t2"


def test_search_session_finish(search_config):
    from factor_optimizer.contracts.search_budget import BudgetTracker

    session = SearchSession(
        session_id="test",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
    )

    assert not session.is_finished()

    session.finish("test_reason")

    assert session.is_finished()
    assert session.stop_reason == "test_reason"
    assert session.finished_at is not None


def test_search_runner_basic():
    budget = SearchBudget(max_trials=5, max_evaluations=5, max_cost_units=100.0)
    config = SearchConfig(budget=budget, plateau_window=3)

    trial_counter = [0]

    def proposal_fn():
        trial_counter[0] += 1
        return Trial(
            trial_id=f"t{trial_counter[0]}",
            mutation_id=f"m{trial_counter[0]}",
            status=TrialStatus.PROPOSED,
        )

    def evaluation_fn(trial, fidelity):
        return {
            "evaluation_id": f"eval-{trial.trial_id}",
            "score": 0.5 + trial_counter[0] * 0.05,
            "cost": 10.0,
        }

    runner = SearchRunner(config, proposal_fn, evaluation_fn)
    session = runner.run("test-session")

    assert session.is_finished()
    assert session.stop_reason == "budget_exhausted"
    assert len(session.trials) == 5
    assert session.budget_tracker.trials_used == 5
    assert session.budget_tracker.evaluations_used == 5


def test_search_runner_plateau_detection():
    budget = SearchBudget(max_trials=100, max_evaluations=100, max_cost_units=1000.0)
    config = SearchConfig(
        budget=budget,
        plateau_window=5,
        plateau_threshold=0.001,
    )

    trial_counter = [0]

    def proposal_fn():
        trial_counter[0] += 1
        return Trial(
            trial_id=f"t{trial_counter[0]}",
            mutation_id=f"m{trial_counter[0]}",
            status=TrialStatus.PROPOSED,
        )

    def evaluation_fn(trial, fidelity):
        # Score plateaus after 10 trials
        if trial_counter[0] <= 10:
            score = 0.5 + trial_counter[0] * 0.1
        else:
            score = 1.5  # Flat

        return {
            "evaluation_id": f"eval-{trial.trial_id}",
            "score": score,
            "cost": 5.0,
        }

    runner = SearchRunner(config, proposal_fn, evaluation_fn)
    session = runner.run("plateau-test")

    assert session.is_finished()
    assert session.stop_reason == "plateau_detected"
    assert session.budget_tracker.trials_used < budget.max_trials


def test_search_runner_custom_plateau_detector():
    budget = SearchBudget(max_trials=50, max_evaluations=50, max_cost_units=500.0)
    config = SearchConfig(budget=budget)

    trial_counter = [0]

    def proposal_fn():
        trial_counter[0] += 1
        return Trial(
            trial_id=f"t{trial_counter[0]}",
            mutation_id=f"m{trial_counter[0]}",
            status=TrialStatus.PROPOSED,
        )

    def evaluation_fn(trial, fidelity):
        return {
            "evaluation_id": f"eval-{trial.trial_id}",
            "score": 0.8,
            "cost": 5.0,
        }

    # Custom detector that triggers after 10 scores
    def custom_plateau(scores):
        return len(scores) >= 10

    runner = SearchRunner(config, proposal_fn, evaluation_fn, custom_plateau)
    session = runner.run("custom-plateau")

    assert session.is_finished()
    assert session.stop_reason == "plateau_detected"
    assert session.budget_tracker.evaluations_used >= 10


def test_search_runner_evaluation_failure():
    budget = SearchBudget(max_trials=10, max_evaluations=10, max_cost_units=100.0)
    config = SearchConfig(budget=budget)

    trial_counter = [0]

    def proposal_fn():
        trial_counter[0] += 1
        return Trial(
            trial_id=f"t{trial_counter[0]}",
            mutation_id=f"m{trial_counter[0]}",
            status=TrialStatus.PROPOSED,
        )

    def evaluation_fn(trial, fidelity):
        if trial_counter[0] % 3 == 0:
            raise RuntimeError("Evaluation failed")
        return {
            "evaluation_id": f"eval-{trial.trial_id}",
            "score": 0.5,
            "cost": 10.0,
        }

    runner = SearchRunner(config, proposal_fn, evaluation_fn)
    session = runner.run("failure-test")

    assert session.is_finished()
    failed_trials = [t for t in session.trials if t.status == TrialStatus.FAILED]
    assert len(failed_trials) > 0


def test_search_runner_successful_trials():
    budget = SearchBudget(max_trials=5, max_evaluations=5, max_cost_units=100.0)
    config = SearchConfig(budget=budget)

    trial_counter = [0]

    def proposal_fn():
        trial_counter[0] += 1
        return Trial(
            trial_id=f"t{trial_counter[0]}",
            mutation_id=f"m{trial_counter[0]}",
            status=TrialStatus.PROPOSED,
        )

    def evaluation_fn(trial, fidelity):
        return {
            "evaluation_id": f"eval-{trial.trial_id}",
            "score": 0.5 + trial_counter[0] * 0.1,
            "cost": 10.0,
        }

    runner = SearchRunner(config, proposal_fn, evaluation_fn)
    session = runner.run("success-test")

    successful = session.successful_trials()
    assert len(successful) == 5
    assert all(t.status == TrialStatus.EVALUATED for t in successful)


def test_search_session_duration():
    budget = SearchBudget(max_trials=1, max_evaluations=1)
    config = SearchConfig(budget=budget)

    from factor_optimizer.contracts.search_budget import BudgetTracker

    session = SearchSession(
        session_id="test",
        config=config,
        budget_tracker=BudgetTracker(budget),
    )

    duration = session.duration_seconds()
    assert duration >= 0

    session.finish("done")
    duration_after = session.duration_seconds()
    assert duration_after >= duration
