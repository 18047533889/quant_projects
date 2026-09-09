"""Tests for SearchRunner."""

import pytest
from datetime import datetime

from factor_optimizer.search.runner import SearchRunner, SearchConfig, SearchSession
from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
import numpy as np


def _protocol(fn):
    """Wrap a generic evaluator callback in the required safe boundary."""
    return EvaluationProtocol(
        SplitPlan("search", [True], [False], [False], {}), fn
    )


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

    with pytest.raises(ValueError, match="serialized"):
        SearchConfig(budget=budget, max_concurrency=2)


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


def test_search_runner_rejects_duplicate_trial_ids_before_evaluation():
    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=10.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False)
    proposals = iter([
        Trial(trial_id="same", mutation_id="m1", status=TrialStatus.PROPOSED),
        Trial(trial_id="same", mutation_id="m2", status=TrialStatus.PROPOSED),
    ])
    evaluated = []

    def evaluate(trial, fidelity):
        evaluated.append(trial.trial_id)
        return {"evaluation_id": "eval-same", "score": 1.0, "cost": 1.0, "treatment_integrity_evidence": _integrity_evidence(trial.trial_id)}

    session = SearchRunner(config, lambda: next(proposals), _protocol(evaluate)).run("duplicates")

    assert evaluated == ["same"]
    assert [trial.trial_id for trial in session.trials] == ["same"]
    assert len(session.duplicate_trials) == 1
    assert session.duplicate_trials[0].status is TrialStatus.DUPLICATE
    assert session.successful_trials()[0].trial_id == "same"


def test_search_runner_does_not_mutate_aliased_duplicate_trial():
    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=10.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False)
    trial = Trial(trial_id="same", mutation_id="m1", status=TrialStatus.PROPOSED)
    evaluated = []

    def evaluate(candidate, fidelity):
        evaluated.append(candidate.trial_id)
        return {"evaluation_id": "eval-same", "score": 1.0, "cost": 1.0, "treatment_integrity_evidence": _integrity_evidence(trial.trial_id)}

    session = SearchRunner(config, lambda: trial, _protocol(evaluate)).run("aliased-duplicate")

    assert evaluated == ["same"]
    assert trial.status is TrialStatus.EVALUATED
    assert session.trials == [trial]
    assert session.successful_trials() == [trial]
    assert session.duplicate_trials[0] is not trial
    assert session.duplicate_trials[0].status is TrialStatus.DUPLICATE
    assert session.budget_tracker.trials_used == 2
    assert session.budget_tracker.evaluations_used == 1


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
            "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            }

    runner = SearchRunner(config, proposal_fn, _protocol(evaluation_fn))
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
            "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            }

    runner = SearchRunner(config, proposal_fn, _protocol(evaluation_fn))
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
            "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            }

    # Custom detector that triggers after 10 scores
    def custom_plateau(scores):
        return len(scores) >= 10

    runner = SearchRunner(config, proposal_fn, _protocol(evaluation_fn), custom_plateau)
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
            "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            }

    runner = SearchRunner(config, proposal_fn, _protocol(evaluation_fn))
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
            "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            }

    runner = SearchRunner(config, proposal_fn, _protocol(evaluation_fn))
    session = runner.run("success-test")

    successful = session.successful_trials()
    assert len(successful) == 5
    assert all(t.status == TrialStatus.EVALUATED for t in successful)


def test_search_runner_cost_greater_than_remaining_does_not_execute():
    budget = SearchBudget(max_trials=5, max_evaluations=5, max_cost_units=10.0)
    config = SearchConfig(budget=budget, evaluation_cost_units=6.0)
    calls = []

    def proposal_fn():
        return Trial(trial_id=f"t{len(calls) + 1}", mutation_id="m", status=TrialStatus.PROPOSED)

    def evaluation_fn(trial, fidelity):
        calls.append(trial.trial_id)
        return {"evaluation_id": trial.trial_id, "score": 1.0, "cost": 6.0, "treatment_integrity_evidence": _integrity_evidence(trial.trial_id)}

    runner = SearchRunner(config, proposal_fn, _protocol(evaluation_fn))
    session = runner.run("cost-boundary")

    assert calls == ["t1"]
    assert session.stop_reason == "evaluation_budget_unavailable"
    assert session.budget_tracker.cost_used == 6.0
    assert session.budget_tracker.evaluations_used == 1
    assert session.trials[-1].status == TrialStatus.FAILED


def test_search_runner_started_failure_charges_reserved_envelope():
    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=10.0)
    config = SearchConfig(budget=budget, evaluation_cost_units=10.0)
    calls = 0

    def proposal_fn():
        return Trial(trial_id=f"t{len(session_trials) + 1}", mutation_id="m", status=TrialStatus.PROPOSED)

    session_trials = []

    def evaluation_fn(trial, fidelity):
        nonlocal calls
        session_trials.append(trial)
        calls += 1
        if calls == 1:
            raise RuntimeError("failed")
        return {"evaluation_id": trial.trial_id, "score": 1.0, "cost": 10.0, "treatment_integrity_evidence": _integrity_evidence(trial.trial_id)}

    session = SearchRunner(config, proposal_fn, _protocol(evaluation_fn)).run("refund")
    # A69: the callback started and consumed resources before failing.  With
    # no worker meter, conservatively charge its reserved envelope; do not
    # refund the expensive failure and dispatch another job on false budget.
    assert calls == 1
    assert session.budget_tracker.cost_used == 10.0
    assert session.budget_tracker.cost_reserved == 0.0
    assert session.budget_tracker.evaluations_used == 1


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


def test_search_session_checkpoint_round_trip(search_config):
    from factor_optimizer.contracts.search_budget import BudgetTracker

    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.EVALUATED)
    trial.update_status(TrialStatus.EVALUATED, evaluation_ref="eval-1", metadata={"score": 0.8})
    session = SearchSession(
        session_id="checkpoint",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
        trials=[trial],
        best_score=0.8,
        best_trial_id="t1",
        recent_scores=[0.8],
    )
    session.budget_tracker.record_trial()
    session.budget_tracker.record_evaluation(2.0)

    restored = SearchSession.from_dict(session.to_dict())

    assert restored.session_id == session.session_id
    assert restored.config.to_dict() == session.config.to_dict()
    assert restored.budget_tracker.to_dict() == session.budget_tracker.to_dict()
    assert restored.trials[0].to_dict() == trial.to_dict()
    assert restored.best_score == 0.8
    assert restored.best_trial_id == "t1"
    assert restored.recent_scores == [0.8]


def test_search_session_rejects_nonfinite_best_score(search_config):
    session = SearchSession(
        session_id="nonfinite-best",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
        best_score=0.8,
    )
    checkpoint = session.to_dict()
    checkpoint["best_score"] = float("nan")

    with pytest.raises(ValueError, match="best_score"):
        SearchSession.from_dict(checkpoint)


def test_search_session_rejects_inconsistent_best_state(search_config):
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.EVALUATED)
    trial.update_status(TrialStatus.EVALUATED, evaluation_ref="e1", metadata={"score": 0.8})
    session = SearchSession(
        session_id="inconsistent-best",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
        trials=[trial],
        best_score=0.8,
        best_trial_id="t1",
    )
    checkpoint = session.to_dict()

    invalid_checkpoints = (
        {**checkpoint, "best_trial_id": None},
        {**checkpoint, "best_trial_id": "missing"},
        {**checkpoint, "best_score": 0.7},
        {**checkpoint, "trials": [dict(trial.to_dict(), status=TrialStatus.FAILED.value)]},
    )
    for invalid in invalid_checkpoints:
        with pytest.raises(ValueError, match="best"):
            SearchSession.from_dict(invalid)


def test_search_session_rejects_inconsistent_recent_score_history(search_config):
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.EVALUATED)
    trial.update_status(TrialStatus.EVALUATED, evaluation_ref="e1", metadata={"score": 0.8})
    session = SearchSession(
        session_id="inconsistent-history",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
        trials=[trial],
        best_score=0.8,
        best_trial_id="t1",
        recent_scores=[0.8],
    )
    checkpoint = session.to_dict()

    invalid_checkpoints = (
        {**checkpoint, "recent_scores": [0.7]},
        {**checkpoint, "recent_scores": list(range(search_config.plateau_window + 1))},
        {
            **checkpoint,
            "trials": [],
            "best_score": None,
            "best_trial_id": None,
            "recent_scores": [0.8],
        },
    )
    for invalid in invalid_checkpoints:
        with pytest.raises(ValueError, match="recent_scores"):
            SearchSession.from_dict(invalid)


def test_search_runner_resume_continues_from_checkpoint():
    budget = SearchBudget(max_trials=3, max_evaluations=3, max_cost_units=30.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False)
    proposals = iter([
        Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.PROPOSED),
        Trial(trial_id="t2", mutation_id="m2", status=TrialStatus.PROPOSED),
    ])
    runner = SearchRunner(
        config,
        lambda: next(proposals),
        _protocol(lambda trial, fidelity: {
            "evaluation_id": trial.trial_id,
            "score": 0.5,
            "cost": 10.0,
            "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
        }),
    )
    session = SearchSession(
        session_id="resume",
        config=config,
        budget_tracker=BudgetTracker(config.budget),
        recent_scores=[0.1],
    )
    session.budget_tracker.record_trial()
    session.budget_tracker.record_evaluation(10.0)
    first = Trial(trial_id="t0", mutation_id="m0", status=TrialStatus.EVALUATED)
    first.update_status(TrialStatus.EVALUATED, evaluation_ref="e0", metadata={"score": 0.1})
    session.add_trial(first)

    restored = SearchSession.from_dict(session.to_dict())
    result = runner.resume(restored)

    assert result.stop_reason == "budget_exhausted"
    assert result.budget_tracker.trials_used == 3
    assert result.budget_tracker.evaluations_used == 3
    assert result.recent_scores == [0.1, 0.5, 0.5]


def test_search_session_rejects_invalid_budget_counters(search_config):
    tracker = BudgetTracker(
        search_config.budget,
        trials_used=-1,
        evaluations_used=-1,
        cost_used=-1.0,
    )
    session = SearchSession(
        session_id="invalid-counters",
        config=search_config,
        budget_tracker=tracker,
    )

    with pytest.raises(ValueError, match="budget counters"):
        SearchSession.from_dict(session.to_dict())

    tracker = BudgetTracker(search_config.budget)
    tracker.reserve_evaluation(1.0)
    session = SearchSession(
        session_id="active",
        config=search_config,
        budget_tracker=tracker,
    )
    with pytest.raises(ValueError, match="active evaluation reservations"):
        SearchSession.from_dict(session.to_dict())

    tracker = BudgetTracker(search_config.budget)
    session = SearchSession(
        session_id="inflight",
        config=search_config,
        budget_tracker=tracker,
        trials=[Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.EVALUATING)],
    )
    with pytest.raises(ValueError, match="non-terminal trials"):
        SearchSession.from_dict(session.to_dict())


def test_search_runner_resume_rejects_finished_or_mismatched_session(search_config):
    from factor_optimizer.contracts.search_budget import BudgetTracker

    session = SearchSession(
        session_id="finished",
        config=search_config,
        budget_tracker=BudgetTracker(search_config.budget),
    )
    session.finish("done")
    runner = SearchRunner(search_config, lambda: None, _protocol(lambda trial, fidelity: {}))
    with pytest.raises(ValueError, match="finished session"):
        runner.resume(session)

    other = SearchConfig(budget=SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=2.0))
    open_session = SearchSession(
        session_id="mismatch",
        config=other,
        budget_tracker=BudgetTracker(other.budget),
    )
    with pytest.raises(ValueError, match="does not match"):
        runner.resume(open_session)


def _integrity_evidence(trial_id="t1", kind=None):
    """Passing TreatmentIntegrityEvidence measured from arrays (R55 P0-9)."""
    from factor_optimizer.contracts.treatment_integrity import (
        build_integrity_evidence,
    )

    rng = np.random.default_rng(abs(hash(trial_id)) % (2 ** 32))
    before = rng.normal(size=32)
    treated_kind = kind if kind else f"treatment::{trial_id}"
    if treated_kind == "raw":
        after = before
    else:
        after = before * 0.5 + 0.01
    return build_integrity_evidence(
        trial_id,
        treated_kind,
        {} if treated_kind == "raw" else {"window": 3},
        before,
        after,
    )
