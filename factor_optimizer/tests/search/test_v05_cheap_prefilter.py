import pytest
from types import SimpleNamespace

from factor_optimizer.contracts.evaluation_artifact import (
    EvaluationStatus, TrialEvaluationArtifact,
)
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.runner import SearchConfig, SearchRunner
from factor_optimizer.search.tiered_evaluation import (
    EvaluationTier, TieredEvaluationPolicy,
)
from tests.search.test_runner import _integrity_evidence


def _request(candidate_id, fidelity, selectable=True):
    request = SimpleNamespace(
        request_id=f"request:{candidate_id}:{fidelity}", content_hash=f"rh:{candidate_id}:{fidelity}",
        policy_id="policy", policy_content_hash="policy-hash",
        comparison_context_hash="context", candidate_set_hash=f"cs:{candidate_id}",
        required_final_fidelity=fidelity,
    )
    request.selectable = selectable
    return request


class _DecisionProvider:
    def decide(self, request):
        candidate_id = request.candidate_set_hash.removeprefix("cs:")
        selected = getattr(request, "selectable", True)
        return SimpleNamespace(
            request_id=request.request_id, request_hash=request.content_hash,
            policy_id=request.policy_id, policy_content_hash=request.policy_content_hash,
            comparison_context_hash=request.comparison_context_hash,
            candidate_set_hash=request.candidate_set_hash,
            final_fidelity=request.required_final_fidelity,
            decision_id=f"decision:{request.request_id}", content_hash=f"dh:{request.request_id}",
            status="SELECTED" if selected else "REJECTED",
            eligibility={candidate_id: selected},
            point_utility={candidate_id: .5 if selected else None},
            conservative_utility={candidate_id: .5 if selected else None},
            winner_id=candidate_id if selected else None,
        )


def _runner(evaluate, trial_id="candidate"):
    policy = TieredEvaluationPolicy((
        EvaluationTier("CHEAP_DECLARED_EVIDENCE", 1.0, 0),
        EvaluationTier("FULL_VALIDATION", 10.0, 4),
    ))
    config = SearchConfig(
        budget=SearchBudget(max_trials=1, max_evaluations=2, max_cost_units=2),
        tiered_evaluation=policy,
        evaluation_cost_units=1,
        screening_only=False,
    )
    protocol = EvaluationProtocol(
        SplitPlan("train-only-v05", [True], [False], [False], {}), evaluate,
    )
    provider = _DecisionProvider()
    return SearchRunner(
        config,
        lambda: Trial(trial_id=trial_id, mutation_id="bounded-v05",
                      status=TrialStatus.PROPOSED),
        protocol,
        decision_provider=provider,
        decision_request_factory=lambda session: _request(
            trial_id, "FULL_VALIDATION", bool(session.successful_trials())
        ),
        stage_decision_request_factory=lambda session, trial, artifact, job: _request(
            trial.trial_id, job.tier_name
        ),
    )


def _artifact(trial, fidelity, *, score, status, evidence_ref):
    return TrialEvaluationArtifact(
        trial_id=trial.trial_id,
        objective_values=[{"name": "score", "value": score}],
        objective_spec_ref="score", split_ref="train-only-v05",
        fidelity=fidelity, evidence_ref=evidence_ref, compute_cost=1,
        status=status,
        treatment_integrity_evidence=_integrity_evidence(trial.trial_id),
    )


@pytest.mark.parametrize("declared_branch", [
    "stable-q10-u-shape", "predeclared-sparse-event-coverage",
    "measured-model-complementarity",
])
def test_low_raw_rank_ic_does_not_override_declared_survivor_branch(declared_branch):
    calls = []

    def evaluate(trial, fidelity):
        calls.append(fidelity)
        # The primary value is deliberately weak. The EvaluationProtocol has
        # nevertheless classified the candidate COMPLETE from its persisted,
        # branch-specific evidence; SearchRunner must not invent a RankIC cut.
        return _artifact(
            trial, fidelity, score=.0001, status=EvaluationStatus.COMPLETE,
            evidence_ref=f"qe:{declared_branch}:f{fidelity}",
        )

    session = _runner(evaluate, declared_branch).run("v05-survivor")
    assert calls == [0, 4]
    assert session.trials[0].status is TrialStatus.EVALUATED
    assert session.best_trial_id == declared_branch


def test_declared_no_signal_prune_is_terminal_and_budget_bounded():
    calls = []

    def evaluate(trial, fidelity):
        calls.append(fidelity)
        return _artifact(
            trial, fidelity, score=0.0, status=EvaluationStatus.PRUNED,
            evidence_ref="qe:cheap:no-signal-after-declared-branches",
        )

    session = _runner(evaluate, "deterministic-noise").run("v05-noise")
    assert calls == [0]
    assert session.trials[0].status is TrialStatus.PRUNED
    assert session.best_trial_id is None
    assert session.budget_tracker.evaluations_used == 1
    assert session.budget_tracker.cost_used == 1
    assert session.stage_execution_trace[0]["evaluation_ref"] == \
        "qe:cheap:no-signal-after-declared-branches"


def test_v6_real_runner_records_issued_jobs_and_bound_outcomes():
    def evaluate(trial, fidelity):
        return _artifact(
            trial, fidelity, score=.1, status=EvaluationStatus.COMPLETE,
            evidence_ref=f"qe:issued:f{fidelity}",
        )

    runner = _runner(evaluate, "issued-candidate")
    session = runner.run("v6-issued-runner")
    state = runner._tiered_scheduler.to_dict()
    assert len(state["issued"]) == 2
    assert len(state["outcomes"]) == 2
    assert state["completed"] == ["issued-candidate"]
    for payload, _result in state["outcomes"].values():
        assert payload["candidate_id"] == "issued-candidate"
        assert payload["recipe_version"] == "bounded-v05"
        assert payload["evaluation_ref"].startswith("qe:issued:")
    assert session.trials[0].status is TrialStatus.EVALUATED


def test_complete_high_score_without_decision_provider_cannot_promote():
    calls = []
    policy = TieredEvaluationPolicy((
        EvaluationTier("CHEAP_DECLARED_EVIDENCE", 1.0, 0),
        EvaluationTier("FULL_VALIDATION", 10.0, 4),
    ))
    config = SearchConfig(
        budget=SearchBudget(max_trials=1, max_evaluations=2, max_cost_units=2),
        tiered_evaluation=policy, evaluation_cost_units=1,
        screening_only=True,
    )
    def evaluate(trial, fidelity):
        calls.append(fidelity)
        return _artifact(trial, fidelity, score=999.0,
                         status=EvaluationStatus.COMPLETE,
                         evidence_ref="qe:malicious-complete")
    protocol = EvaluationProtocol(
        SplitPlan("train-only-v05", [True], [False], [False], {}), evaluate,
    )
    session = SearchRunner(
        config, lambda: Trial("malicious", "m", status=TrialStatus.PROPOSED),
        protocol,
    ).run("screening-only")
    assert calls == [0]
    assert session.best_trial_id is None
