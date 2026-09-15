import pytest

from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.search.runner import SearchConfig, SearchRunner, SearchSession


@pytest.mark.parametrize("mutate_in_proposal", [False, True])
def test_live_config_drift_stops_before_trial_processing(mutate_in_proposal):
    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=2.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False)
    protocol = EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}),
                                  lambda trial, fidelity: None)
    calls = []
    def propose():
        calls.append(True)
        config.plateau_threshold = 0.5
        return None
    runner = SearchRunner(config, propose, protocol)
    session = SearchSession("live-guard", config, BudgetTracker(budget))
    if not mutate_in_proposal:
        config.plateau_threshold = 0.5
    with pytest.raises(ValueError, match="configuration changed"):
        runner._run_session(session)
    assert len(calls) == int(mutate_in_proposal)


@pytest.mark.parametrize("verdict", [True, False, "raise"])
def test_validator_drift_stops_before_verdict_and_evaluation(verdict):
    from factor_optimizer.contracts.trial import Trial, TrialStatus

    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=2.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False)
    protocol = EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}),
                                  lambda trial, fidelity: None)
    trial = Trial(trial_id="t", mutation_id="m", status=TrialStatus.PROPOSED)
    def validate(candidate):
        config.plateau_threshold = 0.5
        if verdict == "raise":
            raise RuntimeError("validator failed after mutation")
        return {"is_legal": verdict}
    runner = SearchRunner(config, lambda: trial, protocol, trial_validator=validate)
    session = SearchSession("validator-guard", config, BudgetTracker(budget))
    with pytest.raises(ValueError, match="configuration changed"):
        runner._run_session(session)
    assert trial.status is TrialStatus.VALIDATING
    assert session.budget_tracker.evaluations_used == 0


def test_evaluator_drift_rejects_output_before_normalization(monkeypatch):
    import factor_optimizer.search.runner as runner_module
    from factor_optimizer.contracts.trial import Trial, TrialStatus

    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=2.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False)
    def evaluate(trial, fidelity):
        config.plateau_threshold = 0.5
        return {}
    protocol = EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}), evaluate)
    trial = Trial(trial_id="eval-drift", mutation_id="m", status=TrialStatus.PROPOSED)
    runner = SearchRunner(config, lambda: trial, protocol)
    session = SearchSession("eval-guard", config, BudgetTracker(budget))
    runner._bind_capabilities(session.session_id)
    normalizations = []
    original = runner_module.normalize_evaluation_result
    def normalize(*args, **kwargs):
        normalizations.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner_module, "normalize_evaluation_result", normalize)
    with pytest.raises(ValueError, match="configuration changed"):
        runner._run_session(session)
    assert normalizations == []
    assert trial.status is TrialStatus.FAILED
    assert session.budget_tracker.evaluations_used == 1
    assert not session.budget_tracker.has_reservation(trial.trial_id)


@pytest.mark.parametrize("boundary", ["entry", "request", "provider"])
def test_final_decision_drift_cannot_seal_session(boundary):
    from tests.search.test_v05_cheap_prefilter import _request, _DecisionProvider

    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=2.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False)
    calls = []
    def request_factory(session):
        calls.append("request")
        if boundary == "request":
            config.plateau_threshold = 0.5
        return _request("candidate", "FULL_VALIDATION", False)
    class Provider(_DecisionProvider):
        def decide(self, request):
            calls.append("provider")
            if boundary == "provider":
                config.plateau_threshold = 0.5
            return super().decide(request)
    protocol = EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}),
                                  lambda trial, fidelity: None)
    runner = SearchRunner(config, lambda: None, protocol,
                          decision_provider=Provider(), decision_request_factory=request_factory)
    session = SearchSession("finish-guard", config, BudgetTracker(budget))
    if boundary == "entry":
        config.plateau_threshold = 0.5
    with pytest.raises(ValueError, match="configuration changed"):
        runner._finish(session, "budget_exhausted")
    assert not session.is_finished()
    assert session.selection_decision_id is None
    assert calls == {"entry": [], "request": ["request"],
                     "provider": ["request", "provider"]}[boundary]


@pytest.mark.parametrize("boundary", ["request", "provider"])
def test_stage_decision_drift_cannot_advance_candidate(boundary, monkeypatch):
    from tests.search.test_v05_cheap_prefilter import _runner, _artifact
    from factor_optimizer.contracts.evaluation_artifact import EvaluationStatus

    evaluations, advances, decisions = [], [], []
    def evaluate(trial, fidelity):
        evaluations.append(fidelity)
        return _artifact(trial, fidelity, score=0.5,
                         status=EvaluationStatus.COMPLETE, evidence_ref="qe:stage")
    runner = _runner(evaluate)
    original_request = runner.stage_decision_request_factory
    def request(*args):
        if boundary == "request":
            runner.config.plateau_threshold = 0.5
        return original_request(*args)
    runner.stage_decision_request_factory = request
    original_decide = runner.decision_provider.decide
    def decide(value):
        decisions.append(True)
        if boundary == "provider":
            runner.config.plateau_threshold = 0.5
        return original_decide(value)
    monkeypatch.setattr(runner.decision_provider, "decide", decide)
    original_advance = runner._tiered_scheduler.advance
    def advance(value):
        advances.append(True)
        return original_advance(value)
    monkeypatch.setattr(runner._tiered_scheduler, "advance", advance)
    with pytest.raises(ValueError, match="configuration changed"):
        runner.run("stage-config-guard")
    assert advances == []
    assert evaluations == [0]
    assert len(decisions) == int(boundary == "provider")


@pytest.mark.parametrize("stop", [False, True])
def test_plateau_drift_stops_before_proposal(stop):
    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=2.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False, plateau_window=1)
    protocol = EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}),
                                  lambda trial, fidelity: None)
    proposals = []
    def detect(scores):
        config.plateau_threshold = 0.5
        return stop
    def propose():
        proposals.append(True)
        return None
    runner = SearchRunner(config, propose, protocol, plateau_detector=detect)
    session = SearchSession("plateau-guard", config, BudgetTracker(budget))
    session.recent_scores.append(1.0)
    with pytest.raises(ValueError, match="configuration changed"):
        runner._run_session(session)
    assert proposals == []
    assert not session.is_finished()
