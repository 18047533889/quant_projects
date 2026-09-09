from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.runner import SearchConfig, SearchRunner, SearchSession
from tests.search.test_runner import _integrity_evidence


def test_real_runner_records_executed_stage_profile_and_survives_checkpoint(tmp_path):
    budget = SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=2)
    runner = SearchRunner(
        SearchConfig(budget=budget, enable_multifidelity=False),
        lambda: Trial(trial_id="trace", mutation_id="m", status=TrialStatus.PROPOSED),
        EvaluationProtocol(
            SplitPlan("validation-sample", [True], [False], [False], {}),
            lambda trial, fidelity: {
                "evaluation_id": "qe:trace", "score": 0.2, "cost": 1,
                "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            },
        ),
    )
    session = runner.run("trace-session")
    assert session.stage_execution_trace == [{
        "trial_id": "trace", "stage": "STAGE5", "fidelity": 4,
        "evidence_profile": "MODEL_FEATURE_DIAGNOSTIC_CN_1D",
        "profile_policy_version": "1.0.0", "split_ref": "validation-sample",
        "evaluation_ref": "qe:trace", "objective_id": "score", "actual_cost": 1.0,
    }]
    path = tmp_path / "checkpoint.json"
    session.checkpoint(path)
    assert SearchSession.resume(path).stage_execution_trace == session.stage_execution_trace
