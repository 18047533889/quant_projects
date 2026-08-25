"""FO-P1-25 regression tests: TrialEvaluationArtifact replaces the magic dict.

The historical runner read a magic dict of ad-hoc keys ("score", "cost",
"rank", "total", "promote", ...) from the evaluation callback.  These tests pin
that a well-formed artifact is the preferred contract and that a legacy magic
dict is normalized into one — an evaluator that renames/removes a key is caught
loudly instead of silently breaking the run.
"""

import math

import pytest

from factor_optimizer.contracts.evaluation_artifact import (
    EvaluationStatus,
    TrialEvaluationArtifact,
    normalize_evaluation_result,
)
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.runner import SearchConfig, SearchRunner


def _protocol(fn):
    return EvaluationProtocol(
        SplitPlan("search", [True], [False], [False], {}), fn
    )


def _trial(tid="t1"):
    return Trial(trial_id=tid, mutation_id="m", status=TrialStatus.PROPOSED)


def _config(max_trials=2, max_eval=2, max_cost=20.0, **kw):
    return SearchConfig(
        budget=SearchBudget(max_trials=max_trials, max_evaluations=max_eval, max_cost_units=max_cost),
        enable_multifidelity=False,
        **kw,
    )


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------


def test_artifact_construction_and_content_hash():
    artifact = TrialEvaluationArtifact(
        trial_id="t1",
        objective_values=[{"name": "rank_ic", "value": 0.5}],
        objective_spec_ref="rank_ic",
        split_ref="p1",
        fidelity=0,
        evidence_ref="ev-1",
        compute_cost=2.0,
        status=EvaluationStatus.COMPLETE,
    )
    assert artifact.primary_objective_value == 0.5
    assert artifact.content_hash
    artifact.verify()  # genuine artifact verifies


def test_artifact_rejects_bool_score():
    with pytest.raises(TypeError, match="non-boolean number"):
        TrialEvaluationArtifact(
            trial_id="t1",
            objective_values=[{"name": "score", "value": True}],
            evidence_ref="e",
        )


def test_artifact_rejects_nan_score():
    with pytest.raises(ValueError, match="finite"):
        TrialEvaluationArtifact(
            trial_id="t1",
            objective_values=[{"name": "score", "value": float("nan")}],
            evidence_ref="e",
        )


def test_artifact_rejects_missing_objective_values():
    with pytest.raises(ValueError, match="objective_values must not be empty"):
        TrialEvaluationArtifact(trial_id="t1", objective_values=[], evidence_ref="e")


def test_artifact_rejects_negative_cost():
    with pytest.raises(ValueError, match="finite and >= 0"):
        TrialEvaluationArtifact(
            trial_id="t1",
            objective_values=[{"name": "score", "value": 0.5}],
            evidence_ref="e",
            compute_cost=-1.0,
        )


def test_artifact_roundtrip_and_tamper_detection():
    artifact = TrialEvaluationArtifact(
        trial_id="t1",
        objective_values=[{"name": "score", "value": 0.5}],
        evidence_ref="ev-1",
        compute_cost=2.0,
    )
    data = artifact.to_dict()
    restored = TrialEvaluationArtifact.from_dict(data)
    assert restored == artifact
    # Tampering with a field after construction must be detected.
    tampered = dict(data)
    tampered["compute_cost"] = 99.0
    with pytest.raises(ValueError, match="content_hash"):
        TrialEvaluationArtifact.from_dict(tampered)


# ---------------------------------------------------------------------------
# Legacy magic dict normalization
# ---------------------------------------------------------------------------


def test_legacy_magic_dict_normalizes_into_artifact():
    artifact = normalize_evaluation_result(
        "t1",
        {"score": 0.5, "cost": 2.0, "evidence_ref": "ev-1", "extra": 1},
    )
    assert isinstance(artifact, TrialEvaluationArtifact)
    assert artifact.primary_objective_value == 0.5
    assert artifact.compute_cost == 2.0
    assert artifact.evidence_ref == "ev-1"


def test_legacy_dict_promotion_keys_folded_into_promotion_evidence():
    artifact = normalize_evaluation_result(
        "t1",
        {"score": 0.5, "rank": 0, "total": 10, "promote": True, "evidence_ref": "e"},
    )
    assert artifact.promotion_evidence == {
        "rank": 0,
        "total": 10,
        "promote": True,
        "baseline_score": None,
    }


def test_legacy_dict_missing_score_raises_loudly():
    with pytest.raises(ValueError, match="missing a primary objective score"):
        normalize_evaluation_result("t1", {"cost": 1.0, "evidence_ref": "e"})


def test_legacy_dict_missing_evidence_raises_loudly():
    with pytest.raises(ValueError, match="evidence_ref"):
        normalize_evaluation_result("t1", {"score": 0.5, "cost": 1.0})


def test_legacy_dict_boolean_score_raises_loudly():
    with pytest.raises(ValueError, match="must not be boolean"):
        normalize_evaluation_result("t1", {"score": True, "evidence_ref": "e"})


# ---------------------------------------------------------------------------
# The runner reads the artifact, never bare magic keys
# ---------------------------------------------------------------------------


def test_runner_evaluates_and_records_artifact():
    calls = [0]

    def eval_fn(trial, fidelity):
        calls[0] += 1
        return {"evaluation_id": f"e-{calls[0]}", "score": 0.5, "cost": 1.0}

    session = SearchRunner(
        _config(1, 1, 10.0), lambda: _trial(), _protocol(eval_fn)
    ).run("artifact-run")
    trial = session.trials[0]
    assert trial.status is TrialStatus.EVALUATED
    assert trial.metadata["score"] == 0.5
    artifact_dict = trial.metadata["evaluation_artifact"]
    assert artifact_dict["evidence_ref"] == "e-1"
    assert artifact_dict["compute_cost"] == 1.0


def test_runner_rejects_renamed_score_key_loudly():
    def eval_fn(trial, fidelity):
        return {"evaluation_id": "e", "rank_ic": 0.5, "cost": 1.0}  # renamed!

    session = SearchRunner(
        _config(1, 1, 10.0), lambda: _trial(), _protocol(eval_fn)
    ).run("renamed-key")
    trial = session.trials[0]
    assert trial.status is TrialStatus.FAILED
    assert "missing a primary objective score" in trial.failure_reason
