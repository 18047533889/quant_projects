"""FO-P1-25 regression tests: TrialEvaluationArtifact replaces the magic dict.

The historical runner read a magic dict of ad-hoc keys ("score", "cost",
"rank", "total", "promote", ...) from the evaluation callback.  These tests pin
that a well-formed artifact is the preferred contract and that a legacy magic
dict is normalized into one — an evaluator that renames/removes a key is caught
loudly instead of silently breaking the run.
"""

import math

import numpy as np
import pytest

from factor_optimizer.contracts.evaluation_artifact import (
    EvaluationStatus,
    TrialEvaluationArtifact,
    normalize_evaluation_result,
)
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.contracts.treatment_integrity import (
    IntegrityCheckResult,
    TreatmentIntegrityEvidence,
    build_integrity_evidence,
    require_integrity_evidence,
)
from factor_optimizer.search.runner import SearchConfig, SearchRunner


def _integrity_evidence(trial_id="t1"):
    """Real TreatmentIntegrityEvidence for a trial (R55 P0-9)."""
    from factor_optimizer.contracts.treatment_integrity import (
        build_integrity_evidence,
    )

    rng = np.random.default_rng(abs(hash(trial_id)) % (2 ** 32))
    before = rng.normal(size=32)
    return build_integrity_evidence(
        trial_id, "treatment::test", {"window": 3}, before, before * 0.5
    )


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
        {
            "score": 0.5,
            "cost": 2.0,
            "evidence_ref": "ev-1",
            "extra": 1,
            "treatment_integrity_evidence": _integrity_evidence("t1"),
        },
    )
    assert isinstance(artifact, TrialEvaluationArtifact)
    assert artifact.primary_objective_value == 0.5
    assert artifact.compute_cost == 2.0
    assert artifact.evidence_ref == "ev-1"
    # R55 P0-9: the evidence rides on the artifact and passes the gate.
    assert artifact.treatment_integrity_evidence is not None
    require_integrity_evidence("t1", artifact.treatment_integrity_evidence)
    # The evidence's content hash is recorded in the artifact metadata.
    assert (
        artifact.metadata["treatment_integrity_evidence"]
        == artifact.treatment_integrity_evidence.content_hash
    )


def test_legacy_dict_without_integrity_evidence_fails_closed():
    """An evaluator that omits integrity evidence is a FAILED trial, not a
    score (R55 P0-9: the historical magic-dict path admitted it silently)."""
    with pytest.raises(ValueError, match="treatment_integrity_evidence"):
        normalize_evaluation_result(
            "t1", {"score": 0.5, "cost": 2.0, "evidence_ref": "ev-1"}
        )


def test_legacy_dict_with_failed_integrity_check_is_gated():
    """Evidence that carries a failing check must not pass the gate."""
    good = _integrity_evidence("t1")
    failed = TreatmentIntegrityEvidence(
        treatment_id=good.treatment_id,
        treatment_kind=good.treatment_kind,
        applied_parameters=dict(good.applied_parameters),
        integrity_checks=tuple(good.integrity_checks)
        + (
            IntegrityCheckResult(
                name="pit_no_leakage",
                passed=False,
                expected="no future observation may reach the treated value",
                detail="lag check failed",
            ),
        ),
        before_digest=good.before_digest,
        after_digest=good.after_digest,
    )
    from factor_optimizer.errors import TreatmentIntegrityError

    with pytest.raises(TreatmentIntegrityError, match="pit_no_leakage"):
        require_integrity_evidence("t1", failed)


def test_legacy_dict_promotion_keys_folded_into_promotion_evidence():
    artifact = normalize_evaluation_result(
        "t1",
        {
            "score": 0.5,
            "rank": 0,
            "total": 10,
            "promote": True,
            "evidence_ref": "e",
            "treatment_integrity_evidence": _integrity_evidence("t1"),
        },
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
        return {
            "evaluation_id": f"e-{calls[0]}",
            "score": 0.5,
            "cost": 1.0,
            "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
        }

    session = SearchRunner(
        _config(1, 1, 10.0), lambda: _trial(), _protocol(eval_fn)
    ).run("artifact-run")
    trial = session.trials[0]
    assert trial.status is TrialStatus.EVALUATED
    assert trial.metadata["score"] == 0.5
    artifact_dict = trial.metadata["evaluation_artifact"]
    assert artifact_dict["evidence_ref"] == "e-1"
    assert artifact_dict["compute_cost"] == 1.0
    # R55 P0-9: the evidence is carried and recorded with its content hash.
    assert artifact_dict["treatment_integrity_evidence"]["treatment_id"] == "t1"
    assert artifact_dict["metadata"]["treatment_integrity_evidence"] == (
        artifact_dict["treatment_integrity_evidence"]["content_hash"]
    )


def test_runner_rejects_renamed_score_key_loudly():
    def eval_fn(trial, fidelity):
        return {"evaluation_id": "e", "rank_ic": 0.5, "cost": 1.0, "treatment_integrity_evidence": _integrity_evidence(trial.trial_id)}  # renamed!

    session = SearchRunner(
        _config(1, 1, 10.0), lambda: _trial(), _protocol(eval_fn)
    ).run("renamed-key")
    trial = session.trials[0]
    assert trial.status is TrialStatus.FAILED
    assert "missing a primary objective score" in trial.failure_reason


def test_runner_fails_trial_without_integrity_evidence():
    """An evaluator that never measured integrity produces a FAILED trial —
    its score can never become the incumbent (R55 P0-9)."""
    session = SearchRunner(
        _config(1, 1, 10.0),
        lambda: _trial(),
        _protocol(lambda trial, fid: {"score": 0.9, "cost": 1.0,
                              "evidence_ref": "e"}),
    ).run("no-integrity")
    trial = session.trials[0]
    assert trial.status is TrialStatus.FAILED
    assert "treatment_integrity_evidence" in trial.failure_reason
    assert session.best_trial_id is None


def test_runner_fails_trial_with_failed_integrity_check():
    """A trial whose evidence carries a failing check is FAILED, and the
    search proceeds without it (the failing check is recorded)."""
    good = _integrity_evidence("t1")
    bad = TreatmentIntegrityEvidence(
        treatment_id=good.treatment_id,
        treatment_kind=good.treatment_kind,
        applied_parameters=dict(good.applied_parameters),
        integrity_checks=tuple(good.integrity_checks)
        + (
            IntegrityCheckResult(
                name="pit_no_leakage",
                passed=False,
                expected="no future observation may reach the treated value",
                detail="lag check failed",
            ),
        ),
        before_digest=good.before_digest,
        after_digest=good.after_digest,
    )
    session = SearchRunner(
        _config(1, 1, 10.0),
        lambda: _trial(),
        _protocol(lambda trial, fid: {"score": 0.9, "cost": 1.0,
                                      "evidence_ref": "e",
                                      "treatment_integrity_evidence": bad}),
    ).run("failed-check")
    trial = session.trials[0]
    assert trial.status is TrialStatus.FAILED
    assert "pit_no_leakage" in trial.failure_reason
    assert session.best_trial_id is None


def test_runner_accepts_trial_with_passing_integrity_evidence():
    """A trial with complete, passing evidence is EVALUATED and becomes the
    incumbent."""
    session = SearchRunner(
        _config(1, 1, 10.0),
        lambda: _trial(),
        _protocol(lambda trial, fid: {"score": 0.9, "cost": 1.0,
                                      "evidence_ref": "e",
                                      "treatment_integrity_evidence":
                                          _integrity_evidence(trial.trial_id)}),
    ).run("passing-integrity")
    trial = session.trials[0]
    assert trial.status is TrialStatus.EVALUATED
    assert session.best_trial_id == "t1"
    artifact = TrialEvaluationArtifact.from_dict(
        trial.metadata["evaluation_artifact"]
    )
    # The gate re-derives the verdict from the recorded evidence.
    artifact.require_passing_integrity()


def test_evaluation_artifact_roundtrip_keeps_integrity_evidence():
    artifact = normalize_evaluation_result(
        "t1",
        {"score": 0.5, "cost": 1.0, "evidence_ref": "e",
         "treatment_integrity_evidence": _integrity_evidence("t1")},
    )
    data = artifact.to_dict()
    restored = TrialEvaluationArtifact.from_dict(data)
    assert restored.treatment_integrity_evidence is not None
    assert restored.treatment_integrity_evidence.content_hash == (
        artifact.treatment_integrity_evidence.content_hash
    )
    restored.require_passing_integrity()
