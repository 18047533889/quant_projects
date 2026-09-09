import hashlib
import json
import pickle
import sqlite3
import time
from pathlib import Path

import pytest

from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.persistent_run_state import PersistentRunState
from factor_engine.runtime.resume_validation import (
    ResumeIdentityError, iter_resume_pending, iter_resume_artifacts_to_revalidate,
    manifest_seal_payload, validate_resume_context,
    verify_previously_committed_artifact,
)


@pytest.fixture
def saved_run(tmp_path):
    run_id = "a" * 32
    directory = tmp_path / run_id
    directory.mkdir()
    policy = resolve_default_policy()
    identity = {"deployment_digest": "approved-fixture", "scope": {"revision": 1}}
    (directory / "identity.json").write_text(json.dumps(dict(
        schema_version="factor_engine.run_identity.v1", run_id=run_id,
        policy_id=policy.policy_id, policy_digest=policy.digest,
        run_identity=identity, deployment_digest=identity["deployment_digest"],
    )))
    manifest = FiniteFactorManifest.ingest(
        [{"name": "alpha"}, {"name": "beta"}], directory / "manifest.sqlite3")
    manifest.close()
    state = PersistentRunState(directory / "state.sqlite3", max_attempts=policy.work_item_max_attempts)
    state.register(0, "alpha")
    state.consume_attempt(0, "execution")
    state.record_commit_intent(0, "b" * 32)
    state.close()
    (directory / "manifest_identity.json").write_text(json.dumps(
        manifest_seal_payload(directory, policy=policy)))
    return tmp_path, directory, run_id, policy, identity


def validate(saved_run):
    root, _, run_id, policy, identity = saved_run
    return validate_resume_context(root, run_id, policy=policy, run_identity=identity)


def test_readonly_resume_preserves_attempt_and_generation_without_unpickling(saved_run, monkeypatch):
    monkeypatch.setattr(pickle, "loads", lambda *args: pytest.fail("must not deserialize during identity validation"))
    context = validate(saved_run)
    assert context.factor_count == context.pending_count == 2
    pending = list(iter_resume_pending(context))
    assert pending[0]["attempts"] == 1
    assert pending[0]["generation"] == "b" * 32
    assert pending[1]["state"] == "UNREGISTERED"
    assert pending[0]["worker_exit_proof"] == "REQUIRED_NOT_AVAILABLE_IN_LEGACY_STATE"
    assert validate(saved_run) == context


def test_resume_rejects_unsealed_legacy_run(saved_run):
    saved_run[1].joinpath("manifest_identity.json").unlink()
    with pytest.raises(ResumeIdentityError, match="control record"):
        validate(saved_run)


def test_resume_rejects_business_identity_even_bool_int_alias(saved_run):
    root, _, run_id, policy, identity = saved_run
    altered = {**identity, "scope": {"revision": True}}
    with pytest.raises(ResumeIdentityError, match="business identity"):
        validate_resume_context(root, run_id, policy=policy, run_identity=altered)


def test_resume_rejects_coherently_rehashed_but_changed_definition(saved_run):
    payload = pickle.dumps({"name": "alpha", "changed": True})
    with sqlite3.connect(saved_run[1] / "manifest.sqlite3") as db:
        db.execute("UPDATE factors SET definition=?,definition_bytes=?,definition_digest=? WHERE ordinal=0",
                   (payload, len(payload), hashlib.sha256(payload).hexdigest()))
    with pytest.raises(ResumeIdentityError, match="sealed manifest identity"):
        validate(saved_run)


@pytest.mark.parametrize("mutation,match", [
    ("UPDATE outcomes SET name='wrong' WHERE ordinal=0", "state ordinal"),
    ("UPDATE outcomes SET attempts=4 WHERE ordinal=0", "attempt count"),
    ("UPDATE outcomes SET artifact_generation=NULL WHERE ordinal=0", "lost its generation"),
    ("UPDATE state_policy SET value='2' WHERE key='max_attempts'", "attempt budget"),
])
def test_resume_rejects_mutated_state_without_resetting_it(saved_run, mutation, match):
    with sqlite3.connect(saved_run[1] / "state.sqlite3") as db:
        db.execute(mutation)
    with pytest.raises(ResumeIdentityError, match=match):
        validate(saved_run)


def test_resume_rejects_incomplete_input(saved_run):
    with sqlite3.connect(saved_run[1] / "manifest.sqlite3") as db:
        db.execute("UPDATE meta SET value='0' WHERE key='input_complete'")
    with pytest.raises(ResumeIdentityError, match="complete sealed"):
        validate(saved_run)


def test_resume_rejects_run_path_traversal(saved_run):
    with pytest.raises(ResumeIdentityError, match="UUID"):
        validate_resume_context(saved_run[0], "../" + saved_run[2],
                                policy=saved_run[3], run_identity=saved_run[4])


@pytest.mark.parametrize("deadline", [0, float("inf"), float("nan")])
def test_resume_validation_deadline_is_finite(saved_run, deadline):
    with pytest.raises(ResumeIdentityError, match="deadline"):
        validate_resume_context(saved_run[0], saved_run[2], policy=saved_run[3],
                                run_identity=saved_run[4], deadline=deadline)


def test_resume_rejects_boolean_seal_count(saved_run):
    seal_path = saved_run[1] / "manifest_identity.json"
    seal = json.loads(seal_path.read_text())
    seal["factor_count"] = True
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ResumeIdentityError, match="sealed manifest identity"):
        validate(saved_run)


@pytest.mark.parametrize("mutation", [
    "UPDATE outcomes SET state='SUCCEEDED',commit_state='GARBAGE'",
    "UPDATE outcomes SET state='SUCCEEDED',commit_state='NOT_STARTED'",
    "UPDATE outcomes SET state='SUCCEEDED',commit_state='VERIFIED',artifact_json='{}'",
    "UPDATE outcomes SET state='RUNNING',attempts=0",
    "UPDATE outcomes SET state='ACCEPTED'",
])
def test_resume_rejects_impossible_state_combinations(saved_run, mutation):
    with sqlite3.connect(saved_run[1] / "state.sqlite3") as db:
        db.execute(mutation)
    with pytest.raises(ResumeIdentityError):
        validate(saved_run)


def test_success_is_separate_artifact_revalidation_obligation(saved_run):
    state = PersistentRunState(saved_run[1] / "state.sqlite3")
    artifact = dict(committed=True, verified=True, generation="b" * 32)
    state.terminal(0, "SUCCEEDED", artifact=artifact, commit_state="VERIFIED")
    state.close()
    context = validate(saved_run)
    assert context.pending_count == 1
    assert context.artifact_revalidation_required
    obligations = list(iter_resume_artifacts_to_revalidate(context))
    assert obligations == [dict(ordinal=0, name="alpha", generation="b" * 32,
                                artifact=artifact, action="VERIFY_EXACT_ARTIFACT")]


def test_failed_unknown_commit_remains_a_reconciliation_obligation(saved_run):
    state = PersistentRunState(saved_run[1] / "state.sqlite3")
    state.mark_unknown_commit(0, "lost ACK")
    state.close()
    context = validate(saved_run)
    assert context.pending_count == 2
    assert list(iter_resume_pending(context))[0]["generation"] == "b" * 32


def test_missing_root_is_typed_identity_failure(saved_run):
    with pytest.raises(ResumeIdentityError, match="root"):
        validate_resume_context(saved_run[0] / "missing", saved_run[2],
                                policy=saved_run[3], run_identity=saved_run[4])


@pytest.fixture
def committed_artifact(saved_run):
    import pandas as pd
    from factor_engine.runtime.durable_artifact_sink import write_verified_factor_artifact

    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"])
    value = pd.Series([1., 2., 3., 4.], index=index)
    root, directory, run_id, policy, _ = saved_run
    receipt = write_verified_factor_artifact(
        root, run_id, 0, "alpha", value, policy=policy,
        budget_bytes=8 * 1024 * 1024, generation="b" * 32)
    state = PersistentRunState(directory / "state.sqlite3")
    state.terminal(0, "SUCCEEDED", artifact=receipt, commit_state="VERIFIED")
    state.close()
    context = validate(saved_run)
    obligation = next(iter_resume_artifacts_to_revalidate(context))
    return context, obligation, policy


def test_previously_verified_artifact_is_checked_without_recompute(committed_artifact):
    context, obligation, policy = committed_artifact
    result = verify_previously_committed_artifact(
        context, obligation, policy=policy, budget_bytes=1024 * 1024,
        deadline=time.monotonic() + 10)
    assert result["resume_bytes_reverified"] is True
    assert result["sha256"] == obligation["artifact"]["sha256"]


@pytest.mark.parametrize("target", ["manifest", "chunk"])
def test_previously_verified_artifact_detects_disk_corruption(committed_artifact, target):
    context, obligation, policy = committed_artifact
    manifest_path = Path(obligation["artifact"]["path"])
    if target == "manifest":
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    else:
        chunk = json.loads(manifest_path.read_text())["chunks"][0]["path"]
        chunk_path = manifest_path.parent / chunk
        data = chunk_path.read_bytes()
        chunk_path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ResumeIdentityError, match="hash changed"):
        verify_previously_committed_artifact(
            context, obligation, policy=policy, budget_bytes=1024 * 1024,
            deadline=time.monotonic() + 10)


def test_pending_manifest_cannot_self_approve_with_a_fabricated_obligation(committed_artifact):
    context, obligation, policy = committed_artifact
    with sqlite3.connect(context.state_path) as db:
        db.execute("UPDATE outcomes SET state='RUNNING',commit_state='INTENT' WHERE ordinal=0")
    with pytest.raises(ResumeIdentityError, match="persisted verified authority"):
        verify_previously_committed_artifact(
            context, obligation, policy=policy, budget_bytes=1024 * 1024,
            deadline=time.monotonic() + 10)


@pytest.mark.parametrize("obligation", [{}, [], {"ordinal": 0}])
def test_malformed_obligation_is_a_typed_identity_failure(saved_run, obligation):
    with pytest.raises(ResumeIdentityError):
        verify_previously_committed_artifact(
            validate(saved_run), obligation, policy=saved_run[3], budget_bytes=1024 * 1024,
            deadline=time.monotonic() + 10)
