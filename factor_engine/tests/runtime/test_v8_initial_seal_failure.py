"""Initial manifest sealing errors must leave finite, inspectable outcomes."""
import json
import sqlite3
from pathlib import Path

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime import resume_validation as validation
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor


@pytest.mark.parametrize("expired", [False, True])
def test_seal_failure_has_abort_receipt_and_zero_attempt_terminals(tmp_path, monkeypatch, expired):
    def forbidden_start(*args, **kwargs):
        pytest.fail("seal failure must precede every compiler/compute/writer start")

    monkeypatch.setattr(pipeline.SupervisedReusableWorker, "start", forbidden_start)
    def fail_seal(*args, **kwargs):
        if expired:
            validation._validation_deadline(-1, resolve_default_policy())
        raise validation.ResumeIdentityError("injected invalid seal")

    monkeypatch.setattr(validation, "manifest_seal_payload", fail_seal)
    with pytest.raises(validation.ResumeIdentityError) as caught:
        pipeline.execute_run_many_durable(
            FakeEngine(), [FakeFactor("alpha"), FakeFactor("beta")],
            policy=resolve_default_policy(), artifact_root=tmp_path,
        )
    if expired:
        assert isinstance(caught.value, TimeoutError)
    receipt = json.loads(Path(caught.value.receipt_path).read_text())
    assert receipt["status"] == "ABORTED"
    assert receipt["counts"] == {"CANCELLED": 2}
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute("select attempts,commit_state from outcomes order by ordinal").fetchall() == [
            (0, "NOT_STARTED"), (0, "NOT_STARTED"),
        ]
    assert not (Path(receipt["identity_path"]).parent / "manifest_identity.json").exists()
