"""Real durable pipeline terminals; synthetic engine, not native certification."""
import json
import sqlite3
from pathlib import Path

import pytest

from factor_engine.runtime.bounded_pipeline import execute_run_many_durable
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor


class BrokenDependenciesFactor:
    name = "broken-dependencies"

    @property
    def dependencies(self):
        raise ValueError("unreadable factor dependencies")


@pytest.mark.parametrize("bad,reason", [
    (FakeFactor("量" * 400), "FACTOR_NAME_TOO_LARGE"),
    (BrokenDependenciesFactor(), "INVALID_FACTOR_DEPENDENCIES"),
])
def test_bad_metadata_is_zero_attempt_rejection_and_peer_is_persisted(
    tmp_path, bad, reason,
):
    receipt = execute_run_many_durable(
        FakeEngine(), [bad, FakeFactor("healthy")],
        policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["counts"] == {"REJECTED": 1, "SUCCEEDED": 1}
    with sqlite3.connect(receipt["state_path"]) as db:
        rows = db.execute(
            "SELECT ordinal,state,error_code,attempts,commit_state "
            "FROM outcomes ORDER BY ordinal"
        ).fetchall()
    assert rows[0] == (0, "REJECTED", reason, 0, "NOT_STARTED")
    assert rows[1][0:4] == (1, "SUCCEEDED", None, 1)
    assert rows[1][4] == "VERIFIED"
    assert json.loads(Path(receipt["receipt_path"]).read_text())["counts"] == receipt["counts"]


def test_legal_large_definition_survives_registration_seal_compute_and_write(tmp_path):
    factor = FakeFactor("legal-large-definition")
    factor.padding = "x" * 32_768
    receipt = execute_run_many_durable(
        FakeEngine(), [factor], policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["counts"] == {"SUCCEEDED": 1}
    with sqlite3.connect(receipt["manifest_path"]) as db:
        assert db.execute("SELECT definition_bytes FROM factors").fetchone()[0] > 32_768
    assert (Path(receipt["identity_path"]).parent / "manifest_identity.json").is_file()
