import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from factor_engine.runtime.bounded_pipeline import execute_run_many_durable
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.resume_validation import ResumeIdentityError
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor
from factor_engine.tests.runtime.test_v7_direct_artifact_pipeline import (
    Broker, Factor, build_continuous_refill_engine,
)


def _roles(receipt):
    path = Path(receipt["manifest_path"]).parent / "worker-ownership.sqlite3"
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "select role,state from worker_instances order by starting_monotonic"
        ).fetchall()
    assert rows and all(state == "EXITED" for _role, state in rows)
    return [role for role, _state in rows]


def _rewrite_identity_markers(receipt, markers):
    identity_path = Path(receipt["identity_path"])
    identity = json.loads(identity_path.read_bytes())
    identity.pop("ownership_store", None)
    identity.pop("ownership_coverage", None)
    identity.update(markers)
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode()
    identity_path.write_bytes(payload)
    seal_path = identity_path.parent / "manifest_identity.json"
    seal = json.loads(seal_path.read_bytes())
    seal["identity_sha256"] = hashlib.sha256(payload).hexdigest()
    seal_path.write_text(json.dumps(
        seal, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")))


def test_new_normal_run_persists_sqlite_selection_and_owned_roles(tmp_path):
    receipt = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=resolve_default_policy(),
        artifact_root=tmp_path,
    )
    identity_path = Path(receipt["identity_path"])
    identity = json.loads(identity_path.read_text())
    assert identity["ownership_store"] == "sqlite-v1"
    assert identity["ownership_coverage"] == "all-durable-pipeline-os-processes-v1"
    assert _roles(receipt) == [
        "manifest-ingestion", "compiler", "compute-primary", "artifact-writer",
    ]


def test_empty_new_run_owns_only_ingestion_process(tmp_path):
    receipt = execute_run_many_durable(
        FakeEngine(), [], policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["execution_batches"] == 0
    assert _roles(receipt) == ["manifest-ingestion"]


def test_direct_refill_registers_every_started_role(tmp_path):
    receipt = execute_run_many_durable(
        None, [Factor(name) for name in "abcde"],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path, run_kwargs={"broker": Broker()},
        engine_factory=build_continuous_refill_engine,
        engine_factory_config={
            "marker": str(tmp_path), "modes": str(tmp_path / "events"), "hold": "none",
        },
    )
    roles = _roles(receipt)
    assert roles.count("manifest-ingestion") == 1
    assert roles.count("compiler") == 1
    assert roles.count("compute-primary") == 1
    assert roles.count("direct-artifact-reconciler") == 1
    assert sum(role in {"compute-spare", "compute-refill", "compute-replacement"}
               for role in roles) >= 1
    assert len({int(line.split(":")[3]) for line in (
        tmp_path / "events"
    ).read_text().splitlines() if ":start:" in line}) == 5


def test_new_terminal_resume_appends_verifier_without_rewriting_identity_or_store(tmp_path):
    policy = resolve_default_policy()
    first = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )
    identity_path = Path(first["identity_path"])
    store_path = identity_path.parent / "worker-ownership.sqlite3"
    identity_before = identity_path.read_bytes()
    identity_digest = hashlib.sha256(identity_before).hexdigest()
    store_inode = store_path.stat().st_ino

    resumed = execute_run_many_durable(
        FakeEngine(), (), policy=policy, artifact_root=tmp_path,
        resume_run_id=first["run_id"],
    )
    assert resumed["execution_batches"] == 0
    assert identity_path.read_bytes() == identity_before
    assert hashlib.sha256(identity_path.read_bytes()).hexdigest() == identity_digest
    assert store_path.stat().st_ino == store_inode
    assert _roles(resumed)[-1] == "resume-artifact-verifier"


@pytest.mark.parametrize("markers", [
    {"ownership_store": "sqlite-v1"},
    {"ownership_coverage": "all-durable-pipeline-os-processes-v1"},
    {"ownership_store": None,
     "ownership_coverage": "all-durable-pipeline-os-processes-v1"},
    {"ownership_store": "sqlite-v1", "ownership_coverage": None},
    {"ownership_store": "sqlite-v2",
     "ownership_coverage": "all-durable-pipeline-os-processes-v1"},
    {"ownership_store": "sqlite-v1", "ownership_coverage": "partial-v1"},
])
def test_resume_rejects_partial_null_or_unknown_ownership_markers(tmp_path, markers):
    policy = resolve_default_policy()
    first = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )
    _rewrite_identity_markers(first, markers)

    with pytest.raises(
            ResumeIdentityError,
            match="resume ownership identity markers are invalid"):
        execute_run_many_durable(
            FakeEngine(), (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )


def test_resume_allows_true_legacy_identity_with_both_markers_absent(tmp_path):
    policy = resolve_default_policy()
    first = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )
    _rewrite_identity_markers(first, {})

    resumed = execute_run_many_durable(
        FakeEngine(), (), policy=policy, artifact_root=tmp_path,
        resume_run_id=first["run_id"],
    )
    assert resumed["execution_batches"] == 0
