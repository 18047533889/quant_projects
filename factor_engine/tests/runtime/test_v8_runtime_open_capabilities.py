from pathlib import Path
import json
import sqlite3

import pytest

from factor_engine.runtime.bounded_pipeline import execute_run_many_durable
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor
from factor_engine.tests.runtime.test_v7_direct_artifact_pipeline import (
    Broker, Factor, build_ack_loss_engine,
)


@pytest.mark.xfail(
    strict=True,
    reason="V8-R02: safe multi-wave worker epoch reuse is not implemented",
)
def test_multi_wave_epoch_reuses_pid_only_after_ownership_proof(tmp_path):
    marker = tmp_path / "normal"
    marker.write_text("no-ack-loss")
    modes = tmp_path / "modes"
    receipt = execute_run_many_durable(
        None, [Factor("first"), Factor("second")],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path / "artifacts",
        run_kwargs={"broker": Broker()},
        engine_factory=build_ack_loss_engine,
        engine_factory_config={"marker": str(marker), "modes": str(modes)},
    )
    assert receipt["status"] == "SUCCEEDED"
    pids = [line.split(":")[1] for line in modes.read_text().splitlines()]
    assert len(pids) == 2
    assert pids[0] == pids[1]


def test_existing_identity_manifest_and_state_resume_same_run(tmp_path):
    first = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=resolve_default_policy(),
        artifact_root=tmp_path,
    )
    resumed = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=resolve_default_policy(),
        artifact_root=tmp_path,
        manifest_path=Path(first["manifest_path"]),
        state_path=Path(first["state_path"]),
    )
    assert resumed["run_id"] == first["run_id"]
    assert resumed["counts"] == {"SUCCEEDED": 1}


def test_explicit_resume_reopens_same_terminal_run_without_consuming_input(tmp_path):
    policy = resolve_default_policy()
    first = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )

    def forbidden_input():
        raise AssertionError("resume must not consume a replacement factor iterable")
        yield

    resumed = execute_run_many_durable(
        FakeEngine(), forbidden_input(), policy=policy, artifact_root=tmp_path,
        resume_run_id=first["run_id"],
    )
    assert resumed["run_id"] == first["run_id"]
    assert resumed["counts"] == {"SUCCEEDED": 1}
    assert resumed["execution_batches"] == 0


def test_explicit_resume_rejects_changed_artifact_bytes_and_releases_lock(tmp_path):
    from factor_engine.runtime.resume_validation import ResumeIdentityError

    policy = resolve_default_policy()
    first = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )
    with sqlite3.connect(first["state_path"]) as db:
        artifact = json.loads(db.execute(
            "select artifact_json from outcomes where ordinal=0"
        ).fetchone()[0])
    manifest = json.loads(Path(artifact["path"]).read_text())
    chunk = Path(artifact["path"]).parent / manifest["chunks"][0]["path"]
    payload = bytearray(chunk.read_bytes())
    payload[len(payload) // 2] ^= 1
    chunk.write_bytes(payload)
    with pytest.raises(ResumeIdentityError, match="hash changed"):
        execute_run_many_durable(
            FakeEngine(), (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )
    from factor_engine.runtime.bounded_pipeline import _RunCoordinatorLock
    probe = _RunCoordinatorLock(Path(first["manifest_path"]).parent)
    probe.acquire(allow_dead_owner=True)
    probe.release()


def test_explicit_resume_pending_intent_fails_without_attempt_or_generation_change(tmp_path):
    from factor_engine.runtime.resume_validation import ResumeIdentityError

    policy = resolve_default_policy()
    first = execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )
    with sqlite3.connect(first["state_path"]) as db:
        generation, attempts = db.execute(
            "select artifact_generation,attempts from outcomes where ordinal=0"
        ).fetchone()
        db.execute(
            "update outcomes set state='RUNNING',commit_state='INTENT',artifact_json=NULL "
            "where ordinal=0"
        )
    with pytest.raises(ResumeIdentityError, match="pending same-run execution"):
        execute_run_many_durable(
            FakeEngine(), (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )
    with sqlite3.connect(first["state_path"]) as db:
        assert db.execute(
            "select artifact_generation,attempts from outcomes where ordinal=0"
        ).fetchone() == (generation, attempts)
