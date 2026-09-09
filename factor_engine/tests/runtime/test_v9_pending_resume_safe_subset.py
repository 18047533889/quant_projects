import json
from contextlib import closing
from dataclasses import replace
import shutil
import sqlite3
import time
from pathlib import Path
import pytest
import pandas as pd
from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.resume_validation import ResumeIdentityError
from factor_engine.tests.runtime.test_v6_bounded_pipeline import (
    FakeEngine, FakeFactor,
)
from factor_engine.tests.runtime.test_v7_direct_artifact_pipeline import Broker, Factor


class DirectSuccessEngine:
    run_mode = "research"

    def __init__(self, config):
        self.marker = Path(config["marker"])
        self.wait_for_refill = config.get("wait_for_refill")

    def compile(self, factor):
        return factor

    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        names = {factor.name for factor in factors}
        if self.wait_for_refill and "new-refill" in names:
            Path(self.wait_for_refill).touch()
        if self.wait_for_refill and "new-slow" in names:
            deadline = time.monotonic() + 15.0
            while not Path(self.wait_for_refill).exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for refill worker to start")
                time.sleep(0.01)
        with self.marker.open("a") as stream:
            stream.write(",".join(factor.name for factor in factors) + "\n")
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-02"), "000001.SZ")],
            names=["time", "instrument"],
        )
        results = {
            factor.name: pd.Series([1.0], index=index, name=factor.name)
            for factor in factors
        }
        if result_policy == "sink":
            for name, value in results.items():
                sink(name, value)
            return {"results": {}}
        return {"results": results}


def build_direct_success_engine(config):
    return DirectSuccessEngine(config)

def test_never_dispatched_accepted_resumes_after_strict_worker_exit_proof(tmp_path):
    policy = resolve_default_policy({"initial_lookahead_factors": 1})
    first = pipeline.execute_run_many_durable(
        FakeEngine(), [FakeFactor("done"), FakeFactor("pending")],
        policy=policy, artifact_root=tmp_path, run_identity={})
    with sqlite3.connect(first["state_path"]) as db:
        evidence_id = db.execute(
            "select evidence_id from fit_failure_evidence_assignments where ordinal=1"
        ).fetchone()[0]
        db.execute("delete from fit_failure_evidence_assignments where evidence_id=?", (evidence_id,))
        db.execute("delete from fit_failure_evidence where evidence_id=?", (evidence_id,))
        db.execute("update outcomes set state='ACCEPTED',attempts=0,error_code=NULL,error_detail=NULL,retryable=0,artifact_json=NULL,commit_state='NOT_STARTED',artifact_generation=NULL where ordinal=1")
    for path in (Path(first["state_path"]).parent / "values").glob("00000001-*"):
        shutil.rmtree(path)
    receipt_path = Path(first["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["counts"] = {"ACCEPTED": 1, "SUCCEEDED": 1}
    receipt["all_terminal"] = receipt["all_requested_terminal"] = receipt["all_outputs_valid"] = False
    receipt["terminal_count"] = 1

    with sqlite3.connect(first["state_path"]) as db:
        summary = db.execute("select count(*),coalesce(max(seq),0),sum(availability='OBSERVED'),sum(availability='UNAVAILABLE') from fit_failure_evidence").fetchone()
    receipt["fit_failure_evidence"].update(wave_count=summary[0], last_seq=summary[1], observed_waves=summary[2], unavailable_waves=summary[3], truncated_waves=0, next_seq=0 if summary[1] else None)
    pipeline._write_control_receipt(receipt_path, receipt)
    with closing(sqlite3.connect(first["state_path"])) as db:
        before = db.execute("select attempts,artifact_json from outcomes where ordinal=0").fetchone()
    second = pipeline.execute_run_many_durable(
        FakeEngine(), (), policy=policy, artifact_root=tmp_path,
        run_identity={}, resume_run_id=first["run_id"],
    )
    assert second["counts"] == {"SUCCEEDED": 2}
    with sqlite3.connect(first["state_path"]) as db:
        assert db.execute("select attempts,artifact_json from outcomes where ordinal=0").fetchone() == before
        assert db.execute("select attempts,state,commit_state from outcomes where ordinal=1").fetchone() == (1,"SUCCEEDED","VERIFIED")


def test_spawn_direct_mixed_resume_filters_main_prefetch_and_refill(tmp_path):
    # Three two-item waves force all admission sites.  The first wave runs in
    # the current slot, the second is prefetched, and its fast completion while
    # the current slot is still live causes the third wave to enter via refill.
    policy = resolve_default_policy({"initial_lookahead_factors": 2})
    factors = [Factor(name) for name in (
        "new-slow", "done-a", "new-next", "done-b", "new-refill", "done-c",
    )]
    marker = tmp_path / "computed"
    config = {"marker": str(marker)}
    first = pipeline.execute_run_many_durable(
        None, factors, policy=policy, artifact_root=tmp_path,
        run_identity={}, run_kwargs={"broker": Broker()},
        engine_factory=build_direct_success_engine, engine_factory_config=config,
    )
    state_path = Path(first["state_path"])
    with sqlite3.connect(state_path) as db:
        before_done = db.execute(
            "select ordinal,attempts,artifact_generation,artifact_json "
            "from outcomes where ordinal in (1,3,5) order by ordinal"
        ).fetchall()
        evidence_ids = [row[0] for row in db.execute(
            "select distinct evidence_id from fit_failure_evidence_assignments "
            "where ordinal in (0,2,4)"
        )]
        for evidence_id in evidence_ids:
            db.execute(
                "delete from fit_failure_evidence_assignments where evidence_id=?",
                (evidence_id,),
            )
            db.execute(
                "delete from fit_failure_evidence where evidence_id=?", (evidence_id,)
            )
        db.execute(
            "update outcomes set state='ACCEPTED',attempts=0,error_code=NULL,"
            "error_detail=NULL,retryable=0,artifact_json=NULL,"
            "commit_state='NOT_STARTED',artifact_generation=NULL "
            "where ordinal in (0,2,4)"
        )
        summary = db.execute(
            "select count(*),coalesce(max(seq),0),sum(availability='OBSERVED'),"
            "sum(availability='UNAVAILABLE') from fit_failure_evidence"
        ).fetchone()
    for ordinal in (0, 2, 4):
        for path in (state_path.parent / "values").glob(f"{ordinal:08d}-*"):
            shutil.rmtree(path)
    receipt_path = Path(first["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt.update(counts={"ACCEPTED": 3, "SUCCEEDED": 3}, all_terminal=False,
                   all_requested_terminal=False, all_outputs_valid=False,
                   terminal_count=3)
    receipt["fit_failure_evidence"].update(
        wave_count=summary[0], last_seq=summary[1], observed_waves=summary[2] or 0,
        unavailable_waves=summary[3] or 0, truncated_waves=0,
        next_seq=0 if summary[1] else None,
    )
    pipeline._write_control_receipt(receipt_path, receipt)
    marker.write_text("")
    ownership_path = state_path.parent / "worker-ownership.sqlite3"
    with sqlite3.connect(ownership_path) as db:
        pre_resume_instances = {
            row[0] for row in db.execute("select instance_id from worker_instances")
        }
    refill_started = tmp_path / "resume-refill-started"
    config["wait_for_refill"] = str(refill_started)
    resumed = pipeline.execute_run_many_durable(
        None, (), policy=policy, artifact_root=tmp_path, run_identity={},
        resume_run_id=first["run_id"], run_kwargs={"broker": Broker()},
        engine_factory=build_direct_success_engine, engine_factory_config=config,
    )
    assert sorted(marker.read_text().splitlines()) == [
        "new-next", "new-refill", "new-slow",
    ]
    assert resumed["counts"] == {"SUCCEEDED": 6}
    with sqlite3.connect(state_path) as db:
        assert db.execute(
            "select ordinal,attempts,artifact_generation,artifact_json "
            "from outcomes where ordinal in (1,3,5) order by ordinal"
        ).fetchall() == before_done
        assert db.execute(
            "select ordinal,attempts,state,commit_state from outcomes "
            "where ordinal in (0,2,4) order by ordinal"
        ).fetchall() == [
            (0, 1, "SUCCEEDED", "VERIFIED"),
            (2, 1, "SUCCEEDED", "VERIFIED"),
            (4, 1, "SUCCEEDED", "VERIFIED"),
        ]
    with sqlite3.connect(ownership_path) as db:
        new_workers = [
            (instance_id, role)
            for instance_id, role in db.execute(
                "select instance_id,role from worker_instances"
            )
            if instance_id not in pre_resume_instances
        ]
    assert refill_started.is_file()
    assert {role for _instance_id, role in new_workers} >= {
        "compute-spare", "compute-refill",
    }


@pytest.mark.parametrize("tamper", ["starting", "bound", "pid", "namespace"])
def test_pending_resume_rejects_incomplete_or_corrupt_exit_proof(tmp_path, tamper):
    policy = resolve_default_policy()
    first = pipeline.execute_run_many_durable(
        FakeEngine(), [FakeFactor("pending")], policy=policy,
        artifact_root=tmp_path, run_identity={},
    )
    with sqlite3.connect(first["state_path"]) as db:
        evidence_id = db.execute(
            "select evidence_id from fit_failure_evidence_assignments where ordinal=0"
        ).fetchone()[0]
        db.execute(
            "delete from fit_failure_evidence_assignments where evidence_id=?",
            (evidence_id,),
        )
        db.execute("delete from fit_failure_evidence where evidence_id=?", (evidence_id,))
        db.execute(
            "update outcomes set state='ACCEPTED',attempts=0,error_code=NULL,"
            "error_detail=NULL,retryable=0,artifact_json=NULL,"
            "commit_state='NOT_STARTED',artifact_generation=NULL where ordinal=0"
        )
    receipt_path = Path(first["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt.update(counts={"ACCEPTED": 1}, all_terminal=False,
                   all_requested_terminal=False, all_outputs_valid=False,
                   terminal_count=0)
    receipt["fit_failure_evidence"].update(
        observed_waves=0, unavailable_waves=0, truncated_waves=0,
        wave_count=0, last_seq=0, next_seq=None,
    )
    pipeline._write_control_receipt(receipt_path, receipt)
    ownership = Path(first["state_path"]).parent / "worker-ownership.sqlite3"
    with sqlite3.connect(ownership) as db:
        instance_id = db.execute(
            "select instance_id from worker_instances limit 1"
        ).fetchone()[0]
        if tamper == "starting":
            db.execute(
                "delete from worker_events where instance_id=? and event in ('BOUND','EXITED')",
                (instance_id,),
            )
            db.execute(
                "update worker_instances set state='STARTING',pid=NULL,starttime=NULL,"
                "boot_id=NULL,machine_id=NULL,pid_namespace=NULL,exit_evidence=NULL,"
                "observed_boot_id=NULL,bound_monotonic=NULL,exited_monotonic=NULL "
                "where instance_id=?", (instance_id,),
            )
            db.execute(
                "update store_counters set event_count=event_count-2 where singleton=1"
            )
        elif tamper == "bound":
            db.execute(
                "delete from worker_events where instance_id=? and event='EXITED'",
                (instance_id,),
            )
            db.execute(
                "update worker_instances set state='BOUND',exit_evidence=NULL,"
                "observed_boot_id=NULL,exited_monotonic=NULL where instance_id=?",
                (instance_id,),
            )
            db.execute(
                "update store_counters set event_count=event_count-1 where singleton=1"
            )
        elif tamper == "pid":
            db.execute(
                "update worker_instances set starttime='1' where instance_id=?",
                (instance_id,),
            )
        else:
            db.execute(
                "update worker_instances set pid_namespace='tampered' where instance_id=?",
                (instance_id,),
            )
    with pytest.raises(ResumeIdentityError, match="persisted worker-exit proof"):
        pipeline.execute_run_many_durable(
            FakeEngine(), (), policy=policy, artifact_root=tmp_path,
            run_identity={}, resume_run_id=first["run_id"],
        )


def _never_dispatched_checkpoint(tmp_path, *, keep_dispatch_evidence=False):
    """Construct a checkpoint fixture; this is not a real crash campaign."""
    policy = resolve_default_policy()
    first = pipeline.execute_run_many_durable(
        FakeEngine(), [FakeFactor("pending")], policy=policy,
        artifact_root=tmp_path, run_identity={},
    )
    with closing(sqlite3.connect(first["state_path"])) as db, db:
        if not keep_dispatch_evidence:
            db.execute("DELETE FROM fit_failure_evidence_assignments")
            db.execute("DELETE FROM fit_failure_evidence")
        db.execute(
            "UPDATE outcomes SET state='ACCEPTED',attempts=0,error_code=NULL,"
            "error_detail=NULL,retryable=0,artifact_json=NULL,"
            "commit_state='NOT_STARTED',artifact_generation=NULL WHERE ordinal=0"
        )
    Path(first["receipt_path"]).unlink()
    return policy, first


def _forbid_writable_resume(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("rejected resume opened writable state before safety checks")
    monkeypatch.setattr(pipeline, "PersistentRunState", forbidden)


def test_pending_original_deadline_expiry_rejects_before_writable_state(tmp_path, monkeypatch):
    from factor_engine.runtime import run_deadline

    policy, first = _never_dispatched_checkpoint(tmp_path)
    original_restore = run_deadline.restore_run_deadline
    identity = json.loads((Path(first["state_path"]).parent / "identity.json").read_text())
    original_record = identity["job_deadline"]
    seen = []

    def expire_original(record, **kwargs):
        seen.append(record)
        assert record == original_record
        return original_restore(
            record, **kwargs,
            monotonic_ns=record["start_monotonic_ns"] + record["duration_ns"] + 1,
        )

    monkeypatch.setattr(run_deadline, "restore_run_deadline", expire_original)
    _forbid_writable_resume(monkeypatch)
    with pytest.raises(ResumeIdentityError, match="restored job deadline exhausted"):
        pipeline.execute_run_many_durable(
            FakeEngine(), (), policy=policy, artifact_root=tmp_path,
            run_identity={}, resume_run_id=first["run_id"],
        )
    assert seen == [original_record]


@pytest.mark.parametrize("state,commit,generation", [
    ("RUNNING", "NOT_STARTED", None),
    ("RUNNING", "INTENT", "a" * 32),
    ("RUNNING", "UNKNOWN", "a" * 32),
])
def test_dispatched_pending_is_not_authorized_to_resume(
        tmp_path, monkeypatch, state, commit, generation):
    policy, first = _never_dispatched_checkpoint(tmp_path)
    with closing(sqlite3.connect(first["state_path"])) as db, db:
        db.execute(
            "UPDATE outcomes SET state=?,attempts=1,commit_state=?,artifact_generation=?",
            (state, commit, generation),
        )
    _forbid_writable_resume(monkeypatch)
    with pytest.raises(ResumeIdentityError, match="dispatched or reconcilable"):
        pipeline.execute_run_many_durable(
            FakeEngine(), (), policy=policy, artifact_root=tmp_path,
            run_identity={}, resume_run_id=first["run_id"],
        )


def test_accepted_with_dispatch_evidence_is_rejected(tmp_path, monkeypatch):
    policy, first = _never_dispatched_checkpoint(tmp_path, keep_dispatch_evidence=True)
    _forbid_writable_resume(monkeypatch)
    with pytest.raises(ResumeIdentityError, match="persisted dispatch evidence"):
        pipeline.execute_run_many_durable(
            FakeEngine(), (), policy=policy, artifact_root=tmp_path,
            run_identity={}, resume_run_id=first["run_id"],
        )


@pytest.mark.parametrize("missing", ["ownership_store", "ownership_coverage",
                                   "fit_failure_evidence_available"])
def test_pending_executor_requires_full_validated_authority(tmp_path, monkeypatch, missing):
    from factor_engine.runtime import resume_validation

    policy, first = _never_dispatched_checkpoint(tmp_path)
    original_validate = resume_validation.validate_resume_context

    def incomplete(*args, **kwargs):
        context = original_validate(*args, **kwargs)
        return replace(context, **{missing: None})

    monkeypatch.setattr(resume_validation, "validate_resume_context", incomplete)
    _forbid_writable_resume(monkeypatch)
    with pytest.raises(ResumeIdentityError, match="full SQLite ownership and indexed evidence"):
        pipeline.execute_run_many_durable(
            FakeEngine(), (), policy=policy, artifact_root=tmp_path,
            run_identity={}, resume_run_id=first["run_id"],
        )
