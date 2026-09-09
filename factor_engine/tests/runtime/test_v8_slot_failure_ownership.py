from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from factor_engine.runtime.bounded_pipeline import execute_run_many_durable
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v7_direct_artifact_pipeline import (
    AckLossEngine,
    Broker,
    Factor,
    build_verified_then_hang_engine,
)


class HeldARefillAckLossEngine(AckLossEngine):
    """Keep A live while B commits its own artifact and loses only its ACK."""

    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        factor = factors[0]
        root = Path(self.marker)
        if result_policy == "return":
            (root / f"{factor.name}.reconciled").write_text("unexpected")
        (root / f"{factor.name}.started").write_text(str(os.getpid()))
        if factor.name == "a":
            deadline = time.monotonic() + 10
            while not (root / "release-a").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("A release timed out")
                time.sleep(0.01)
        sink(factor.name, self._value(factor.name))
        if factor.name == "b":
            (root / "b.artifact-written").write_text("written")
            os._exit(29)
        return {"results": {}}


def build_held_a_refill_ack_loss_engine(config):
    return HeldARefillAckLossEngine(config)


def _release_retained_authorities(exc):
    for name in ("run_state", "run_manifest"):
        authority = getattr(exc, name, None)
        if authority is not None:
            authority.close()
    lock = getattr(exc, "coordinator_lock", None)
    if lock is not None:
        lock.release()
    broker_ipc = getattr(exc, "broker_ipc", None)
    if broker_ipc is not None:
        broker_ipc.close()
    for lease in getattr(exc, "egress_leases", ()) or ():
        lease.release()


def test_refill_ack_loss_while_a_is_held_keeps_b_slot_identity(tmp_path):
    with pytest.raises(BaseException) as raised:
        execute_run_many_durable(
            None,
            [Factor("a"), Factor("b")],
            policy=resolve_default_policy({
                "initial_lookahead_factors": 1,
                "cooperative_cancel_grace_seconds": 0.01,
                "worker_exit_observation_seconds": 0.5,
                "job_unknown_seconds": 5,
                "job_min_seconds": 5,
                "job_max_seconds": 5,
            }),
            artifact_root=tmp_path / "artifacts",
            run_kwargs={"broker": Broker()},
            engine_factory=build_held_a_refill_ack_loss_engine,
            engine_factory_config={
                "marker": str(tmp_path), "modes": str(tmp_path / "modes")
            },
        )

    exc = raised.value
    slot = getattr(exc, "failing_slot", None)
    assert slot is not None, "await-any failure lost its worker/proxy/wave owner"
    assert slot.names == ["b"]
    assert [factor.name for _ordinal, factor in slot.admitted] == ["b"]
    assert set(slot.assignments) == {"b"}
    assert slot.worker is not None and slot.proxy is not None and slot.handle is not None
    assert getattr(exc, "direct_slot_fatal", False) is True
    assert (tmp_path / "b.artifact-written").is_file()

    receipt = json.loads(Path(exc.receipt_path).read_text())
    with sqlite3.connect(receipt["state_path"]) as db:
        rows = db.execute(
            "select name,attempts,state,error_code,commit_state "
            "from outcomes order by ordinal"
        ).fetchall()
        b_detail = db.execute(
            "select error_detail from outcomes where name='b'"
        ).fetchone()[0]
        evidence = db.execute(
            "select availability,assignment_count from fit_failure_evidence"
        ).fetchall()
        evidence_ordinals = db.execute(
            "select ordinal from fit_failure_evidence_assignments"
        ).fetchall()
    assert rows == [
        ("a", 1, "CANCELLED", "RUN_ABORTED", "UNKNOWN"),
        ("b", 1, "FAILED", "WORKER_PROTOCOL_INTEGRITY", "UNKNOWN"),
    ]
    assert b_detail.startswith("WorkerTransportFailed:")
    assert receipt["counts"] == {"CANCELLED": 1, "FAILED": 1}
    assert receipt["fit_failure_evidence"]["availability"] == "indexed"
    assert receipt["fit_failure_evidence"]["observed_waves"] == 0
    assert receipt["fit_failure_evidence"]["unavailable_waves"] == 1
    assert evidence == [("UNAVAILABLE", 1)]
    assert evidence_ordinals == [(1,)]
    # This failure is an explicit run-level STOPPING gate: it must not invoke
    # reconciliation with A's local direct_context, nor rewrite B's artifact.
    assert not (tmp_path / "a.reconciled").exists()
    assert not (tmp_path / "b.reconciled").exists()
    manifests = list(
        (Path(receipt["state_path"]).parent / "values").glob("*/manifest.json")
    )
    assert len(manifests) == 1
    artifact = json.loads(manifests[0].read_text())
    assert artifact["factor_id"] == "b"
    assert artifact["ordinal"] == 1
    assert artifact["generation"] == slot.assignments["b"][1]
    _release_retained_authorities(exc)


class TrackingLease:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class RetentionBroker(Broker):
    def __init__(self, leases):
        self.leases = leases

    def acquire_protected_egress(self, *args, **kwargs):
        return tuple(self.leases)


def _run_cancelled_quarantine(tmp_path, monkeypatch, *, add_final_cleanup_error=False):
    from factor_engine.runtime.exceptions import CancellationToken
    from factor_engine.runtime.supervised_worker import AsyncWorkerCall, WorkerQuarantined
    from factor_engine.runtime.supervised_worker import SupervisedReusableWorker

    token = CancellationToken()
    leases = [TrackingLease(), TrackingLease()]
    captured = []
    cleanup_allowed = threading.Event()
    original_cancel = AsyncWorkerCall.cancel_and_retire
    original_close = SupervisedReusableWorker.close
    inner_cleanup = RuntimeError("inner slot cleanup failed")
    final_cleanup = RuntimeError("final worker cleanup failed")
    raise_final = {"ready": False, "done": False}

    def injected_cancel(self):
        progress = dict(self._worker.last_progress or {})
        original_cancel(self)
        if progress.get("ordinal") == 1:
            failure = WorkerQuarantined("post-retirement quarantine oracle")
            if add_final_cleanup_error:
                failure.cleanup_errors = [inner_cleanup]
                raise_final["ready"] = True
            raise failure

    def close_with_final_error(self):
        result = original_close(self)
        if raise_final["ready"] and not raise_final["done"]:
            raise_final["done"] = True
            raise final_cleanup
        return result

    monkeypatch.setattr(AsyncWorkerCall, "cancel_and_retire", injected_cancel)
    if add_final_cleanup_error:
        monkeypatch.setattr(SupervisedReusableWorker, "close", close_with_final_error)

    def run():
        try:
            execute_run_many_durable(
                None, [Factor("first"), Factor("second")],
                policy=resolve_default_policy({
                    "initial_lookahead_factors": 1,
                    "cooperative_cancel_grace_seconds": 0.01,
                    "worker_exit_observation_seconds": 0.5,
                }),
                artifact_root=tmp_path / "artifacts",
                run_kwargs={"broker": RetentionBroker(leases)},
                engine_factory=build_verified_then_hang_engine,
                engine_factory_config={
                    "marker": str(tmp_path), "modes": str(tmp_path / "modes")
                },
                cancellation_token=token,
            )
        except BaseException as exc:
            captured.append(exc)
            cleanup_allowed.wait(8)
            _release_retained_authorities(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while not (tmp_path / "second.started").is_file():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    token.cancel()
    capture_deadline = time.monotonic() + 8
    while not captured:
        assert time.monotonic() < capture_deadline
        time.sleep(0.01)
    assert len(captured) == 1
    return (
        captured[0], leases, inner_cleanup, final_cleanup,
        cleanup_allowed, thread,
    )


def test_cleanup_pending_primary_retains_broker_and_egress_when_final_close_succeeds(
    tmp_path, monkeypatch,
):
    exc, leases, _inner, _final, cleanup_allowed, thread = _run_cancelled_quarantine(
        tmp_path, monkeypatch
    )
    try:
        assert getattr(exc, "cleanup_pending", False)
        assert exc.broker_ipc is not None
        assert exc.egress_leases == tuple(leases)
        assert exc.coordinator_lock.owned
        assert exc.run_state is not None and exc.run_manifest is not None
        assert not any(lease.released for lease in leases)
    finally:
        cleanup_allowed.set()
        thread.join(8)
    assert not thread.is_alive()


def test_final_cleanup_errors_append_to_existing_slot_cleanup_errors(tmp_path, monkeypatch):
    exc, leases, inner, final, cleanup_allowed, thread = _run_cancelled_quarantine(
        tmp_path, monkeypatch, add_final_cleanup_error=True
    )
    try:
        assert exc.cleanup_errors[0] is inner
        assert final in exc.cleanup_errors[1:]
        assert exc.broker_ipc is not None
        assert exc.egress_leases == tuple(leases)
    finally:
        cleanup_allowed.set()
        thread.join(8)
    assert not thread.is_alive()
