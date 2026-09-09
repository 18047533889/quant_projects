import time
import sqlite3
import json
from pathlib import Path

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import (
    FakeBroker, FakeEngine, FakeFactor,
)


class Lease:
    def release(self):
        pass


class TrackingLease(Lease):
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class TemporaryPressureBroker:
    hard_memory_limit = 1024

    def __init__(self, succeed_after=1):
        self.calls = 0
        self.waits = []
        self.succeed_after = succeed_after

    def acquire_protected_egress(self, *args, **kwargs):
        self.calls += 1
        if self.succeed_after is None or self.calls <= self.succeed_after:
            return None
        return Lease(), Lease()

    def wait_for_resource_change(self, timeout):
        self.waits.append(timeout)


def test_temporary_egress_pressure_waits_then_succeeds_without_new_identity():
    broker = TemporaryPressureBroker(succeed_after=1)
    leases, error = pipeline._acquire_protected_egress_until(
        broker, 100, 100, lease_id="one-run", deadline=time.monotonic() + 1,
        poll_seconds=.01,
    )
    assert leases is not None
    assert error is None
    assert broker.calls == 2
    assert len(broker.waits) == 1


def test_egress_pressure_has_finite_wait_terminal():
    broker = TemporaryPressureBroker(succeed_after=None)
    leases, error = pipeline._acquire_protected_egress_until(
        broker, 100, 100, lease_id="one-run", deadline=time.monotonic() + .01,
        poll_seconds=.002,
    )
    assert leases is None
    assert error == "RESOURCE_WAIT_EXHAUSTED"
    assert broker.calls > 1


def test_impossible_egress_requirement_rejects_without_waiting():
    broker = TemporaryPressureBroker(succeed_after=0)
    leases, error = pipeline._acquire_protected_egress_until(
        broker, 800, 800, lease_id="one-run", deadline=time.monotonic() + 1,
        poll_seconds=.01,
    )
    assert leases is None
    assert error == "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
    assert broker.calls == 0
    assert broker.waits == []


@pytest.mark.parametrize("queue_bytes,writer_bytes", [(0, 1), (1, 0), (0, 0)])
def test_zero_egress_budget_is_never_promoted_even_if_broker_would_grant(
    queue_bytes, writer_bytes
):
    broker = TemporaryPressureBroker(succeed_after=0)
    leases, error = pipeline._acquire_protected_egress_until(
        broker, queue_bytes, writer_bytes, lease_id="one-run",
        deadline=time.monotonic() + 1, poll_seconds=.01,
    )
    assert leases is None
    assert error == "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
    assert broker.calls == 0


def test_nonpositive_hard_cap_is_unknown_and_uses_finite_wait():
    broker = TemporaryPressureBroker(succeed_after=None)
    broker.hard_memory_limit = 0
    leases, error = pipeline._acquire_protected_egress_until(
        broker, 1, 1, lease_id="one-run", deadline=time.monotonic() + .01,
        poll_seconds=.002,
    )
    assert leases is None
    assert error == "RESOURCE_WAIT_EXHAUSTED"
    assert broker.calls > 0
    assert broker.waits


def test_expired_deadline_cannot_grant_a_late_lease():
    broker = TemporaryPressureBroker(succeed_after=0)
    leases, error = pipeline._acquire_protected_egress_until(
        broker, 1, 1, lease_id="one-run", deadline=time.monotonic() - .001,
        poll_seconds=.01,
    )
    assert leases is None
    assert error == "RESOURCE_WAIT_EXHAUSTED"
    assert broker.calls == 0


def test_lease_returned_after_blocking_acquire_deadline_is_released():
    leases = TrackingLease(), TrackingLease()

    class SlowGrantBroker:
        hard_memory_limit = 100

        def acquire_protected_egress(self, *args, **kwargs):
            time.sleep(.01)
            return leases

    granted, error = pipeline._acquire_protected_egress_until(
        SlowGrantBroker(), 1, 1, lease_id="one-run",
        deadline=time.monotonic() + .002, poll_seconds=.001,
    )
    assert granted is None and error == "RESOURCE_WAIT_EXHAUSTED"
    assert all(lease.released for lease in leases)


def test_spurious_resource_wakeup_retains_bounded_poll_cadence():
    broker = TemporaryPressureBroker(succeed_after=None)
    started = time.monotonic()
    leases, error = pipeline._acquire_protected_egress_until(
        broker, 1, 1, lease_id="one-run", deadline=started + .012,
        poll_seconds=.004,
    )
    elapsed = time.monotonic() - started
    assert leases is None and error == "RESOURCE_WAIT_EXHAUSTED"
    assert elapsed >= .01
    assert broker.calls < 10


def test_egress_wait_cancellation_exits_without_execution_attempt():
    from factor_engine.runtime.exceptions import Cancellation, CancellationToken

    token = CancellationToken()

    class CancellingPressure(TemporaryPressureBroker):
        def wait_for_resource_change(self, timeout):
            token.cancel()
            return True

    with pytest.raises(Cancellation):
        pipeline._acquire_protected_egress_until(
            CancellingPressure(succeed_after=None), 1, 1,
            lease_id="cancelled-wait", deadline=time.monotonic() + 1,
            poll_seconds=.01, cancellation_token=token,
        )


def test_precancelled_run_starts_no_worker_or_durable_identity(tmp_path):
    from factor_engine.runtime.exceptions import Cancellation, CancellationToken

    token = CancellationToken()
    token.cancel()
    with pytest.raises(Cancellation):
        pipeline.execute_run_many_durable(
            FakeEngine(), [FakeFactor("never")], policy=resolve_default_policy(),
            artifact_root=tmp_path / "not-created", cancellation_token=token,
        )
    assert not (tmp_path / "not-created").exists()


def test_real_broker_explicit_token_interrupts_long_egress_wait_promptly(tmp_path):
    import threading
    from factor_engine.runtime.auto_memory_budget import AutoMemoryBudget
    from factor_engine.runtime.exceptions import Cancellation, CancellationToken
    from factor_engine.runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(
        hard_memory_limit=1024**3, cpu_slots=1,
        min_host_reserve_gb=0, min_host_reserve_fraction=0,
    )
    broker._refresh_auto_budget = lambda: AutoMemoryBudget(
        hard_memory_limit=1024**3, emergency_reserve=0,
        safe_live_budget=128 * 1024**2, execution_budget=128 * 1024**2,
        safety_factor=.8, measurement_state="TEST_INJECTED",
    )
    # Keep the real FIFO waiter/condition implementation while forcing
    # temporary pressure at its acquisition boundary.
    entered_wait = threading.Event()
    def deny_egress(*args, **kwargs):
        entered_wait.set()
        return None
    broker.acquire_protected_egress = deny_egress
    token = CancellationToken()
    canceller = threading.Thread(target=lambda: (entered_wait.wait(2), token.cancel()))
    canceller.start()
    started = time.monotonic()
    engine = FakeEngine()
    engine.resource_broker = broker
    with pytest.raises(Cancellation) as caught:
        pipeline.execute_run_many_durable(
            engine, [FakeFactor("cancelled")],
            policy=resolve_default_policy({
                "resource_wait_seconds": 5, "sample_interval_seconds": 5,
            }),
            artifact_root=tmp_path, cancellation_token=token,
        )
    canceller.join()
    assert entered_wait.is_set()
    assert time.monotonic() - started < .25
    with sqlite3.connect(json.loads(Path(caught.value.receipt_path).read_text())["state_path"]) as db:
        assert db.execute("select attempts from outcomes where ordinal=0").fetchone() == (0,)


def test_completed_descriptor_wins_cancellation_race_before_prefetch_cleanup():
    from factor_engine.runtime.exceptions import CancellationToken
    from factor_engine.runtime.supervised_worker import WorkerResult

    class Completed:
        done = True
        def result(self, timeout=None):
            return WorkerResult("verified-envelope", "a" * 32)

    class Prefetched:
        cancelled = False
        def cancel_and_retire(self):
            self.cancelled = True

    token = CancellationToken()
    token.cancel()
    prefetched = Prefetched()
    result = pipeline._await_direct_handles(Completed(), prefetched, token)
    assert result.value == "verified-envelope"
    # Parent must validate/persist this descriptor before the wave boundary
    # retires a prefetched slot; cleanup failure cannot erase verified success.
    assert prefetched.cancelled is False


def test_stage_timeout_is_bounded_by_one_job_deadline():
    deadline = time.monotonic() + .02
    first = pipeline._stage_timeout(deadline, 10, "compile")
    assert 0 < first <= .02
    time.sleep(.025)
    try:
        pipeline._stage_timeout(deadline, 10, "write")
    except pipeline.JobDeadlineExceeded as exc:
        assert exc.reason_code == "JOB_DEADLINE_EXCEEDED"
    else:
        raise AssertionError("expired job deadline was reset between stages")


class PipelinePressureBroker(FakeBroker):
    hard_memory_limit = 64 * 1024 * 1024

    def __init__(self, release_after=None):
        self.acquire_calls = 0
        self.release_after = release_after

    def acquire_protected_egress(self, *args, **kwargs):
        self.acquire_calls += 1
        if self.release_after is not None and self.acquire_calls > self.release_after:
            return Lease(), Lease()
        return None

    def wait_for_resource_change(self, timeout):
        time.sleep(timeout)


def test_pipeline_temporary_egress_pressure_does_not_consume_execution_attempt(tmp_path):
    broker = PipelinePressureBroker(release_after=1)
    engine = FakeEngine()
    engine.resource_broker = broker
    receipt = pipeline.execute_run_many_durable(
        engine, [FakeFactor("alpha")], policy=resolve_default_policy({
            "resource_wait_seconds": .1, "sample_interval_seconds": .002,
        }), artifact_root=tmp_path,
    )
    assert receipt["status"] == "SUCCEEDED"
    assert broker.acquire_calls == 2
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute("select attempts from outcomes where ordinal=0").fetchone() == (1,)


def test_pipeline_permanent_egress_pressure_has_finite_failed_terminal(tmp_path):
    broker = PipelinePressureBroker()
    engine = FakeEngine()
    engine.resource_broker = broker
    started = time.monotonic()
    receipt = pipeline.execute_run_many_durable(
        engine, [FakeFactor("alpha")], policy=resolve_default_policy({
            "resource_wait_seconds": .02, "sample_interval_seconds": .002,
        }), artifact_root=tmp_path,
    )
    assert time.monotonic() - started < 1
    assert receipt["counts"] == {"FAILED": 1}
    assert receipt["error_groups"][0]["code"] == "RESOURCE_WAIT_EXHAUSTED"
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute("select attempts from outcomes where ordinal=0").fetchone() == (0,)


def test_one_job_deadline_aborts_slow_compile_and_persists_terminals(tmp_path):
    class SlowCompileEngine(FakeEngine):
        def compile(self, factor):
            time.sleep(1)
            return factor

    with pytest.raises(TimeoutError) as caught:
        pipeline.execute_run_many_durable(
            SlowCompileEngine(), [FakeFactor("alpha")],
            policy=resolve_default_policy({
                "job_unknown_seconds": .05,
                "job_min_seconds": .01,
                "job_max_seconds": .05,
                "compute_unknown_seconds": 1,
                "input_ingestion_deadline_seconds": .5,
                "cooperative_cancel_grace_seconds": .01,
                "worker_exit_observation_seconds": .1,
            }), artifact_root=tmp_path,
        )
    persisted = json.loads(Path(caught.value.receipt_path).read_text())
    assert persisted["status"] == "ABORTED"
    assert persisted["counts"] == {"CANCELLED": 1}
    with sqlite3.connect(persisted["state_path"]) as db:
        assert db.execute("select attempts from outcomes where ordinal=0").fetchone() == (0,)


def test_one_job_deadline_is_not_reset_between_multiple_waves(tmp_path):
    class SlowWaveEngine(FakeEngine):
        def run_many_parallel(self, factors, **kwargs):
            time.sleep(.035)
            return super().run_many_parallel(factors, **kwargs)

    with pytest.raises(TimeoutError) as caught:
        pipeline.execute_run_many_durable(
            SlowWaveEngine(), [FakeFactor("first"), FakeFactor("second")],
            policy=resolve_default_policy({
                "initial_lookahead_factors": 1,
                "job_unknown_seconds": .06,
                "job_min_seconds": .01,
                "job_max_seconds": .06,
                "compute_unknown_seconds": 1,
                "input_ingestion_deadline_seconds": .5,
                "cooperative_cancel_grace_seconds": .01,
                "worker_exit_observation_seconds": .1,
            }), artifact_root=tmp_path,
        )
    persisted = json.loads(Path(caught.value.receipt_path).read_text())
    assert persisted["status"] == "ABORTED"
    assert sum(persisted["counts"].values()) == 2
    assert persisted["counts"].get("CANCELLED", 0) >= 1
