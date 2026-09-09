from __future__ import annotations

import time
import threading

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.finite_manifest import FiniteFactorManifest, ManifestInputTimeout
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker, _build_spawn_engine
from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
from factor_engine.runtime.resource_broker import ResourceBroker


class Lease:
    def __init__(self): self.released = False
    def release(self): self.released = True


class SequenceBroker:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
    def acquire_memory(self, *_args, **_kwargs):
        self.calls += 1
        return self.responses.pop(0) if self.responses else None


class Cancellation:
    def __init__(self): self.checks = 0
    def raise_if_cancelled(self): self.checks += 1


class SlowNameFactor:
    def __init__(self, ordinal): self.ordinal = ordinal
    @property
    def name(self):
        time.sleep(0.06)
        return f"slow-{self.ordinal}"


def test_temporary_none_retries_same_broker_then_real_spawn_releases(tmp_path):
    lease = Lease()
    broker = SequenceBroker([None, None, lease])
    token = Cancellation()
    view = pipeline._DeadlineBoundMemoryBroker(
        broker, deadline=time.monotonic() + 1, cancellation_token=token,
    )
    manifest = FiniteFactorManifest.ingest_supervised(
        [{"name": "alpha"}], tmp_path / "manifest.sqlite3",
        process_context="spawn", deadline_seconds=2,
        serialization_broker=view,
    )
    try:
        assert manifest.input_complete and len(manifest) == 1
    finally:
        manifest.close()
    assert broker.calls == 3
    assert token.checks == 4
    assert lease.released


def test_sustained_none_has_finite_typed_exit_before_process(tmp_path):
    broker = SequenceBroker([])
    view = pipeline._DeadlineBoundMemoryBroker(
        broker, deadline=time.monotonic() + 0.03, cancellation_token=None,
    )
    started = time.monotonic()
    with pytest.raises(pipeline.JobDeadlineExceeded, match="job deadline"):
        FiniteFactorManifest.ingest_supervised(
            [{"name": "alpha"}], tmp_path / "manifest.sqlite3",
            process_context="spawn", deadline_seconds=2,
            serialization_broker=view,
        )
    assert 0.02 <= time.monotonic() - started < 0.5


def test_cancellation_interrupts_retry_without_starting_process(tmp_path):
    broker = SequenceBroker([])

    class Cancelled(RuntimeError): pass
    class CancelOnSecondCheck:
        def __init__(self): self.checks = 0
        def raise_if_cancelled(self):
            self.checks += 1
            if self.checks == 2: raise Cancelled("cancelled")

    token = CancelOnSecondCheck()
    view = pipeline._DeadlineBoundMemoryBroker(
        broker, deadline=time.monotonic() + 1, cancellation_token=token,
    )
    with pytest.raises(Cancelled):
        FiniteFactorManifest.ingest_supervised(
            [{"name": "alpha"}], tmp_path / "manifest.sqlite3",
            process_context="spawn", deadline_seconds=2,
            serialization_broker=view,
        )
    assert broker.calls == 1


def test_expired_deadline_rejects_before_broker_call():
    broker = SequenceBroker([Lease()])
    view = pipeline._DeadlineBoundMemoryBroker(
        broker, deadline=time.monotonic() - 1, cancellation_token=None,
    )
    with pytest.raises(pipeline.JobDeadlineExceeded):
        view.acquire_memory("MANIFEST_BUFFER", 1, lease_id="expired")
    assert broker.calls == 0


def test_resource_wait_expiry_is_not_misreported_as_job_deadline():
    now = time.monotonic()
    view = pipeline._DeadlineBoundMemoryBroker(
        SequenceBroker([]), job_deadline=now + 10, input_deadline=now + 5,
        resource_deadline=now + 0.02, cancellation_token=None,
    )
    with pytest.raises(pipeline.ResourceWaitExhausted) as caught:
        view.acquire_memory("MANIFEST_BUFFER", 1, lease_id="resource-first")
    assert caught.value.reason_code == "RESOURCE_WAIT_EXHAUSTED"
    assert not isinstance(caught.value, pipeline.JobDeadlineExceeded)


def test_true_job_deadline_expiry_keeps_job_reason():
    now = time.monotonic()
    view = pipeline._DeadlineBoundMemoryBroker(
        SequenceBroker([]), job_deadline=now + 0.02, input_deadline=now + 5,
        resource_deadline=now + 10, cancellation_token=None,
    )
    with pytest.raises(pipeline.JobDeadlineExceeded) as caught:
        view.acquire_memory("MANIFEST_BUFFER", 1, lease_id="job-first")
    assert caught.value.reason_code == "JOB_DEADLINE_EXCEEDED"


def test_input_deadline_expiry_has_input_reason():
    now = time.monotonic()
    view = pipeline._DeadlineBoundMemoryBroker(
        SequenceBroker([]), job_deadline=now + 10, input_deadline=now + 0.02,
        resource_deadline=now + 5, cancellation_token=None,
    )
    with pytest.raises(ManifestInputTimeout) as caught:
        view.acquire_memory("MANIFEST_BUFFER", 1, lease_id="input-first")
    assert caught.value.reason_code == "INPUT_INGESTION_TIMEOUT"
    assert isinstance(caught.value, TimeoutError)


def test_late_grant_is_released_and_rejected():
    lease = Lease()

    class LateBroker(SequenceBroker):
        def acquire_memory(self, *args, **kwargs):
            time.sleep(0.02)
            return super().acquire_memory(*args, **kwargs)

    broker = LateBroker([lease])
    view = pipeline._DeadlineBoundMemoryBroker(
        broker, deadline=time.monotonic() + 0.01, cancellation_token=None,
    )
    with pytest.raises(pipeline.JobDeadlineExceeded):
        view.acquire_memory("MANIFEST_BUFFER", 1, lease_id="late")
    assert lease.released


def test_cancelled_grant_is_released_and_original_cancel_propagates():
    lease = Lease()
    class Cancelled(RuntimeError): pass
    class Token:
        cancelled = False
        def raise_if_cancelled(self):
            if self.cancelled: raise Cancelled("cancelled after grant")
    token = Token()
    class CancellingBroker(SequenceBroker):
        def acquire_memory(self, *args, **kwargs):
            result = super().acquire_memory(*args, **kwargs)
            token.cancelled = True
            return result
    view = pipeline._DeadlineBoundMemoryBroker(
        CancellingBroker([lease]), deadline=time.monotonic() + 1,
        cancellation_token=token,
    )
    with pytest.raises(Cancelled, match="after grant"):
        view.acquire_memory("MANIFEST_BUFFER", 1, lease_id="cancel")
    assert lease.released


def test_late_grant_release_failure_retains_exact_lease():
    class FailingLease(Lease):
        def release(self): raise RuntimeError("release failed")
    lease = FailingLease()
    class LateBroker(SequenceBroker):
        def acquire_memory(self, *args, **kwargs):
            time.sleep(0.02)
            return super().acquire_memory(*args, **kwargs)
    view = pipeline._DeadlineBoundMemoryBroker(
        LateBroker([lease]), deadline=time.monotonic() + 0.01,
        cancellation_token=None,
    )
    with pytest.raises(pipeline.JobDeadlineExceeded) as caught:
        view.acquire_memory("MANIFEST_BUFFER", 1, lease_id="late-release-fail")
    assert caught.value.cleanup_pending is True
    assert caught.value.serialization_buffer_lease is lease
    assert "release failed" in str(caught.value.cleanup_errors[0])


def test_admission_wait_and_child_join_share_one_absolute_deadline(tmp_path):
    lease = Lease()
    broker = SequenceBroker([None, None, lease])
    absolute = time.monotonic() + 0.16
    view = pipeline._DeadlineBoundMemoryBroker(
        broker, deadline=absolute, cancellation_token=None,
    )
    manifest = FiniteFactorManifest.ingest_supervised(
        [SlowNameFactor(i) for i in range(3)], tmp_path / "manifest.sqlite3",
        process_context="spawn", deadline_seconds=5,
        deadline_monotonic=absolute, serialization_broker=view,
    )
    try:
        assert not manifest.input_complete
        assert len(manifest) < 3
        assert "deadline exceeded" in manifest.input_error
    finally:
        manifest.close()
    assert lease.released


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_factors": 1.5}, {"max_factors": True},
        {"max_definition_bytes": 1.5}, {"max_definition_bytes": True},
        {"deadline_seconds": float("nan")}, {"deadline_seconds": True},
    ],
)
def test_standalone_limits_reject_before_database_open(tmp_path, kwargs):
    path = tmp_path / "manifest.sqlite3"
    with pytest.raises(ValueError):
        FiniteFactorManifest.ingest([], path, **kwargs)
    assert not path.exists()


def test_default_factory_pipeline_passes_shared_broker_to_real_spawn(
    tmp_path, monkeypatch,
):
    shared = FakeBroker()
    observed = {}
    original = FiniteFactorManifest.ingest_supervised

    def inspect(*args, **kwargs):
        observed["serialization_broker"] = kwargs.get("serialization_broker")
        observed["deadline_monotonic"] = kwargs.get("deadline_monotonic")
        return original(*args, **kwargs)

    monkeypatch.setattr(FiniteFactorManifest, "ingest_supervised", inspect)
    receipt = pipeline.execute_run_many_durable(
        None, [], policy=resolve_default_policy(),
        artifact_root=tmp_path / "artifacts",
        run_kwargs={"broker": shared},
        engine_factory=_build_spawn_engine, engine_factory_config={},
    )
    assert receipt["status"] == "SUCCEEDED"
    assert isinstance(observed["serialization_broker"], pipeline._DeadlineBoundMemoryBroker)
    assert observed["serialization_broker"]._broker is shared
    assert observed["deadline_monotonic"] is not None


def test_manifest_wait_never_enters_or_blocks_real_broker_egress_fifo(monkeypatch):
    broker = ResourceBroker(hard_memory_limit=100, cpu_slots=2)
    monkeypatch.setattr(broker, "execution_budget", lambda: 100)
    blocker = broker.acquire_memory(
        MemoryLeaseKind.CSE_CACHE, 80, lease_id="blocker"
    )
    assert blocker is not None
    acquired = []

    def wait_for_manifest():
        view = pipeline._DeadlineBoundMemoryBroker(
            broker, deadline=time.monotonic() + 1, cancellation_token=None,
        )
        acquired.append(view.acquire_memory(
            MemoryLeaseKind.MANIFEST_BUFFER, 30, lease_id="manifest"
        ))

    thread = threading.Thread(target=wait_for_manifest)
    thread.start()
    time.sleep(0.06)
    assert not broker._protected_egress_waiters
    with broker.protected_egress_waiter("writer", time.monotonic() + 0.5):
        writer = broker.acquire_protected_egress(5, 5, lease_id="writer")
    assert writer is not None
    for lease in writer:
        lease.release()
    blocker.release()
    thread.join(1)
    assert not thread.is_alive()
    assert len(acquired) == 1
    acquired[0].release()
