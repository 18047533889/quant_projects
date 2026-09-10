from __future__ import annotations

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline


def _broker():
    from factor_engine.runtime.auto_memory_budget import AutoMemoryBudget
    from factor_engine.runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=4,
        min_host_reserve_gb=0,
        min_host_reserve_fraction=0,
    )
    broker._refresh_auto_budget = lambda: AutoMemoryBudget(
        hard_memory_limit=8 * 1024**3,
        emergency_reserve=0,
        safe_live_budget=1024**3,
        execution_budget=1024**3,
        safety_factor=0.8,
        measurement_state="TEST_INJECTED",
    )
    return broker


def test_parent_broker_ipc_captures_active_job_lease(monkeypatch):
    from factor_engine.runtime import host_resource_coordinator as coordinator_module
    from factor_engine.runtime import resource_broker_ipc as ipc_module

    job_lease = object()
    captured = {}

    class CapturingIPC:
        def __init__(self, broker, *, rpc_timeout_seconds, job_lease):
            captured.update(
                broker=broker,
                rpc_timeout_seconds=rpc_timeout_seconds,
                job_lease=job_lease,
            )

    monkeypatch.setattr(coordinator_module, "get_active_job_lease", lambda: job_lease)
    monkeypatch.setattr(ipc_module, "ParentBrokerIPC", CapturingIPC)
    broker = object()
    created = pipeline._create_parent_broker_ipc(broker, rpc_timeout_seconds=3.5)
    assert isinstance(created, CapturingIPC)
    assert captured == {
        "broker": broker,
        "rpc_timeout_seconds": 3.5,
        "job_lease": job_lease,
    }


def test_parent_broker_ipc_preserves_standalone_none_scope(monkeypatch):
    from factor_engine.runtime import host_resource_coordinator as coordinator_module
    from factor_engine.runtime import resource_broker_ipc as ipc_module

    captured = {}

    class CapturingIPC:
        def __init__(self, _broker, *, rpc_timeout_seconds, job_lease):
            captured.update(timeout=rpc_timeout_seconds, job_lease=job_lease)

    monkeypatch.setattr(coordinator_module, "get_active_job_lease", lambda: None)
    monkeypatch.setattr(ipc_module, "ParentBrokerIPC", CapturingIPC)
    pipeline._create_parent_broker_ipc(object(), rpc_timeout_seconds=2.0)
    assert captured == {"timeout": 2.0, "job_lease": None}


def test_real_ipc_protected_egress_remains_charged_until_each_remote_release():
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    mib = 1024**2
    broker = _broker()
    coordinator = HostResourceCoordinator(broker=broker)
    job = coordinator.request_job_lease(
        owner="bounded-pipeline-ipc-test",
        memory_bytes=48 * mib,
        cpu_tokens=1,
    )
    assert job is not None

    with job.bind_context():
        ipc = pipeline._create_parent_broker_ipc(broker, rpc_timeout_seconds=2.0)
    try:
        proxy = ipc.create_proxy()
        egress = proxy.acquire_protected_egress(
            16 * mib,
            16 * mib,
            lease_id="bounded-pipeline-egress",
        )
        assert egress is not None
        result_queue_lease, writer_batch_lease = egress
        assert job.remaining_memory_bytes == 16 * mib

        assert proxy.acquire_memory(
            "WRITER_BATCH", 17 * mib, lease_id="over-job-capacity"
        ) is None
        assert job.remaining_memory_bytes == 16 * mib

        result_queue_lease.release()
        # Protected egress is one atomic child reservation.  Releasing only
        # half must not return capacity while the other physical lease lives.
        assert job.remaining_memory_bytes == 16 * mib
        writer_batch_lease.release()
        assert job.remaining_memory_bytes == 48 * mib

        job.release()
        assert proxy.acquire_memory(
            "WRITER_BATCH", mib, lease_id="closed-job"
        ) is None
    finally:
        ipc.close()
        job.release()


def test_active_job_rejects_missing_factory_before_workers_or_durable_writes(
    tmp_path, monkeypatch,
):
    from factor_engine.runtime import host_resource_coordinator as coordinator_module
    from factor_engine.runtime.default_engine import DeploymentConfigurationError
    from factor_engine.runtime.default_execution_policy import resolve_default_policy
    from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker, FakeEngine

    effects = {"iterated": False, "worker_started": False}

    def factors():
        effects["iterated"] = True
        raise AssertionError("factor ingestion must not start")
        yield

    class ForbiddenWorker:
        def __init__(self, *args, **kwargs):
            effects["worker_started"] = True
            raise AssertionError("worker must not start")

    monkeypatch.setattr(coordinator_module, "get_active_job_lease", lambda: object())
    monkeypatch.setattr(pipeline, "SupervisedReusableWorker", ForbiddenWorker)
    artifact_root = tmp_path / "must-not-be-created"

    with pytest.raises(DeploymentConfigurationError) as caught:
        pipeline.execute_run_many_durable(
            FakeEngine(), factors(),
            policy=resolve_default_policy(),
            artifact_root=artifact_root,
            run_kwargs={"broker": FakeBroker()},
            engine_factory=None,
            sink=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("sink must not write")
            ),
        )

    assert caught.value.reason_code == "DEPLOYMENT_CONFIGURATION_REQUIRED"
    assert caught.value.missing_fields == ("engine_factory",)
    assert effects == {"iterated": False, "worker_started": False}
    assert not artifact_root.exists()
