from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from factor_engine.runtime.job_scoped_lease import (
    acquire_job_scoped_memory,
    reserve_job_scoped_task,
)


class _Lease:
    def __init__(self, release_fn):
        self._release_fn = release_fn
        self.released = False

    def release(self):
        if not self.released:
            self.released = True
            self._release_fn()


class _Broker:
    def __init__(self, cap):
        self.cap = cap
        self.live = 0
        self.fail = False

    def _take(self, n):
        if self.fail or self.live + n > self.cap:
            return None
        self.live += n
        return _Lease(lambda: setattr(self, "live", self.live - n))

    def acquire_memory(self, kind, nbytes, *, lease_id):
        return self._take(nbytes)

    def try_reserve(self, contract, *, task_id):
        return self._take(contract.admissible_peak_bytes)


class _Job:
    def __init__(self, lease_id, cap):
        self.lease_id = lease_id
        self.memory_bytes = cap
        self.released = False
        self.live = 0
        self.children = set()

    def request_child(self, *, owner, kind, memory_bytes, **kwargs):
        if self.released or self.live + memory_bytes > self.memory_bytes:
            return None
        child = _Child(self.lease_id, memory_bytes)
        self.live += memory_bytes
        self.children.add(id(child))
        return child

    def release_child(self, child):
        if id(child) not in self.children:
            return False
        self.children.remove(id(child))
        self.live -= child.memory_bytes
        return True


@dataclass
class _Child:
    parent_lease_id: str
    memory_bytes: int


@dataclass
class _Contract:
    admissible_peak_bytes: int


def test_same_job_wave_plus_task_cannot_exceed_job_cap():
    broker = _Broker(32)
    job = _Job("job-a", 8)
    wave = acquire_job_scoped_memory(
        broker, job, "READ_WAVE", 4, lease_id="job-a:s:wave:1"
    )
    assert wave is not None
    task = reserve_job_scoped_task(
        broker, job, _Contract(5), task_id="job-a:s:task:1"
    )
    assert task is None
    assert job.live == 4 and broker.live == 4
    wave.release()


def test_two_jobs_share_global_broker_without_cross_scope_accounting():
    broker = _Broker(8)
    a, b = _Job("job-a", 8), _Job("job-b", 8)
    la = acquire_job_scoped_memory(broker, a, "READ_WAVE", 4, lease_id="a:w")
    lb = acquire_job_scoped_memory(broker, b, "READ_WAVE", 4, lease_id="b:w")
    assert la is not None and lb is not None and broker.live == 8
    rejected = reserve_job_scoped_task(
        broker, a, _Contract(1), task_id="a:t"
    )
    assert rejected is None
    assert a.live == 4  # broker rejection rolled back only A's new child
    assert b.live == 4
    la.release()
    lb.release()


def test_broker_failure_rolls_back_child_and_release_is_idempotent():
    broker = _Broker(8)
    job = _Job("job-a", 8)
    broker.fail = True
    assert acquire_job_scoped_memory(
        broker, job, "READ_WAVE", 4, lease_id="a:w"
    ) is None
    assert job.live == 0 and broker.live == 0
    broker.fail = False
    lease = reserve_job_scoped_task(
        broker, job, _Contract(4), task_id="a:t"
    )
    assert lease is not None
    lease.release()
    lease.release()
    assert job.live == 0 and broker.live == 0


def test_explicit_job_object_does_not_depend_on_contextvar_propagation():
    broker = _Broker(8)
    job = _Job("job-a", 4)
    lease = reserve_job_scoped_task(
        broker, job, _Contract(4), task_id="job-a:s:task:worker"
    )
    assert lease is not None
    job.released = True
    assert reserve_job_scoped_task(
        broker, job, _Contract(1), task_id="job-a:s:task:late"
    ) is None
    lease.release()


class _CoordinatorBroker:
    hard_cpu_slots = 8

    def resource_envelope(self):
        return SimpleNamespace(
            safe_memory_bytes=8, target_cpu_tokens=8,
            io_capacity_score=1.0, spill_free_bytes=0,
        )

    def cpu_budget(self):
        return 8


def test_real_coordinator_enforces_combined_job_children_and_prunes_ids(monkeypatch):
    import factor_engine.runtime.resource_broker as broker_module
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    raw = _CoordinatorBroker()
    monkeypatch.setattr(broker_module, "require_broker", lambda broker, **kwargs: broker)
    coordinator = HostResourceCoordinator(broker=raw)
    job = coordinator.request_job_lease(owner="job:a", memory_bytes=8)
    assert job is not None
    wave = job.request_child(owner="wave", memory_bytes=4)
    assert wave is not None
    assert job.remaining_memory_bytes == 4
    assert job.request_child(owner="task", memory_bytes=5) is None
    assert job.release_child(wave)
    assert job.remaining_memory_bytes == 8
    root = coordinator._leases[job.lease_id]
    assert root.child_lease_ids == []


def test_real_job_context_is_explicitly_bound_in_executor_thread(monkeypatch):
    import factor_engine.runtime.resource_broker as broker_module
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    raw = _CoordinatorBroker()
    monkeypatch.setattr(broker_module, "require_broker", lambda broker, **kwargs: broker)
    coordinator = HostResourceCoordinator(broker=raw)
    job = coordinator.request_job_lease(owner="job:a", memory_bytes=8)
    assert job is not None

    def worker():
        assert coordinator.active_job_lease() is None
        with job.bind_context():
            return coordinator.active_job_lease() is job

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(worker).result() is True
    assert coordinator.active_job_lease() is None


def test_job_root_close_waits_for_live_child(monkeypatch):
    import factor_engine.runtime.resource_broker as broker_module
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    raw = _CoordinatorBroker()
    monkeypatch.setattr(broker_module, "require_broker", lambda broker, **kwargs: broker)
    coordinator = HostResourceCoordinator(broker=raw)
    job = coordinator.request_job_lease(owner="job:a", memory_bytes=8)
    child = job.request_child(owner="late-future", memory_bytes=4)
    job.release()
    assert job.lease_id in coordinator._leases
    assert job.request_child(owner="too-late", memory_bytes=1) is None
    assert job.release_child(child)
    assert job.lease_id not in coordinator._leases


def test_scheduler_submit_binds_captured_job_inside_real_thread(monkeypatch):
    import factor_engine.runtime.resource_broker as broker_module
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    raw = _CoordinatorBroker()
    monkeypatch.setattr(broker_module, "require_broker", lambda broker, **kwargs: broker)
    coordinator = HostResourceCoordinator(broker=raw)
    job = coordinator.request_job_lease(owner="job:a", memory_bytes=8)

    class _Executor:
        prefer_process = False

        def submit(self, backend, fn, *args, **kwargs):
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(fn, *args)
            future.add_done_callback(lambda _: pool.shutdown(wait=False))
            return future

    scheduler = object.__new__(AdaptiveBatchScheduler)
    scheduler.job_lease = job
    scheduler.executor = _Executor()
    scheduler._execution_policy = None
    future = scheduler._submit(
        "pandas", lambda: coordinator.active_job_lease() is job
    )
    assert future.result() is True
    assert coordinator.active_job_lease() is None


def test_materialize_zero_budget_fails_closed_before_starting_writer():
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    class _ZeroBroker:
        def resource_decision(self, **kwargs):
            raise RuntimeError("no decision")

        def current_sink_budget(self):
            return 0

    scheduler = object.__new__(AdaptiveBatchScheduler)
    scheduler.broker = _ZeroBroker()
    scheduler.job_lease = None
    try:
        scheduler.materialize(
            object(), backend=object(), ctx=object(), writer=lambda items: None
        )
    except ValueError as exc:
        assert "queue_bytes total budget must be positive" in str(exc)
    else:
        raise AssertionError("zero sink budget must fail before writer startup")
