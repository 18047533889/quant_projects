"""Small synthetic regression tests: no market data or large allocations."""
import gc
import weakref
from types import SimpleNamespace

import pytest

from factor_engine.runtime.batch_service import (
    _consume_bounded_roots,
    execute_run_many,
    execute_run_many_parallel,
)
from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler


@pytest.mark.parametrize("entry", [execute_run_many, execute_run_many_parallel])
@pytest.mark.parametrize("policy,sink", [("sink", None), ("sink", object()), ("typo", None)])
def test_invalid_policy_rejected_before_compilation(entry, policy, sink):
    with pytest.raises(ValueError):
        entry(object(), [], result_policy=policy, sink=sink)


def test_bounded_roots_release_results_and_backpressure():
    refs = []
    consumed = []
    produced = 0

    class Payload:
        pass

    def items():
        nonlocal produced
        for index in range(1000):
            produced += 1
            yield index

    def execute(index):
        payload = Payload()
        refs.append(weakref.ref(payload))
        return index, payload

    def consume(index, payload):
        # Input is not exhausted up front; even an arbitrarily slow sink leaves
        # at most the configured concurrency outstanding.
        assert produced - len(consumed) <= 3
        assert sum(ref() is not None for ref in refs) <= 3
        consumed.append(index)

    _consume_bounded_roots(items(), execute, consume, max_workers=3)
    gc.collect()
    assert sorted(consumed) == list(range(1000))
    assert all(ref() is None for ref in refs)


def test_sink_failure_stops_admission():
    produced = []

    def items():
        for index in range(1000):
            produced.append(index)
            yield index

    def fail(*args):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _consume_bounded_roots(items(), lambda i: (i,), fail, max_workers=2)
    assert len(produced) == 2


def serial_scheduler():
    # Isolate result ownership from the host resource coordinator: the actual
    # serial scheduler method is exercised with a one-root physical DAG.
    scheduler = object.__new__(AdaptiveBatchScheduler)
    scheduler.sink = None
    scheduler._results = {}
    scheduler._task_timing = {}
    scheduler._explanations = []
    scheduler._wave_summary = {}
    scheduler.broker = SimpleNamespace(summary=lambda: {})
    scheduler.executor = SimpleNamespace(summary=lambda: {})
    scheduler._execute_read_waves = lambda *a, **k: None
    scheduler._record_timing = lambda *a: None
    scheduler._record_task_calibration = lambda *a: None
    scheduler._release_consumed = lambda *a: None
    task = SimpleNamespace(task_type="ROOT", factor_name="factor")
    dag = SimpleNamespace(tasks={"root": task}, topological_order=lambda: ["root"])
    return scheduler, dag


def test_serial_sink_does_not_retain_results():
    scheduler, dag = serial_scheduler()
    written = []
    sink = SimpleNamespace(
        submit=lambda name, result: written.append((name, result)) or True,
        finish=lambda: None,
    )
    output = scheduler.run_serial_fused(
        dag, backend=None, ctx=None, execute_root=lambda task: 42, sink=sink
    )
    assert written == [("factor", 42)]
    assert output["results"] == {}


def test_serial_sink_rejection_aborts():
    scheduler, dag = serial_scheduler()
    sink = SimpleNamespace(submit=lambda *a: False, finish=lambda: None)
    with pytest.raises(RuntimeError, match="sink.submit returned False"):
        scheduler.run_serial_fused(
            dag, backend=None, ctx=None, execute_root=lambda task: 42, sink=sink
        )


def test_serial_without_sink_still_returns_results():
    scheduler, dag = serial_scheduler()
    output = scheduler.run_serial_fused(
        dag, backend=None, ctx=None, execute_root=lambda task: 42
    )
    assert output["results"] == {"factor": 42}


@pytest.mark.parametrize("micro_batch", [False, True])
def test_parallel_sink_does_not_retain_results(monkeypatch, micro_batch):
    from factor_engine.planner.physical_factor_dag import PhysicalFactorDAG, PhysicalFactorTask
    from factor_engine.runtime.resource_broker import ResourceBroker
    from factor_engine.runtime.task_resource_contract import TaskResourceContract
    import factor_engine.runtime.adaptive_batch_scheduler as module

    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    if not micro_batch:
        monkeypatch.setattr(module, "_MICRO_BATCH_MIN_ROOTS", 10**9)
    dag = PhysicalFactorDAG()
    for i in range(20):
        dag.add_task(PhysicalFactorTask(
            task_id=f"root:{i}", factor_name=str(i), op="add", task_type="ROOT",
            preferred_backend="pandas_numpy", source_scope="mock", execution_scope="all",
            resource_contract=TaskResourceContract(
                predicted_elapsed_ms=1, peak_memory_bytes=1024, output_bytes=1024,
                cpu_tokens=1, backend_threads=1,
            ),
        ))
    dag.roots = tuple(dag.tasks)
    scheduler = AdaptiveBatchScheduler(
        broker=ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=2,
                              min_host_reserve_gb=1, min_host_reserve_fraction=0.05),
        max_concurrency=2,
    )
    written = {}
    sink = SimpleNamespace(
        submit=lambda name, result: written.__setitem__(name, result) or True,
        finish=lambda: None, backpressure_ratio=0.0, fatal_error=None,
        set_target_bytes=lambda value: None,
    )
    output = scheduler.run(
        dag, backend=None,
        ctx=SimpleNamespace(run_mode="research", runtime_stats={}, shared_result_cache={}),
        execute_root=lambda task: int(task.factor_name), sink=sink,
    )
    assert written == {str(i): i for i in range(20)}
    assert output["results"] == {}
    assert (output["scheduler_stats"]["micro_batch_task_count"] > 0) is micro_batch
    scheduler.executor.shutdown()


def test_worker_replan_preserves_legacy_resource_plan_contract():
    from factor_engine.runtime.batch_service import _replan_duckdb_threads
    from factor_engine.runtime.execution_resources import ResourcePlan

    original = ResourcePlan(n_jobs=8, duckdb_threads=4, polars_threads=4,
                            io_concurrency=2, memory_budget_gb=4,
                            total_runnable=32, memory_budget_bytes=4 * 1024**3)
    plan = _replan_duckdb_threads(original, SimpleNamespace(cpu_budget=4, workers=2))
    assert plan.n_jobs == 2
    assert plan.duckdb_threads == plan.polars_threads == 2
    assert plan.total_runnable == 4
    assert plan.memory_budget_bytes == original.memory_budget_bytes


def test_scheduler_executor_respects_concurrency():
    from factor_engine.runtime.resource_broker import ResourceBroker

    scheduler = AdaptiveBatchScheduler(
        broker=ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=2),
        max_concurrency=2,
    )
    assert scheduler.executor.max_thread_workers == 2
    assert scheduler.executor.max_process_workers == 2


@pytest.mark.parametrize("error_type,expected_attempts", [(ConnectionError, 2), (RuntimeError, 1)])
def test_microbatch_future_failure_has_bounded_retries(monkeypatch, error_type, expected_attempts):
    import factor_engine.runtime.adaptive_batch_scheduler as module
    from factor_engine.tests.r39.test_perf_scheduler_2026_08 import _bare_dag, _Ctx, _make_scheduler

    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    attempts = []

    def failing_dispatch(*args):
        attempts.append(1)
        raise error_type("batch transport failure")

    monkeypatch.setattr(module, "dispatch_micro_batch", failing_dispatch)
    scheduler = _make_scheduler(cpu_slots=2)
    try:
        with pytest.raises(error_type, match="batch transport failure"):
            scheduler.run(_bare_dag(20), backend=None, ctx=_Ctx(),
                          execute_root=lambda task: 42)
        assert len(attempts) == expected_attempts
    finally:
        scheduler.executor.shutdown()


def test_scope_byte_limits_are_accepted_by_real_duckdb(monkeypatch):
    import os
    import duckdb
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan, ExecutionResourceScope

    scope = ExecutionResourceScope(ExecutionResourcePlan.auto(max_workers=1))
    monkeypatch.setattr(scope, "_apply_live_pragma", lambda threads: None)
    monkeypatch.setenv("DUCKDB_MEMORY_LIMIT", "123MB")
    with scope:
        conn = duckdb.connect(":memory:")
        try:
            conn.execute("SET memory_limit = ?", [os.environ["DUCKDB_MEMORY_LIMIT"]])
            conn.execute("SET max_temp_directory_size = ?", [os.environ["DUCKDB_MAX_TEMP_DIRECTORY_SIZE"]])
        finally:
            conn.close()
    assert os.environ["DUCKDB_MEMORY_LIMIT"] == "123MB"


def test_scope_failed_enter_restores_environment(monkeypatch):
    import os
    from factor_engine.runtime.resource_governor import ExecutionResourceScope

    scope = ExecutionResourceScope(strict=True)
    monkeypatch.setenv("DUCKDB_MAX_THREADS", "2")
    original = {key: os.environ.get(key) for key in (
        "DUCKDB_MAX_THREADS", "POLARS_MAX_THREADS", "DUCKDB_MEMORY_LIMIT",
        "DUCKDB_TEMP_DIRECTORY", "DUCKDB_MAX_TEMP_DIRECTORY_SIZE",
    )}

    def fail(threads):
        raise RuntimeError("apply failed")

    monkeypatch.setattr(scope, "_apply_live_pragma", fail)
    with pytest.raises(RuntimeError, match="apply failed"):
        scope.__enter__()
    assert {key: os.environ.get(key) for key in original} == original


def test_microbatch_sink_failure_releases_lease(monkeypatch):
    from factor_engine.tests.r39.test_perf_scheduler_2026_08 import _bare_dag, _Ctx, _make_scheduler

    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    scheduler = _make_scheduler(cpu_slots=4)

    def fail(name, result):
        raise OSError("writer failed")

    try:
        with pytest.raises(OSError, match="writer failed"):
            scheduler.run(_bare_dag(20), backend=None, ctx=_Ctx(),
                          execute_root=lambda task: 42, result_handler=fail)
    finally:
        scheduler.executor.shutdown()
    assert scheduler.broker.summary()["running_tasks"] == 0
    assert scheduler.broker.summary()["cpu"]["in_use"] == 0


@pytest.mark.parametrize("repeat", range(5))
def test_sink_failure_releases_inflight_only_after_completion(monkeypatch, repeat):
    import threading
    import factor_engine.runtime.adaptive_batch_scheduler as module
    from factor_engine.tests.r39.test_perf_scheduler_2026_08 import _bare_dag, _Ctx, _make_scheduler

    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    monkeypatch.setattr(module, "_MICRO_BATCH_MIN_ROOTS", 10**9)
    scheduler = _make_scheduler(cpu_slots=4)
    scheduler.max_concurrency = 2
    # This is a lease-lifecycle oracle, not a live host-controller test. A
    # previously started global autopilot may legitimately recommend one task;
    # then these two intentionally interdependent test roots cannot both start.
    # Fix external resource signals while retaining real token admission,
    # ReservationLease objects, Future completion and cleanup.
    from factor_engine.runtime.resource_broker import ResourceSnapshot
    from factor_engine.runtime.resource_autopilot import ResourceDecision
    snapshot = ResourceSnapshot(
        timestamp_ms=0, hard_cpu_slots=2, system_cpu_util=0, our_cpu_util=0,
        external_cpu_util=0, loadavg=0, hard_memory_limit=8 * 1024**3,
        cgroup_memory_current=None, host_mem_available=8 * 1024**3,
        process_rss=0, worker_rss=0, process_family_rss=0, process_family_pss=0,
        spill_free_bytes=8 * 1024**3, spill_total_bytes=16 * 1024**3, disk_busy=0,
    )
    decision = ResourceDecision(
        target_concurrency=2, target_cpu_tokens=2, read_wave_bytes=1024**2,
        factor_block_bytes=1024**2, result_queue_bytes=1024**2, io_concurrency=2,
        remote_concurrency=2, cache_budget_bytes=1024**2, spill_budget_bytes=1024**2,
        pressure_state="NORMAL",
    )
    monkeypatch.setattr(scheduler.broker, "_refresh", lambda **kwargs: snapshot)
    monkeypatch.setattr(scheduler.broker, "resource_decision", lambda **kwargs: decision)
    scheduler.broker._cpu.set_soft_budget(2)
    started = threading.Event()
    finish = threading.Event()

    def execute(task):
        if task.factor_name == "F1":
            started.set()
            assert finish.wait(10), "test failed to release blocked root"
        else:
            assert started.wait(10), "second root was not admitted concurrently"
        return 42

    def fail(name, result):
        raise OSError("writer failed")

    try:
        with pytest.raises(OSError, match="writer failed"):
            scheduler.run(_bare_dag(2), backend=None, ctx=_Ctx(),
                          execute_root=execute, result_handler=fail)
        assert scheduler.broker.summary()["running_tasks"] == 1
        assert scheduler.broker.summary()["cpu"]["in_use"] == 1
    finally:
        finish.set()
        scheduler.executor.shutdown()
    assert scheduler.broker.summary()["running_tasks"] == 0
    assert scheduler.broker.summary()["cpu"]["in_use"] == 0


@pytest.mark.parametrize("accepted", [None, True, False])
def test_callback_sink_rejection_is_not_success(accepted):
    from factor_engine.runtime.batch_service import _handle_result

    paths = {}
    if accepted is False:
        with pytest.raises(RuntimeError, match="sink rejected result"):
            _handle_result("sink", lambda *args: False, {}, "factor", 42, {}, paths)
        assert paths == {}
    else:
        results = {}
        _handle_result("sink", lambda *args: accepted, results, "factor", 42, {}, paths)
        assert results == {}
        assert paths == {"factor": {}}


def test_uncertified_telemetry_is_local_and_does_not_resample_resources(monkeypatch):
    from factor_engine.backend.cleaned_bridge import _record_uncertified
    from factor_engine.backend.path_summary import snapshot_backend_path
    import factor_engine.runtime.resource_telemetry as telemetry

    def forbidden(*args, **kwargs):
        pytest.fail("operator-level telemetry must not probe all process memory")

    monkeypatch.setattr(telemetry, "resource_telemetry_summary", forbidden)
    contexts = [SimpleNamespace(runtime_stats={}) for _ in range(64)]
    for ctx in contexts:
        _record_uncertified("add", "research degradation", ctx=ctx)
        assert ctx.runtime_stats["parameter_certification_degradation_count"] == 1
    _record_uncertified("sub", "second degradation", ctx=contexts[0])
    assert contexts[0].runtime_stats["parameter_certification_degradation_count"] == 2
    assert contexts[1].runtime_stats["parameter_certification_degradation_count"] == 1
    path = snapshot_backend_path(contexts[0].runtime_stats)
    assert path["parameter_certification_degradation_count"] == 2
    assert path["param_domain_membership_skipped"] == "sub"
