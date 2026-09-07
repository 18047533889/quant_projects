import gc
from types import SimpleNamespace

import pandas as pd
import pytest

from factor_engine.planner.physical_factor_dag import TASK_SOURCE_SCAN
from factor_engine.planner.read_wave_planner import ReadWave, ReadWavePlan, ReadWavePlanner
from factor_engine.runtime.adaptive_batch_scheduler import (
    AdaptiveBatchScheduler,
    _retain_wave_lease,
)
from factor_engine.runtime.buffer_ref import SourceWaveExecutor
from factor_engine.runtime.resource_errors import ResourceBudgetExceeded
from factor_engine.storage.sources.read_session import DataSourceReadSession


def _register(planner, task_id, column, time_range=None):
    planner.register_scan_task(
        task_id, dataset="d", source_scope="dataset:d", snapshot_id="s",
        time_range=time_range, columns=[column], estimated_scan_bytes=80,
        estimated_memory_bytes=80,
    )


def test_same_window_union_over_budget_remains_split():
    planner = ReadWavePlanner(rows_estimate=10, wave_memory_budget=120)
    _register(planner, "a", "close", ("2024-01-01", "2024-01-31"))
    _register(planner, "b", "open", ("2024-01-01", "2024-01-31"))
    plan = planner.plan()
    assert len(plan.waves) == 2
    assert all(w.estimated_memory_bytes <= 120 for w in plan.waves)


def test_different_halo_ranges_never_coalesce_without_range_aware_executor():
    planner = ReadWavePlanner(rows_estimate=10, wave_memory_budget=1_000)
    _register(planner, "a", "close", ("2024-01-10", "2024-01-31"))
    _register(planner, "b", "open", ("2024-01-01", "2024-01-31"))
    assert len(planner.plan().waves) == 2


def test_atomic_request_over_budget_is_rejected_before_execution():
    planner = ReadWavePlanner(rows_estimate=10, wave_memory_budget=79)
    _register(planner, "a", "close")
    with pytest.raises(ResourceBudgetExceeded, match="atomic read request"):
        planner.plan()


def test_wave_read_failure_propagates_without_source_commit():
    class BrokenSource:
        def load_columns(self, names):
            raise OSError("read failed")

    wave = ReadWave(
        wave_id=0, source_scope="dataset:d", dataset="d", snapshot_id="s",
        columns=frozenset({"close"}), time_range=None, source_tasks=("scan",),
        estimated_memory_bytes=80,
    )
    scheduler = AdaptiveBatchScheduler.__new__(AdaptiveBatchScheduler)
    scheduler._wave_executor = None
    scheduler._wave_refs = {}
    class Lease:
        released = False
        def release(self): self.released = True
    lease = Lease()
    scheduler.broker = SimpleNamespace(acquire_memory=lambda *a, **k: lease)
    task = SimpleNamespace(task_id="scan", task_type=TASK_SOURCE_SCAN, inputs=())
    committed, remaining = set(), {"scan"}
    with pytest.raises(OSError, match="read failed"):
        scheduler._execute_read_waves(
            SimpleNamespace(read_waves=ReadWavePlan([wave])),
            SimpleNamespace(tasks={"scan": task}), committed, remaining,
            SimpleNamespace(data_source=BrokenSource(), runtime_stats={}),
        )
    assert committed == set()
    assert remaining == {"scan"}
    assert scheduler._wave_refs == {}
    assert lease.released


def test_runtime_broker_denial_prevents_wave_read():
    class Source:
        called = False
        def load_columns(self, names):
            self.called = True
            return {}

    source = Source()
    wave = ReadWave(
        wave_id=0, source_scope="dataset:d", dataset="d", snapshot_id="s",
        columns=frozenset({"close"}), time_range=None, source_tasks=("scan",),
        estimated_memory_bytes=80,
    )
    scheduler = AdaptiveBatchScheduler.__new__(AdaptiveBatchScheduler)
    scheduler._wave_executor = None
    scheduler._wave_refs = {}
    scheduler.broker = SimpleNamespace(acquire_memory=lambda *a, **k: None)
    task = SimpleNamespace(task_id="scan", task_type=TASK_SOURCE_SCAN, inputs=())
    with pytest.raises(ResourceBudgetExceeded, match="broker admission denied"):
        scheduler._execute_read_waves(
            SimpleNamespace(read_waves=ReadWavePlan([wave])),
            SimpleNamespace(tasks={"scan": task}), set(), {"scan"},
            SimpleNamespace(data_source=source, runtime_stats={}),
        )
    assert not source.called


def test_prefetched_mapping_drives_dq_without_cache_or_column_reread():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "A")], names=["timestamp", "instrument"]
    )

    class ZeroCacheSource:
        dataset = ""

        def __init__(self):
            self.calls = 0

        def load_columns(self, names):
            self.calls += 1
            return {name: pd.Series([1.0], index=index) for name in names}

        def load_column(self, name):
            raise AssertionError("DQ must consume the prefetched mapping")

    source = ZeroCacheSource()
    session = DataSourceReadSession.__new__(DataSourceReadSession)
    session._source = source
    session._last_scope_key = None
    report = session.prepare_batch(["close", "open"], input_dq_check=True)
    assert report.passed
    assert source.calls == 1


def test_wave_lease_tracks_mapping_series_value_and_index_owners():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "A"), (pd.Timestamp("2024-01-02"), "B")],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.0, 2.0], index=index)
    mapping = {"close": series}
    values = series.to_numpy(copy=False)
    code = index.codes[0]
    class Ref:
        pass

    ref = Ref()
    ref.meta = {"loaded_columns": mapping}

    class Lease:
        released = False

        def release(self):
            self.released = True

    lease = Lease()
    _retain_wave_lease(ref, lease)

    del ref
    gc.collect()
    assert not lease.released
    del mapping
    gc.collect()
    assert not lease.released
    del series
    gc.collect()
    assert not lease.released
    del values
    gc.collect()
    assert not lease.released
    del index
    gc.collect()
    assert not lease.released
    del code
    gc.collect()
    assert lease.released


def test_physical_lowering_resolves_nested_cse_source_closure_and_native_backend():
    from factor_engine.planner.dag import DAGPlan, FactorPlan
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.physical_factor_dag import TASK_CSE_SHARED, TASK_SOURCE_SCAN
    from factor_engine.planner.physical_lowerer import lower_batch_dag

    column = PlanNode("column", attrs={"name": "close"})
    inner = PlanNode("ts_mean", inputs=(PlanNode("plan_ref", attrs={"sid": "column"}),),
                     attrs={"window": 2})
    root = PlanNode("rank", inputs=(PlanNode("plan_ref", attrs={"sid": "mean"}),))
    dag = DAGPlan(
        roots=[FactorPlan("factor", root)],
        shared_nodes={"column": column, "mean": inner},
    )

    class NativeSource:
        dataset = "d"

        def scan_polars_long(self, columns):
            raise AssertionError("lowering must not execute the source")

    ctx = SimpleNamespace(
        data_source=NativeSource(), selected_backend="polars_long", market="",
    )
    scan_cost = SimpleNamespace(
        estimated_rows=6, selected_bytes=78, projection_bytes=78, remote=False,
        scan_bytes=78, decoded_bytes=96, peak_memory_bytes=96,
    )
    physical = lower_batch_dag(
        dag, ctx=ctx, rows=6, scan_cost_map={"dataset:d": scan_cost},
    )

    scans = [t for t in physical.tasks.values() if t.task_type == TASK_SOURCE_SCAN]
    cse = [t for t in physical.tasks.values() if t.task_type == TASK_CSE_SHARED]
    assert scans and all(t.required_columns == ("close",) for t in scans)
    assert all(t.required_columns == ("close",) for t in cse)
    assert all(t.preferred_backend == "polars" for t in (*scans, *cse))
    scan_ids = {t.task_id for t in scans}
    assert all(scan_ids & set(t.inputs) for t in cse)


def test_two_waves_make_progress_in_small_pool_after_first_consumer_releases():
    class Lease:
        def __init__(self, broker):
            self.broker = broker
            self.released = False

        def release(self):
            if not self.released:
                self.released = True
                self.broker.busy = False

    class OneWaveBroker:
        busy = False

        def execution_budget(self):
            return 80

        def acquire_memory(self, kind, nbytes, lease_id=""):
            if self.busy:
                return None
            self.busy = True
            return Lease(self)

    class Source:
        def load_columns(self, names):
            index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-01-01"), "A")],
                names=["timestamp", "instrument"],
            )
            return {name: pd.Series([1.0], index=index) for name in names}

    waves = [
        ReadWave(
            wave_id=i, source_scope="dataset:d", dataset="d", snapshot_id="s",
            columns=frozenset({column}), time_range=None,
            source_tasks=(f"scan{i}",), consumer_tasks=(f"consumer{i}",),
            estimated_memory_bytes=80,
        )
        for i, column in enumerate(("close", "open"))
    ]
    tasks = {
        "scan0": SimpleNamespace(task_id="scan0", task_type=TASK_SOURCE_SCAN, inputs=(), op="source_scan", factor_name="", preferred_backend="polars"),
        "scan1": SimpleNamespace(task_id="scan1", task_type=TASK_SOURCE_SCAN, inputs=(), op="source_scan", factor_name="", preferred_backend="polars"),
    }
    scheduler = AdaptiveBatchScheduler.__new__(AdaptiveBatchScheduler)
    scheduler._wave_executor = None
    scheduler._wave_refs = {}
    scheduler._wave_pending_consumers = {}
    scheduler._wave_source_tasks = {}
    scheduler._buffer_results = {}
    scheduler._wave_summary = {"events": [], "waves_executed": 0}
    scheduler._input_dq_reports = []
    scheduler._done = 0
    scheduler._wave_covered_tasks = set()
    scheduler._task_started_at = {}
    scheduler._task_timing = {}
    scheduler.broker = OneWaveBroker()
    committed, remaining = set(), {"scan0", "scan1"}
    plan = SimpleNamespace(read_waves=ReadWavePlan(waves))
    dag = SimpleNamespace(tasks=tasks)
    ctx = SimpleNamespace(data_source=Source(), runtime_stats={})

    assert scheduler._execute_read_waves(
        plan, dag, committed, remaining, ctx, defer_on_pressure=True,
    ) == 1
    assert committed == {"scan0"}
    assert any(
        isinstance(event, str) and event.startswith("deferred:1:")
        for event in scheduler._wave_summary["events"]
    )

    scheduler._mark_wave_consumer_done("consumer0")
    gc.collect()
    assert not scheduler.broker.busy
    assert 0 not in scheduler._wave_refs
    assert "scan0" not in scheduler._buffer_results

    assert scheduler._execute_read_waves(
        plan, dag, committed, remaining, ctx, defer_on_pressure=True,
    ) == 1
    assert committed == {"scan0", "scan1"}


def test_native_polars_derivatives_cannot_trigger_early_lease_release():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"x": [1, 2]})
    derived_lazy = frame.lazy().select("x")
    derived_series = frame["x"].slice(0, 1)

    class Ref:
        meta = {"loaded_columns": {}, "native_buffer": frame}

    class Lease:
        released = False

        def release(self):
            self.released = True

    ref = Ref()
    lease = Lease()
    from factor_engine.runtime.adaptive_batch_scheduler import _retain_wave_lease
    _retain_wave_lease(ref, lease)
    del ref, frame
    gc.collect()
    assert derived_lazy.collect().height == 2
    assert derived_series.len() == 1
    assert not lease.released


def test_serial_fused_interleaves_two_waves_in_one_wave_pool():
    from factor_engine.planner.physical_factor_dag import (
        PhysicalFactorDAG, PhysicalFactorTask, TASK_ROOT,
    )

    class Lease:
        def __init__(self, broker):
            self.broker = broker
        def release(self):
            self.broker.busy = False

    class Broker:
        busy = False
        denials = 0
        hard_memory_limit = 80
        def execution_budget(self): return 80
        def acquire_memory(self, kind, nbytes, lease_id=""):
            if self.busy:
                self.denials += 1
                return None
            self.busy = True
            return Lease(self)
        def summary(self): return {"busy": self.busy}

    class Executor:
        def summary(self): return {}

    class Source:
        def load_columns(self, names):
            index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-01-01"), "A")],
                names=["timestamp", "instrument"],
            )
            return {name: pd.Series([1.0], index=index) for name in names}

    dag = PhysicalFactorDAG()
    for i in range(2):
        scan_id, root_id = f"scan{i}", f"root:r{i}"
        dag.add_task(PhysicalFactorTask(
            task_id=scan_id, op="source_scan", task_type=TASK_SOURCE_SCAN,
            consumers=(root_id,), required_columns=(f"c{i}",),
        ))
        dag.add_task(PhysicalFactorTask(
            task_id=root_id, op="root", task_type=TASK_ROOT,
            inputs=(scan_id,), factor_name=f"r{i}",
        ))
    dag.roots = ("root:r0", "root:r1")
    waves = [
        ReadWave(
            wave_id=i, source_scope="dataset:d", dataset="d", snapshot_id="s",
            columns=frozenset({f"c{i}"}), time_range=None,
            source_tasks=(f"scan{i}",), consumer_tasks=(f"root:r{i}",),
            estimated_memory_bytes=80,
        )
        for i in range(2)
    ]
    scheduler = AdaptiveBatchScheduler.__new__(AdaptiveBatchScheduler)
    scheduler.broker, scheduler.executor, scheduler.sink = Broker(), Executor(), None
    scheduler._wave_executor = None
    scheduler._wave_refs = {}
    scheduler._wave_pending_consumers = {}
    scheduler._wave_source_tasks = {}
    scheduler._buffer_results = {}
    scheduler._wave_summary = {"events": [], "waves_executed": 0}
    scheduler._input_dq_reports = []
    scheduler._done = 0
    scheduler._wave_covered_tasks = set()
    scheduler._task_started_at = {}
    scheduler._task_timing = {}
    scheduler._results = {}
    scheduler._explanations = []
    scheduler._scheduler_stats = {}
    scheduler._record_task_calibration = lambda *args: None
    scheduler._release_consumed = lambda *args: None
    order = []
    result = scheduler.run_serial_fused(
        SimpleNamespace(physical_dag=dag, read_waves=ReadWavePlan(waves)),
        backend=object(),
        ctx=SimpleNamespace(data_source=Source(), runtime_stats={}),
        execute_root=lambda task: order.append(task.factor_name) or task.factor_name,
        materialize_shared=lambda *args: None,
    )
    assert order == ["r0", "r1"]
    assert result["results"] == {"r0": "r0", "r1": "r1"}
    assert scheduler.broker.denials >= 1
