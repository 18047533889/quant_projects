"""The runtime observer must run before compute admission on both lanes."""
from types import SimpleNamespace
import pytest

from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler


def test_serial_observer_runs_after_reads_before_reservation():
    from factor_engine.tests.runtime.test_run_many_sink_backpressure import serial_scheduler
    scheduler, dag = serial_scheduler()
    events = []
    scheduler._execute_read_waves = lambda *args, **kwargs: events.append("read")
    def refresh(owner, plan, committed, remaining):
        assert owner is scheduler
        assert remaining == {"root"}
        events.append("refresh")
    original = scheduler._reserve_serial_task
    def reserve(*args, **kwargs):
        assert events[-1] == "refresh"
        events.append("reserve")
        return original(*args, **kwargs)
    scheduler._reserve_serial_task = reserve
    plan = SimpleNamespace(physical_dag=dag, refresh_task_budgets=refresh)
    result = scheduler.run_serial_fused(
        plan, backend=None, ctx=None, execute_root=lambda task: 42
    )
    assert result["results"] == {"factor": 42}
    assert events[:3] == ["read", "refresh", "reserve"]


def test_observer_failure_never_dispatches_compute():
    from factor_engine.tests.runtime.test_run_many_sink_backpressure import serial_scheduler
    scheduler, dag = serial_scheduler()
    executed = []
    def fail(*args):
        raise ValueError("invalid observed scope")
    plan = SimpleNamespace(physical_dag=dag, refresh_task_budgets=fail)
    with pytest.raises(ValueError, match="invalid observed scope"):
        scheduler.run_serial_fused(plan, backend=None, ctx=None,
                                  execute_root=lambda task: executed.append(task))
    assert executed == []


def test_parallel_observer_precedes_real_root_admission(monkeypatch):
    import pandas as pd
    from benchmarks.benchmark_run_many_streaming_20260906 import Source
    from factor_engine.api import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.runtime.perf_config import PerfConfig

    events = []
    plan_method = AdaptiveBatchScheduler.plan
    reserve_method = AdaptiveBatchScheduler._reserve_task
    def plan_with_observer(self, *args, **kwargs):
        plan = plan_method(self, *args, **kwargs)
        def refresh(owner, current, committed, remaining):
            assert owner is self
            assert current is plan
            events.append("refresh")
        plan.refresh_task_budgets = refresh
        return plan
    def reserve_after_observer(self, contract, *, task_id):
        assert "refresh" in events
        events.append("reserve")
        return reserve_method(self, contract, task_id=task_id)
    monkeypatch.setattr(AdaptiveBatchScheduler, "plan", plan_with_observer)
    monkeypatch.setattr(AdaptiveBatchScheduler, "_reserve_task", reserve_after_observer)
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["A"]],
        names=["timestamp", "instrument"],
    )
    engine = FactorEngine(PandasBackend(), Source(pd.Series([1., 2., 3.], index=index)))
    output = engine.run_many(
        [Factor(str(i), col("close") + i) for i in range(5)],
        perf=PerfConfig(max_workers=2),
    )
    assert len(output["results"]) == 5
    assert events[0] == "refresh"
    assert "reserve" in events
