from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from factor_engine.backend.context import ExecutionContext


def test_pre_m11_positional_arguments_keep_their_meaning() -> None:
    legacy_args = (object(), "research", 7, "evidence-v1", "exec-1", "warn")
    ctx = ExecutionContext(*legacy_args)
    assert ctx.registry_version == 7
    assert ctx.evidence_version == "evidence-v1"
    assert ctx.execution_id == "exec-1"
    assert ctx.production_fallback_policy == "warn"
    assert ctx.profile_id is ctx.run_id is ctx.task_id is ctx.factor_id is None


def test_unowned_context_keeps_legacy_pickle_path_lock_free() -> None:
    ctx = ExecutionContext(data_source=object())
    assert ctx.profile_id is ctx.run_id is ctx.task_id is ctx.factor_id is None
    assert ctx.fit_failure_sink is None
    # Isolate the context transport contract from the pre-existing logical
    # data-source wrapper, which has its own serialization protocol.
    ctx.data_source = None
    ctx._logical_wrapper = None
    restored = pickle.loads(pickle.dumps(ctx))
    assert restored.fit_failure_sink is None
    assert restored.factor_id is None


def test_root_context_is_immutable_copy_and_shared_context_stays_unowned() -> None:
    from factor_engine.runtime.batch_service import _execute_root_with_path

    seen = []

    class Backend:
        def execute(self, plan, ctx):
            seen.append(ctx)
            return 1.0

    shared = ExecutionContext(data_source=object(), run_id="run-1", profile_id="profile-1")
    result, _ = _execute_root_with_path(
        Backend(), SimpleNamespace(node_id="n"), shared,
        factor_name="alpha", task_id="root:alpha", factor_id="factor-alpha",
    )
    assert result == 1.0
    root = seen[0]
    assert (root.task_id, root.factor_id) == ("root:alpha", "factor-alpha")
    assert (root.run_id, root.profile_id) == ("run-1", "profile-1")
    assert shared.task_id is shared.factor_id is None


def test_bridge_binds_real_owner_then_resets(monkeypatch) -> None:
    import factor_engine.backend.cleaned_bridge as bridge
    from factor_engine.cleaned_operators.ts_model import _rolling_core as rc
    from factor_engine.planner.logical_plan import PlanNode

    captured = []

    class Operator:
        metadata = None

        def calculate(self):
            captured.append(rc.current_fit_scope())
            return 1.0

    monkeypatch.setattr(bridge, "ensure_cleaned_loaded", lambda: None)
    monkeypatch.setattr(bridge, "_resolve_canonical", lambda op: "ts_test")
    monkeypatch.setattr(
        bridge, "_resolve_operator",
        lambda canonical, ctx, row_count_estimate: (Operator(), "pandas_numpy"),
    )
    ctx = ExecutionContext(
        data_source=object(), execution_id="exec-1", profile_id="profile-1",
        run_id="run-1", task_id="root:alpha", factor_id="factor-alpha",
    )
    kernel = bridge.make_cleaned_kernel(lambda node, context: None, "ts_test")
    assert kernel(PlanNode(op="ts_test", inputs=[], attrs={}), ctx) == 1.0
    scope = captured[0]
    assert scope is not None
    assert scope.canonical == "ts_test"
    assert scope.backend == "pandas_numpy"
    assert scope.execution_id == "exec-1"
    assert scope.profile == "profile-1"
    assert scope.run_id == "run-1"
    assert scope.task_id == "root:alpha"
    assert scope.factor_id == "factor-alpha"
    assert scope.instrument is None
    assert rc.current_fit_scope() is None


def test_bridge_exception_restores_outer_scope_and_sink(monkeypatch) -> None:
    import factor_engine.backend.cleaned_bridge as bridge
    from factor_engine.cleaned_operators.ts_model import _rolling_core as rc
    from factor_engine.planner.logical_plan import PlanNode

    class Operator:
        metadata = None

        def calculate(self):
            assert rc.current_fit_scope().factor_id == "inner-factor"
            raise RuntimeError("forced operator failure")

    monkeypatch.setattr(bridge, "ensure_cleaned_loaded", lambda: None)
    monkeypatch.setattr(bridge, "_resolve_canonical", lambda op: "ts_test")
    monkeypatch.setattr(
        bridge, "_resolve_operator",
        lambda canonical, ctx, row_count_estimate: (Operator(), "pandas_numpy"),
    )
    outer_sink = rc.BoundedFitFailureSink()
    inner_sink = rc.BoundedFitFailureSink()
    outer_scope = rc.FitScope(canonical="outer", factor_id="outer-factor")
    ctx = ExecutionContext(
        data_source=object(), factor_id="inner-factor", fit_failure_sink=inner_sink,
    )
    kernel = bridge.make_cleaned_kernel(lambda node, context: None, "ts_test")
    with rc.fit_receipt_scope(outer_scope), rc.fit_failure_receipts(outer_sink):
        with pytest.raises(RuntimeError, match="forced operator failure"):
            kernel(PlanNode(op="ts_test", inputs=[], attrs={}), ctx)
        assert rc.current_fit_scope() == outer_scope
        rc.record_current_fit(
            rc.FitResult(None, rc.FitStatus(False, "outer-only")),
            instrument="AAA", window_start=0, window_end=1, output_row=1,
            fit_cutoff=1, maturity_cutoff=1,
        )
    assert [item.status.reason for item in outer_sink.page()] == ["outer-only"]
    assert inner_sink.page() == ()
    assert rc.current_fit_scope() is None


def test_parallel_root_scopes_do_not_cross(monkeypatch) -> None:
    import factor_engine.backend.cleaned_bridge as bridge
    from factor_engine.cleaned_operators.ts_model import _rolling_core as rc
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.batch_service import _execute_root_with_path

    barrier = Barrier(2)
    seen = {}

    class Operator:
        metadata = None

        def calculate(self):
            before = rc.current_fit_scope()
            barrier.wait(timeout=5)
            after = rc.current_fit_scope()
            seen[before.factor_id] = (before.task_id, after.factor_id, after.task_id)
            return 1.0

    monkeypatch.setattr(bridge, "ensure_cleaned_loaded", lambda: None)
    monkeypatch.setattr(bridge, "_resolve_canonical", lambda op: "ts_test")
    monkeypatch.setattr(
        bridge, "_resolve_operator",
        lambda canonical, ctx, row_count_estimate: (Operator(), "pandas_numpy"),
    )
    kernel = bridge.make_cleaned_kernel(lambda node, context: None, "ts_test")

    class Backend:
        def execute(self, plan, ctx):
            return kernel(plan, ctx)

    shared = ExecutionContext(data_source=object())
    node = PlanNode(op="ts_test", inputs=[], attrs={})

    def run(owner):
        return _execute_root_with_path(
            Backend(), node, shared, factor_name=owner,
            task_id=f"task:{owner}", factor_id=owner,
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(run, ("factor-a", "factor-b"))) == [1.0, 1.0]
    assert seen == {
        "factor-a": ("task:factor-a", "factor-a", "task:factor-a"),
        "factor-b": ("task:factor-b", "factor-b", "task:factor-b"),
    }
    assert shared.task_id is shared.factor_id is None


def test_bridge_uses_worker_local_sink_without_inventing_owner(monkeypatch) -> None:
    import factor_engine.backend.cleaned_bridge as bridge
    from factor_engine.cleaned_operators.ts_model import _rolling_core as rc
    from factor_engine.planner.logical_plan import PlanNode

    class Operator:
        metadata = None

        def calculate(self):
            result = rc.FitResult(None, rc.FitStatus(False, "forced"))
            rc.record_current_fit(
                result, instrument="AAA", window_start=0, window_end=2,
                output_row=2, fit_cutoff=2, maturity_cutoff=2,
            )
            return 1.0

    monkeypatch.setattr(bridge, "ensure_cleaned_loaded", lambda: None)
    monkeypatch.setattr(bridge, "_resolve_canonical", lambda op: "ts_test")
    monkeypatch.setattr(
        bridge, "_resolve_operator",
        lambda canonical, ctx, row_count_estimate: (Operator(), "pandas_numpy"),
    )
    sink = rc.BoundedFitFailureSink()
    ctx = ExecutionContext(data_source=object(), fit_failure_sink=sink)
    kernel = bridge.make_cleaned_kernel(lambda node, context: None, "ts_test")
    assert kernel(PlanNode(op="ts_test", inputs=[], attrs={}), ctx) == 1.0
    receipt = sink.page()[0]
    assert receipt.scope.scope_kind == "kernel_only"
    assert receipt.scope.run_id is None
    assert receipt.scope.factor_id is None


def test_shared_cse_bridge_execution_remains_factor_unowned(monkeypatch) -> None:
    import factor_engine.backend.cleaned_bridge as bridge
    from factor_engine.cleaned_operators.ts_model import _rolling_core as rc
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.batch_service import _materialize_shared_subplan

    captured = []

    class Operator:
        metadata = None

        def calculate(self):
            captured.append(rc.current_fit_scope())
            return 2.0

    monkeypatch.setattr(bridge, "ensure_cleaned_loaded", lambda: None)
    monkeypatch.setattr(bridge, "_resolve_canonical", lambda op: "ts_shared")
    monkeypatch.setattr(
        bridge, "_resolve_operator",
        lambda canonical, ctx, row_count_estimate: (Operator(), "pandas_numpy"),
    )
    kernel = bridge.make_cleaned_kernel(lambda node, context: None, "ts_shared")

    class Backend:
        supports_lazy_shared = False

        def execute(self, plan, ctx):
            return kernel(plan, ctx)

    ctx = ExecutionContext(
        data_source=object(), execution_id="exec-1", profile_id="profile-1",
        run_id="run-1", shared_result_cache={},
    )
    assert _materialize_shared_subplan(
        Backend(), PlanNode(op="ts_shared", inputs=[], attrs={}), ctx, "shared-1"
    )
    assert ctx.shared_result_cache["shared-1"] == 2.0
    scope = captured[0]
    assert scope.execution_id == "exec-1"
    assert scope.run_id == "run-1"
    assert scope.task_id is None
    assert scope.factor_id is None


def test_scheduler_execution_entry_creates_bounded_worker_local_sink(monkeypatch) -> None:
    import factor_engine.runtime.adaptive_batch_scheduler as scheduler_module
    from factor_engine.runtime.batch_service import _execute_run_many_scheduler
    from factor_engine.cleaned_operators.ts_model import _rolling_core as rc

    ctx = ExecutionContext(data_source=object())
    engine = SimpleNamespace(
        _make_context=lambda **kwargs: ctx,
        backend=SimpleNamespace(runtime_backend_label="pandas_numpy"),
        resource_broker=None,
    )

    class StopAfterContext:
        def __init__(self, **kwargs):
            assert isinstance(ctx.fit_failure_sink, rc.BoundedFitFailureSink)
            raise RuntimeError("observed worker-local context")

    monkeypatch.setattr(scheduler_module, "AdaptiveBatchScheduler", StopAfterContext)
    with pytest.raises(RuntimeError, match="observed worker-local context"):
        _execute_run_many_scheduler(
            engine, [], SimpleNamespace(roots=[]), {},
            perf=SimpleNamespace(max_workers=1), engine_to_use=engine,
            per_windows={}, input_report=None, run_mode="research",
            source_bars_per_day=1, result_policy="return", sink=None,
            enable_cse=False,
        )
    sink = ctx.fit_failure_sink
    assert isinstance(sink, rc.BoundedFitFailureSink)
    assert sink.detail_capacity == 64
    assert sink.group_capacity == 128


def test_v2_worker_forwards_only_authoritative_profile_and_run(monkeypatch) -> None:
    import factor_engine.runtime.bounded_pipeline as pipeline

    captured = {}

    class Deployment:
        def to_dict(self):
            return {"profile_id": "approved-profile"}

    runtime = pipeline._SpawnFactoryRuntime.__new__(pipeline._SpawnFactoryRuntime)
    runtime.config = SimpleNamespace(deployment=Deployment())
    runtime.broker_proxy = object()
    runtime.mode = "compute"
    runtime._get_engine = lambda: object()

    def fake_compute(*args, **kwargs):
        captured.update(kwargs)
        return b"ok"

    monkeypatch.setattr(pipeline, "_compute_wave_to_artifacts", fake_compute)
    result = runtime(
        [SimpleNamespace(name="alpha")], {}, transport_budget=8 * 1024 * 1024,
        artifact_plan={"run_id": "run-1"}, evidence_id="a" * 32,
    )
    assert captured["transport_budget_bytes"] == 8 * 1024 * 1024
    assert captured["evidence_id"] == "a" * 32
    assert result == b"ok"
    assert captured["execution_owner"] == {
        "profile_id": "approved-profile", "run_id": "run-1"
    }
