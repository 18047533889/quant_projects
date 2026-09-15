from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

import pytest

from factor_engine.backend.context import ExecutionContext
from factor_engine.planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalRegionPlan,
    Representation,
    TransferEdge,
    TransferTransform,
)
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.physical_factor_dag import (
    PhysicalFactorDAG,
    PhysicalFactorTask,
    TASK_CSE_SHARED,
    TASK_ROOT,
)
from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
from factor_engine.runtime.engine import (
    PhysicalPlanRequiredError,
    _admit_ready_single_region_batch,
    _execute_ready_single_region_plan,
    _execute_ready_physical_shared_task,
    _physical_backend_for_region,
)
from factor_engine.runtime.batch_service import _execute_root_with_path, _materialize_shared_subplan
from factor_engine.runtime.buffer_store import GovernedBufferStore


class PandasBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[PlanNode, object]] = []

    def execute(self, plan: PlanNode, ctx: object) -> str:
        self.calls.append((plan, ctx))
        return "pandas-result"


class PolarsBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[PlanNode, object]] = []

    def execute(self, plan: PlanNode, ctx: object) -> object:
        self.calls.append((plan, ctx))
        return plan.inputs[0].attrs["value"] if plan.inputs else "polars-result"


class HybridBackend:
    def __init__(self) -> None:
        self._pandas = PandasBackend()
        self._polars = PolarsBackend()
        self.route_calls = 0

    def execute(self, plan: PlanNode, ctx: object) -> object:
        self.route_calls += 1
        raise AssertionError("HybridBackend.execute must not route a physical plan")


@dataclass
class _Context:
    runtime_stats: dict[str, object]
    task_id: str | None = None
    factor_id: str | None = None
    shared_result_cache: object = None
    shared_long_lazy_cache: object = None
    materialized_long_lazy: object = None
    materialized_series: object = None
    panel_cache: object = None


def _region(
    backend: PhysicalBackend = PhysicalBackend.PANDAS_NUMPY,
    *,
    region_id: str = "r1",
    node_ids: tuple[str, ...] = ("factor",),
    representation: Representation = Representation.PANDAS_LONG,
) -> BackendRegion:
    return BackendRegion(
        region_id=region_id,
        backend=backend,
        representation=representation,
        node_ids=node_ids,
        execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
        estimated_rows=1,
        estimated_compute_ms=1.0,
        estimated_memory_bytes=8,
    )


def _plan(
    *regions: BackendRegion,
    edges: tuple[TransferEdge, ...] = (),
    roots: tuple[str, ...] | None = None,
    topological_order: tuple[str, ...] | None = None,
) -> PhysicalRegionPlan:
    return PhysicalRegionPlan(
        plan_id="physical",
        regions=regions,
        edges=edges,
        topological_order=topological_order or tuple(
            region.region_id for region in regions
        ),
        root_region_ids=roots or (regions[-1].region_id,),
        total_compute_ms=1.0,
        total_transfer_ms=0.0,
        total_ttdc_ms=1.0,
        peak_memory_bytes=8,
        plan_hash="physical",
        logical_node_count=sum(len(region.node_ids) for region in regions),
        backend_switch_count=sum(
            edge.source_backend != edge.target_backend for edge in edges
        ),
        native_fraction=1.0,
    )


def _optimization(plan: PhysicalRegionPlan, *, ready: bool = True) -> object:
    return SimpleNamespace(
        physical_plan=plan,
        production_ready=ready,
        readiness_reason="not certified" if not ready else "",
    )


def test_ready_single_region_executes_original_root_without_hybrid_routing() -> None:
    root = PlanNode(op="literal", attrs={"value": 1}, node_id="node-debug-id")
    backend = HybridBackend()
    ctx = SimpleNamespace(runtime_stats={})

    result = _execute_ready_single_region_plan(
        _optimization(_plan(_region())),
        root,
        backend,
        ctx,
        logical_root_id="factor",
    )

    assert result == "pandas-result"
    assert backend.route_calls == 0
    assert backend._pandas.calls == [(root, ctx)]
    assert ctx.runtime_stats["physical_plan"]["planned_backend"] == "pandas_numpy"
    assert ctx.runtime_stats["physical_plan"]["actual_backend"] == "pandas_numpy"


@pytest.mark.parametrize(
    ("optimization", "message"),
    [
        (_optimization(_plan(_region()), ready=False), "not production-ready"),
        (
            _optimization(_plan(_region(PhysicalBackend.CLICKHOUSE_SQL))),
            "not runtime-capable",
        ),
        (
            _optimization(_plan(_region(node_ids=("other",)))),
            "does not contain batch roots",
        ),
    ],
)
def test_physical_execution_rejects_unexecutable_plans(
    optimization: object, message: str
) -> None:
    with pytest.raises(PhysicalPlanRequiredError, match=message):
        _execute_ready_single_region_plan(
            optimization,
            PlanNode(op="literal", node_id="factor"),
            HybridBackend(),
            object(),
            logical_root_id="factor",
        )


def test_physical_execution_materializes_valid_transfer_in_topological_order() -> None:
    first = _region(region_id="r1", node_ids=("child",))
    region = _region(region_id="r2", node_ids=("factor",))
    edge = TransferEdge(
        edge_id="e1",
        producer_region="r1",
        consumer_region="r2",
        source_backend=PhysicalBackend.PANDAS_NUMPY,
        target_backend=PhysicalBackend.PANDAS_NUMPY,
        source_representation=Representation.PANDAS_LONG,
        target_representation=Representation.PANDAS_LONG,
        transform=TransferTransform.SAME_BACKEND_NATIVE,
        estimated_rows=1,
        estimated_bytes=8,
        estimated_transfer_ms=1.0,
    )
    optimization = _optimization(
        _plan(
            region,
            first,
            edges=(edge,),
            roots=("r2",),
            topological_order=("r1", "r2"),
        )
    )
    root = PlanNode(
        op="add",
        inputs=(PlanNode(op="literal", attrs={"value": 2}, node_id="child"),),
        node_id="debug-root",
    )
    backend = HybridBackend()

    result = _execute_ready_single_region_plan(
        optimization,
        root,
        backend,
        SimpleNamespace(runtime_stats={}),
        logical_root_id="factor",
    )

    assert result == "pandas-result"
    assert backend.route_calls == 0
    assert [call[0].node_id for call in backend._pandas.calls] == [
        "child",
        "debug-root",
    ]
    assert backend._pandas.calls[1][0].inputs[0].attrs["value"] == "pandas-result"



def test_non_first_root_executes_only_its_reachable_region_and_reports_it() -> None:
    first = _region(region_id="r1", node_ids=("first",))
    second = _region(region_id="r2", node_ids=("second",))
    optimization = _optimization(_plan(first, second, roots=("r1", "r2")))
    backend = HybridBackend()
    ctx = SimpleNamespace(runtime_stats={})

    result = _execute_ready_single_region_plan(
        optimization,
        PlanNode(op="literal", attrs={"value": 2}, node_id="debug-second"),
        backend,
        ctx,
        logical_root_id="second",
    )

    assert result == "pandas-result"
    assert len(backend._pandas.calls) == 1
    assert backend._pandas.calls[0][0].node_id == "debug-second"
    assert ctx.runtime_stats["physical_plan"]["region_id"] == "r2"
    assert ctx.runtime_stats["physical_plan"]["planned_backend"] == "pandas_numpy"


def test_root_closure_ignores_unrelated_regions_and_executes_real_pandas_to_polars() -> None:
    producer = _region(region_id="r1", node_ids=("child",))
    consumer = _region(
        PhysicalBackend.POLARS_PANEL,
        region_id="r2",
        node_ids=("first",),
        representation=Representation.POLARS_LONG,
    )
    unrelated = _region(region_id="r3", node_ids=("second",))
    edge = TransferEdge(
        edge_id="e1",
        producer_region="r1",
        consumer_region="r2",
        source_backend=PhysicalBackend.PANDAS_NUMPY,
        target_backend=PhysicalBackend.POLARS_PANEL,
        source_representation=Representation.PANDAS_LONG,
        target_representation=Representation.POLARS_LONG,
        transform=TransferTransform.PANDAS_TO_POLARS,
        estimated_rows=2,
        estimated_bytes=16,
        estimated_transfer_ms=1.0,
    )
    optimization = _optimization(
        _plan(
            producer,
            consumer,
            unrelated,
            edges=(edge,),
            roots=("r2", "r3"),
            topological_order=("r1", "r2", "r3"),
        )
    )
    frame = pd.DataFrame({"value": [1.0, 2.0]})
    backend = HybridBackend()
    backend._pandas.execute = lambda plan, ctx: frame
    ctx = SimpleNamespace(runtime_stats={})

    result = _execute_ready_single_region_plan(
        optimization,
        PlanNode(
            op="identity",
            inputs=(PlanNode(op="literal", node_id="child"),),
            node_id="debug-first",
        ),
        backend,
        ctx,
        logical_root_id="first",
    )

    import polars as pl

    assert isinstance(result, pl.DataFrame)
    assert result.to_dict(as_series=False) == {"value": [1.0, 2.0]}
    assert len(backend._polars.calls) == 1
    transferred = backend._polars.calls[0][0].inputs[0].attrs["value"]
    assert isinstance(transferred, pl.DataFrame)
    assert ctx.runtime_stats["physical_plan"]["region_id"] == "r2"
    assert ctx.runtime_stats["physical_plan"]["planned_backend"] == "polars_panel"
    assert ctx.runtime_stats["physical_plan"]["materialization_count"] == 1


def test_physical_executor_runs_real_polars_to_pandas_transfer() -> None:
    producer = _region(
        PhysicalBackend.POLARS_PANEL,
        region_id="r1",
        node_ids=("child",),
        representation=Representation.POLARS_LONG,
    )
    consumer = _region(region_id="r2", node_ids=("factor",))
    edge = TransferEdge(
        edge_id="e1",
        producer_region="r1",
        consumer_region="r2",
        source_backend=PhysicalBackend.POLARS_PANEL,
        target_backend=PhysicalBackend.PANDAS_NUMPY,
        source_representation=Representation.POLARS_LONG,
        target_representation=Representation.PANDAS_LONG,
        transform=TransferTransform.POLARS_TO_PANDAS,
        estimated_rows=2,
        estimated_bytes=16,
        estimated_transfer_ms=1.0,
    )
    optimization = _optimization(
        _plan(producer, consumer, edges=(edge,), roots=("r2",))
    )
    import polars as pl

    source = pl.DataFrame({"value": [3.0, 4.0]})
    backend = HybridBackend()
    backend._polars.execute = lambda plan, ctx: source
    observed: list[pd.DataFrame] = []

    def consume(plan, ctx):
        observed.append(plan.inputs[0].attrs["value"])
        return observed[-1]

    backend._pandas.execute = consume
    ctx = SimpleNamespace(runtime_stats={})

    result = _execute_ready_single_region_plan(
        optimization,
        PlanNode(
            op="identity",
            inputs=(PlanNode(op="literal", node_id="child"),),
            node_id="debug-factor",
        ),
        backend,
        ctx,
        logical_root_id="factor",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.to_dict(orient="list") == {"value": [3.0, 4.0]}
    assert ctx.runtime_stats["physical_plan"]["transfers"][0]["edge_id"] == "e1"
    assert ctx.runtime_stats["physical_plan"]["transfer_metrics"]["direct_count"] == 1

def test_physical_execution_rejects_malformed_transfer_residency() -> None:
    first = _region(region_id="r1", node_ids=("child",))
    region = _region(region_id="r2", node_ids=("factor",))
    edge = TransferEdge(
        edge_id="e1",
        producer_region="r1",
        consumer_region="r2",
        source_backend=PhysicalBackend.POLARS_PANEL,
        target_backend=PhysicalBackend.PANDAS_NUMPY,
        source_representation=Representation.PANDAS_LONG,
        target_representation=Representation.PANDAS_LONG,
        transform=TransferTransform.POLARS_TO_NUMPY,
        estimated_rows=1,
        estimated_bytes=8,
        estimated_transfer_ms=1.0,
    )
    with pytest.raises(ValueError, match="source residency"):
        _plan(first, region, edges=(edge,), roots=("r2",))

def test_concrete_backend_must_match_fixed_physical_backend() -> None:
    PolarsBackend = type("PolarsBackend", (), {"execute": lambda self, plan, ctx: None})

    with pytest.raises(PhysicalPlanRequiredError, match="does not match"):
        _execute_ready_single_region_plan(
            _optimization(_plan(_region())),
            PlanNode(op="literal", node_id="factor"),
            PolarsBackend(),
            object(),
            logical_root_id="factor",
        )


def test_batch_admission_accepts_two_roots_in_one_region() -> None:
    optimization = _optimization(_plan(_region(node_ids=("a", "b", "shared"))))

    assert _admit_ready_single_region_batch(optimization, ("a", "b")) is optimization


def test_batch_admission_accepts_two_regions_with_valid_topology() -> None:
    optimization = _optimization(
        _plan(
            _region(region_id="r1", node_ids=("a",)),
            _region(region_id="r2", node_ids=("b",)),
            roots=("r1", "r2"),
        )
    )

    assert _admit_ready_single_region_batch(optimization, ("a", "b")) is optimization


def test_execute_root_with_path_uses_real_snapshot_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = PlanNode(op="literal", attrs={"value": 1}, node_id="debug")
    optimization = _optimization(_plan(_region(node_ids=("a", "b"))))
    backend = HybridBackend()
    ctx = _Context(runtime_stats={})
    monkeypatch.setattr(
        "factor_engine.runtime.batch_service.assert_production_fastpath_runtime",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "factor_engine.runtime.batch_service._assert_no_native_certified_fallback",
        lambda *args, **kwargs: None,
    )

    result, path = _execute_root_with_path(
        backend,
        root,
        ctx,
        factor_name="a",
        physical_optimization=optimization,
        physical_root_id="a",
    )

    assert result == "pandas-result"
    assert backend.route_calls == 0
    assert backend._pandas.calls[0][0] is root
    assert path["planned_backend"] == "pandas_numpy"
    assert path["actual_backend"] == "pandas_numpy"
    assert path["physical_region_id"] == "r1"
    assert path["physical_plan"] == {
        "plan_id": "physical",
        "plan_hash": "physical",
        "planned_backend": "pandas_numpy",
        "actual_backend": "pandas_numpy",
        "region_id": "r1",
        "backend_switch_count": 0,
        "estimated_peak_memory_bytes": 8,
        "memory_basis": "estimated-not-measured",
        "materialization_count": 0,
        "resident_reuse_count": 0,
        "python_to_q_bytes": 0,
        "source_residency": "",
    }
    assert "backend_path_summary" in path


def test_hybrid_long_is_rejected_as_rerouting_executor() -> None:
    class HybridLongBackend:
        def execute(self, plan: object, ctx: object) -> object:
            raise AssertionError("must not execute hybrid long")

    with pytest.raises(PhysicalPlanRequiredError, match="does not match"):
        _physical_backend_for_region(HybridLongBackend(), PhysicalBackend.POLARS_LONG)


def test_hybrid_resolves_underlying_polars_long_without_routing() -> None:
    PolarsLongBackend = type("PolarsLongBackend", (), {})
    concrete = PolarsLongBackend()
    hybrid_long = SimpleNamespace(_polars_long=concrete)
    backend = HybridBackend()
    backend._long_backend = lambda: hybrid_long

    assert (
        _physical_backend_for_region(backend, PhysicalBackend.POLARS_LONG)
        is concrete
    )
    assert backend.route_calls == 0


def test_lazy_literal_shared_materialization_reports_zero() -> None:
    class LazyBackend:
        supports_lazy_shared = True

        def execute(self, sub: object, ctx: object) -> object:
            raise AssertionError("lazy literal must not execute")

    assert _materialize_shared_subplan(
        LazyBackend(),
        PlanNode(op="literal"),
        _Context(runtime_stats={}, shared_result_cache={}),
        "sid",
    ) is False


def test_compile_only_shared_materialization_reports_zero() -> None:
    class LazyBackend:
        supports_lazy_shared = True

        def compile_lazy_shared(self, sub: object, ctx: object, *, sid: str) -> bool:
            return True

    assert _materialize_shared_subplan(
        LazyBackend(),
        PlanNode(op="add"),
        _Context(runtime_stats={}, shared_result_cache={}),
        "sid",
    ) is False


def test_eager_shared_materialization_reports_one() -> None:
    class EagerBackend:
        supports_lazy_shared = False

        def execute(self, sub: object, ctx: object) -> str:
            return "value"

    ctx = _Context(runtime_stats={}, shared_result_cache={})
    assert _materialize_shared_subplan(EagerBackend(), PlanNode(op="add"), ctx, "sid") is True


def test_precomputed_shared_value_skips_backend_and_uses_governed_store() -> None:
    class MustNotExecute:
        supports_lazy_shared = True

        def compile_lazy_shared(self, *args, **kwargs):
            raise AssertionError("precomputed value must skip lazy compilation")

        def execute(self, *args, **kwargs):
            raise AssertionError("precomputed value must skip backend execution")

    backing: dict[str, object] = {}
    store = GovernedBufferStore(backing=backing)
    value = pd.Series([3.0, 4.0])
    ctx = _Context(runtime_stats={}, shared_result_cache=backing)
    ctx.shared_buffers = store
    assert _materialize_shared_subplan(
        MustNotExecute(), PlanNode("neg"), ctx, "sid", precomputed_value=value
    ) is True
    assert store.get_ref("sid") is value


def test_serial_scheduler_shared_and_root_use_fixed_concrete_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from factor_engine.runtime.task_resource_contract import TaskResourceContract
    shared = PlanNode(op="literal", attrs={"value": 1}, node_id="shared")
    root = PlanNode(op="literal", attrs={"value": 2}, node_id="root")
    dag = PhysicalFactorDAG(roots=("root:a",))
    dag.add_task(
        PhysicalFactorTask(
            task_id="cse:shared",
            op="literal",
            task_type=TASK_CSE_SHARED,
            consumers=("root:a",),
            node_ref=shared,
            resource_contract=TaskResourceContract(peak_memory_bytes=1024 * 1024),
        )
    )
    dag.add_task(
        PhysicalFactorTask(
            task_id="root:a",
            op="literal",
            task_type=TASK_ROOT,
            inputs=("cse:shared",),
            node_ref=root,
            factor_name="a",
            resource_contract=TaskResourceContract(peak_memory_bytes=1024 * 1024),
        )
    )
    backend = HybridBackend()
    concrete = backend._pandas
    ctx = _Context(runtime_stats={}, shared_result_cache={})
    monkeypatch.setattr(
        "factor_engine.backend.plan_cost_router.choose_plan_route",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("route selection must not run")
        ),
    )

    seen: list[tuple[str, object]] = []

    def materialize(sid: str, node: object) -> None:
        seen.append((sid, concrete.execute(node, ctx)))

    def execute_root(task: PhysicalFactorTask) -> object:
        return concrete.execute(task.node_ref, ctx)

    AdaptiveBatchScheduler().run_serial_fused(
        dag,
        backend=concrete,
        ctx=ctx,
        materialize_shared=materialize,
        execute_root=execute_root,
    )

    assert seen == [("shared", "pandas-result")]
    assert [call[0] for call in concrete.calls] == [shared, root]
    assert backend.route_calls == 0


def test_physical_plan_ref_reuses_real_governed_value_without_releasing_it() -> None:
    shared = pd.Series([1.0, 2.0])
    store = GovernedBufferStore(backing={"shared": shared})
    region = _region(node_ids=("factor", "ref", "shared"))
    root = PlanNode(
        op="identity",
        inputs=(PlanNode(op="plan_ref", attrs={"sid": "shared"}, node_id="ref"),),
        node_id="debug-factor",
    )
    backend = HybridBackend()

    def consume(plan: PlanNode, ctx: object) -> object:
        return plan.inputs[0].attrs["value"]

    backend._pandas.execute = consume
    ctx = SimpleNamespace(runtime_stats={}, shared_buffers=store)

    result = _execute_ready_single_region_plan(
        _optimization(_plan(region)), root, backend, ctx, logical_root_id="factor"
    )

    assert result is shared
    assert store.get_ref("shared") is shared
    assert len(backend._pandas.calls) == 0
    assert ctx.runtime_stats["physical_plan"]["resident_reuse_count"] == 1
    store.release("shared")
    assert store.get_ref("shared") is None


def test_physical_plan_ref_same_region_representation_mismatch_fails_closed() -> None:
    import polars as pl

    store = GovernedBufferStore(backing={"shared": pd.Series([1.0])})
    region = _region(
        PhysicalBackend.POLARS_PANEL,
        node_ids=("factor", "ref", "shared"),
        representation=Representation.POLARS_LONG,
    )
    root = PlanNode(
        op="identity",
        inputs=(PlanNode(op="plan_ref", attrs={"sid": "shared"}, node_id="ref"),),
        node_id="debug-factor",
    )

    with pytest.raises(PhysicalPlanRequiredError, match="does not match same-region"):
        _execute_ready_single_region_plan(
            _optimization(_plan(region)),
            root,
            HybridBackend(),
            SimpleNamespace(runtime_stats={}, shared_buffers=store),
            logical_root_id="factor",
        )


def test_physical_plan_ref_cross_region_uses_declared_real_conversion() -> None:
    import polars as pl

    shared_region = _region(region_id="shared-region", node_ids=("shared",))
    consumer_region = _region(
        PhysicalBackend.POLARS_PANEL,
        region_id="consumer-region",
        node_ids=("factor", "ref"),
        representation=Representation.POLARS_LONG,
    )
    edge = TransferEdge(
        edge_id="shared-to-consumer",
        producer_region="shared-region",
        consumer_region="consumer-region",
        source_backend=PhysicalBackend.PANDAS_NUMPY,
        target_backend=PhysicalBackend.POLARS_PANEL,
        source_representation=Representation.PANDAS_LONG,
        target_representation=Representation.POLARS_LONG,
        transform=TransferTransform.PANDAS_TO_POLARS,
        estimated_rows=2,
        estimated_bytes=16,
        estimated_transfer_ms=1.0,
    )
    frame = pd.DataFrame({"value": [5.0, 6.0]})
    store = GovernedBufferStore(backing={"shared": frame})
    backend = HybridBackend()
    root = PlanNode(
        op="identity",
        inputs=(PlanNode(op="plan_ref", attrs={"sid": "shared"}, node_id="ref"),),
        node_id="debug-factor",
    )

    result = _execute_ready_single_region_plan(
        _optimization(
            _plan(
                shared_region,
                consumer_region,
                edges=(edge,),
                roots=("consumer-region",),
            )
        ),
        root,
        backend,
        SimpleNamespace(runtime_stats={}, shared_buffers=store),
        logical_root_id="factor",
    )

    assert isinstance(result, pl.DataFrame)
    assert result.to_dict(as_series=False) == {"value": [5.0, 6.0]}
    assert backend._pandas.calls == []
    assert len(backend._polars.calls) == 1


def test_real_global_optimizer_executes_plan_ref_nodes_without_debug_ids() -> None:
    from factor_engine.planner.batch_global_optimizer import BatchGlobalOptimizer
    from factor_engine.runtime.default_execution_policy import ExecutionPurpose

    shared_definition = PlanNode("literal", attrs={"value": 0})
    optimized_root = PlanNode(
        "neg", inputs=(PlanNode("plan_ref", attrs={"sid": "shared"}),)
    )
    ctx = SimpleNamespace(
        run_mode="research",
        execution_purpose=ExecutionPurpose(),
        runtime_stats={
            "row_count_estimate": 2,
            "estimated_bytes": 32,
            "estimated_memory_bytes": 16,
        },
    )
    optimization = BatchGlobalOptimizer(
        forced_backend="pandas_numpy"
    ).optimize_batch(
        {"factor": optimized_root}, {"shared": shared_definition}, {}, ctx
    )
    shared = pd.Series([7.0, 8.0])
    ctx.shared_buffers = GovernedBufferStore(backing={"shared": shared})
    backend = HybridBackend()
    backend._pandas.execute = lambda plan, local_ctx: plan.inputs[0].attrs["value"]

    # Scheduler roots are a different (pre-binding) object graph and likewise
    # carry no node IDs. Execution must use the optimizer-owned bound graph and
    # its discovered identities rather than mutating PlanNode debug IDs.
    scheduler_root = PlanNode(
        "neg", inputs=(PlanNode("plan_ref", attrs={"sid": "shared"}),)
    )
    result = _execute_ready_single_region_plan(
        optimization,
        scheduler_root,
        backend,
        ctx,
        logical_root_id="factor",
    )

    assert result is shared
    assert optimization.logical_roots["factor"] is optimized_root
    assert all(node.node_id is None for node in (optimized_root, *optimized_root.inputs))


def test_real_global_optimizer_executes_shared_kernel_without_hybrid_routing() -> None:
    from dataclasses import replace
    from factor_engine.backend.hybrid_backend import HybridBackend as RealHybridBackend
    from factor_engine.planner.batch_global_optimizer import BatchGlobalOptimizer

    values = pd.Series(
        [2.0, -5.0],
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-01"), "A"), (pd.Timestamp("2026-01-02"), "A")],
            names=["timestamp", "instrument"],
        ),
    )
    shared = PlanNode("neg", inputs=(PlanNode("literal", attrs={"value": values}),))
    root = PlanNode("plan_ref", attrs={"sid": "shared"})
    optimizer_ctx = SimpleNamespace(
        run_mode="research",
        runtime_stats={
            "row_count_estimate": 2,
            "estimated_bytes": 32,
            "estimated_memory_bytes": 16,
        },
    )
    optimization = BatchGlobalOptimizer(
        forced_backend="pandas_numpy"
    ).optimize_batch({"factor": root}, {"shared": shared}, {}, optimizer_ctx)
    optimization = replace(optimization, production_ready=True, readiness_reason="")
    backend = RealHybridBackend()
    backend.execute = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("HybridBackend.execute rerouted a fixed physical shared task")
    )
    ctx = ExecutionContext(data_source=object(), runtime_stats={})

    result = _execute_ready_physical_shared_task(
        optimization, "shared", backend, ctx
    )

    pd.testing.assert_series_equal(result, -values)


def test_execution_index_builds_once_for_many_roots_and_rebuilds_for_new_plan() -> None:
    from dataclasses import replace
    from factor_engine.runtime.multibackend.batch_global_optimizer import BatchOptimizationResult
    from factor_engine.runtime.physical_execution_index import PhysicalExecutionIndexCache

    regions = tuple(_region(region_id=f"r{i}", node_ids=(f"factor-{i}",)) for i in range(20))
    plan = _plan(*regions, roots=tuple(region.region_id for region in regions))
    cache = PhysicalExecutionIndexCache()
    optimization = BatchOptimizationResult(
        physical_plan=plan, per_node_choices={}, shared_benefits={},
        total_shared_benefit_ms=0.0, optimization_basis="test",
        production_ready=True, execution_ready=True, execution_index_cache=cache,
    )
    for i in range(20):
        _admit_ready_single_region_batch(optimization, (f"factor-{i}",))
    assert cache.build_count == 1
    _admit_ready_single_region_batch(replace(optimization, physical_plan=replace(plan, plan_id="new")), ("factor-0",))
    assert cache.build_count == 2


def test_execution_index_revalidates_replacement_plan_and_fails_closed() -> None:
    from dataclasses import replace
    from factor_engine.runtime.physical_execution_index import PhysicalExecutionIndexCache

    optimization = SimpleNamespace(
        physical_plan=_plan(_region()), production_ready=True, execution_ready=True,
        execution_index_cache=PhysicalExecutionIndexCache(),
    )
    _admit_ready_single_region_batch(optimization, ("factor",))
    malformed = replace(optimization.physical_plan, plan_id="malformed")
    object.__setattr__(malformed, "topological_order", ("unknown-region",))
    optimization.physical_plan = malformed
    with pytest.raises(PhysicalPlanRequiredError, match="invalid topology"):
        _admit_ready_single_region_batch(optimization, ("factor",))


def test_each_root_execution_does_not_scan_full_index_tables() -> None:
    from dataclasses import replace
    from factor_engine.runtime.multibackend.batch_global_optimizer import BatchOptimizationResult
    from factor_engine.runtime.physical_execution_index import PhysicalExecutionIndexCache

    class NoItems(dict):
        def items(self):
            raise AssertionError("root execution scanned the full edge index")

    class NoIteration(tuple):
        def __iter__(self):
            raise AssertionError("root execution scanned full topological order")

    roots = {f"factor-{i}": PlanNode("literal", node_id=f"factor-{i}") for i in range(20)}
    regions = tuple(_region(region_id=f"r{i}", node_ids=(f"factor-{i}",)) for i in range(20))
    plan = _plan(*regions, roots=tuple(region.region_id for region in regions))
    cache = PhysicalExecutionIndexCache()
    optimization = BatchOptimizationResult(
        physical_plan=plan, per_node_choices={}, shared_benefits={},
        total_shared_benefit_ms=0.0, optimization_basis="test",
        logical_roots=roots, production_ready=True, execution_ready=True,
        execution_index_cache=cache,
    )
    _admit_ready_single_region_batch(optimization, tuple(roots))
    cache._index = replace(
        cache._index,
        edges_by_pair=NoItems(cache._index.edges_by_pair),
        topological_order=NoIteration(cache._index.topological_order),
    )
    backend = HybridBackend()
    for root_id, root in roots.items():
        assert _execute_ready_single_region_plan(
            optimization, root, backend, SimpleNamespace(runtime_stats={}),
            logical_root_id=root_id,
        ) == "pandas-result"
    assert cache.build_count == 1


def test_execution_index_concurrent_plan_replacement_never_cross_pairs() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from factor_engine.runtime.physical_execution_index import PhysicalExecutionIndexCache

    first = _plan(_region(region_id="first", node_ids=("a",)), roots=("first",))
    second = _plan(_region(region_id="second", node_ids=("b",)), roots=("second",))
    cache = PhysicalExecutionIndexCache()

    def lookup(plan):
        index = cache.get(plan)
        assert index.plan is plan
        return index

    plans = [first, second] * 50
    with ThreadPoolExecutor(max_workers=8) as executor:
        indexes = list(executor.map(lookup, plans))
    assert all(index.plan is plan for index, plan in zip(indexes, plans))


def test_batch_admission_rejects_requested_root_in_undeclared_region() -> None:
    optimization = _optimization(
        _plan(
            _region(region_id="r1", node_ids=("a",)),
            _region(region_id="r2", node_ids=("b",)),
            roots=("r2",),
        )
    )
    with pytest.raises(PhysicalPlanRequiredError, match="declared root regions"):
        _admit_ready_single_region_batch(optimization, ("a", "b"))


def test_high_degree_incoming_validation_uses_smaller_local_set() -> None:
    from dataclasses import replace
    from types import MappingProxyType
    from factor_engine.runtime.physical_execution_index import PhysicalExecutionIndexCache

    class HighDegreeIncoming:
        def __len__(self):
            return 100_000

        def __contains__(self, value):
            return value == "producer"

        def __iter__(self):
            raise AssertionError("iterated high-degree incoming collection")

    producer = _region(region_id="producer", node_ids=("child",))
    consumer = _region(region_id="consumer", node_ids=("factor",))
    edge = TransferEdge(
        edge_id="edge", producer_region="producer", consumer_region="consumer",
        source_backend=PhysicalBackend.PANDAS_NUMPY,
        target_backend=PhysicalBackend.PANDAS_NUMPY,
        source_representation=Representation.PANDAS_LONG,
        target_representation=Representation.PANDAS_LONG,
        transform=TransferTransform.SAME_BACKEND_NATIVE,
        estimated_rows=1, estimated_bytes=8, estimated_transfer_ms=0.0,
    )
    plan = _plan(producer, consumer, edges=(edge,), roots=("consumer",))
    cache = PhysicalExecutionIndexCache()
    optimization = SimpleNamespace(
        physical_plan=plan, production_ready=True, execution_ready=True,
        execution_index_cache=cache,
    )
    _admit_ready_single_region_batch(optimization, ("factor",))
    cache._index = replace(
        cache._index,
        incoming_regions=MappingProxyType({"consumer": HighDegreeIncoming()}),
    )
    root = PlanNode(
        "identity", inputs=(PlanNode("literal", node_id="child"),), node_id="factor"
    )
    assert _execute_ready_single_region_plan(
        optimization, root, HybridBackend(), SimpleNamespace(runtime_stats={}),
        logical_root_id="factor",
    ) == "pandas-result"
