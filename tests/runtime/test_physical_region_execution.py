from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from backend.context import ExecutionContext
from planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalRegionPlan,
    Representation,
    TransferEdge,
    TransferTransform,
)
from planner.logical_plan import PlanNode
from planner.physical_factor_dag import (
    PhysicalFactorDAG,
    PhysicalFactorTask,
    TASK_CSE_SHARED,
    TASK_ROOT,
)
from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
from runtime.engine import (
    PhysicalPlanRequiredError,
    _admit_ready_single_region_batch,
    _execute_ready_single_region_plan,
    _physical_backend_for_region,
)
from runtime.batch_service import _execute_root_with_path, _materialize_shared_subplan


class PandasBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[PlanNode, object]] = []

    def execute(self, plan: PlanNode, ctx: object) -> str:
        self.calls.append((plan, ctx))
        return "pandas-result"


class HybridBackend:
    def __init__(self) -> None:
        self._pandas = PandasBackend()
        self.route_calls = 0

    def execute(self, plan: PlanNode, ctx: object) -> object:
        self.route_calls += 1
        raise AssertionError("HybridBackend.execute must not route a physical plan")


@dataclass
class _Context:
    runtime_stats: dict[str, object]
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
) -> BackendRegion:
    return BackendRegion(
        region_id=region_id,
        backend=backend,
        representation=Representation.PANDAS_LONG,
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
            _optimization(
                _plan(
                    _region(region_id="r1", node_ids=("child",)),
                    _region(region_id="r2", node_ids=("factor",)),
                )
            ),
            "exactly one materialized output",
        ),
        (
            _optimization(_plan(_region(PhysicalBackend.CLICKHOUSE_SQL))),
            "unsupported physical backend",
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
        "runtime.batch_service.assert_production_fastpath_runtime",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "runtime.batch_service._assert_no_native_certified_fallback",
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


def test_serial_scheduler_shared_and_root_use_fixed_concrete_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        )
    )
    backend = HybridBackend()
    concrete = backend._pandas
    ctx = _Context(runtime_stats={}, shared_result_cache={})
    monkeypatch.setattr(
        "backend.plan_cost_router.choose_plan_route",
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
