"""Execution contract for same-residency dependencies split across regions."""
from types import SimpleNamespace

import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.planner.backend_region import PhysicalBackend, Representation
from factor_engine.runtime.physical_execution_index import build_physical_execution_index
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import _execute_ready_single_region_plan
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    NodeBackendChoice,
    PhysicalBatchGlobalOptimizer,
)


class _ScalarBackend:
    def __init__(self, label):
        self.runtime_backend_label = label
        self.calls = []
        self.results = []

    def execute(self, plan, _ctx):
        self.calls.append(plan)

        def evaluate(node):
            if node.op == "literal":
                return node.attrs["value"]
            values = [evaluate(child) for child in node.inputs]
            if node.op == "negate":
                return -values[0]
            if node.op == "add":
                return values[0] + values[1]
            if node.op == "where":
                return values[1] if values[0] else values[2]
            raise AssertionError(f"unexpected test operator {node.op!r}")

        result = evaluate(plan)
        self.results.append(result)
        return result


class HybridBackend:
    def __init__(self):
        self._pandas = _ScalarBackend("pandas_numpy")
        self._polars = _ScalarBackend("polars_panel")


def _choice(node_id, backend, representation):
    return NodeBackendChoice(
        node_id=node_id,
        backend=backend,
        representation=representation,
        compute_cost_ms=1.0,
        transfer_from_children_ms=0.0,
        total_cost_ms=1.0,
        execution_kind=ExecutionKind.PANDAS_REFERENCE,
        production_certified=True,
    )


def test_same_backend_cross_region_dependency_is_materialized_not_recomputed():
    leaf = PlanNode("literal", attrs={"value": 7, "origin": "leaf"}, node_id="leaf")
    middle = PlanNode("negate", inputs=(leaf,), node_id="middle")
    root = PlanNode("add", inputs=(leaf, middle), node_id="root-debug")
    graph = {"leaf": [], "middle": ["leaf"], "root": ["leaf", "middle"]}
    nodes = {"leaf": leaf, "middle": middle, "root": root}
    choices = {
        "leaf": _choice(
            "leaf", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG
        ),
        "middle": _choice(
            "middle", PhysicalBackend.POLARS_PANEL, Representation.POLARS_LONG
        ),
        "root": _choice(
            "root", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG
        ),
    }
    plan = PhysicalBatchGlobalOptimizer()._build_physical_plan(
        choices,
        {},
        0.0,
        3.0,
        0.0,
        SimpleNamespace(),
        graph,
        ("root",),
        {key: (1, 8, 8) for key in graph},
        nodes,
    )
    owner = {
        node_id: region.region_id
        for region in plan.regions
        for node_id in region.node_ids
    }
    same_backend_pair = (owner["leaf"], owner["root"])
    assert same_backend_pair in {
        (edge.producer_region, edge.consumer_region) for edge in plan.edges
    }

    backend = HybridBackend()
    ctx = SimpleNamespace(runtime_stats={})
    optimization = SimpleNamespace(
        physical_plan=plan,
        production_ready=True,
        readiness_reason="",
        discovered_node_ids={},
        execution_index_cache=None,
    )
    result = _execute_ready_single_region_plan(
        optimization, root, backend, ctx, logical_root_id="root"
    )

    assert result == 0
    assert len(backend._pandas.calls) == 2
    assert len(backend._polars.calls) == 1
    assert backend._pandas.calls[0].attrs["origin"] == "leaf"
    root_input = backend._pandas.calls[1].inputs[0]
    assert root_input.op == "literal" and root_input.attrs == {"value": 7}
    assert ctx.runtime_stats["physical_plan"]["materialization_count"] == 3


def test_multiple_logical_dependencies_between_region_pair_are_unambiguous():
    graph = {
        "a": [],
        "b": ["a"],
        "x": ["a"],
        "c": ["a", "x"],
        "d": ["b", "c"],
    }
    a = PlanNode("literal", attrs={"value": 2, "origin": "a"}, node_id="a")
    b = PlanNode("negate", inputs=(a,), node_id="b")
    x = PlanNode("negate", inputs=(a,), node_id="x")
    c = PlanNode("add", inputs=(a, x), node_id="c")
    d = PlanNode("add", inputs=(b, c), node_id="d-debug")
    nodes = {"a": a, "b": b, "x": x, "c": c, "d": d}
    choices = {
        key: _choice(
            key,
            PhysicalBackend.POLARS_PANEL if key == "x" else PhysicalBackend.PANDAS_NUMPY,
            Representation.POLARS_LONG if key == "x" else Representation.PANDAS_LONG,
        )
        for key in graph
    }
    plan = PhysicalBatchGlobalOptimizer()._build_physical_plan(
        choices,
        {},
        0.0,
        5.0,
        0.0,
        SimpleNamespace(),
        graph,
        ("d",),
        {key: (1, 8, 8) for key in graph},
        nodes,
    )
    owner = {
        node_id: region.region_id
        for region in plan.regions
        for node_id in region.node_ids
    }
    # a and b carry distinct values across different outgoing boundaries, so
    # fusing them would violate the runtime's one-materialized-output contract.
    assert owner["a"] != owner["b"]
    assert owner["c"] == owner["d"]
    assert owner["a"] != owner["c"]
    pairs = [(edge.producer_region, edge.consumer_region) for edge in plan.edges]
    assert len(pairs) == len(set(pairs))
    build_physical_execution_index(plan)

    backend = HybridBackend()
    ctx = SimpleNamespace(runtime_stats={})
    optimization = SimpleNamespace(
        physical_plan=plan,
        production_ready=True,
        readiness_reason="",
        discovered_node_ids={id(node): node_id for node_id, node in nodes.items()},
        execution_index_cache=None,
    )
    result = _execute_ready_single_region_plan(
        optimization, d, backend, ctx, logical_root_id="d"
    )
    assert result == -2
    # The two pandas producer calls prove a=2 and b=-2 were independently
    # materialized rather than being collapsed to one region output.
    assert 2 in backend._pandas.results and -2 in backend._pandas.results


class _SharedStore:
    def __init__(self, values):
        self.values = values

    def get_ref(self, sid):
        return self.values.get(sid)


def _same_region_plan_ref_optimization(shared_plan, value):
    shared = shared_plan
    ref = PlanNode("plan_ref", attrs={"sid": "shared"}, node_id="ref")
    two = PlanNode("literal", attrs={"value": 2}, node_id="two")
    root = PlanNode("add", inputs=(ref, two), node_id="root-debug")
    nodes = {"ref": ref, "two": two, "root": root}
    graph = {"ref": ["shared"], "two": [], "root": ["ref", "two"]}

    def register(node, node_id):
        nodes[node_id] = node
        child_ids = []
        for index, child in enumerate(node.inputs):
            child_id = child.node_id or f"{node_id}_child_{index}"
            register(child, child_id)
            child_ids.append(child_id)
        graph[node_id] = child_ids

    register(shared, "shared")
    choices = {
        key: _choice(
            key, PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG
        )
        for key in graph
    }
    plan = PhysicalBatchGlobalOptimizer()._build_physical_plan(
        choices, {}, 0.0, 4.0, 0.0, SimpleNamespace(), graph, ("root",),
        {key: (1, 32, 32) for key in graph}, nodes,
    )
    optimization = SimpleNamespace(
        physical_plan=plan,
        production_ready=True,
        readiness_reason="",
        discovered_node_ids={id(node): node_id for node_id, node in nodes.items()},
        logical_shared_nodes={"shared": shared},
        execution_index_cache=None,
    )
    ctx = SimpleNamespace(runtime_stats={}, shared_buffers=_SharedStore({"shared": value}))
    return optimization, root, ctx


def test_proven_scalar_shared_value_is_rebuilt_as_literal_for_panel_region():
    shared = PlanNode("literal", attrs={"value": 3}, node_id="shared")
    optimization, root, ctx = _same_region_plan_ref_optimization(shared, 3)
    assert _execute_ready_single_region_plan(
        optimization, root, HybridBackend(), ctx, logical_root_id="root"
    ) == 5


def test_proven_scalar_where_constant_branch_preserves_actual_value():
    shared = PlanNode(
        "where",
        inputs=(
            PlanNode("literal", attrs={"value": True}),
            PlanNode("literal", attrs={"value": 7.0}),
            PlanNode("literal", attrs={"value": 9.0}),
        ),
        node_id="shared",
    )
    optimization, root, ctx = _same_region_plan_ref_optimization(shared, 7.0)
    assert _execute_ready_single_region_plan(
        optimization, root, HybridBackend(), ctx, logical_root_id="root"
    ) == 9.0


def test_panel_shared_plan_returning_scalar_is_rejected():
    shared = PlanNode("column", attrs={"name": "close"}, node_id="shared")
    optimization, root, ctx = _same_region_plan_ref_optimization(shared, 3.0)
    from factor_engine.runtime.engine import PhysicalPlanRequiredError

    with pytest.raises(PhysicalPlanRequiredError, match="does not match same-region"):
        _execute_ready_single_region_plan(
            optimization, root, HybridBackend(), ctx, logical_root_id="root"
        )


def test_boundary_validation_expands_shared_definition_closure():
    leaf = PlanNode("literal", attrs={"value": 2}, node_id="leaf")
    shared = PlanNode("negate", inputs=(leaf,), node_id="shared")
    sibling = PlanNode("negate", inputs=(leaf,), node_id="sibling")
    ref = PlanNode("plan_ref", attrs={"sid": "shared"}, node_id="ref")
    root = PlanNode("add", inputs=(ref, sibling), node_id="root-debug")
    nodes = {
        "leaf": leaf,
        "shared": shared,
        "sibling": sibling,
        "ref": ref,
        "root": root,
    }
    graph = {
        "leaf": [],
        "shared": ["leaf"],
        "sibling": ["leaf"],
        "ref": ["shared"],
        "root": ["ref", "sibling"],
    }
    choices = {
        node_id: _choice(
            node_id,
            PhysicalBackend.POLARS_PANEL
            if node_id in {"leaf", "sibling"}
            else PhysicalBackend.PANDAS_NUMPY,
            Representation.POLARS_LONG
            if node_id in {"leaf", "sibling"}
            else Representation.PANDAS_LONG,
        )
        for node_id in graph
    }
    plan = PhysicalBatchGlobalOptimizer()._build_physical_plan(
        choices, {}, 0.0, 5.0, 0.0, SimpleNamespace(), graph, ("root",),
        {key: (1, 32, 32) for key in graph}, nodes,
    )
    optimization = SimpleNamespace(
        physical_plan=plan,
        production_ready=True,
        readiness_reason="",
        discovered_node_ids={id(node): node_id for node_id, node in nodes.items()},
        logical_shared_nodes={"shared": shared},
        execution_index_cache=None,
    )
    ctx = SimpleNamespace(runtime_stats={}, shared_buffers=_SharedStore({"shared": -2}))

    from factor_engine.runtime.engine import PhysicalPlanRequiredError

    # Reaching the representation check proves boundary validation accepted
    # leaf -> shared from the expanded shared-definition closure.  Without the
    # closure it fails earlier by misclassifying that planned edge as extra.
    with pytest.raises(PhysicalPlanRequiredError, match="does not match same-region"):
        _execute_ready_single_region_plan(
            optimization, root, HybridBackend(), ctx, logical_root_id="root"
        )
