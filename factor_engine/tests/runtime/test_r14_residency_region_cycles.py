"""Residency fusion must not introduce cycles into an acyclic logical DAG."""
from types import SimpleNamespace
import pytest
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.planner.backend_region import PhysicalBackend, Representation
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    NodeBackendChoice, PhysicalBatchGlobalOptimizer,
)


def build(graph):
    nodes = {key: PlanNode("abs") for key in graph}
    choices = {}
    for key in graph:
        pandas = key != "middle"
        choices[key] = NodeBackendChoice(
            node_id=key,
            backend=PhysicalBackend.PANDAS_NUMPY if pandas else PhysicalBackend.POLARS_PANEL,
            representation=Representation.PANDAS_LONG if pandas else Representation.POLARS_LONG,
            compute_cost_ms=1., transfer_from_children_ms=0., total_cost_ms=1.,
            execution_kind=ExecutionKind.PANDAS_REFERENCE, production_certified=True,
        )
    return PhysicalBatchGlobalOptimizer()._build_physical_plan(
        choices, {}, 0., float(len(nodes)), 0., SimpleNamespace(), graph,
        ("root",), {key: (12, 96, 192) for key in graph}, nodes,
    )


def test_diamond_backend_reentry_keeps_all_dependencies_without_region_cycle():
    graph = {"leaf": [], "middle": ["leaf"], "root": ["leaf", "middle"]}
    plan = build(graph)
    owner = {node: region.region_id for region in plan.regions for node in region.node_ids}
    assert len(owner) == 3
    assert owner["leaf"] != owner["root"]
    order = {key: index for index, key in enumerate(plan.topological_order)}
    edges = {(edge.producer_region, edge.consumer_region) for edge in plan.edges}
    for parent, children in graph.items():
        for child in children:
            if owner[child] != owner[parent]:
                assert order[owner[child]] < order[owner[parent]]
                assert (owner[child], owner[parent]) in edges
    # Same-backend reentry still needs an explicit dependency edge.
    assert (owner["leaf"], owner["root"]) in edges


def test_actual_node_cycle_is_rejected():
    with pytest.raises(ValueError, match="cyclic"):
        build({"leaf": ["root"], "middle": ["leaf"], "root": ["middle"]})


def test_same_residency_chain_stays_fused():
    plan = build({"leaf": [], "root": ["leaf"]})
    assert len(plan.regions) == 1
    assert not plan.edges
