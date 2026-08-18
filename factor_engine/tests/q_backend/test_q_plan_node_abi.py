from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.q_backend.q_backend import QBackend
from backend.q_backend.q_compiler import QCompiler
from backend.q_backend.q_errors import QPhysicalRegionNotImplemented
from ir.analyzer import Analyzer
from api import ts_mean
from api.columns import col
from planner.logical_plan import PlanNode
from planner.lowerer import Lowerer


def _backend_with_compiler(compiler: object) -> QBackend:
    backend = QBackend.__new__(QBackend)
    backend._compiler = compiler
    backend._production_mode = False
    return backend


def test_lowerer_assigns_deterministic_ids_accepted_by_q_extraction() -> None:
    expr = ts_mean(col("close"), 5)
    first = Lowerer().to_logical_plan(Analyzer().lower(expr).ir)
    second = Lowerer().to_logical_plan(Analyzer().lower(expr).ir)

    assert first.node_id is not None
    assert first.inputs[0].node_id is not None
    assert first.node_id == second.node_id
    assert first.inputs[0].node_id == second.inputs[0].node_id
    nodes = _backend_with_compiler(object())._extract_nodes_topological(first)
    assert all(node["node_id"] for node in nodes)


def test_extract_nodes_uses_canonical_plan_node_abi() -> None:
    leaf = PlanNode(
        op="column",
        inputs=(),
        attrs={"name": "close"},
        semantic_attrs={"unit": "price"},
        node_id="leaf",
    )
    root = PlanNode(
        op="ts_mean",
        inputs=(leaf,),
        attrs={"window": 5},
        semantic_attrs={"frequency": "daily"},
        node_id="root",
    )

    nodes = _backend_with_compiler(object())._extract_nodes_topological(root)

    assert nodes == [
        {
            "node_id": "leaf",
            "op": "column",
            "inputs": [],
            "attrs": {"name": "close"},
            "semantic_attrs": {"unit": "price"},
        },
        {
            "node_id": "root",
            "op": "ts_mean",
            "inputs": ["leaf"],
            "attrs": {"window": 5},
            "semantic_attrs": {"frequency": "daily"},
        },
    ]


def test_plan_to_q_region_passes_canonical_nodes_to_compile_preparation() -> None:
    leaf = PlanNode(op="column", attrs={"name": "close"}, node_id="leaf")
    root = PlanNode(op="ts_mean", inputs=(leaf,), attrs={"window": 5}, node_id="root")
    calls: dict[str, object] = {}

    class Compiler:
        def validate_region(self, nodes, *, mode):
            calls["validated"] = (nodes, mode)
            return True, []

        def compile_region(self, region_id, nodes, *, input_tables, output_name, mode):
            calls["compiled"] = (region_id, nodes, input_tables, output_name, mode)
            return SimpleNamespace(
                region_id=region_id,
                node_ids=tuple(n["node_id"] for n in nodes),
            )

    result = _backend_with_compiler(Compiler())._plan_to_q_region(root, None)

    assert result.region_id == "region_root"
    assert calls["validated"] == (calls["compiled"][1], "research")
    assert calls["compiled"][1][-1]["attrs"] == {"window": 5}
    assert calls["compiled"][1][-1]["semantic_attrs"] == {}


def test_real_compiler_rejects_whole_tree_source_node() -> None:
    leaf = PlanNode(op="column", attrs={"name": "close"}, node_id="close")
    root = PlanNode(op="ts_mean", inputs=(leaf,), attrs={"window": 5}, node_id="mean")

    backend = _backend_with_compiler(QCompiler())
    with pytest.raises(
        ValueError,
        match=r"unsupported operators \['column'\]",
    ):
        backend._plan_to_q_region(root, None)


def test_real_compiler_rejects_positional_literal_as_region_node() -> None:
    leaf = PlanNode(op="column", attrs={"name": "close"}, node_id="close")
    window = PlanNode(op="literal", attrs={"value": 5}, node_id="window")
    root = PlanNode(op="ts_mean", inputs=(leaf, window), attrs={}, node_id="mean")

    backend = _backend_with_compiler(QCompiler())
    with pytest.raises(
        ValueError,
        match=r"unsupported operators \['column', 'literal'\]",
    ):
        backend._plan_to_q_region(root, None)


def test_real_compiler_accepts_explicit_region_inputs_and_canonical_params() -> None:
    region = QCompiler().compile_region(
        "mean_region",
        [
            {
                "node_id": "mean",
                "op": "ts_mean",
                "inputs": ["input_table"],
                "attrs": {"window": 5},
                "semantic_attrs": {"frequency": "daily"},
            }
        ],
        input_tables=["input_table"],
        output_name="result",
        mode="research",
    )

    assert "mean: 5 mavg input_table;" in region.q_code
    assert region.node_ids == ("mean",)
    assert region.input_tables == ("input_table",)


def test_production_whole_tree_execution_remains_disabled() -> None:
    backend = QBackend.__new__(QBackend)
    backend._production_mode = True
    plan = PlanNode(op="column", attrs={"name": "close"}, node_id="root")

    with pytest.raises(QPhysicalRegionNotImplemented, match="whole-tree"):
        backend.execute(plan, None)


def test_extract_nodes_fails_closed_without_canonical_node_id() -> None:
    root = PlanNode(op="column", attrs={"name": "close"})

    with pytest.raises(ValueError, match="canonical PlanNode.node_id"):
        _backend_with_compiler(object())._extract_nodes_topological(root)


def test_extract_nodes_fails_closed_without_child_node_id() -> None:
    child = PlanNode(op="column", attrs={"name": "close"})
    root = PlanNode(op="ts_mean", inputs=(child,), attrs={"window": 5}, node_id="root")

    with pytest.raises(ValueError, match="node_id on every input"):
        _backend_with_compiler(object())._extract_nodes_topological(root)
