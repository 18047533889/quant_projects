from types import SimpleNamespace

import pytest

from factor_engine.planner.backend_region import PhysicalBackend
from factor_engine.ir.nodes import IRNode
from factor_engine.planner.lowerer import Lowerer
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    PhysicalBatchGlobalOptimizer as Optimizer,
)


PAIRWISE_STAGED = (
    "ts_corr",
    "ts_cov",
    "ts_regression_slope",
    "ts_regression_intercept",
    "ts_regression_resid",
    "ts_regression_r2",
)


def pair_node(op: str, window: int, *, custom: bool = False) -> PlanNode:
    return PlanNode(
        "custom" if custom else op,
        inputs=(
            PlanNode("column", attrs={"name": "x"}),
            PlanNode("column", attrs={"name": "y"}),
            PlanNode("literal", attrs={"value": window}),
        ),
        attrs={"canonical": op} if custom else {},
    )


@pytest.mark.parametrize("op", PAIRWISE_STAGED)
def test_polars_panel_pairwise_staged_workspace_scales_with_window(op):
    choice = SimpleNamespace(backend=PhysicalBackend.POLARS_PANEL)
    small = Optimizer._node_additional_workspace_bytes(pair_node(op, 7), choice, 1_000)
    large = Optimizer._node_additional_workspace_bytes(pair_node(op, 700), choice, 1_000)
    assert small == 1_000 * (96 * 7 + 24)
    assert large == 1_000 * (96 * 700 + 24)
    assert large > small


def test_custom_node_uses_bound_canonical_for_workspace():
    choice = SimpleNamespace(backend=PhysicalBackend.POLARS_PANEL)
    node = pair_node("ts_regression_r2", 11, custom=True)
    assert Optimizer._node_additional_workspace_bytes(node, choice, 100) == 100 * (96 * 11 + 24)


def test_ts_cov_polars_long_keeps_more_conservative_existing_bound():
    choice = SimpleNamespace(backend=PhysicalBackend.POLARS_LONG)
    assert Optimizer._node_additional_workspace_bytes(
        pair_node("ts_cov", 7), choice, 100
    ) == 100 * (192 * 7 + 64)


def test_pairwise_workspace_is_added_to_base_memory():
    choice = SimpleNamespace(backend=PhysicalBackend.POLARS_PANEL)
    node = pair_node("ts_regression_resid", 7)
    assert Optimizer._node_execution_memory_bytes(node, choice, 100, 800) == (
        800 + 100 * (96 * 7 + 24)
    )


@pytest.mark.parametrize("op", PAIRWISE_STAGED[2:])
def test_lowered_regression_literal_window_reaches_workspace_budget(op):
    ir = IRNode(op, inputs=(
        IRNode("column", attrs={"name": "y"}),
        IRNode("column", attrs={"name": "x"}),
        IRNode("literal", attrs={"value": 17}),
    ))
    compiled = Lowerer().to_logical_plan(ir)
    choice = SimpleNamespace(backend=PhysicalBackend.POLARS_PANEL)
    assert Optimizer._node_additional_workspace_bytes(compiled, choice, 100) == (
        100 * (96 * 17 + 24)
    )


def test_lowered_keyword_window_and_bound_identity_agree():
    ir = IRNode("ts_regression_resid", inputs=(
        IRNode("column", attrs={"name": "y"}),
        IRNode("column", attrs={"name": "x"}),
    ), attrs={"window": 19, "min_periods": 3})
    compiled = Lowerer().to_logical_plan(ir)
    choice = SimpleNamespace(backend=PhysicalBackend.POLARS_PANEL)
    assert Optimizer._bound_parameter_identity(compiled) == '{"min_periods":3,"window":19}'
    assert Optimizer._node_additional_workspace_bytes(compiled, choice, 100) == (
        100 * (96 * 19 + 24)
    )


def test_real_analyzer_keyword_window_reaches_workspace_budget():
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import col
    from factor_engine.ir.analyzer import Analyzer

    expression = make_cleaned_call_factory("ts_regression_slope")(
        col("y"), col("x"), window=19
    )
    compiled = Lowerer().to_logical_plan(Analyzer().lower(expression).ir)
    choice = SimpleNamespace(backend=PhysicalBackend.POLARS_PANEL)
    assert compiled.op == "ts_regression_slope"
    assert Optimizer._node_additional_workspace_bytes(compiled, choice, 100) == (
        100 * (96 * 19 + 24)
    )


def test_alias_and_unknown_custom_canonicals_are_not_conflated():
    choice = SimpleNamespace(backend=PhysicalBackend.POLARS_PANEL)
    alias = pair_node("ts_regression", 13)
    unknown = pair_node("not_a_pairwise_operator", 13, custom=True)
    assert Optimizer._node_additional_workspace_bytes(alias, choice, 100) == (
        100 * (96 * 13 + 24)
    )
    assert Optimizer._node_additional_workspace_bytes(unknown, choice, 100) == 0
