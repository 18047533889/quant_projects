# -*- coding: utf-8 -*-
"""R57: pairwise rolling operators must read their window from the declared
positional index, not from the "first non-column child" heuristic.

Regression for the bug where ``ts_corr(ret, log_returns(volume), 120)`` failed on
the polars backend with ``PlanParamError: window 必须为整数 literal，收到动态输入
'ts_log_return'`` while the pandas backend executed it fine. The pairwise contract
is ``x, y, window[, min_periods]``, so the second operand is itself a series
expression and must never be mistaken for the window.

The unary case must keep its strictness: a genuinely dynamic window on a unary
rolling operator is still rejected.
"""
from __future__ import annotations

import pytest

from factor_engine.backend.pair_window_spec import PairWindowSpec
from factor_engine.backend.plan_params import PlanParamError, window_spec_from_plan_node
from factor_engine.backend.window_spec import WindowSpec
from factor_engine.planner.logical_plan import PlanNode


def col(name: str = "x") -> PlanNode:
    return PlanNode(op="column", inputs=(), attrs={"name": name})


def lit(value) -> PlanNode:
    return PlanNode(op="literal", inputs=(), attrs={"value": value})


def derived(op: str = "ts_log_return") -> PlanNode:
    """A series-valued expression, i.e. not a plain column and not a literal."""
    return PlanNode(op=op, inputs=(col("v"),), attrs={})


# ---------------------------------------------------------------------------
# pairwise: window lives at positional index 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [2, 20, 60, 120])
def test_pairwise_window_read_from_index_two(window: int) -> None:
    node = PlanNode(op="ts_corr", inputs=(col("ret"), derived(), lit(window)), attrs={})
    assert PairWindowSpec.from_plan_node(node).size == window


@pytest.mark.parametrize("op", ["ts_corr", "ts_cov", "ts_beta", "ts_regression_slope"])
@pytest.mark.parametrize("second_operand", ["derived", "column", "plan_ref"])
def test_pairwise_window_index_two_for_every_operand_shape(op: str, second_operand: str) -> None:
    if second_operand == "derived":
        right = derived("subtract")
    elif second_operand == "plan_ref":
        right = PlanNode(op="plan_ref", inputs=(), attrs={"sid": 7})
    else:
        right = col("volume")
    node = PlanNode(op=op, inputs=(col("ret"), right, lit(60)), attrs={})
    assert PairWindowSpec.from_plan_node(node).size == 60


def test_pairwise_window_omitted_uses_default() -> None:
    node = PlanNode(op="ts_corr", inputs=(col("ret"), derived()), attrs={})
    assert PairWindowSpec.from_plan_node(node).size == 20


def test_pairwise_window_attr_wins_over_position() -> None:
    node = PlanNode(
        op="ts_corr", inputs=(col("ret"), derived(), lit(60)), attrs={"window": 33}
    )
    assert PairWindowSpec.from_plan_node(node).size == 33


def test_pairwise_dynamic_window_still_rejected() -> None:
    """An explicit positional window that is not a literal is still an error."""
    node = PlanNode(op="ts_corr", inputs=(col("ret"), col("volume"), derived("subtract")), attrs={})
    with pytest.raises(PlanParamError):
        PairWindowSpec.from_plan_node(node)


def test_pairwise_min_periods_attrs_still_applied() -> None:
    node = PlanNode(
        op="ts_corr",
        inputs=(col("ret"), derived(), lit(60)),
        attrs={"min_periods": 10},
    )
    spec = PairWindowSpec.from_plan_node(node)
    assert spec.size == 60
    assert spec.min_periods == 10


# ---------------------------------------------------------------------------
# unary: the generic heuristic is unchanged and stays strict
# ---------------------------------------------------------------------------


def test_unary_positional_literal_unchanged() -> None:
    node = PlanNode(op="ts_std", inputs=(col("ret"), lit(20)), attrs={})
    assert WindowSpec.from_plan_node(node).size == 20


def test_unary_window_attr_unchanged() -> None:
    node = PlanNode(op="ts_std", inputs=(col("ret"),), attrs={"window": 15})
    assert WindowSpec.from_plan_node(node).size == 15


def test_unary_dynamic_window_still_rejected() -> None:
    """Strictness for unary rolling ops must not be weakened by this fix."""
    node = PlanNode(op="ts_std", inputs=(col("ret"), derived("subtract")), attrs={})
    with pytest.raises(PlanParamError):
        WindowSpec.from_plan_node(node)


def test_unary_omitted_window_uses_default() -> None:
    node = PlanNode(op="ts_std", inputs=(col("ret"),), attrs={})
    assert WindowSpec.from_plan_node(node, default_size=7).size == 7


def test_window_input_index_is_opt_in() -> None:
    """Without the index argument the old heuristic would still misfire."""
    node = PlanNode(op="ts_corr", inputs=(col("ret"), derived(), lit(60)), attrs={})
    with pytest.raises(PlanParamError):
        WindowSpec.from_plan_node(node)
    assert WindowSpec.from_plan_node(node, window_input_index=2).size == 60


def test_plan_params_passthrough() -> None:
    node = PlanNode(op="ts_corr", inputs=(col("ret"), derived(), lit(45)), attrs={})
    assert window_spec_from_plan_node(node, window_input_index=2).size == 45
