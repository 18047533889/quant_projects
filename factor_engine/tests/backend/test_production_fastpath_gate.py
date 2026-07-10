# -*- coding: utf-8
"""Production fast path gate 单元测试。"""
from __future__ import annotations

import pytest

from backend.production_fastpath_gate import check_production_fastpath_plan_ops
from backend.sql_pushdown.plan_fixtures import column, minimal_plan


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


@pytest.mark.parametrize(
    "op",
    [
        "ts_mean",
        "rank",
        "protected_div",
        "group_mean",
        "add",
        "winsorize",
        "group_winsorize",
        "vwap",
        "ts_rank",
        "ts_sharpe",
        "cs_resid",
    ],
)
def test_production_fastpath_ops_pass(_loaded, op: str):
    plan = minimal_plan(op)
    result = check_production_fastpath_plan_ops(plan)
    assert result.ok, result.violations


@pytest.mark.parametrize(
    "op",
    [
        "WMA",
        "ts_quantile",
        "ts_skew",
        "ts_regression",
        "Slope",
        "ts_product",
        "group_decay_linear",
        "signed_log",
        "signed_sqrt",
        "cum_std",
        "expanding_std",
        "ewm_mean",
        "ewm_std",
        "ewm_var",
        "ts_ema",
        "RSI_WILDER",
        "ATR_WILDER",
    ],
)
def test_deferred_ops_fail_fastpath(_loaded, op: str):
    from backend.production_fastpath_tiers import resolve_polars_native_canonical

    plan = minimal_plan(op)
    result = check_production_fastpath_plan_ops(plan)
    assert not result.ok
    resolved = resolve_polars_native_canonical(op)
    assert any(resolved in v or op in v for v in result.violations)


def test_composite_plan_fastpath(_loaded):
    from planner.logical_plan import PlanNode

    plan = PlanNode(
        op="add",
        inputs=[minimal_plan("ts_mean"), minimal_plan("ts_delta")],
        attrs={},
    )
    result = check_production_fastpath_plan_ops(plan)
    assert result.ok, result.violations


def test_python_rolling_op_blocked(_loaded):
    from backend.polars_long_policy import infer_polars_long_tier

    op = "ts_decay_linear"
    assert infer_polars_long_tier(op) == "python_rolling"
    plan = minimal_plan(op)
    result = check_production_fastpath_plan_ops(plan)
    assert not result.ok


def test_map_groups_op_blocked(_loaded):
    from backend.polars_long_policy import infer_polars_long_tier

    op = "ewm_corr"
    assert infer_polars_long_tier(op) == "map_groups"
    plan = minimal_plan(op)
    result = check_production_fastpath_plan_ops(plan)
    assert not result.ok
