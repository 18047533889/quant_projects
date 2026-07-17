# -*- coding: utf-8 -*-
from __future__ import annotations

from backend.sql_pushdown.sql_registry import is_sql_capable
from factor_recipes.planner_bridge import compile_recipe_plans
from planner.optimizer import Optimizer


def _find(root, op):
    found = []
    seen = set()

    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        if node.op == op:
            found.append(node)
        for child in node.inputs:
            visit(child)

    visit(root)
    return found


def test_recipe_batch_bridges_to_shared_main_plans():
    batch = compile_recipe_plans({
        "upper": ("bollinger_upper", {"x": "close", "window": 5, "width": 2.0}),
        "lower": ("bollinger_lower", {"x": "close", "window": 5, "width": 2.0}),
    })
    upper, lower = batch.plans["upper"], batch.plans["lower"]
    upper_mean = _find(upper, "ts_mean")[0]
    lower_mean = _find(lower, "ts_mean")[0]
    upper_std = _find(upper, "ts_std")[0]
    lower_std = _find(lower, "ts_std")[0]
    assert upper_mean is lower_mean
    assert upper_std is lower_std
    assert batch.shared_node_count >= 2
    assert is_sql_capable(upper)
    assert is_sql_capable(lower)
    assert Optimizer().optimize(upper).op == "add"
