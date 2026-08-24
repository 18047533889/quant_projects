# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace
import pandas as pd
import pytest

from factor_engine.backend.sql_pushdown.sql_registry import is_sql_capable
from factor_engine.cleaned_operators import load_all
from factor_engine.factor_recipes.compiler import RecipeCompiler, RecipeExpansionError
from factor_engine.factor_recipes.planner_bridge import compile_recipe_plans
from factor_engine.factor_recipes.registry import FactorRecipe, FactorRecipeRegistry
from factor_engine.planner.optimizer import Optimizer


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def test_recipe_expands_to_primitives() -> None:
    formula = FactorRecipeRegistry.expand("momentum", {"x": "close", "window": 20})
    assert formula == "ts_delta(close, 20)"
    nested = FactorRecipeRegistry.expand(
        "aroon_oscillator",
        {"high": "high", "low": "low", "window": 25},
    )
    assert "ts_argmax(high, 25)" in nested
    assert "ts_argmin(low, 25)" in nested
    assert "aroon_up" not in nested and "aroon_down" not in nested


def test_optional_recipe_requires_explicit_status_opt_in() -> None:
    with pytest.raises(RecipeExpansionError, match="status"):
        FactorRecipeRegistry.expand(
            "dpo_causal", {"close": "close", "window": 20}, allowed_statuses=("test",)
        )
    expanded = FactorRecipeRegistry.expand(
        "dpo_causal", {"close": "close", "window": 20},
        allowed_statuses=("production", "optional"),
    )
    assert "ts_delay" in expanded


def test_recipe_compiler_rejects_unsafe_syntax_and_cycles() -> None:
    unsafe = FactorRecipe("unsafe_test_recipe", "test", "test", "__import__('os')", (), status="test")
    first = FactorRecipe("cycle_test_a", "test", "test", "cycle_test_b(x)", ("x",), status="test")
    second = FactorRecipe("cycle_test_b", "test", "test", "cycle_test_a(x)", ("x",), status="test")
    for recipe in (unsafe, first, second):
        FactorRecipeRegistry.register(recipe)
    try:
        with pytest.raises(RecipeExpansionError):
            RecipeCompiler(allowed_statuses=("test",)).expand("unsafe_test_recipe", {})
        with pytest.raises(RecipeExpansionError, match="cyclic"):
            RecipeCompiler(allowed_statuses=("test",)).expand("cycle_test_a", {"x": "close"})
    finally:
        for name in (unsafe.name, first.name, second.name):
            FactorRecipeRegistry._recipes.pop(name, None)


def test_batch_compilation_eliminates_common_subexpressions() -> None:
    batch = FactorRecipeRegistry.compile_batch({
        "upper": ("bollinger_upper", {"x": "close", "window": 5, "width": 2.0}),
        "lower": ("bollinger_lower", {"x": "close", "window": 5, "width": 2.0}),
    })
    calls = [node.op for node in batch.nodes.values() if node.kind == "call"]
    assert calls.count("ts_mean") == 1
    assert calls.count("ts_std") == 1
    assert batch.shared_node_count >= 2


def test_compiled_batch_executes_each_shared_operator_once() -> None:
    batch = FactorRecipeRegistry.compile_batch({
        "upper": ("bollinger_upper", {"x": "close", "window": 3, "width": 2.0}),
        "lower": ("bollinger_lower", {"x": "close", "window": 3, "width": 2.0}),
    })
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    counts: dict[str, int] = {}

    def resolver(name: str, backend: str):
        del backend
        if name == "ts_mean":
            def calculate(x, window):
                counts[name] = counts.get(name, 0) + 1
                return x.rolling(int(window), min_periods=1).mean()
        elif name == "ts_std":
            def calculate(x, window):
                counts[name] = counts.get(name, 0) + 1
                return x.rolling(int(window), min_periods=1).std(ddof=1)
        else:
            return None
        return SimpleNamespace(calculate=calculate)

    result = batch.execute({"close": close}, resolver=resolver)
    mean = close.rolling(3, min_periods=1).mean()
    std = close.rolling(3, min_periods=1).std(ddof=1)
    pd.testing.assert_frame_equal(result["upper"], mean + 2.0 * std)
    pd.testing.assert_frame_equal(result["lower"], mean - 2.0 * std)
    assert counts == {"ts_mean": 1, "ts_std": 1}


def test_dag_node_ids_are_deterministic() -> None:
    requests = {"factor": ("gross_profitability", {
        "gross_profit": "gross_profit", "total_assets": "total_assets", "period_id": "period_id",
        "periods_per_year": 4,
    })}
    first = FactorRecipeRegistry.compile_batch(requests)
    second = FactorRecipeRegistry.compile_batch(requests)
    assert first.roots == second.roots
    assert first.nodes == second.nodes


def _find_nodes(root, op):
    found, seen = [], set()
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


def test_recipe_batch_enters_main_planner_with_shared_nodes() -> None:
    batch = compile_recipe_plans({
        "upper": ("bollinger_upper", {"x": "close", "window": 5, "width": 2.0}),
        "lower": ("bollinger_lower", {"x": "close", "window": 5, "width": 2.0}),
    })
    upper, lower = batch.plans["upper"], batch.plans["lower"]
    assert _find_nodes(upper, "ts_mean")[0] is _find_nodes(lower, "ts_mean")[0]
    assert _find_nodes(upper, "ts_std")[0] is _find_nodes(lower, "ts_std")[0]
    assert batch.shared_node_count >= 2
    assert is_sql_capable(upper) and is_sql_capable(lower)
    assert Optimizer().optimize(upper).op == "add"
