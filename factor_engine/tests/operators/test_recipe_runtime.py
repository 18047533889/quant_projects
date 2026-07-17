# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from factor_recipes.compiler import RecipeCompiler, RecipeExpansionError
from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry


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
    assert "aroon_up" not in nested
    assert "aroon_down" not in nested


def test_optional_recipe_requires_explicit_status_opt_in() -> None:
    with pytest.raises(RecipeExpansionError, match="status"):
        FactorRecipeRegistry.expand("dpo_causal", {"close": "close", "window": 20})
    expanded = FactorRecipeRegistry.expand(
        "dpo_causal",
        {"close": "close", "window": 20},
        allowed_statuses=("production", "optional"),
    )
    assert "ts_delay" in expanded


def test_recipe_compiler_rejects_unsafe_syntax_and_cycles() -> None:
    unsafe = FactorRecipe("__unsafe_test", "test", "test", "__import__('os')", ())
    first = FactorRecipe("__cycle_a", "test", "test", "__cycle_b(x)", ("x",))
    second = FactorRecipe("__cycle_b", "test", "test", "__cycle_a(x)", ("x",))
    for recipe in (unsafe, first, second):
        FactorRecipeRegistry.register(recipe)
    try:
        with pytest.raises(RecipeExpansionError):
            RecipeCompiler().expand("__unsafe_test", {})
        with pytest.raises(RecipeExpansionError, match="cyclic"):
            RecipeCompiler().expand("__cycle_a", {"x": "close"})
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
    requests = {
        "factor": ("gross_profitability", {
            "gross_profit": "gross_profit",
            "total_assets": "total_assets",
            "period_id": "period_id",
        })
    }
    first = FactorRecipeRegistry.compile_batch(requests)
    second = FactorRecipeRegistry.compile_batch(requests)
    assert first.roots == second.roots
    assert first.nodes == second.nodes
