# -*- coding: utf-8 -*-
"""Execution and backend-capability verification for declarative recipes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from factor_recipes.planner_bridge import compile_recipe_plans
from factor_recipes.registry import FactorRecipeRegistry


@dataclass(frozen=True)
class RecipeVerification:
    pandas_execution_verified: bool
    polars_plan_capable: bool
    duckdb_plan_capable: bool
    missing_polars_operators: tuple[str, ...]
    missing_duckdb_operators: tuple[str, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scalar_parameter(name: str) -> bool:
    return name == "window" or name.endswith("_window") or name in {
        "width", "constant", "lambda_param"
    }


def _scalar_value(name: str) -> float | int:
    if name == "window":
        return 4
    if name.endswith("_window"):
        return 3
    return {"width": 2.0, "constant": 0.015, "lambda_param": 0.5}[name]


def _panel(name: str, rows: int = 16, columns: int = 6) -> pd.DataFrame:
    row = np.arange(rows, dtype=float)[:, None]
    col = np.arange(columns, dtype=float)[None, :]
    values = 20.0 + row * 0.7 + col * 1.1 + np.sin(row + col) * 0.2
    lowered = name.lower()
    if lowered in {"returns", "ret"}:
        values = 0.002 + np.sin(row + col) * 0.01
    elif "volume" in lowered:
        values = 1000.0 + row * 17.0 + col * 31.0
    elif "liabilities" in lowered or "debt" in lowered:
        values = 8.0 + row * 0.2 + col * 0.3
    elif "inventory" in lowered or "receivable" in lowered:
        values = 4.0 + row * 0.1 + col * 0.2
    elif "cash" in lowered:
        values = 6.0 + row * 0.15 + col * 0.1
    elif "market_cap" in lowered or "assets" in lowered or "equity" in lowered:
        values = 100.0 + row * 2.0 + col * 5.0
    elif lowered == "low":
        values = values - 1.0
    elif lowered == "high":
        values = values + 1.0
    elif lowered == "open":
        values = values - 0.2
    elif lowered == "ann_factor":
        values = np.full_like(values, 252.0)
    values = values.astype(float)
    values[2, 1] = np.nan
    return pd.DataFrame(
        values,
        index=pd.Index(range(rows), name="ts"),
        columns=pd.Index([f"S{i}" for i in range(columns)], name="inst"),
    )


def _period_panel(rows: int = 16, columns: int = 6) -> pd.DataFrame:
    periods = [f"2023Q{1 + min(index // 4, 3)}" for index in range(rows)]
    return pd.DataFrame(
        np.repeat(np.asarray(periods, dtype=object)[:, None], columns, axis=1),
        index=pd.Index(range(rows), name="ts"),
        columns=pd.Index([f"S{i}" for i in range(columns)], name="inst"),
    )


def recipe_fixture(recipe_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe = FactorRecipeRegistry.get(recipe_name)
    if recipe is None:
        raise KeyError(recipe_name)
    bindings: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for parameter in recipe.parameters:
        if _scalar_parameter(parameter):
            bindings[parameter] = _scalar_value(parameter)
        else:
            bindings[parameter] = parameter
            inputs[parameter] = _period_panel() if parameter == "period_id" else _panel(parameter)
    return bindings, inputs


def _called_operators(plan) -> set[str]:
    names: set[str] = set()
    visited: set[int] = set()

    def visit(node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        if node.op not in {"column", "literal"}:
            names.add(node.op)
        for child in node.inputs:
            visit(child)

    visit(plan)
    return names


def verify_recipe(recipe_name: str, *, allowed_statuses=("production",)) -> RecipeVerification:
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from backend.sql_pushdown.emitter import plan_is_sql_capable
    from backend.polars_expr_emitter import plan_is_polars_long_capable

    load_all()
    bindings, inputs = recipe_fixture(recipe_name)
    logical = compile_recipe_plans(
        {recipe_name: (recipe_name, bindings)}, allowed_statuses=allowed_statuses
    )
    plan = logical.plans[recipe_name]
    calls = _called_operators(plan)
    missing_polars = tuple(sorted(
        name for name in calls
        if name not in {"add", "subtract", "multiply", "divide", "power", "neg", "identity"}
        and "polars" not in OperatorRegistry.backends_for(name)
    ))
    missing_duckdb = tuple(sorted(
        name for name in calls
        if name not in {"add", "subtract", "multiply", "divide", "power", "neg", "identity"}
        and "sql" not in OperatorRegistry.backends_for(name)
    ))

    batch = FactorRecipeRegistry.compile_batch(
        {recipe_name: (recipe_name, bindings)}, allowed_statuses=allowed_statuses
    )
    try:
        output = batch.execute(inputs, backend="pandas_numpy")[recipe_name]
        if not isinstance(output, pd.DataFrame):
            raise TypeError(f"recipe output is not a panel: {type(output)!r}")
        pandas_verified = True
        error = None
    except Exception as exc:
        pandas_verified = False
        error = f"{type(exc).__name__}: {exc}"

    return RecipeVerification(
        pandas_execution_verified=pandas_verified,
        polars_plan_capable=plan_is_polars_long_capable(plan),
        duckdb_plan_capable=plan_is_sql_capable(plan),
        missing_polars_operators=missing_polars,
        missing_duckdb_operators=missing_duckdb,
        error=error,
    )


def verify_recipes(*, statuses=("production",)) -> Mapping[str, RecipeVerification]:
    result = {}
    for name in FactorRecipeRegistry.list_names():
        recipe = FactorRecipeRegistry.get(name)
        if recipe is not None and recipe.status in statuses:
            result[name] = verify_recipe(name, allowed_statuses=statuses)
    return result


__all__ = ["RecipeVerification", "recipe_fixture", "verify_recipe", "verify_recipes"]
