# -*- coding: utf-8 -*-
"""Production recipe certification beyond the portable three-backend subset."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from backend.context import ExecutionContext
from backend.factory import build_backend
from factor_recipes.planner_bridge import compile_recipe_plans
from tests.helpers import InMemorySeriesSource

_SCALARS: dict[str, Any] = {
    "window": 3,
    "smooth_window": 2,
    "width": 2.0,
    "ann_factor": 252.0,
    "periods": 1,
    "periods_per_year": 4,
    "body_window": 3,
    "shadow_window": 3,
    "penetration": 0.3,
    "market_cap_window": 3,
    "constant": 0.015,
    "min_obs": 4,
    "add_intercept": True,
}
_PANEL_IDENTIFIERS = frozenset({
    "period_id",
    "fiscal_quarter",
    "group",
    "industry",
    "sector",
})


def _is_scalar_parameter(name: str) -> bool:
    return (
        name in _SCALARS
        or name.endswith("_window")
        or name.endswith("_periods")
        or name.endswith("_factor")
    )


def _scalar_value(name: str):
    if name in _SCALARS:
        return _SCALARS[name]
    if name.endswith("_window"):
        return 3
    if name.endswith("_periods"):
        return 4
    if name.endswith("_factor"):
        return 1.0
    raise KeyError(name)


def _bindings(recipe):
    return {
        parameter: (
            _scalar_value(parameter)
            if _is_scalar_parameter(parameter)
            else parameter
        )
        for parameter in recipe.parameters
    }


def _source(recipes):
    timestamps = pd.date_range("2024-01-01", periods=36, freq="B")
    assets = ["A", "B"]
    index = pd.MultiIndex.from_product(
        [timestamps, assets], names=["timestamp", "instrument"]
    )
    size = len(index)
    base = 20.0 + np.arange(size, dtype=float) * 0.03
    fields = {
        parameter
        for recipe in recipes
        for parameter in recipe.parameters
        if not _is_scalar_parameter(parameter)
    }
    data: dict[str, pd.Series] = {}
    for offset, name in enumerate(sorted(fields)):
        if name == "period_id":
            values = np.repeat(np.arange(len(timestamps)) // 3, len(assets))
            data[name] = pd.Series(values.astype(object), index=index)
            continue
        if name == "fiscal_quarter":
            report_number = np.arange(len(timestamps)) // 3
            quarter = report_number % 4 + 1
            values = np.repeat(quarter, len(assets))
            data[name] = pd.Series(values.astype(float), index=index)
            continue
        if name in {"group", "industry", "sector"}:
            values = np.tile(np.array(["G0", "G1"], dtype=object), len(timestamps))
            data[name] = pd.Series(values, index=index)
            continue

        values = base * (1.0 + 0.01 * offset) + 0.1 * np.sin(
            np.arange(size) / 3 + offset
        )
        if name == "high":
            values = values + 1.0
        if name == "low":
            values = values - 1.0
        if name == "open":
            values = values - 0.2
        if name in {
            "volume",
            "market_cap",
            "total_assets",
            "total_equity",
            "current_assets",
            "current_liabilities",
            "inventory",
            "receivables",
            "cash",
            "short_debt",
            "long_debt",
            "invested_capital",
            "fundamental_scale",
            "scale_base",
            "expected_std",
        }:
            values = np.abs(values) * 1e6 + 1e6
        data[name] = pd.Series(values, index=index)

    # Expanded formulas can reference common symbols not explicit in the outer
    # recipe signature.
    for name in ("close", "open", "high", "low", "volume"):
        if name in data:
            continue
        values = base.copy()
        if name == "high":
            values += 1
        if name == "low":
            values -= 1
        if name == "open":
            values -= 0.2
        if name == "volume":
            values = np.abs(values) * 1e6
        data[name] = pd.Series(values, index=index)
    return InMemorySeriesSource(data=data)


def _plan_ops(plan) -> set[str]:
    names: set[str] = set()

    def visit(node) -> None:
        name = str(getattr(node, "op", "") or "")
        if name:
            names.add(name)
        for child in getattr(node, "inputs", ()) or ():
            visit(child)

    visit(plan)
    return names


def test_every_production_recipe_executes_on_certified_reference_path():
    os.environ["FACTOR_ENGINE_CERTIFY_RECIPE_EVIDENCE"] = "1"
    from backend.operator_capability import production_eligible_backends
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from factor_recipes import FactorRecipeRegistry

    load_all()
    recipes = [
        FactorRecipeRegistry.get(name)
        for name in FactorRecipeRegistry.list_names(status="production")
    ]
    recipes = [recipe for recipe in recipes if recipe is not None]
    source = _source(recipes)
    backend = build_backend("pandas")

    for recipe in recipes:
        request = {recipe.name: (recipe.name, _bindings(recipe))}
        batch = compile_recipe_plans(request)
        plan = batch.plans[recipe.name]
        for raw_name in _plan_ops(plan):
            if raw_name in {"column", "literal"}:
                continue
            canonical = OperatorRegistry.resolve_canonical_strict(raw_name)
            assert production_eligible_backends(canonical), (
                recipe.name,
                canonical,
                "recipe leaf has no evidence-backed production backend",
            )

        output = backend.execute(
            plan,
            ExecutionContext(data_source=source, run_mode="research"),
        )
        assert isinstance(output, pd.Series), recipe.name
        assert output.index.equals(source.data["close"].index), recipe.name


def test_production_recipe_names_are_unique_and_registered():
    from factor_recipes import FactorRecipeRegistry

    names = FactorRecipeRegistry.list_names(status="production")
    assert len(names) == len(set(names))
    assert names
