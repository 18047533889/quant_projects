# -*- coding: utf-8 -*-
"""Test profitability_derived recipes can expand correctly."""
import os

import pytest

from factor_engine.factor_recipes.registry import FactorRecipeRegistry


@pytest.mark.parametrize(
    "recipe_name,bindings",
    [
        ("gross_profit_derived", {"gross_profit": "gross_profit"}),
        (
            "gross_margin_derived",
            {"gross_profit": "gross_profit", "operating_revenue": "operating_revenue"},
        ),
        (
            "operating_margin_derived",
            {"operating_profit": "operating_profit", "operating_revenue": "operating_revenue"},
        ),
        (
            "net_margin_derived",
            {"net_profit": "net_profit", "operating_revenue": "operating_revenue"},
        ),
        (
            "industry_peer_mean",
            {"metric": "metric", "industry_code": "industry_code"},
        ),
    ],
)
def test_profitability_derived_recipe_expansion(recipe_name, bindings, monkeypatch):
    """Verify Phase 0 derived recipes can expand without error."""
    # Allow expansion without evidence for new recipes
    monkeypatch.setenv("FACTOR_ENGINE_EXPAND_RECIPE_USAGE", "1")

    recipe = FactorRecipeRegistry.get(recipe_name)
    assert recipe is not None
    assert recipe.status == "production"

    # Expand should not raise
    expression = FactorRecipeRegistry.expand(recipe_name, bindings)
    assert len(expression) > 0


def test_profitability_derived_all_registered():
    """Verify all 5 profitability_derived recipes are registered."""
    expected = {
        "gross_profit_derived",
        "gross_margin_derived",
        "operating_margin_derived",
        "net_margin_derived",
        "industry_peer_mean",
    }
    actual = set(FactorRecipeRegistry.list_names(status="production"))
    assert expected.issubset(actual), f"Missing: {expected - actual}"
