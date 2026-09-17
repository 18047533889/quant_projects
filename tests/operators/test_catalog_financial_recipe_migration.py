from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("current_ratio(Assets,Liabilities)", "fin_ratio(Assets,Liabilities)"),
        ("debt_to_equity(Debt, Equity)", "fin_ratio(Debt, Equity)"),
        ("operating_margin(Profit, Revenue)", "fin_ratio(Profit, Revenue)"),
        (
            "quick_ratio(Assets,Inventory,Liabilities)",
            "fin_ratio(subtract(Assets, Inventory), Liabilities)",
        ),
    ],
)
def test_exact_financial_recipe_calls_migrate(source: str, expected: str) -> None:
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == expected
    assert len(result.changes) == 1


@pytest.mark.parametrize(
    "source",
    [
        "current_ratio(Assets)",
        "current_ratio(Assets, Liabilities, Period)",
        "current_ratio(current_assets=Assets, current_liabilities=Liabilities)",
        "quick_ratio(Assets, Inventory)",
        "quick_ratio(Assets, Inventory, Liabilities, Period)",
        "quick_ratio(Assets - Other, Inventory, Liabilities)",
        "namespace.operating_margin(Profit, Revenue)",
        "field('debt_to_equity')",
    ],
)
def test_ambiguous_or_non_catalog_financial_forms_are_preserved(source: str) -> None:
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == source
    assert result.changes == ()


def test_recipe_helpers_match_legacy_zero_and_missing_semantics() -> None:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.fundamental.ops import (
        CurrentRatioOp,
        DebtToEquityOp,
        OperatingMarginOp,
        QuickRatioOp,
    )
    from factor_engine.factor_recipes.fundamental_ratios import (
        current_ratio,
        debt_to_equity,
        operating_margin,
        quick_ratio,
    )

    load_all()
    numerator = pd.DataFrame({"A": [12.0, np.nan, 9.0, 8.0]})
    denominator = pd.DataFrame({"A": [3.0, 2.0, 0.0, np.nan]})
    inventory = pd.DataFrame({"A": [2.0, 1.0, 3.0, np.nan]})

    pd.testing.assert_frame_equal(
        CurrentRatioOp().calculate(numerator, denominator),
        current_ratio(numerator, denominator),
    )
    pd.testing.assert_frame_equal(
        DebtToEquityOp().calculate(numerator, denominator),
        debt_to_equity(numerator, denominator),
    )
    pd.testing.assert_frame_equal(
        OperatingMarginOp().calculate(numerator, denominator),
        operating_margin(numerator, denominator),
    )
    pd.testing.assert_frame_equal(
        QuickRatioOp().calculate(numerator, inventory, denominator),
        quick_ratio(numerator, inventory, denominator),
    )
