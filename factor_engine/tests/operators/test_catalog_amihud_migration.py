import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


def test_three_argument_amount_recipe_migrates_to_price_impact():
    source = "amihud_illiquidity(ret, amount, 20)"
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == "price_impact(ret, amount, 20)"
    assert result.changes
    assert migrate_catalog_recipe_formula(result.formula).formula == result.formula


def test_nested_amount_recipe_avoids_overlapping_primitive_edit():
    import ast

    result = migrate_catalog_recipe_formula(
        "ROC(amihud_illiquidity(ret, amount, 20), 5)"
    )
    ast.parse(result.formula, mode="eval")
    assert result.formula == "ROC(price_impact(ret, amount, 20), 5)"


def test_migrated_amount_recipe_compiles_with_three_parameter_contract():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    formula = migrate_catalog_recipe_formula(
        "amihud_illiquidity(ret, amount, 20)"
    ).formula
    ir = Analyzer().lower(DSLParser(surface="compat_research").parse(formula)).ir
    assert ir.op == "price_impact"
    assert len(ir.inputs) == 3
    assert ir.inputs[2].op == "literal"
    assert ir.inputs[2].attrs["value"] == 20


@pytest.mark.parametrize(
    "source",
    [
        "amihud_illiquidity(return_1d, amount, 20)",
        "amihud_illiquidity(ret, dollar_volume, 20)",
        "amihud_illiquidity(ret, amount, 20.0)",
        "amihud_illiquidity(ret, amount, True)",
        "amihud_illiquidity(ret, amount, window=20)",
        "amihud_illiquidity(ret, amount, *windows)",
        "amihud_illiquidity(ret, close, volume, 20)",
    ],
)
def test_amihud_recipe_migration_does_not_guess(source):
    assert migrate_catalog_recipe_formula(source).formula == source


def test_price_impact_is_numerically_the_amount_form_of_amihud():
    from factor_engine.cleaned_operators.price_volume.liquidity_v2 import (
        amihud_illiquidity,
        price_impact,
    )

    ret = pd.DataFrame({"a": [0.01, -0.02, np.nan, 0.03, 0.01, -0.01]})
    close = pd.DataFrame({"a": [10.0, 11.0, 12.0, 0.0, 9.0, 8.0]})
    volume = pd.DataFrame({"a": [100.0, 0.0, 120.0, 90.0, 80.0, 70.0]})
    amount = close.abs() * volume

    expected = amihud_illiquidity(ret, close, volume, 2)
    actual = price_impact(ret, amount, 2)
    pd.testing.assert_frame_equal(actual, expected)
