from __future__ import annotations


def test_average_volume_field_and_operator_are_unambiguous():
    from cleaned_operators import load_all
    from cleaned_operators.layer_governance import formula_field_names
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    fields = formula_field_names()
    assert "average_volume" in fields
    assert "average_volume" not in set(OperatorRegistry.list_canonical())
    assert "average_volume" not in OperatorRegistry._aliases
    assert "ts_average_volume" in set(OperatorRegistry.list_canonical())


def test_collision_free_average_volume_operator_parses_and_executes():
    import numpy as np
    import pandas as pd

    from api.dsl_parser import parse_factor
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from ir.analyzer import Analyzer

    load_all()
    factor = parse_factor(
        "ts_average_volume(volume,3)",
        name="liquidity::average_volume",
        surface="extended",
    )
    analysis = Analyzer().lower(factor.expr)
    assert analysis.ir.op == "ts_average_volume"
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    volume = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    output = OperatorRegistry.get("ts_average_volume").calculate(volume, 3)
    np.testing.assert_allclose(
        output["A"].to_numpy(), [np.nan, np.nan, 2.0, 3.0, 4.0], equal_nan=True
    )


def test_extended_cold_start_uses_collision_free_name():
    from factor_cold_start.catalog import load_catalog

    for market in ("ashare", "us"):
        rows = load_catalog(market, "extended")
        formulas = {row.formula for row in rows}
        assert "ts_average_volume(volume,20)" in formulas
        assert "average_volume(volume,20)" not in formulas
