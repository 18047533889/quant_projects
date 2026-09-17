import ast
import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.ashare_fiscal_period import ashare_fiscal_quarter_from_period_end
from factor_engine.cleaned_operators.fundamental.flow_semantics_v2 import fin_quarter_from_cumulative
from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


def test_quarter_end_mapping_is_strict():
    periods = pd.DataFrame({"A": pd.to_datetime([
        "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
        "2024-06-29", None,
    ])})
    result = ashare_fiscal_quarter_from_period_end(periods)["A"].to_numpy()
    np.testing.assert_allclose(result[:4], [1, 2, 3, 4])
    assert np.isnan(result[4:]).all()
    encoded = pd.DataFrame({"A": [20240331, "20240630", 20240629, 202403, True]})
    encoded_result = ashare_fiscal_quarter_from_period_end(encoded)["A"].to_numpy()
    np.testing.assert_allclose(encoded_result[:2], [1, 2])
    assert np.isnan(encoded_result[2:]).all()


def test_single_period_oracle_missing_predecessor_and_revision():
    idx = pd.date_range("2024-04-30", periods=5)
    period = pd.DataFrame({"A": pd.to_datetime([
        "2024-03-31", "2024-06-30", "2024-09-30", "2024-06-30", "2025-03-31"
    ])}, index=idx)
    cumulative = pd.DataFrame({"A": [10.0, 25.0, 50.0, 28.0, 12.0]}, index=idx)
    quarter = ashare_fiscal_quarter_from_period_end(period)
    result = fin_quarter_from_cumulative(cumulative, period, quarter)["A"]
    # Q1 passes through; Q2 subtracts the visible Q1. Q3 is 50-25. A later
    # revision of Q2 updates that period only and never rewrites the earlier Q3.
    np.testing.assert_allclose(result.iloc[[0, 1, 2, 3, 4]], [10, 15, 25, 18, 12])
    missing_period = period.iloc[[1]].copy()
    missing_value = cumulative.iloc[[1]].copy()
    missing_quarter = quarter.iloc[[1]].copy()
    assert np.isnan(fin_quarter_from_cumulative(missing_value, missing_period, missing_quarter).iloc[0, 0])


def test_only_exact_supported_same_table_recipes_migrate():
    revenue = migrate_catalog_recipe_formula(
        "fiscal_standardized_surprise(OperatingRevenue_SP,FiscalPeriodId)"
    )
    assert "StockIncome" in revenue.formula
    assert "fin_quarter_from_cumulative" in revenue.formula
    cash = migrate_catalog_recipe_formula(
        "fiscal_standardized_surprise(NetOperateCashFlow_SP, FiscalPeriodId)"
    )
    assert "StockCashFlow" in cash.formula
    assert "fin_quarter_from_cumulative" in cash.formula
    assert migrate_catalog_recipe_formula("fiscal_standardized_surprise(x,FiscalPeriodId)").changes == ()
    assert migrate_catalog_recipe_formula("add(OperatingRevenue_SP,NetOperateCashFlow_SP)").changes == ()


def test_fiscal_rewrite_is_parseable_idempotent_and_overlap_safe():
    source = "MOM(fiscal_standardized_surprise(OperatingRevenue_SP,FiscalPeriodId), 4)"
    first = migrate_catalog_recipe_formula(source)
    ast.parse(first.formula, mode="eval")
    # The outer primitive is intentionally deferred rather than crossing the
    # inner whole-call replacement span.
    assert first.formula.startswith("MOM(")
    assert "fin_quarter_from_cumulative" in first.formula
    second = migrate_catalog_recipe_formula(first.formula)
    ast.parse(second.formula, mode="eval")
    assert second.formula != first.formula
    third = migrate_catalog_recipe_formula(second.formula)
    assert third.formula == second.formula


def test_registry_selected_quarterization_matches_oracle():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    idx = pd.date_range("2024-04-30", periods=2)
    periods = pd.DataFrame({"A": pd.to_datetime(["2024-03-31", "2024-06-30"])}, index=idx)
    cumulative = pd.DataFrame({"A": [10.0, 25.0]}, index=idx)
    quarter_op = OperatorRegistry.get("ashare_fiscal_quarter_from_period_end", "pandas_numpy")
    convert_op = OperatorRegistry.get("fin_quarter_from_cumulative", "pandas_numpy")
    quarters = quarter_op.calculate(periods)
    result = convert_op.calculate(cumulative, periods, quarters)
    np.testing.assert_allclose(result["A"], [10.0, 15.0])
