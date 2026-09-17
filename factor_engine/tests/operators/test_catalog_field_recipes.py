from factor_engine.tools.catalog_field_recipes import migrate_catalog_field_formula


def test_canonical_financial_derivations_follow_visible_revisions():
    import numpy as np
    import pandas as pd

    from factor_engine.cleaned_operators.fundamental.flow_semantics_v2 import (
        fin_quarter_from_cumulative,
    )
    from factor_engine.cleaned_operators.fundamental.transforms_v2 import (
        fin_average_balance,
    )

    index = pd.date_range("2024-04-01", periods=4, freq="B")
    period = pd.DataFrame({"A": ["2023Q1", "2023Q1", "2023Q2", "2023Q2"]}, index=index)
    quarter = pd.DataFrame({"A": [1, 1, 2, 2]}, index=index)
    cumulative = pd.DataFrame({"A": [10.0, 10.0, 30.0, 32.0]}, index=index)
    assets = pd.DataFrame({"A": [100.0, 110.0, 120.0, 130.0]}, index=index)

    np.testing.assert_allclose(
        fin_quarter_from_cumulative(cumulative, period, quarter)["A"],
        [10.0, 10.0, 20.0, 22.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        fin_average_balance(assets, period)["A"],
        [np.nan, np.nan, 115.0, 120.0],
        equal_nan=True,
    )


def test_disabled_field_recipe_is_source_preserving():
    source = "fiscal_pct_change(Inventories,FiscalPeriodId)"
    assert migrate_catalog_field_formula(
        source, tables="StockBalance", enabled=False
    ).formula == source


def test_balance_fiscal_period_and_average_assets_are_derived_from_same_table():
    source = (
        "fin_delta_noa(TotalAssets,CashEquivalents,TotalLiability,ShorttermLoan,"
        "LongtermLoan,FiscalAvgAssets,FiscalPeriodId)"
    )
    result = migrate_catalog_field_formula(source, tables="StockBalance", enabled=True)
    period = 'field("report_period_end_date", table="StockBalance")'
    assets = 'field("total_assets", table="StockBalance")'
    assert result.formula == (
        "fin_delta_noa(TotalAssets,CashEquivalents,TotalLiability,ShorttermLoan,"
        f"LongtermLoan,fin_average_balance({assets}, {period}),{period})"
    )
    assert len(result.changes) == 2
    assert all("ORIGINAL=" in change and "DECISION=" in change and "NEW=" in change
               for change in result.changes)


def test_income_single_period_flow_uses_canonical_quarter_conversion():
    source = "subtract(OperatingRevenue_SP,OperatingCost_SP)"
    result = migrate_catalog_field_formula(source, tables="StockIncome", enabled=True)
    period = 'field("report_period_end_date", table="StockIncome")'
    assert result.formula == (
        'subtract(fin_quarter_from_cumulative(field("operating_revenue", '
        f'table="StockIncome"), {period}, ashare_fiscal_quarter_from_period_end({period})),'
        'fin_quarter_from_cumulative(field("operating_cost", table="StockIncome"), '
        f'{period}, ashare_fiscal_quarter_from_period_end({period})))'
    )


def test_stock_indicator_dates_are_qualified_only_with_reviewed_context():
    source = "report_filing_delay_surprise(PubDate,ReportPeriodEndDate)"
    qualified = migrate_catalog_field_formula(
        source, tables="StockIndicator|StockMinuteBar", enabled=True
    )
    assert qualified.formula == (
        'report_filing_delay_surprise(field("pub_date", table="StockIndicator"),'
        'field("report_period_end_date", table="StockIndicator"))'
    )
    assert migrate_catalog_field_formula(source, tables="", enabled=True).changes == ()


def test_unavailable_or_identity_ambiguous_fields_fail_closed():
    source = "add(add(Top10PledgeRatio,IndexReturn),IndexWeight)"
    result = migrate_catalog_field_formula(
        source,
        tables="StockTopTenShareholder|IndexDailyBar|IndexConstituent",
        enabled=True,
    )
    assert result.formula == source
    assert result.changes == ()


def test_mixed_table_unknown_fiscal_call_does_not_guess_balance_period():
    source = "unknown_fiscal_recipe(x,FiscalPeriodId)"
    result = migrate_catalog_field_formula(
        source, tables="StockBalance|StockIncome", enabled=True
    )
    assert result.formula == source
    assert result.changes == ()


def test_legacy_field_spelling_used_as_call_target_is_not_rewritten():
    source = "OperatingRevenue_SP(x)"
    result = migrate_catalog_field_formula(
        source, tables="StockIncome", enabled=True
    )
    assert result.formula == source
    assert result.changes == ()


def test_table_context_tokens_are_trimmed():
    result = migrate_catalog_field_formula(
        "fiscal_pct_change(Inventories,FiscalPeriodId)",
        tables=" StockBalance | StockIndicator ",
        enabled=True,
    )
    assert 'table="StockBalance"' in result.formula
