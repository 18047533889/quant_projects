import pytest

from factor_engine.tools.catalog_r19_financial_field_recipes import migrate_formula


@pytest.mark.parametrize("legacy,table,field", [
    ("RdExpenses_SP", "StockIncome", "rd_expenses"),
    ("OperatingProfit_SP", "StockIncome", "operating_profit"),
    ("BorrowingRepayment_SP", "StockCashFlow", "borrowing_repayment"),
    ("FixIntanOtherAssetAcquiCash_SP", "StockCashFlow", "fix_intan_other_asset_acquis_cash"),
    ("TaxPayments_SP", "StockCashFlow", "tax_payments"),
])
def test_registered_cumulative_fields_expand_to_same_table_quarters(legacy, table, field):
    got, changes = migrate_formula(legacy)
    period = f'field("report_period_end_date", table="{table}")'
    assert got == f'fin_quarter_from_cumulative(field("{field}", table="{table}"), {period}, ashare_fiscal_quarter_from_period_end({period}))'
    assert changes and legacy in changes[0] and f"{table}.{field}" in changes[0]
    assert migrate_formula(got) == (got, [])


def test_average_equity_uses_registered_balance_field_and_period():
    got, changes = migrate_formula("safe_div_null(NetProfit, FiscalAvgEquity)")
    assert got == 'safe_div_null(NetProfit, fin_average_balance(field("total_owner_equities", table="StockBalance"), field("report_period_end_date", table="StockBalance")))'
    assert "adjacent fiscal-period average equity" in changes[0]


def test_multiple_tables_are_explicit_and_unknown_shorthand_fails_closed():
    source = "add(RdExpenses_SP, BorrowingRepayment_SP)"
    got, changes = migrate_formula(source)
    assert 'table="StockIncome"' in got and 'table="StockCashFlow"' in got
    assert len(changes) == 2
    assert migrate_formula("UnknownFlow_SP") == ("UnknownFlow_SP", [])
    assert migrate_formula("RdExpenses_SP(x)") == ("RdExpenses_SP(x)", [])


def test_invalid_or_oversized_inputs_fail_closed_or_bounded():
    assert migrate_formula("RdExpenses_SP(") == ("RdExpenses_SP(", [])
    with pytest.raises(ValueError): migrate_formula("x" * 65_537)


@pytest.mark.parametrize("source", [
    "RdExpenses_SP",
    "BorrowingRepayment_SP",
    "safe_div_null(OperatingProfit_SP, FiscalAvgEquity)",
])
def test_repaired_fields_bind_and_compile_without_reads(source):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators import load_all
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.sources.datasource import DataSource

    class NoRead(DataSource):
        def _no(self, *args, **kwargs): raise AssertionError("compile attempted data read")
        load_column = load_columns = prefetch_columns = scan_polars_long = scan_index_long = _no

    load_all()
    formula, changes = migrate_formula(source)
    assert changes
    expr = DSLParser(surface="compat_research").parse(formula)
    FactorEngine(PandasBackend(), NoRead(), run_mode="research").compile(
        Factor(name="r19-financial-field", expr=expr, source_expr=formula, surface="compat_research")
    )
