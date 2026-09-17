import pytest
from factor_engine.api.columns import field
from factor_engine.fields.resolver import resolve_market_field

@pytest.mark.parametrize("stem", ["open", "high", "low", "close", "volume", "amount", "vwap"])
def test_catalog_uppercase_minute_alias_is_exact_adjusted_identity(stem):
    alias = "Minute" + stem.capitalize()
    ref = field(alias)
    expected = field("minute_" + stem, table="StockMinuteBarAdj")
    assert ref.field_id == expected.field_id
    assert ref.source_name == expected.source_name
    assert resolve_market_field(ref, "ashare", strict=True).spec == resolve_market_field(expected, "ashare", strict=True).spec

def test_catalog_pcf_ratio2_spelling_preserves_second_cashflow_ratio():
    ref = field("pcf_ratio2")
    assert ref.table == "StockValuationDaily"
    assert ref.source_name == "PcfRatio2"
    assert ref.field_id == field("pcf_ratio_2").field_id
    assert ref.field_id != field("pcf_ratio").field_id


@pytest.mark.parametrize(
    ("alias", "canonical", "table", "source_name"),
    [
        ("sale_expense", "selling_expense", "StockIncome", "SaleExpense"),
        ("constru_in_process", "construction_in_progress", "StockBalance", "ConstruInProcess"),
        ("good_will", "goodwill", "StockBalance", "GoodWill"),
        ("taxs_payable", "taxes_payable", "StockBalance", "TaxsPayable"),
        ("deferred_tax_liability", "deferred_tax_liabilities", "StockBalance", "DeferredTaxLiability"),
        ("net_operate_cash_flow", "operating_cash_flow", "StockCashFlow", "NetOperateCashFlow"),
        ("net_invest_cash_flow", "investing_cash_flow", "StockCashFlow", "NetInvestCashFlow"),
        ("net_finance_cash_flow", "financing_cash_flow", "StockCashFlow", "NetFinanceCashFlow"),
        ("fix_intan_other_asset_acqui_cash", "fix_intan_other_asset_acquis_cash", "StockCashFlow", "FixIntanOtherAssetAcquiCash"),
        ("staff_cash_paid", "staff_behalf_paid", "StockCashFlow", "StaffBehalfPaid"),
    ],
)
def test_catalog_legacy_financial_spelling_is_exact_field_identity(alias, canonical, table, source_name):
    ref = field(alias)
    expected = field(canonical)
    assert ref.field_id == expected.field_id
    assert ref.table == table
    assert ref.source_name == source_name
    assert resolve_market_field(ref, "ashare", strict=True).spec == resolve_market_field(expected, "ashare", strict=True).spec
