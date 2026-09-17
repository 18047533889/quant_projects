import ast
import pytest
from factor_engine.tools.catalog_migration import migrate_adjusted_price_fields

def migrate(value):
    return migrate_adjusted_price_fields(value, market="ashare")

def test_lowercase_and_economic_parameters_are_unchanged():
    text = "rank(ts_mean(close, 20) / volume)"
    assert migrate(text).formula == text
    assert not migrate(text).changes

@pytest.mark.parametrize("old", ["Close", "StockDailyBar.Close", "DailyBar.Close", "StockDailyBar().Close", 'col("Close")', 'field("Close")', 'field("Close", table="StockDailyBar")'])
def test_explicit_legacy_close_becomes_adjusted_field(old):
    result = migrate(old)
    assert ast.dump(ast.parse(result.formula)) == ast.dump(ast.parse("field('close', table='StockDailyBarAdj')"))
    assert result.changes
    assert not migrate(result.formula).changes

def test_other_table_qualified_data_and_strings_not_rewritten():
    text = "op(StockIncome.NetProfit, label='Close', window=20)"
    assert migrate(text).formula == text

def test_function_names_are_not_interpreted_as_fields():
    assert migrate("Close(x)").formula == "Close(x)"

def test_control_flags_preserved():
    out = migrate("field('close', table='StockDailyBar', strict=True, for_mining=True)").formula
    assert "strict=True" in out and "for_mining=True" in out
    assert "StockDailyBarAdj" in out

def test_filter_parameters_not_dropped():
    assert migrate("StockDailyBar(universe='x').Close").formula == "StockDailyBar(universe='x').Close"

def test_no_market_guess_or_oversized_parse():
    with pytest.raises(ValueError, match="ashare"):
        migrate_adjusted_price_fields("Close", market="us")
    with pytest.raises(ValueError, match="budget"):
        migrate(" " * 65537)
