import ast
import pytest
from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


@pytest.mark.parametrize(('call', 'count'), [
    ('ashare_limit_open_up_streak(open,high_limit,subtract(1.0,is_suspend),0.005)', 2),
    ('ashare_limit_open_down_streak(open,low_limit,subtract(1.0,is_suspend),0.005)', 2),
    ('ashare_limit_one_price(open,high,low,close,high_limit,low_limit,"up",0.005)', 6),
    ('ashare_limit_one_price(open=open,high=high,low=low,close=close,upper_limit=high_limit,lower_limit=low_limit)', 6),
])
def test_official_open_and_one_price_migration(call, count):
    source = f'add({call},ts_mean(close,20))'
    result = migrate_catalog_recipe_formula(source)
    assert result.formula.count('table="StockDailyBar"') == count
    assert result.formula.endswith(',ts_mean(close,20))')
    assert migrate_catalog_recipe_formula(result.formula).formula == result.formula
    ast.parse(result.formula, mode='eval')


def test_open_limit_never_guesses_expressions_or_nonprice_slots():
    source = 'ashare_limit_open_up_streak(ts_mean(open,2),custom_limit,close,high)'
    assert migrate_catalog_recipe_formula(source).formula == source


@pytest.mark.parametrize('call', [
    'ashare_limit_open_up_streak(open,high_limit,subtract(1.0,is_suspend),0.005)',
    'ashare_limit_open_down_streak(open,low_limit,subtract(1.0,is_suspend),0.005)',
    'ashare_limit_one_price(open,high,low,close,high_limit,low_limit,"up",0.005)',
])
def test_migrated_official_price_basis_passes_typed_ir(call):
    from factor_engine.cleaned_operators import load_all
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer
    load_all()
    expr = DSLParser(surface='compat_research').parse(migrate_catalog_recipe_formula(call).formula)
    Analyzer(production=True, market='ashare').lower(expr)
