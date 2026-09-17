import ast
import pytest
from factor_engine.tools.catalog_migration import migrate_adjusted_price_fields
from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


def pipeline(source):
    return migrate_catalog_recipe_formula(migrate_adjusted_price_fields(source, market='ashare').formula).formula


@pytest.mark.parametrize('source', [
    'ashare_limit_open_up_streak(open,high_limit,subtract(1,is_suspend),0.005)',
    'ashare_limit_open_down_streak(open,low_limit,subtract(1,is_suspend),0.005)',
    'ashare_limit_one_price(open,high,low,close,high_limit,low_limit,"up",0.005)',
    'ashare_limit_down_touch(low=low,low_limit=low_limit,tick_tolerance=0.005)',
])
def test_corrected_official_price_dsl_survives_full_pipeline_reimport(source):
    once = pipeline(source)
    twice = pipeline(once)
    assert ast.dump(ast.parse(twice)) == ast.dump(ast.parse(once))
    assert 'StockDailyBarAdj' not in twice


@pytest.mark.parametrize('leaf', ['Low', 'StockDailyBar.Low', 'DailyBar().Low', 'field("Low",table="StockDailyBar",strict=True)'])
def test_legacy_raw_price_inside_limit_call_is_not_adjusted(leaf):
    result = ast.parse(pipeline(f'add(ashare_limit_down_touch({leaf},LowLimit,0.005),Close)'), mode='eval')
    limit, outside = result.body.args
    for price in limit.args[:2]:
        assert isinstance(price, ast.Call)
        assert next(k.value.value for k in price.keywords if k.arg=='table') == 'StockDailyBar'
    assert next(k.value.value for k in outside.keywords if k.arg=='table') == 'StockDailyBarAdj'
    if leaf.startswith('field'):
        assert next(k.value.value for k in limit.args[0].keywords if k.arg=='strict') is True


def test_nonprice_slot_still_follows_regular_adjusted_price_migration():
    result = ast.parse(pipeline('ashare_limit_open_up_streak(open,high_limit,Close,0.005)'), mode='eval')
    assert next(k.value.value for k in result.body.args[2].keywords if k.arg=='table') == 'StockDailyBarAdj'
