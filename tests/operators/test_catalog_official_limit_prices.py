import pytest
from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


@pytest.mark.parametrize("call", [
    "ashare_limit_down_streak(close, low_limit, subtract(1.0,is_suspend),0.005)",
    "ashare_limit_touch_count(high,low,high_limit,low_limit,20,'up',0.005)",
    "ashare_limit_down_touch(low,low_limit,0.005)",
    "ashare_failed_limit_count(high,low,close,high_limit,low_limit,20,'up',0.005)",
    "ashare_limit_up_touch(high,high_limit,0.005)",
    "ashare_limit_up_streak(close,high_limit,subtract(1.0,is_suspend),0.005)",
    "ashare_limit_failed(high,close,high_limit,0.0)",
])
def test_official_limit_prices_are_explicit_and_only_local(call):
    formula = f"add({call},ts_mean(close,20))"
    migrated = migrate_catalog_recipe_formula(formula)
    assert 'table="StockDailyBar"' in migrated.formula
    assert migrated.formula.endswith(",ts_mean(close,20))")
    assert migrate_catalog_recipe_formula(migrated.formula).formula == migrated.formula
    assert migrated.changes


def test_unknown_expressions_and_nonprice_slots_are_not_guessed():
    formula = "ashare_limit_down_streak(ts_mean(close,2),other,close,low)"
    assert migrate_catalog_recipe_formula(formula).formula == formula


def test_up_streak_rewrites_only_price_slots():
    formula = "add(ashare_limit_up_streak(close,high_limit,subtract(1.0,is_suspend),0.005),ts_mean(close,20))"
    migrated = migrate_catalog_recipe_formula(formula)
    assert migrated.formula.count('table="StockDailyBar"') == 2
    assert "subtract(1.0,is_suspend)" in migrated.formula
    assert migrated.formula.endswith(",ts_mean(close,20))")


@pytest.mark.parametrize(("source", "expected_count"), [
    ("ashare_limit_up_touch(high,high_limit,0.005)", 2),
    ("ashare_limit_up_streak(close,high_limit,valid_trade,0.005)", 2),
    ("ashare_limit_failed(high,close,high_limit,0.0)", 3),
])
def test_new_official_limit_calls_rewrite_exact_price_slots(source, expected_count):
    migrated = migrate_catalog_recipe_formula(source)
    assert migrated.formula.count('table="StockDailyBar"') == expected_count
    assert migrate_catalog_recipe_formula(migrated.formula).formula == migrated.formula


def test_nested_limit_leaf_migration_does_not_overlap_primitive_expansion():
    import ast
    source = "ROC(ashare_limit_up_touch(high,high_limit,0.005),5)"
    result = migrate_catalog_recipe_formula(source)
    ast.parse(result.formula, mode="eval")
    assert result.formula.startswith("ROC(ashare_limit_up_touch(")
    assert result.formula.count('table="StockDailyBar"') == 2


def test_raw_limit_binding_compiles_with_official_basis():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer
    load_all()
    formula = migrate_catalog_recipe_formula("ashare_limit_down_touch(low,low_limit,0.005)").formula
    ir = Analyzer(production=True, market="ashare").lower(
        DSLParser(surface="compat_research").parse(formula)
    ).ir
    from factor_engine.expr.column import ColumnRef
    from factor_engine.fields.resolver import resolve_market_field
    expr = DSLParser(surface="compat_research").parse(formula)
    refs = [child for child in expr.children() if isinstance(child, ColumnRef)]
    assert len(refs) == 2
    specs = [resolve_market_field(child, "ashare", strict=True).spec for child in refs]
    assert all(spec.table == "StockDailyBar" for spec in specs)
    assert [spec.price_basis for spec in specs] == ["RAW", "RAW_OFFICIAL_LIMIT"]
