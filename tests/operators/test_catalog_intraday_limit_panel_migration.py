import ast

import pytest

from factor_engine.tools.catalog_migration import migrate_adjusted_price_fields
from factor_engine.tools.catalog_recipe_migration import (
    migrate_catalog_recipe_formula,
    redesign_catalog_intraday_limit_formula,
)


@pytest.mark.parametrize(
    ("legacy", "canonical", "scalar_tail"),
    [
        ("intraday_limit_first_hit_time", "intra_limit_first_hit_time", ', "up"'),
        ("intraday_limit_duration", "intra_limit_duration", ', "up"'),
        ("intraday_limit_reopen_count", "intra_limit_reopen_count", ', "up", "open"'),
    ],
)
def test_exact_three_panel_limit_recipe_uses_raw_sources(
    legacy, canonical, scalar_tail
):
    source = f"{legacy}(MinuteClose,HighLimit,LowLimit)"
    redesigned = redesign_catalog_intraday_limit_formula(source, enabled=True)
    adjusted = migrate_adjusted_price_fields(redesigned.formula, market="ashare").formula
    result = migrate_catalog_recipe_formula(adjusted)

    assert result.formula.startswith(f"{canonical}(")
    assert 'field("minute_close", table="StockMinuteBar")' in result.formula
    assert result.formula.count('table="StockDailyBar"') == 2
    assert "StockDailyBarAdj" not in result.formula
    assert result.formula.endswith(scalar_tail + ")")
    assert migrate_catalog_recipe_formula(result.formula).formula == result.formula
    assert redesigned.changes and redesigned.changes[0].startswith("SEMANTIC_REDESIGN")
    ast.parse(result.formula, mode="eval")


def test_first_hit_uses_real_raw_minute_ohl_for_current_native_touch_definition():
    result = redesign_catalog_intraday_limit_formula(
        "intraday_limit_first_hit_time(MinuteClose,HighLimit,LowLimit)", enabled=True
    )
    for field in ("minute_close", "minute_high", "minute_low"):
        assert f'field("{field}", table="StockMinuteBar")' in result.formula
    assert "touch=HIGH_FOR_UP/LOW_FOR_DOWN" in result.changes[0]


def test_semantic_redesign_is_disabled_by_default():
    source = "intraday_limit_duration(MinuteClose,HighLimit,LowLimit)"
    assert redesign_catalog_intraday_limit_formula(source).formula == source
    assert migrate_catalog_recipe_formula(source).formula == source


@pytest.mark.parametrize(
    "source",
    [
        "intraday_limit_duration(ts_mean(MinuteClose,2),HighLimit,LowLimit)",
        "intraday_limit_duration(MinuteClose,custom_limit,LowLimit)",
        "intraday_limit_duration(MinuteClose,HighLimit,LowLimit,\"down\")",
        "intraday_limit_duration(close=MinuteClose,high_limit=HighLimit,low_limit=LowLimit)",
    ],
)
def test_intraday_limit_recipe_does_not_guess_nonexact_signatures(source):
    assert redesign_catalog_intraday_limit_formula(source, enabled=True).formula == source


@pytest.mark.parametrize(
    "source",
    [
        "intraday_limit_first_hit_time(MinuteClose,HighLimit,LowLimit)",
        "intraday_limit_duration(MinuteClose,HighLimit,LowLimit)",
        "intraday_limit_reopen_count(MinuteClose,HighLimit,LowLimit)",
    ],
)
def test_migrated_intraday_limit_recipe_lowers_with_cross_grain_contract(source):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.cleaned_operators import load_all
    from factor_engine.ir.analyzer import Analyzer

    load_all()
    migrated = redesign_catalog_intraday_limit_formula(source, enabled=True).formula
    expr = DSLParser(surface="compat_research").parse(migrated)
    lowered = Analyzer(production=False, market="ashare").lower(expr)
    assert lowered.ir is not None
