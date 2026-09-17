import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r19_context_recipes import migrate_formula


@pytest.mark.parametrize(
    ("source", "expected_fragment", "change_fragment"),
    [
        ("ts_mean(atr_pct, 5)", "atr_pct(high, low, close, window=20)", "20-session"),
        ("add(overnight_ret, intraday_ret)", "overnight_return(open, pre_close)", "prior close"),
        ("rank(vwap_dist)", "vwap_distance_pct(close, volume, window=20)", "daily volume"),
        ("ts_mean(rolling_vwap, 5)", "rolling_vwap(close, volume, window=20)", "placeholder"),
        ("event_historical_response_mean(ret, volume_shock)",
         "gt(volume_shock(volume, window=20), 2.0)", "formal MaskBool"),
        ("event_historical_response_mean(ret, turnover_shock)",
         "gt(turnover_shock(turnover_ratio, window=20), 2.0)", "formal MaskBool"),
    ],
)
def test_reviewed_placeholders_get_explicit_definitions(source, expected_fragment, change_fragment):
    formula, changes = migrate_formula(source)
    assert expected_fragment in formula
    assert any(change_fragment in change for change in changes)
    assert migrate_formula(formula) == (formula, [])


def test_real_operator_calls_and_unconstructable_placeholders_are_unchanged():
    for formula in (
        "atr_pct(high, low, close, window=20)",
        "rolling_vwap(close, volume, window=20)",
        "volume_shock(volume, window=20)",
        "add(oi_spread, edge_spread)",
        "add(chip_profit_share, chip_mode_distance)",
        "event_historical_response_mean(ret, index_entry)",
        "event_historical_response_mean(ret, capital_supply_shock)",
        "add(gap_up, limit_up_touch)",
    ):
        assert migrate_formula(formula) == (formula, [])


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("limit_up_touch", "ashare_limit_up_touch(field('high', table='StockDailyBar')"),
        ("limit_down_touch", "ashare_limit_down_touch(field('low', table='StockDailyBar')"),
        ("failed_limit", "ashare_limit_failed(field('high', table='StockDailyBar')"),
        ("gap_up", "gt(overnight_return(open, pre_close), 0.0)"),
        ("gap_down", "lt(overnight_return(open, pre_close), 0.0)"),
        ("suspension_resume", "and_(eq(is_suspend, 0.0), eq(delay(is_suspend, 1), 1.0))"),
        ("st_transition", "ne(is_st, delay(is_st, 1))"),
    ],
)
def test_event_placeholders_expand_only_in_explicit_event_keyword(name, fragment):
    source = f"event_historical_response_mean(response=ret, event={name})"
    formula, changes = migrate_formula(source)
    assert fragment in formula
    assert len(changes) == 1
    assert "SEMANTIC_DEFINITION" in changes[0]
    assert migrate_formula(formula) == (formula, [])


@pytest.mark.parametrize("name", [
    "limit_up_touch", "limit_down_touch", "failed_limit", "gap_up", "gap_down",
    "suspension_resume", "st_transition",
])
def test_event_definitions_parse_lower_bind_and_produce_boolean(name):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = f"event_historical_response_mean(response=ret, event={name})"
    formula, _ = migrate_formula(source)
    result = Analyzer().lower(DSLParser(surface="compat_research").parse(formula))
    assert result.ir.inputs[1].semantic_attrs["semantic_kind"] in {"MaskBool", "EventBool"}
    assert result.referenced_columns


def test_migrated_contexts_parse_lower_bind_and_have_boolean_events():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    sources = (
        "add(atr_pct, add(overnight_ret, intraday_ret))",
        "add(vwap_dist, safe_div_null(close, rolling_vwap))",
        "event_historical_response_mean(ret, volume_shock)",
        "event_historical_response_mean(ret, turnover_shock)",
    )
    for source in sources:
        formula, _ = migrate_formula(source)
        result = Analyzer().lower(DSLParser(surface="compat_research").parse(formula))
        assert result.referenced_columns
        if "historical_response" in formula:
            assert result.ir.inputs[1].semantic_attrs["semantic_kind"] == "MaskBool"


def test_definitions_execute_with_expected_numeric_oracles():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    idx = pd.date_range("2025-01-01", periods=80)
    close = pd.DataFrame({"A": np.linspace(100.0, 120.0, 80)}, index=idx)
    open_px = close * 0.99
    pre_close = close.shift(1)
    high, low = close * 1.01, close * 0.98
    volume = pd.DataFrame({"A": np.r_[100.0 + np.arange(79) % 5, 1000.0]}, index=idx)
    turnover = volume / 10_000.0

    overnight = OperatorRegistry.get("overnight_return", "pandas_numpy").calculate(
        open_px, pre_close, price_basis="continuous")
    intraday = OperatorRegistry.get("open_close_return", "pandas_numpy").calculate(
        open_px, close, price_basis="continuous")
    assert np.isclose(overnight.iloc[-1, 0], open_px.iloc[-1, 0] / pre_close.iloc[-1, 0] - 1)
    assert np.isclose(intraday.iloc[-1, 0], close.iloc[-1, 0] / open_px.iloc[-1, 0] - 1)

    atr = OperatorRegistry.get("atr_pct", "pandas_numpy").calculate(high, low, close, window=20)
    vwap = OperatorRegistry.get("rolling_vwap", "pandas_numpy").calculate(close, volume, window=20)
    dist = OperatorRegistry.get("vwap_distance_pct", "pandas_numpy").calculate(
        close, volume, window=20)
    assert np.isfinite(atr.iloc[-1, 0])
    assert np.isclose(dist.iloc[-1, 0], (close.iloc[-1, 0] - vwap.iloc[-1, 0]) / close.iloc[-1, 0])

    for name, panel in (("volume_shock", volume), ("turnover_shock", turnover)):
        shock = OperatorRegistry.get(name, "pandas_numpy").calculate(panel, window=20)
        mask = OperatorRegistry.get("gt", "pandas_numpy").calculate(shock, 2.0)
        assert mask.iloc[-1, 0] == 1.0


def test_limit_gap_and_state_event_definitions_have_numeric_oracles():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    idx = pd.date_range("2025-01-01", periods=3)
    upper = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=idx)
    lower = pd.DataFrame({"A": [8.0, 8.0, 8.0]}, index=idx)
    high = pd.DataFrame({"A": [9.0, 10.0, 10.0]}, index=idx)
    low = pd.DataFrame({"A": [9.0, 8.0, 9.0]}, index=idx)
    close = pd.DataFrame({"A": [9.0, 10.0, 9.5]}, index=idx)
    up = OperatorRegistry.get("ashare_limit_up_touch", "pandas_numpy").calculate(
        high, upper, 0.005)
    down = OperatorRegistry.get("ashare_limit_down_touch", "pandas_numpy").calculate(
        low, lower, 0.005)
    failed = OperatorRegistry.get("ashare_limit_failed", "pandas_numpy").calculate(
        high, close, upper, 0.005)
    assert up.iloc[:, 0].tolist() == [0.0, 1.0, 1.0]
    assert down.iloc[:, 0].tolist() == [0.0, 1.0, 0.0]
    assert failed.iloc[:, 0].tolist() == [0.0, 0.0, 1.0]

    open_px = pd.DataFrame({"A": [101.0, 99.0, 100.0]}, index=idx)
    pre_close = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
    overnight = OperatorRegistry.get("overnight_return", "pandas_numpy").calculate(
        open_px, pre_close, price_basis="continuous")
    gap_up = OperatorRegistry.get("gt", "pandas_numpy").calculate(overnight, 0.0)
    gap_down = OperatorRegistry.get("lt", "pandas_numpy").calculate(overnight, 0.0)
    assert gap_up.iloc[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert gap_down.iloc[:, 0].tolist() == [0.0, 1.0, 0.0]

    suspended = pd.Series([1.0, 0.0, 0.0], index=idx)
    is_st = pd.Series([0.0, 0.0, 1.0], index=idx)
    resume = ((suspended == 0.0) & (suspended.shift(1) == 1.0)).astype(float)
    transition = (is_st != is_st.shift(1)).astype(float).where(is_st.shift(1).notna())
    assert resume.tolist() == [0.0, 1.0, 0.0]
    assert transition.iloc[1:].tolist() == [0.0, 1.0]


def test_invalid_and_oversized_inputs_are_bounded():
    malformed = "event_historical_response_mean("
    assert migrate_formula(malformed) == (malformed, [])
    with pytest.raises(ValueError, match="input budget"):
        migrate_formula("x" * 65_537)
    with pytest.raises(TypeError):
        migrate_formula("x", logic=None)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ashare_limit_up_touch(field('open', table='StockDailyBar'), "
         "field('close', table='StockDailyBar'), field('high_limit', "
         "table='StockDailyBarAdj'), field('low_limit', table='StockDailyBarAdj'))",
         "ashare_limit_up_touch(field('high', table='StockDailyBar'), "
         "field('high_limit', table='StockDailyBar'), 0.005)"),
        ("ashare_limit_down_touch(field('open', table='StockDailyBar'), "
         "field('close', table='StockDailyBar'), field('high_limit', "
         "table='StockDailyBarAdj'), field('low_limit', table='StockDailyBarAdj'), "
         "field('volume', table='StockDailyBarAdj'))",
         "ashare_limit_down_touch(field('low', table='StockDailyBar'), "
         "field('low_limit', table='StockDailyBar'), 0.005)"),
        ("ashare_limit_failed(field('open', table='StockDailyBar'), "
         "field('close', table='StockDailyBar'), field('high_limit', "
         "table='StockDailyBar'), field('low_limit', table='StockDailyBarAdj'), "
         "field('volume', table='StockDailyBarAdj'))",
         "ashare_limit_failed(field('high', table='StockDailyBar'), "
         "field('close', table='StockDailyBar'), "
         "field('high_limit', table='StockDailyBar'), 0.005)"),
    ],
)
def test_exact_malformed_limit_templates_rebuild_formal_raw_calls(source, expected):
    formula, changes = migrate_formula(source)
    assert formula == expected
    assert changes and changes[0].startswith("SEMANTIC_REPAIR")
    assert migrate_formula(formula) == (formula, [])


def test_near_match_malformed_limit_templates_are_left_for_confirmation():
    sources = (
        "ashare_limit_up_touch(field('high', table='StockDailyBar'), "
        "field('close', table='StockDailyBar'), field('high_limit', "
        "table='StockDailyBarAdj'), field('low_limit', table='StockDailyBarAdj'))",
        "ashare_limit_down_streak(field('open', table='StockDailyBar'), "
        "field('close', table='StockDailyBar'), field('high_limit', "
        "table='StockDailyBarAdj'), field('low_limit', table='StockDailyBarAdj'), "
        "field('volume', table='StockDailyBarAdj'))",
    )
    for source in sources:
        assert migrate_formula(source) == (source, [])
