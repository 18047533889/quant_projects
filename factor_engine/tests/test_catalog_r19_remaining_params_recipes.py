import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r19_remaining_params_recipes import migrate_formula


@pytest.mark.parametrize(("source", "expected"), [
    (
        "ts_extreme_cluster_ratio(ret, 60, 2.0, 0.0, 5, 5)",
        "ts_extreme_cluster_ratio(ret, 60, 2.0, 0.9, 'absolute', 5)",
    ),
    (
        "ts_first_passage_bias(ret)",
        "ts_first_passage_bias(ret, ts_std(ret, 20))",
    ),
    (
        "ts_first_passage_bias(ret, ts_std(ret,20), window=120, min_periods=3)",
        "ts_first_passage_bias(ret, ts_std(ret, 20), window=120, min_anchors=3)",
    ),
    (
        "ts_crossing_speed(ret)",
        "ts_crossing_speed(ret, ts_mean(ret, 20))",
    ),
    (
        "ts_crossing_acceleration(TurnoverRatio)",
        "ts_crossing_acceleration(TurnoverRatio, ts_mean(TurnoverRatio, 20))",
    ),
    (
        "ts_current_drawdown_area(field('ret', table='StockDailyBarAdj'))",
        "ts_current_drawdown_area(field('close', table='StockDailyBarAdj'), 60)",
    ),
])
def test_reviewed_remaining_parameter_repairs(source, expected):
    got, audit = migrate_formula(source, "reviewed")
    assert got == expected
    assert audit
    assert migrate_formula(got, "reviewed") == (got, [])


def test_ambiguous_or_already_complete_calls_fail_closed():
    samples = [
        "ts_crossing_speed(x, benchmark)",
        "ts_first_passage_bias(x, scale)",
        "ts_current_drawdown_area(price)",
        "ts_current_drawdown_area(ret)",
        "ts_current_drawdown_area(field(" + chr(39) + "ret" + chr(39) + ", table=" + chr(39) + "StockDailyBar" + chr(39) + "))",
        "ts_current_drawdown_area(field(" + chr(39) + "ret" + chr(39) + ", table=" + chr(39) + "StockDailyBarAdj" + chr(39) + ", adjustment=" + chr(39) + "x" + chr(39) + "))",
        "ts_extreme_cluster_ratio(x, 20, 'quantile', 0.95, 'upper', 10)",
    ]
    for source in samples:
        assert migrate_formula(source) == (source, [])
    assert migrate_formula("ts_crossing_speed(") == ("ts_crossing_speed(", [])
    with pytest.raises(ValueError):
        migrate_formula("x" * 65_537)


def test_repaired_formulas_really_parse_and_compile():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.cleaned_operators import load_all
    from factor_engine.ir.analyzer import Analyzer

    load_all()
    parser = DSLParser(surface="compat_research")
    for source in (
        "ts_extreme_cluster_ratio(ret, 60, 2.0, 0.0, 5, 5)",
        "ts_first_passage_bias(ret)",
        "ts_crossing_speed(ret)",
        "ts_crossing_acceleration(ret)",
    ):
        repaired, _ = migrate_formula(source)
        Analyzer().lower(parser.parse(repaired))


def test_small_numeric_semantics_use_real_scale_baseline_and_wealth_level():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    idx = pd.date_range("2026-01-01", periods=80)
    x = pd.DataFrame({"A": np.sin(np.arange(80) / 5.0)}, index=idx)
    scale = x.rolling(20, min_periods=1).std(ddof=1)
    fp = OperatorRegistry.get("ts_first_passage_bias", "pandas_numpy").calculate(
        x, scale, window=40, horizon=5, min_anchors=3
    )
    assert fp.shape == x.shape and np.isfinite(fp.iloc[-1, 0])

    price = pd.DataFrame({"A": np.cumprod(1.0 + np.r_[np.zeros(20), -0.01 * np.ones(10), 0.02 * np.ones(50)])}, index=idx)
    area = OperatorRegistry.get("ts_current_drawdown_area", "pandas_numpy").calculate(price, window=60)
    assert area.shape == price.shape

@pytest.mark.parametrize(("source", "expected"), [
    ("ts_support_level(low, 3, 3, 3, 3)", "ts_support_level(low, 3, 3, 60, 3)"),
    ("ts_resistance_slope(high, 3, 3, 3, 3)", "ts_resistance_slope(high, 3, 3, 60, 3)"),
    ("ts_distance_to_support(close, low, 3, 3, 3, 3)", "ts_distance_to_support(close, low, 3, 3, 60, 3)"),
    ("ts_resistance_break(close, high, 3, 3, 3, 3)", "ts_resistance_break(close, high, 3, 3, 60, 3)"),
])
def test_impossible_three_row_pivot_history_is_expanded(source, expected):
    got, audit = migrate_formula(source)
    assert got == expected
    assert "cannot contain three confirmed pivots" in audit[0]
    assert migrate_formula(got) == (got, [])


def test_repaired_support_level_compiles_and_is_finite_on_oscillating_lows():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.ir.analyzer import Analyzer

    load_all()
    repaired, _ = migrate_formula("ts_support_level(low, 3, 3, 3, 3)")
    Analyzer().lower(DSLParser(surface="compat_research").parse(repaired))
    values = 10.0 + np.sin(np.arange(160) * np.pi / 4.0)
    low = pd.DataFrame({"A": values})
    result = OperatorRegistry.get("ts_support_level", "pandas_numpy").calculate(low, 3, 3, 60, 3)
    assert np.isfinite(result.to_numpy()).any()
