import ast

import pytest

from factor_engine.tools.catalog_r19_operator_recipes import migrate_formula


def _f(name):
    return f"field('{name}', table='StockDailyBarAdj')"


def _ohlcv(name):
    return f"{name}({_f('open')}, {_f('high')}, {_f('low')}, {_f('close')}, {_f('volume')})"


@pytest.mark.parametrize(
    ("name", "required"),
    [
        ("RSI", "RSI_WILDER"),
        ("ATR", "ATR_WILDER"),
        ("MACD", "MACD_line"),
        ("MACD_line", "MACD_line"),
        ("MACD_signal", "MACD_signal"),
        ("MOM", "ts_delay"),
        ("ROC", "safe_div_null"),
        ("BollingerUpper", "ts_std"),
        ("BollingerLower", "ts_std"),
        ("AROON", "ts_argmax"),
        ("AROON_up", "ts_argmax"),
        ("AROON_down", "ts_argmin"),
        ("StochasticD", "ts_max"),
        ("TRIX", "ts_ema"),
        ("ADXR", "ADX"),
    ],
)
def test_reviewed_ohlcv_template_is_repaired_and_compiles(name, required):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    repaired, changes = migrate_formula(_ohlcv(name), "verified technical state")
    assert required in repaired
    assert changes and changes[0].startswith("R19_V1 SEMANTIC_REPAIR")
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(repaired)
    ) is not None


def test_sw_l1_subscript_becomes_registered_group_field_and_compiles():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = f"industry_neutralize({_f('ret')}, IndustryCode[sw_l1])"
    repaired, changes = migrate_formula(source)
    assert "field('industry_code', table='StockIndustry')" in repaired
    assert changes == [
        "EXACT_TAXONOMY_RECIPE IndustryCode[sw_l1]: use the registered "
        "StockIndustry industry_code field; runtime required-filter default is "
        "explicitly sw_l1"
    ]
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(repaired)
    ) is not None


def test_persistence_entropy_has_explicit_non_equivalent_complexity_replacement():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = "ts_persistence_entropy(ret, window=60, tau=1, dim=3)"
    repaired, changes = migrate_formula(source, "rolling complexity")
    assert repaired == (
        "ts_permutation_entropy(ret, window=60, order=3, delay=1, normalize=True)"
    )
    assert changes and "SEMANTIC_REDESIGN" in changes[0]
    assert "not topological persistence entropy" in changes[0]
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(repaired)
    ) is not None


def test_volume_clock_roughness_adds_real_price_dependency_and_compiles():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    repaired, changes = migrate_formula(
        "intraday_volume_clock_path_roughness(MinuteVolume)",
        "volume-clock path roughness",
    )
    assert "intraday_volume_clock_path_efficiency" in repaired
    assert "field('minute_close', table='StockMinuteBarAdj')" in repaired
    assert "field('minute_volume', table='StockMinuteBarAdj')" in repaired
    assert "original omitted price" in changes[0]
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(repaired)
    ) is not None


def test_other_roughness_shapes_fail_closed():
    source = "intraday_volume_clock_path_roughness(volume)"
    assert migrate_formula(source) == (source, [])


@pytest.mark.parametrize("taxonomy", ["sw_l2", "sw_l3", "jq_l1", "jq_l2", "zjw"])
def test_non_default_taxonomies_are_explicitly_downgraded(taxonomy):
    source = f"industry_neutralize(ret, IndustryCode[{taxonomy}])"
    repaired, changes = migrate_formula(source)
    assert "field('industry_code', table='StockIndustry')" in repaired
    assert taxonomy in changes[0]
    assert "NOT equivalent" in changes[0]


@pytest.mark.parametrize(
    "source",
    [
        "RSI(close)",
        "RSI(open, high, low, close, volume)",
        "RSI(field('close', table='StockDailyBarAdj'), 14)",
        "ATR(field('high', table='StockDailyBarAdj'), field('low', table='StockDailyBarAdj'), field('close', table='StockDailyBarAdj'), 14)",
    ],
)
def test_ambiguous_or_non_catalog_shapes_fail_closed(source):
    repaired, changes = migrate_formula(source)
    assert repaired == source
    assert changes == []


def test_nested_calls_are_repaired_without_losing_outer_expression():
    source = f"multiply({_ohlcv('AROON')}, TurnoverRatio)"
    repaired, changes = migrate_formula(source)
    assert repaired.startswith("multiply(") and "ts_argmax" in repaired
    assert len(changes) == 1
    ast.parse(repaired, mode="eval")


def test_type_and_byte_budget_guards():
    with pytest.raises(TypeError, match="formula"):
        migrate_formula(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="logic"):
        migrate_formula("ret", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input budget"):
        migrate_formula("测" * 21_846)


def test_invalid_expression_is_preserved_for_full_catalog_streaming():
    source = "not valid formula prose; keep it"
    assert migrate_formula(source) == (source, [])


def test_wilder_replacements_have_finite_flat_series_boundaries():
    import numpy as np
    import pandas as pd

    from factor_engine.cleaned_operators.technical.signal import ATRWilder, RSIWilder

    index = pd.date_range("2026-01-01", periods=20)
    close = pd.DataFrame({"A": np.full(20, 10.0)}, index=index)
    high = close + 1.0
    low = close - 1.0
    rsi = RSIWilder().calculate(close, window=14)
    atr = ATRWilder().calculate(high, low, close, window=14)
    assert np.isfinite(rsi.iloc[-1, 0])
    assert rsi.iloc[-1, 0] == pytest.approx(50.0)
    assert np.isfinite(atr.iloc[-1, 0])
    assert atr.iloc[-1, 0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "source",
    [
        "max_drawdown(ret)",
        "max_drawdown(field('ret', table='StockDailyBarAdj'))",
    ],
)
def test_max_drawdown_redefinition_uses_adjusted_close_and_compiles(source):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    repaired, changes = migrate_formula(source, "historical drawdown")
    assert repaired == (
        "ts_max_drawdown(field('close', table='StockDailyBarAdj'), 252)"
    )
    assert "fixed 252-session adjusted-price" in changes[0]
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(repaired)
    ) is not None


def test_max_drawdown_other_shapes_fail_closed():
    for source in (
        "max_drawdown(close)",
        "max_drawdown(field('ret', table='OtherTable'))",
        "max_drawdown(ret, 252)",
    ):
        assert migrate_formula(source) == (source, [])


def test_ts_max_drawdown_price_path_oracle():
    import polars as pl

    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    levels = pl.DataFrame({"asset": [100.0, 120.0, 90.0, 110.0]})
    operator = OperatorRegistry.get("ts_max_drawdown", backend="polars", mode="research")
    actual = operator.calculate(levels, window=4)
    assert actual["asset"][-1] == pytest.approx(-0.25)


def test_one_argument_volume_clock_efficiency_adds_real_price_and_compiles():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    repaired, changes = migrate_formula("intraday_volume_clock_path_efficiency(MinuteVolume)")
    assert "field('minute_close', table='StockMinuteBarAdj')" in repaired
    assert "buckets=16" in changes[0]
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(repaired)
    ) is not None


def test_adjusted_close_max_drawdown_redefinition_compiles():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = "max_drawdown(field('close', table='StockDailyBarAdj'))"
    repaired, changes = migrate_formula(source)
    assert repaired == "ts_max_drawdown(field('close', table='StockDailyBarAdj'), 252)"
    assert changes[0].startswith("R19_V2 SEMANTIC_REDESIGN")
    assert Analyzer().lower(DSLParser(surface="compat_research").parse(repaired)) is not None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("overnight_return", "overnight_return"),
        ("open_gap", "overnight_return"),
        ("close_gap", "open_close_return"),
        ("open_close_return", "open_close_return"),
        ("open_to_vwap_return", "open_to_vwap_return"),
    ],
)
def test_malformed_return_template_uses_registered_decomposition(name, expected):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = (
        f"{name}({_f('open')}, {_f('close')}, "
        f"{_f('pre_close')}, {_f('vwap')})"
    )
    repaired, changes = migrate_formula(source)
    assert repaired.startswith(expected + "(")
    assert changes[0].startswith("R19_V2 TEMPLATE_REPAIR")
    assert Analyzer().lower(DSLParser(surface="compat_research").parse(repaired)) is not None


def test_malformed_return_template_other_shape_fails_closed():
    source = "open_gap(open, close, pre_close, vwap)"
    assert migrate_formula(source) == (source, [])


def test_transition_count_continuous_return_becomes_boolean_state_and_compiles():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = "ts_transition_count(ret, 20, 'break')"
    repaired, changes = migrate_formula(source, "mean-reversion speed")
    assert repaired == "ts_transition_count(gt(ret, 0.0), 20, 'break')"
    assert changes and changes[0].startswith("R19_V3 SEMANTIC_REDESIGN")
    assert "requires a ConditionBool" in changes[0]
    assert "NOT equivalent" in changes[0]
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(repaired)
    ) is not None


def test_transition_count_rejects_continuous_return_during_planning():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer, TypedInputContractError

    with pytest.raises(TypedInputContractError, match="ts_transition_count.*condition"):
        Analyzer().lower(
            DSLParser(surface="compat_research").parse(
                "ts_transition_count(ret, 20, 'break')"
            )
        )


def test_transition_count_boolean_repair_kernel_accepts_output():
    import numpy as np
    import pandas as pd

    from factor_engine.cleaned_operators.state_event import TsTransitionCount

    index = pd.date_range("2026-01-01", periods=5)
    positive_state = pd.DataFrame(
        {"A": [1.0, 1.0, 0.0, 0.0, 1.0]}, index=index
    )
    actual = TsTransitionCount().calculate(
        positive_state, window=5, missing_policy="break"
    )
    assert np.isfinite(actual.iloc[-1, 0])
    assert actual.iloc[-1, 0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "source",
    [
        "ts_transition_count(ret, 60, 'break')",
        "ts_transition_count(ret, 20, 'carry')",
        "ts_transition_count(gt(ret, 0.0), 20, 'break')",
    ],
)
def test_transition_count_other_shapes_fail_closed(source):
    assert migrate_formula(source) == (source, [])


def test_quantile_regression_beta_uses_registered_explicit_median_slope():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = "ts_quantile_regression_beta(ret, TurnoverRatio)"
    repaired, changes = migrate_formula(source)
    assert repaired == (
        "ts_quantile_regression_slope(ret, TurnoverRatio, 120, 0.5, 5)"
    )
    assert "implementation differs" in changes[0]
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(repaired)
    ) is not None


def test_pandas_candlestick_pattern_hammer_is_supported_and_bounded():
    import numpy as np
    import pandas as pd

    from factor_engine.cleaned_operators.price_volume.candle_pattern_engine_v2 import (
        CandlestickPatternEngine,
        candlestick_active_params,
    )

    index = pd.date_range("2026-01-01", periods=3)
    open_ = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=index)
    high = pd.DataFrame({"A": [10.22, 10.8, 10.0]}, index=index)
    low = pd.DataFrame({"A": [9.0, 9.9, 10.0]}, index=index)
    close = pd.DataFrame({"A": [10.2, 10.7, 10.0]}, index=index)

    actual = CandlestickPatternEngine().calculate(
        open_, high, low, close, pattern="hammer"
    )
    assert actual.iloc[0, 0] == 1.0
    assert actual.iloc[1, 0] == 0.0
    assert np.isnan(actual.iloc[2, 0])
    assert candlestick_active_params("hammer") == frozenset({"pattern"})
