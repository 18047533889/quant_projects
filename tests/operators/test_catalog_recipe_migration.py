from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_recipe_migration import (
    migrate_average_volume_calls,
    migrate_catalog_recipe_formula,
)


def test_rewrites_only_bare_call_target_and_preserves_source_exactly() -> None:
    source = "add( average_volume(volume, window=20),\n average_volume (other, 5) )"
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == (
        "add( ts_average_volume(volume, window=20),\n"
        " ts_average_volume (other, 5) )"
    )
    assert result.changes == (
        "average_volume -> ts_average_volume",
        "average_volume -> ts_average_volume",
    )


@pytest.mark.parametrize(
    "source",
    [
        "average_volume",
        "field('average_volume')",
        "eq(label, 'average_volume(volume, 20)')",
        "LQTP.average_volume(volume, 20)",
        "namespace.average_volume(volume, 20)",
    ],
)
def test_does_not_rewrite_strings_fields_bare_names_or_attributes(source: str) -> None:
    result = migrate_average_volume_calls(source)
    assert result.formula == source
    assert result.changes == ()


def test_five_ohlcv_arguments_are_retained_instead_of_guessed_or_dropped() -> None:
    source = "average_volume(Open, High, Low, Close, Volume)"
    result = migrate_average_volume_calls(source)
    assert result.formula == "ts_average_volume(Open, High, Low, Close, Volume)"
    old_call = ast.parse(source, mode="eval").body
    new_call = ast.parse(result.formula, mode="eval").body
    assert isinstance(old_call, ast.Call) and isinstance(new_call, ast.Call)
    assert [ast.dump(arg) for arg in new_call.args] == [
        ast.dump(arg) for arg in old_call.args
    ]
    assert len(new_call.args) == 5

    from factor_engine.backend.operator_errors import OperatorParameterError
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    operator = OperatorRegistry.get("ts_average_volume", backend="pandas_numpy")
    assert operator is not None
    panel = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(OperatorParameterError, match="received 5 positional"):
        operator.calculate(panel, panel, panel, panel, panel)


def test_invalid_or_oversized_input_fails_closed() -> None:
    with pytest.raises(SyntaxError):
        migrate_average_volume_calls("average_volume(volume, 20")
    with pytest.raises(TypeError, match="string"):
        migrate_average_volume_calls(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="budget"):
        migrate_average_volume_calls("x" * 65_537)


def test_real_dsl_and_numeric_two_parameter_volume20_contract() -> None:
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.price_volume.liquidity_v2 import (
        average_volume as implementation_helper,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.ir.analyzer import Analyzer

    migrated = migrate_average_volume_calls("average_volume(volume, 20)")
    assert migrated.formula == "ts_average_volume(volume, 20)"
    factor = parse_factor(migrated.formula, surface="compat_research")
    lowered = Analyzer().lower(factor.expr).ir
    assert lowered.op == "ts_average_volume"

    load_all()
    operator = OperatorRegistry.get("ts_average_volume", backend="pandas_numpy")
    assert operator is not None
    values = np.arange(1.0, 26.0)
    volume = pd.DataFrame(
        {"A": values},
        index=pd.date_range("2024-01-02", periods=len(values), freq="B"),
    )
    actual = operator.calculate(volume, 20)
    helper = implementation_helper(volume, 20)
    expected = np.full(len(values), np.nan)
    for end in range(20, len(values) + 1):
        expected[end - 1] = sum(values[end - 20 : end]) / 20.0

    np.testing.assert_allclose(actual["A"].to_numpy(), expected, equal_nan=True)
    pd.testing.assert_frame_equal(actual, helper)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ROC(close)", "multiply(ts_pct(close, 10), 100.0)"),
        ("ROC(price=close, window=5)", "multiply(ts_pct(close, 5), 100.0)"),
        ("MOM(close, 7)", "ts_delta(close, 7)"),
        ("RSI(x=close)", "rsi_sma(close=close, window=14)"),
        ("ATR(high, low, close)", "atr_sma(high, low, close, window=14)"),
        (
            "BollingerUpper(x=close, std_dev=2.5)",
            "add(ts_mean(close, 20), multiply(2.5, ts_std(close, 20)))",
        ),
        ("BollingerLower(close, 10)", "subtract(ts_mean(close, 10), multiply(2, ts_std(close, 10)))"),
        ("BollingerBands(close)", "ts_mean(close, 20)"),
    ],
)
def test_migrates_only_unambiguous_legacy_signatures(source: str, expected: str) -> None:
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == expected
    assert len(result.changes) == 1


@pytest.mark.parametrize(
    "source",
    [
        "ROC(Open, High, Low, Close, Volume)",
        "RSI(Open, High, Low, Close, Volume)",
        "ATR(Open, High, Low, Close, Volume)",
        "BollingerUpper(Open, High, Low, Close, Volume)",
        "ROC(close, price=other)",
        "MOM(*args)",
        "ATR(high, low, close, **kwargs)",
        "RSI(close, smoothing='wilder')",
        "namespace.ROC(close, 10)",
        "ADXR(high, low, close, 14)",
    ],
)
def test_preserves_malformed_ambiguous_and_attribute_recipe_calls(source: str) -> None:
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == source
    assert result.changes == ()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "rank(ROC(close, 10))",
            "rank(multiply(ts_pct(close, 10), 100.0))",
        ),
        (
            "ROC(add(close, open), 10)",
            "multiply(ts_pct(add(close, open), 10), 100.0)",
        ),
        (
            "ROC(MOM(close, 5), 10)",
            "ROC(ts_delta(close, 5), 10)",
        ),
    ],
)
def test_nested_migration_changes_only_non_overlapping_call_spans(
    source: str, expected: str
) -> None:
    assert migrate_catalog_recipe_formula(source).formula == expected


def test_reviewed_recipe_expressions_match_legacy_numeric_implementations() -> None:
    from factor_engine.cleaned_operators.technical.signal import (
        ATR,
        MOM,
        ROC,
        RSI,
        BollingerLower,
        BollingerUpper,
    )

    index = pd.date_range("2024-01-02", periods=30, freq="B")
    close = pd.DataFrame({"A": [10 + i + (i % 4) for i in range(30)]}, index=index)
    high = close + pd.DataFrame({"A": np.linspace(0.5, 1.5, 30)}, index=index)
    low = close - pd.DataFrame({"A": np.linspace(0.4, 1.2, 30)}, index=index)

    lag = close.shift(5)
    pd.testing.assert_frame_equal(ROC().calculate(close, 5), (close / lag - 1) * 100)
    pd.testing.assert_frame_equal(MOM().calculate(close, 5), close - lag)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(6, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(6, min_periods=1).mean()
    expected_rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    expected_rsi = expected_rsi.mask((loss == 0) & (gain > 0), 100.0)
    expected_rsi = expected_rsi.mask((gain == 0) & (loss > 0), 0.0)
    expected_rsi = expected_rsi.mask((gain == 0) & (loss == 0), 50.0)
    pd.testing.assert_frame_equal(RSI().calculate(close, 6), expected_rsi)

    true_range = np.maximum(
        np.maximum(high - low, (high - close.shift(1)).abs()),
        (low - close.shift(1)).abs(),
    )
    pd.testing.assert_frame_equal(
        ATR().calculate(high, low, close, 6), true_range.rolling(6, min_periods=1).mean()
    )
    mean = close.rolling(6, min_periods=1).mean()
    std = close.rolling(6, min_periods=1).std()
    pd.testing.assert_frame_equal(BollingerUpper().calculate(close, 6, 2.5), mean + 2.5 * std)
    pd.testing.assert_frame_equal(BollingerLower().calculate(close, 6, 2.5), mean - 2.5 * std)


@pytest.mark.parametrize(
    "source",
    [
        "MOM(close, 5)",
        "ROC(close, 5)",
        "BollingerUpper(close, 6, 2.5)",
        "BollingerLower(close, 6, 2.5)",
        "BollingerBands(close, 6, 2.5)",
    ],
)
def test_primitive_recipe_migrations_are_accepted_by_real_dsl(source: str) -> None:
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.cleaned_operators import load_all

    load_all()
    migrated = migrate_catalog_recipe_formula(source)
    DSLParser(surface="compat_research", dialect="native").parse(migrated.formula)


@pytest.mark.parametrize(
    ("source", "legacy_name", "legacy_args"),
    [
        ("MOM(close, 5)", "MOM", (5,)),
        ("ROC(close, 5)", "ROC", (5,)),
        ("BollingerUpper(close, 6, 2.5)", "BollingerUpper", (6, 2.5)),
        ("BollingerLower(close, 6, 2.5)", "BollingerLower", (6, 2.5)),
        ("BollingerBands(close, 6, 2.5)", "BollingerBands", (6, 2.5)),
    ],
)
def test_migrated_primitive_recipe_matches_legacy_numeric_output(
    source: str, legacy_name: str, legacy_args: tuple[object, ...]
) -> None:
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.technical import signal
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    load_all()
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    panel = pd.DataFrame(
        {"A": [10.0 + i + (i % 4) for i in range(30)]}, index=dates
    )
    series = panel.stack()
    series.index.names = ["timestamp", "instrument"]
    migrated = migrate_catalog_recipe_formula(source)
    expr = DSLParser(surface="compat_research", dialect="native").parse(
        migrated.formula
    )
    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": series}),
        run_mode="research",
    )
    actual = engine.run(Factor(name="migrated", expr=expr))["result"].sort_index()
    legacy = getattr(signal, legacy_name)().calculate(panel, *legacy_args)
    expected = legacy.stack(dropna=False)
    expected.index.names = ["timestamp", "instrument"]
    pd.testing.assert_series_equal(
        actual, expected.sort_index(), check_names=False, rtol=1e-12, atol=1e-12
    )


def test_markov_min_periods_maps_to_documented_transition_count_floor() -> None:
    source = "ts_markov_persistence(ret, window=60, bins=3, lag=1, min_periods=3)"
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == (
        "ts_markov_persistence(ret, window=60, bins=3, lag=1, min_count=3)"
    )
    assert result.changes == (
        "ts_markov_persistence.min_periods -> min_count",
    )


@pytest.mark.parametrize(
    "source",
    [
        "ts_markov_persistence(ret, 60, 3, 1, 3, min_periods=3)",
        "ts_markov_persistence(ret, min_count=3, min_periods=3)",
        "ts_markov_persistence(ret, min_periods=3, **settings)",
        "namespace.ts_markov_persistence(ret, min_periods=3)",
    ],
)
def test_markov_min_periods_collision_or_ambiguity_is_preserved(source: str) -> None:
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == source
    assert result.changes == ()
