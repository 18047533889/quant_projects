import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r18_technical_recipes import (
    migrate_catalog_r18_technical_formula,
)


def _field(name: str, table: str = "StockDailyBarAdj") -> str:
    return f"field('{name}', table='{table}')"


def test_exact_trix_lowers_and_compiles():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = f"TRIX({_field('close')}, 15)"
    result = migrate_catalog_r18_technical_formula(source, enabled=True)

    assert result.changes == (
        "EXACT_SHAPE_RECIPE TRIX: preserved close/window=15; triple adjust=False "
        "span EMA, exact-zero prior mapped to null, then raw one-row percent change",
    )
    assert "TRIX(" not in result.formula
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(result.formula)
    ) is not None


def test_exact_adxr_lowers_and_compiles():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    source = (
        f"ADXR({_field('high')}, {_field('low')}, {_field('close')}, 14)"
    )
    result = migrate_catalog_r18_technical_formula(source, enabled=True)

    assert result.changes == (
        "EXACT_SHAPE_RECIPE ADXR: preserved high/low/close/window=14; active "
        "canonical ADX plus its exact per-instrument window lag",
    )
    assert "ADXR(" not in result.formula
    assert Analyzer().lower(
        DSLParser(surface="compat_research").parse(result.formula)
    ) is not None


def test_trix_expansion_run_many_matches_canonical_oracle():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators.technical.signal import TRIX
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dates = pd.date_range("2026-01-01", periods=48)
    close = pd.DataFrame(
        {
            "trend": 10.0 + np.arange(48, dtype=float) * 0.25,
            "flat": np.full(48, 5.0),
            "tiny": 1.0 + np.arange(48, dtype=float) * 5e-13,
            "leading_zero": np.r_[np.zeros(8), np.linspace(0.25, 10.0, 40)],
            "sign_cross": np.linspace(-8.0, 8.0, 48),
            "all_null": np.full(48, np.nan),
        },
        index=dates,
    )
    close.iloc[17, 0] = np.nan
    close.index.name = "timestamp"
    close.columns.name = "instrument"
    data = {"close": close.stack(future_stack=True).rename("close")}

    source = f"TRIX({_field('close')}, 15)"
    formula = migrate_catalog_r18_technical_formula(source, enabled=True).formula
    formula = formula.replace(_field("close"), "close")
    factor = Factor(
        name="trix",
        expr=DSLParser(surface="compat_research").parse(formula),
        source_expr=formula,
        surface="compat_research",
    )
    actual = FactorEngine(
        PandasBackend(), InMemorySeriesSource(data=data), run_mode="research"
    ).run_many([factor])["results"]["trix"]
    expected = TRIX().calculate(close, window=15).stack(future_stack=True)
    expected.name = actual.name

    pd.testing.assert_series_equal(actual, expected)


def test_existing_adxr_recipe_matches_active_adx_composition():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dates = pd.date_range("2026-02-01", periods=70)
    steps = np.arange(70, dtype=float)
    high = pd.DataFrame(
        {
            "trend": 12.0 + steps,
            "flat_zero": np.zeros(70),
            "nan_gap": 20.0 + steps * 0.1,
            "all_null": np.full(70, np.nan),
        },
        index=dates,
    )
    low = pd.DataFrame(
        {
            "trend": 8.0 + steps,
            "flat_zero": np.zeros(70),
            "nan_gap": 18.0 + steps * 0.1,
            "all_null": np.full(70, np.nan),
        },
        index=dates,
    )
    close = (high + low) * 0.5
    high.iloc[37, 2] = np.nan
    for frame in (high, low, close):
        frame.index.name = "timestamp"
        frame.columns.name = "instrument"
    data = {
        name: frame.stack(future_stack=True).rename(name)
        for name, frame in {"high": high, "low": low, "close": close}.items()
    }

    source = (
        f"ADXR({_field('high')}, {_field('low')}, {_field('close')}, 14)"
    )
    formula = migrate_catalog_r18_technical_formula(source, enabled=True).formula
    for name in ("high", "low", "close"):
        formula = formula.replace(_field(name), name)
    adx_formula = "ADX(high, low, close, 14)"
    factors = [
        Factor(
            name=name,
            expr=DSLParser(surface="compat_research").parse(source),
            source_expr=source,
            surface="compat_research",
        )
        for name, source in (("adx", adx_formula), ("adxr_recipe", formula))
    ]
    results = FactorEngine(
        PandasBackend(), InMemorySeriesSource(data=data), run_mode="research"
    ).run_many(factors)["results"]
    actual = results["adxr_recipe"]
    adx = results["adx"]
    delayed = (
        adx.unstack("instrument")
        .shift(14)
        .stack(future_stack=True)
        .reindex(adx.index)
    )
    expected = 0.5 * (adx + delayed)
    expected.name = actual.name
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "source",
    [
        "TRIX(close, 15)",
        f"TRIX({_field('close')}, 15.0)",
        f"TRIX({_field('close')}, window=15)",
        f"TRIX({_field('open')}, 15)",
        f"TRIX({_field('close', 'OtherTable')}, 15)",
        (
            f"TRIX({_field('open')}, {_field('high')}, {_field('low')}, "
            f"{_field('close')}, {_field('volume')})"
        ),
        "ADXR(high, low, close, 14)",
        f"ADXR({_field('high')}, {_field('low')}, {_field('close')}, 14.0)",
        f"ADXR({_field('high')}, {_field('low')}, {_field('close')}, window=14)",
        (
            f"ADXR({_field('open')}, {_field('high')}, {_field('low')}, "
            f"{_field('close')}, {_field('volume')})"
        ),
    ],
)
def test_non_exact_contracts_fail_closed(source):
    result = migrate_catalog_r18_technical_formula(source, enabled=True)
    assert result.formula == source
    assert not result.changes


def test_disabled_and_noop_text_are_preserved_byte_for_byte():
    exact = f"TRIX({_field('close')}, 15)"
    assert migrate_catalog_r18_technical_formula(exact).formula == exact

    source = "(\n  multiply(field('close', table='StockDailyBarAdj'), 1.0)  # keep\n)\n"
    result = migrate_catalog_r18_technical_formula(source, enabled=True)
    assert result.formula == source
    assert not result.changes


def test_utf8_byte_budget_is_enforced():
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_r18_technical_formula("测" * 21_846, enabled=True)
