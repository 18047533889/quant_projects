import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r17_technical_recipes import migrate_catalog_r17_technical_formula


def _field(name):
    return f"field('{name}', table='StockDailyBarAdj')"


@pytest.mark.parametrize("name,args", [
    ("StochasticD", f"{_field('high')}, {_field('low')}, {_field('close')}, 14"),
    ("AROON", f"{_field('close')}, 25"),
])
def test_exact_outer_calls_lower_and_compile(name, args):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer
    result = migrate_catalog_r17_technical_formula(f"{name}({args})", enabled=True)
    assert result.changes and f"{name}(" not in result.formula
    assert Analyzer().lower(DSLParser(surface="compat_research").parse(result.formula)) is not None


def test_formal_recipe_contracts_match_canonical_oracles():
    from factor_engine.cleaned_operators.technical.signal import Aroon, StochasticD
    dates = pd.date_range("2026-01-01", periods=35)
    close = pd.DataFrame({"x": np.r_[np.arange(20.0), np.repeat(19.0, 15)]}, index=dates)
    high, low = close + 2, close - 2
    high.iloc[7, 0] = np.nan
    lowest = low.rolling(14, min_periods=1).min()
    highest = high.rolling(14, min_periods=1).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    pd.testing.assert_frame_equal(k.rolling(3, min_periods=1).mean(), StochasticD().calculate(high, low, close, window=14))
    age_min = close.rolling(26, min_periods=26).apply(lambda x: np.argmin(x[::-1]), raw=True)
    age_max = close.rolling(26, min_periods=26).apply(lambda x: np.argmax(x[::-1]), raw=True)
    pd.testing.assert_frame_equal(100 * (age_min - age_max) / 25, Aroon().calculate(close, window=25))


def test_lowered_outer_calls_run_many_match_canonical_oracles():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators.technical.signal import Aroon, StochasticD
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource
    dates = pd.date_range("2026-02-01", periods=35)
    cols = ["trend", "ties"]
    close = pd.DataFrame({"trend": np.arange(35.0), "ties": np.r_[np.arange(20.0), np.repeat(19.0, 15)]}, index=dates)
    high, low = close + 2.0, close - 2.0
    high.iloc[7, 0] = np.nan
    for frame in (high, low, close):
        frame.index.name = "timestamp"; frame.columns.name = "instrument"
    data = {name: frame.stack(future_stack=True).rename(name) for name, frame in {"high": high, "low": low, "close": close}.items()}
    sources = {
        "d": f"StochasticD({_field('high')}, {_field('low')}, {_field('close')}, 14)",
        "aroon": f"AROON({_field('close')}, 25)",
    }
    factors = []
    for name, source in sources.items():
        formula = migrate_catalog_r17_technical_formula(source, enabled=True).formula
        for field in ("high", "low", "close"):
            formula = formula.replace(_field(field), field)
        factors.append(Factor(name=name, expr=DSLParser(surface="compat_research").parse(formula), source_expr=formula, surface="compat_research"))
    actual = FactorEngine(PandasBackend(), InMemorySeriesSource(data=data), run_mode="research").run_many(factors)["results"]
    for name, panel in {
        "d": StochasticD().calculate(high, low, close, window=14),
        "aroon": Aroon().calculate(close, window=25),
    }.items():
        expected = panel.stack(future_stack=True); expected.name = actual[name].name
        pd.testing.assert_series_equal(actual[name], expected)


def test_uncertain_shapes_fail_closed():
    for source in ("StochasticD(high, low, close, 14)", f"AROON({_field('high')}, 25)", f"AROON({_field('close')}, 25.0)"):
        result = migrate_catalog_r17_technical_formula(source, enabled=True)
        assert result.formula == source and not result.changes


def test_noop_preserves_formatting_whitespace_and_comments_byte_for_byte():
    source = (
        "(\n"
        "    multiply(\n"
        "        field('close', table='StockDailyBarAdj'),  # keep this comment\n"
        "        1.0,\n"
        "    )\n"
        ")  # and this trailing comment\n"
    )
    result = migrate_catalog_r17_technical_formula(source, enabled=True)
    assert result.formula == source
    assert result.changes == ()


def test_input_type_syntax_and_byte_budget_guards():
    with pytest.raises(TypeError):
        migrate_catalog_r17_technical_formula(1, enabled=True)
    with pytest.raises(SyntaxError):
        migrate_catalog_r17_technical_formula("AROON(", enabled=True)
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_r17_technical_formula("测" * 21_846, enabled=True)
