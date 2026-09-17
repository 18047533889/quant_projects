import numpy as np
import pandas as pd
import polars as pl
import pytest


def test_strict_mean_abs_deviation_backend_parity_and_poisoning():
    from factor_engine.cleaned_operators.common.statistics import StrictMeanAbsoluteDeviation
    from factor_engine.cleaned_operators.common.polars_statistics import StrictMeanAbsDeviationPolars
    values = [1.0, 2.0, np.nan, 4.0, 4.0, np.inf, 5.0, 5.0]
    pdf = pd.DataFrame({"x": values})
    expected = StrictMeanAbsoluteDeviation().calculate(pdf, window=3)["x"]
    actual = StrictMeanAbsDeviationPolars().calculate(pl.DataFrame({"x": values}), window=3)["x"].to_pandas()
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    assert expected.iloc[2:5].isna().all()
    assert expected.iloc[5:8].isna().all()


def test_strict_mean_abs_deviation_is_registered_on_both_research_backends():
    from factor_engine.cleaned_operators import OperatorRegistry, load_all
    load_all()
    pandas_op = OperatorRegistry.get(
        "ts_mean_abs_deviation_strict", backend="pandas_numpy", mode="research"
    )
    polars_op = OperatorRegistry.get(
        "ts_mean_abs_deviation_strict", backend="polars", mode="research"
    )
    assert pandas_op is not None
    assert polars_op is not None
    expected = pandas_op.calculate(pd.DataFrame({"x": [1.0, 2.0, 4.0]}), window=2)["x"]
    actual = polars_op.calculate(pl.DataFrame({"x": [1.0, 2.0, 4.0]}), window=2)["x"].to_pandas()
    pd.testing.assert_series_equal(actual, expected, check_names=False)


@pytest.mark.parametrize("window", [True, False, 3.5, 0, -1, None])
def test_strict_mean_abs_deviation_public_binding_rejects_invalid_windows(window):
    from factor_engine.cleaned_operators.common.statistics import StrictMeanAbsoluteDeviation
    from factor_engine.cleaned_operators.common.polars_statistics import StrictMeanAbsDeviationPolars
    pdf = pd.DataFrame({"x": [1.0, 2.0]})
    pldf = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises((TypeError, ValueError)):
        StrictMeanAbsoluteDeviation().calculate(pdf, window=window)
    with pytest.raises((TypeError, ValueError)):
        StrictMeanAbsDeviationPolars().calculate(pldf, window=window)


@pytest.mark.parametrize("window", [True, False, 3.0, 3.5, 0, -1, "3", None])
def test_strict_mean_abs_deviation_raw_kernels_reject_non_strict_windows(window):
    from factor_engine.cleaned_operators.common.statistics import StrictMeanAbsoluteDeviation
    from factor_engine.cleaned_operators.common.polars_statistics import StrictMeanAbsDeviationPolars
    pdf = pd.DataFrame({"x": [1.0, 2.0]})
    pldf = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises((TypeError, ValueError)):
        StrictMeanAbsoluteDeviation()._calculate_series(pdf, window=window)
    with pytest.raises((TypeError, ValueError)):
        StrictMeanAbsDeviationPolars()._calculate_series(pldf, window=window)


@pytest.mark.parametrize("window", [3.0, "3"])
def test_strict_mean_abs_deviation_public_binding_normalizes_exact_integers(window):
    from factor_engine.cleaned_operators.common.statistics import StrictMeanAbsoluteDeviation
    from factor_engine.cleaned_operators.common.polars_statistics import StrictMeanAbsDeviationPolars
    pdf = pd.DataFrame({"x": [1.0, 2.0, 4.0]})
    expected = StrictMeanAbsoluteDeviation().calculate(pdf, window=3)
    pd.testing.assert_frame_equal(
        StrictMeanAbsoluteDeviation().calculate(pdf, window=window), expected
    )
    actual = StrictMeanAbsDeviationPolars().calculate(
        pl.DataFrame({"x": [1.0, 2.0, 4.0]}), window=window
    )
    baseline = StrictMeanAbsDeviationPolars().calculate(
        pl.DataFrame({"x": [1.0, 2.0, 4.0]}), window=3
    )
    assert actual.equals(baseline)


def test_strict_mean_abs_deviation_declares_exact_history():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.runtime.execution_contract import _declared_history_extension
    load_all()
    assert _declared_history_extension(
        "ts_mean_abs_deviation_strict", {"window": 20}
    ) == 19


def test_cci_recipe_matches_canonical_pandas_oracle():
    from factor_engine.cleaned_operators.technical.signal import CCI
    from factor_engine.cleaned_operators.common.statistics import StrictMeanAbsoluteDeviation
    dates = pd.date_range("2026-01-01", periods=9)
    high = pd.DataFrame({"x": [3,4,5,np.nan,7,8,9,10,11], "flat": 5.0}, index=dates)
    low = high - 2
    close = high - 1
    tp = (high + low + close) / 3.0
    mad = StrictMeanAbsoluteDeviation().calculate(tp, window=3)
    expanded = (tp - tp.rolling(3, min_periods=1).mean()) / (0.015 * mad.replace(0, np.nan))
    expected = CCI().calculate(high, low, close, window=3)
    pd.testing.assert_frame_equal(expanded, expected)


def test_catalog_exact_and_repaired_calls_compile():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.tools.catalog_r16_technical_recipes import migrate_catalog_r16_technical_formula
    f = lambda n: f"field('{n}', table='StockDailyBarAdj')"
    exact = f"CCI({f('high')}, {f('low')}, {f('close')}, 3)"
    repaired = f"CCI({f('open')}, {f('high')}, {f('low')}, {f('close')}, {f('volume')})"
    for source, logic in ((exact, "generic oscillator"), (repaired, "CCI oscillator")):
        result = migrate_catalog_r16_technical_formula(source, logic=logic, enabled=True)
        assert result.changes
        assert "CCI(" not in result.formula
        assert Analyzer().lower(DSLParser(surface="compat_research").parse(result.formula)) is not None


def test_expanded_cci_runs_many_against_canonical_boundary_oracle():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators.technical.signal import CCI
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.tools.catalog_r16_technical_recipes import migrate_catalog_r16_technical_formula
    from tests.helpers import InMemorySeriesSource

    dates = pd.date_range("2026-02-01", periods=9)
    columns = ["normal", "nan", "posinf", "neginf", "tiny", "flat"]
    base = np.arange(9.0)
    high = pd.DataFrame(index=dates, columns=columns, dtype=float)
    low = high.copy()
    close = high.copy()
    for i in range(9):
        high.iloc[i] = [12+i, 12+i, 12+i, 12+i, 1.0+(i+1)*5e-13, 5.0]
        low.iloc[i] = [8+i, 8+i, 8+i, 8+i, 1.0, 5.0]
        close.iloc[i] = [9+i, 9+i, 9+i, 9+i, 1.0+(i+1)*2e-13, 5.0]
    close.iloc[2, close.columns.get_loc("nan")] = np.nan
    high.iloc[2, high.columns.get_loc("posinf")] = np.inf
    low.iloc[2, low.columns.get_loc("neginf")] = -np.inf
    panels = {"high": high, "low": low, "close": close}
    for frame in panels.values():
        frame.index.name = "timestamp"
        frame.columns.name = "instrument"
    data = {name: frame.stack(future_stack=True).rename(name) for name, frame in panels.items()}
    field = lambda n: f"field('{n}', table='StockDailyBarAdj')"
    source_formula = f"CCI({field('high')}, {field('low')}, {field('close')}, 3)"
    formula = migrate_catalog_r16_technical_formula(
        source_formula, logic="generic oscillator", enabled=True
    ).formula
    for name in panels:
        formula = formula.replace(field(name), name)
    factor = Factor(
        name="cci", expr=DSLParser(surface="compat_research").parse(formula),
        source_expr=formula, surface="compat_research",
    )
    actual = FactorEngine(
        PandasBackend(), InMemorySeriesSource(data=data), run_mode="research"
    ).run_many([factor])["results"]["cci"]
    expected = CCI().calculate(high, low, close, window=3).stack(future_stack=True)
    expected.name = actual.name
    pd.testing.assert_series_equal(actual, expected)
    assert actual.xs("flat", level="instrument").isna().all()
    assert actual.xs("tiny", level="instrument").notna().iloc[1:].all()
    for name in ("nan", "posinf", "neginf"):
        series = actual.xs(name, level="instrument")
        assert series.iloc[2:5].isna().all()
        assert np.isfinite(series.iloc[5])


def test_generic_ohlcv_is_fail_closed():
    from factor_engine.tools.catalog_r16_technical_recipes import migrate_catalog_r16_technical_formula
    f = lambda n: f"field('{n}', table='StockDailyBarAdj')"
    source = f"CCI({f('open')}, {f('high')}, {f('low')}, {f('close')}, {f('volume')})"
    result = migrate_catalog_r16_technical_formula(source, logic="generic oscillator", enabled=True)
    assert result.formula == source and not result.changes
