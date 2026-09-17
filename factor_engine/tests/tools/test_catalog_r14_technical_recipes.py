import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r14_technical_recipes import (
    migrate_catalog_r14_technical_formula,
)


def _field(name, table="StockDailyBarAdj"):
    return f"field('{name}', table='{table}')"


OHLCV = ", ".join(_field(name) for name in ("open", "high", "low", "close", "volume"))


def _expected(window=14):
    high, low, close = _field("high"), _field("low"), _field("close")
    denominator = f"subtract(ts_max({high}, {window}), ts_min({low}, {window}))"
    numerator = f"subtract(ts_max({high}, {window}), {close})"
    return (
        f"where(eq({denominator}, 0.0), safe_div_null(0.0, 0.0), "
        f"multiply(-100.0, divide({numerator}, {denominator})))"
    )


@pytest.mark.parametrize(
    ("source", "logic", "expected", "decision_kind"),
    [
        (f"WilliamsR({_field('high')}, {_field('low')}, {_field('close')}, 14)", "williams_centered", _expected(), "EXACT_SHAPE_RECIPE"),
        (f"WilliamsR({_field('high')}, {_field('low')}, {_field('close')}, 9)", "Williams range", _expected(9), "EXACT_SHAPE_RECIPE"),
        (f"WilliamsR({OHLCV})", "verified technical state WilliamsR", _expected(), "EXPLICIT_CONTRACT_REPAIR"),
    ],
)
def test_exact_shapes_expand_and_are_idempotent(source, logic, expected, decision_kind):
    result = migrate_catalog_r14_technical_formula(source, logic=logic, enabled=True)
    assert result.formula == expected
    assert result.changes[0].startswith(decision_kind)
    assert "min_periods=1" in result.changes[0]
    assert migrate_catalog_r14_technical_formula(result.formula, logic=logic, enabled=True).changes == ()


def test_default_is_disabled():
    source = f"WilliamsR({_field('high')}, {_field('low')}, {_field('close')}, 14)"
    assert migrate_catalog_r14_technical_formula(source, logic="WilliamsR").formula == source


@pytest.mark.parametrize(
    "formula,logic",
    [
        (f"WilliamsR({_field('high')}, {_field('low')}, {_field('close')}, 14)", "generic oscillator"),
        ("WilliamsR(high, low, close, 14)", "WilliamsR"),
        (f"WilliamsR({_field('high')}, {_field('low')}, {_field('close')}, window=14)", "WilliamsR"),
        (f"WilliamsR({_field('high')}, {_field('low')}, {_field('close')}, 14.0)", "WilliamsR"),
        (f"WilliamsR({_field('high', 'Other')}, {_field('low')}, {_field('close')}, 14)", "WilliamsR"),
        (f"WilliamsR({OHLCV}, 14)", "WilliamsR"),
        (f"other.WilliamsR({OHLCV})", "WilliamsR"),
    ],
)
def test_uncertain_shapes_fail_closed(formula, logic):
    result = migrate_catalog_r14_technical_formula(formula, logic=logic, enabled=True)
    assert result.formula == formula
    assert result.changes == ()


def test_pointwise_oracle_matches_legacy_for_nan_warmup_and_constant_ranges():
    from factor_engine.cleaned_operators import OperatorRegistry, load_all
    from factor_engine.cleaned_operators.technical.signal import WilliamsR

    load_all()
    index = pd.date_range("2026-01-01", periods=12)
    high = pd.DataFrame({"trend": np.arange(12.0) + 11, "flat": 5.0}, index=index)
    low = pd.DataFrame({"trend": np.arange(12.0) + 8, "flat": 5.0}, index=index)
    close = pd.DataFrame({"trend": np.arange(12.0) + 9, "flat": 5.0}, index=index)
    high.iloc[4, 0] = np.nan
    low.iloc[7, 0] = np.nan
    close.iloc[9, 0] = np.nan
    legacy = WilliamsR().calculate(high, low, close, window=5)
    rolling_high = OperatorRegistry.get("ts_max").calculate(high, window=5)
    rolling_low = OperatorRegistry.get("ts_min").calculate(low, window=5)
    denominator = rolling_high - rolling_low
    expanded = -100.0 * (rolling_high - close) / denominator.replace(0, np.nan)
    pd.testing.assert_frame_equal(expanded, legacy)
    assert np.isfinite(expanded["trend"].iloc[0])
    assert expanded["flat"].isna().all()
    assert np.isnan(expanded["trend"].iloc[9])


def test_real_dsl_run_many_matches_legacy_on_boundary_panel():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators.technical.signal import WilliamsR
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dates = pd.date_range("2026-02-01", periods=8)
    instruments = ["flat", "infinite", "missing", "tiny", "trend"]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["timestamp", "instrument"]
    )
    panels = {
        "high": pd.DataFrame(index=dates, columns=instruments, dtype=float),
        "low": pd.DataFrame(index=dates, columns=instruments, dtype=float),
        "close": pd.DataFrame(index=dates, columns=instruments, dtype=float),
    }
    for panel in panels.values():
        panel.index.name = "timestamp"
        panel.columns.name = "instrument"
    for i in range(len(dates)):
        panels["high"].iloc[i] = [5.0, 10 + i, 9 + i, 1.0 + 5e-13, 12 + i]
        panels["low"].iloc[i] = [5.0, 8 + i, 7 + i, 1.0, 8 + i]
        panels["close"].iloc[i] = [5.0, 9 + i, 8 + i, 1.0 + 2e-13, 9 + i]
    panels["close"].iloc[3, panels["close"].columns.get_loc("missing")] = np.nan
    panels["high"].iloc[5, panels["high"].columns.get_loc("infinite")] = np.inf
    source_data = {
        name: frame.stack(future_stack=True).rename(name) for name, frame in panels.items()
    }
    formula = migrate_catalog_r14_technical_formula(
        f"WilliamsR({_field('high')}, {_field('low')}, {_field('close')}, 3)",
        logic="williams_centered", enabled=True,
    ).formula
    for name in ("high", "low", "close"):
        formula = formula.replace(_field(name), name)
    factor = Factor(
        name="williams", expr=DSLParser(surface="compat_research").parse(formula),
        source_expr=formula, surface="compat_research",
    )
    actual = FactorEngine(
        PandasBackend(), InMemorySeriesSource(data=source_data), run_mode="research"
    ).run_many([factor])["results"]["williams"]
    expected_panel = WilliamsR().calculate(
        panels["high"], panels["low"], panels["close"], window=3
    )
    expected = expected_panel.stack(future_stack=True)
    expected.name = actual.name
    pd.testing.assert_series_equal(actual, expected)
    tiny = actual.xs("tiny", level="instrument")
    assert tiny.notna().all(), "tiny nonzero ranges must not be epsilon-masked"
    assert actual.xs("flat", level="instrument").isna().all()


def test_rewrite_compiles_with_real_no_read_engine():
    sys.path.insert(0, str(Path("evidence/factor_catalog_20260915").resolve()))
    try:
        from compile_catalog import build_runtime
        from factor_engine.api.factor import Factor

        parser, engine = build_runtime()
        formula = migrate_catalog_r14_technical_formula(
            f"WilliamsR({_field('high')}, {_field('low')}, {_field('close')}, 14)",
            logic="williams_centered", enabled=True,
        ).formula
        assert engine.compile(Factor(
            name="r14_williams_r", expr=parser.parse(formula), source_expr=formula,
            surface="compat_research",
        )) is not None
    finally:
        sys.path.pop(0)


def test_input_bounds_and_types():
    with pytest.raises(TypeError):
        migrate_catalog_r14_technical_formula(1, logic="WilliamsR", enabled=True)
    with pytest.raises(TypeError):
        migrate_catalog_r14_technical_formula("x", logic=None, enabled=True)
    with pytest.raises(SyntaxError):
        migrate_catalog_r14_technical_formula("WilliamsR(", logic="WilliamsR", enabled=True)
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_r14_technical_formula("测" * 22_000, enabled=True)
