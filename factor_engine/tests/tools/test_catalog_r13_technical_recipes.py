import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r13_technical_recipes import (
    migrate_catalog_r13_technical_formula,
)


def _field(name, table="StockDailyBarAdj"):
    return f"field('{name}', table='{table}')"


OHLCV = ", ".join(_field(name) for name in ("open", "high", "low", "close", "volume"))


@pytest.mark.parametrize(
    ("source", "logic", "expected"),
    [
        (
            "DPO(close)", "detrended_cycle_position",
            "subtract(close, ts_delay(ts_mean(close, 20, 1), 11))",
        ),
        (
            f"DPO({_field('close')}, 20)", "detrended_cycle_position",
            f"subtract({_field('close')}, ts_delay(ts_mean({_field('close')}, 20, 1), 11))",
        ),
        (
            "DPO(close, 9)", "DPO cycle",
            "subtract(close, ts_delay(ts_mean(close, 9, 1), 5))",
        ),
        (
            f"DPO({OHLCV})", "verified technical operator DPO",
            f"subtract({_field('close')}, ts_delay(ts_mean({_field('close')}, 20, 1), 11))",
        ),
    ],
)
def test_exact_shapes_expand_with_explicit_support_policy(source, logic, expected):
    result = migrate_catalog_r13_technical_formula(source, logic=logic, enabled=True)
    assert result.formula == expected
    assert "min_periods=1" in result.changes[0]
    assert migrate_catalog_r13_technical_formula(
        result.formula, logic=logic, enabled=True
    ).changes == ()


def test_disabled_by_default():
    assert migrate_catalog_r13_technical_formula("DPO(close)", logic="DPO").changes == ()


@pytest.mark.parametrize(
    "formula,logic",
    [
        ("DPO(close)", "generic cycle"),
        ("DPO(open, 20)", "DPO"),
        ("DPO(close, window=20)", "DPO"),
        ("DPO(close, 20.0)", "DPO"),
        ("DPO(close, 0)", "DPO"),
        (f"DPO({_field('close', 'Other')}, 20)", "DPO"),
        (f"DPO({OHLCV}, 20)", "DPO"),
        ("other.DPO(close, 20)", "DPO"),
    ],
)
def test_uncertain_shapes_fail_closed(formula, logic):
    result = migrate_catalog_r13_technical_formula(formula, logic=logic, enabled=True)
    assert result.formula == formula
    assert result.changes == ()


def test_golden_matches_legacy_dpo_including_warmup_and_nan_behavior():
    from factor_engine.cleaned_operators.technical.signal import DPO
    from factor_engine.cleaned_operators import OperatorRegistry, load_all

    load_all()
    index = pd.date_range("2026-01-01", periods=35)
    close = pd.DataFrame(
        {"A": np.linspace(10.0, 30.0, 35), "B": np.linspace(40.0, 20.0, 35)},
        index=index,
    )
    close.loc[index[4], "A"] = np.nan
    close.loc[index[18], "B"] = np.nan
    legacy = DPO().calculate(close, window=20)
    mean = OperatorRegistry.get("ts_mean").calculate(close, window=20, min_periods=1)
    expanded = close - OperatorRegistry.get("ts_delay").calculate(mean, n=11)
    pd.testing.assert_frame_equal(expanded, legacy)
    assert expanded.iloc[:11].isna().all().all()
    assert expanded.iloc[11:].notna().any().all()


def test_rewrite_compiles_with_real_no_read_engine():
    sys.path.insert(0, str(Path("evidence/factor_catalog_20260915").resolve()))
    try:
        from compile_catalog import build_runtime
        from factor_engine.api.factor import Factor

        parser, engine = build_runtime()
        formula = migrate_catalog_r13_technical_formula(
            f"DPO({_field('close')}, 20)", logic="detrended_cycle_position", enabled=True
        ).formula
        plan = engine.compile(Factor(
            name="r13_dpo", expr=parser.parse(formula), source_expr=formula,
            surface="compat_research",
        ))
        assert plan is not None
    finally:
        sys.path.pop(0)


def test_real_run_many_matches_small_synthetic_oracle():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dates = pd.date_range("2026-01-01", periods=35)
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B"]], names=["timestamp", "instrument"]
    )
    close = pd.Series(
        [float(i + offset) for i in range(35) for offset in (10, 20)],
        index=index,
        name="close",
    )
    parser = DSLParser(surface="compat_research")
    formulas = [
        migrate_catalog_r13_technical_formula("DPO(close, 20)", logic="DPO", enabled=True).formula,
        migrate_catalog_r13_technical_formula("DPO(close, 9)", logic="DPO", enabled=True).formula,
    ]
    factors = [Factor(name=f"dpo_{i}", expr=parser.parse(formula), source_expr=formula, surface="compat_research") for i, formula in enumerate(formulas)]
    output = FactorEngine(
        PandasBackend(), InMemorySeriesSource(data={"close": close}), run_mode="research"
    ).run_many(factors)
    assert set(output["results"]) == {"dpo_0", "dpo_1"}
    panel = close.unstack("instrument")
    for position, window in enumerate((20, 9)):
        lag = window // 2 + 1
        expected = (
            panel - panel.rolling(window, min_periods=1).mean().shift(lag)
        ).stack(future_stack=True)
        expected.name = None
        pd.testing.assert_series_equal(output["results"][f"dpo_{position}"], expected)


def test_invalid_and_oversized_inputs_are_bounded():
    with pytest.raises(SyntaxError):
        migrate_catalog_r13_technical_formula("DPO(", logic="DPO", enabled=True)
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_r13_technical_formula("x" * 65_537, enabled=True)


def test_unicode_before_multiple_edits_preserves_exact_source_text():
    formula = "add(field('中文字段', table='Demo'), add(DPO(close), DPO(close, 9)))"
    result = migrate_catalog_r13_technical_formula(
        formula, logic="去趋势价格振荡", enabled=True,
    )
    assert result.formula == (
        "add(field('中文字段', table='Demo'), "
        "add(subtract(close, ts_delay(ts_mean(close, 20, 1), 11)), "
        "subtract(close, ts_delay(ts_mean(close, 9, 1), 5))))"
    )
    assert len(result.changes) == 2


def test_nested_call_rewrites_only_valid_inner_shape_without_span_corruption():
    formula = "DPO(DPO(close), 20)"
    result = migrate_catalog_r13_technical_formula(formula, logic="DPO", enabled=True)
    assert result.formula == (
        "DPO(subtract(close, ts_delay(ts_mean(close, 20, 1), 11)), 20)"
    )
    assert len(result.changes) == 1


@pytest.mark.parametrize(
    "formula",
    [
        "DPO(close, window=20)",
        "DPO(close, window=20, window=20)",
        "DPO(close, **opts)",
        "DPO(close, 20, **opts)",
    ],
)
def test_keywords_duplicates_and_expansions_are_never_discarded(formula):
    result = migrate_catalog_r13_technical_formula(formula, logic="DPO", enabled=True)
    assert result.formula == formula
    assert result.changes == ()


@pytest.mark.parametrize("formula", [None, b"DPO(close)", 1])
def test_formula_must_be_string_even_when_disabled(formula):
    with pytest.raises(TypeError, match="formula must be a string"):
        migrate_catalog_r13_technical_formula(formula, enabled=False)


def test_utf8_budget_counts_bytes_before_disabled_noop():
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_r13_technical_formula("'" + "界" * 22_000 + "'", enabled=False)
