import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r13_expression_recipes import (
    migrate_catalog_r13_expression_formula,
)


def _field(name, table="StockDailyBarAdj"):
    return f"field('{name}', table='{table}')"


OHLCV = ", ".join(_field(name) for name in ("open", "high", "low", "close", "volume"))


def test_exact_reviewed_shape_is_opt_in_and_idempotent():
    source = f"rank(OBV({OHLCV}))"
    assert migrate_catalog_r13_expression_formula(source, logic="OBV state").formula == source
    result = migrate_catalog_r13_expression_formula(source, logic="verified OBV state", enabled=True)
    expected = f"rank(OBV({_field('close')}, {_field('volume')}))"
    assert result.formula == expected
    assert len(result.changes) == 1
    assert "price=close" in result.changes[0]
    assert migrate_catalog_r13_expression_formula(
        result.formula, logic="verified OBV state", enabled=True
    ).changes == ()


@pytest.mark.parametrize(
    "formula,logic",
    [
        (f"OBV({OHLCV})", "generic volume state"),
        ("OBV(close, volume)", "OBV state"),
        (f"OBV({OHLCV}, window=20)", "OBV state"),
        (f"OBV({_field('open', 'Other')}, " + ", ".join(_field(n) for n in ("high", "low", "close", "volume")) + ")", "OBV state"),
        (f"OBV({_field('high')}, {_field('open')}, {_field('low')}, {_field('close')}, {_field('volume')})", "OBV state"),
        (f"other.OBV({OHLCV})", "OBV state"),
    ],
)
def test_ambiguous_keyword_nonfield_and_lookalike_calls_fail_closed(formula, logic):
    result = migrate_catalog_r13_expression_formula(formula, logic=logic, enabled=True)
    assert result.formula == formula
    assert result.changes == ()


def test_exact_candidate_nested_in_noncandidate_parent_is_rewritten_safely():
    nested = f"OBV({_field('open')}, {_field('high')}, {_field('low')}, OBV({OHLCV}), {_field('volume')})"
    result = migrate_catalog_r13_expression_formula(nested, logic="OBV", enabled=True)
    assert result.formula == (
        f"OBV({_field('open')}, {_field('high')}, {_field('low')}, "
        f"OBV({_field('close')}, {_field('volume')}), {_field('volume')})"
    )
    assert len(result.changes) == 1


def test_rewrite_compiles_with_real_no_read_engine():
    sys.path.insert(0, str(Path("evidence/factor_catalog_20260915").resolve()))
    try:
        from compile_catalog import build_runtime
        from factor_engine.api.factor import Factor

        parser, engine = build_runtime()
        formula = migrate_catalog_r13_expression_formula(
            f"OBV({OHLCV})", logic="verified OBV state", enabled=True
        ).formula
        expr = parser.parse(formula)
        plan = engine.compile(Factor(
            name="r13_obv_exact_shape", expr=expr, source_expr=formula,
            surface="compat_research",
        ))
        assert plan is not None
    finally:
        sys.path.pop(0)


def test_canonical_obv_matches_small_manual_golden_fixture():
    from factor_engine.cleaned_operators import OperatorRegistry, load_all

    load_all()
    index = pd.date_range("2026-01-01", periods=5)
    close = pd.DataFrame({"A": [10.0, 11.0, 10.0, 10.0, 12.0]}, index=index)
    volume = pd.DataFrame({"A": [100.0, 200.0, 300.0, 500.0, 400.0]}, index=index)
    actual = OperatorRegistry.get("OBV", backend="pandas_numpy").calculate(close, volume)
    expected = np.array([0.0, 200.0, -100.0, -100.0, 300.0])
    np.testing.assert_allclose(actual["A"].to_numpy(), expected)


def test_invalid_and_oversized_inputs_are_bounded():
    with pytest.raises(SyntaxError):
        migrate_catalog_r13_expression_formula("OBV(", logic="OBV", enabled=True)
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_r13_expression_formula("x" * 65_537, enabled=True)
