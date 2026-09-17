import numpy as np
import pandas as pd
import pytest


def _lower(formula):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    return Analyzer().lower(DSLParser(surface="compat_research").parse(formula)).ir


@pytest.mark.parametrize("op", ["gt", "ge", "lt", "le", "eq", "ne"])
def test_comparison_outputs_are_formal_masks(op):
    ir = _lower(f"{op}(ret, 0.0)")
    assert ir.semantic_attrs["semantic_kind"] == "MaskBool"
    assert ir.semantic_attrs["unit"] == "boolean"
    assert "price_basis" not in ir.semantic_attrs
    assert "mixed_semantic_kind" not in ir.semantic_attrs


@pytest.mark.parametrize("formula", [
    "and_(gt(ret, 0.0), le(ret, 0.02))",
    "or_(lt(ret, 0.0), ge(ret, 0.02))",
    "not_(eq(ret, 0.0))",
])
def test_logical_combinators_keep_mask_type(formula):
    assert _lower(formula).semantic_attrs["semantic_kind"] == "MaskBool"


def test_where_numeric_type_comes_from_value_branches_not_selector():
    ir = _lower("where(gt(ret, 0.0), ret, neg(ret))")
    assert ir.semantic_attrs["semantic_kind"] == "ReturnDecimal"
    assert ir.semantic_attrs["unit"] == "ratio"


def test_event_response_accepts_comparison_mask_and_executes_small_panel():
    ir = _lower("event_historical_response_mean(ret, gt(abs(ret), 0.02))")
    assert ir.op == "event_historical_response_mean"
    assert ir.inputs[1].semantic_attrs["semantic_kind"] == "MaskBool"

    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    idx = pd.date_range("2025-01-01", periods=100)
    ret = pd.DataFrame({"A": np.tile([-0.01, 0.03, 0.01, -0.04, 0.0], 20)}, index=idx)
    event = OperatorRegistry.get("gt", backend="pandas_numpy").calculate(abs(ret), 0.02)
    out = OperatorRegistry.get(
        "event_historical_response_mean", backend="pandas_numpy"
    ).calculate(ret, event, history_window=60, horizon=5, mode="sum", min_events=2)
    assert np.isfinite(out.iloc[-1, 0])


def test_r19_recipe_now_reorders_proven_comparison_mask():
    from factor_engine.tools.catalog_r19_type_recipes import migrate_formula

    source = "event_historical_response_mean(gt(abs(ret), 0.02), ret)"
    expected = "event_historical_response_mean(ret, gt(abs(ret), 0.02))"
    migrated, changes = migrate_formula(source)
    assert migrated == expected
    assert len(changes) == 1
    assert _lower(migrated).inputs[1].semantic_attrs["semantic_kind"] == "MaskBool"
