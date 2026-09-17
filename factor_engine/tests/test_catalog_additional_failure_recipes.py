import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_additional_failure_recipes import (
    migrate_catalog_additional_failure_formula,
)


@pytest.mark.parametrize(
    ("formula", "logic", "expected", "fragment"),
    [
        (
            "rank(ts_transition_count(ret, 20, 0.0))",
            "均值回复速度和方向切换频率",
            "rank(ts_transition_count(gt(ret, 0.0), 20, 'break'))",
            "direction threshold 0.0",
        ),
        (
            "rank(ts_gap_fill_ratio(close, open, pre_close, 0.01))",
            "隔夜缺口填补与日内价格修复",
            "rank(ts_gap_fill_ratio(close, open, pre_close, 60))",
            "legacy 0.01 cannot be interpreted",
        ),
        (
            "rank(event_historical_response_mean(ashare_limit_up_touch(high, high_limit, 0.005), ret))",
            "涨停触板事件后的历史收益响应",
            "rank(event_historical_response_mean(ret, ashare_limit_up_touch(high, high_limit, 0.005)))",
            "event expression first argument",
        ),
    ],
)
def test_exact_reviewed_shapes_are_semantically_redesigned(formula, logic, expected, fragment):
    result = migrate_catalog_additional_failure_formula(formula, logic=logic, enabled=True)
    assert result.formula == expected
    assert len(result.changes) == 1
    assert result.changes[0].startswith("SEMANTIC_REDESIGN")
    assert fragment in result.changes[0]
    assert migrate_catalog_additional_failure_formula(
        result.formula, logic=logic, enabled=True
    ).changes == ()


def test_migration_is_disabled_by_default_and_requires_corroborating_logic():
    formula = "ts_transition_count(ret, 20, 0.0)"
    assert migrate_catalog_additional_failure_formula(
        formula, logic="方向切换频率"
    ).formula == formula
    assert migrate_catalog_additional_failure_formula(
        formula, logic="generic signal", enabled=True
    ).changes == ()


@pytest.mark.parametrize(
    "formula,logic",
    [
        ("ts_autocorr_decay_half_life(ret, 60, 1, 20, 0.05)", "自相关衰减"),
        ("event_historical_response_mean(ret, gt(abs(ret), 0.02))", "历史事件响应"),
        ("event_historical_response_mean(foo(ret), ret)", "历史事件响应"),
        ("ts_gap_fill_ratio(close, open, pre_close, 20)", "缺口填补"),
    ],
)
def test_unapproved_or_already_canonical_shapes_fail_closed(formula, logic):
    result = migrate_catalog_additional_failure_formula(formula, logic=logic, enabled=True)
    assert result.formula == formula
    assert result.changes == ()


@pytest.mark.parametrize(
    "source,logic,expected_op",
    [
        ("ts_transition_count(ret, 20, 0.0)", "方向切换频率", "ts_transition_count"),
        ("ts_gap_fill_ratio(close, open, pre_close, 0.01)", "缺口填补", "ts_gap_fill_ratio"),
    ],
)
def test_rewrites_parse_and_lower_against_real_contract(source, logic, expected_op):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    formula = migrate_catalog_additional_failure_formula(
        source, logic=logic, enabled=True
    ).formula
    ir = Analyzer().lower(DSLParser(surface="compat_research").parse(formula)).ir
    assert ir.op == expected_op


def test_redesigned_calls_execute_on_small_real_panels():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    index = pd.date_range("2025-01-01", periods=80)
    ret = pd.DataFrame({"A": np.tile([-0.01, 0.02], 40)}, index=index)
    condition = (ret > 0.0).astype(float)
    transitions = OperatorRegistry.get(
        "ts_transition_count", backend="pandas_numpy"
    ).calculate(condition, window=20, missing_policy="break")
    assert transitions.iloc[-1, 0] == 19.0

    pre_close = pd.DataFrame({"A": np.full(80, 100.0)}, index=index)
    open_px = pd.DataFrame({"A": np.full(80, 102.0)}, index=index)
    close = pd.DataFrame({"A": np.full(80, 100.0)}, index=index)
    gap = OperatorRegistry.get("ts_gap_fill_ratio", backend="pandas_numpy").calculate(
        close, open_px, pre_close, window=60
    )
    assert np.isfinite(gap.iloc[-1, 0])

    event = pd.DataFrame({"A": np.arange(80) % 7 == 0}, index=index).astype(float)
    response = OperatorRegistry.get(
        "event_historical_response_mean", backend="pandas_numpy"
    ).calculate(ret, event, history_window=60, horizon=5, mode="sum", min_events=2)
    assert np.isfinite(response.iloc[-1, 0])


def test_invalid_and_oversized_inputs_are_bounded():
    with pytest.raises(SyntaxError):
        migrate_catalog_additional_failure_formula("ts_gap_fill_ratio(", enabled=True)
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_additional_failure_formula("x" * 65_537, enabled=True)
