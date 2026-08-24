# -*- coding: utf-8 -*-
"""Golden semantic tests for audited factor-engine operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.api.dsl_parser import parse_expr
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.ir.analyzer import Analyzer


@pytest.fixture(scope="module", autouse=True)
def _load_operators():
    load_all()


def test_ts_product_preserves_zero_and_negative_sign():
    op = OperatorRegistry.get("ts_product")
    x = pd.DataFrame({"A": [-2.0, 3.0, 0.0, 4.0]})
    result = op.calculate(x, 2, min_periods=2)

    expected = pd.DataFrame({"A": [np.nan, -6.0, 0.0, 0.0]})
    pd.testing.assert_frame_equal(result, expected)


def test_ts_product_overflow_fails_closed():
    op = OperatorRegistry.get("ts_product")
    x = pd.DataFrame({"A": [1e308, 1e308]})
    result = op.calculate(x, 2, min_periods=2)
    assert np.isnan(result.iloc[-1, 0])


def test_ts_mad_is_median_absolute_deviation():
    op = OperatorRegistry.get("ts_mad")
    x = pd.DataFrame({"A": [1.0, 2.0, 100.0]})
    result = op.calculate(x, 3, min_periods=3)
    assert result.iloc[-1, 0] == pytest.approx(1.0)


def test_group_percentile_defaults_to_top_tail():
    op = OperatorRegistry.get("group_percentile")
    x = pd.DataFrame([[10.0, 1.0, 8.0, 2.0]], columns=list("ABCD"))
    group = pd.DataFrame([[1.0, 1.0, 2.0, 2.0]], columns=list("ABCD"))

    top = op.calculate(x, group, 0.5)
    bottom = op.calculate(x, group, 0.5, side="bottom")

    pd.testing.assert_frame_equal(
        top,
        pd.DataFrame([[1.0, 0.0, 1.0, 0.0]], columns=list("ABCD")),
    )
    pd.testing.assert_frame_equal(
        bottom,
        pd.DataFrame([[0.0, 1.0, 0.0, 1.0]], columns=list("ABCD")),
    )


def test_group_percentile_does_not_silently_fallback():
    op = OperatorRegistry.get("group_percentile")
    x = pd.DataFrame([[10.0, 1.0]], columns=["A", "B"])
    with pytest.raises(ValueError, match="group labels are missing"):
        op.calculate(x, None, 0.5)


def test_div_or_null_really_returns_null():
    op = OperatorRegistry.get("div_or_null")
    x = pd.DataFrame({"A": [4.0, 4.0, 4.0]})
    y = pd.DataFrame({"A": [2.0, 0.0, np.nan]})
    result = op.calculate(x, y)

    expected = pd.DataFrame({"A": [2.0, np.nan, np.nan]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "canonical",
    [
        # 2026-08 第三轮:group_decay_linear / ts_sum_decay / trade_when / rank_corr
        # 已通过生产认证升到 daily,移出本守卫。
        "vp_weighted_price",
        "vpmacd",
        "vpmacd_signal",
        "rolling_beta_to_market",
        "micro_vpin",
    ],
)
def test_misleading_or_experimental_ops_are_not_daily(canonical):
    assert classify_canonical(canonical) != "daily"


def test_audited_nonproduction_ops_remain_fail_closed():
    # 2026-08 daily migration: these audited factor operators are production targets and
    # now live on the daily surface; they must never be research/unsafe.
    assert classify_canonical("ts_product") == "daily"
    assert classify_canonical("ts_mad") == "daily"
    assert classify_canonical("group_percentile") == "daily"
    assert classify_canonical("div_or_null") != "daily"


def test_nested_delay_and_rolling_lookback_is_additive():
    expression = parse_expr("ts_mean(ts_delay(close, 5), 20)")
    analysis = Analyzer().lower(expression)
    assert analysis.lookback == 24


def test_nested_rolling_lookback_is_additive():
    expression = parse_expr("ts_mean(ts_mean(close, 20), 20)")
    analysis = Analyzer().lower(expression)
    assert analysis.lookback == 38


def test_multivariate_window_parameter_is_discovered():
    expression = parse_expr("ts_corr(close, volume, 20)")
    analysis = Analyzer().lower(expression)
    assert analysis.lookback == 19
    assert analysis.referenced_columns == {"close", "volume"}
