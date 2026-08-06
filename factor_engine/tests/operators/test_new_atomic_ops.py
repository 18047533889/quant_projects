# -*- coding: utf-8 -*-
"""Numeric and causal-invariance tests for the 2026-08 operator expansion (P0)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

P0_CANONICALS = frozenset(
    """ts_quantile_range ts_trimmed_mean ts_robust_zscore
    ts_positive_ratio ts_negative_ratio ts_zero_ratio
    ts_abs_concentration ts_abs_entropy
    ts_min_if ts_max_if ts_quantile_if ts_corr_if ts_beta_if ts_regression_resid_if
    ts_transition_count ts_time_since_change ts_event_spacing_mean ts_event_spacing_cv
    ts_downside_deviation ts_upside_deviation ts_current_drawdown_duration
    ts_time_under_water ts_best_lag_corr ts_price_delay
    group_ex_self_mean group_ex_self_weighted_mean hierarchical_group_neutralize
    cs_robust_resid overnight_return open_close_return open_to_vwap_return
    vwap_to_close_return ashare_limit_distance ashare_limit_touch
    ashare_limit_one_price ashare_limit_failed ashare_limit_open_break
    """.split()
)


def _panel(rows: int = 60, cols: int = 3, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    names = ["A", "B", "C"][:cols]
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((rows, cols)), index=idx, columns=names)


def _group(rows: int = 60, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    names = ["A", "B", "C"][:cols]
    return pd.DataFrame(np.resize(np.array(["X", "Y", "X"]), (rows, cols)), index=idx, columns=names)


def _cond(panel: pd.DataFrame) -> pd.DataFrame:
    return (panel > 0).astype(float)


@pytest.mark.parametrize("name", sorted(P0_CANONICALS))
def test_p0_operator_registered_and_extended_surface(name: str) -> None:
    assert OperatorRegistry.get(name) is not None
    assert classify_canonical(name) == "extended"


def test_p0_operators_preserve_shape_and_are_deterministic() -> None:
    panel = _panel()
    calls: dict[str, list[object]] = {
        "ts_quantile_range": [panel],
        "ts_trimmed_mean": [panel],
        "ts_robust_zscore": [panel],
        "ts_positive_ratio": [panel],
        "ts_negative_ratio": [panel],
        "ts_zero_ratio": [panel],
        "ts_abs_concentration": [panel],
        "ts_abs_entropy": [panel],
        "ts_min_if": [panel, _cond(panel)],
        "ts_max_if": [panel, _cond(panel)],
        "ts_quantile_if": [panel, _cond(panel)],
        "ts_corr_if": [panel, panel * 2.0 + 0.1, _cond(panel)],
        "ts_beta_if": [panel * 2.0 + 0.1, panel, _cond(panel)],
        "ts_regression_resid_if": [panel * 2.0 + 0.1, panel, _cond(panel)],
        "ts_transition_count": [_cond(panel)],
        "ts_time_since_change": [_cond(panel)],
        "ts_event_spacing_mean": [_cond(panel)],
        "ts_event_spacing_cv": [_cond(panel)],
        "ts_downside_deviation": [panel],
        "ts_upside_deviation": [panel],
        "ts_current_drawdown_duration": [panel],
        "ts_time_under_water": [panel],
        "ts_best_lag_corr": [panel, panel * 0.5],
        "ts_price_delay": [panel],
        "group_ex_self_mean": [panel, _group()],
        "group_ex_self_weighted_mean": [panel, panel.abs() + 1.0, _group()],
        "hierarchical_group_neutralize": [panel, _group(), _group()],
        "cs_robust_resid": [panel, panel * 0.5],
        "overnight_return": [panel + 10.0, panel + 9.0],
        "open_close_return": [panel + 10.0, panel + 11.0],
        "open_to_vwap_return": [panel + 10.0, panel + 10.5],
        "vwap_to_close_return": [panel + 10.5, panel + 11.0],
        "ashare_limit_distance": [panel + 10.0, panel + 10.05],
        "ashare_limit_touch": [panel + 10.0, panel + 10.0],
        "ashare_limit_one_price": [panel + 10.0, panel + 10.0, panel + 10.0],
        "ashare_limit_failed": [panel + 10.0, panel + 10.05],
        "ashare_limit_open_break": [panel + 10.0, panel + 10.0],
    }
    for name, args in calls.items():
        op = OperatorRegistry.get(name)
        first = op.calculate(*args)
        second = op.calculate(*args)
        assert first.shape == panel.shape, f"{name}: shape changed"
        assert first.index.equals(panel.index) and first.columns.equals(panel.columns)
        pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_p0_operators_are_prefix_causal() -> None:
    """结果只依赖当前行及历史行：截断未来输入后，历史输出不变。"""
    panel = _panel()
    name = "ts_robust_zscore"
    op = OperatorRegistry.get(name)
    full = op.calculate(panel)
    truncated = op.calculate(panel.iloc[:30])
    assert full.iloc[:30].equals(truncated)


def test_ts_quantile_range_is_iqr() -> None:
    panel = _panel()
    op = OperatorRegistry.get("ts_quantile_range")
    out = op.calculate(panel, window=60, q_low=0.25, q_high=0.75)
    expected = panel.quantile(0.75, axis=0) - panel.quantile(0.25, axis=0)
    for column in panel.columns:
        assert out[column].iloc[-1] == pytest.approx(float(expected[column]))


def test_overnight_return_numeric() -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    cols = ["A"]
    open_px = pd.DataFrame([10.0, 10.0, 10.0], index=idx, columns=cols)
    pre_close = pd.DataFrame([9.0, 10.0, 0.0], index=idx, columns=cols)
    op = OperatorRegistry.get("overnight_return")
    out = op.calculate(open_px, pre_close)
    assert out.iloc[0, 0] == pytest.approx(10.0 / 9.0 - 1.0)
    assert np.isnan(out.iloc[2, 0])


def test_group_ex_self_mean_excludes_own_contribution() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    x = pd.DataFrame([[1.0, 2.0, 3.0]], index=idx, columns=["A", "B", "C"])
    group = pd.DataFrame([["X", "X", "Y"]], index=idx, columns=["A", "B", "C"])
    op = OperatorRegistry.get("group_ex_self_mean")
    out = op.calculate(x, group)
    # A excludes self: mean of B only = 2.0 ; B excludes self: mean of A only = 1.0
    assert out.iloc[0, 0] == pytest.approx(2.0)
    assert out.iloc[0, 1] == pytest.approx(1.0)
    assert np.isnan(out.iloc[0, 2])


def test_conditional_beta_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    x = _panel(seed=2)
    y = 2.0 * x + 0.5 + rng.standard_normal(x.shape) * 0.01
    cond = _cond(x)
    op = OperatorRegistry.get("ts_beta_if")
    out = op.calculate(y, x, cond, window=60, min_periods=2)
    for column in x.columns:
        mask = cond[column] > 0
        xs = x[column][mask].to_numpy()
        ys = y[column][mask].to_numpy()
        slope = float(np.mean((xs - np.mean(xs)) * (ys - np.mean(ys))) / np.var(xs))
        assert out[column].iloc[-1] == pytest.approx(slope, rel=1e-2)
