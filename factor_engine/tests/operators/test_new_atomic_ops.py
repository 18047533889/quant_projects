# -*- coding: utf-8 -*-
"""Numeric and causal-invariance tests for the 2026-08 operator expansion (P0)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

P0_CANONICALS = frozenset(
    """ts_quantile_range ts_trimmed_mean ts_robust_zscore_inclusive ts_robust_zscore_prior
    ts_positive_ratio ts_negative_ratio ts_zero_ratio
    ts_abs_concentration ts_abs_entropy
    ts_min_if ts_max_if ts_quantile_if ts_corr_if ts_beta_if ts_regression_resid_if
    ts_transition_count ts_time_since_change ts_event_spacing_mean ts_event_spacing_cv
    ts_downside_deviation ts_upside_deviation ts_current_drawdown_duration
    ts_time_under_water ts_best_lag_corr ts_price_delay
    group_ex_self_mean group_ex_self_weighted_mean hierarchical_group_neutralize
    cs_robust_resid overnight_return open_close_return open_to_vwap_return
    vwap_to_close_return ashare_limit_distance ashare_limit_up_touch
    ashare_limit_down_touch ashare_limit_one_price ashare_limit_failed
    ashare_open_at_upper_limit ashare_limit_open_failed
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
    # Daily production migration promotes the non-fail-closed P0 factor operators to the
    # daily surface; fail-closed (experimental/isolation) ones remain extended.
    assert classify_canonical(name) in {"daily", "extended"}


def test_p0_operators_preserve_shape_and_are_deterministic() -> None:
    panel = _panel()
    calls: dict[str, list[object]] = {
        "ts_quantile_range": [panel],
        "ts_trimmed_mean": [panel],
        "ts_robust_zscore_inclusive": [panel],
        "ts_robust_zscore_prior": [panel],
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
        "ts_price_delay": [panel, panel * 0.5],
        "group_ex_self_mean": [panel, _group()],
        "group_ex_self_weighted_mean": [panel, panel.abs() + 1.0, _group()],
        "hierarchical_group_neutralize": [panel, _group(), _group()],
        "cs_robust_resid": [panel, panel * 0.5],
        "overnight_return": [panel + 10.0, panel + 9.0, "raw"],
        "open_close_return": [panel + 10.0, panel + 11.0, "raw"],
        "open_to_vwap_return": [panel + 10.0, panel + 10.5, "raw"],
        "vwap_to_close_return": [panel + 10.5, panel + 11.0, "raw"],
        "ashare_limit_distance": [panel + 10.0, panel + 10.05],
        "ashare_limit_up_touch": [panel + 10.0, panel + 10.0],
        "ashare_limit_down_touch": [panel + 10.0, panel + 10.0],
        "ashare_limit_one_price": [panel + 10.0, panel + 10.0, panel + 10.0, panel + 10.0, panel + 10.0, panel + 10.0],
        "ashare_limit_failed": [panel + 10.0, panel + 10.0, panel + 10.05],
        "ashare_open_at_upper_limit": [panel + 10.0, panel + 10.0],
        "ashare_limit_open_failed": [panel + 10.0, panel + 10.0, panel + 10.0],
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
    out = op.calculate(open_px, pre_close, price_basis="raw")
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


def test_ashare_limit_ops_registration() -> None:
    """Verify all A-share limit operators are registered."""
    limit_ops = [
        "ashare_limit_distance",
        "ashare_limit_up_touch",
        "ashare_limit_down_touch",
        "ashare_limit_one_price",
        "ashare_limit_failed",
        "ashare_open_at_upper_limit",
        "ashare_limit_open_failed",
    ]
    for name in limit_ops:
        op = OperatorRegistry.get(name)
        assert op is not None, f"{name} not registered"
        assert op.metadata.category == "ashare"
        assert "pit_safe" in op.metadata.tags


def test_ashare_limit_distance_measures_gap() -> None:
    """ashare_limit_distance = close/upper_limit - 1."""
    idx = pd.date_range("2024-01-01", periods=3)
    close = pd.DataFrame([10.0, 10.5, 11.0], index=idx, columns=["A"])
    upper_limit = pd.DataFrame([10.0, 10.0, 10.0], index=idx, columns=["A"])
    op = OperatorRegistry.get("ashare_limit_distance")
    out = op.calculate(close, upper_limit)
    assert out.iloc[0, 0] == pytest.approx(0.0)
    assert out.iloc[1, 0] == pytest.approx(0.05)
    assert out.iloc[2, 0] == pytest.approx(0.1)


def test_ashare_limit_up_touch_within_tolerance() -> None:
    """ashare_limit_up_touch: high >= upper_limit - tolerance."""
    idx = pd.date_range("2024-01-01", periods=4)
    high = pd.DataFrame([10.0, 10.048, 10.002, 9.95], index=idx, columns=["A"])
    upper_limit = pd.DataFrame([10.05, 10.05, 10.05, 10.05], index=idx, columns=["A"])
    op = OperatorRegistry.get("ashare_limit_up_touch")
    out = op.calculate(high, upper_limit, tick_tolerance=0.005)
    # 10.0 < 10.05 - 0.005 = 10.045 → False
    # 10.048 >= 10.045 → True
    # 10.002 < 10.045 → False
    # 9.95 < 10.045 → False
    assert out.iloc[0, 0] == 0.0
    assert out.iloc[1, 0] == 1.0
    assert out.iloc[2, 0] == 0.0
    assert out.iloc[3, 0] == 0.0


def test_ashare_limit_one_price_both_bounds() -> None:
    """ashare_limit_one_price: OHLC all at limit price within tolerance."""
    idx = pd.date_range("2024-01-01", periods=2)
    # Case 1: one-price board at upper limit
    open_px = pd.DataFrame([10.05, 9.0], index=idx, columns=["A"])
    high = pd.DataFrame([10.05, 9.1], index=idx, columns=["A"])
    low = pd.DataFrame([10.05, 8.9], index=idx, columns=["A"])
    close = pd.DataFrame([10.05, 9.0], index=idx, columns=["A"])
    upper_limit = pd.DataFrame([10.05, 10.05], index=idx, columns=["A"])
    lower_limit = pd.DataFrame([9.0, 9.0], index=idx, columns=["A"])
    op = OperatorRegistry.get("ashare_limit_one_price")
    out = op.calculate(open_px, high, low, close, upper_limit, lower_limit, side="up", tick_tolerance=0.005)
    assert out.iloc[0, 0] == 1.0  # All OHLC at upper limit
    assert out.iloc[1, 0] == 0.0  # Range too wide


def test_ashare_limit_failed_detects_broken_limit() -> None:
    """ashare_limit_failed: high touched limit but close did not hold."""
    idx = pd.date_range("2024-01-01", periods=3)
    high = pd.DataFrame([10.05, 10.05, 10.0], index=idx, columns=["A"])
    close = pd.DataFrame([10.0, 10.05, 10.0], index=idx, columns=["A"])
    upper_limit = pd.DataFrame([10.05, 10.05, 10.05], index=idx, columns=["A"])
    op = OperatorRegistry.get("ashare_limit_failed")
    out = op.calculate(high, close, upper_limit, tick_tolerance=0.005)
    # Day 0: high=10.05 (touched), close=10.0 (not held) → Failed
    # Day 1: high=10.05 (touched), close=10.05 (held) → Not failed
    # Day 2: high=10.0 (not touched) → Not failed
    assert out.iloc[0, 0] == 1.0
    assert out.iloc[1, 0] == 0.0
    assert out.iloc[2, 0] == 0.0


def test_ashare_open_at_upper_limit_detects_limit_open() -> None:
    """ashare_open_at_upper_limit: open >= upper_limit - tolerance."""
    idx = pd.date_range("2024-01-01", periods=3)
    open_px = pd.DataFrame([10.05, 10.048, 10.0], index=idx, columns=["A"])
    upper_limit = pd.DataFrame([10.05, 10.05, 10.05], index=idx, columns=["A"])
    op = OperatorRegistry.get("ashare_open_at_upper_limit")
    out = op.calculate(open_px, upper_limit, tick_tolerance=0.005)
    # 10.05 >= 10.045 → True
    # 10.048 >= 10.045 → True
    # 10.0 < 10.045 → False
    assert out.iloc[0, 0] == 1.0
    assert out.iloc[1, 0] == 1.0
    assert out.iloc[2, 0] == 0.0


def test_ashare_limit_open_failed_detects_break() -> None:
    """ashare_limit_open_failed: opened at limit then broke below intraday."""
    idx = pd.date_range("2024-01-01", periods=3)
    open_px = pd.DataFrame([10.05, 10.05, 10.0], index=idx, columns=["A"])
    low = pd.DataFrame([10.0, 10.05, 10.0], index=idx, columns=["A"])
    upper_limit = pd.DataFrame([10.05, 10.05, 10.05], index=idx, columns=["A"])
    op = OperatorRegistry.get("ashare_limit_open_failed")
    out = op.calculate(open_px, low, upper_limit, tick_tolerance=0.005)
    # Day 0: open=10.05 (at limit), low=10.0 (broke) → Failed
    # Day 1: open=10.05 (at limit), low=10.05 (held) → Not failed
    # Day 2: open=10.0 (not at limit) → Not failed
    assert out.iloc[0, 0] == 1.0
    assert out.iloc[1, 0] == 0.0
    assert out.iloc[2, 0] == 0.0


def test_group_ext_registration() -> None:
    """所有 group_ext 算子已正确注册并归类到 extended surface。"""
    group_ext_ops = [
        "group_ex_self_mean",
        "group_ex_self_weighted_mean",
        "hierarchical_group_neutralize",
        "cs_trimmed_ols_resid",
        "cs_huber_resid",
        "cs_lad_resid",
    ]
    for name in group_ext_ops:
        assert OperatorRegistry.get(name) is not None, f"{name} not registered"
        surface = classify_canonical(name)
        assert surface in {"daily", "extended"}, f"{name} surface={surface}"


def test_group_ex_self_mean_excludes_own_value() -> None:
    """group_ex_self_mean 应排除自身值，只计算组内其他成员的均值。"""
    idx = pd.date_range("2024-01-01", periods=1)
    x = pd.DataFrame([[1.0, 2.0, 3.0, 10.0]], index=idx, columns=["A", "B", "C", "D"])
    group = pd.DataFrame([["G1", "G1", "G2", "G1"]], index=idx, columns=["A", "B", "C", "D"])
    op = OperatorRegistry.get("group_ex_self_mean")
    out = op.calculate(x, group)
    # G1 group: A, B, D (values 1.0, 2.0, 10.0)
    # A excludes self: mean(2.0, 10.0) = 6.0
    # B excludes self: mean(1.0, 10.0) = 5.5
    # D excludes self: mean(1.0, 2.0) = 1.5
    # C is alone in G2: NaN
    assert out.iloc[0, 0] == pytest.approx(6.0)
    assert out.iloc[0, 1] == pytest.approx(5.5)
    assert np.isnan(out.iloc[0, 2])
    assert out.iloc[0, 3] == pytest.approx(1.5)


def test_hierarchical_group_neutralize_two_stages() -> None:
    """hierarchical_group_neutralize 应使用复合 (group, subgroup) 键去均值。"""
    idx = pd.date_range("2024-01-01", periods=1)
    # 4 stocks: (G1,S1): [1,2], (G1,S2): [8], (G2,S1): [5]
    x = pd.DataFrame([[1.0, 2.0, 8.0, 5.0]], index=idx, columns=["A", "B", "C", "D"])
    group = pd.DataFrame([["G1", "G1", "G1", "G2"]], index=idx, columns=["A", "B", "C", "D"])
    subgroup = pd.DataFrame([["S1", "S1", "S2", "S1"]], index=idx, columns=["A", "B", "C", "D"])
    op = OperatorRegistry.get("hierarchical_group_neutralize")
    out = op.calculate(x, group, subgroup)
    # (G1,S1): mean=1.5 -> residuals: -0.5, +0.5
    # (G1,S2): mean=8.0 -> residual: 0.0
    # (G2,S1): mean=5.0 -> residual: 0.0
    assert out.iloc[0, 0] == pytest.approx(-0.5)
    assert out.iloc[0, 1] == pytest.approx(0.5)
    assert out.iloc[0, 2] == pytest.approx(0.0)
    assert out.iloc[0, 3] == pytest.approx(0.0)


def test_cs_robust_resid_trims_outliers() -> None:
    """cs_trimmed_ols_resid 应截尾 trim_ratio 的极端 x 样本后拟合。"""
    idx = pd.date_range("2024-01-01", periods=1)
    # x: 20 stocks from 1 to 20; y = 2*x + noise, but add an outlier
    x_vals = list(range(1, 21))
    y_vals = [2.0 * v + 0.1 for v in x_vals]
    y_vals[-1] = 100.0  # outlier at x=20
    x = pd.DataFrame([x_vals], index=idx)
    y = pd.DataFrame([y_vals], index=idx)
    op = OperatorRegistry.get("cs_trimmed_ols_resid")
    # trim_ratio=0.1 removes 2 from each tail (x=1,2 and x=19,20)
    out = op.calculate(y, x, trim_ratio=0.1, add_intercept=True)
    # After trimming, the fit should be close to y=2*x
    # The residual at x=10 (middle) should be near 0
    mid_idx = 9  # x=10
    assert abs(out.iloc[0, mid_idx]) < 1.0


def test_cs_huber_resid_robust_to_outliers() -> None:
    """cs_huber_resid 应对 y 端离群点稳健（Huber M-估计量）。"""
    idx = pd.date_range("2024-01-01", periods=1)
    x_vals = list(range(1, 21))
    # Nonzero inlier noise gives the adaptive-MAD estimator a defined scale.
    # An exactly linear inlier majority plus one outlier is scale-degenerate
    # and intentionally fails closed (covered by the M01 shared-kernel tests).
    y_vals = [2.0 * v + 0.05 * np.sin(v) for v in x_vals]
    y_vals[0] = 100.0  # y outlier at x=1
    x = pd.DataFrame([x_vals], index=idx)
    y = pd.DataFrame([y_vals], index=idx)
    op = OperatorRegistry.get("cs_huber_resid")
    out = op.calculate(y, x, add_intercept=False)
    # Huber should downweight the outlier; the middle residuals should be small
    mid_idx = 10  # x=11
    assert abs(out.iloc[0, mid_idx]) < 2.0
    assert not np.isnan(out.iloc[0, mid_idx])


def test_cs_lad_resid_median_regression() -> None:
    """cs_lad_resid 应最小化绝对残差和（L1 / median regression）。"""
    idx = pd.date_range("2024-01-01", periods=1)
    x_vals = list(range(1, 21))
    y_vals = [2.0 * v for v in x_vals]
    y_vals[0] = 100.0  # y outlier
    x = pd.DataFrame([x_vals], index=idx)
    y = pd.DataFrame([y_vals], index=idx)
    op = OperatorRegistry.get("cs_lad_resid")
    out = op.calculate(y, x, add_intercept=False)
    # LAD should be robust; the middle residuals should be small
    mid_idx = 10
    assert abs(out.iloc[0, mid_idx]) < 2.0
    assert not np.isnan(out.iloc[0, mid_idx])


def test_direction_concentration_registration() -> None:
    """Direction concentration operators are registered and on extended surface."""
    operators = ["ts_positive_ratio", "ts_negative_ratio", "ts_zero_ratio",
                 "ts_abs_concentration", "ts_abs_entropy"]
    for name in operators:
        assert OperatorRegistry.get(name) is not None, f"{name} not registered"
        surface = classify_canonical(name)
        assert surface in {"daily", "extended"}, f"{name} surface={surface}"


def test_ts_positive_negative_ratio_sum_to_one_when_no_zeros() -> None:
    """When no values are exactly zero, positive_ratio + negative_ratio should ≈ 1."""
    idx = pd.date_range("2024-01-01", periods=20)
    # Create values that are clearly positive or negative (no zeros)
    values = np.array([1.0, -2.0, 3.0, -1.5, 2.5, -3.0, 0.5, -0.8, 1.2, -1.8,
                       2.1, -2.3, 1.7, -1.1, 0.9, -2.8, 1.4, -0.6, 2.0, -1.3])
    x = pd.DataFrame(values, index=idx, columns=["A"])

    op_pos = OperatorRegistry.get("ts_positive_ratio")
    op_neg = OperatorRegistry.get("ts_negative_ratio")

    pos_ratio = op_pos.calculate(x, window=20, threshold=0.0, min_periods=1)
    neg_ratio = op_neg.calculate(x, window=20, threshold=0.0, min_periods=1)

    # At the last row, sum should be 1.0 (all values are either > 0 or < 0)
    total = pos_ratio.iloc[-1, 0] + neg_ratio.iloc[-1, 0]
    assert total == pytest.approx(1.0, rel=1e-6)


def test_ts_abs_concentration_hhi_formula() -> None:
    """ts_abs_concentration computes HHI = sum(share_i^2) of absolute values."""
    idx = pd.date_range("2024-01-01", periods=4)
    # Simple case: |values| = [1, 2, 3, 4], total = 10
    # shares = [0.1, 0.2, 0.3, 0.4]
    # HHI = 0.01 + 0.04 + 0.09 + 0.16 = 0.30
    x = pd.DataFrame([1.0, -2.0, 3.0, -4.0], index=idx, columns=["A"])
    op = OperatorRegistry.get("ts_abs_concentration")
    out = op.calculate(x, window=4, min_periods=1)

    expected_hhi = 0.01 + 0.04 + 0.09 + 0.16
    assert out.iloc[-1, 0] == pytest.approx(expected_hhi, rel=1e-6)


def test_ts_abs_entropy_normalized_range() -> None:
    """ts_abs_entropy with normalize=True should output values in [0, 1]."""
    idx = pd.date_range("2024-01-01", periods=20)
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.standard_normal(20), index=idx, columns=["A"])

    op = OperatorRegistry.get("ts_abs_entropy")
    out = op.calculate(x, window=10, normalize=True, min_periods=1)

    # All finite values should be in [0, 1] when normalized
    finite_values = out[np.isfinite(out)].to_numpy().flatten()
    assert np.all(finite_values >= 0.0), "Normalized entropy should be >= 0"
    assert np.all(finite_values <= 1.0), "Normalized entropy should be <= 1"


def test_downside_risk_registration() -> None:
    """Verify all downside risk operators are registered."""
    names = [
        "ts_downside_deviation",
        "ts_upside_deviation",
        "ts_current_drawdown_duration",
        "ts_time_under_water",
        "ts_best_lag_corr",  # alias of ts_best_lag_corr_raw
        "ts_price_delay",
    ]
    for name in names:
        op = OperatorRegistry.get(name)
        assert op is not None, f"Operator {name} should be registered"


def test_ts_downside_deviation_only_negative_values() -> None:
    """Downside deviation only considers negative values below target."""
    idx = pd.date_range("2024-01-01", periods=5)
    # Mix of positive and negative returns
    x = pd.DataFrame([0.02, -0.01, 0.03, -0.02, -0.01], index=idx, columns=["A"])
    op = OperatorRegistry.get("ts_downside_deviation")
    out = op.calculate(x, window=5, target=0.0, min_periods=2)

    # At the last row, window includes all 5 values: [0.02, -0.01, 0.03, -0.02, -0.01]
    # Below target=0: [0, -0.01, 0, -0.02, -0.01]
    # Downside dev = sqrt(mean([0^2, 0.01^2, 0^2, 0.02^2, 0.01^2]))
    below = np.array([0.0, -0.01, 0.0, -0.02, -0.01])
    expected = np.sqrt(np.mean(below ** 2))
    assert out.iloc[-1, 0] == pytest.approx(expected, rel=1e-6)


def test_ts_current_drawdown_duration_resets_at_new_high() -> None:
    """Drawdown duration resets to 0 when price reaches new high."""
    idx = pd.date_range("2024-01-01", periods=8)
    # Price sequence: goes up, then down, then new high
    x = pd.DataFrame([10.0, 12.0, 11.0, 10.5, 10.0, 11.0, 13.0, 12.5], index=idx, columns=["A"])
    op = OperatorRegistry.get("ts_current_drawdown_duration")
    out = op.calculate(x, window=20)

    # At index 1: new high (12.0), duration = 0
    assert out.iloc[1, 0] == 0.0
    # At index 2: below 12.0, duration = 1
    assert out.iloc[2, 0] == 1.0
    # At index 3: still below 12.0, duration = 2
    assert out.iloc[3, 0] == 2.0
    # At index 6: new high (13.0), duration = 0
    assert out.iloc[6, 0] == 0.0
    # At index 7: below 13.0, duration = 1
    assert out.iloc[7, 0] == 1.0


def test_ts_best_lag_corr_finds_optimal_lag() -> None:
    """Best lag correlation finds the lag with highest |corr|."""
    idx = pd.date_range("2024-01-01", periods=30)
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.standard_normal(30), index=idx, columns=["A"])
    # y lags x by 2 periods with some noise
    y_vals = np.concatenate([np.full(2, np.nan), x["A"].values[:-2]]) + rng.standard_normal(30) * 0.1
    y = pd.DataFrame(y_vals, index=idx, columns=["A"])

    op = OperatorRegistry.get("ts_best_lag_corr")
    out = op.calculate(y, x, window=20, max_lag=5)

    # Should find a correlation (exact value depends on noise, but should be > 0)
    assert out.iloc[-1, 0] > 0.5, "Should detect strong lagged correlation"


# ---- State Event 族测试 (Phase 1 A-class atomic operators) ----


def test_state_event_registration() -> None:
    """验证状态事件族四个算子已正确注册并归类到 extended surface。"""
    state_event_ops = [
        "ts_transition_count",
        "ts_time_since_change",
        "ts_event_spacing_mean",
        "ts_event_spacing_cv",
    ]
    for name in state_event_ops:
        assert OperatorRegistry.get(name) is not None, f"{name} not registered"
        surface = classify_canonical(name)
        assert surface in {"daily", "extended"}, f"{name} surface={surface}"


def test_ts_transition_count_detects_rising_edge() -> None:
    """ts_transition_count 应正确识别状态切换（0→1 或 1→0）。"""
    idx = pd.date_range("2024-01-01", periods=10)
    # 状态序列: 0 0 1 1 0 1 0 0 1 1
    # Transitions: idx 2 (0→1), idx 4 (1→0), idx 5 (0→1), idx 6 (1→0), idx 8 (0→1) = 5 total
    condition = pd.DataFrame([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], index=idx, columns=["A"])
    op = OperatorRegistry.get("ts_transition_count")
    out = op.calculate(condition, window=10, missing_policy="break")
    # 最后一行窗口内应有 5 次状态切换
    assert out.iloc[-1, 0] == 5.0


def test_ts_time_since_change_counts_bars_correctly() -> None:
    """ts_time_since_change 应正确计算距上次状态翻转的行数。"""
    idx = pd.date_range("2024-01-01", periods=8)
    # 状态序列: 0 0 1 1 1 0 0 1 → transitions at idx 2 (0→1), 5 (1→0), 7 (0→1)
    condition = pd.DataFrame([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0], index=idx, columns=["A"])
    op = OperatorRegistry.get("ts_time_since_change")
    out = op.calculate(condition, max_lookback=None, missing_policy="break", initial_semantics="since_transition")
    # 最后一行 (idx 7) 距上次切换 (idx 7 本身, 0→1) 应为 0
    assert out.iloc[-1, 0] == 0.0
    # idx 6 距上次切换 (idx 5, 1→0) 应为 1
    assert out.iloc[6, 0] == 1.0
    # idx 4 距上次切换 (idx 2, 0→1) 应为 2
    assert out.iloc[4, 0] == 2.0


def test_ts_event_spacing_mean_cv() -> None:
    """ts_event_spacing_mean 和 ts_event_spacing_cv 应正确计算事件间隔统计。"""
    idx = pd.date_range("2024-01-01", periods=12)
    # 事件位置: idx 1, 4, 7, 10 → 间隔 3, 3, 3 → mean=3.0, cv=0.0
    condition = pd.DataFrame([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], index=idx, columns=["A"])

    op_mean = OperatorRegistry.get("ts_event_spacing_mean")
    out_mean = op_mean.calculate(condition, window=12, min_events=2)
    # 最后一行窗口内事件间隔均值应为 3.0
    assert out_mean.iloc[-1, 0] == pytest.approx(3.0)

    op_cv = OperatorRegistry.get("ts_event_spacing_cv")
    out_cv = op_cv.calculate(condition, window=12, min_events=3)
    # 间隔完全一致 (3, 3, 3)，CV 应为 0.0
    assert out_cv.iloc[-1, 0] == pytest.approx(0.0, abs=1e-9)
