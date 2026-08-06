# -*- coding: utf-8 -*-
"""Tests for the 2026-08 P2 model-type regression operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

P2_CANONICALS = frozenset(
    """ts_huber_regression_resid ts_ridge_regression_resid
    ts_quantile_regression_slope ts_ar_coefficient ts_variance_ratio
    ts_cusum_break_score ts_level_shift_score ts_vol_shift_score
    """.split()
)


def _panel(n: int = 80, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, 2)), index=idx, columns=["A", "B"])


@pytest.mark.parametrize("name", sorted(P2_CANONICALS))
def test_p2_operator_registered_and_extended(name: str) -> None:
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None
    assert classify_canonical(name) in {"daily", "extended"}


def test_regression_resid_is_approximately_zero_on_perfect_line() -> None:
    x = _panel(seed=1)
    y = 2.0 * x + 0.5
    op = OperatorRegistry.get("ts_huber_regression_resid")
    out = op.calculate(y, x, window=60, min_periods=6)
    assert out.iloc[-1, 0] == pytest.approx(0.0, abs=1e-6)
    # Ridge 有正则化偏置：极小 alpha 时应接近 0
    ridge = OperatorRegistry.get("ts_ridge_regression_resid")
    out_r = ridge.calculate(y, x, window=60, alpha=1e-6, min_periods=6)
    assert out_r.iloc[-1, 0] == pytest.approx(0.0, abs=1e-4)


def test_quantile_regression_slope_matches_ols_on_symmetric_data() -> None:
    x = _panel(seed=2)
    y = 1.5 * x + 0.3 + np.random.default_rng(3).standard_normal(x.shape) * 0.01
    op = OperatorRegistry.get("ts_quantile_regression_slope")
    out = op.calculate(y, x, window=60, q=0.5, min_periods=6)
    # q=0.5（中位数回归）应接近 OLS 斜率 1.5
    assert out.iloc[-1, 0] == pytest.approx(1.5, abs=0.2)


def test_ar_coefficient_recovers_ar1_rho() -> None:
    rng = np.random.default_rng(0)
    n = 200
    rho = 0.7
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.standard_normal()
    idx = pd.date_range("2024-01-01", periods=n)
    x = pd.DataFrame(series, index=idx, columns=["A"])
    out = OperatorRegistry.get("ts_ar_coefficient").calculate(x, window=120, lag=1, min_periods=10)
    assert out["A"].iloc[-1] == pytest.approx(rho, abs=0.15)


def test_variance_ratio_is_zero_for_random_walk() -> None:
    rng = np.random.default_rng(0)
    n = 400
    walk = np.cumsum(rng.standard_normal(n))
    idx = pd.date_range("2024-01-01", periods=n)
    x = pd.DataFrame(walk, index=idx, columns=["A"])
    out = OperatorRegistry.get("ts_variance_ratio").calculate(x, window=200, q=10, min_periods=20)
    # 随机游走方差比接近 0
    assert abs(out["A"].iloc[-1]) < 0.3


def test_level_shift_score_detects_mean_shift() -> None:
    rng = np.random.default_rng(0)
    n = 100
    series = rng.standard_normal(n)
    series[60:] += 1.5  # 后半段水平位移
    idx = pd.date_range("2024-01-01", periods=n)
    x = pd.DataFrame(series, index=idx, columns=["A"])
    out = OperatorRegistry.get("ts_level_shift_score").calculate(x, window=80, min_periods=10)
    # 窗口覆盖位移点后，水平位移得分应显著为正
    assert out["A"].iloc[-1] > 0.5


def test_p2_operators_preserve_shape_and_determinism() -> None:
    x = _panel()
    y = 1.5 * x + np.random.default_rng(1).standard_normal(x.shape) * 0.01
    calls = {
        "ts_huber_regression_resid": [y, x],
        "ts_ridge_regression_resid": [y, x],
        "ts_quantile_regression_slope": [y, x],
        "ts_ar_coefficient": [x],
        "ts_variance_ratio": [x],
        "ts_cusum_break_score": [x],
        "ts_level_shift_score": [x],
        "ts_vol_shift_score": [x],
    }
    for name, args in calls.items():
        op = OperatorRegistry.get(name)
        first = op.calculate(*args)
        second = op.calculate(*args)
        assert first.shape == x.shape, f"{name}: shape changed"
        pd.testing.assert_frame_equal(first, second, check_dtype=False)
