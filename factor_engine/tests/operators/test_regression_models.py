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
    """ts_huber_regression_in_sample_resid ts_ridge_regression_in_sample_resid
    ts_quantile_regression_slope ts_ar_coefficient ts_variance_ratio_proxy
    ts_cumulative_deviation_score ts_level_shift_score ts_vol_shift_score
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


# ---------------------------------------------------------------------------
# Window maturity poison tests (2026-08-13 audit)
# ---------------------------------------------------------------------------


def test_predictive_regression_uses_exactly_w_prior_observations() -> None:
    """POISON TEST: Predictive regression must fit on exactly W prior obs [t-W, t-1].

    Perturbing a row at t-W-1 (just outside the declared window) should NOT change
    the output at t. Perturbing a row at t-W (the first row of the declared window)
    MUST change the output.
    """
    rng = np.random.default_rng(42)
    n = 100
    w = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x_base = pd.DataFrame(rng.standard_normal((n, 2)), index=idx, columns=["A", "B"])
    y_base = 2.0 * x_base + 0.5 + rng.standard_normal((n, 2)) * 0.1

    op = OperatorRegistry.get("ts_huber_regression_predictive_resid")
    base_out = op.calculate(y_base, x_base, window=w, min_periods=5)

    # For last row at index n-1=99, window=20: prior window is [79, 98]
    # Poison row at (n-1)-w-1 = 78 (outside the window)
    last_row = n - 1
    x_poison_outside = x_base.copy()
    y_poison_outside = y_base.copy()
    x_poison_outside.iloc[last_row - w - 1, :] = 1e10
    y_poison_outside.iloc[last_row - w - 1, :] = 1e10
    out_outside = op.calculate(y_poison_outside, x_poison_outside, window=w, min_periods=5)

    # Output at last row should NOT change (row at t-W-1 is outside the window)
    np.testing.assert_allclose(
        base_out.iloc[-1].to_numpy(),
        out_outside.iloc[-1].to_numpy(),
        rtol=1e-9,
        err_msg="Predictive regression changed when row at t-W-1 was poisoned (window too wide)",
    )

    # Poison row at (n-1)-w = 79 (first row of the window [79, 98])
    x_poison_inside = x_base.copy()
    y_poison_inside = y_base.copy()
    x_poison_inside.iloc[last_row - w, :] = 1e10
    y_poison_inside.iloc[last_row - w, :] = 1e10
    out_inside = op.calculate(y_poison_inside, x_poison_inside, window=w, min_periods=5)

    # Output at last row MUST change (row at t-W is inside the window)
    diff = np.abs(base_out.iloc[-1].to_numpy() - out_inside.iloc[-1].to_numpy())
    assert np.all(diff > 1e-3), f"Predictive regression did NOT change when row at t-W was poisoned (window too narrow): diff={diff}"


def test_predictive_regression_excludes_current_observation() -> None:
    """POISON TEST: Predictive fit must NOT see the current observation (x_t, y_t).

    Perturbing only the current row should NOT change the fitted coefficients
    (only the residual magnitude changes). We verify by checking that two different
    extreme perturbations of the current row yield residuals that differ by exactly
    the y-perturbation difference (same fitted β).
    """
    rng = np.random.default_rng(43)
    n = 100
    w = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x_base = pd.DataFrame(rng.standard_normal((n, 1)), index=idx, columns=["A"])
    y_base = 2.0 * x_base + 0.5 + rng.standard_normal((n, 1)) * 0.1

    op = OperatorRegistry.get("ts_ridge_regression_predictive_resid")

    # Perturb current row y[t] by +1000
    y_perturb_high = y_base.copy()
    y_perturb_high.iloc[-1, 0] += 1000.0
    out_high = op.calculate(y_perturb_high, x_base, window=w, alpha=0.1, min_periods=5)

    # Perturb current row y[t] by -1000
    y_perturb_low = y_base.copy()
    y_perturb_low.iloc[-1, 0] -= 1000.0
    out_low = op.calculate(y_perturb_low, x_base, window=w, alpha=0.1, min_periods=5)

    # The residuals should differ by exactly 2000 (same fitted β, only y_t changed)
    residual_diff = out_high.iloc[-1, 0] - out_low.iloc[-1, 0]
    assert residual_diff == pytest.approx(2000.0, rel=1e-6), (
        f"Predictive fit saw the current observation: residual diff {residual_diff} != 2000 "
        "(if β changed, the diff would not be exactly the y-perturbation difference)"
    )


def test_ar_coefficient_uses_exactly_w_pairs() -> None:
    """POISON TEST: AR(lag) must use exactly W pairs (W+lag observations).

    For window=W, lag=L, the segment must be [t-W-L+1, t], yielding W pairs after
    lagging. Perturbing t-W-L (outside) should NOT change output; perturbing
    t-W-L+1 (inside, the first lagged obs) MUST change output.
    """
    rng = np.random.default_rng(44)
    n = 100
    w = 20
    lag = 2
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x_base = pd.DataFrame(rng.standard_normal((n, 2)), index=idx, columns=["A", "B"])

    op = OperatorRegistry.get("ts_ar_coefficient")
    base_out = op.calculate(x_base, window=w, lag=lag, min_periods=5)

    # For last row at index n-1=99, window=20, lag=2: need segment [78, 99] = 22 rows → 20 pairs
    # Poison row at (n-1)-w-lag = 99-20-2 = 77 (outside the window)
    last_row = n - 1
    x_poison_outside = x_base.copy()
    x_poison_outside.iloc[last_row - w - lag, :] = 1e10
    out_outside = op.calculate(x_poison_outside, window=w, lag=lag, min_periods=5)

    np.testing.assert_allclose(
        base_out.iloc[-1].to_numpy(),
        out_outside.iloc[-1].to_numpy(),
        rtol=1e-9,
        err_msg="AR coefficient changed when row at t-W-lag was poisoned (window too wide)",
    )

    # Poison row at (n-1)-w-lag+1 = 78 (first row of the window [78, 99])
    x_poison_inside = x_base.copy()
    x_poison_inside.iloc[last_row - w - lag + 1, :] = 1e10
    out_inside = op.calculate(x_poison_inside, window=w, lag=lag, min_periods=5)

    diff = np.abs(base_out.iloc[-1].to_numpy() - out_inside.iloc[-1].to_numpy())
    assert np.all(diff > 1e-6), (
        f"AR coefficient did NOT change when row at t-W-lag+1 was poisoned (window too narrow): diff={diff}"
    )

    # Poison row 78 (t-W-lag+1 = 78, first row of the window [78, 99])
    x_poison_inside = x_base.copy()
    x_poison_inside.iloc[n - w - lag + 1, :] = 1e10
    out_inside = op.calculate(x_poison_inside, window=w, lag=lag, min_periods=5)

    diff = np.abs(base_out.iloc[-1].to_numpy() - out_inside.iloc[-1].to_numpy())
    assert np.all(diff > 1e-6), (
        f"AR coefficient did NOT change when row at t-W-lag+1 was poisoned (window too narrow): diff={diff}"
    )


def test_predictive_regression_manual_window_parity() -> None:
    """VERIFICATION TEST: Operator with window=W matches manual fit on exactly W prior obs."""
    rng = np.random.default_rng(45)
    n = 100
    w = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x = pd.DataFrame(rng.standard_normal((n, 1)), index=idx, columns=["A"])
    y = 2.0 * x + 0.5 + rng.standard_normal((n, 1)) * 0.1

    op = OperatorRegistry.get("ts_huber_regression_predictive_resid")
    op_out = op.calculate(y, x, window=w, min_periods=5)

    # Manual: fit on [n-w-1, n-2] (W prior observations), score at n-1
    y_fit = y.iloc[n - w - 1 : n - 1, 0].to_numpy()
    x_fit = x.iloc[n - w - 1 : n - 1, 0].to_numpy()
    valid = np.isfinite(y_fit) & np.isfinite(x_fit)

    # Huber fit (simplified, 0 iterations for reproducibility)
    design = np.column_stack([np.ones(valid.sum()), x_fit[valid]])
    beta, *_ = np.linalg.lstsq(design, y_fit[valid], rcond=None)

    # Residual at current row
    y_cur = y.iloc[-1, 0]
    x_cur = x.iloc[-1, 0]
    manual_resid = y_cur - (beta[0] + beta[1] * x_cur)

    # The operator output should match the manual computation
    assert op_out.iloc[-1, 0] == pytest.approx(manual_resid, abs=0.05), (
        f"Operator residual {op_out.iloc[-1, 0]} != manual {manual_resid} "
        f"(mismatch suggests operator is not using exactly W prior observations)"
    )
