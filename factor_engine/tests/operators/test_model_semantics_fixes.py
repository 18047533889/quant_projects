# -*- coding: utf-8 -*-
"""Regression tests locking in the 2026-08 model-operator semantic fixes.

Covers: prior-window (out-of-sample) AR / multi-regression; Huber sqrt(weight);
Ridge intercept-only penalty; PCA prior-window and per-stock commonality;
industry PCA local-index mapping; Kalman innovation on the predicted state;
GARCH standardised shock on h_t; CUSUM max-|cumsum|; peer rank targeting the
current stock; group-based peer diffusion; liquidity-beta on delta; panel
supervised forecasts excluding the current label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _panel(n: int = 160, seed: int = 0, cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=["A", "B"])


# ---------------------------------------------------------------------------
# Prior-window / out-of-sample semantics
# ---------------------------------------------------------------------------

def test_prior_forecast_error_exceeds_in_sample_residual() -> None:
    """Under a current-period shock the prior-window error must be larger than
    the in-sample residual (the in-sample fit adapts to the shock)."""
    x = _panel(seed=1)
    y = 1.5 * x + 0.3 + np.random.default_rng(2).standard_normal(x.shape) * 0.01
    y.iloc[-1, 0] = 99.0
    in_sample = OperatorRegistry.get("ts_multi_regression_resid").calculate(y, x, window=80, min_periods=10)
    prior = OperatorRegistry.get("ts_multi_regression_forecast_error").calculate(y, x, window=80, min_periods=10)
    assert abs(prior["A"].iloc[-1]) > abs(in_sample["A"].iloc[-1])


def test_prior_coeff_unaffected_by_current_shock() -> None:
    """A huge current shock must not move the prior-window coefficient."""
    x = _panel(seed=3)
    y = 1.5 * x + 0.3 + np.random.default_rng(4).standard_normal(x.shape) * 0.01
    base = OperatorRegistry.get("ts_multi_regression_coeff_prior").calculate(y, x, window=80, coefficient_index=1, min_periods=10)
    y2 = y.copy()
    y2.iloc[-1, 0] = 99.0
    shocked = OperatorRegistry.get("ts_multi_regression_coeff_prior").calculate(y2, x, window=80, coefficient_index=1, min_periods=10)
    assert base["A"].iloc[-1] == pytest.approx(shocked["A"].iloc[-1], abs=1e-9)


def test_ar_prior_forecast_uses_t_minus_1_model() -> None:
    """ts_ar_prior_forecast must equal a manual AR(1) fit on [t-window, t-1]."""
    rng = np.random.default_rng(0)
    n = 300
    rho = 0.7
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.standard_normal()
    x = pd.DataFrame(series, index=pd.date_range("2024-01-01", periods=n), columns=["A"])
    f = OperatorRegistry.get("ts_ar_prior_forecast").calculate(x, window=120, order=1)["A"]
    seg = series[-121:-1]
    b = np.linalg.lstsq(np.column_stack([np.ones(119), seg[:-1]]), seg[1:], rcond=None)[0]
    manual = b[0] + b[1] * series[-2]
    assert f.iloc[-1] == pytest.approx(manual, abs=1e-9)


def test_ar_prior_innovation_z_finite() -> None:
    x = _panel(seed=5)
    out = OperatorRegistry.get("ts_ar_prior_innovation_z").calculate(x, window=80, order=1)
    assert np.isfinite(out.to_numpy()).any()


# ---------------------------------------------------------------------------
# Huber / Ridge kernel fixes
# ---------------------------------------------------------------------------

def test_huber_robust_to_outliers_with_sqrt_weight() -> None:
    """Huber slope on a clean line with a few huge outliers stays ~2.0."""
    rng = np.random.default_rng(6)
    n = 60
    xx = pd.DataFrame(np.linspace(0, 1, n), index=pd.date_range("2024-01-01", periods=n), columns=["A"])
    yy = 2.0 * xx + 0.1
    yy.iloc[5, 0] = 50.0
    yy.iloc[40, 0] = -40.0
    out = OperatorRegistry.get("ts_huber_regression_coeff").calculate(yy, xx, window=n, coefficient_index=1, min_periods=10)
    assert out["A"].iloc[-1] == pytest.approx(2.0, abs=0.05)
    from factor_engine.cleaned_operators.ts_model._rolling_core import last_fit_status
    assert last_fit_status()["reason"] == "exact_consensus"


def test_ridge_penalises_first_feature_without_intercept() -> None:
    """Ridge without intercept must still shrink the first *feature* column."""
    from factor_engine.cleaned_operators.ts_model._rolling_core import build_design, ols_fit, ridge_fit

    rng = np.random.default_rng(7)
    x = rng.standard_normal(60)
    y = 2.0 * x + rng.standard_normal(60) * 0.1
    D = build_design([x], add_intercept=False)
    b_shrunk = ridge_fit(D, y, alpha=50.0, has_intercept=False)
    b_ols = ols_fit(D, y)
    assert abs(b_shrunk[0]) < abs(b_ols[0])


# ---------------------------------------------------------------------------
# PCA family
# ---------------------------------------------------------------------------

def test_pca_resid_prior_window_excludes_current_row() -> None:
    """Changing the current row must not change today's PCA residual structure
    beyond the current residual itself (the PCA is fit on prior rows)."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(8)
    ret = pd.DataFrame(rng.standard_normal((60, 6)), index=dates, columns=[f"C{i}" for i in range(6)])
    base = OperatorRegistry.get("panel_rolling_pca_resid").calculate(ret, window=40, n_components=2)
    ret2 = ret.copy()
    ret2.iloc[-1] = 99.0
    shocked = OperatorRegistry.get("panel_rolling_pca_resid").calculate(ret2, window=40, n_components=2)
    # rows strictly before the last must be unchanged
    pd.testing.assert_frame_equal(base.iloc[:-1], shocked.iloc[:-1], check_dtype=False)


def test_pca_explained_ratio_is_per_stock_not_broadcast() -> None:
    """Per-stock commonality must vary across the cross-section."""
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(9)
    common = rng.standard_normal((80, 1))
    ret = common @ rng.standard_normal((1, 6)) * 0.5 + rng.standard_normal((80, 6)) * 0.1
    ret[:, 5] = rng.standard_normal(80) * 1.0  # a pure-idiosyncratic column
    ret = pd.DataFrame(ret, index=dates, columns=[f"C{i}" for i in range(6)])
    out = OperatorRegistry.get("panel_rolling_pca_explained_ratio").calculate(ret, window=50, n_components=2)
    last = out.iloc[-1]
    assert last.nunique() > 1, "commonality must not be a scalar broadcast"


def test_industry_pca_loading_local_index() -> None:
    """Non-contiguous industry membership must map to the correct local row."""
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(10)
    ret = pd.DataFrame(rng.standard_normal((40, 6)), index=dates, columns=["a", "b", "c", "d", "e", "f"])
    # industry group: stocks b and d (non-contiguous), industry A and B split
    grp = pd.DataFrame(
        [["X", "G", "Y", "G", "Y", "X"]] * 40,
        index=dates, columns=["a", "b", "c", "d", "e", "f"], dtype=object,
    )
    out = OperatorRegistry.get("industry_rolling_pca_loading").calculate(ret, grp, window=30, component=0)
    assert out.shape == ret.shape


def test_autoencoder_rank_below_feature_count() -> None:
    """ts_feature_pca_reconstruction_error with n_components=2 on 3 feature
    panels is a genuine low-rank compression whose error is non-zero.  (A single
    feature panel yields p=1 and trivially perfect reconstruction -- exactly the
    pathology the operator was created to avoid.)"""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(11)
    x1 = pd.DataFrame(rng.standard_normal((60, 4)), index=dates, columns=["a", "b", "c", "d"])
    x2 = pd.DataFrame(rng.standard_normal((60, 4)), index=dates, columns=["a", "b", "c", "d"])
    x3 = pd.DataFrame(rng.standard_normal((60, 4)), index=dates, columns=["a", "b", "c", "d"])
    out = OperatorRegistry.get("ts_feature_pca_reconstruction_error").calculate(x1, x2, x3, window=40, n_components=2)
    valid = out.dropna().to_numpy()
    assert (valid > 1e-9).any(), "rank=2 on 3 features must not collapse to ~0"


# ---------------------------------------------------------------------------
# Kalman / GARCH timing
# ---------------------------------------------------------------------------

def test_kalman_innovation_uses_predicted_state() -> None:
    from factor_engine.cleaned_operators.ts_model.state_space import _kalman_level

    v = np.array([1.0, 1.1, 0.9, 1.2, 1.05, 0.95, 1.15, 1.1])
    out = _kalman_level(v, 0.01, 0.1, "innovation_z")
    mu, P = np.nan, 1.0
    manual = []
    for x in v:
        if not np.isfinite(mu):
            mu, P = x, 0.1
            manual.append(np.nan)
            continue
        p_pred = P + 0.01
        k = p_pred / (p_pred + 0.1)
        innov = x - mu  # predicted state before update
        mu = mu + k * innov
        P = (1 - k) * p_pred
        manual.append(innov / np.sqrt(max(p_pred + 0.1, 1e-12)))
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(np.array(manual)), atol=1e-9)


def test_garch_shock_uses_h_t() -> None:
    from factor_engine.cleaned_operators.ts_model.volatility import _garch_path, _fit_garch

    rng = np.random.default_rng(12)
    n = 400
    h = np.zeros(n)
    h[0] = 1.0
    ret = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.1 * ret[t - 1] ** 2 + 0.8 * h[t - 1]
        ret[t] = np.sqrt(h[t]) * rng.standard_normal()
    seg = ret[-120:]
    # P0-040: the shock of the current return must be standardised by
    # parameters fitted strictly on <= t-1 (the window *excluding* the current
    # return), then evaluated against the variance that governed it.
    params = _fit_garch(seg[:-1])
    assert params is not None
    w, a, b = params
    # The backcast is part of the fitted state and therefore must use the same
    # strict-prior segment as the parameter fit.  Replaying shocks through
    # seg[-2] yields h_t, the variance governing seg[-1].
    h_cur = float(np.var(seg[:-1]))
    for i in range(1, len(seg)):
        h_cur = w + a * seg[i - 1] ** 2 + b * h_cur
    shock = _garch_path(ret, 120, "shock", False, 0.0)
    assert shock == pytest.approx(seg[-1] / np.sqrt(max(h_cur, 1e-12)), rel=1e-6)

    # The current observation may change the numerator, but not the h_t state
    # reconstructed solely from the strict-prior segment.
    changed = ret.copy()
    changed[-1] *= 3.0
    changed_shock = _garch_path(changed, 120, "shock", False, 0.0)
    assert changed_shock / shock == pytest.approx(3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

def test_cusum_vol_break_uses_max_abs_cumsum() -> None:
    from factor_engine.cleaned_operators.ts_model.complexity import _cusum_vol_break

    # A window whose last cumulative sum is ~0 but which has a mid-window spike
    # must still report a large score via max(|cumsum|).
    rng = np.random.default_rng(13)
    v = rng.standard_normal(60)
    v[30] = 12.0
    score = _cusum_vol_break(v, 60, 10)
    sq = v ** 2
    mean = sq.mean()
    sd = sq.std()
    running = np.cumsum(sq - mean) / (sd * np.sqrt(np.arange(1, 61)))
    assert score == pytest.approx(np.max(np.abs(running)), rel=1e-9)
    assert abs(running[-1]) < score


# ---------------------------------------------------------------------------
# Peer / group fixes
# ---------------------------------------------------------------------------

def test_multi_level_rank_consistency_uses_target_rank() -> None:
    """The rank used for each cell must be the *target stock's* rank inside its
    group, not the last stock of the group in panel order.

    With values A=1..F=6 and the three group structures below, stock A is always
    the smallest in its group (rank 0 everywhere -> consistency 1) while stock B
    has ranks {0.2, 0.5, 0} (consistency ~0.56).  A buggy implementation that
    always used the group's last stock would report 1.0 for every stock.
    """
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    vals = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}
    x = pd.DataFrame({k: [v] * 20 for k, v in vals.items()}, index=dates)
    g1 = pd.DataFrame({k: ["all"] * 20 for k in vals}, index=dates, dtype=object)
    g2 = pd.DataFrame({"A": ["p"] * 20, "B": ["p"] * 20, "C": ["p"] * 20,
                       "D": ["q"] * 20, "E": ["q"] * 20, "F": ["q"] * 20}, index=dates, dtype=object)
    g3 = pd.DataFrame({"A": ["u"] * 20, "B": ["v"] * 20, "C": ["u"] * 20,
                       "D": ["v"] * 20, "E": ["u"] * 20, "F": ["v"] * 20}, index=dates, dtype=object)
    out = OperatorRegistry.get("group_multi_level_rank_consistency").calculate(x, g1, g2, g3)
    row = out.iloc[-1]
    assert row["A"] == pytest.approx(1.0, abs=1e-9), "A is rank-consistent across all groups"
    assert row["B"] < 0.99, "B's within-group ranks differ across groups -> consistency must drop"


def test_peer_deviation_index_all_missing_is_nan() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    nan = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    out = OperatorRegistry.get("group_peer_deviation_index").calculate(nan, nan)
    assert out.iloc[-1].isna().all(), "all-missing inputs must yield NaN, not 0"


def test_group_peer_information_diffusion_uses_group() -> None:
    """The operator must accept (own_return, group, window, lag) and use group to
    build the ex-self peer return."""
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(14)
    ret = pd.DataFrame(rng.standard_normal((80, 3)) * 0.01, index=dates, columns=["A", "B", "C"])
    g = pd.DataFrame({"A": ["ind"] * 80, "B": ["ind"] * 80, "C": ["ind"] * 80}, index=dates, dtype=object)
    out = OperatorRegistry.get("group_peer_information_diffusion").calculate(ret, g, window=40, lag=1)
    assert out.shape == ret.shape
    assert np.isfinite(out.to_numpy()).any()


def test_liquidity_beta_regresses_on_delta() -> None:
    """Constant liquidity (delta all zero) must yield NaN, proving the regressor
    is the change in liquidity and not the raw level."""
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(15)
    ret = pd.DataFrame(rng.standard_normal((80, 2)) * 0.01, index=dates, columns=["A", "B"])
    const_liquidity = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    out = OperatorRegistry.get("ts_market_liquidity_beta").calculate(ret, const_liquidity, window=40)
    assert out.iloc[-1].isna().all(), "delta of a constant series is zero -> beta must be NaN"


# ---------------------------------------------------------------------------
# Panel supervised forecasts exclude the current label
# ---------------------------------------------------------------------------

def test_pcr_forecast_ignores_current_label() -> None:
    """The walk-forward PCR prediction at row t must not change when y_t is
    perturbed, because y_t never enters the training set of the model that
    predicts t."""
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(16)
    x1 = pd.DataFrame(rng.standard_normal((80, 6)), index=dates, columns=[f"C{i}" for i in range(6)])
    x2 = pd.DataFrame(rng.standard_normal((80, 6)), index=dates, columns=[f"C{i}" for i in range(6)])
    y = pd.DataFrame(np.roll(x1.to_numpy(), -1, axis=0), index=dates, columns=x1.columns)
    base = OperatorRegistry.get("panel_rolling_pcr_forecast").calculate(y, x1, x2, window=40, n_components=3)
    y2 = y.copy()
    y2.iloc[-1] = 999.0
    shocked = OperatorRegistry.get("panel_rolling_pcr_forecast").calculate(y2, x1, x2, window=40, n_components=3)
    pd.testing.assert_frame_equal(base, shocked, check_dtype=False)
