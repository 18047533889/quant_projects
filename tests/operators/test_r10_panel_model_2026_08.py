# -*- coding: utf-8 -*-
"""Review-10 PCA / panel-model fixes (R10-P0-004..007, 009, 029, 030).

* R10-P0-004 — ``industry_rolling_pca_loading`` crashed with ``TypeError`` the
  moment any industry had >= 4 members (the loading kernel is now stateless and
  takes 3 args).
* R10-P0-005 — PCA loading is STATELESS: the same window gives the same sign
  whether it is reached in a full run or a chunked run (no hidden previous-
  loading sign state).
* R10-P0-006 — one stock missing TODAY no longer blanks every other stock's PCA
  residual; its own residual is NaN, its peers stay finite.
* R10-P0-007 — the active set is coverage-gated (``min_obs = ceil(window*0.7)``),
  with ``active_breadth`` / ``median_coverage`` / ``min_coverage`` telemetry; a
  low-coverage stock is excluded, not mean-imputed into the fit.
* R10-P0-009 — panel-model integer knobs declare ``ParamSpec(dtype=int)`` so a
  fractional ``n_experts=3.9`` fails at the call boundary instead of silently
  ``int()``-truncating.
* R10-P0-029 — ``ts_feature_pca_reconstruction_error`` requires >= 2 feature
  panels (a 1-feature call was a legal-but-always-NaN dead end).
* R10-P0-030 — a degenerate (constant) market_state collapses the MoE expert
  bins and the cell fails closed instead of running fewer real experts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

load_all()


def _get(canon: str):
    op = OperatorRegistry.get(canon, backend="pandas_numpy")
    assert op is not None, f"missing {canon}"
    return op


def _panel(rows: int, cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(rng.standard_normal((rows, cols)), index=dates, columns=[f"C{i}" for i in range(cols)])


# ---------------------------------------------------------------------------
# R10-P0-004: industry loading no longer crashes on >= 4 members
# ---------------------------------------------------------------------------

def test_industry_pca_loading_four_member_industry_no_typeerror():
    rows, cols = 40, 8
    ret = _panel(rows, cols, seed=1)
    grp = pd.DataFrame(
        [["G"] * 4 + ["H"] * 4] * rows,
        index=ret.index, columns=ret.columns, dtype=object,
    )
    out = _get("industry_rolling_pca_loading").calculate(ret, grp, window=30, component=0)
    assert out.shape == ret.shape
    # some industry cells must be computed (the old code raised before writing)
    assert np.isfinite(out.to_numpy()).any()


# ---------------------------------------------------------------------------
# R10-P0-005: stateless — chunked == full for the same window
# ---------------------------------------------------------------------------

def test_pca_loading_stateless_chunk_equals_full():
    ret = _panel(200, 8, seed=2)
    W = 60
    full = _get("panel_rolling_pca_loading").calculate(ret, window=W, component=0)
    for t in (90, 130, 180):
        # a chunk whose last window of W rows exactly matches the full-run
        # window at t must reproduce full.iloc[t] bit-for-bit (no hidden state)
        chunk = _get("panel_rolling_pca_loading").calculate(
            ret.iloc[t - W: t + 1], window=W, component=0
        )
        np.testing.assert_array_equal(
            chunk.iloc[-1].to_numpy(), full.iloc[t].to_numpy(), err_msg=f"t={t}"
        )


def test_pca_loading_mid_series_start_matches_full():
    """Starting the run mid-series (no previous loading history) gives the same
    loading for a shared window — the exact hidden-state symptom the review
    described."""
    ret = _panel(240, 6, seed=3)
    W = 50
    full = _get("panel_rolling_pca_loading").calculate(ret, window=W, component=1)
    t = 150
    # simulate "computed from the middle": only rows [t-W, t] available
    mid = _get("panel_rolling_pca_loading").calculate(
        ret.iloc[t - W: t + 1], window=W, component=1
    )
    np.testing.assert_array_equal(mid.iloc[-1].to_numpy(), full.iloc[t].to_numpy())


# ---------------------------------------------------------------------------
# R10-P0-006: one missing stock must not poison the cross-section
# ---------------------------------------------------------------------------

def test_pca_resid_single_missing_stock_isolates():
    rng = np.random.default_rng(4)
    rows, cols = 80, 10
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    common = rng.standard_normal((rows, 1))
    ret = common @ rng.standard_normal((1, cols)) * 0.5 + rng.standard_normal((rows, cols)) * 0.1
    ret = pd.DataFrame(ret, index=dates, columns=[f"C{i}" for i in range(cols)])

    ret2 = ret.copy()
    ret2.iloc[-1, 3] = np.nan  # stock C3 suspended today
    base = _get("panel_rolling_pca_resid").calculate(ret, window=50, n_components=2)
    shocked = _get("panel_rolling_pca_resid").calculate(ret2, window=50, n_components=2)

    # the suspended stock's own residual is NaN today
    assert np.isnan(shocked.iloc[-1, 3])
    # every OTHER stock's residual stays finite today (no contamination)
    for j in range(cols):
        if j == 3:
            continue
        assert np.isfinite(shocked.iloc[-1, j]), f"column C{j} poisoned"
    # rows strictly before the shock are unchanged
    pd.testing.assert_frame_equal(base.iloc[:-1], shocked.iloc[:-1], check_dtype=False)


# ---------------------------------------------------------------------------
# R10-P0-007: coverage-gated active set + telemetry
# ---------------------------------------------------------------------------

def test_pca_svd_excludes_low_coverage_stock_and_reports_telemetry():
    from cleaned_operators.cross_section.panel_model import _pca_svd

    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 4))
    X[2:, 3] = np.nan  # stock 3 has only 2 finite rows in the 60-row window
    pca = _pca_svd(X, 2)
    assert pca is not None
    assert pca["active"][3] == False, "low-coverage stock must be INACTIVE"
    assert pca["active_breadth"] == 3
    # coverage percentiles are over the ACTIVE sub-space, which requires >= 0.7
    assert pca["min_coverage"] >= 0.7 - 1e-9
    assert pca["median_coverage"] >= 0.7 - 1e-9


def test_pca_resid_low_coverage_stock_stays_nan_not_imputed():
    rng = np.random.default_rng(6)
    rows, cols = 60, 5
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    ret = pd.DataFrame(rng.standard_normal((rows, cols)), index=dates, columns=[f"C{i}" for i in range(cols)])
    ret.iloc[2:, 4] = np.nan  # stock C4 present for only 2 rows
    out = _get("panel_rolling_pca_resid").calculate(ret, window=50, n_components=2)
    # C4 never becomes active -> its residual is NaN everywhere (fail-closed)
    assert out["C4"].notna().sum() == 0
    # other columns are computed once the window matures
    assert out["C0"].notna().sum() > 0


# ---------------------------------------------------------------------------
# R10-P0-008: resid_vol / resid_momentum declare the nested 2W history
# ---------------------------------------------------------------------------

def test_pca_resid_vol_history_is_double_window():
    from runtime.execution_contract import history_requirement

    for canon in ("panel_rolling_pca_resid_vol", "panel_rolling_pca_resid_momentum"):
        req = history_requirement(canon, {"window": 120})
        assert req.kind == "finite"
        # rolling PCA (W) -> rolling std/sum (W): ~2*(W-1) mature rows
        assert req.rows == 2 * (120 - 1)
        # the generic one-window parser would have said 119 — must differ
        assert req.rows != 119


# ---------------------------------------------------------------------------
# R10-P0-009: integer knobs reject fractional values at the call boundary
# ---------------------------------------------------------------------------

def test_panel_model_fractional_integer_knob_rejected():
    y = _panel(60, 6, seed=7)
    x1 = _panel(60, 6, seed=8)
    ms = _panel(60, 6, seed=9)
    with pytest.raises(Exception, match="n_experts"):
        _get("panel_mixture_of_experts_score").calculate(
            y, x1, None, None, None, ms, window=40, n_experts=3.9, label_horizon=1
        )


def test_panel_model_fractional_window_rejected():
    y = _panel(60, 6, seed=10)
    x1 = _panel(60, 6, seed=11)
    with pytest.raises(Exception, match="window"):
        _get("panel_rolling_pcr_forecast").calculate(
            y, x1, None, None, None, window=59.5, n_components=2, label_horizon=1
        )


# ---------------------------------------------------------------------------
# R10-P0-029: autoencoder reconstruction needs >= 2 features
# ---------------------------------------------------------------------------

def test_pca_reconstruction_error_single_feature_rejected():
    x1 = _panel(60, 6, seed=12)
    with pytest.raises(ValueError, match="at least 2 feature"):
        _get("ts_feature_pca_reconstruction_error").calculate(x1, window=40, n_components=1)


# ---------------------------------------------------------------------------
# R10-P0-030: degenerate market_state collapses expert bins -> fail closed
# ---------------------------------------------------------------------------

def test_moe_constant_market_state_fails_closed():
    y = _panel(80, 6, seed=13)
    x1 = _panel(80, 6, seed=14)
    x2 = _panel(80, 6, seed=15)
    ms = pd.DataFrame(1.0, index=y.index, columns=y.columns)
    out = _get("panel_mixture_of_experts_score").calculate(
        y, x1, x2, None, None, ms, window=40, n_experts=3, label_horizon=1
    )
    # quantile edges collapse to one value -> effective experts < requested
    assert out.to_numpy().shape == y.shape
    assert not np.isfinite(out.to_numpy()).any()
