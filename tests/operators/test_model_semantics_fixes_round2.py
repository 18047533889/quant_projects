# -*- coding: utf-8 -*-
"""Regression tests locking in the 2026-08 round-2 operator fixes.

Covers: true max-drawdown peak/trough via the running peak; ShareholderId-
matched turnover (rank swap with unchanged holdings must give zero churn);
scale-free VWAP path variants; honest tail-variation aliases; return-profile
operators that compute returns internally; robust cross-sectional LOF / MAD
Mahalanobis / residual percentile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _panel(n: int = 60, seed: int = 0, cols: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=[f"C{i}" for i in range(cols)])


# ---------------------------------------------------------------------------
# True max drawdown via running peak
# ---------------------------------------------------------------------------

def test_drawdown_running_peak_finds_hidden_drawdown() -> None:
    """[5, 10, 6, 11] has a real max drawdown 10 -> 6 (-0.4) whose peak is not
    the global high (the final 11).  The global-peak-then-min-after approach
    reports NaN; the running-peak approach must find -0.4."""
    from cleaned_operators.intraday.vwap_path import _drawdown_depth, _drawdown_duration

    price = np.array([5.0, 10.0, 6.0, 11.0])
    assert _drawdown_depth(price) == pytest.approx(-0.4, abs=1e-9)
    assert _drawdown_duration(price) == pytest.approx(1.0, abs=1e-9)


def test_drawdown_matches_pandas_intra_max_drawdown_depth() -> None:
    """intra_drawdown_depth must agree with intra_max_drawdown on random paths."""
    from cleaned_operators.intraday.vwap_path import _drawdown_depth, _max_drawdown

    rng = np.random.default_rng(0)
    for _ in range(20):
        path = np.cumsum(rng.standard_normal(30)) + 100.0
        d = _drawdown_depth(path)
        m = _max_drawdown(path, "down")
        if np.isfinite(d) and np.isfinite(m):
            assert d == pytest.approx(m, abs=1e-9)


# ---------------------------------------------------------------------------
# ShareholderId matched turnover
# ---------------------------------------------------------------------------

def _shareholder_rank_swap_panels():
    """Previous ranking X#1(5%),Y#2(3%); current Y#1(3%),X#2(5%) -- same
    holders, swapped ranks, unchanged holdings."""
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = ["A"]
    nan = pd.DataFrame(np.nan, index=idx, columns=cols)
    blank = pd.DataFrame([[""]] * 2, index=idx, columns=cols)
    s1 = pd.DataFrame([[0.05], [0.05]], index=idx, columns=cols)
    s2 = pd.DataFrame([[0.03], [0.03]], index=idx, columns=cols)
    sid1 = pd.DataFrame([["Y"], ["Y"]], index=idx, columns=cols)
    sid2 = pd.DataFrame([["X"], ["X"]], index=idx, columns=cols)
    p1 = pd.DataFrame([[0.03], [0.03]], index=idx, columns=cols)
    p2 = pd.DataFrame([[0.05], [0.05]], index=idx, columns=cols)
    psid1 = pd.DataFrame([["X"], ["X"]], index=idx, columns=cols)
    psid2 = pd.DataFrame([["Y"], ["Y"]], index=idx, columns=cols)
    return ([s1, s2] + [nan] * 8 + [sid1, sid2] + [blank] * 8 +
            [p1, p2] + [nan] * 8 + [psid1, psid2] + [blank] * 8)


def test_holder_id_matched_churn_zero_on_rank_swap() -> None:
    args = _shareholder_rank_swap_panels()
    assert len(args) == 40
    # holder_weighted_churn was reworked to the ShareholderId-matched union pair
    # (2026-08), so it must also report zero on a pure rank swap (same holders,
    # same ratios).  It equals the dedicated holder_id_matched_churn.
    churn = OperatorRegistry.get("holder_weighted_churn").calculate(*args)
    matched = OperatorRegistry.get("holder_id_matched_churn").calculate(*args)
    assert abs(churn.iloc[-1, 0]) < 1e-12
    assert abs(matched.iloc[-1, 0]) < 1e-12


def test_holder_id_matched_entry_exit_overlap() -> None:
    args = _shareholder_rank_swap_panels()
    assert abs(OperatorRegistry.get("holder_id_matched_entry_share").calculate(*args).iloc[-1, 0]) < 1e-12
    assert abs(OperatorRegistry.get("holder_id_matched_exit_share").calculate(*args).iloc[-1, 0]) < 1e-12
    assert OperatorRegistry.get("holder_id_overlap_ratio").calculate(*args).iloc[-1, 0] == pytest.approx(1.0)
    # replacement case: current adds Z 4% replacing Y
    args2 = list(args)
    args2[1] = pd.DataFrame([[0.04], [0.04]], index=args2[1].index, columns=args2[1].columns)
    args2[11] = pd.DataFrame([["Z"], ["Z"]], index=args2[11].index, columns=args2[11].columns)
    entry = OperatorRegistry.get("holder_id_matched_entry_share").calculate(*args2).iloc[-1, 0]
    assert entry == pytest.approx(0.04, abs=1e-9)
    assert OperatorRegistry.get("holder_id_overlap_ratio").calculate(*args2).iloc[-1, 0] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Scale-free VWAP path variants
# ---------------------------------------------------------------------------

def test_vwap_path_pct_is_scale_invariant() -> None:
    """Two stocks whose cum-VWAP paths are identical relative to their own
    first price but at very different absolute levels must have the same
    ``_pct`` slope."""
    from cleaned_operators.intraday.vwap_path import _vwap_path_pct_common

    # intraday minute series for two days so _cum_vwap has a real day
    amt = np.linspace(1, 2, 10)
    vol = np.linspace(10, 20, 10)
    cheap = np.linspace(5.0, 6.0, 10)
    expensive = cheap * 100.0
    s_cheap = _vwap_path_pct_common(cheap, amt, vol, 1, 1)
    s_expensive = _vwap_path_pct_common(expensive, amt, vol, 1, 1)
    assert s_cheap == pytest.approx(s_expensive, abs=1e-9)
    assert np.isfinite(s_cheap)


# ---------------------------------------------------------------------------
# Honest tail-variation aliases
# ---------------------------------------------------------------------------

def test_tail_variation_aliases_match_historic_threshold_ops() -> None:
    dates = pd.date_range("2024-01-03 09:31:00", periods=60, freq="min")
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal(60) * 0.01, axis=0)) * 100, index=dates, columns=["A"]
    )
    pos_new = OperatorRegistry.get("intra_positive_tail_variation").calculate(close, threshold_scale=2.0)
    pos_old = OperatorRegistry.get("intra_positive_jump_variation").calculate(close, threshold_scale=2.0)
    pd.testing.assert_frame_equal(pos_new, pos_old, check_dtype=False)
    cnt_new = OperatorRegistry.get("intra_tail_event_count").calculate(close, threshold_scale=2.0)
    cnt_old = OperatorRegistry.get("intra_jump_count").calculate(close, threshold_scale=2.0)
    pd.testing.assert_frame_equal(cnt_new, cnt_old, check_dtype=False)


# ---------------------------------------------------------------------------
# Return-profile operators compute returns internally
# ---------------------------------------------------------------------------

def test_signed_return_profile_cosine_accepts_close() -> None:
    # needs >= 3 days so the rolling historical profile has a full window
    dates = pd.date_range("2024-01-03", periods=3, freq="D")
    timestamps = []
    for d in dates:
        for m in list(range(571, 691))[:60]:
            timestamps.append(d + pd.Timedelta(minutes=m))
    rng = np.random.default_rng(2)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((len(timestamps), 1)) * 0.001, axis=0)) * 100,
        index=pd.DatetimeIndex(timestamps), columns=["A"],
    )
    out = OperatorRegistry.get("intra_signed_return_profile_cosine").calculate(close, window=2)
    assert out.shape[1] == 1
    assert np.isfinite(out.to_numpy()).any()


# ---------------------------------------------------------------------------
# Robust cross-sectional operators
# ---------------------------------------------------------------------------

def test_lof_detects_outlier() -> None:
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.standard_normal((50, 20)), index=dates, columns=[f"C{i}" for i in range(20)])
    x["C0"] = 50.0
    lof = OperatorRegistry.get("cs_actual_lof_score").calculate(x, k=8)
    row = lof.iloc[-1]
    assert row["C0"] > row.drop("C0").median()


def test_mad_mahalanobis_and_residual_percentile() -> None:
    x = _panel(seed=4, cols=12)
    d = OperatorRegistry.get("cs_robust_mahalanobis_mad").calculate(x)
    assert d.shape == x.shape and np.isfinite(d.to_numpy()).any()
    p = OperatorRegistry.get("cs_residual_percentile").calculate(x)
    flat = p.to_numpy()[np.isfinite(p.to_numpy())]
    assert (flat >= 0).all() and (flat <= 1).all()


def test_knn_distance_blockwise_matches_reference() -> None:
    """The blockwise KNN must agree with a direct full-distance computation."""
    from cleaned_operators.cross_section.robust_cs import _knn_blockwise

    rng = np.random.default_rng(5)
    Xn = rng.standard_normal((37, 4))
    out = _knn_blockwise(Xn, 5)
    d2 = np.sum((Xn[:, None, :] - Xn[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    part = np.partition(d2, kth=4, axis=1)[:, :5]
    ref = np.sqrt(part).mean(axis=1)
    assert np.allclose(out, ref, atol=1e-9)
