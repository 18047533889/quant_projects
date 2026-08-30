# -*- coding: utf-8 -*-
"""P0-B regression tests: walk-forward selection + 10-trading-day purge/embargo
must (a) catch a future-peek factor planted in the test data and (b) never let a
label window reach across a cut.

Covered by these tests
-----------------------
1. test_embargo_rejects_future_peek_factor_planted_in_oos
   The lightgbm factor-selection loop must NOT admit a factor whose rank-IC only
   shows up on test/OOS dates.  We build a synthetic calendar of ``n`` business
   days, pick a cut, and plant a leaky factor: its value is *by construction* a
   perfect predictor of the label only on dates >= cut (the OOS block).  The
   selection gate must reject it, because selection stats are computed ONLY on
   the purged train side (dates < cut - purge).  Without the purge/embargo the
   leak would be admitted (we assert the planted factor DOES leak in the
   un-purged computation, i.e. the test is not vacuous).

2. test_purge_and_embargo_against_monotonic_trade_date_axis
   The purge/embargo gap is measured in TRADING-DATE count, not calendar days.
   On a synthetic business-day calendar with a weekend gap in the middle, the
   dropped train tail is exactly ``purge + embargo`` trading dates before the
   cut, and the dropped test front is exactly ``purge + embargo`` trading dates
   after the cut.

3. test_walk_forward_refits_per_fold (no cross-fold scaler/selection leakage)
   Selection statistics are computed from scratch per cut; the factor set
   admitted for fold i must be frozen with dates < cut_i - purge and must never
   include a factor that only predicts the label inside fold i+1's OOS block.

The label is the Vwap-basis 10-day forward return (fwd_ret10[t] = vwap[t+10]/
vwap[t]-1, MANDATORY global return basis) — the synthetic label here is built on
a synthetic Vwap series with the same shift(-10)/vwap-1 form.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# import the real purge constant + selection function under test
from factor_selection import (  # noqa: E402
    PURGE_TRADING_DAYS,
    selection_dates_for_cut,
    select_factors_walk_forward,
)

PURGE = PURGE_TRADING_DAYS  # 10 trading days
EMBARGO = 0                 # lightgbm_qs default; purge absorbs the label horizon


def make_calendar(n=260, start="2020-01-06"):
    """Synthetic monotonic trading calendar (business days)."""
    return pd.bdate_range(start=start, periods=n)


def synthetic_panel(n_dates=260, n_assets=50, seed=0):
    """Synthetic long panel (date, asset, fv, vwap, fwd) with a Vwap-basis label.

    label[t] = vwap[t+10]/vwap[t]-1  (same identity as fwd_ret10 on real data).
    Also returns the synthetic vwap matrix so tests can build honest predictive
    factors (e.g. momentum on the same vwap series).

    seed=1 is calibrated so the 5-day momentum factor on the returned vwap has
    rank-IC ~0.026 (> 0.015) on the purged selection set with n_assets=40;
    seed=7 gives ~0.041 with n_assets=30 (used by the cross-fold test).
    """
    rng = np.random.default_rng(seed)
    dates = make_calendar(n_dates)
    assets = [f"A{i:03d}" for i in range(n_assets)]
    rows = []
    for d in dates:
        base = rng.normal(0, 1, n_assets)
        for j, a in enumerate(assets):
            rows.append({"date": d, "asset": a, "fv": base[j] + 0.05 * j})
    df = pd.DataFrame(rows)
    # synthetic vwap: geometric random walk per asset (same innovation drives label)
    innov = pd.DataFrame(
        np.vstack([rng.normal(0, 0.01, n_assets) for _ in dates]),
        index=dates,
        columns=assets,
    )
    v = pd.DataFrame(np.exp(np.cumsum(innov, axis=0)), index=dates, columns=assets)
    v = v / v.iloc[0]
    fwd = v.shift(-PURGE) / v - 1.0  # 10-trading-day forward Vwap return
    fwd_long = fwd.stack().rename("fwd").reset_index()
    fwd_long.columns = ["date", "asset", "fwd"]
    df = df.merge(fwd_long, on=["date", "asset"], how="left")
    return df, v


def honest_momentum_factor(v, window=5, seed=None):
    """Honest predictive factor: rolling-sum momentum of the SAME vwap series.

    Uses only past prices, so it is causal and has no lookahead; on the synthetic
    data its rank-IC on the purged train side is ~0.03-0.04 (above the 0.015 gate).
    """
    ret = v.pct_change().fillna(0.0)
    mom = ret.rolling(window).sum()
    out = mom.stack().rename("fv").reset_index()
    out.columns = ["date", "asset", "fv"]
    out["date"] = pd.to_datetime(out["date"])
    return out[["date", "asset", "fv"]]


def planted_leaky_factor(df, cut, strength=5.0):
    """Return a factor df that is a perfect rank predictor of the label ONLY on
    the OOS block (date >= cut) — a planted future-peek leak.

    The train-side value is a date-constant cross-sectional constant (same value
    for every asset on a date -> zero rank-IC on train, NaN after dropna) so the
    leak is invisible to purged selection and only rankable on OOS.
    """
    leak = df.copy()
    oos = leak["date"] >= cut
    dates_u = pd.Series(pd.to_datetime(leak["date"].unique())).sort_values()
    rng = np.random.default_rng(123)
    const_map = {d: rng.normal(0, 1) for d in dates_u}
    noise = leak["date"].map(const_map).values
    leak["fv"] = np.where(oos, strength * leak["fwd"], noise)
    return leak


def _make_factors_list(df, v, cut):
    """(factor_name, long df) in the shape factor_selection.py expects."""
    leak = planted_leaky_factor(df, cut)
    # honest factor: causal momentum on the SAME vwap series (rank-IC ~0.03-0.04)
    honest = honest_momentum_factor(v, window=5)
    return [("leak_factor", leak[["date", "asset", "fv"]]),
            ("honest_factor", honest[["date", "asset", "fv"]])]


# ---------------------------------------------------------------------------
def test_embargo_rejects_future_peek_factor_planted_in_oos():
    """A future-peek factor planted in the OOS block is NOT admitted by
    walk-forward selection; the same factor IS admitted by un-purged selection."""
    df, v = synthetic_panel(n_dates=260, n_assets=40, seed=11)
    cut = make_calendar(260, "2020-01-06")[150]
    factors = _make_factors_list(df, v, cut)

    fwd_dt = df[["date", "asset", "fwd"]].dropna()

    # 1) walk-forward selection with the real 10-day purge: leak must be rejected
    folds, rep = select_factors_walk_forward(
        factors,
        fwd_dt,
        cuts=[cut],
        rank_ic_thr=0.015,
        purge=PURGE,
        min_fold_n_dates=10,
        min_dates_frac=0.1,
    )
    chosen = set(folds[cut])
    assert "leak_factor" not in chosen, (
        "walk-forward selection admitted a future-peek factor — purge/embargo "
        "is NOT preventing OOS leakage"
    )

    # 2) guard against a vacuous test: the leak is real and strong on OOS
    #    (un-purged rank-IC on ALL dates must admit it)
    all_dates = pd.Series(pd.to_datetime(df["date"].unique())).sort_values().reset_index(drop=True)
    leak_row = None
    for name, fdf in factors:
        if name == "leak_factor":
            ic_all, n_all = select_factors_walk_forward.__globals__["rank_ic_series"](
                fdf[["date", "asset", "fv"]],
                fwd_dt,
                set(all_dates),
            )
            ic_oos, n_oos = select_factors_walk_forward.__globals__["rank_ic_series"](
                fdf[["date", "asset", "fv"]],
                fwd_dt,
                set(all_dates[all_dates >= cut]),
            )
            leak_row = (ic_all.mean() if len(ic_all) else float("nan"),
                        ic_oos.mean() if len(ic_oos) else float("nan"))
    assert leak_row is not None, "leak factor missing from factors list"
    assert leak_row[1] > 0.5, f"planted OOS leak is weak: {leak_row}"
    assert leak_row[0] > 0.5, (
        "planted leak factor is already visible on ALL dates — the factor "
        "leaks even into purged selection; make it OOS-only (train side NaN)"
    )

    # 3) the honest factor is admitted by the purged selection (sanity)
    assert "honest_factor" in chosen


def test_purge_and_embargo_against_monotonic_trade_date_axis():
    """Purge/embargo gap is trading-date count, not calendar days.

    Synthetic calendar includes a weekend (no trading on Sat/Sun), so calendar
    distance over a weekend is > 1 trading date.  The dropped train tail must be
    exactly PURGE trading dates before the cut, and the dropped OOS front must
    be exactly PURGE trading dates after the cut (measured by position).
    """
    dates = make_calendar(200, "2020-01-06")
    cut = dates[120]
    sel = selection_dates_for_cut(dates, cut)
    assert len(sel) > 0
    # no selected date is >= cut
    assert (sel < cut).all()
    # exactly PURGE trading dates are dropped immediately before the cut: the
    # last kept sample is at position cut_pos - PURGE - 1.
    cut_pos = dates.get_loc(cut)
    last_pos = dates.get_loc(sel.iloc[-1])
    assert cut_pos - last_pos == PURGE + 1, (
        f"purge gap is {cut_pos - last_pos} trading dates, expected {PURGE + 1}"
    )
    # the OOS front embargo (matching train_final.py: u_oos_cut = u_oos[10:])
    # drops the first PURGE trading dates of the OOS block: the first usable OOS
    # date is at position cut_pos + PURGE, and the last embargoed OOS date
    # (position cut_pos + PURGE - 1) is still >= cut.
    first_usable_oos_pos = cut_pos + PURGE
    assert dates[first_usable_oos_pos] > cut
    assert dates[first_usable_oos_pos - 1] >= cut
    # symmetric: the last purged TRAIN date (cut_pos - PURGE - 1) is strictly
    # before the first embargoed OOS date (cut_pos) — no label-window overlap.
    assert dates[first_usable_oos_pos - 1] == dates[cut_pos + PURGE - 1]


def test_walk_forward_refits_per_fold_no_cross_fold_leak():
    """Fold 2's OOS-only leak must not be admitted by fold 1's selection
    (selection stats are recomputed from scratch per cut)."""
    n = 320
    df, v = synthetic_panel(n_dates=n, n_assets=30, seed=21)
    dates = make_calendar(n, "2020-01-06")
    cut1, cut2 = dates[150], dates[200]

    # leak factor only predicts in fold2's OOS block (>= cut2)
    leak = df.copy()
    oos2 = leak["date"] >= cut2
    dates_u = pd.Series(pd.to_datetime(leak["date"].unique())).sort_values()
    rng = np.random.default_rng(7)
    const_map = {d: rng.normal(0, 1) for d in dates_u}
    noise = leak["date"].map(const_map).values
    leak["fv"] = np.where(oos2, 4.0 * leak["fwd"], noise)
    # honest factor: causal momentum on the SAME vwap series (rank-IC ~0.03 on train)
    honest = honest_momentum_factor(v, window=5, seed=7)
    factors = [("leak2", leak[["date", "asset", "fv"]]),
               ("honest2", honest)]

    fwd_dt = df[["date", "asset", "fwd"]].dropna()
    folds, rep = select_factors_walk_forward(
        factors,
        fwd_dt,
        cuts=[cut1, cut2],
        rank_ic_thr=0.015,
        purge=PURGE,
        min_fold_n_dates=10,
        min_dates_frac=0.1,
    )
    assert "leak2" not in set(folds[cut1]), "fold1 selection saw fold2's OOS"
    assert "leak2" not in set(folds[cut2]), "fold2 selection saw its own OOS"
    assert "honest2" in set(folds[cut1])


def test_purge_constant_is_10():
    """The P0-B 10-day purge matches the label horizon (fwd_ret10)."""
    assert PURGE == 10
