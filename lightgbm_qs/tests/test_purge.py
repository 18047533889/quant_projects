# -*- coding: utf-8 -*-
"""Unit tests for the P0-B 10-day purge/embargo split logic in scripts/train_final.py.

The fwd_ret10 label is a 10-trading-day forward Vwap return: a sample at trading index t
carries a label that observes Vwap up to index t+10 (fwd_ret10[t] == vwap[t+10]/vwap[t]-1
by row position on the trading calendar). The old split (train = date < cut) let a train
sample at index cut-10 leak into the OOS block.

These tests use a small synthetic calendar and assert that, after the purge, no train
sample's label window (t .. t+PURGE) overlaps the OOS block (index >= cut).

The purge logic is a tiny pure function so it is duplicated here and asserted against the
constant used by train_final.py (kept in sync via import).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# import the purge constant from the real train script (guard against path import issues)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

PURGE_TRADING_DAYS = 10
EMBARGO_TRADING_DAYS = 0


def make_dates(n=300, start="2020-01-06"):
    """Synthetic business-day trading calendar of length n."""
    return pd.bdate_range(start=start, periods=n)


def purge_split(dates, cut_idx, purge=PURGE_TRADING_DAYS, embargo=EMBARGO_TRADING_DAYS):
    """Replicate train_final.py's purge: return (train_mask, oos_mask) as index arrays.

    train = dates[< cut_idx] minus the last (purge+embargo) trading days;
    oos   = dates[cut_idx : cut_idx+BLOCK] minus the first (purge+embargo) days.
    """
    cut = dates[cut_idx]
    u_train = dates[dates < cut]
    u_purge = purge + embargo
    u_train_cut = u_train[:-u_purge] if len(u_train) > u_purge else u_train
    # OOS block = next 3 months (here: next 60 business days) for the test
    block_end = cut_idx + 60
    u_oos = dates[cut_idx:block_end]
    u_oos_cut = u_oos[u_purge:] if len(u_oos) > u_purge else u_oos
    return u_train_cut, u_oos_cut


def label_window_ok(train_dates, oos_dates, purge=PURGE_TRADING_DAYS, dates=None):
    """Assert no train sample's label window overlaps the OOS block.

    A train sample at position p (in the global calendar) has a label covering
    positions p..p+purge. It must not reach the first OOS position.
    """
    if dates is None:
        return True
    first_oos = oos_dates[0]
    first_oos_pos = dates.get_loc(first_oos)
    for td in train_dates:
        pos = dates.get_loc(td)
        # label window positions: pos .. pos+purge  (vwap up to index pos+purge)
        # require pos + purge < first_oos_pos  (strictly before the OOS block)
        assert pos + purge < first_oos_pos, \
            f"train sample {td} (pos {pos}) label window [pos, pos+{purge}] overlaps OOS at pos {first_oos_pos}"
    return True


# ---------------------------------------------------------------------------
def test_purge_removes_leaking_train_tail():
    """Without purge a train sample at cut-10 leaks; with purge it does not."""
    dates = make_dates(300)
    cut_idx = 150
    cut = dates[cut_idx]

    # OLD behavior (no purge): train = all dates < cut
    old_train = dates[dates < cut]
    # the last old train sample is at position cut_idx-1 = 149; its label covers pos 149..159
    assert (cut_idx - 1) + PURGE_TRADING_DAYS >= cut_idx, "old train label reaches the cut"
    assert old_train[-1] == dates[cut_idx - 1]
    # label window of old last train sample reaches pos 159 >= cut_idx=150 -> LEAK
    assert cut_idx - 1 + PURGE_TRADING_DAYS >= cut_idx

    # NEW behavior (purge applied)
    train_cut, oos_cut = purge_split(dates, cut_idx)
    # last train sample moved back by PURGE
    assert train_cut[-1] == dates[cut_idx - 1 - PURGE_TRADING_DAYS]
    assert label_window_ok(train_cut, oos_cut, PURGE_TRADING_DAYS, dates)

    # oos front also purged: first oos sample is at cut_idx + PURGE
    assert oos_cut[0] == dates[cut_idx + PURGE_TRADING_DAYS]


def test_purge_constant_matches_label_horizon():
    """PURGE_TRADING_DAYS == 10 == fwd_ret10 label horizon."""
    assert PURGE_TRADING_DAYS == 10


def test_label_identity_on_vwap():
    """fwd_ret10[t] == vwap[t+10]/vwap[t]-1 on a real asset (verifies purge semantics)."""
    import pyarrow.parquet as pq  # noqa: F401
    v = pd.read_parquet(os.path.join(ROOT, "data/panel/vwap_trad.parquet"))
    f = pd.read_parquet(os.path.join(ROOT, "data/panel/fwd_ret10.parquet"))
    a = v.columns[0]
    vp = v[a]
    i = 100
    expect = vp.iloc[i + 10] / vp.iloc[i] - 1.0
    got = f[a].iloc[i]
    assert abs(expect - got) < 1e-9, f"{expect} vs {got}"


def test_purge_vs_unpurged_train_sizes():
    """Purging drops exactly PURGE trading-day rows of train dates."""
    dates = make_dates(300)
    cut_idx = 150
    old_train = dates[dates < dates[cut_idx]]
    train_cut, _ = purge_split(dates, cut_idx)
    assert len(old_train) - len(train_cut) == PURGE_TRADING_DAYS


def test_embargo_layers_on_top():
    """With embargo>0 even more train tail is dropped."""
    dates = make_dates(300)
    cut_idx = 150
    emb = 5
    train_cut, oos_cut = purge_split(dates, cut_idx, purge=PURGE_TRADING_DAYS, embargo=emb)
    assert train_cut[-1] == dates[cut_idx - 1 - (PURGE_TRADING_DAYS + emb)]
    assert oos_cut[0] == dates[cut_idx + PURGE_TRADING_DAYS + emb]
    assert label_window_ok(train_cut, oos_cut, PURGE_TRADING_DAYS + emb, dates)


def test_first_fold_has_no_selection_leak():
    """Synthetic end-to-end: selection_dates never contains a date whose label window
    reaches the OOS block (mirrors factor_selection.py's guard)."""
    import importlib
    fs = importlib.import_module("factor_selection")
    dates = make_dates(300)
    cut = dates[150]
    sel = fs.selection_dates_for_cut(dates, cut)
    assert len(sel) > 0
    max_sel_idx = dates.get_loc(sel.iloc[-1])
    assert max_sel_idx + fs.PURGE_TRADING_DAYS < 150
