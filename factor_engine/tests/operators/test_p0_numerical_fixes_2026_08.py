# -*- coding: utf-8 -*-
"""Golden tests locking in the P0 deterministic-numerical fixes (2026-08-08).

Covers:
* P0-5  ``group_tail_lead_score`` — prefix invariance (future leak fixed).
* P0-6  ``cs_weighted_percentile_rank`` — unsort corrected (hand-computed).
* P0-7  ``ts_score_rank_weighted_mean`` — weight/target alignment (hand-computed).
* P0-12 ``_diagram_w1`` — Hungarian illegal 0-cost edges (non-identical diagrams).
* P1-23 ``group_topk_mean`` — tie policy column-permutation invariant.
* P1-24 ``event_level_survival_share`` — true survival across the path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _col(vals, name: str = "A") -> pd.DataFrame:
    return pd.DataFrame(np.asarray(vals, dtype=float).reshape(-1, 1), columns=[name])


# ---------------------------------------------------------------------------
# P0-5: group_tail_lead_score prefix invariance.
# ---------------------------------------------------------------------------
def test_group_tail_lead_score_prefix_invariance() -> None:
    from cleaned_operators.tail_systemic import _tail_lead_series

    rng = np.random.default_rng(7)
    rows, cols = 40, 4
    x = rng.normal(size=(rows, cols))
    g = np.tile(np.array(["A", "A", "B", "B"]), (rows, 1))
    w, q, lag = 15, 0.2, 2

    full = _tail_lead_series(x, g, w, q, "lower", lag)
    before = full[:20].copy()

    # Randomly scramble the FUTURE rows (rows 20..39) — a prefix-causal operator
    # must be bit-identical on rows 0..19.
    x_perturbed = x.copy()
    for c in range(cols):
        x_perturbed[20:, c] = np.roll(x_perturbed[20:, c], 3)
    after = _tail_lead_series(x_perturbed, g, w, q, "lower", lag)[:20]

    ok = np.isfinite(before)
    np.testing.assert_allclose(before[ok], after[ok], atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# P0-6: cs_weighted_percentile_rank hand-computed unsort.
# ---------------------------------------------------------------------------
def test_cs_weighted_percentile_rank_hand_golden() -> None:
    from cleaned_operators.gather_ext import _cs_weighted_percentile_rank

    x = pd.DataFrame([[30.0, 10.0, 20.0]], columns=["a", "b", "c"])
    w = pd.DataFrame([[1.0, 1.0, 1.0]], columns=["a", "b", "c"])
    out = _cs_weighted_percentile_rank(x, w)
    # weighted mid-rank: x=10 -> (0+0.5)/3=0.1667, x=20 -> 0.5, x=30 -> 0.8333
    np.testing.assert_allclose(
        out.values,
        [[0.83333333, 0.16666667, 0.5]],
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# P0-7: ts_score_rank_weighted_mean weight/target alignment.
# ---------------------------------------------------------------------------
def test_score_rank_weighted_mean_hand_golden() -> None:
    from cleaned_operators.advanced_information import _score_rank_weighted_mean

    # sv=[3,1,2] -> ranks [0,2,1]; tv=[30,10,20]; decay=0.5
    # weights = [1, 0.25, 0.5] normalized -> mean = (30 + 2.5 + 10)/1.75 = 24.2857.
    # The misaligned version multiplied weight_i by target_{order_i} and returned
    # (30 + 0.25*20 + 0.5*10)/1.75 = 22.857.
    out = _score_rank_weighted_mean(np.array([30.0, 10.0, 20.0]), np.array([3.0, 1.0, 2.0]), 0.5)
    assert out == pytest.approx(24.2857142857, rel=1e-6)


# ---------------------------------------------------------------------------
# P0-12: persistence-diagram W1 must not resolve via illegal 0-cost edges.
# ---------------------------------------------------------------------------
def test_diagram_w1_no_illegal_zero_edges() -> None:
    from cleaned_operators.advanced_topology import _diagram_w1

    # A=[(1,3)], B=[(1,3),(10,20)]. True W1: A matches B1 at cost 0, B2 matches
    # the diagonal at (20-10)/2=5 -> total 5.0 (TOTAL assignment cost, not the
    # old /max(m1,m2)=2.5 mean; R6-120 removed that division).  The old
    # construction resolved B1->B2-diagonal (0) + B2->A-diagonal (0) at cost 0.
    assert _diagram_w1([(1.0, 3.0)], [(1.0, 3.0), (10.0, 20.0)]) == pytest.approx(5.0)
    # identical diagrams -> exactly 0
    assert _diagram_w1([(1.0, 2.0)], [(1.0, 2.0)]) == pytest.approx(0.0)
    # empty-vs-nonempty is the SUM of the non-empty diagram's diagonal distances
    # (diag costs 1.0 + 5.0 = 6.0), symmetric, and identical to the general-case
    # total scale — never a mean (would depend on diagram size and break the metric).
    assert _diagram_w1([], [(1.0, 3.0), (10.0, 20.0)]) == pytest.approx(6.0)
    assert _diagram_w1([(1.0, 3.0), (10.0, 20.0)], []) == pytest.approx(6.0)
    assert _diagram_w1([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# P1-23: group_topk_mean tie policy is column-permutation invariant.
# ---------------------------------------------------------------------------
def test_group_topk_mean_tie_column_permutation_invariant() -> None:
    from cleaned_operators.gather_ext import _group_topk_mean

    tgt = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], columns=["a", "b", "c", "d", "e"])
    sc = pd.DataFrame([[5.0, 5.0, 5.0, 1.0, 0.0]], columns=["a", "b", "c", "d", "e"])
    grp = pd.DataFrame([["g"] * 5], columns=["a", "b", "c", "d", "e"])

    r1 = _group_topk_mean(tgt, sc, grp, k=2).values
    perm = ["e", "d", "c", "b", "a"]
    r2 = _group_topk_mean(tgt[perm], sc[perm], grp[perm], k=2).values
    np.testing.assert_allclose(r1, r2[:, ::-1], equal_nan=True)


# ---------------------------------------------------------------------------
# P1-24: event_level_survival_share tracks the whole path, not today only.
# ---------------------------------------------------------------------------
def test_event_level_survival_share_path_aware() -> None:
    from cleaned_operators.gather_ext import _event_level_survival_share

    ev = pd.DataFrame([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    lv = pd.DataFrame([[10.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    # event at row0 level=10 (up); x dips to 9 then recovers to 12 -> the level
    # was BROKEN even though today's close is above it.
    x_dip = pd.DataFrame([[11.0, 5.0], [9.0, 5.0], [12.0, 5.0]])
    out = _event_level_survival_share(ev, lv, x_dip, history_window=10, direction="up")
    np.testing.assert_allclose(out.iloc[:, 0].to_numpy(), [np.nan, 0.0, 0.0], equal_nan=True)
    # no dip -> survives
    x_ok = pd.DataFrame([[11.0, 5.0], [12.0, 5.0], [13.0, 5.0]])
    out2 = _event_level_survival_share(ev, lv, x_ok, history_window=10, direction="up")
    np.testing.assert_allclose(out2.iloc[:, 0].to_numpy(), [np.nan, 1.0, 1.0], equal_nan=True)
