# -*- coding: utf-8 -*-
"""Regression tests for the R11 grouped rotation cohort semantics.

Two P0 bugs fixed in ``cleaned_operators/stateful/rotation.py``:

1. ``CsRankCompositionChurn`` grouped branch: the pool-membership churn
   ``|A_t Δ A_{t-lag}| / |A_t ∪ A_{t-lag}|`` must be computed over each date's
   NATIVE membership set (finite-observable names of the label on that date),
   not over the intersection ``(g_row==lab) & (gv[r-lk]==lab)``.  A stock that
   moved A -> B is in neither intersection, so its genuine A-exit / B-entry
   must still be reflected in both groups' churn.

2. ``CsTailRetention`` grouped branch: the tail must be defined over each
   date's FULL group cohort (so a stock in A's lagged tail that moved A -> B
   counts as a retention miss for A, not a redefinition of A's lagged tail).
   Only then is the overlap / cohort denominator applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.stateful.rotation import CsRankCompositionChurn
from cleaned_operators.stateful.rotation import CsTailRetention


def _panel(values, groups, cols=None):
    dates = pd.bdate_range("2024-01-02", periods=len(values))
    if cols is None:
        cols = [f"S{i}" for i in range(len(values[0]))]
    x = pd.DataFrame(values, index=dates, columns=cols, dtype=float)
    g = pd.DataFrame(groups, index=dates, columns=cols, dtype=object)
    return x, g


# --------------------------------------------------------------------------
# ISSUE 1 — CsRankCompositionChurn grouped branch
# --------------------------------------------------------------------------

def test_composition_churn_grouped_counts_exit_as_turnover():
    """S2 moving A->B must raise A's churn on date t even though S2 is not in A today."""
    cols = ["S0", "S1", "S2", "S3"]
    # All values finite on both dates; only the group labels change.
    x, group = _panel(
        [[1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0]],
        [["A", "A", "A", "B"],
         ["A", "A", "B", "B"]],  # S2 migrated A -> B
        cols=cols,
    )
    out = CsRankCompositionChurn()._calculate_series(x, lag=1, group=group)
    row = out.iloc[1]
    # Group A: A_prev={S0,S1,S2}, A_cur={S0,S1} -> (3-2)/3 = 1/3, broadcast to
    # A's remaining current members (S0, S1).  The exit of S2 is NOT dropped.
    assert row["S0"] == pytest.approx(1.0 / 3.0)
    assert row["S1"] == pytest.approx(1.0 / 3.0)
    # Group B: B_prev={S3}, B_cur={S2,S3} -> (2-1)/2 = 1/2, broadcast to S2, S3.
    assert row["S2"] == pytest.approx(0.5)
    assert row["S3"] == pytest.approx(0.5)
    # Row 0 has no lag history -> NaN.
    assert np.isnan(out.iloc[0]["S0"])


def test_composition_churn_stable_group_zero_and_migration_increases():
    """A no-migration control gives churn 0; the A->B migration strictly increases A's churn."""
    cols = ["S0", "S1", "S2", "S3"]
    x = pd.DataFrame(
        [[1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0]],
        index=pd.bdate_range("2024-01-02", periods=2), columns=cols,
    )
    stable = pd.DataFrame(
        [["A", "A", "A", "B"],
         ["A", "A", "A", "B"]],
        index=x.index, columns=cols, dtype=object,
    )
    migrated = pd.DataFrame(
        [["A", "A", "A", "B"],
         ["A", "A", "B", "B"]],
        index=x.index, columns=cols, dtype=object,
    )
    out_stable = CsRankCompositionChurn()._calculate_series(x, lag=1, group=stable)
    out_mig = CsRankCompositionChurn()._calculate_series(x, lag=1, group=migrated)
    # Stable: A_prev == A_cur == {S0,S1,S2} -> churn 0.
    assert out_stable.iloc[1]["S0"] == pytest.approx(0.0)
    assert out_stable.iloc[1]["S1"] == pytest.approx(0.0)
    # Migration strictly increases group-A churn on date t.
    assert out_mig.iloc[1]["S0"] > out_stable.iloc[1]["S0"]
    assert out_mig.iloc[1]["S1"] > out_stable.iloc[1]["S1"]


# --------------------------------------------------------------------------
# ISSUE 2 — CsTailRetention grouped branch
# --------------------------------------------------------------------------

def test_tail_retention_grouped_migration_is_retention_miss():
    """A stock in A's lagged tail that moved A->B is a retention MISS for A.

    The lagged tail must be defined over A's FULL lagged cohort: S2 was the
    top-1 tail of A yesterday, so its departure today must drop A's retention
    to 0 — the old intersection-sample bug would have redefined the lagged tail
    inside {S0} and reported 1.0.
    """
    cols = ["S0", "S1", "S2", "S3"]
    x, group = _panel(
        [[50.0, 100.0, 60.0, 1.0],
         [55.0, 105.0, 65.0, 2.0]],
        [["A", "B", "A", "B"],
         ["A", "B", "B", "B"]],  # S2 migrated A -> B
        cols=cols,
    )
    out = CsTailRetention()._calculate_series(
        x, lag=1, quantile=0.5, side="top", group=group, cohort="intersection"
    )
    # T_prev = {S2} (A's lagged cohort {S0,S2}, top-1), T_cur = {S0} (A today).
    # num = 0; intersection den = T_prev observable today = 1 -> retention 0.
    assert out.iloc[1]["S0"] == pytest.approx(0.0)
    # S2 left A -> it is a current B member now, so it carries B's retention
    # (B kept its whole lagged top-1 tail S1 -> B retention 1.0), not A's value.
    assert out.iloc[1]["S2"] == pytest.approx(1.0)
    assert out.iloc[1]["S1"] == pytest.approx(1.0)


def test_tail_retention_grouped_control_no_migration():
    """Same cohort, no migration -> top-1 tail is fully retained (retention 1.0)."""
    cols = ["S0", "S1", "S2", "S3"]
    x, group = _panel(
        [[50.0, 100.0, 60.0, 1.0],
         [55.0, 105.0, 65.0, 2.0]],
        [["A", "B", "A", "B"],
         ["A", "B", "A", "B"]],  # no migration
        cols=cols,
    )
    out = CsTailRetention()._calculate_series(
        x, lag=1, quantile=0.5, side="top", group=group, cohort="intersection"
    )
    # T_prev = {S2}, T_cur = {S2} -> num=1, den=1 -> retention 1.0, broadcast to
    # both of today's A members (S0, S2).
    assert out.iloc[1]["S0"] == pytest.approx(1.0)
    assert out.iloc[1]["S2"] == pytest.approx(1.0)


def test_tail_retention_grouped_cohort_denominators():
    """Exercise intersection / historical / current denominators under migration.

    A at t-lag = {S0,S1,S2} with top-2 tail {S0,S2}; S2 moves A->B, so today
    A = {S0,S1} with top-1 tail {S0}.  T_prev={S0,S2}, T_cur={S0}:
    - intersection: den = |T_prev| observable today = 2 -> 1/2
    - historical:   den = |T_prev|                    = 2 -> 1/2
    - current:      den = |T_cur|                     = 1 -> 1.0
    """
    cols = ["S0", "S1", "S2", "S3"]
    x, group = _panel(
        [[100.0, 50.0, 90.0, 1.0],
         [98.0, 45.0, 92.0, 2.0]],
        [["A", "A", "A", "B"],
         ["A", "A", "B", "B"]],  # S2 migrated A -> B
        cols=cols,
    )
    for cohort, expected in [("intersection", 0.5), ("historical", 0.5), ("current", 1.0)]:
        out = CsTailRetention()._calculate_series(
            x, lag=1, quantile=0.5, side="top", group=group, cohort=cohort
        )
        assert out.iloc[1]["S0"] == pytest.approx(expected), f"cohort={cohort}"
        assert out.iloc[1]["S1"] == pytest.approx(expected), f"cohort={cohort}"


def test_tail_retention_grouped_fractional_boundary_tie():
    """Fractional boundary-tie mass in the grouped branch.

    A at t-lag = {S0(10), S1(10), S2(5)}, q=1/3 -> k=1, boundary tie splits
    0.5/0.5 between S0 and S1.  S1 moves A->B, so today A = {S0(10), S2(6)}
    with top-1 {S0}.  num = min(0.5,1) = 0.5; intersection den = 1.0 (the
    migrated S1 still carries its 0.5 lagged mass and is a miss) -> 0.5.
    """
    cols = ["S0", "S1", "S2", "S3"]
    x, group = _panel(
        [[10.0, 10.0, 5.0, 1.0],
         [10.0, 9.0, 6.0, 2.0]],
        [["A", "A", "A", "B"],
         ["A", "B", "A", "B"]],  # S1 migrated A -> B
        cols=cols,
    )
    out = CsTailRetention()._calculate_series(
        x, lag=1, quantile=1.0 / 3.0, side="top", group=group, cohort="intersection"
    )
    assert out.iloc[1]["S0"] == pytest.approx(0.5)
    assert out.iloc[1]["S2"] == pytest.approx(0.5)
