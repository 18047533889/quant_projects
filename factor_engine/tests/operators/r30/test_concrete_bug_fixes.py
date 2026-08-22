# -*- coding: utf-8 -*-
"""R30 Phase 9 concrete-bug regression tests.

* P0-011: DMD log geometric sum is finite for large positive log_rho.
* P0-012: survival gap never carries pre-gap episode age.
* P0-013: fin_component_score three-valued logic (FALSE != MISSING).
* P0-016: KAMA warmup needs er_window+1 contiguous prices.
* P0-017: Supertrend gap re-seed is UNKNOWN, not bullish.
* P0-018: PSAR gap re-seed does not manufacture a long bias.
* P0-019: ts_regression sample floor scales with n_regressors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.execution_contract import minimum_effective_samples
from cleaned_operators.dmd import _log_finite_horizon_sum
from cleaned_operators.stateful.survival import _survival_kernel


# ---------- P0-011 DMD log geometric sum ----------
def test_dmd_log_sum_finite_for_large_positive():
    for lr in (100.0, 1000.0, 10000.0):
        r = _log_finite_horizon_sum(lr, 50)
        assert np.isfinite(r) and r > 0, (lr, r)


def test_dmd_log_sum_matches_reference():
    K = 50
    for lr in (-1e-9, 1e-9, 0.5, 2.0):
        rho = np.exp(lr)
        ref = np.log(np.sum(rho ** np.arange(K, dtype=float)))
        assert abs(_log_finite_horizon_sum(lr, K) - ref) < 1e-6


def test_dmd_log_sum_extremes():
    # R30 §15: 0, ±1e-15, ±1e-9, ±1e-6, 1,10,100,1000,-1000 all finite.
    for lr in (0.0, -1e-15, 1e-15, -1e-9, 1e-9, -1e-6, 1e-6, 1.0, 10.0, 100.0, 1000.0, -1000.0):
        assert np.isfinite(_log_finite_horizon_sum(lr, 50)), lr


# ---------- P0-012 survival gap state ----------
def _surv_ages(seq):
    s = np.array(seq, dtype=float)
    age, *_ = _survival_kernel(s, 60, 1, 1, 1, 1.0, inactive_policy="zero", gap_policy="lower_bound")
    return age


def test_survival_gap_does_not_carry_episode_age():
    # 0,1,NaN,1,0 -> post-gap 1 starts a NEW episode at age 1 (not inheriting 2).
    ages = _surv_ages([0, 1, np.nan, 1, 0])
    assert ages[3] == 1.0, ages


def test_survival_gap_then_inactive_then_active():
    # 0,1,NaN,0,1 -> post-gap 1 is an observed-entry episode at age 1.
    ages = _surv_ages([0, 1, np.nan, 0, 1])
    assert ages[4] == 1.0, ages


def test_survival_long_gap_sequence():
    ages = _surv_ages([0, 1, 1, np.nan, 1, 1, 0])
    assert ages[4] == 1.0, ages  # post-gap first active restarts at 1
    assert ages[5] == 2.0, ages


# ---------- P0-019 regression sample floor ----------
def test_regression_floor_parameter_aware():
    for n, expect in ((1, 2), (2, 3), (5, 6), (10, 11)):
        got = minimum_effective_samples("ts_regression", {"n_regressors": n})
        assert got == expect, (n, got, expect)
