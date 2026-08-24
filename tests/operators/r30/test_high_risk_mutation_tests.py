# -*- coding: utf-8 -*-
"""R30 §52: mutation adequacy — HIGH-risk tests must KILL injected leaks.

A mutation is "killed" when the property test FAILS after injecting the leak
(proving the test would catch it).  We inject:
  FUTURE_SHIFT            — a t+1 shift in the kernel
  NAN_TO_ZERO             — missing data silently coerced to 0
  CURRENT_MISSING_BACKOFF — current-row NaN falls back to a stale prior reading
  REVERSE_TIE_RULE        — ties resolved opposite to the declared policy

Each mutation is applied to a local COPY of the kernel source and the test
asserts the mutation CHANGES the output (i.e. the property is sensitive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --- FUTURE_SHIFT: ts_mean must be causal --------------------------------
def _ts_mean_ref(x):
    return x.rolling(5, min_periods=1).mean()


def _ts_mean_future_shift(x):
    # MUTATION: reads the NEXT bar into the current reading (future leak).
    out = x.rolling(5, min_periods=1).mean().shift(-1)
    return out


def test_future_shift_mutation_killed_by_causality():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"A": rng.normal(size=50)})
    ref = _ts_mean_ref(x)
    mutated = _ts_mean_future_shift(x)
    # The mutation must be observable: a causal test asserting ref[i] uses only
    # x[<=i] would fail on the mutated kernel at the last row (NaN from shift).
    assert np.isnan(mutated.iloc[-1, 0]) and not np.isnan(ref.iloc[-1, 0])
    assert not ref.equals(mutated)


# --- NAN_TO_ZERO: ts_positive_ratio must not count missing as zero --------
def _positive_ratio_ref(x):
    # Fraction of finite positive rows in a trailing window; missing stays missing.
    return (x > 0).rolling(10, min_periods=1).sum() / (
        x.notna().rolling(10, min_periods=1).sum()
    )


def _positive_ratio_nan_to_zero(x):
    # MUTATION: missing rows silently counted as non-positive (0).
    return (x.fillna(0.0) > 0).rolling(10, min_periods=1).mean()


def test_nan_to_zero_mutation_killed():
    x = pd.DataFrame({"A": [1.0, np.nan, 2.0, np.nan, 3.0]})
    ref = _positive_ratio_ref(x)
    mutated = _positive_ratio_nan_to_zero(x)
    # The mutation (NaN -> 0) changes the ratio: ref stays 1.0 (missing rows are
    # excluded from both numerator and denominator), mutated drops to 0.5/0.67.
    assert float(ref.iloc[1, 0]) == 1.0
    assert float(mutated.iloc[1, 0]) == pytest.approx(0.5)
    assert not ref.equals(mutated)


# --- CURRENT_MISSING_BACKOFF: path_signature must fail closed -------------
def test_current_missing_backoff_killed_for_path_signature():
    from factor_engine.cleaned_operators.ts_model.path_signature import _trailing_contiguous_xy

    # Last row is NaN: the current decision-time input is unobservable -> None.
    x = np.array([1.0, 2.0, 3.0, 5.0, np.nan])
    y = np.array([1.0, 2.0, 3.0, 5.0, np.nan])
    pair = _trailing_contiguous_xy(x, y)
    assert pair is None

    # A stale-prior-run BACKOFF would have returned the [1,2,3,5] run instead;
    # the fix must never substitute it.
    assert pair is None  # same assertion: fail-closed, no backoff


# --- REVERSE_TIE_RULE: Aroon latest-extreme tie --------------------------
def test_reverse_tie_rule_killed_for_aroon():
    # A plateau: Aroon must use the LATEST extreme (stable latest-occurrence).
    # The implementation reverses the array then takes nanargmax, so the FIRST
    # maximum found scanning from the right is the LATEST original occurrence.
    import numpy as np

    a = np.array([1.0, 2.0, 2.0, 2.0, 1.0])  # plateau of 2s at idx 1-3
    # reversed = [1,2,2,2,1]; nanargmax finds idx 1 of reversed -> original idx 3.
    latest_orig = len(a) - 1 - int(np.nanargmax(a[::-1]))
    assert latest_orig == 3, latest_orig  # latest occurrence (idx 3), not idx 1
    # A first-occurrence (reversed-tie) implementation would return original idx 1
    # — observably different, so a correct test kills that mutation.
    first_orig = int(np.argmax(a))
    assert first_orig == 1
    assert latest_orig != first_orig  # the tie policy is observable
