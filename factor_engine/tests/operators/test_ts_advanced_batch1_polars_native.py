"""Focused parity and PIT tests for ts_advanced_batch1 native implementations."""

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators.polars_native.ts_advanced_batch1 import (
    TSAbsConcentrationPolarsNative,
    TSConfirmedPivotHighPolarsNative,
    TSConfirmedPivotLowPolarsNative,
)


def _series(values):
    return pd.Series(values, index=pd.RangeIndex(len(values)), name="x", dtype=float)


def _expected_hhi(values, window):
    out = []
    for i in range(len(values)):
        if i < window - 1:
            out.append(np.nan)
            continue
        chunk = np.asarray(values[i - window + 1 : i + 1], dtype=float)
        chunk = np.abs(chunk[np.isfinite(chunk)])
        total = chunk.sum()
        out.append(np.nan if total <= 0 else float(np.square(chunk / total).sum()))
    return np.asarray(out)


def test_abs_concentration_matches_pandas_reference_and_zero_is_nan():
    feature = _series([1.0, -2.0, 3.0, 0.0, -4.0, np.nan])
    out = TSAbsConcentrationPolarsNative()._calculate_series(feature, window=3)
    expected = _expected_hhi(feature.to_numpy(), 3)
    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)
    assert np.isnan(out.iloc[3]) is False


def test_confirmed_pivot_high_emits_only_after_right_bars():
    feature = _series([1.0, 3.0, 2.0, 1.0, 4.0, 2.0])
    out = TSConfirmedPivotHighPolarsNative()._calculate_series(feature, left_bars=1, right_bars=1)
    expected = np.array([np.nan, np.nan, 3.0, np.nan, np.nan, 4.0])
    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)
    # Poisoning future values after the confirmation timestamp cannot alter it.
    poisoned = feature.copy()
    poisoned.iloc[4:] = [-100.0, -100.0]
    poisoned_out = TSConfirmedPivotHighPolarsNative()._calculate_series(poisoned, 1, 1)
    assert poisoned_out.iloc[2] == 3.0


def test_confirmed_pivot_low_emits_only_after_right_bars():
    feature = _series([3.0, 1.0, 2.0, 4.0, 0.0, 2.0])
    out = TSConfirmedPivotLowPolarsNative()._calculate_series(feature, left_bars=1, right_bars=1)
    expected = np.array([np.nan, np.nan, 1.0, np.nan, np.nan, 0.0])
    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)
