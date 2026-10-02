"""Exact warmup, lag, and nonfinite semantics for FP robust_ewma."""

import numpy as np
import pandas as pd

from factor_preprocess.transforms.smoothing import (
    _robust_ewma_reference,
    robust_ewma,
)


def _frame(values):
    return pd.DataFrame({
        "asset_id": ["a"] * len(values),
        "date": np.arange(len(values)),
        "value": values,
    })


def test_robust_ewma_keeps_finite_value_when_warmup_bounds_are_nan_and_lags_once():
    frame = _frame([2.0, 4.0, 8.0, 16.0, 32.0])
    expected = np.array([np.nan, 2.0, 3.0, 5.5, 10.75])

    actual = robust_ewma(frame, halflife=1.0)
    oracle = _robust_ewma_reference(frame, halflife=1.0)

    # First output is NaN because it consumes the initial lag. At row 1 the
    # rolling bounds are still NaN (one observation), so clip leaves 2 intact.
    # Later outputs consume x[t-1] once; an extra lag would change this vector.
    np.testing.assert_array_equal(actual.to_numpy(), expected)
    np.testing.assert_array_equal(actual.to_numpy(), oracle.to_numpy())


def test_robust_ewma_clips_lagged_infinity_when_rolling_bounds_are_finite():
    frame = _frame([0.0, 0.0, 0.0, 0.0, 10.0, np.inf, np.nan, 0.0])
    actual = robust_ewma(frame, halflife=2.0, winsor_std=1.0)
    oracle = _robust_ewma_reference(frame, halflife=2.0, winsor_std=1.0)

    # Rolling moments skip the infinity, but pandas clip clamps the lagged
    # infinity to the available finite bound. The later NaN remains an EWMA gap.
    assert np.isfinite(actual.iloc[5])
    assert np.isfinite(actual.iloc[6])
    assert not np.isinf(actual.to_numpy()).any()
    np.testing.assert_array_equal(actual.to_numpy(), oracle.to_numpy())
