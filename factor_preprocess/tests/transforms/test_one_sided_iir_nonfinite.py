import numpy as np
import pandas as pd

import pytest

from factor_preprocess.transforms.smoothing import (
    kalman_local_level, one_sided_iir_lowpass,
)


def _frame(values):
    return pd.DataFrame({
        "asset_id": ["A"] * len(values),
        "date": np.arange(len(values)),
        "value": values,
    })


def test_one_sided_iir_nonfinite_lags_reset_state_and_remain_causal():
    frame = _frame([10.0, 1.0, np.inf, 3.0, -np.inf, 5.0, 6.0])
    original = frame.copy(deep=True)

    actual = one_sided_iir_lowpass(frame, alpha=0.5)

    # The current observation is excluded; every non-finite lag resets the
    # state, and the following finite lag seeds a fresh recurrence.
    expected = np.array([np.nan, 10.0, 5.5, np.nan, 3.0, np.nan, 5.0])
    np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)
    pd.testing.assert_frame_equal(frame, original)

    # Recomputing any prefix must preserve all outputs in that prefix.
    for stop in range(1, len(frame) + 1):
        prefix = one_sided_iir_lowpass(frame.iloc[:stop].copy(), alpha=0.5)
        np.testing.assert_allclose(
            prefix.to_numpy(), actual.iloc[:stop].to_numpy(), equal_nan=True
        )

def test_one_sided_iir_nonfinite_matches_nan_and_resets_independent_assets():
    frame = pd.DataFrame({
        "asset_id": ["A", "B"] * 4,
        "date": [0, 0, 1, 1, 2, 2, 3, 3],
        "value": [10.0, 7.0, np.inf, -np.inf, 2.0, 3.0, 4.0, 5.0],
    })
    expected = np.array([np.nan, np.nan, 10.0, 7.0,
                         np.nan, np.nan, 2.0, 3.0])
    actual = one_sided_iir_lowpass(frame, alpha=1.0)
    np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)

    nan_equivalent = frame.copy(deep=True)
    nan_equivalent["value"] = nan_equivalent["value"].replace(
        [np.inf, -np.inf], np.nan
    )
    np.testing.assert_allclose(
        actual.to_numpy(),
        one_sided_iir_lowpass(nan_equivalent, alpha=1.0).to_numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.01, np.nan, np.inf])
def test_one_sided_iir_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha must be in"):
        one_sided_iir_lowpass(_frame([1.0]), alpha=alpha)


def test_kalman_nonfinite_lags_reset_state_without_infinite_output():
    frame = _frame([1.0, 2.0, 3.0, np.inf, 4.0, -np.inf, 5.0, 6.0])
    expected = frame.copy(deep=True)
    expected["value"] = expected["value"].replace([np.inf, -np.inf], np.nan)

    actual = kalman_local_level(frame, process_noise=0.01, measurement_noise=0.3)
    reference = kalman_local_level(
        expected, process_noise=0.01, measurement_noise=0.3
    )
    np.testing.assert_allclose(actual.to_numpy(), reference.to_numpy(), equal_nan=True)
    assert np.isnan(actual.iloc[4]) and np.isnan(actual.iloc[6])
    assert np.isfinite(actual.iloc[7])
    pd.testing.assert_frame_equal(
        frame, _frame([1.0, 2.0, 3.0, np.inf, 4.0, -np.inf, 5.0, 6.0])
    )
