import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.long_ewm import lagged_iir_lowpass


def _reference(frame, alpha):
    result = np.full(len(frame), np.nan, dtype=np.float64)
    raw = frame["value"].reset_index(drop=True).to_numpy(dtype=float)
    for positions in frame.groupby("asset_id", sort=False).indices.values():
        state = np.nan
        for i, pos in enumerate(positions):
            sample = raw[positions[i - 1]] if i else np.nan
            if not np.isfinite(sample):
                state = np.nan
            elif not np.isfinite(state):
                state = sample
            else:
                state = (1.0 - alpha) * state + alpha * sample
            result[pos] = state
    return result


@pytest.mark.parametrize("alpha", [1.0, 0.37, np.nextafter(0.0, 1.0)])
def test_lagged_iir_matches_scalar_oracle_for_gaps_and_interleaved_keys(alpha):
    frame = pd.DataFrame({
        "asset_id": pd.Categorical(
            ["a", "b", "a", None, "b", "a", "b", "a", "b", "a"],
            categories=["a", "b", "unused"],
        ),
        "date": [1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        "value": [1e100, 7., 2e100, 999., -3., np.inf, 5., 4., -np.inf, 8.],
    }, index=[4, 4, 7, 7, 4, 1, 7, 1, 4, 1])
    original = frame.copy(deep=True)
    actual = lagged_iir_lowpass(frame, alpha=alpha)
    np.testing.assert_allclose(actual.to_numpy(), _reference(frame, alpha),
                               rtol=2e-15, atol=0.0, equal_nan=True)
    assert actual.index.equals(frame.index)
    pd.testing.assert_frame_equal(frame, original)
    for stop in range(1, len(frame) + 1):
        prefix = lagged_iir_lowpass(frame.iloc[:stop].copy(), alpha=alpha)
        np.testing.assert_allclose(prefix.to_numpy(), actual.iloc[:stop].to_numpy(),
                                   rtol=2e-15, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.01, np.nan, np.inf])
def test_lagged_iir_rejects_invalid_alpha(alpha):
    frame = pd.DataFrame({"asset_id": ["a"], "date": [1], "value": [1.]})
    with pytest.raises(ValueError, match="alpha must be"):
        lagged_iir_lowpass(frame, alpha=alpha)


def test_lagged_iir_empty_and_all_null_keys():
    frame = pd.DataFrame({"asset_id": [None, None], "date": [1, 2], "value": [1., 2.]})
    assert lagged_iir_lowpass(frame, alpha=0.5).isna().all()
    assert lagged_iir_lowpass(frame.iloc[:0], alpha=0.5).empty


def test_lagged_iir_keeps_subnormal_alpha_exact():
    alpha = np.nextafter(0.0, 1.0)
    frame = pd.DataFrame({"asset_id": ["a"] * 3, "date": [1, 2, 3],
                          "value": [0.0, 1.0, 1.0]})
    actual = lagged_iir_lowpass(frame, alpha=alpha)
    assert np.isnan(actual.iloc[0])
    assert actual.iloc[1] == 0.0
    assert actual.iloc[2] == alpha
