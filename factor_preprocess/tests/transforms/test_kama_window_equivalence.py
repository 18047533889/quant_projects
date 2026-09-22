import numpy as np
import pandas as pd
import pytest
from factor_preprocess.transforms.smoothing import kama


def oracle(x, window, current):
    z = x.copy() if current else np.r_[np.nan, x[:-1]].astype(x.dtype)
    z = np.where(np.isfinite(z), z, np.nan)
    out = np.full(len(z), np.nan)
    state = np.nan
    for i in range(window, len(z)):
        if not np.isfinite(state):
            if not np.isnan(z[i]):
                state = z[i]
            continue
        history = z[i-window:i+1]
        if not np.isfinite(history).all():
            out[i] = state
            continue
        path = float(np.abs(np.diff(history)).sum())
        direction = float(abs(history[-1]-history[0]))
        er = direction/path if path > 0 else 0.
        # The public kernel stores efficiency/gain in float64 even when the
        # observations are float32; retain that promotion in this oracle.
        gain = np.float64((er*(2/3-2/31)+2/31)**2)
        state = state + gain*(z[i]-state)
        out[i] = state
    return out


@pytest.mark.parametrize('window', [1, 3, 10, 100])
@pytest.mark.parametrize('current', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_kama_interleaved_irregular_groups_match_scalar_oracle(window, current, dtype):
    rng = np.random.default_rng(911)
    x = rng.normal(size=(80, 2)).astype(dtype)
    x[9:12, 0] = np.nan
    x[35, 1] = np.inf
    x[45:55, 1] = 2.
    frame = pd.DataFrame(dict(date=np.repeat(np.arange(80)*2, 2),
        asset_id=np.tile(['b', 'a'], 80), value=x.ravel()))
    frame = frame.drop(index=[22, 23, 44, 87]).copy()
    frame.index = np.arange(len(frame)) % 7
    actual = kama(frame, period_er=window, use_current=current)
    expected = np.full(len(frame), np.nan)
    for indices in frame.groupby('asset_id', sort=False).indices.values():
        expected[indices] = oracle(frame.iloc[indices].value.to_numpy(), window, current)
    assert actual.index.equals(frame.index)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13, equal_nan=True)
    prefix = kama(frame.iloc[:83], period_er=window, use_current=current)
    np.testing.assert_array_equal(actual.iloc[:83].to_numpy(), prefix.to_numpy())
