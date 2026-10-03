import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.long_ewm import event_decay_native


@pytest.mark.parametrize("halflife", [0.5, 2.0, 30.0])
@pytest.mark.parametrize("min_periods", [1, 3])
def test_event_decay_native_stable_extremes_match_scalar_and_every_prefix(
    halflife, min_periods
):
    """Catch EWM overflow, warmup, gap reset, or future-dependent prefix output."""
    alpha = min(np.log(2.0) / halflife, 1.0)
    frame = pd.DataFrame({
        "asset_id": ["a"] * 9,
        "date": range(9),
        "value": [1e308, -1e308, 1e308, -1e308, np.nan, 3.0,
                  np.inf, -6.0, 8.0],
    }, index=[3, 3, 8, 8, 3, 1, 1, 3, 8])
    actual_series = event_decay_native(
        frame, halflife=halflife, min_periods=min_periods
    )
    actual = actual_series.to_numpy() / 1e308

    scaled = [1.0, -1.0, 1.0, -1.0, np.nan, 3e-308,
              np.inf, -6e-308, 8e-308]
    expected = np.full(len(scaled), np.nan)
    state = np.nan
    count = 0
    for pos in range(1, len(scaled)):
        lagged = scaled[pos - 1]
        if not np.isfinite(lagged):
            state = np.nan
            count = 0
            continue
        count += 1
        state = lagged if not np.isfinite(state) else (1 - alpha) * state + alpha * lagged
        if count >= min_periods:
            expected[pos] = state
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-16,
                               equal_nan=True)
    assert actual_series.index.equals(frame.index)
    for stop in range(1, len(frame) + 1):
        prefix = event_decay_native(
            frame.iloc[:stop], halflife=halflife, min_periods=min_periods
        ).to_numpy() / 1e308
        np.testing.assert_allclose(prefix, actual[:stop], rtol=0, atol=5e-16,
                                   equal_nan=True)

def test_event_decay_native_interleaved_assets_null_asset_and_gap_warmup():
    """Catch cross-asset leakage, null-asset output, or missing reset warmup."""
    frame = pd.DataFrame({
        "asset_id": ["a", "b", None, "a", "b", "a", "b", "a"],
        "date": [1, 1, 1, 2, 2, 3, 3, 4],
        "value": [2.0, 20.0, 999.0, 4.0, 40.0, np.inf, 80.0, 8.0],
    }, index=[5, 5, 9, 5, 9, 2, 2, 5])
    actual = event_decay_native(frame, halflife=2.0, min_periods=2)
    expected = np.array([
        np.nan, np.nan, np.nan, np.nan, np.nan,
        2.0 + 2.0 * np.log(2.0) / 2.0, 20.0 + 20.0 * np.log(2.0) / 2.0,
        np.nan,
    ])
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=0, atol=4e-15,
                               equal_nan=True)
    assert actual.index.equals(frame.index)


@pytest.mark.parametrize("halflife", [True, False, 0, -1, np.nan, np.inf, -np.inf])
def test_event_decay_native_rejects_invalid_halflife(halflife):
    frame = pd.DataFrame({"asset_id": ["a"], "date": [1], "value": [1.0]})
    with pytest.raises(ValueError):
        event_decay_native(frame, halflife=halflife)


@pytest.mark.parametrize("min_periods", [True, 0, -1, 1.5, np.nan])
def test_event_decay_native_rejects_invalid_min_periods(min_periods):
    frame = pd.DataFrame({"asset_id": ["a"], "date": [1], "value": [1.0]})
    with pytest.raises(ValueError):
        event_decay_native(frame, halflife=2.0, min_periods=min_periods)


def test_event_decay_native_rejects_nonmonotone_dates():
    frame = pd.DataFrame({
        "asset_id": ["a", "a"], "date": [2, 1], "value": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="monotone increasing"):
        event_decay_native(frame, halflife=2.0)
