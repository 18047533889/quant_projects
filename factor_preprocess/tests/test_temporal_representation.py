import numpy as np
import pandas as pd
import pytest


def frame(x):
    return pd.DataFrame({"date": np.arange(len(x)), "asset_id": "a", "value": x})


def test_current_rank_uses_only_declared_previous_window_and_average_or_min_ties():
    from factor_preprocess.transforms.temporal_representation import time_series_rank
    data = frame([1., 2., 2., 2., 5.])
    np.testing.assert_allclose(time_series_rank(data, window=3),
                               [np.nan, np.nan, np.nan, 2/3, 1.], equal_nan=True)
    np.testing.assert_allclose(time_series_rank(data, window=3, method="min"),
                               [np.nan, np.nan, np.nan, 1/3, 1.], equal_nan=True)


def test_zscore_uses_lagged_sample_std_caps_extremes_and_rejects_zero_variance():
    from factor_preprocess.transforms.temporal_representation import capped_time_series_zscore
    np.testing.assert_allclose(capped_time_series_zscore(frame([1., 2., 3., 4., 20.]),
                                                        window=3, cap=3.),
                               [np.nan, np.nan, np.nan, 2., 3.], equal_nan=True)
    assert np.isnan(capped_time_series_zscore(frame([2., 2., 2., 4.]), window=3).iloc[-1])
    assert capped_time_series_zscore(frame([1., 2., 3., 1e308]), window=3).iloc[-1] == 3.
    assert np.isfinite(capped_time_series_zscore(frame([-1e308, 0., 1e308, 1e308]),
                                                window=3).iloc[-1])


@pytest.mark.parametrize("kind", ["rank", "zscore"])
def test_temporal_representation_preserves_prefix_asset_isolation_and_missing_windows(kind):
    from factor_preprocess.transforms.temporal_representation import time_series_rank, capped_time_series_zscore
    fn = time_series_rank if kind == "rank" else capped_time_series_zscore
    data = pd.concat([frame([1., 2., 3., np.nan, 5., 6., 7., 8.]),
                      frame([8., 5., 2., 4., 9., 7., 8., 2.]).assign(asset_id="b")])
    data = data.sort_values("date", kind="stable")
    data.index = [1]*len(data)
    out = fn(data, window=3)
    assert out.index.equals(data.index)
    np.testing.assert_array_equal(out[data.date < 5], fn(data[data.date < 5], window=3))
    a = out.to_numpy()[data.asset_id.to_numpy()=="a"]
    assert np.isnan(a[3:7]).all()
    assert np.isfinite(a[7])
    order = np.arange(len(data)).reshape(-1, 2)[:, ::-1].ravel()
    np.testing.assert_array_equal(out, fn(data.iloc[order], window=3).to_numpy()[np.argsort(order)])


@pytest.mark.parametrize("window", [True, 1, 253, 2.5])
def test_temporal_window_requires_explicit_bounded_integer(window):
    from factor_preprocess.transforms.temporal_representation import time_series_rank
    with pytest.raises(ValueError, match="window"):
        time_series_rank(frame([1., 2., 3.]), window=window)
