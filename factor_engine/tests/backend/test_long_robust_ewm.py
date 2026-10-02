import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.long_robust_ewm import lagged_robust_ewma


def _oracle(frame, halflife, winsor_std, min_periods):
    width = winsor_std if winsor_std > 0 else 1e-12
    out = np.full(len(frame), np.nan)
    raw = frame["value"].reset_index(drop=True).to_numpy(dtype=float)
    for positions in frame.groupby("asset_id", sort=False, observed=True).indices.values():
        lag = pd.Series(raw[positions]).shift(1)
        finite = lag.where(np.isfinite(lag))
        mean = finite.rolling(10, min_periods=2).mean()
        std = finite.rolling(10, min_periods=2).std(ddof=1).clip(lower=1e-12)
        clipped = lag.clip(lower=mean - width * std, upper=mean + width * std)
        clipped = clipped.where(np.isfinite(clipped))
        out[positions] = clipped.ewm(
            halflife=halflife, min_periods=min_periods,
            adjust=False, ignore_na=False,
        ).mean().to_numpy()
    return out


def _frame():
    return pd.DataFrame({
        "asset_id": pd.Categorical(
            ["a", "b", "a", None, "b", "a", "b", "a", "b", "a",
             "b", "a", "b", "a", "b", "a", "b"],
            categories=["a", "b", "unused"],
        ),
        "date": [1, 1, 1, None, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        "value": [10., 3., 12., 999., np.nan, 13., np.inf, 50., -np.inf,
                  12., 4., np.nan, 5., 14., 6., 15., 7.],
    }, index=[3, 3, 1, 1, 3, 8, 1, 8, 1, 3, 8, 3, 1, 8, 3, 1, 8])


@pytest.mark.parametrize("min_periods", [0, 1, 3])
def test_pandas_boundary_preserves_interleaved_rows_and_duplicate_index(min_periods):
    frame = _frame()
    original = frame.copy(deep=True)
    actual = lagged_robust_ewma(
        frame, halflife=2.5, winsor_std=1.5, min_periods=min_periods,
    )
    expected = _oracle(frame, 2.5, 1.5, min_periods)
    np.testing.assert_allclose(
        actual.to_numpy(), expected, rtol=2e-13, atol=1e-13, equal_nan=True,
    )
    assert actual.index.equals(frame.index)
    assert actual.name == "value"
    pd.testing.assert_frame_equal(frame, original)


def test_empty_and_all_missing_assets_keep_input_index():
    empty = pd.DataFrame({"asset_id": pd.Series(dtype="object"),
                          "date": pd.Series(dtype="int64"),
                          "value": pd.Series(dtype="float64")},
                         index=pd.Index([], dtype="int64"))
    assert lagged_robust_ewma(empty, halflife=2.0).empty
    frame = pd.DataFrame({"asset_id": [None, np.nan], "date": [1, 2],
                          "value": [1., 2.]}, index=[4, 4])
    actual = lagged_robust_ewma(frame, halflife=2.0)
    assert actual.index.equals(frame.index)
    assert actual.isna().all()


@pytest.mark.parametrize("kwargs", [
    {"halflife": 0.0}, {"halflife": np.nan}, {"halflife": np.inf},
    {"halflife": True}, {"halflife": 2.0, "min_periods": -1},
    {"halflife": 2.0, "min_periods": True},
])
def test_rejects_invalid_ewma_parameters(kwargs):
    frame = pd.DataFrame({"asset_id": ["a"], "date": [1], "value": [1.]})
    with pytest.raises(ValueError):
        lagged_robust_ewma(frame, **kwargs)


def test_rejects_decreasing_per_asset_times_but_allows_equal_times():
    frame = pd.DataFrame({"asset_id": ["a", "a"], "date": [2, 1],
                          "value": [1., 2.]})
    with pytest.raises(ValueError, match="monotone"):
        lagged_robust_ewma(frame, halflife=2.0)
    tied = frame.assign(date=[1, 1])
    assert len(lagged_robust_ewma(tied, halflife=2.0)) == 2


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"asset_id": pd.Series(dtype="object"),
                  "date": pd.Series(dtype="int64"),
                  "value": pd.Series(dtype="float64")}),
    pd.DataFrame({"asset_id": [None, None], "date": [1, 2],
                  "value": [1., 2.]}),
])
def test_validates_winsor_parameters_before_empty_or_all_missing_shortcuts(frame):
    with pytest.raises(ValueError, match="winsor_std"):
        lagged_robust_ewma(frame, halflife=2.0, winsor_std=np.nan)


def test_validates_distinct_column_names_before_empty_shortcut():
    empty = pd.DataFrame({"asset_id": pd.Series(dtype="object"),
                          "date": pd.Series(dtype="int64"),
                          "value": pd.Series(dtype="float64")})
    with pytest.raises(ValueError, match="distinct"):
        lagged_robust_ewma(
            empty, halflife=2.0, time_col="asset_id",
        )
