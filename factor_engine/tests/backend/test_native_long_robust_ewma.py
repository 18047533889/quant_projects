import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.native_long_robust_ewma import lagged_robust_ewma


def _pandas_oracle(frame, *, halflife, winsor_std=4.0, min_periods=1):
    width = winsor_std if winsor_std > 0 else 1e-12
    result = np.full(frame.shape[0], np.nan, dtype=np.float64)
    for positions in frame.groupby("asset_id", sort=False, dropna=True).indices.values():
        lagged = pd.Series(frame["value"].to_numpy(dtype=float)[positions]).shift(1)
        finite = lagged.where(np.isfinite(lagged))
        mean = finite.rolling(window=10, min_periods=2).mean()
        std = finite.rolling(window=10, min_periods=2).std(ddof=1).clip(lower=1e-12)
        clipped = lagged.clip(lower=mean - width * std, upper=mean + width * std)
        clipped = clipped.where(np.isfinite(clipped))
        result[positions] = clipped.ewm(
            halflife=halflife,
            min_periods=min_periods,
            adjust=False,
            ignore_na=False,
        ).mean().to_numpy()
    return result


def _fixture():
    return pd.DataFrame({
        "asset_id": ["a", "b", "a", None, "b", "a", "b", "a", "b",
                     "a", "b", "a", "b", "a", "b", "a", "b"],
        # Interleaved monotone groups, duplicate timestamps, and a null-key row.
        "date": [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        "value": [10., 3., 12., 999., np.nan, 13., np.inf, 50., -np.inf,
                  12., 4., np.nan, 5., 14., 6., 15., 7.],
        "row_tag": list("abcdefghijklmnopq"),
    })


@pytest.mark.parametrize("min_periods", [0, 1, 3])
def test_matches_pandas_clip_and_ewm_for_interleaved_assets(min_periods):
    pandas_frame = _fixture()
    source = pl.from_pandas(pandas_frame)
    original = source.clone()
    actual = lagged_robust_ewma(
        source, halflife=2.5, winsor_std=1.5, min_periods=min_periods,
    )
    expected = _pandas_oracle(
        pandas_frame, halflife=2.5, winsor_std=1.5, min_periods=min_periods,
    )
    np.testing.assert_allclose(
        actual["lagged_robust_ewma"].to_numpy(), expected,
        rtol=2e-13, atol=1e-13, equal_nan=True,
    )
    assert actual.height == source.height
    assert actual.drop("lagged_robust_ewma").equals(original)
    assert actual["row_tag"].to_list() == pandas_frame["row_tag"].tolist()


def test_matches_pandas_at_large_but_well_scaled_magnitudes():
    values = [1e150, 2e150, 3e150, 4e150, 9e150, 5e150, 6e150, 7e150,
              8e150, 9e150, 1e151, 1.1e151]
    frame = pd.DataFrame({"asset_id": ["x"] * len(values),
                          "date": list(range(len(values))), "value": values})
    actual = lagged_robust_ewma(
        pl.from_pandas(frame), halflife=5.0, winsor_std=1.25,
    )["lagged_robust_ewma"].to_numpy()
    expected = _pandas_oracle(frame, halflife=5.0, winsor_std=1.25)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0, equal_nan=True)


def test_prefix_causality_and_nonpositive_winsor_compatibility():
    frame = pl.from_pandas(_fixture())
    actual = lagged_robust_ewma(frame, halflife=3.0, winsor_std=0.0)
    negative = lagged_robust_ewma(frame, halflife=3.0, winsor_std=-2.0)
    np.testing.assert_allclose(
        actual["lagged_robust_ewma"].to_numpy(),
        negative["lagged_robust_ewma"].to_numpy(), equal_nan=True,
    )
    for stop in range(1, frame.height + 1):
        prefix = lagged_robust_ewma(frame.head(stop), halflife=3.0, winsor_std=0.0)
        np.testing.assert_allclose(
            prefix["lagged_robust_ewma"].to_numpy(),
            actual["lagged_robust_ewma"].head(stop).to_numpy(),
            rtol=2e-13, atol=1e-13, equal_nan=True,
        )


@pytest.mark.parametrize("kwargs", [
    {"halflife": 0.0}, {"halflife": np.nan}, {"halflife": np.inf},
    {"halflife": True}, {"halflife": 2.0, "winsor_std": np.nan},
    {"halflife": 2.0, "winsor_std": np.inf},
    {"halflife": 2.0, "min_periods": -1},
    {"halflife": 2.0, "min_periods": 1.5},
    {"halflife": 2.0, "min_periods": True},
])
def test_rejects_invalid_parameters(kwargs):
    frame = pl.DataFrame({"asset_id": ["a"], "date": [1], "value": [1.]})
    with pytest.raises(ValueError):
        lagged_robust_ewma(frame, **kwargs)


def test_rejects_bad_group_order_and_missing_identity_columns():
    unordered = pl.DataFrame({"asset_id": ["a", "a"], "date": [2, 1],
                              "value": [1., 2.]})
    with pytest.raises(ValueError, match="monotone"):
        lagged_robust_ewma(unordered, halflife=2.0)
    missing = pl.DataFrame({"asset_id": ["a"], "date": [1], "other": [1.]})
    with pytest.raises(ValueError, match="missing columns"):
        lagged_robust_ewma(missing, halflife=2.0)


def test_empty_and_all_null_assets_preserve_rows_with_null_output():
    empty = pl.DataFrame({"asset_id": pl.Series([], dtype=pl.String),
                          "date": pl.Series([], dtype=pl.Int64),
                          "value": pl.Series([], dtype=pl.Float64)})
    assert lagged_robust_ewma(empty, halflife=2.0).height == 0
    null_keys = pl.DataFrame({"asset_id": [None, None], "date": [1, 2],
                              "value": [1., 2.]})
    output = lagged_robust_ewma(null_keys, halflife=2.0)
    assert output.height == null_keys.height
    assert output["lagged_robust_ewma"].null_count() == null_keys.height


def test_float_nan_asset_keys_follow_pandas_dropna_grouping():
    frame = pl.DataFrame({
        "asset_id": [float("nan"), float("nan"), 4.0],
        "date": [1, 2, 1],
        "value": [10.0, 12.0, 3.0],
    })
    output = lagged_robust_ewma(frame, halflife=2.0)
    values = output["lagged_robust_ewma"].to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None


def test_rejects_non_numeric_value_schema_instead_of_nulling_bad_values():
    frame = pl.DataFrame({"asset_id": ["a", "a"], "date": [1, 2],
                          "value": ["1.0", "not-a-number"]})
    with pytest.raises(TypeError, match="numeric Polars dtype"):
        lagged_robust_ewma(frame, halflife=2.0)


def test_large_offset_small_variation_matches_pandas_clip_decisions():
    values = [1e12 + i * 1e-3 for i in (0, 1, 2, 3, 4, 5, 20, 7, 8, 9, 10, 11)]
    frame = pd.DataFrame({"asset_id": ["x"] * len(values),
                          "date": list(range(len(values))), "value": values})
    actual = lagged_robust_ewma(
        pl.from_pandas(frame), halflife=3.0, winsor_std=1.0,
    )["lagged_robust_ewma"].to_numpy()
    expected = _pandas_oracle(frame, halflife=3.0, winsor_std=1.0)
    base = 1e12
    centered_mask = np.isfinite(actual) & np.isfinite(expected)
    np.testing.assert_allclose(
        actual[centered_mask] - base,
        expected[centered_mask] - base,
        rtol=0.0, atol=2.0 * np.spacing(base),
    )

    lagged = pd.Series(values, dtype=float).shift(1)
    finite = lagged.where(np.isfinite(lagged))
    pandas_mean = finite.rolling(10, min_periods=2).mean().to_numpy()
    pandas_std = finite.rolling(10, min_periods=2).std(ddof=1).clip(lower=1e-12).to_numpy()
    native_lag = pl.Series("_lag", values, dtype=pl.Float64).shift(1)
    stats = (
        pl.DataFrame({"_lag": native_lag})
        .with_columns(
            pl.when(pl.col("_lag").is_finite())
            .then(pl.col("_lag")).otherwise(None).alias("_finite")
        )
        .with_columns(
            pl.col("_finite").rolling_mean(window_size=10, min_samples=2).alias("_mean"),
            pl.col("_finite").rolling_std(
                window_size=10, min_samples=2, ddof=1,
            ).clip(lower_bound=1e-12).alias("_std"),
        )
    )
    native_mean = stats["_mean"].to_numpy()
    native_std = stats["_std"].to_numpy()
    mean_mask = np.isfinite(pandas_mean) & np.isfinite(native_mean)
    std_mask = np.isfinite(pandas_std) & np.isfinite(native_std)
    np.testing.assert_allclose(
        native_mean[mean_mask] - base, pandas_mean[mean_mask] - base,
        rtol=0.0, atol=2.0 * np.spacing(base),
    )
    np.testing.assert_allclose(
        native_std[std_mask], pandas_std[std_mask], rtol=0.0, atol=2e-15,
    )
    pandas_lag = lagged.to_numpy()
    native_lag_values = native_lag.to_numpy()
    pandas_clip = (
        np.isfinite(pandas_lag) & np.isfinite(pandas_mean) & np.isfinite(pandas_std)
        & ((pandas_lag < pandas_mean - pandas_std)
           | (pandas_lag > pandas_mean + pandas_std))
    )
    native_clip = (
        np.isfinite(native_lag_values) & np.isfinite(native_mean) & np.isfinite(native_std)
        & ((native_lag_values < native_mean - native_std)
           | (native_lag_values > native_mean + native_std))
    )
    np.testing.assert_array_equal(native_clip, pandas_clip)
