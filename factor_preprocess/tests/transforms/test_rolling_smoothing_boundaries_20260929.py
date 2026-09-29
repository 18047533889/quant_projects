"""Boundary regressions for vectorized rolling and smoothing transforms."""

import numpy as np
import pandas as pd
import pytest

from factor_preprocess.transforms import rolling as R
from factor_preprocess.transforms import smoothing as S


TRANSFORMS = [
    (R.rolling_mean, {"window": 2, "min_periods": 1}),
    (R.rolling_std, {"window": 2, "min_periods": 1}),
    (R.rolling_zscore, {"window": 2, "min_periods": 1}),
    (R.ewma, {"halflife": 2, "min_periods": 1}),
    (S.trailing_sma, {"window": 2, "min_periods": 1}),
    (S.trailing_median, {"window": 2, "min_periods": 1}),
    (S.robust_ewma, {"halflife": 2, "min_periods": 1}),
]


def _panel(asset_kind):
    if asset_kind == "object":
        assets = ["a", "a", "a", "b", "b", "b", None, None]
    elif asset_kind == "nullable_int":
        assets = pd.array([1, 1, 1, 2, 2, 2, pd.NA, pd.NA], dtype="Int64")
    else:
        assets = pd.Categorical(
            ["a", "a", "a", "b", "b", "b", None, None],
            categories=["a", "b", "unused"],
        )
    return pd.DataFrame(
        {
            "asset_id": assets,
            "date": [1, 2, 3, 1, 2, 3, 1, 2],
            "value": [1., 3., 5., 10., 14., 18., 100., 200.],
        },
        index=[7, 7, 2, 2, 7, 1, 1, 7],
    )


@pytest.mark.parametrize("name,transform,kwargs", [
    (fn.__name__, fn, kwargs) for fn, kwargs in TRANSFORMS
])
def test_asset_id_null_and_dtype_edges_are_isolated(name, transform, kwargs):
    object_frame = _panel("object")
    expected = transform(object_frame, **kwargs).to_numpy()

    for kind in ("nullable_int", "categorical"):
        frame = _panel(kind)
        actual = transform(frame, **kwargs)
        assert actual.index.equals(frame.index), name
        np.testing.assert_array_equal(actual.to_numpy(), expected, err_msg=f"{name}: {kind}")
        np.testing.assert_array_equal(actual.iloc[-2:].to_numpy(), [np.nan, np.nan])


@pytest.mark.parametrize("transform,kwargs", [
    (R.rolling_mean, {"window": 2}),
    (R.rolling_std, {"window": 2}),
    (R.rolling_zscore, {"window": 2}),
    (S.trailing_sma, {"window": 2}),
    (S.trailing_median, {"window": 2}),
])
def test_rolling_min_periods_zero_and_window_boundaries(transform, kwargs):
    frame = _panel("object").iloc[:6].copy()
    zero = transform(frame, min_periods=0, **kwargs)
    one = transform(frame, min_periods=1, **kwargs)
    np.testing.assert_array_equal(zero.to_numpy(), one.to_numpy())
    transform(frame, min_periods=2, **kwargs)
    with pytest.raises(ValueError):
        transform(frame, min_periods=3, **kwargs)
    with pytest.raises(ValueError):
        transform(frame, min_periods=-1, **kwargs)



@pytest.mark.parametrize("transform,kwargs", [
    (R.ewma, {"halflife": 2}),
    (S.robust_ewma, {"halflife": 2}),
])
def test_ewma_min_periods_zero_and_oversized(transform, kwargs):
    frame = _panel("object").iloc[:6].copy()
    zero = transform(frame, min_periods=0, **kwargs)
    one = transform(frame, min_periods=1, **kwargs)
    np.testing.assert_array_equal(zero.to_numpy(), one.to_numpy())
    assert transform(frame, min_periods=len(frame) + 1, **kwargs).isna().all()


@pytest.mark.parametrize("transform,kwargs", [
    (R.ewma, {"halflife": 2}), (S.robust_ewma, {"halflife": 2})
])
def test_ewma_rejects_negative_min_periods(transform, kwargs):
    with pytest.raises(ValueError):
        transform(_panel("object"), min_periods=-1, **kwargs)



@pytest.mark.parametrize("transform,kwargs", [
    (R.ewma, {"halflife": 2}), (S.robust_ewma, {"halflife": 2})
])
@pytest.mark.parametrize("invalid", [1.5, "1", True])
def test_ewma_min_periods_requires_integer(transform, kwargs, invalid):
    with pytest.raises(ValueError, match="min_periods must be a non-negative integer"):
        transform(_panel("object"), min_periods=invalid, **kwargs)


@pytest.mark.parametrize("transform,kwargs", [
    (R.ewma, {"halflife": 2}), (S.robust_ewma, {"halflife": 2})
])
def test_ewma_accepts_numpy_integer_min_periods(transform, kwargs):
    result = transform(_panel("object"), min_periods=np.int64(1), **kwargs)
    assert result.notna().any()
