"""Dimensionless statistics must not overflow squared moment products."""
import numpy as np
import polars as pl
import pytest

from factor_engine.cleaned_operators.common.polars_ts_rolling import (
    TSCorrNative, TSRegressionR2Native, TSRegressionSlopeNative,
)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_correlation_is_invariant_to_finite_units(scale):
    x = np.array([1., 3., 2., 6., 4., 8.])
    y = np.array([4., 2., 7., 3., 9., 5.])
    actual = TSCorrNative()._calculate_series(
        pl.DataFrame({"A": x * scale}), pl.DataFrame({"A": y * scale}),
        window=4, min_periods=2,
    )["A"].to_numpy()
    expected = np.array([np.nan] + [
        np.corrcoef(x[max(0, end-3):end+1], y[max(0, end-3):end+1])[0, 1]
        for end in range(1, len(x))
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
@pytest.mark.parametrize("cls,extra", [(TSRegressionR2Native, {}),
                                      (TSRegressionSlopeNative, {"retval": "r2"})])
def test_intercept_regression_r2_is_invariant_to_finite_units(scale, cls, extra):
    x = np.array([1., 3., 2., 6., 4., 8.])
    y = np.array([4., 2., 7., 3., 9., 5.])
    actual = cls()._calculate_series(
        pl.DataFrame({"A": y * scale}), pl.DataFrame({"A": x * scale}),
        window=4, min_periods=2, add_intercept=True, **extra,
    )["A"].to_numpy()
    expected = np.array([np.nan] + [
        np.corrcoef(x[max(0, end-3):end+1], y[max(0, end-3):end+1])[0, 1] ** 2
        for end in range(1, len(x))
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
