from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factor_engine.cleaned_operators.common.polars_ts_rolling import (
    TSRegressionInterceptNative,
    TSRegressionR2Native,
    TSRegressionResidNative,
    TSRegressionSlopeNative,
)

CLASSES = {
    "slope": TSRegressionSlopeNative,
    "intercept": TSRegressionInterceptNative,
    "resid": TSRegressionResidNative,
    "r2": TSRegressionR2Native,
}


def _oracle(x, y, window, min_periods, output, add_intercept):
    result = np.full(len(x), np.nan)
    for end in range(len(x)):
        start = max(0, end - window + 1)
        xx = np.asarray(x[start : end + 1], float)
        yy = np.asarray(y[start : end + 1], float)
        valid = np.isfinite(xx) & np.isfinite(yy)
        endpoint_valid = bool(valid[-1])
        xx, yy = xx[valid], yy[valid]
        if len(xx) < min_periods:
            continue
        if add_intercept:
            xc, yc = xx - xx.mean(), yy - yy.mean()
            ssx, ssy = np.dot(xc, xc), np.dot(yc, yc)
            if ssx <= 0:
                continue
            slope = np.dot(xc, yc) / ssx
            intercept = yy.mean() - slope * xx.mean()
            resid = yy[-1] - intercept - slope * xx[-1] if endpoint_valid else np.nan
            r2 = np.nan if ssy <= 0 else np.dot(xc, yc) ** 2 / (ssx * ssy)
        else:
            ssx, ssy = np.dot(xx, xx), np.dot(yy, yy)
            if ssx <= 0:
                continue
            slope = np.dot(xx, yy) / ssx
            intercept = 0.0
            resid = yy[-1] - slope * xx[-1] if endpoint_valid else np.nan
            sse = np.dot(yy - slope * xx, yy - slope * xx)
            r2 = np.nan if ssy <= 0 else 1 - sse / ssy
        result[end] = dict(slope=slope, intercept=intercept, resid=resid, r2=r2)[output]
    return result


@pytest.mark.parametrize("window", [20, 60])
@pytest.mark.parametrize("add_intercept", [True, False])
@pytest.mark.parametrize("output", ["slope", "intercept", "resid", "r2"])
def test_staged_regression_large_offset_finite_pair_oracle(window, add_intercept, output):
    rng = np.random.default_rng(2209)
    n = 96
    x = 1e9 + np.arange(n) * 0.01 + rng.normal(0, 0.02, n)
    y = -3e8 + 1.75 * x + rng.normal(0, 0.03, n)
    x[13], y[29], x[47] = np.nan, np.inf, -np.inf
    xf = pl.DataFrame({"A": x})
    yf = pl.DataFrame({"A": y})
    cls = CLASSES[output]
    actual = cls()._calculate_series(
        yf, xf, window=window, min_periods=5, add_intercept=add_intercept
    )["A"].to_numpy()
    expected = _oracle(x, y, window, 5, output, add_intercept)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6, equal_nan=True)


@pytest.mark.parametrize("retval", ["slope", "beta", "intercept", "resid", "residual", "r2", "r_squared"])
def test_slope_retval_branches_and_lag(retval):
    x = np.arange(80, dtype=float)
    y = 4.0 + 2.0 * np.roll(x, 1)
    y[0] = np.nan
    xf, yf = pl.DataFrame({"A": x}), pl.DataFrame({"A": y})
    out = TSRegressionSlopeNative()._calculate_series(
        yf, xf, 20, lag=1, retval=retval, min_periods=5, add_intercept=True
    )
    assert out.height == 80
    assert out["A"].is_not_null().sum() > 0


@pytest.mark.parametrize("cls", list(CLASSES.values()))
def test_staged_regression_preserves_meta_and_empty(cls):
    dates = [1, 2, 3]
    y = pl.DataFrame({"date": dates, "stock_code": ["A"] * 3, "__xw0": [1.0, 2.0, 3.0]})
    x = pl.DataFrame({"date": dates, "stock_code": ["A"] * 3, "__xw0": [2.0, 3.0, 5.0]})
    out = cls()._calculate_series(y, x, window=2, min_periods=2)
    assert out.columns == y.columns
    assert out["date"].to_list() == dates
    assert out["stock_code"].to_list() == ["A"] * 3
    empty = cls()._calculate_series(y.head(0), x.head(0), window=20, min_periods=3)
    assert empty.shape == y.head(0).shape
