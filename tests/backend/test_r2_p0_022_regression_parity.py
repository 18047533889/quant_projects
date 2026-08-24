"""R2-P0-022: selectable Polars rolling regression parity."""
from __future__ import annotations

import math

import pytest
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.operator_errors import FutureReferenceError
from factor_engine.cleaned_operators.common.polars_ts_rolling import (
    TSRegressionInterceptNative,
    TSRegressionR2Native,
    TSRegressionResidNative,
    TSRegressionSlopeNative,
)


def values(frame):
    return frame["a"].to_list()


def test_registered_regression_no_intercept_matches_origin_oracle() -> None:
    load_all()
    x = pl.DataFrame({"a": [1.0, 2.0]})
    y = pl.DataFrame({"a": [3.0, 5.0]})
    expected = {
        "ts_regression_slope": [None, 2.6],
        "ts_regression_intercept": [None, 0.0],
        "ts_regression_resid": [None, -0.2],
    }
    for canonical, values_expected in expected.items():
        operator = OperatorRegistry.get(canonical, "polars", mode="research")
        actual = operator.calculate(
            y, x, window=2, min_periods=2, add_intercept=False
        )["a"].to_list()
        for got, want in zip(actual, values_expected):
            if want is None:
                assert got is None or (isinstance(got, float) and math.isnan(got))
            else:
                assert got == pytest.approx(want)


def test_regression_default_min_periods_is_three():
    y = pl.DataFrame({"a": [1.0, 3.0]})
    x = pl.DataFrame({"a": [1.0, 2.0]})
    assert values(TSRegressionSlopeNative()._calculate_series(y, x, window=3)) == [None, None]


def test_regression_rejects_invalid_min_periods():
    y = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    x = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="min_periods must be <= window"):
        TSRegressionSlopeNative()._calculate_series(y, x, window=2, min_periods=3)


def test_through_origin_regression_outputs_match_formula():
    x = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pl.DataFrame({"a": [2.0, 2.0, 6.0]})
    assert values(TSRegressionSlopeNative()._calculate_series(y, x, window=3, min_periods=2, add_intercept=False))[-1] == pytest.approx(24 / 14)
    assert values(TSRegressionInterceptNative()._calculate_series(y, x, window=3, min_periods=2, add_intercept=False))[-1] == 0.0
    assert values(TSRegressionResidNative()._calculate_series(y, x, window=3, min_periods=2, add_intercept=False))[-1] == pytest.approx(6 - (24 / 14) * 3)
    assert values(TSRegressionR2Native()._calculate_series(y, x, window=3, min_periods=2, add_intercept=False))[-1] == pytest.approx(0.935064935064935)


def test_slope_legacy_lag_and_retval_and_future_guard():
    x = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pl.DataFrame({"a": [2.0, 4.0, 6.0, 8.0]})
    result = TSRegressionSlopeNative()._calculate_series(y, x, 3, 1, "intercept", 2)
    assert values(result)[-1] == pytest.approx(2.0)
    with pytest.raises(FutureReferenceError):
        TSRegressionSlopeNative()._calculate_series(y, x, 3, -1)
