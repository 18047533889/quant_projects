# -*- coding: utf-8 -*-
"""R2-P0-021 direct-window regression tests for rolling central statistics."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

from factor_engine.cleaned_operators.common.time_series import TSKurtosisPolars, TSSkewnessPolars
from factor_engine.cleaned_operators.research_polars import TSMomentNativePolars

TSMomentNative = TSMomentNativePolars
TSKurtNative = TSKurtosisPolars
TSSkewNative = TSSkewnessPolars


def _windows(values: list[float], window: int):
    for end in range(len(values)):
        start = max(0, end - window + 1)
        raw = np.asarray(values[start : end + 1], dtype=float)
        yield raw[np.isfinite(raw)]


def _moment_oracle(values: list[float], window: int, order: int) -> np.ndarray:
    result = []
    for current in _windows(values, window):
        if current.size < window:
            result.append(np.nan)
            continue
        mean = sum(float(value) for value in current) / current.size
        result.append(sum((float(value) - mean) ** order for value in current) / current.size)
    return np.asarray(result)


def _skew_oracle(values: list[float], window: int) -> np.ndarray:
    result = []
    for current in _windows(values, window):
        count = current.size
        if count < window:
            result.append(np.nan)
            continue
        mean = sum(float(value) for value in current) / count
        centered = [float(value) - mean for value in current]
        sum_sq = sum(value * value for value in centered)
        if sum_sq == 0.0:
            result.append(np.nan)
            continue
        sum_cube = sum(value * value * value for value in centered)
        result.append(count * sum_cube / ((count - 1) * (count - 2) * (sum_sq / (count - 1)) ** 1.5))
    return np.asarray(result)


def _kurt_oracle(values: list[float], window: int) -> np.ndarray:
    result = []
    for current in _windows(values, window):
        count = current.size
        if count < window:
            result.append(np.nan)
            continue
        mean = sum(float(value) for value in current) / count
        centered = [float(value) - mean for value in current]
        second = sum(value * value for value in centered)
        if second == 0.0:
            result.append(np.nan)
            continue
        fourth = sum(value ** 4 for value in centered)
        biased = count * fourth / (second * second) - 3.0
        result.append((count - 1) / ((count - 2) * (count - 3)) * ((count + 1) * biased + 6.0))
    return np.asarray(result)


def _values(result: pl.DataFrame) -> np.ndarray:
    return np.asarray(result["x"].to_numpy(), dtype=float)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_ts_moment_uses_each_current_trailing_window(order: int) -> None:
    values = [2.0, 3.0, 8.0, 5.0, 13.0, 21.0, 1.0, 34.0]
    actual = _values(TSMomentNative().calculate(pl.DataFrame({"x": values}), 4, order))
    np.testing.assert_allclose(actual, _moment_oracle(values, 4, order), equal_nan=True)


@pytest.mark.parametrize(
    ("operator", "oracle", "window"),
    [(TSSkewNative(), _skew_oracle, 3), (TSKurtNative(), _kurt_oracle, 4)],
)
def test_standardized_moments_use_direct_window_oracle(operator, oracle, window: int) -> None:
    values = [1.0, 2.0, 9.0, 4.0, 15.0, 6.0, 30.0, 7.0]
    actual = _values(operator.calculate(pl.DataFrame({"x": values}), window))
    np.testing.assert_allclose(actual, oracle(values, window), rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    ("operator", "args", "oracle", "window"),
    [
        (TSMomentNative(), (4, 3), lambda values, window: _moment_oracle(values, window, 3), 4),
        (TSSkewNative(), (3,), _skew_oracle, 3),
        (TSKurtNative(), (4,), _kurt_oracle, 4),
    ],
)
def test_moments_anchor_rollout_nan_inf_and_zero_scale(operator, args, oracle, window: int) -> None:
    prefix = [1000.0, -2000.0, 500.0]
    anchor = [4.0, np.nan, 7.0, np.inf, 11.0, 13.0, 17.0, 19.0, 23.0]
    full = prefix + anchor
    full_actual = _values(operator.calculate(pl.DataFrame({"x": full}), *args))
    anchor_actual = _values(operator.calculate(pl.DataFrame({"x": anchor}), *args))
    np.testing.assert_allclose(full_actual[len(prefix) + window - 1 :], anchor_actual[window - 1 :], equal_nan=True)
    np.testing.assert_allclose(full_actual, oracle(full, window), rtol=1e-7, atol=1e-7, equal_nan=True)

    constant = _values(operator.calculate(pl.DataFrame({"x": [7.0] * (window + 2)}), *args))
    if operator.metadata.name == "ts_kurtosis":
        # The canonical pandas authority explicitly defines constant full
        # windows as -3.0; this is a compatibility assertion, separate from
        # the direct mathematical oracle above where zero scale is undefined.
        np.testing.assert_allclose(constant[window - 1 :], -3.0)
    elif operator.metadata.name == "ts_skewness":
        assert np.isnan(constant[window - 1 :]).all()
    else:
        np.testing.assert_allclose(constant[window - 1 :], 0.0)


def test_moment_parameter_semantics() -> None:
    with pytest.raises(ValueError, match="k"):
        TSMomentNative().calculate(pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), 4, 1)


def test_bootstrap_selects_only_repaired_moment_authorities() -> None:
    assert type(OperatorRegistry.get("ts_moment", backend="polars")) is TSMomentNativePolars
    assert type(OperatorRegistry.get("ts_kurt", backend="polars")) is TSKurtosisPolars
    assert type(OperatorRegistry.get("ts_skew", backend="polars")) is TSSkewnessPolars
