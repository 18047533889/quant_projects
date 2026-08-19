"""POL2-P0-004 rolling moment/kurtosis parity regressions."""
from __future__ import annotations

import numpy as np
import polars as pl

from cleaned_operators.common.time_series import TSKurtosisPolars
from cleaned_operators.research_polars import TSMomentNativePolars


def _rolling(values: list[float], window: int) -> list[np.ndarray]:
    return [np.asarray(values[max(0, i - window + 1) : i + 1], dtype=float) for i in range(len(values))]


def _central_moment(values: list[float], window: int, order: int) -> np.ndarray:
    result = []
    for current in _rolling(values, window):
        if current.size < window:
            result.append(np.nan)
            continue
        mean = float(np.mean(current))
        result.append(float(np.mean((current - mean) ** order)))
    return np.asarray(result)


def _unbiased_excess_kurtosis(values: list[float], window: int) -> np.ndarray:
    result = []
    for current in _rolling(values, window):
        if current.size < window:
            result.append(np.nan)
            continue
        count = current.size
        centered = current - float(np.mean(current))
        second = float(np.sum(centered**2))
        if second == 0.0:
            result.append(-3.0)
            continue
        biased_excess = count * float(np.sum(centered**4)) / (second * second) - 3.0
        result.append(
            (count - 1.0) / ((count - 2.0) * (count - 3.0))
            * ((count + 1.0) * biased_excess + 6.0)
        )
    return np.asarray(result)


def test_ts_moment_uses_current_trailing_mean_not_expanding_mean() -> None:
    values = [0.0, 1.0, 2.0, 10.0]
    actual = TSMomentNativePolars().calculate(pl.DataFrame({"x": values}), 3, 3)["x"].to_numpy()

    expected = _central_moment(values, 3, 3)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    # The final rolling window is [1, 2, 10], not the global/expanding window.
    np.testing.assert_allclose(actual[-1], expected[-1], rtol=1e-12, atol=1e-12)
    assert not np.isclose(actual[-1], np.mean((np.asarray(values) - np.mean(values)) ** 3))


def test_ts_kurt_uses_current_trailing_mean_not_global_mean() -> None:
    values = [0.0, 1.0, 2.0, 10.0, 20.0]
    actual = TSKurtosisPolars().calculate(pl.DataFrame({"x": values}), 4)["x"].to_numpy()

    expected = _unbiased_excess_kurtosis(values, 4)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    rolling = np.asarray(values[-4:], dtype=float)
    global_values = np.asarray(values, dtype=float)
    assert not np.isclose(
        actual[-1],
        _unbiased_excess_kurtosis(global_values.tolist(), len(global_values))[-1],
    )
    assert np.isfinite(rolling.mean())
