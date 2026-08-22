# -*- coding: utf-8 -*-
"""R2-P0-022 direct-window trimmed-mean regression tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

load_all()


PANDAS_AUTHORITY = OperatorRegistry.get("ts_trimmed_mean", backend="pandas_numpy")
POLARS_AUTHORITY = OperatorRegistry.get("ts_trimmed_mean", backend="polars")


def _window_oracle(values: list[float], window: int, trim_ratio: float, min_periods: int) -> np.ndarray:
    result: list[float] = []
    for end in range(len(values)):
        current = np.asarray(values[max(0, end - window + 1) : end + 1], dtype=float)
        current = current[np.isfinite(current)]
        if current.size < min_periods:
            result.append(math.nan)
            continue
        ordered = np.sort(current)
        cut = int(np.floor(trim_ratio * ordered.size))
        if cut * 2 >= ordered.size:
            result.append(math.nan)
            continue
        result.append(float(np.mean(ordered[cut : ordered.size - cut])))
    return np.asarray(result)


def _polars_values(result: pl.DataFrame) -> np.ndarray:
    return np.asarray(result["x"].to_numpy(), dtype=float)


def test_bootstrap_selects_direct_window_authorities() -> None:
    assert type(PANDAS_AUTHORITY).__module__ == "cleaned_operators.robust_stats"
    assert type(PANDAS_AUTHORITY).__name__ == "TsTrimmedMean"
    assert type(POLARS_AUTHORITY).__module__ == "cleaned_operators.common.polars_robust_stats"
    assert type(POLARS_AUTHORITY).__name__ == "PolarsRobustStats_ts_trimmed_mean"
    assert OperatorRegistry.get("ts_trimmed_mean", backend="duckdb") is None


def test_trimmed_mean_deletes_tails_in_each_current_window() -> None:
    values = [0.0, 1.0, 2.0, 3.0, 100.0, 4.0]
    expected = _window_oracle(values, window=5, trim_ratio=0.2, min_periods=5)
    pandas_actual = PANDAS_AUTHORITY.calculate(
        pd.DataFrame({"x": values}), window=5, trim_ratio=0.2, min_periods=5
    )["x"].to_numpy(dtype=float)
    polars_actual = _polars_values(
        POLARS_AUTHORITY.calculate(
            pl.DataFrame({"x": values}), window=5, trim_ratio=0.2, min_periods=5
        )
    )
    np.testing.assert_allclose(pandas_actual, expected, equal_nan=True)
    np.testing.assert_allclose(polars_actual, expected, equal_nan=True)

    # Percentile clipping (winsorization) keeps five observations and therefore
    # differs from deleting one observation from each tail.
    clipped = np.clip(
        np.asarray(values[:5]),
        np.quantile(values[:5], 0.2),
        np.quantile(values[:5], 0.8),
    )
    assert expected[4] == 2.0
    np.testing.assert_allclose(float(np.mean(clipped)), 5.84)
    assert float(np.mean(clipped)) != expected[4]


def test_trimmed_mean_filters_nan_inf_and_respects_min_periods() -> None:
    values = [1.0, np.nan, 4.0, np.inf, 2.0, 3.0, 5.0]
    expected = _window_oracle(values, window=5, trim_ratio=0.25, min_periods=3)
    pandas_actual = PANDAS_AUTHORITY.calculate(
        pd.DataFrame({"x": values}), window=5, trim_ratio=0.25, min_periods=3
    )["x"].to_numpy(dtype=float)
    polars_actual = _polars_values(
        POLARS_AUTHORITY.calculate(
            pl.DataFrame({"x": values}), window=5, trim_ratio=0.25, min_periods=3
        )
    )
    np.testing.assert_allclose(pandas_actual, expected, equal_nan=True)
    np.testing.assert_allclose(polars_actual, expected, equal_nan=True)
    assert np.isnan(expected[:4]).all()
    np.testing.assert_allclose(expected[4:], [7.0 / 3.0, 3.0, 3.5])


@pytest.mark.parametrize("trim_ratio", [-0.01, 0.5, np.nan, np.inf])
def test_trimmed_mean_rejects_invalid_trim_ratio(trim_ratio: float) -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ValueError, match="trim"):
        POLARS_AUTHORITY.calculate(frame, window=5, trim_ratio=trim_ratio, min_periods=5)
