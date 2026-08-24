"""R2-P0-024 direct-window Expected Shortfall parity coverage."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

PANDAS = OperatorRegistry.get("ts_expected_shortfall", backend="pandas_numpy")
POLARS = OperatorRegistry.get("ts_expected_shortfall", backend="polars")


def _es_oracle(values: np.ndarray, q: float, side: str, min_tail_count: int) -> float:
    valid = values[np.isfinite(values)]
    minimum = max(2, int(min_tail_count))
    if valid.size < minimum:
        return math.nan
    threshold = np.quantile(valid, q if side == "lower" else 1.0 - q)
    tail = valid[valid <= threshold] if side == "lower" else valid[valid >= threshold]
    return float(np.mean(tail)) if tail.size >= minimum else math.nan


def _rolling_oracle(values: np.ndarray, window: int, q: float, side: str, min_tail_count: int) -> np.ndarray:
    return np.asarray([
        _es_oracle(values[max(0, i - window + 1) : i + 1], q, side, min_tail_count)
        for i in range(values.size)
    ])


def test_selected_authorities_have_canonical_expected_shortfall_abi() -> None:
    assert PANDAS is not None
    assert POLARS is not None
    assert PANDAS.metadata.param_names == ["x", "window", "q", "side", "min_tail_count"]
    assert POLARS.metadata.param_names == ["x", "window", "q", "side", "min_tail_count"]
    assert type(POLARS).__module__ == "factor_engine.cleaned_operators.rolling_pack"


def test_expected_shortfall_selected_backends_match_direct_window_oracle() -> None:
    values = np.asarray([-10.0, -2.0, 0.0, 1.0, 2.0, 100.0, 3.0, np.nan, 4.0, np.inf])
    kwargs = {"window": 6, "q": 0.25, "side": "lower", "min_tail_count": 2}
    expected = _rolling_oracle(values, **kwargs)
    pandas_actual = PANDAS.calculate(pd.DataFrame({"x": values}), **kwargs)["x"].to_numpy()
    polars_actual = POLARS.calculate(pl.DataFrame({"x": values}), **kwargs)["x"].to_numpy()
    np.testing.assert_allclose(pandas_actual, expected, equal_nan=True)
    np.testing.assert_allclose(polars_actual, expected, equal_nan=True)


def test_expected_shortfall_tail_count_is_not_window_size() -> None:
    values = np.arange(1.0, 11.0)
    kwargs = {"window": 10, "q": 0.25, "side": "lower", "min_tail_count": 5}
    pandas_actual = PANDAS.calculate(pd.DataFrame({"x": values}), **kwargs)["x"].iloc[-1]
    polars_actual = POLARS.calculate(pl.DataFrame({"x": values}), **kwargs)["x"].to_numpy()[-1]
    assert np.isnan(pandas_actual)
    assert np.isnan(polars_actual)


def test_expected_shortfall_upper_tail_and_future_invariance() -> None:
    values = np.asarray([-4.0, -1.0, 0.0, 2.0, 3.0, 9.0, 10.0])
    kwargs = {"window": 5, "q": 0.4, "side": "upper", "min_tail_count": 2}
    first = POLARS.calculate(pl.DataFrame({"x": values}), **kwargs)["x"].to_numpy()
    extended = POLARS.calculate(pl.DataFrame({"x": np.r_[values, 10000.0]}), **kwargs)["x"].to_numpy()
    expected = _rolling_oracle(values, **kwargs)
    np.testing.assert_allclose(first, expected, equal_nan=True)
    np.testing.assert_array_equal(first, extended[: values.size])
