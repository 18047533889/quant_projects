"""R2-P0-023 regression tests for the genuine Qn scale estimator."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

PANDAS_AUTHORITY = OperatorRegistry.get("ts_qn_scale", backend="pandas_numpy")
POLARS_AUTHORITY = OperatorRegistry.get("ts_qn_scale", backend="polars")


def _qn_oracle(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return math.nan
    h = n // 2 + 1
    k = h * (h - 1) // 2
    diffs = np.asarray(
        [abs(values[i] - values[j]) for i in range(n) for j in range(i + 1, n)]
    )
    corrections = {
        2: 0.399356,
        3: 0.99365,
        4: 0.51321,
        5: 0.84401,
        6: 0.61220,
        7: 0.85877,
        8: 0.66993,
        9: 0.87344,
        10: 0.72014,
        11: 0.88906,
        12: 0.75743,
    }
    if n in corrections:
        dn = corrections[n]
    elif n % 2:
        dn = 1 / (1 + 1.60188 / n - 2.1284 / n**2 - 5.172 / n**3)
    else:
        dn = 1 / (1 + 3.67561 / n + 1.9654 / n**2 + 6.987 / n**3 - 77 / n**4)
    return 2.21914446598508 * dn * np.partition(diffs, k - 1)[k - 1]


def test_bootstrap_selects_genuine_qn_authorities() -> None:
    assert type(PANDAS_AUTHORITY).__module__ == "factor_engine.cleaned_operators.gemini_v2_common"
    assert type(POLARS_AUTHORITY).__module__ == "factor_engine.cleaned_operators.gemini_v2_common"
    assert PANDAS_AUTHORITY.metadata.param_names == ["x", "window", "min_periods"]
    assert POLARS_AUTHORITY.metadata.param_names == ["x", "window", "min_periods"]
    assert OperatorRegistry.get("ts_qn_scale", backend="duckdb") is None


def test_qn_is_pairwise_order_statistic_not_iqr_normalization() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 100.0])
    expected = _qn_oracle(values)
    actual = PANDAS_AUTHORITY.calculate(
        pd.DataFrame({"x": values}), window=4, min_periods=4
    )["x"].iloc[-1]
    iqr_scale = (np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349
    assert np.isclose(actual, expected)
    assert not np.isclose(actual, iqr_scale)


def test_qn_pandas_and_polars_match_independent_window_oracle() -> None:
    values = np.asarray([1.0, 8.0, 2.0, 100.0, 3.0, 4.0, 5.0, 6.0])
    window = 5
    expected = np.asarray(
        [
            math.nan if i + 1 < window else _qn_oracle(values[i - window + 1 : i + 1])
            for i in range(values.size)
        ]
    )
    pandas_actual = PANDAS_AUTHORITY.calculate(
        pd.DataFrame({"x": values}), window=window, min_periods=window
    )["x"].to_numpy(dtype=float)
    polars_actual = POLARS_AUTHORITY.calculate(
        pl.DataFrame({"x": values}), window=window, min_periods=window
    )["x"].to_numpy()
    np.testing.assert_allclose(pandas_actual, expected, equal_nan=True)
    np.testing.assert_allclose(polars_actual, expected, equal_nan=True)


def test_qn_fails_closed_across_nonfinite_gaps_and_handles_negative_infinity() -> None:
    values = np.asarray([2.0, 2.0, np.nan, 2.0, np.inf, -np.inf, 2.0, 3.0])
    actual = PANDAS_AUTHORITY.calculate(
        pd.DataFrame({"x": values}), window=5, min_periods=2
    )["x"].to_numpy(dtype=float)
    expected = np.asarray(
        [
            math.nan,
            0.0,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            _qn_oracle(values[-2:]),
        ]
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_qn_prior_results_are_future_invariant() -> None:
    values = np.asarray([1.0, 8.0, 2.0, 100.0, 3.0, 4.0])
    first = PANDAS_AUTHORITY.calculate(
        pd.DataFrame({"x": values}), window=4, min_periods=4
    )["x"].to_numpy(dtype=float)
    extended = PANDAS_AUTHORITY.calculate(
        pd.DataFrame({"x": np.r_[values, 10_000.0]}), window=4, min_periods=4
    )["x"].to_numpy(dtype=float)
    np.testing.assert_array_equal(first, extended[: values.size])
