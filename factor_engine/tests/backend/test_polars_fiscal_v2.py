# -*- coding: utf-8 -*-
"""Parity tests for fiscal-signal, confirmed-pivot and component-score Polars."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(7)
    index = pd.date_range("2022-01-01", periods=90, freq="D")
    x = pd.DataFrame(np.nan, index=index, columns=["A", "B"], dtype=float)
    pid = pd.DataFrame(np.nan, index=index, columns=["A", "B"], dtype=object)
    for col in ["A", "B"]:
        for k in range(12):
            start = k * 7
            if start >= 90:
                break
            end = min(90, start + 7)
            pid.iloc[start:end, x.columns.get_loc(col)] = f"2022Q{k % 4 + 1}"
            x.iloc[start:end, x.columns.get_loc(col)] = rng.choice([-1.0, 0.0, 1.0, 2.0])
    high = pd.DataFrame(np.cumsum(rng.normal(0, 1, (90, 2)), axis=0) + 100, index=index, columns=["A", "B"])
    low = high - rng.uniform(0.5, 2, (90, 2))
    comp = pd.DataFrame({"c1": [1.0, -1.0, 0.0, np.nan, 2.0, -3.0] * 15, "c2": [0.0, 1.0, -1.0, 2.0, np.nan, 1.0] * 15})
    return x, pid, high, low, comp


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    data = {}
    for column in frame.columns:
        if frame[column].dtype == object:
            values = frame[column].tolist()
            cleaned = [
                None
                if value is None or (isinstance(value, float) and np.isnan(value))
                else value
                for value in values
            ]
            data[column] = pl.Series(cleaned, dtype=pl.Object) if any(not isinstance(v, str) for v in cleaned) else cleaned
        else:
            data[column] = frame[column].to_numpy()
    return pl.DataFrame(data)


def _assert_parity(name, args, kwargs, rtol=1e-8, atol=1e-8):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None, f"{name} missing pandas"
    assert polars_op is not None, f"{name} missing polars"
    pandas_out = pandas_op.calculate(*args, **kwargs)
    polars_out = polars_op.calculate(*[_polars(arg) for arg in args], **kwargs)
    assert list(pandas_out.columns) == list(polars_out.columns)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_native_polars_fiscal_v2_matches_pandas(panels):
    x, pid, high, low, comp = panels
    cases = [
        ("fiscal_true_streak", (x, pid), {"require_consecutive": True}),
        ("fiscal_true_streak", (x, pid), {"require_consecutive": False}),
        ("fiscal_sign_consistency", (x, pid), {"periods": 8, "min_periods": 3}),
        ("fiscal_reversal_ratio", (x, pid), {"periods": 8, "min_pairs": 3}),
        ("fiscal_autocorr", (x, pid), {"periods": 12, "lag": 1, "min_pairs": 3}),
        ("fiscal_standardized_surprise", (x, pid), {"seasonal_lag": 4, "lookback_periods": 8, "min_history": 4}),
        ("fiscal_sign_agreement", (x, x, pid), {"periods": 8, "min_periods": 3}),
        ("fiscal_change_direction_agreement", (x, x, pid), {"periods": 8, "min_periods": 3}),
        ("ts_confirmed_pivot_high", (high,), {"left_window": 3, "right_window": 3}),
        ("ts_confirmed_pivot_low", (low,), {"left_window": 3, "right_window": 3}),
        ("fin_component_score", (comp,), {}),
        ("fin_component_score", (comp,), {"component_directions": ["up", "down"], "score_weights": [1.0, 0.5]}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
