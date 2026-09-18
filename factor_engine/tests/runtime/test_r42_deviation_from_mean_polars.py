"""Contracts for the Polars-only rolling deviation operator."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _oracle(values, window):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    for row in range(window - 1, len(values)):
        out[row] = abs(values[row] - np.mean(values[row - window + 1:row + 1]))
    return out


def _fixture(time_name="timestamp"):
    rng = np.random.default_rng(4207)
    values = rng.normal(size=72).cumsum()
    time = pd.Timestamp("2025-02-01") + pd.to_timedelta(np.arange(len(values)), unit="D")
    return values, pl.DataFrame({time_name: time, "A": values, "B": values * -0.5 + 2})


@pytest.mark.parametrize("time_name", ["date", "timestamp"])
def test_deviation_from_mean_matches_independent_oracle(time_name):
    load_all()
    values, frame = _fixture(time_name)
    actual = OperatorRegistry.get("ts_deviation_from_mean", "polars", mode="any").calculate(frame, window=20)
    assert actual[time_name].to_list() == frame[time_name].to_list()
    np.testing.assert_allclose(actual["A"].to_numpy(), _oracle(values, 20), equal_nan=True)
    np.testing.assert_allclose(actual["B"].to_numpy(), _oracle(values * -0.5 + 2, 20), equal_nan=True)


def test_deviation_from_mean_nan_and_future_prefix_contracts():
    load_all()
    values, frame = _fixture()
    values[31] = np.nan
    frame = frame.with_columns(pl.Series("A", values))
    op = OperatorRegistry.get("ts_deviation_from_mean", "polars", mode="any")
    before = op.calculate(frame, window=20)["A"].to_numpy()
    changed = frame.with_columns(pl.when(pl.int_range(pl.len()) >= 50).then(pl.col("A") * 7 - 3).otherwise(pl.col("A")).alias("A"))
    after = op.calculate(changed, window=20)["A"].to_numpy()
    np.testing.assert_allclose(before[:50], after[:50], equal_nan=True)
    np.testing.assert_allclose(before, _oracle(values, 20), equal_nan=True)


def test_deviation_from_mean_empty_validation_and_backend_boundary():
    load_all()
    op = OperatorRegistry.get("ts_deviation_from_mean", "polars", mode="any")
    empty = pl.DataFrame(schema={"date": pl.Datetime, "A": pl.Float64})
    assert op.calculate(empty, window=20).shape == (0, 2)
    _, frame = _fixture()
    with pytest.raises((ValueError, TypeError)):
        op.calculate(frame, window=19)
    with pytest.raises(TypeError, match="polars.DataFrame"):
        op.calculate(pd.DataFrame({"A": np.arange(30.0)}), window=20)
