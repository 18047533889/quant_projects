"""Independent contracts for the effective Polars extremal-index registration."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _oracle(values, window=18, side="upper", q=0.75, min_exceed=3, run_length=2):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    for row in range(len(values)):
        chunk = values[max(0, row - window + 1):row + 1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size < 3:
            continue
        threshold = np.quantile(valid, q if side == "upper" else 1 - q)
        count = clusters = gap = 0
        seen = False
        for value in chunk:
            if not np.isfinite(value):
                gap = run_length
                continue
            extreme = value > threshold if side == "upper" else value < threshold
            if extreme:
                count += 1
                if not seen or gap >= run_length:
                    clusters += 1
                seen, gap = True, 0
            else:
                gap += 1
        if count >= min_exceed:
            out[row] = clusters / count
    return out


def _fixture(time_name="timestamp"):
    rng = np.random.default_rng(4017)
    values = rng.normal(size=70)
    values[12:16] += 6
    values[31] = np.nan
    values[32:35] += 7
    values[51:53] -= 6
    increments = np.resize([1, 2, 5, 1], len(values))
    time = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.cumsum(increments), unit="D")
    return values, time, pl.DataFrame({time_name: time, "A": values})


@pytest.mark.parametrize("time_name", ["date", "timestamp"])
@pytest.mark.parametrize("side", ["upper", "lower"])
def test_extremal_index_matches_independent_runs_oracle(time_name, side):
    load_all()
    values, time, frame = _fixture(time_name)
    params = dict(window=18, side=side, q=0.75, min_exceed=3, run_length=2)
    actual = OperatorRegistry.get("ts_extremal_index", "polars", mode="any").calculate(frame, **params)
    assert actual[time_name].to_list() == frame[time_name].to_list()
    np.testing.assert_allclose(actual["A"].to_numpy(), _oracle(values, **params), equal_nan=True)


def test_extremal_index_defaults_match_pandas_and_explicit_values():
    load_all()
    values, time, frame = _fixture("date")
    pandas_frame = pd.DataFrame({"A": values}, index=time)
    pandas_op = OperatorRegistry.get("ts_extremal_index", "pandas_numpy", mode="any")
    polars_op = OperatorRegistry.get("ts_extremal_index", "polars", mode="any")
    pi = pandas_op.calculate(pandas_frame).to_numpy()
    li = polars_op.calculate(frame).select("A").to_numpy()
    le = polars_op.calculate(frame, window=120, side="upper", q=0.9, min_exceed=3, run_length=1).select("A").to_numpy()
    np.testing.assert_allclose(li, pi, equal_nan=True)
    np.testing.assert_allclose(li, le, equal_nan=True)


def test_extremal_index_nan_breaks_cluster_and_future_prefix_is_invariant():
    load_all()
    values, _, frame = _fixture("timestamp")
    params = dict(window=18, side="upper", q=0.75, min_exceed=3, run_length=4)
    op = OperatorRegistry.get("ts_extremal_index", "polars", mode="any")
    before = op.calculate(frame, **params)["A"].to_numpy()
    changed = frame.with_columns(
        pl.when(pl.int_range(pl.len()) >= 50).then(pl.col("A") * -17 + 9).otherwise(pl.col("A")).alias("A")
    )
    after = op.calculate(changed, **params)["A"].to_numpy()
    np.testing.assert_allclose(before[:50], after[:50], equal_nan=True)
    np.testing.assert_allclose(before, _oracle(values, **params), equal_nan=True)


def test_extremal_index_empty_and_parameter_rejection():
    load_all()
    op = OperatorRegistry.get("ts_extremal_index", "polars", mode="any")
    empty = pl.DataFrame(schema={"timestamp": pl.Datetime, "A": pl.Float64})
    assert op.calculate(empty).shape == (0, 2)
    _, _, frame = _fixture()
    for bad in ({"side": "middle"}, {"q": 1.0}, {"min_exceed": 1}, {"run_length": 0}):
        with pytest.raises((ValueError, TypeError)):
            op.calculate(frame, **bad)
