"""Independent contracts for the effective Polars GPD-PWM registration."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _oracle(values, window=24, side="upper", tail_fraction=0.25, min_tail_count=4):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    for row in range(len(values)):
        chunk = values[max(0, row - window + 1):row + 1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size < min_tail_count + 2:
            continue
        threshold = np.quantile(valid, 1 - tail_fraction if side == "upper" else tail_fraction)
        exceedances = valid[valid > threshold] - threshold if side == "upper" else threshold - valid[valid < threshold]
        if exceedances.size < min_tail_count:
            continue
        ordered = np.sort(exceedances)
        count = ordered.size
        b0 = ordered.mean()
        b1 = np.sum(np.arange(count) / (count - 1) * ordered) / count
        denominator = 2 * b1 - b0
        if abs(denominator) <= np.finfo(float).eps:
            continue
        estimate = 2 - b0 / denominator
        if np.isfinite(estimate):
            out[row] = estimate
    return out


def _fixture(time_name="timestamp"):
    rng = np.random.default_rng(4103)
    values = rng.standard_t(df=3, size=96)
    values[[17, 38, 71]] = np.nan
    values[42:47] += 8
    time = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.cumsum(np.resize([1, 3, 2, 5], len(values))), unit="D")
    return values, time, pl.DataFrame({time_name: time, "A": values})


@pytest.mark.parametrize("time_name", ["date", "timestamp"])
@pytest.mark.parametrize("side", ["upper", "lower"])
def test_gpd_shape_pwm_matches_independent_oracle(time_name, side):
    load_all()
    values, _, frame = _fixture(time_name)
    params = dict(window=24, side=side, tail_fraction=0.25, min_tail_count=4)
    actual = OperatorRegistry.get("ts_gpd_shape_pwm", "polars", mode="any").calculate(frame, **params)
    assert actual[time_name].to_list() == frame[time_name].to_list()
    np.testing.assert_allclose(actual["A"].to_numpy(), _oracle(values, **params), equal_nan=True)


def test_gpd_shape_pwm_defaults_match_pandas_and_explicit_values():
    load_all()
    values, time, frame = _fixture("date")
    pandas_frame = pd.DataFrame({"A": values}, index=time)
    pandas_op = OperatorRegistry.get("ts_gpd_shape_pwm", "pandas_numpy", mode="any")
    polars_op = OperatorRegistry.get("ts_gpd_shape_pwm", "polars", mode="any")
    expected = pandas_op.calculate(pandas_frame).to_numpy()
    implicit = polars_op.calculate(frame).select("A").to_numpy()
    explicit = polars_op.calculate(frame, window=120, side="upper", tail_fraction=0.2, min_tail_count=10).select("A").to_numpy()
    np.testing.assert_allclose(implicit, expected, equal_nan=True)
    np.testing.assert_allclose(implicit, explicit, equal_nan=True)


def test_gpd_shape_pwm_nan_support_and_future_prefix_invariance():
    load_all()
    values, _, frame = _fixture()
    params = dict(window=24, side="upper", tail_fraction=0.25, min_tail_count=4)
    op = OperatorRegistry.get("ts_gpd_shape_pwm", "polars", mode="any")
    before = op.calculate(frame, **params)["A"].to_numpy()
    changed = frame.with_columns(pl.when(pl.int_range(pl.len()) >= 64).then(pl.col("A") * -11 + 4).otherwise(pl.col("A")).alias("A"))
    after = op.calculate(changed, **params)["A"].to_numpy()
    np.testing.assert_allclose(before[:64], after[:64], equal_nan=True)
    np.testing.assert_allclose(before, _oracle(values, **params), equal_nan=True)


def test_gpd_shape_pwm_empty_and_parameter_rejection():
    load_all()
    op = OperatorRegistry.get("ts_gpd_shape_pwm", "polars", mode="any")
    empty = pl.DataFrame(schema={"timestamp": pl.Datetime, "A": pl.Float64})
    assert op.calculate(empty).shape == (0, 2)
    _, _, frame = _fixture()
    for bad in ({"side": "both"}, {"tail_fraction": 0.0}, {"tail_fraction": 0.6}, {"min_tail_count": 2}, {"window": 1}):
        with pytest.raises((ValueError, TypeError)):
            op.calculate(frame, **bad)
