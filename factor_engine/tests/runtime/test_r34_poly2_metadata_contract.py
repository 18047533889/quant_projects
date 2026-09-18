"""Metadata and causality contracts for effective robust-statistics Poly2 kernels."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _coeff_oracle(values, d):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    bars = np.arange(d, dtype=float)
    for row in range(d - 1, len(values)):
        window = values[row - d + 1:row + 1]
        valid = np.isfinite(window)
        if valid.sum() < 3:
            continue
        design = np.column_stack((bars[valid] ** 2, bars[valid], np.ones(valid.sum())))
        out[row] = np.linalg.lstsq(design, window[valid], rcond=None)[0][0]
    return out


def _resid_oracle(y, x, d):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    out = np.full(y.shape, np.nan)
    for row in range(d - 1, len(y)):
        yw, xw = y[row - d + 1:row + 1], x[row - d + 1:row + 1]
        valid = np.isfinite(yw) & np.isfinite(xw)
        if valid.sum() < 3:
            continue
        design = np.column_stack((xw[valid] ** 2, xw[valid], np.ones(valid.sum())))
        quadratic, linear, intercept = np.linalg.lstsq(design, yw[valid], rcond=None)[0]
        fitted_last = quadratic * xw[-1] ** 2 + linear * xw[-1] + intercept
        out[row] = yw[-1] - fitted_last
    return out


def _inputs(time_name):
    n = 43
    bar = np.arange(n, dtype=float)
    x = 0.2 * bar + np.sin(bar / 3.7)
    y = 1.3 - 0.4 * x + 0.07 * x ** 2 + 0.11 * np.cos(bar / 2.9)
    x[[5, 18, 29]] = np.nan
    y[[8, 18, 34]] = np.nan
    increments = np.resize(np.array([1, 3, 2, 6]), n)
    time = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.cumsum(increments), unit="D")
    return pl.DataFrame({time_name: time, "A": y}), pl.DataFrame({time_name: time, "A": x}), y, x


@pytest.mark.parametrize("time_name", ["date", "timestamp"])
def test_effective_poly2_kernels_ignore_time_metadata_and_match_lstsq(time_name):
    load_all()
    y_frame, x_frame, y, x = _inputs(time_name)
    d = 12
    coeff = OperatorRegistry.get("ts_poly2_coeff", "polars", mode="any").calculate(y_frame, d=d)
    resid = OperatorRegistry.get("ts_poly2_resid", "polars", mode="any").calculate(y_frame, x_frame, d=d)
    assert coeff.columns == ["A"] and resid.columns == ["A"]
    np.testing.assert_allclose(coeff["A"].to_numpy(), _coeff_oracle(y, d), rtol=1e-10, atol=1e-11, equal_nan=True)
    np.testing.assert_allclose(resid["A"].to_numpy(), _resid_oracle(y, x, d), rtol=1e-10, atol=1e-11, equal_nan=True)


@pytest.mark.parametrize("time_name,dtype", [("date", pl.Date), ("timestamp", pl.Datetime)])
def test_effective_poly2_kernels_accept_empty_metadata_panels(time_name, dtype):
    load_all()
    frame = pl.DataFrame(schema={time_name: dtype, "A": pl.Float64})
    coeff = OperatorRegistry.get("ts_poly2_coeff", "polars", mode="any").calculate(frame, d=8)
    resid = OperatorRegistry.get("ts_poly2_resid", "polars", mode="any").calculate(frame, frame, d=8)
    assert coeff.schema == {"A": pl.Float64}
    assert resid.schema == {"A": pl.Float64}
    assert coeff.height == resid.height == 0


def test_effective_poly2_kernels_are_future_prefix_invariant_with_nan_rows():
    load_all()
    y_frame, x_frame, _, _ = _inputs("timestamp")
    cut, d = 27, 10
    y_changed = y_frame.with_columns(
        pl.when(pl.int_range(pl.len()) >= cut).then(pl.col("A") * -13 + 17).otherwise(pl.col("A")).alias("A")
    )
    x_changed = x_frame.with_columns(
        pl.when(pl.int_range(pl.len()) >= cut).then(pl.col("A") * 9 - 4).otherwise(pl.col("A")).alias("A")
    )
    coeff_op = OperatorRegistry.get("ts_poly2_coeff", "polars", mode="any")
    resid_op = OperatorRegistry.get("ts_poly2_resid", "polars", mode="any")
    coeff_before = coeff_op.calculate(y_frame, d=d)["A"].to_numpy()
    coeff_after = coeff_op.calculate(y_changed, d=d)["A"].to_numpy()
    resid_before = resid_op.calculate(y_frame, x_frame, d=d)["A"].to_numpy()
    resid_after = resid_op.calculate(y_changed, x_changed, d=d)["A"].to_numpy()
    np.testing.assert_allclose(coeff_before[:cut], coeff_after[:cut], equal_nan=True)
    np.testing.assert_allclose(resid_before[:cut], resid_after[:cut], equal_nan=True)


@pytest.mark.parametrize(
    "missing_y,missing_x,expect_nan",
    [(False, False, False), (True, False, True), (False, True, True), (True, True, True)],
)
def test_poly2_resid_current_row_missing_pair_semantics(missing_y, missing_x, expect_nan):
    load_all()
    d, n = 8, 20
    x = np.linspace(-1.2, 2.1, n)
    y = 0.4 - 0.7 * x + 0.2 * x ** 2 + 0.03 * np.sin(np.arange(n))
    if missing_y:
        y[-1] = np.nan
    if missing_x:
        x[-1] = np.nan
    timestamp = pd.date_range("2025-07-01", periods=n, freq="3D")
    y_frame = pl.DataFrame({"timestamp": timestamp, "A": y})
    x_frame = pl.DataFrame({"timestamp": timestamp, "A": x})
    actual = OperatorRegistry.get("ts_poly2_resid", "polars", mode="any").calculate(
        y_frame, x_frame, d=d
    )["A"].to_numpy()
    expected = _resid_oracle(y, x, d)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11, equal_nan=True)
    assert bool(np.isnan(actual[-1])) is expect_nan


@pytest.mark.parametrize("time_name", ["date", "timestamp"])
def test_poly2_resid_exact_non_symmetric_quadratic_has_zero_residual(time_name):
    load_all()
    x = np.linspace(-2.3, 3.7, 32)
    # Deliberately distinct intercept and quadratic coefficient catches coefficient reversal.
    y = 7.25 - 1.4 * x + 0.18 * x ** 2
    time = pd.date_range("2025-09-01", periods=len(x), freq="2D")
    y_frame = pl.DataFrame({time_name: time, "A": y})
    x_frame = pl.DataFrame({time_name: time, "A": x})
    actual = OperatorRegistry.get("ts_poly2_resid", "polars", mode="any").calculate(
        y_frame, x_frame, d=10
    )["A"].to_numpy()
    np.testing.assert_allclose(actual[9:], 0.0, rtol=0.0, atol=1e-11)


def test_poly2_resid_rejects_shifted_time_and_reordered_instrument_axes():
    load_all()
    timestamp = pd.date_range("2025-10-01", periods=20)
    y = pl.DataFrame({"timestamp": timestamp, "A": np.arange(20.0), "B": np.arange(20.0) ** 2})
    x = pl.DataFrame({"timestamp": timestamp, "A": np.linspace(-1, 1, 20), "B": np.linspace(2, 4, 20)})
    op = OperatorRegistry.get("ts_poly2_resid", "polars", mode="any")
    shifted = x.with_columns((pl.col("timestamp") + pl.duration(days=1)).alias("timestamp"))
    with pytest.raises(ValueError):
        op.calculate(y, shifted, d=8)
    with pytest.raises(ValueError):
        op.calculate(y, x.select("timestamp", "B", "A"), d=8)


@pytest.mark.parametrize("predictor", ["zero", "constant", "two_values"])
def test_poly2_resid_degenerate_constant_predictor_matches_missing_terminal_contract(predictor):
    load_all()
    n, d = 24, 9
    timestamp = pd.date_range("2025-11-01", periods=n)
    x = (np.zeros(n) if predictor == "zero" else
         np.ones(n) if predictor == "constant" else np.arange(n) % 2.0)
    y = np.sin(np.arange(n) / 3.0)
    pandas_y = pd.DataFrame({"A": y}, index=timestamp)
    pandas_x = pd.DataFrame({"A": x}, index=timestamp)
    polars_y = pl.DataFrame({"timestamp": timestamp, "A": y})
    polars_x = pl.DataFrame({"timestamp": timestamp, "A": x})
    pandas_result = OperatorRegistry.get("ts_poly2_resid", "pandas_numpy", mode="any").calculate(
        pandas_y, pandas_x, d=d
    ).to_numpy()[:, 0]
    polars_result = OperatorRegistry.get("ts_poly2_resid", "polars", mode="any").calculate(
        polars_y, polars_x, d=d
    )["A"].to_numpy()
    np.testing.assert_allclose(polars_result, pandas_result, equal_nan=True)
    assert np.isnan(polars_result).all()
