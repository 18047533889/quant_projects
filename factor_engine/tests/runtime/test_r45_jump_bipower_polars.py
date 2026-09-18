"""Independent contracts for the Polars-only daily bipower canonical."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.polars_native.ts_advanced_batch5 import TSJumpBipowerPolarsNative


def _oracle(values, window):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    for row in range(window - 1, len(values)):
        chunk = values[row - window + 1:row + 1]
        if np.isfinite(chunk).all():
            out[row] = (np.pi / 2.0) * np.sum(np.abs(chunk[1:]) * np.abs(chunk[:-1]))
    return out


def _fixture(time_name="timestamp"):
    values = np.array([0.01, -0.03, 0.02, 0.08, -0.01, 0.04, -0.02, 0.03, 0.01])
    dates = pd.date_range("2025-03-01", periods=len(values), freq="D")
    return values, pl.DataFrame({time_name: dates, "A": values, "B": -2 * values})


@pytest.mark.parametrize("time_name", ["date", "timestamp"])
def test_jump_bipower_matches_independent_formula_and_preserves_identity(time_name):
    values, frame = _fixture(time_name)
    actual = TSJumpBipowerPolarsNative().calculate(frame, window=5)
    assert actual[time_name].to_list() == frame[time_name].to_list()
    np.testing.assert_allclose(actual["A"].to_numpy(), _oracle(values, 5), equal_nan=True)
    np.testing.assert_allclose(actual["B"].to_numpy(), _oracle(-2 * values, 5), equal_nan=True)


def test_jump_bipower_missing_window_and_future_prefix_contracts():
    values, frame = _fixture()
    values[5] = np.nan
    frame = frame.with_columns(pl.Series("A", values))
    op = TSJumpBipowerPolarsNative()
    before = op.calculate(frame, window=5)["A"].to_numpy()
    changed = frame.with_columns(
        pl.when(pl.int_range(pl.len()) >= 7).then(pl.col("A") * 11).otherwise(pl.col("A")).alias("A")
    )
    after = op.calculate(changed, window=5)["A"].to_numpy()
    np.testing.assert_allclose(before[:7], after[:7], equal_nan=True)
    np.testing.assert_allclose(before, _oracle(values, 5), equal_nan=True)


def test_jump_bipower_nonfinite_window_fails_closed_without_infinity():
    values, frame = _fixture()
    values[5] = np.inf
    actual = TSJumpBipowerPolarsNative().calculate(
        frame.with_columns(pl.Series("A", values)), window=5
    )["A"].to_numpy()
    np.testing.assert_allclose(actual, _oracle(values, 5), equal_nan=True)
    assert not np.isinf(actual).any()


def test_jump_bipower_empty_validation_backend_and_physical_path():
    op = TSJumpBipowerPolarsNative()
    empty = pl.DataFrame(schema={"date": pl.Datetime, "A": pl.Float64})
    assert op.calculate(empty, window=5).shape == (0, 2)
    _, frame = _fixture()
    with pytest.raises((TypeError, ValueError)):
        op.calculate(frame, window=4)
    with pytest.raises(TypeError, match="polars.DataFrame"):
        op.calculate(pd.DataFrame({"A": np.arange(8.0)}), window=5)
    spec = op._physical_spec
    assert spec.execution_kind is ExecutionKind.POLARS_NATIVE_EXPR
    assert not spec.materializes_full_panel
    assert not spec.supports_lazy


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_jump_bipower_recovers_after_nonfinite_window_exits(bad):
    values = np.tile([.01,-.02,.03,.04,.01], 6)
    values[7] = bad
    frame = pl.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=len(values)), "A": values})
    actual = TSJumpBipowerPolarsNative().calculate(frame, window=5)["A"].to_numpy()
    np.testing.assert_allclose(actual, _oracle(values, 5), equal_nan=True)
    assert np.isfinite(actual[12:]).all()
