"""Independent contracts for physical-bar lag-1 autocorrelation."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.polars_native.ts_advanced_batch5 import TSLag1AutocorrPolarsNative


def _oracle(values, window):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    for row in range(len(values)):
        if row < window:
            continue
        chunk = values[row - window:row + 1]
        left, right = chunk[:-1], chunk[1:]
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() != window:
            continue
        a, b = left[valid], right[valid]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            out[row] = np.corrcoef(a, b)[0, 1]
    return out


def _fixture(time_name="timestamp"):
    rng = np.random.default_rng(4601)
    values = 0.65 * np.sin(np.arange(48) / 3.0) + rng.normal(scale=0.2, size=48)
    dates = pd.Timestamp("2025-04-01") + pd.to_timedelta(np.cumsum(np.resize([1, 2, 4], 48)), unit="D")
    return values, pl.DataFrame({time_name: dates, "A": values, "B": -values + 0.3})


@pytest.mark.parametrize("time_name", ["date", "timestamp"])
def test_lag1_autocorr_matches_physical_adjacency_oracle(time_name):
    values, frame = _fixture(time_name)
    actual = TSLag1AutocorrPolarsNative().calculate(frame, window=20, n_bins=5)
    assert actual[time_name].to_list() == frame[time_name].to_list()
    np.testing.assert_allclose(actual["A"].to_numpy(), _oracle(values, 20), equal_nan=True)
    np.testing.assert_allclose(actual["B"].to_numpy(), _oracle(-values + 0.3, 20), equal_nan=True)


def test_lag1_autocorr_missing_bar_is_not_compressed_and_reconnected():
    values, frame = _fixture()
    values[13] = np.nan
    frame = frame.with_columns(pl.Series("A", values))
    actual = TSLag1AutocorrPolarsNative().calculate(frame, window=20)["A"].to_numpy()
    expected = _oracle(values, 20)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    chunk = values[0:21]
    compressed = chunk[np.isfinite(chunk)]
    compressed_corr = np.corrcoef(compressed[:-1], compressed[1:])[0, 1]
    assert np.isnan(actual[20])
    assert np.isfinite(compressed_corr)


def test_lag1_autocorr_is_scale_safe_for_tiny_and_huge_values_and_n_bins_is_noop():
    values, frame = _fixture()
    op = TSLag1AutocorrPolarsNative()
    baseline = op.calculate(frame, window=20, n_bins=3)["A"].to_numpy()
    tiny = frame.with_columns(pl.Series("A", values * 1e-250))
    huge = frame.with_columns(pl.Series("A", values * 1e250))
    np.testing.assert_allclose(op.calculate(tiny, window=20, n_bins=5)["A"].to_numpy(), baseline, equal_nan=True, atol=1e-14)
    np.testing.assert_allclose(op.calculate(huge, window=20, n_bins=10)["A"].to_numpy(), baseline, equal_nan=True, atol=1e-14)


def test_lag1_autocorr_prefix_validation_backend_and_physical_path():
    values, frame = _fixture()
    op = TSLag1AutocorrPolarsNative()
    before = op.calculate(frame, window=20)["A"].to_numpy()
    changed = frame.with_columns(
        pl.when(pl.int_range(pl.len()) >= 35).then(pl.col("A") * -7 + 2).otherwise(pl.col("A")).alias("A")
    )
    after = op.calculate(changed, window=20)["A"].to_numpy()
    np.testing.assert_allclose(before[:35], after[:35], equal_nan=True)
    with pytest.raises((TypeError, ValueError)):
        op.calculate(frame, window=19)
    with pytest.raises(TypeError, match="polars.DataFrame"):
        op.calculate(pd.DataFrame({"A": values}), window=20)
    spec = op._physical_spec
    assert spec.execution_kind is ExecutionKind.POLARS_NUMPY_KERNEL
    assert spec.materializes_full_panel
    assert not spec.supports_lazy
