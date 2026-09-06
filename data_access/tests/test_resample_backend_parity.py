"""Small synthetic regression oracles for standalone frequency resampling."""

import pandas as pd
import pyarrow as pa
import pytest

from data_access.read.resample import (
    _apply_frequency_pandas,
    _apply_frequency_polars,
    apply_frequency,
)


def test_non_numeric_first_skips_null_and_preserves_column_order():
    table = pa.Table.from_pandas(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03"]),
        "symbol": ["A", "A", "A"],
        "label": [None, "valid", "third"],
        "value": [1.0, 3.0, 5.0],
    }), preserve_index=False)
    kwargs = dict(alias="D", time_column="date", instrument_column="symbol")
    expected = _apply_frequency_pandas(table, **kwargs).to_pandas()
    actual = _apply_frequency_polars(table, **kwargs)
    assert actual is not None, "Supported ordinary grouped input must use fast path"
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)


def test_fast_path_time_order_is_monotonic():
    table = pa.Table.from_pandas(pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=64),
        "symbol": ["A"] * 64,
        "value": list(range(64)),
    }), preserve_index=False)
    actual = _apply_frequency_polars(
        table, alias="D", time_column="date", instrument_column="symbol"
    )
    assert actual is not None
    assert actual.to_pandas()["date"].is_monotonic_increasing


def test_ungrouped_frequency_preserves_empty_time_buckets():
    table = pa.Table.from_pandas(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        "value": [1.0, 5.0],
    }), preserve_index=False)
    expected = _apply_frequency_pandas(
        table, alias="D", time_column="date", instrument_column=None
    ).to_pandas()
    actual = apply_frequency(table, frequency="daily", time_column="date").to_pandas()
    assert len(expected) == 3
    pd.testing.assert_frame_equal(actual, expected)
