# -*- coding: utf-8
"""DataAccessSource.scan_polars_long 测试。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_data_access_scan_polars_long_renames_axes():
    pytest.importorskip("polars")
    import polars as pl

    from storage.sources.data_access_source import DataAccessSource

    src = DataAccessSource(
        dataset="test_ds",
        fields={"close": "Close", "volume": "Volume"},
    )
    mock_store = MagicMock()
    mock_ds = MagicMock()
    mock_ds.time_column = "TradeDate"
    mock_ds.instrument_column = "Symbol"
    mock_store.get_dataset.return_value = mock_ds

    base = pl.LazyFrame(
        {
            "TradeDate": ["2024-01-01", "2024-01-01"],
            "Symbol": ["A", "B"],
            "Close": [10.0, 20.0],
            "Volume": [100.0, 200.0],
        }
    )
    mock_store.scan.return_value = base

    with patch("storage.sources.data_access_source._get_store", return_value=mock_store):
        lf = src.scan_polars_long(["close", "volume"])

    cols = lf.collect_schema().names()
    assert cols == ["ts", "inst", "close", "volume"]
    row = lf.collect().row(0)
    assert row[0] == "2024-01-01"
    assert row[1] == "A"
    mock_store.scan.assert_called_once()
    call_cols = mock_store.scan.call_args.kwargs.get("columns")
    assert "Close" in call_cols
