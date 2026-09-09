"""DataAccessSource LazyColumnBundle prefetch 测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from factor_engine.tests.storage.test_data_access_source_concurrent import admitted_cache


def test_prefetch_lazy_bundle_single_collect(admitted_cache):
    pytest.importorskip("polars")

    from factor_engine.storage.sources.data_access_source import DataAccessSource

    src = DataAccessSource(
        dataset="test_ds",
        read_auto=True,
        params={"lazy_scan": True},
    )
    src.enable_lazy_scan(True)

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    series_close = pd.Series([1.0, 2.0], index=idx)
    series_open = pd.Series([0.5, 1.5], index=idx)

    mock_store = MagicMock()
    mock_ds = MagicMock()
    mock_ds.time_column = "align_time"
    mock_ds.instrument_column = "ticker"
    mock_store.get_dataset.return_value = mock_ds
    mock_store.describe_dataset.return_value = MagicMock(snapshot_id="snap_lazy")

    mock_lf = MagicMock()
    mock_store.scan_polars.return_value = mock_lf

    collect_count = {"n": 0}

    def fake_materialize(physical_columns, *, output_names=None):
        collect_count["n"] += 1
        out = {}
        for c in physical_columns:
            logical = (output_names or {}).get(c, c)
            out[logical] = series_close if logical == "close" else series_open
        return out

    with patch(
        "factor_engine.storage.sources.data_access_source._get_store",
        return_value=mock_store,
    ):
        with patch(
            "data_access.store.adapter_options_for_dataset",
            return_value={"normalize_timestamp": False},
        ):
            with patch(
                "factor_engine.backend.polars_lazy.build_lazy_column_bundle",
            ) as build_bundle:
                bundle = MagicMock()
                bundle.snapshot_id = "snap_lazy"
                bundle.missing_physical.return_value = []
                bundle.materialize_columns.side_effect = fake_materialize
                build_bundle.return_value = bundle

                src.prefetch_columns(["close", "open"])
                src.load_column("close")
                src.load_column("open")

    assert collect_count["n"] == 1
    assert src._column_cache["close"] is series_close
