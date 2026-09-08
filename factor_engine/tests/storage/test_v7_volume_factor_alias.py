"""Synthetic field-order oracle; not real HFQ data acceptance."""
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from factor_engine.storage.sources import data_access_source as module
from factor_engine.storage.sources.data_access_source import DataAccessSource


@pytest.mark.parametrize("requests", [
    [["volume"]],
    [["volume", "adj_factor"]],
    [["adj_factor", "volume"]],
    [["adj_factor"], ["volume"]],
    [["volume"], ["volume", "adj_factor"]],
    [["volume", "Factor"]],
    [["Factor", "volume"]],
    [["volume", "adj_factor", "Factor"]],
    [["adj_factor", "Factor", "volume"]],
])
def test_adjusted_volume_independent_of_factor_alias_and_cache(monkeypatch, requests):
    table = pa.Table.from_pydict({
        "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "Symbol": ["A", "A", "A"],
        "Volume": [100.0, 240.0, 300.0], "Factor": [2.0, 4.0, 0.0],
    })

    class Store:
        def describe_dataset(self, *args, **kwargs):
            return SimpleNamespace(snapshot_id="synthetic-content-1")

        def read(self, *args, columns, **kwargs):
            selected = table.select(columns)
            return SimpleNamespace(snapshot=SimpleNamespace(snapshot_id="synthetic-content-1"),
                                   to_arrow=lambda: selected)

    monkeypatch.setattr(module, "_get_store", lambda: Store())
    source = DataAccessSource(dataset="ashare_stock_daily_adj",
                              fields={"volume": "Volume", "adj_factor": "Factor"})
    source._cache_broker = SimpleNamespace(
        acquire_memory=lambda *args, **kwargs: SimpleNamespace(release=lambda: None))
    source._max_cache_bytes = 1024 * 1024
    monkeypatch.setattr(source, "_ensure_field_plans", lambda names: {})
    monkeypatch.setattr(source, "_preflight_logical_columns", lambda names: None)
    monkeypatch.setattr(source, "_adapter_options", lambda store: (
        SimpleNamespace(time_column="TradeDate", instrument_column="Symbol"), True, None))
    try:
        for names in requests:
            output = source.load_columns(names)
            assert set(output) == set(names)
            assert set(names).issubset(source._column_cache), "warm-cache oracle must actually cache"
            if "volume" in output:
                expected = pd.Series([50.0, 60.0, float("nan")], index=output["volume"].index)
                pd.testing.assert_series_equal(output["volume"], expected, check_names=False)
            if "adj_factor" in output:
                assert output["adj_factor"].tolist() == [2.0, 4.0, 0.0]
            if "Factor" in output:
                assert output["Factor"].tolist() == [2.0, 4.0, 0.0]
    finally:
        source.close()
