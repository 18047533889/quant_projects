from types import SimpleNamespace

from factor_engine.storage.sources import data_access_source as module


def test_estimate_scan_cost_uses_store_scope_api_and_semantic_filters(monkeypatch):
    seen = {}

    class Store:
        def estimate_scan_cost(self, dataset, **kwargs):
            seen["dataset"] = dataset
            seen.update(kwargs)
            return SimpleNamespace(estimated_rows=7, projection_bytes=42)

    source = object.__new__(module.DataAccessSource)
    source.dataset = "minute"
    source.semantic_filters = {"timeframe": "intraday"}
    source.params = {"market": "CN"}
    source._lazy_scan_base = False
    source._resolve_columns = lambda fields: (["Close"], {})
    source._time_range = lambda: ("2026-04-20", "2026-04-24")
    monkeypatch.setattr(module, "_get_store", lambda: Store())

    cost = source.estimate_scan_cost(fields=["close"], instruments=["000001.SZ"])

    assert cost.estimated_rows == 7
    assert seen["dataset"] == "minute"
    assert seen["columns"] == ["Close"]
    assert seen["filters"] == {"timeframe": "intraday"}
    assert seen["instrument_filter"] == ["000001.SZ"]
    assert seen["market"] == "CN"
