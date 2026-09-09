from types import SimpleNamespace

import pandas as pd
import pytest

from factor_engine.storage.sources import data_access_source as module
from factor_engine.storage.sources.data_access_source import DataAccessSource
from factor_engine.storage.sources.field_plan import NormalizedFieldPlan


class _Broker:
    def current_read_budget(self):
        return 64 * 1024 * 1024

    def acquire_memory(self, *args, **kwargs):
        return SimpleNamespace(release=lambda: None)


class _Handle:
    def __init__(self, digest):
        self.snapshot = SimpleNamespace(snapshot_id="snap-a")
        self.read_identity = SimpleNamespace(
            dataset="ashare_stock_daily", source_snapshot=digest,
        )

    def to_arrow(self):
        return object()

    def close(self):
        return None


def _values(raw=False):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "000001.SZ"),
         (pd.Timestamp("2024-01-03"), "000001.SZ")],
        names=["time", "instrument"],
    )
    return pd.Series([100.0, -200.0] if raw else [0.01, -0.02], index=index)


def _source(monkeypatch):
    monkeypatch.setattr(module, "_data_cache_authority", lambda: _Broker())
    source = DataAccessSource(
        dataset="ashare_stock_daily", fields={"ret": "Return"}, params={"lazy_scan": True},
    )
    source._data_snapshot_id = "snap-a"
    source.refresh_snapshot = lambda **kwargs: source._data_snapshot_id
    source._resolve_columns = lambda names: (["Return"], {"Return": "ret"})
    source._adapter_options = lambda store: (
        SimpleNamespace(time_column="time", instrument_column="instrument"),
        False,
        None,
    )
    plan = NormalizedFieldPlan(
        logical_concept="ret",
        physical_dataset="ashare_stock_daily",
        physical_fields=("Return",),
        scale=0.0001,
        source="catalog",
        canonical_unit="decimal",
    )
    source._ensure_field_plans = lambda names: {"ret": plan}
    source._max_cache_bytes = 64 * 1024 * 1024
    return source


def test_requested_lazy_approved_eager_uses_actual_normalized_units_once(monkeypatch):
    source = _source(monkeypatch)
    calls = []
    digest = "a" * 32

    class Store:
        def read(self, *args, **kwargs):
            calls.append(kwargs)
            return _Handle(digest)

    monkeypatch.setattr(module, "_get_store", lambda: Store())
    monkeypatch.setattr(
        "data_access.read.adapters.arrow_table_to_multiindex_columns",
        lambda *args, **kwargs: {"ret": _values(raw=False)},
    )
    source.bind_approved_content_digest(digest)

    cold = source.load_columns(["ret"])["ret"]
    warm = source.load_columns(["ret"])["ret"]

    assert source._lazy_scan is True
    assert len(calls) == 1
    assert calls[0]["normalize_units"] is True
    assert cold.tolist() == pytest.approx([0.01, -0.02])
    assert warm is cold


def test_actual_lazy_raw_catalog_units_are_scaled_exactly_once(monkeypatch):
    source = _source(monkeypatch)
    lazy_calls = []

    class Store:
        pass

    def scan(*args, **kwargs):
        lazy_calls.append(kwargs)
        return {"ret": _values(raw=True)}

    monkeypatch.setattr(module, "_get_store", lambda: Store())
    monkeypatch.setattr(
        "factor_engine.backend.polars_lazy.scan_dataset_columns", scan,
    )

    result = source.load_columns(["ret"])["ret"]

    assert len(lazy_calls) == 1
    assert result.tolist() == pytest.approx([0.01, -0.02])


def test_normalizer_explicit_unit_state_is_mutation_sensitive(monkeypatch):
    source = _source(monkeypatch)
    already_normalized = {"ret": _values(raw=False)}
    source._normalize_contract_columns(
        already_normalized, ["ret"], units_normalized=True,
    )
    assert already_normalized["ret"].tolist() == pytest.approx([0.01, -0.02])

    raw = {"ret": _values(raw=True)}
    source._normalize_contract_columns(
        raw, ["ret"], units_normalized=False,
    )
    assert raw["ret"].tolist() == pytest.approx([0.01, -0.02])
