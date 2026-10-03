"""Loader integration contracts with COS and price reads replaced by fixtures."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest


def _load_example():
    path = Path(__file__).parents[1] / "examples" / "cos_batch_audit.py"
    spec = importlib.util.spec_from_file_location("cos_session_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Engine:
    def __init__(self, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


class _PriceStore:
    def __init__(self, calendar, *, invalid_price=None, price_mode="normal"):
        self.calendar = pd.DatetimeIndex(calendar)
        self.invalid_price = invalid_price
        self.price_mode = price_mode
        self.read_calls = 0

    def read(self, dataset, *, time_range, instrument_filter, **kwargs):
        self.read_calls += 1
        if self.price_mode == "empty":
            empty_rows = pd.DataFrame(columns=["TradeDate", "Symbol", "AdjVwap"])
            return pa.Table.from_pandas(empty_rows, preserve_index=False)
        sessions = self.calendar[
            (self.calendar >= pd.Timestamp(time_range[0]))
            & (self.calendar <= pd.Timestamp(time_range[1]))
        ]
        rows = []
        for session_position, session in enumerate(self.calendar):
            if session not in sessions:
                continue
            for asset_position, asset in enumerate(instrument_filter):
                value = 100.0 + session_position + asset_position
                if self.price_mode == "zero":
                    value = 0.0
                if self.invalid_price == (session, asset):
                    if asset_position % 2 == 0:
                        continue
                    value = 0.0
                rows.append({"TradeDate": session, "Symbol": asset, "AdjVwap": value})
        return pa.Table.from_pylist(rows)


def _install_fakes(monkeypatch, *, n_days, price_store, calendar_source="registry"):
    example = _load_example()
    assets = [f"A{index:03d}.SZ" for index in range(20)]
    calendar = pd.bdate_range("2020-01-02", periods=n_days + 2)
    panel = pd.DataFrame({
        "timestamp": calendar[:n_days],
        **{asset: np.arange(1, n_days + 1, dtype=np.float64) for asset in assets},
    })
    sha = "a" * 64
    record = {
        "uri": f"{example.POOL}/{sha}/fixture.parquet",
        "sha256": sha,
        "bytes": 128,
        "verified": True,
        "status": "evaluated_optimization_pending",
    }
    manifest_table = pa.Table.from_pylist([{"factors": {"fixture": record}}])
    panel_table = pa.Table.from_pandas(panel, preserve_index=False)
    factor = SimpleNamespace(
        table=panel_table,
        source_uri=record["uri"],
        source_etag="fixture-etag",
        content_sha256=sha,
        downloaded_bytes=128,
    )
    bound = SimpleNamespace(
        factor=factor,
        treatment_signature=(),
        manifest_sha256="b" * 64,
        source_status=record["status"],
        expression="fixture_expression",
    )
    authoritative_calendar = SimpleNamespace(
        source=calendar_source, has_data=True, trading_days=calendar,
    )
    monkeypatch.setattr(example, "DuckDBEngine", _Engine)
    monkeypatch.setattr(example, "DataAccessStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        example, "read_declared_cos_object",
        lambda *args, **kwargs: SimpleNamespace(table=manifest_table),
    )
    monkeypatch.setattr(example, "read_bound_factor", lambda *args, **kwargs: bound)
    monkeypatch.setattr(example, "get_store", lambda: price_store)
    monkeypatch.setattr(
        example, "get_market_calendar",
        lambda *args, **kwargs: authoritative_calendar,
    )
    return example, calendar, assets


@pytest.mark.parametrize("n_days", [500, 1260])
def test_loader_preserves_last_decision_and_real_tplus2_sessions(monkeypatch, n_days):
    calendar = pd.bdate_range("2020-01-02", periods=n_days + 2)
    price_store = _PriceStore(calendar)
    example, calendar, assets = _install_fakes(
        monkeypatch, n_days=n_days, price_store=price_store,
    )

    batch, labels, provenance = example.load_cos_sample(
        n_factors=1, n_assets=20, n_days=n_days, coverage_policy="strict",
    )

    assert batch.time_axis.size == n_days
    assert labels.decision_time[-1] == calendar[n_days - 1]
    assert labels.execution_time[-1] == calendar[n_days]
    assert labels.label_start_time[-1] == calendar[n_days]
    assert labels.label_end_time[-1] == calendar[n_days + 1]
    assert labels.validity[-1].all()
    assert provenance["calendar_source"] == "registry"
    assert provenance["days"] == n_days
    assert price_store.read_calls > 0
    assert list(batch.asset_axis.values) == assets


@pytest.mark.parametrize("missing_or_zero", ["missing", "zero"])
def test_missing_or_zero_endpoint_invalidates_label_without_shifting_dates(
    monkeypatch, missing_or_zero,
):
    n_days = 500
    calendar = pd.bdate_range("2020-01-02", periods=n_days + 2)
    first_asset = "A000.SZ"
    invalid_value = (calendar[-1], first_asset)
    price_store = _PriceStore(calendar, invalid_price=invalid_value)
    example, calendar, _ = _install_fakes(
        monkeypatch, n_days=n_days, price_store=price_store,
    )
    if missing_or_zero == "zero":
        price_store.invalid_price = (calendar[-1], "A001.SZ")

    _, labels, _ = example.load_cos_sample(
        n_factors=1, n_assets=20, n_days=n_days, coverage_policy="strict",
    )

    invalid_asset = 0 if missing_or_zero == "missing" else 1
    assert not labels.validity[-1, invalid_asset]
    assert labels.validity[-1, (invalid_asset + 1) % 20]
    assert labels.execution_time[-1] == calendar[-2]
    assert labels.label_end_time[-1] == calendar[-1]


def test_loader_rejects_fallback_calendar_before_reading_prices(monkeypatch):
    n_days = 500
    calendar = pd.bdate_range("2020-01-02", periods=n_days + 2)
    price_store = _PriceStore(calendar)
    example, _, _ = _install_fakes(
        monkeypatch, n_days=n_days, price_store=price_store, calendar_source="fallback",
    )

    with pytest.raises(ValueError, match="authoritative ashare_calendar"):
        example.load_cos_sample(n_factors=1, n_assets=20, n_days=n_days)

    assert price_store.read_calls == 0


@pytest.mark.parametrize("price_mode", ["empty", "zero"])
def test_loader_rejects_cohort_with_no_valid_labels(monkeypatch, price_mode):
    n_days = 500
    calendar = pd.bdate_range("2020-01-02", periods=n_days + 2)
    price_store = _PriceStore(calendar, price_mode=price_mode)
    example, _, _ = _install_fakes(
        monkeypatch, n_days=n_days, price_store=price_store,
    )

    with pytest.raises(ValueError, match="no valid.*labels"):
        example.load_cos_sample(n_factors=1, n_assets=20, n_days=n_days)

    assert price_store.read_calls > 0

def test_materialization_budget_rejects_before_any_price_read(monkeypatch):
    n_days = 500
    calendar = pd.bdate_range("2020-01-02", periods=n_days + 2)
    price_store = _PriceStore(calendar)
    example, _, _ = _install_fakes(
        monkeypatch, n_days=n_days, price_store=price_store,
    )

    with pytest.raises(MemoryError, match="materialization budget exceeded"):
        example.load_cos_sample(
            n_factors=1, n_assets=20, n_days=n_days,
            materialization_budget_bytes=1,
        )

    assert price_store.read_calls == 0
