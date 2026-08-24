from __future__ import annotations

from pathlib import Path

import pytest

from factor_engine.storage.factory import build_data_source
from workspace_paths import resolve_path


def test_build_data_source_expands_tilde_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data_root = tmp_path / "quant_projects" / "data" / "kline"
    data_root.mkdir(parents=True)

    source = build_data_source(
        {
            "type": "parquet_kline",
            "root": "~/quant_projects/data/kline",
            "instrument_col": "ticker",
            "timestamp_col": "window_start",
        }
    )
    assert Path(source.root).resolve() == data_root.resolve()


def test_resolve_path_relative_to_quant_projects(monkeypatch):
    from workspace_paths import quant_projects_root

    root = quant_projects_root()
    resolved = resolve_path("data/us_stock/massive_data/StockDailyBar")
    assert resolved == (root / "data/us_stock/massive_data/StockDailyBar").resolve()


def test_composite_top_level_date_range_propagates_to_subsources():
    from factor_engine.storage.data_access_source import DataAccessSource

    composite = build_data_source(
        {
            "type": "composite",
            "anchor": "price",
            "anchor_column": "close",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "sources": {
                "price": {
                    "type": "data_access",
                    "dataset": "us_stocks_sip_day_aggs",
                },
                "ratios": {
                    "type": "data_access",
                    "dataset": "financials_ratios",
                    "start_date": "2023-01-01",
                },
            },
            "joins": {"ratios": "asof_backward"},
        }
    )
    price = composite.sources["price"]
    ratios = composite.sources["ratios"]
    assert isinstance(price, DataAccessSource)
    assert price.start_date == "2024-01-01"
    assert price.end_date == "2024-06-30"
    assert ratios.start_date == "2023-01-01"
    assert ratios.end_date == "2024-06-30"
