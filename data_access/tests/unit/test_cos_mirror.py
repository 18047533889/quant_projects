"""cos_mirror 单元测试。"""
from __future__ import annotations

from datetime import date

from data_access.cos_mirror import (
    ASHARE_DATASET_TABLE_MAP,
    DATASET_MIRROR_REGISTRY,
    US_DATASET_TABLE_MAP,
    _iter_dates,
    mirror_spec_for_dataset,
    table_for_dataset,
)


def test_mirror_registry_covers_markets():
    ashare = [n for n in DATASET_MIRROR_REGISTRY if n.startswith("ashare_")]
    us = [n for n in DATASET_MIRROR_REGISTRY if n.startswith("us_")]
    assert len(ashare) == 20
    assert len(us) == 23


def test_ashare_table_map():
    assert len(ASHARE_DATASET_TABLE_MAP) == 20
    assert ASHARE_DATASET_TABLE_MAP["ashare_stock_daily"] == "StockDailyBar"


def test_us_massive_tables():
    assert US_DATASET_TABLE_MAP["us_stock_daily"] == "StockDailyBar"
    assert "us_universe_daily" not in US_DATASET_TABLE_MAP


def test_hive_specs():
    uni = mirror_spec_for_dataset("us_universe_daily")
    assert uni is not None
    assert uni.layout == "hive_year"
    adj = mirror_spec_for_dataset("us_adj_factor")
    assert adj is not None
    assert adj.layout == "hive_date"


def test_table_for_dataset():
    assert table_for_dataset("us_stock_daily") == "StockDailyBar"
    assert table_for_dataset("us_adj_factor") is None


def test_iter_dates():
    days = _iter_dates(date(2024, 1, 1), date(2024, 1, 3))
    assert days == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
