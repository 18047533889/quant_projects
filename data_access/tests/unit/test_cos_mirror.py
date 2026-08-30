"""cos_mirror 单元测试。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from data_access.cos.mirror import (
    ASHARE_DATASET_TABLE_MAP,
    DATASET_MIRROR_REGISTRY,
    US_DATASET_TABLE_MAP,
    _iter_dates,
    datasets_requiring_cos_mirror,
    mirror_spec_for_dataset,
    table_for_dataset,
)
from data_access.registry import load_registry


def test_mirror_registry_covers_markets():
    ashare = [n for n in DATASET_MIRROR_REGISTRY if n.startswith("ashare_")]
    us = [n for n in DATASET_MIRROR_REGISTRY if n.startswith("us_")]
    # 23 = 21 基础 + ashare_stock_daily_adj + ashare_stock_minute_adj
    # （ADJ_FIELD_MIGRATION 2026-08-28：后复权生成表镜像）
    assert len(ashare) == 23
    # 23 基础 +5 新增/拆分：us_stock_capital_split / us_stock_capital_shares /
    # us_security_master_daily_snap / us_ticker_shares_snapshot / us_fact_news
    # （原 us_stock_capital_daily 拆成 split+shares 两个镜像）
    assert len(us) == 27


def test_ashare_table_map():
    assert len(ASHARE_DATASET_TABLE_MAP) == 23  # 21 基础 + 2 后复权生成表
    assert ASHARE_DATASET_TABLE_MAP["ashare_stock_daily"] == "StockDailyBar"
    assert ASHARE_DATASET_TABLE_MAP["ashare_turnover_base_daily"] == "TurnoverBaseDaily"
    assert ASHARE_DATASET_TABLE_MAP["ashare_stock_daily_adj"] == "StockDailyBarAdj"


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


def test_datasets_requiring_cos_mirror_excludes_sip_and_factor_lake():
    config_path = Path(__file__).resolve().parents[2] / "config" / "datasets.yaml"
    reg = load_registry(config_path)
    eligible = datasets_requiring_cos_mirror(reg)
    assert "ashare_stock_daily" in eligible
    assert "us_universe_daily" in eligible
    assert "us_stocks_sip_day_aggs" not in eligible
    assert "factor_lake" not in eligible
