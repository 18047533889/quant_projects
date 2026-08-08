# -*- coding: utf-8 -*-
"""跨市场（A股/美股）schema 与语义一致性测试。

覆盖 2026-08-08 两本 COS 数据字典（COS_ashare_lqtp_data_dictionary.md /
COS_us_massive_data_dictionary.md）与 dataaccess 的对照落地：
    1. A股 datasets.yaml schema 对齐字典（Factor/HighLimit/LowLimit/IsSuspend、
       Status.PublicStatus（不是 ListedState）、Industry.IndustrySource）。
    2. 美股 schema 修正（Valuation/Indicator 用 snake_case，不是 A股 PascalCase；
       Capital 双 schema 拆分 split/shares，禁止 glob 混读拼炸）。
    3. SemanticFieldCatalog 跨市场消歧：ret/roe/total_assets/revenue/market_cap
       按 dataset 前缀命中对应市场字段，单位 scale 不串味。
    4. required_filters 支持列过滤（timeframe/IndustrySource 走 filters 而非 params）。
"""
from __future__ import annotations

import pytest

from data_access.cos_registry_runtime import patch_registry
from data_access.read.semantic_catalog import get_semantic_catalog
from data_access.registry import load_registry


@pytest.fixture(scope="module")
def catalog():
    return get_semantic_catalog()


@pytest.fixture(scope="module")
def registry():
    return patch_registry(load_registry())


# ---------------------------------------------------------------------------
# 1. A股 schema 对齐字典
# ---------------------------------------------------------------------------

def test_ashare_daily_schema_has_factor_and_limits(registry):
    ds = registry.get("ashare_stock_daily")
    assert "Factor" in ds.schema          # 后复权乘数（Close×Factor）
    assert "HighLimit" in ds.schema       # 涨停价
    assert "LowLimit" in ds.schema        # 跌停价
    assert "IsSuspend" in ds.schema       # 停牌标记
    assert "PreClose" in ds.schema


def test_ashare_status_uses_public_status_not_listed_state(registry):
    ds = registry.get("ashare_stock_status")
    assert "PublicStatus" in ds.schema    # 字典：可交易过滤 PublicStatus=='正常上市'
    assert "ListedState" not in ds.schema  # 字典：不存在 ListedState 列


def test_ashare_industry_has_source(registry):
    ds = registry.get("ashare_stock_industry")
    assert "IndustrySource" in ds.schema  # 字典：必须先 filter 单一 IndustrySource


def test_ashare_valuation_schema_expanded(registry):
    ds = registry.get("ashare_stock_valuation_daily")
    for col in ("PeRatioLyr", "PsRatio", "PcfRatio", "DividendRatio", "FreeCap", "ACap", "AMarketCap"):
        assert col in ds.schema


# ---------------------------------------------------------------------------
# 2. 美股 schema 修正（snake_case / 双 schema 拆分）
# ---------------------------------------------------------------------------

def test_us_valuation_uses_snake_case_not_pascal(registry):
    ds = registry.get("us_stock_valuation_daily")
    assert "market_cap" in ds.schema
    assert "price_to_earnings" in ds.schema
    assert "return_on_equity" in ds.schema
    assert "dividend_yield" in ds.schema
    # 字典：美股 Valuation 不是 A股 PascalCase 列名
    assert "MarketCap" not in ds.schema
    assert "PeRatio" not in ds.schema
    # instrument 列是 ticker（不是 Ticker）
    assert ds.instrument_column == "ticker"


def test_us_indicator_snake_case(registry):
    ds = registry.get("us_stock_indicator")
    assert "return_on_assets" in ds.schema
    assert "price_to_book" in ds.schema


def test_us_capital_split_excludes_shares(registry):
    """字典 C13：同一目录混放拆分事件 + shares PIT，禁止 glob 混读。"""
    split = registry.get("us_stock_capital_split")
    shares = registry.get("us_stock_capital_shares")
    # glob 精确分流：split 只命中 {date}.parquet，shares 只命中 shares_*.parquet
    assert split.glob == "[0-9]*.parquet"
    assert shares.glob == "shares_*.parquet"
    # schema 完全不同：split 是事件（id/execution_date/split_from...），
    # shares 是股本（pit_basic_shares_outstanding...）
    assert "execution_date" in split.schema
    assert "split_from" in split.schema
    assert "pit_basic_shares_outstanding" in shares.schema
    assert "pit_diluted_shares_outstanding" in shares.schema
    # 旧双 schema 数据集不再注册（避免 union_by_name 拼炸）
    assert "us_stock_capital_daily" not in registry


def test_us_financial_timeframe_present(registry):
    for name in ("us_stock_balance", "us_stock_income", "us_stock_cashflow"):
        ds = registry.get(name)
        assert "timeframe" in ds.schema
        assert ds.time_column == "filing_date"
        assert ds.instrument_column == "ticker"


# ---------------------------------------------------------------------------
# 3. SemanticFieldCatalog 跨市场消歧
# ---------------------------------------------------------------------------

def test_ret_disambiguates_by_market(catalog):
    a = catalog.resolve_one("ret", dataset="ashare_stock_daily")
    u = catalog.resolve_one("ret", dataset="us_stock_daily")
    assert a is not None and u is not None
    assert a.physical_name == "Return" and a.scale == 0.0001   # A股 bp → /10000
    assert u.physical_name == "Ret" and u.scale is None          # 美股小数，不再除
    assert a.market == "ashare" and u.market == "us"


def test_financial_fields_disambiguate_by_market(catalog):
    # total_assets：A股 StockBalance.TotalAssets vs 美股 StockBalance.total_assets
    a = catalog.resolve_one("total_assets", dataset="ashare_stock_balance")
    u = catalog.resolve_one("total_assets", dataset="us_stock_balance")
    assert a.dataset == "ashare_stock_balance" and a.physical_name == "TotalAssets"
    assert u.dataset == "us_stock_balance" and u.physical_name == "total_assets"
    # revenue：A股 OperatingRevenue alias vs 美股 StockIncome.revenue
    a = catalog.resolve_one("revenue", dataset="ashare_stock_income")
    u = catalog.resolve_one("revenue", dataset="us_stock_income")
    assert a.physical_name == "OperatingRevenue"
    assert u.physical_name == "revenue"
    assert u.dataset == "us_stock_income"
    # market_cap：A股 MarketCap vs 美股 market_cap
    a = catalog.resolve_one("market_cap", dataset="ashare_stock_valuation_daily")
    u = catalog.resolve_one("market_cap", dataset="us_stock_daily")
    assert a.physical_name == "MarketCap"
    assert u.physical_name == "market_cap" and u.dataset == "us_stock_valuation_daily"


def test_roe_is_ashare_percent_and_us_decimal(catalog):
    a = catalog.resolve_one("roe", dataset="ashare_stock_indicator")
    u = catalog.resolve_one("return_on_equity", dataset="us_stock_valuation_daily")
    assert a is not None and a.scale == 0.01       # A股 ROE % → /100
    assert u is not None and u.scale is None        # 美股 ROE 小数，不再 /100


def test_us_financial_required_timeframe(catalog):
    f = catalog.resolve_one("total_assets", dataset="us_stock_balance")
    assert "timeframe" in f.required_filters


# ---------------------------------------------------------------------------
# 4. required_filters 支持列过滤（filters 而非 params）
# ---------------------------------------------------------------------------

def test_required_filters_accept_params(catalog):
    from data_access.read.semantic_catalog import SemanticField
    from data_access.store import _production_mode
    # 用 index_weight（required_filters=[IndexSymbol]）验证 params 途径仍工作
    f = catalog.resolve_one("index_weight")
    assert f is not None and "IndexSymbol" in f.required_filters


def test_catalog_required_filters_flag_present(catalog):
    """美股财务字段的 required_filters 应含 timeframe（列过滤）。"""
    for name in ("total_assets", "total_liabilities", "revenue", "net_income"):
        f = catalog.resolve_one(name, dataset="us_stock_income")
        assert f is not None
        assert "timeframe" in f.required_filters
