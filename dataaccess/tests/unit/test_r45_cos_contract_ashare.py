# -*- coding: utf-8 -*-
"""R45: A-share COS contract completeness + derived-dataset source-class split + TopTen PIT."""
from __future__ import annotations

import pytest

from data_access.core.exceptions import ValidationError
from data_access.cos_contract import (
    COS_DATASET_CONTRACTS,
    DERIVED_DATASET_CONTRACTS,
    cos_physical_contracts,
    require_cos_contract,
)

_TOP_TEN = ("ashare_stock_topten_shareholder", "ashare_stock_topten_float_shareholder")
_E1_FIN = ("ashare_stock_balance", "ashare_stock_income", "ashare_stock_cashflow", "ashare_stock_indicator")


def test_topten_unique_key_uses_shareholder_rank():
    # R45: TopTen S1 snapshot grain = 一股票×一快照日×一名股东，名次字段是
    # ShareholderRank（1–10），**不是** Rank。断言 unique_key 不得退化为 Rank。
    for name in _TOP_TEN:
        c = COS_DATASET_CONTRACTS[name]
        assert c.unique_key == ("TradeDate", "Symbol", "ShareholderRank"), name
        # 名次字段必须显式是 ShareholderRank（1–10 前十大名次），而非裸 "Rank"。
        assert c.unique_key[2] == "ShareholderRank", name


def test_ashare_universe_daily_not_cos_physical():
    """universe_daily 是本地 clean_data 研究 universe（字典 §8 非 COS 表），
    不得伪装成 COS 物理源。它已移入 DERIVED 注册表，require_cos_contract 必须拒绝。"""
    assert "ashare_universe_daily" in COS_DATASET_CONTRACTS
    assert "ashare_universe_daily" in DERIVED_DATASET_CONTRACTS
    assert "ashare_universe_daily" not in cos_physical_contracts()
    # 物理源集合不含任何 derived 数据集
    for name, c in cos_physical_contracts().items():
        assert c.source_class == "cos_physical", name
    # require_cos_contract 对派生源 fail-closed（不能当 COS 物理契约用）
    with pytest.raises(ValidationError):
        require_cos_contract("ashare_universe_daily")
    # 真物理源仍可通过
    assert require_cos_contract("ashare_stock_daily").source_class == "cos_physical"


def test_ashare_capital_daily_staleness_gate():
    """StockCapitalDaily 为 S1 股本快照，字典实测滞后行情（最末日 2026-06-14 vs
    行情 2026-07-31）——必须携带 freshness/staleness 门槛，禁止同日当 S1-ready。"""
    c = COS_DATASET_CONTRACTS["ashare_stock_capital_daily"]
    assert c.temporal_model == "S1"
    assert c.max_staleness is not None and c.max_staleness != ""  # 必须有滞后门槛


def test_ashare_physical_completeness_filled():
    """R45 完整性：可确证 COS A 股契约应带 coverage/cadence；TopTen 快照自然日 S1。"""
    for name in _TOP_TEN:
        c = COS_DATASET_CONTRACTS[name]
        assert c.calendar_domain == "calendar_day"
        assert c.coverage_start is not None
        assert c.expected_cadence is not None
        # PIT vintage 生产门槛：note 必须声明生产采用需证明真 PIT vintage
        assert "PIT vintage" in c.note, name
        assert "DATA_GATED" in c.note, name


def test_ashare_e1_finance_fills_cadence_staleness():
    for name in _E1_FIN:
        c = COS_DATASET_CONTRACTS[name]
        assert c.coverage_start is not None
        assert c.expected_cadence is not None
        assert c.max_staleness is not None
        assert c.missing_partition_semantics == "warn"
        assert c.availability_column == "PubDate"
        assert c.pit_fidelity == "knowledge_date_pit"
