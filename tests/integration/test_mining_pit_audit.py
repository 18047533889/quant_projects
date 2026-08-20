# -*- coding: utf-8 -*-
"""Mining 默认数据源 PiT join 策略审计。"""

from __future__ import annotations


def test_mining_datasets_contract_ok():
    from api.mining_integration import audit_mining_datasets_contract

    report = audit_mining_datasets_contract()
    assert report["ok"], report.get("violations")


def test_us_daily_market_summary_preset_registered():
    from api.datasets_contract import audit_mining_dataset_contract

    report = audit_mining_dataset_contract()
    assert "daily_market_summary" in report["datasets_checked"]
    assert report["ok"], report.get("violations")


def test_default_presets_follow_explicit_join_contracts():
    from api.mining_integration import audit_default_data_source_configs

    report = audit_default_data_source_configs()
    # A-share valuation follows its exact daily table contract; event/PIT
    # composites keep their explicit backward-asof contracts.
    assert report["ashare_pv_valuation"] == []
    assert report["us_pv_valuation"] == []
    assert report["ashare_pv_universe"] == []
    assert report["us_polygon_floats"] == []
    assert report["us_sip_day_ratios"] == []
    assert report["us_sip_balance_sheet"] == []
    assert report["us_sip_cash_flow"] == []
    assert report["us_sip_income_statement"] == []
    assert report["us_sip_floats"] == []
    # 美股 universe 允许 exact（日对齐 universe 表）
    us_uni = report["us_pv_universe"]
    assert all("universe" in v for v in us_uni) or us_uni == []


def test_audit_flags_valuation_asof_join():
    from api.mining_integration import audit_composite_join_policies

    cfg = {
        "type": "composite",
        "anchor": "pv",
        "sources": {
            "pv": {},
            "valuation": {"type": "data_access", "dataset": "ashare_stock_valuation_daily"},
        },
        "joins": {"valuation": "asof_backward"},
    }
    assert audit_composite_join_policies(cfg) == [
        "valuation: join='asof_backward' (expected exact)"
    ]
