from __future__ import annotations

from factor_engine.api.mining_integration import (
    export_dsl_allowlist_json,
    validate_manifest_for_execution,
    validate_us_dsl,
)


def test_validate_us_dsl_ok():
    ok, msg = validate_us_dsl("rank(ts_mean(close, 3))")
    assert ok is True
    assert msg == "OK"


def test_validate_us_dsl_rejects_unknown_op():
    ok, msg = validate_us_dsl("totally_unknown_op(close)")
    assert ok is False
    assert "totally_unknown_op" in msg or "Unknown" in msg or msg


def test_validate_manifest_ashare_dsl():
    ok, msg = validate_manifest_for_execution(
        market="ashare",
        expression_type="dsl",
        formula="rank(ts_mean(close, 3))",
    )
    assert ok is True
    assert msg == "OK"


def test_validate_manifest_legacy_lqtp_dsl_type():
    """历史 manifest 可能仍标 lqtp_dsl；语法校验与 dsl 相同。"""
    ok, msg = validate_manifest_for_execution(
        market="ashare",
        expression_type="lqtp_dsl",
        formula="rank(ts_mean(close, 3))",
    )
    assert ok is True
    assert msg == "OK"


def test_export_dsl_allowlist_contains_rank():
    payload = export_dsl_allowlist_json(market="us")
    assert payload["operator_policy"] == "afv_us_pv_daily"
    assert payload["market"] == "us"
    assert "rank" in payload["operators"]
    assert "col" in payload["operators"]


def test_export_dsl_allowlist_ashare_policy():
    payload = export_dsl_allowlist_json(market="ashare")
    assert payload["operator_policy"] == "lqtp_pv_daily"
    assert payload["market"] == "ashare"


def test_default_ashare_pv_data_source_uses_data_access_without_max_files():
    from factor_engine.api.mining_integration import default_ashare_pv_data_source_config

    cfg = default_ashare_pv_data_source_config()
    assert cfg["type"] == "data_access"
    assert cfg["dataset"] == "ashare_stock_daily_adj"
    assert cfg["fields"]["close"] == "AdjClose"
    assert cfg["fields"]["vwap"] == "AdjVwap"


def test_default_us_pv_valuation_composite():
    from factor_engine.api.mining_integration import default_us_pv_valuation_data_source_config

    cfg = default_us_pv_valuation_data_source_config()
    assert cfg["type"] == "composite"
    # US valuation/indicator 是 X0 稀疏当前快照（snapshot_only），不能 asof 回填进
    # 历史日线（asof_backward 会发明历史）——current_only 是正确设计
    # （api.mining_integration.default_us_pv_valuation_data_source_config）。
    assert cfg["joins"]["valuation"] == "current_only"
    assert cfg["sources"]["valuation"]["dataset"] == "us_stock_valuation_daily"
    assert cfg["sources"]["pv"]["dataset"] == "us_stock_daily"
    # X0 快照真实列名是 price_to_earnings（P0-013 修复：旧 A 股字段名 PeRatio 是 bug）。
    assert cfg["aliases"]["pe"] == "valuation.price_to_earnings"


def test_default_us_stocks_sip_day_aggs_uses_data_access():
    from factor_engine.api.mining_integration import default_us_stocks_sip_day_aggs_data_source_config

    cfg = default_us_stocks_sip_day_aggs_data_source_config()
    assert cfg["type"] == "data_access"
    assert cfg["dataset"] == "us_stocks_sip_day_aggs"
    assert cfg["fields"]["close"] == "close"

    smoke = default_us_stocks_sip_day_aggs_data_source_config(max_files=3)
    assert smoke["type"] == "data_access"
    assert smoke["dataset"] == "us_stocks_sip_day_aggs"
    assert smoke["start_date"] == "2024-01-01"


def test_default_ashare_pv_valuation_has_pe_alias():
    from factor_engine.api.mining_integration import default_ashare_pv_valuation_data_source_config

    cfg = default_ashare_pv_valuation_data_source_config()
    assert cfg["aliases"]["pe"] == "valuation.pe"


def test_default_us_pv_data_source_uses_data_access():
    from factor_engine.api.mining_integration import default_us_pv_data_source_config

    cfg = default_us_pv_data_source_config()
    assert cfg["type"] == "data_access"
    assert cfg["dataset"] == "us_stock_daily"
    assert cfg["fields"]["close"] == "Close"

    smoke = default_us_pv_data_source_config(max_files=5)
    assert smoke["type"] == "parquet"
    assert smoke["root"].startswith("~/quant_projects/")


def test_default_ashare_pv_data_source():
    from factor_engine.api.mining_integration import default_ashare_pv_data_source_config

    cfg = default_ashare_pv_data_source_config(
        max_files=5,
        start_date="2019-01-01",
        end_date="2019-12-31",
    )
    # ADJ_FIELD_MIGRATION: max_files no longer switches to a direct parquet read —
    # the config stays data_access / ashare_stock_daily_adj (no non-data_access IO).
    assert cfg["type"] == "data_access"
    assert cfg["dataset"] == "ashare_stock_daily_adj"
    assert cfg["fields"]["vwap"] == "AdjVwap"
    assert cfg["fields"]["close"] == "AdjClose"
    assert cfg["max_files"] == 5
    assert cfg["start_date"] == "2019-01-01"
    assert cfg["end_date"] == "2019-12-31"


def test_default_ashare_pv_universe_composite():
    from factor_engine.api.mining_integration import default_ashare_pv_universe_data_source_config

    cfg = default_ashare_pv_universe_data_source_config(index_symbol="000300.SH")
    assert cfg["type"] == "composite"
    assert cfg["joins"]["status"] == "asof_backward"
    assert cfg["joins"]["constituent"] == "exact"
    assert cfg["sources"]["constituent"]["dataset"] == "ashare_index_constituent"
    assert cfg["sources"]["constituent"]["semantic_filters"]["IndexSymbol"] == "000300.SH"
    assert cfg["sources"]["status"]["fields"]["public_status"] == "ListedState"


def test_default_us_sip_day_ratios_composite():
    from factor_engine.api.mining_integration import default_us_sip_day_ratios_composite_config

    cfg = default_us_sip_day_ratios_composite_config()
    assert cfg["type"] == "composite"
    assert cfg["joins"]["ratios"] == "asof_backward"
    assert cfg["sources"]["price"]["dataset"] == "us_stocks_sip_day_aggs"
    assert cfg["sources"]["ratios"]["dataset"] == "financials_ratios"
    assert cfg["aliases"]["pe"] == "ratios.price_to_earnings"


def test_default_us_sip_cash_flow_composite():
    from factor_engine.api.mining_integration import default_us_sip_cash_flow_composite_config

    cfg = default_us_sip_cash_flow_composite_config()
    assert cfg["joins"]["cash_flow"] == "asof_backward"
    assert cfg["aliases"]["operating_cf"] == "cash_flow.net_cash_from_operating_activities"


def test_default_financials_ratios_uses_data_access():
    from factor_engine.api.mining_integration import default_financials_ratios_data_source_config

    cfg = default_financials_ratios_data_source_config()
    assert cfg["type"] == "data_access"
    assert cfg["dataset"] == "financials_ratios"
    assert cfg["fields"]["pe"] == "price_to_earnings"


def test_default_us_pv_universe_composite():
    from factor_engine.api.mining_integration import default_us_pv_universe_data_source_config

    cfg = default_us_pv_universe_data_source_config()
    assert cfg["type"] == "composite"
    assert cfg["joins"]["universe"] == "exact"
    assert cfg["sources"]["universe"]["dataset"] == "us_universe_daily"


def test_default_us_stocks_sip_quotes_uses_data_access():
    from factor_engine.api.mining_integration import default_us_stocks_sip_quotes_data_source_config

    cfg = default_us_stocks_sip_quotes_data_source_config()
    assert cfg["type"] == "data_access"
    assert cfg["dataset"] == "us_stocks_sip_quotes"
    assert cfg["normalize_timestamp"] is True
    assert cfg["timestamp_unit"] == "ns"


def test_default_massive_ticks_parametric_kind():
    from factor_engine.api.mining_integration import default_massive_ticks_data_source_config

    cfg = default_massive_ticks_data_source_config(kind="quotes_v1")
    assert cfg["dataset"] == "massive_ticks"
    assert cfg["kind"] == "quotes_v1"
    assert "bid_price" in cfg["fields"]
