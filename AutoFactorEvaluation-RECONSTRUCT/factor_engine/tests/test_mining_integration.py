from __future__ import annotations

from api.mining_integration import (
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
    payload = export_dsl_allowlist_json()
    assert payload["operator_policy"] == "afv_us_pv_daily"
    assert "rank" in payload["operators"]
    assert "col" in payload["operators"]


def test_default_us_pv_data_source_uses_tilde_root():
    from api.mining_integration import default_us_pv_data_source_config

    cfg = default_us_pv_data_source_config()
    assert cfg["root"].startswith("~/quant_projects/")
    assert cfg["fields"]["close"] == "Close"


def test_default_ashare_pv_data_source():
    from api.mining_integration import default_ashare_pv_data_source_config

    cfg = default_ashare_pv_data_source_config(
        max_files=5,
        start_date="2019-01-01",
        end_date="2019-12-31",
    )
    assert "a_share/lqtp_data" in cfg["root"]
    assert cfg["instrument_col"] == "Symbol"
    assert cfg["fields"]["vwap"] == "Vwap"
    assert cfg["max_files"] == 5
    assert cfg["start_date"] == "2019-01-01"
