# -*- coding: utf-8 -*-
"""Mining preset / prod profile 与 datasets.yaml 契约测试。"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FE_ROOT = Path(__file__).resolve().parents[2]


def test_mining_presets_reference_registered_datasets():
    from factor_engine.api.datasets_contract import audit_mining_dataset_contract

    report = audit_mining_dataset_contract()
    assert report["ok"], report["violations"]
    assert "ashare_stock_daily" in report["datasets_checked"]
    assert "us_stock_daily" in report["datasets_checked"]
    assert "daily_market_summary" in report["datasets_checked"]
    assert "stocks_floats" in report["datasets_checked"]
    assert "us_stocks_sip_day_aggs" in report["datasets_checked"]
    assert "financials_ratios" in report["datasets_checked"]


def test_prod_profile_staging_clickhouse_requires_lake_datasets():
    from factor_engine.api.datasets_contract import audit_prod_profile_contract

    report = audit_prod_profile_contract(profile_name="prod")
    assert report["ok"], report["violations"]
    assert report["materialization_target"] == "staging_clickhouse"


def test_validate_dataset_field_bindings_catches_missing_column(tmp_path):
    from factor_engine.api.datasets_contract import validate_dataset_field_bindings
    from data_access.registry import load_registry

    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "demo_ds": {
                    "kind": "static",
                    "access_mode": "published",
                    "layout": "plain",
                    "root": str(tmp_path / "data"),
                    "glob": "**/*.parquet",
                    "time_column": "TradeDate",
                    "instrument_column": "Symbol",
                    "hive_partitioning": False,
                    "union_by_name": True,
                    "schema": {"TradeDate": "date", "Symbol": "string", "Close": "double"},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    reg = load_registry(cfg)
    violations = validate_dataset_field_bindings(
        {"demo_ds": {"close": "MissingCol"}},
        registry=reg,
    )
    assert any("MissingCol" in v for v in violations)


FE_ROOT = Path(__file__).resolve().parents[2]


def test_us_polygon_daily_profile_loads():
    from factor_engine.runtime.config import load_config

    cfg = FE_ROOT / "examples" / "profiles" / "us_polygon_daily.yaml"
    loaded = load_config(cfg)
    assert loaded.data_source.options["dataset"] == "daily_market_summary"
    assert loaded.factor.name == "us_polygon_momentum"


def test_us_polygon_floats_profile_loads():
    from factor_engine.runtime.config import load_config

    loaded = load_config(FE_ROOT / "examples" / "profiles" / "us_polygon_floats.yaml")
    assert loaded.factor.name == "us_liquidity_mom"
    assert loaded.data_source.type == "composite"
    assert loaded.data_source.options["aliases"]["free_float_percent"] == "floats.free_float_percent"


def test_us_sip_day_aggs_profile_loads():
    from factor_engine.runtime.config import load_config

    loaded = load_config(FE_ROOT / "examples" / "profiles" / "us_sip_day_aggs.yaml")
    assert loaded.factor.name == "us_sip_momentum"
    assert loaded.data_source.type == "data_access"
    assert loaded.data_source.options["dataset"] == "us_stocks_sip_day_aggs"


def test_pr5_datasets_have_schema():
    from factor_engine.api.datasets_contract import audit_pr5_datasets_contract

    report = audit_pr5_datasets_contract()
    assert report["ok"], report["violations"]
    assert "financials_ratios" in report["datasets"]
    assert "us_stocks_sip_minute_aggs" in report["datasets"]


def test_us_sip_fundamental_profile_loads():
    from factor_engine.runtime.config import load_config

    loaded = load_config(FE_ROOT / "examples" / "profiles" / "us_sip_fundamental.yaml")
    assert loaded.factor.name == "us_sip_pe_rank"
    assert loaded.data_source.type == "composite"
    assert loaded.data_source.options["aliases"]["pe"] == "ratios.price_to_earnings"


def test_mining_presets_json_synced():
    import subprocess
    import sys

    env = {**dict(__import__("os").environ), "PYTHONPATH": f"{FE_ROOT.parent}:{FE_ROOT}"}
    for script in (
        "export_mining_data_source_presets.py",
        "sync_factor_engine_llm_prompt_txt.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(FE_ROOT / "scripts" / script), "--check"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(FE_ROOT),
        )
        assert proc.returncode == 0, f"{script}: {proc.stderr or proc.stdout}"


def test_full_contract_script_entrypoint():
    import subprocess
    import sys

    script = FE_ROOT / "scripts" / "validate_datasets_mining_alignment.py"
    env = {**dict(__import__("os").environ), "PYTHONPATH": f"{FE_ROOT.parent}:{FE_ROOT}"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(FE_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_explicit_operator_policies_cover_core_tier1():
    from factor_engine.cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    core = {
        "ts_mean",
        "ts_std",
        "rank",
        "zscore",
        "winsorize",
        "normalize",
        "ts_delay",
        "ts_rank",
        "ts_ema",
    }
    missing = sorted(core - set(_EXPLICIT_POLICIES))
    assert not missing, f"核心算子缺少显式 policy: {missing}"
    assert len(_EXPLICIT_POLICIES) >= 50
