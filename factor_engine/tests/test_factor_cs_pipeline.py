# -*- coding: utf-8
"""factor_cs_pipeline 模板与 PR5 datasets 契约。"""
from __future__ import annotations

from pathlib import Path

from runtime.config import load_config


def test_factor_cs_pipeline_profile_loads():
    root = Path(__file__).resolve().parent.parent
    cfg = root / "examples" / "profiles" / "factor_cs_pipeline.yaml"
    loaded = load_config(cfg)
    assert loaded.factor.name == "cs_momentum_z"
    assert "winsorize" in loaded.factor.expr
    assert loaded.data_source.options["dataset"] == "us_stock_daily"
    assert loaded.materialization is not None
    assert loaded.materialization.factor_id == "cs_momentum_z_v1"


def test_pr5_datasets_registered_with_schema():
    from api.datasets_contract import audit_pr5_datasets_contract

    report = audit_pr5_datasets_contract()
    assert report["ok"], report["violations"]
