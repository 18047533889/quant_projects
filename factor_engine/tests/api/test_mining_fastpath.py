# -*- coding: utf-8
"""Mining API fastpath allowlist / 校验。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_production_fastpath_dsl_accepts_ts_mean(_loaded):
    from api.mining_integration import validate_production_fastpath_dsl

    ok, msg = validate_production_fastpath_dsl("ts_mean(col('close'), 5)")
    assert ok, msg


def test_fastpath_allowlists_json_shape(_loaded):
    from api.mining_integration import export_fastpath_allowlists_json

    payload = export_fastpath_allowlists_json()
    assert "research_allowlist" in payload
    assert "production_allowlist" in payload
    assert "production_fastpath_allowlist" in payload
    assert payload["counts"]["production_fastpath"] <= payload["counts"]["production"]


def test_validate_manifest_fastpath_when_env(monkeypatch):
    from api.mining_integration import validate_manifest_for_execution

    monkeypatch.setenv("FACTOR_ENGINE_MINING_REQUIRE_FASTPATH", "1")
    ok, msg = validate_manifest_for_execution(
        market="us",
        expression_type="dsl",
        formula="ts_mean(close, 3)",
    )
    assert ok is True, msg


def test_default_mining_operator_allowlist_fastpath(_loaded):
    from api.mining_integration import default_mining_operator_allowlist

    ops = default_mining_operator_allowlist(tier="production_fastpath")
    assert "ts_mean" in ops


def test_default_mining_search_space_config(_loaded):
    from api.mining_integration import default_mining_search_space_config

    cfg = default_mining_search_space_config(tier="production_fastpath")
    assert cfg["allowlist_tier"] == "production_fastpath"
    assert cfg["require_fastpath_validation"] is True
    assert cfg["count"] == len(cfg["operators"])
    assert "ts_mean" in cfg["operators"]


def test_validate_formula_in_mining_allowlist(_loaded):
    from api.mining_integration import validate_formula_in_mining_allowlist

    ok, msg = validate_formula_in_mining_allowlist(
        "rank(ts_mean(col('close'), 3))",
        tier="production_fastpath",
    )
    assert ok, msg

    ok_bad, msg_bad = validate_formula_in_mining_allowlist(
        "ewm_corr(col('close'), col('volume'), 5)",
        tier="production_fastpath",
    )
    assert ok_bad is False
    assert "ewm_corr" in msg_bad


def test_resolve_mining_allowlist_tier_env(monkeypatch, _loaded):
    from api.mining_integration import resolve_mining_allowlist_tier

    monkeypatch.setenv("FACTOR_ENGINE_MINING_ALLOWLIST_TIER", "research")
    assert resolve_mining_allowlist_tier() == "research"
    assert resolve_mining_allowlist_tier(tier="production") == "production"
