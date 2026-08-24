# -*- coding: utf-8
"""Mining API fastpath allowlist / 校验。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_production_fastpath_dsl_accepts_ts_mean(_loaded):
    from factor_engine.api.mining_integration import validate_production_fastpath_dsl

    ok, msg = validate_production_fastpath_dsl("ts_mean(col('close'), 5)")
    assert ok, msg


def test_fastpath_allowlists_json_shape(_loaded):
    from factor_engine.api.mining_integration import export_fastpath_allowlists_json

    payload = export_fastpath_allowlists_json()
    assert "research_allowlist" in payload
    assert "production_allowlist" in payload
    assert "production_fastpath_allowlist" in payload
    assert payload["counts"]["production_fastpath"] <= payload["counts"]["production"]


def test_validate_manifest_fastpath_when_env(monkeypatch):
    from factor_engine.api.mining_integration import validate_manifest_for_execution

    monkeypatch.setenv("FACTOR_ENGINE_MINING_REQUIRE_FASTPATH", "1")
    ok, msg = validate_manifest_for_execution(
        market="us",
        expression_type="dsl",
        formula="ts_mean(close, 3)",
    )
    assert ok is True, msg


def test_default_mining_operator_allowlist_fastpath(_loaded):
    from factor_engine.api.mining_integration import default_mining_operator_allowlist

    ops = default_mining_operator_allowlist(tier="production_fastpath")
    assert "ts_mean" in ops


def test_default_mining_search_space_config(_loaded):
    from factor_engine.api.mining_integration import default_mining_search_space_config

    cfg = default_mining_search_space_config(tier="production_fastpath")
    assert cfg["schema_version"] == "factor_engine.mining_search_space.v2"
    assert cfg["allowlist_tier"] == "production_fastpath"
    assert cfg["require_fastpath_validation"] is True
    assert cfg["count"] == len(cfg["operators"])
    assert "ts_mean" in cfg["v1_allowlist"]
    assert cfg["fields"]
    assert all(field["field_expr"].startswith("field(") for field in cfg["fields"])


def test_typed_mining_search_space_v2_preserves_v1_allowlist(_loaded):
    from factor_engine.api.mining_integration import default_mining_search_space_config

    v1 = default_mining_search_space_config(tier="research", version="v1")
    v2 = default_mining_search_space_config(tier="research", version="v2", max_cost=1)
    assert v2["schema_version"] == "factor_engine.mining_search_space.v2"
    assert v2["v1_allowlist"] == v1["operators"]
    assert v2["constraints"]["max_domains"] == 2
    assert v2["field_dq_policy"] == "drop"
    assert all({"frequency", "domain", "cardinality", "unit", "dq_policy"} <= set(field) for field in v2["fields"])
    assert all({"signature", "inputs", "output", "cost", "domains"} <= set(op) for op in v2["operators"])
    earnings = next(op for op in v2["operators"] if op["name"] == "earnings_yield")
    assert earnings["domains"] == ["valuation"]
    assert earnings["unit"] == "ratio"


def test_typed_mining_rejects_invalid_constraints(_loaded):
    from factor_engine.api.mining_integration import default_mining_search_space_config

    with pytest.raises(ValueError, match="max_domains"):
        default_mining_search_space_config(tier="research", version="v2", max_domains=3)
    with pytest.raises(ValueError, match="missing required metadata"):
        default_mining_search_space_config(
            tier="research", version="v2", fields=[{"name": "close"}]
        )


def test_typed_mining_uses_field_catalog_hash_and_filters_cost(tmp_path, _loaded):
    from factor_engine.api.mining_integration import (
        default_mining_search_space_config,
        write_mining_search_space,
    )
    from factor_engine.fields import compute_field_catalog_hash

    cfg = default_mining_search_space_config(
        tier="research", version="v2", max_cost=0.5
    )
    assert cfg["field_catalog_hash"] == compute_field_catalog_hash()
    assert all(op["cost"] <= 0.5 for op in cfg["operators"])

    target = tmp_path / "space.json"
    write_mining_search_space(target, tier="research", version="v2", max_cost=0.5)
    first = target.read_bytes()
    write_mining_search_space(target, tier="research", version="v2", max_cost=0.5)
    assert target.read_bytes() == first


def test_validate_formula_in_mining_allowlist(_loaded):
    from factor_engine.api.mining_integration import validate_formula_in_mining_allowlist

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


def test_validate_manifest_python_is_research_only():
    from factor_engine.api.mining_integration import validate_manifest_for_execution

    ok, msg = validate_manifest_for_execution(
        market="ashare", expression_type="python", formula="close.mean()"
    )
    assert ok is True
    assert "valid_for_production=false" in msg
    prod_ok, prod_msg = validate_manifest_for_execution(
        market="ashare",
        expression_type="python",
        formula="close.mean()",
        require_production=True,
    )
    assert prod_ok is False
    assert "research-only" in prod_msg


def test_resolve_mining_allowlist_tier_env(monkeypatch, _loaded):
    from factor_engine.api.mining_integration import resolve_mining_allowlist_tier

    monkeypatch.setenv("FACTOR_ENGINE_MINING_ALLOWLIST_TIER", "research")
    assert resolve_mining_allowlist_tier() == "research"
    assert resolve_mining_allowlist_tier(tier="production") == "production"
