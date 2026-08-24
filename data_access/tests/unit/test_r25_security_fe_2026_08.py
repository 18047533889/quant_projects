# -*- coding: utf-8 -*-
"""R25 §95/§98/§99 —— security / FE 边界 / HTTP strict 测试。

    §95 security 保留 R24（server A basic / server B premium / 403 不 fallback /
        production 不 parse ~/.cos.yaml / STS token / derived premium factor reject）
    T-FE-001  FactorEngine production raw parquet path → reject（allowlist/GovernedFrame）
    T-FE-002  automated research ambiguous semantic field → reject，不 warning continue
    T-FE-003  Factor from premium dataset → materialized classification premium
    T-FE-004  naked DataFrame unknown provenance → production reject
    §98 HTTP：unknown fields -> 422（extra=forbid）
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# T-FE-004 — naked DataFrame unknown provenance
# ---------------------------------------------------------------------------
def test_tfe004_naked_frame_production_rejected():
    from data_access.core.exceptions import UnknownProvenanceError
    from data_access.security.governed_frame import (
        ExecutionEnvironmentIdentity,
        GovernedFrame,
        require_governed_provenance,
    )
    from data_access.snapshot.source_snapshot import (
        ResolvedObject,
        ResolvedSourceSnapshot,
    )

    with pytest.raises(UnknownProvenanceError):
        require_governed_provenance({"a": 1}, run_mode="production")
    # R26-P0-023：缺真实 provenance 的 GovernedFrame（伪造帧）production 拒绝。
    gf = GovernedFrame(
        table_or_frame={"a": 1},
        source_snapshot="s",
        lineage="l",
        security_digest="d",
    )
    with pytest.raises(UnknownProvenanceError, match="无法证明 provenance"):
        require_governed_provenance(gf, run_mode="production")
    # 带完整 provenance 的 GovernedFrame production 通过。
    snap = ResolvedSourceSnapshot(
        dataset="t",
        objects=(ResolvedObject(uri="s3://b/t/f.parquet", etag="e1", content_length=1),),
        content_digest="abc123",
    )
    gf_ok = GovernedFrame(
        table_or_frame={"a": 1},
        source_snapshot=snap,
        lineage={"dataset": "t", "params": {}},
        execution_environment=ExecutionEnvironmentIdentity(run_mode="production"),
        security_digest="valid-digest",
    )
    out = require_governed_provenance(gf_ok, run_mode="production")
    assert out.has_provenance
    # research 显式 unsafe_external_frame 放行
    research = require_governed_provenance(
        {"a": 1}, run_mode="research", unsafe_external_frame=True
    )
    assert research.security_digest == "unsafe_external_frame"


# ---------------------------------------------------------------------------
# T-FE-002 — automated research strict semantic (INV-07)
# ---------------------------------------------------------------------------
def test_tfe002_automated_research_strict():
    from data_access.read.query_budget import is_strict_semantics
    from data_access.security.run_mode import RunMode

    assert RunMode("automated_research").strict_semantics is True
    assert RunMode("interactive_research").strict_semantics is False
    # automated_research 不允许 publish
    from data_access.core.exceptions import ValidationError
    from data_access.security.run_mode import require_publish_permission

    with pytest.raises(ValidationError, match="publish"):
        require_publish_permission(RunMode("automated_research"))


def test_tfe002b_automated_research_is_strict_semantics(monkeypatch):
    """automated_research 通过 is_strict_semantics() 生效（INV-07）。"""
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.setenv("DATA_ACCESS_RUN_MODE", "automated_research")
    from data_access.read.query_budget import is_strict_semantics

    assert is_strict_semantics() is True
    monkeypatch.delenv("DATA_ACCESS_RUN_MODE", raising=False)


# ---------------------------------------------------------------------------
# T-FE-003 — Factor derived classification (premium propagates)
# ---------------------------------------------------------------------------
def test_tfe003_derived_classification_propagates():
    from factor_engine.security.access import (
        derive_derived_access_tags,
        max_sensitivity,
    )

    source_tags = ("alt.premium", "internal.restricted")
    derived = derive_derived_access_tags(source_tags)
    assert "alt.premium" in derived
    assert "internal.restricted" in derived
    # internal.restricted=50 > public=10；max = 50（不降密）
    assert max_sensitivity(("public", "internal.restricted")) == 50
    assert max_sensitivity(("public",)) == 10


# ---------------------------------------------------------------------------
# §95 / T-S — production secret boundary / derived factor reject（R24 已覆盖，
# 这里补 production 不 parse ~/.cos.yaml 的回归）
# ---------------------------------------------------------------------------
def test_s95_production_no_cos_yaml_parse(monkeypatch, tmp_path):
    from data_access.core.exceptions import ValidationError
    from data_access.security.credentials import (
        CosCliConfigProvider,
        allow_coscli_config_parse,
    )

    monkeypatch.delenv("DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE", raising=False)
    assert allow_coscli_config_parse() is False  # 默认不解析
    # research 显式 opt-in
    monkeypatch.setenv("DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE", "1")
    assert allow_coscli_config_parse() is True


# ---------------------------------------------------------------------------
# §98 HTTP — unknown fields -> 422 (extra=forbid)
# ---------------------------------------------------------------------------
def test_s98_http_unknown_field_422(monkeypatch):
    from fastapi.testclient import TestClient

    from data_access.security.api_principals import reset_api_principal_registry
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    monkeypatch.setenv("DATA_ACCESS_API_PRINCIPALS", "{}")
    reset_api_principal_registry()
    app = create_app(
        ServiceSettings(api_key="testkey", production_mode=False, allow_open=True)
    )
    client = TestClient(app)
    resp = client.post(
        "/v1/read",
        json={
            "dataset": "ashare_stock_daily",
            "columns": ["close"],
            "unknown_field_xyz": 1,  # R25 §59：extra=forbid → 422
        },
        headers={"X-API-Key": "testkey"},
    )
    assert resp.status_code == 422


def test_s98b_http_failed_column_length(monkeypatch):
    """列名超长 → 422（R25 §59）。"""
    from fastapi.testclient import TestClient

    from data_access.security.api_principals import reset_api_principal_registry
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    monkeypatch.setenv("DATA_ACCESS_API_PRINCIPALS", "{}")
    reset_api_principal_registry()
    app = create_app(
        ServiceSettings(api_key="testkey", production_mode=False, allow_open=True)
    )
    client = TestClient(app)
    long_col = "x" * 300  # > MAX_COLUMN_LEN=128
    resp = client.post(
        "/v1/read",
        json={"dataset": "d", "columns": [long_col]},
        headers={"X-API-Key": "testkey"},
    )
    assert resp.status_code == 422
