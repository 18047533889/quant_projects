# -*- coding: utf-8 -*-
"""R24 P0-S1..S5 / P1-S6..S9 安全 destructive tests（§30 T-S01..T-S13）。

覆盖：
    T-S01  server scope：principal 只允许 ashare → us AuthorizationError
    T-S02  不得 privilege fallback：403 → AccessDeniedError，绝不换 credential 重试
    T-S03  production secret boundary：不解析 ~/.cos.yaml、不注入 env
    T-S04  repr redaction：repr(S3Credentials) 无 secret/token
    T-S05  STS：access key + secret + session token 后端可用
    T-S06  high cache 不可被低权限复用：cache root 按 principal 隔离
    T-S07  权限过宽：production cache root world-readable → fail-closed
    T-S08  symlink/tmp 攻击：下载不能覆盖 root 之外目标
    T-S09  filtered dataset catalog：/v1/datasets 不暴露 premium dataset 名
    T-S10  basic principal POST /v1/read premium dataset → 403（backend 前）
    T-S11  basic principal /v1/read_uri premium prefix → 403
    T-S12  production + ALLOW_OPEN=1 → 仍要求 auth（已在 test_read_service 覆盖）
    T-S13  premium source 派生因子 → basic principal 读 → 403
"""
from __future__ import annotations

import os
import stat as _stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    AccessDeniedError,
    AuthorizationError,
    ValidationError,
)
from data_access.registry import load_registry
from data_access.security.api_principals import (
    ApiPrincipalRegistry,
    reset_api_principal_registry,
)
from data_access.security.credentials import EnvCredentialProvider
from data_access.security.principal import AccessPolicy, DataPrincipal
from data_access.security.policy import DefaultAuthorizer, set_authorizer
from data_access.store import DataAccessStore


def _scoped_store(
    tmp_path,
    *,
    allowed_datasets: tuple[str, ...] = ("ashare_stock_daily",),
    allow_uri_read: bool = False,
    principal_id: str = "server-a",
    strict: bool = True,
) -> DataAccessStore:
    policy = AccessPolicy(
        allowed_datasets=frozenset(allowed_datasets),
        allowed_actions=frozenset(
            {
                "dataset:list",
                "dataset:read",
                "factor:list",
                "factor:read",
                "metadata:read",
            }
        ),
        allow_uri_read=allow_uri_read,
    )
    principal = DataPrincipal(principal_id=principal_id, server_id=principal_id)
    authorizer = DefaultAuthorizer(
        policy=policy, principal=principal, strict_default_deny=strict
    )
    registry = load_registry()
    engine = DuckDBEngine(threads=2, enable_object_cache=False)
    return DataAccessStore(
        registry=registry, engine=engine, principal=principal, authorizer=authorizer
    )


@pytest.fixture(autouse=True)
def _clean_globals(monkeypatch):
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("DATA_ACCESS_PRINCIPAL_ID", raising=False)
    monkeypatch.delenv("DATA_ACCESS_ALLOWED_DATASETS", raising=False)
    monkeypatch.delenv("DATA_ACCESS_API_PRINCIPALS", raising=False)
    set_authorizer(None)
    reset_api_principal_registry()
    yield
    set_authorizer(None)
    reset_api_principal_registry()


# ---------------------------------------------------------------------------
# T-S01 — server scope
# ---------------------------------------------------------------------------
def test_ts01_server_scope_ashare_allowed_us_denied(tmp_path):
    store = _scoped_store(tmp_path, allowed_datasets=("ashare_stock_daily",))
    store.authorize_dataset("ashare_stock_daily")  # 不抛
    with pytest.raises(AuthorizationError):
        store.authorize_dataset("us_stock_daily")


# ---------------------------------------------------------------------------
# T-S02 — 不得 privilege fallback
# ---------------------------------------------------------------------------
def test_ts02_no_privilege_fallback():
    """受限 httpfs credential 403 → AccessDeniedError（不是 backend 错误，不换身份）。"""
    import duckdb

    err = duckdb.Error("HTTP Error: 403 Forbidden - access denied for object")
    from data_access.core.retry import ErrorClass, classify_exception

    assert classify_exception(err) == ErrorClass.AUTH
    from data_access.core.engine import DuckDBEngine

    wrapped = DuckDBEngine._wrap_query_error("SELECT 1", err)
    assert isinstance(wrapped, AccessDeniedError)


# ---------------------------------------------------------------------------
# T-S03 — production secret boundary
# ---------------------------------------------------------------------------
def test_ts03_production_secret_boundary(monkeypatch, tmp_path):
    monkeypatch.delenv("COS_SECRET_ID", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    cfg = tmp_path / "cos.yaml"
    cfg.write_text(
        "cos:\n  base:\n    secretid: 'WIDE_ID'\n    secretkey: 'WIDE_KEY'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ACCESS_COS_YAML", str(cfg))
    monkeypatch.setenv("DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE", "1")
    from data_access.cos.remote import resolve_s3_credentials

    with pytest.raises(ValidationError):
        resolve_s3_credentials()
    assert os.environ.get("COS_SECRET_ID") is None
    assert os.environ.get("COS_SECRET_KEY") is None


# ---------------------------------------------------------------------------
# T-S04 — repr redaction
# ---------------------------------------------------------------------------
def test_ts04_repr_redaction():
    from data_access.cos.remote import S3Credentials

    creds = S3Credentials(
        access_key_id="AKIDTEST123456",
        secret_access_key="super_secret_key_xyz",
        endpoint="cos.ap-guangzhou.myqcloud.com",
        region="ap-guangzhou",
        session_token="sts_token_abc",
    )
    r = repr(creds)
    assert "super_secret_key_xyz" not in r
    assert "sts_token_abc" not in r
    assert "AKIDTEST123456" not in r  # SecretId 全量不显示
    safe = creds.to_safe_dict()
    assert "super_secret_key_xyz" not in str(safe)
    assert "secret_access_key" not in str(safe)


# ---------------------------------------------------------------------------
# T-S05 — STS session token
# ---------------------------------------------------------------------------
def test_ts05_sts_session_token(monkeypatch):
    monkeypatch.setenv("COS_SECRET_ID", "STS_ID")
    monkeypatch.setenv("COS_SECRET_KEY", "STS_KEY")
    monkeypatch.setenv("COS_SESSION_TOKEN", "STS_TOKEN")
    monkeypatch.setenv("DATA_ACCESS_COS_S3_ENDPOINT", "cos.example.com")
    material = EnvCredentialProvider().resolve()
    assert material.has_session_token
    assert material.session_token == "STS_TOKEN"
    from data_access.cos.remote import resolve_s3_credentials

    creds = resolve_s3_credentials()
    assert creds.has_session_token
    assert creds.session_token == "STS_TOKEN"


# ---------------------------------------------------------------------------
# T-S06 — high cache 不可被低权限复用（principal 隔离）
# ---------------------------------------------------------------------------
def test_ts06_cache_principal_isolation(monkeypatch):
    monkeypatch.delenv("DATA_ACCESS_PRINCIPAL_ID", raising=False)
    monkeypatch.setenv("QUANTSOCIETY_WORKSPACE_DATA_ROOT", "/tmp/ws")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    from data_access.cos.remote import _cache_principal_scope, cos_cache_root

    s1 = _cache_principal_scope()
    monkeypatch.setenv("DATA_ACCESS_PRINCIPAL_ID", "server-high")
    s2 = _cache_principal_scope()
    assert s1 != s2, "不同 principal 必须不同 cache scope"
    r1 = cos_cache_root()
    monkeypatch.setenv("DATA_ACCESS_PRINCIPAL_ID", "server-low")
    r2 = cos_cache_root()
    assert r1 != r2
    assert "server-high" in str(r1)
    assert "server-low" in str(r2)


# ---------------------------------------------------------------------------
# T-S07 — 权限过宽 fail-closed
# ---------------------------------------------------------------------------
def test_ts07_unsafe_permissions(tmp_path, monkeypatch):
    root = tmp_path / "cache_root"
    root.mkdir(mode=0o777)
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_CACHE_ROOT", str(tmp_path))
    from data_access.cos.remote import ensure_cache_root_secure

    with pytest.raises(ValidationError, match="world"):
        ensure_cache_root_secure(root)


# ---------------------------------------------------------------------------
# T-S08 — symlink/tmp 攻击
# ---------------------------------------------------------------------------
def test_ts08_symlink_attack_refused(tmp_path, monkeypatch):
    """恶意预建 symlink 作为目标 → 下载拒绝覆盖 root 之外目标。"""
    target = tmp_path / "dest.parquet"
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim"
    victim.write_bytes(b"precious")
    target.symlink_to(victim)

    from data_access.cos.mirror import _sync_cos_file

    with pytest.raises(ValidationError, match="symlink"):
        _sync_cos_file("cos://b/k/f.parquet", target)
    assert victim.read_bytes() == b"precious"  # 未被覆盖


def test_ts08b_secure_temp_not_predictable(tmp_path, monkeypatch):
    """临时文件用 O_EXCL 随机名，不再用可预测的 .tmp。"""
    dest = tmp_path / "f.parquet"
    from data_access.cos.mirror import _sync_cos_file

    def fake_cp(args):
        # fake clean-cos-ro 写文件到最后一个 arg（tmp 路径）
        target = Path(args[-1])
        target.write_bytes(b"\x00\x01parquet")

    monkeypatch.setattr("data_access.cos.mirror._run_cos_cli", fake_cp)
    _sync_cos_file("cos://b/k/f.parquet", dest)
    assert dest.exists()
    # 随机 O_EXCL tmp 用完即清理，不残留可预测 .tmp
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# T-S09 / T-S10 / T-S11 — HTTP scopes
# ---------------------------------------------------------------------------
def _http_client(principal_json: str, monkeypatch):
    from fastapi.testclient import TestClient

    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    monkeypatch.setenv("DATA_ACCESS_API_PRINCIPALS", principal_json)
    reset_api_principal_registry()
    app = create_app(
        ServiceSettings(
            api_key="testkey",
            production_mode=False,
            allow_open=False,
        )
    )
    return TestClient(app)


def _mock_store_with_premium():
    store = MagicMock()
    # premium_alt_data + basic_market 两个数据集
    basic = MagicMock(
        name="basic_market",
        kind="static",
        access_mode="published",
        time_column="t",
        instrument_column="s",
        params_schema={},
    )
    premium = MagicMock(
        name="premium_alt_data",
        kind="static",
        access_mode="published",
        time_column="t",
        instrument_column="s",
        params_schema={},
    )
    store.registry.names.return_value = ["basic_market", "premium_alt_data"]
    store.registry.get.side_effect = lambda n: basic if n == "basic_market" else premium
    store.registry.registry_fingerprint.return_value = "fp"
    return store


def test_ts09_filtered_dataset_catalog(monkeypatch):
    from data_access.security.api_principals import hash_api_key

    key = "basic-key"
    cfg = {
        hash_api_key(key): {
            "principal_id": "basic",
            "allowed_datasets": ["basic_market"],
            "allowed_actions": ["dataset:list", "dataset:read", "metadata:read"],
        }
    }
    import json as _json

    client = _http_client(_json.dumps(cfg), monkeypatch)
    with patch("data_access.service.app.get_store", return_value=_mock_store_with_premium()):
        resp = client.get("/v1/datasets", headers={"X-API-Key": key})
    assert resp.status_code == 200
    names = [d["name"] for d in resp.json()]
    assert "basic_market" in names
    assert "premium_alt_data" not in names  # T-S09：premium dataset 名不暴露


def test_ts10_http_read_premium_denied(monkeypatch):
    from data_access.security.api_principals import hash_api_key

    key = "basic-key"
    cfg = {
        hash_api_key(key): {
            "principal_id": "basic",
            "allowed_datasets": ["basic_market"],
            "allowed_actions": ["dataset:list", "dataset:read"],
        }
    }
    import json as _json

    client = _http_client(_json.dumps(cfg), monkeypatch)
    with patch("data_access.service.app.get_store", return_value=_mock_store_with_premium()):
        resp = client.post(
            "/v1/read",
            json={"dataset": "premium_alt_data", "columns": ["a"], "format": "parquet"},
            headers={"X-API-Key": key},
        )
    assert resp.status_code == 403  # backend 之前就被拒


def test_ts11_http_read_uri_denied(monkeypatch):
    from data_access.security.api_principals import hash_api_key

    key = "basic-key"
    cfg = {
        hash_api_key(key): {
            "principal_id": "basic",
            "allowed_datasets": ["basic_market"],
            "allowed_actions": ["dataset:list", "dataset:read"],
        }
    }
    import json as _json

    client = _http_client(_json.dumps(cfg), monkeypatch)
    with patch("data_access.service.app.get_store", return_value=_mock_store_with_premium()):
        resp = client.post(
            "/v1/read_uri",
            json={"uri": "s3://premium-bucket/key.parquet", "format": "parquet"},
            headers={"X-API-Key": key},
        )
    assert resp.status_code == 403  # uri:read 默认拒绝


# ---------------------------------------------------------------------------
# T-S13 — premium source 派生因子 → basic principal 读 → 403
# ---------------------------------------------------------------------------
def test_ts13_derived_factor_denied(tmp_path):
    from data_access.read.factors import FactorCatalog, FactorMeta

    store = _scoped_store(
        tmp_path, allowed_datasets=("factor_lake",)
    )
    # 给 store 挂一个带 namespace 限制的 policy
    policy = AccessPolicy(
        allowed_datasets=frozenset({"factor_lake"}),
        allowed_factor_namespaces=frozenset({"market.basic"}),
        allowed_actions=frozenset({"factor:list", "factor:read"}),
    )
    principal = DataPrincipal(principal_id="basic", server_id="basic")
    authorizer = DefaultAuthorizer(
        policy=policy, principal=principal, strict_default_deny=True
    )
    store._authorizer_sec = authorizer
    store._access_policy = policy

    meta = FactorMeta(
        factor_id="alpha_secret",
        derived_access_tags=("internal.restricted",),
        source_access_tags=("internal.restricted",),
    )

    with patch.object(store, "get_factor_catalog") as gc:
        cat = MagicMock()
        cat.records = {"alpha_secret": meta}
        gc.return_value = cat
        with pytest.raises(AccessDeniedError, match="factor access tags"):
            store.read_factors(["alpha_secret"], layout="long")
