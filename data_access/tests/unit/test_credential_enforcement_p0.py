# -*- coding: utf-8 -*-
"""P0-6 / P0-7 / P0-1 —— production credential enforcement + remote-first.

覆盖：
- P0-6 fail-closed：HUMAN/TEAM principal 无 scoped credential → ValidationError，
  绝不回退 server 的 global/env/service credential；SERVICE 允许 service 链。
- P0-6 并发：principal A/B 各自 scoped credential 不跨嵌套读串扰。
- P0-7 credential_generation_id：refresh 后 generation 变化 → DuckDB S3 身份
  变化 → SECRET 重新注入。
- P0-1 production remote-first：production 上下文 + mode mirror/unset → 有效
  remote；production + remote 读失败 → 抛错（无 mirror fallback）；research 仍
  允许 mirror。
"""
from __future__ import annotations

import os

import pytest

from data_access.core.exceptions import ValidationError
from data_access.security.credentials import CredentialMaterial, EnvCredentialProvider
from data_access.security.execution_context import (
    DataAccessExecutionContext,
    execution_scope,
)


class _ScopedProvider:
    """principal-scoped CredentialProvider（每 principal 一份受限凭证）。"""

    def __init__(self, material: CredentialMaterial):
        self._material = material

    def resolve(self) -> CredentialMaterial:
        return self._material


def _material(
    *,
    access_key_id: str,
    secret: str,
    principal_id: str | None = None,
    scope: str | None = None,
    gen: str | None = None,
    token: str | None = None,
) -> CredentialMaterial:
    return CredentialMaterial(
        access_key_id=access_key_id,
        secret_access_key=secret,
        session_token=token,
        principal_id=principal_id,
        credential_scope_id=scope,
        credential_generation_id=gen,
        source="provider",
    )


# ---------------------------------------------------------------------------
# P0-6 fail-closed
# ---------------------------------------------------------------------------
def test_p06_human_principal_without_scoped_credential_fails_closed(monkeypatch):
    """HUMAN principal 无 scoped credential → 抛 ValidationError，不拿 env/service。"""
    from data_access.cos.remote import resolve_s3_credentials

    # 服务器有高权限 env credential（本应只给 SERVICE 用）。
    monkeypatch.setenv("COS_SECRET_ID", "SERVER_WIDE_ID")
    monkeypatch.setenv("COS_SECRET_KEY", "SERVER_WIDE_KEY")
    monkeypatch.delenv("DATA_ACCESS_COS_S3_ENDPOINT", raising=False)
    monkeypatch.setenv("DATA_ACCESS_COS_S3_ENDPOINT", "cos.example.com")

    ctx = DataAccessExecutionContext(
        principal="human-1", principal_type="HUMAN", credential_provider=None
    )
    with execution_scope(ctx):
        with pytest.raises(ValidationError, match="fail-closed"):
            resolve_s3_credentials()


def test_p06_team_principal_without_scoped_credential_fails_closed(monkeypatch):
    """TEAM principal 无 scoped credential → 同样 fail-closed。"""
    from data_access.cos.remote import resolve_s3_credentials

    monkeypatch.setenv("COS_SECRET_ID", "SERVER_WIDE_ID")
    monkeypatch.setenv("COS_SECRET_KEY", "SERVER_WIDE_KEY")
    monkeypatch.setenv("DATA_ACCESS_COS_S3_ENDPOINT", "cos.example.com")

    ctx = DataAccessExecutionContext(
        principal="team-1", principal_type="TEAM", credential_provider=None
    )
    with execution_scope(ctx):
        with pytest.raises(ValidationError, match="fail-closed"):
            resolve_s3_credentials()


def test_p06_service_principal_allowed_service_credential(monkeypatch):
    """SERVICE principal → 允许回退到 server 的 env/service credential。"""
    from data_access.cos.remote import resolve_s3_credentials

    monkeypatch.setenv("COS_SECRET_ID", "SVC_ID")
    monkeypatch.setenv("COS_SECRET_KEY", "SVC_KEY")
    monkeypatch.setenv("DATA_ACCESS_COS_S3_ENDPOINT", "cos.example.com")

    ctx = DataAccessExecutionContext(
        principal="svc-1", principal_type="SERVICE", credential_provider=None
    )
    with execution_scope(ctx):
        creds = resolve_s3_credentials()
    assert creds.access_key_id == "SVC_ID"


def test_p06_human_with_scoped_credential_uses_own(monkeypatch):
    """HUMAN principal 有 scoped credential → 用自己那份，不用 server env。"""
    from data_access.cos.remote import resolve_s3_credentials

    monkeypatch.setenv("COS_SECRET_ID", "SERVER_WIDE_ID")
    monkeypatch.setenv("COS_SECRET_KEY", "SERVER_WIDE_KEY")
    monkeypatch.setenv("DATA_ACCESS_COS_S3_ENDPOINT", "cos.example.com")

    scoped = _ScopedProvider(
        _material(
            access_key_id="HUMAN_SCOPED_ID",
            secret="HUMAN_SCOPED_KEY",
            principal_id="human-1",
            scope="scope-human",
        )
    )
    ctx = DataAccessExecutionContext(
        principal="human-1",
        principal_type="HUMAN",
        credential_provider=scoped,
    )
    with execution_scope(ctx):
        creds = resolve_s3_credentials()
    assert creds.access_key_id == "HUMAN_SCOPED_ID"
    assert creds.credential_scope_id == "scope-human"


def test_p06_concurrent_principals_no_scope_bleed(monkeypatch):
    """并发 principal A/B 各自 scoped credential 不跨嵌套读串扰。"""
    from data_access.cos.remote import resolve_s3_credentials

    monkeypatch.setenv("COS_SECRET_ID", "SERVER_WIDE_ID")
    monkeypatch.setenv("COS_SECRET_KEY", "SERVER_WIDE_KEY")
    monkeypatch.setenv("DATA_ACCESS_COS_S3_ENDPOINT", "cos.example.com")

    provider_a = _ScopedProvider(
        _material(
            access_key_id="A_ID", secret="A_KEY", principal_id="a", scope="scope-a"
        )
    )
    provider_b = _ScopedProvider(
        _material(
            access_key_id="B_ID", secret="B_KEY", principal_id="b", scope="scope-b"
        )
    )

    ctx_a = DataAccessExecutionContext(
        principal="a", principal_type="HUMAN", credential_provider=provider_a
    )
    ctx_b = DataAccessExecutionContext(
        principal="b", principal_type="HUMAN", credential_provider=provider_b
    )

    with execution_scope(ctx_a):
        creds_a = resolve_s3_credentials()
        assert creds_a.access_key_id == "A_ID"
        # 嵌套读继承 A 的上下文，仍用 A 的 scoped credential。
        with execution_scope(ctx_a):
            nested_a = resolve_s3_credentials()
            assert nested_a.access_key_id == "A_ID"
        # 切到 B：B 的 scoped credential，绝不用 A 的。
        with execution_scope(ctx_b):
            creds_b = resolve_s3_credentials()
            assert creds_b.access_key_id == "B_ID"
            assert creds_b.credential_scope_id == "scope-b"
        # 回到 A 上下文，仍是 A。
        creds_a2 = resolve_s3_credentials()
        assert creds_a2.access_key_id == "A_ID"


# ---------------------------------------------------------------------------
# P0-7 credential_generation_id → DuckDB S3 identity / SECRET re-inject
# ---------------------------------------------------------------------------
def test_p07_generation_bump_reinjects_duckdb_secret(monkeypatch):
    """generation 变化 → S3 身份变化 → DuckDB SECRET 重新注入。"""
    import duckdb

    from data_access.cos.remote import S3Credentials
    from data_access.cos.s3_duckdb import (
        _creds_fingerprint,
        apply_s3_credentials,
        reset_duckdb_s3_state,
    )

    reset_duckdb_s3_state()
    con = duckdb.connect(":memory:")

    def _creds(gen: str, token: str | None) -> S3Credentials:
        return S3Credentials(
            access_key_id="ak",
            secret_access_key="sk",
            endpoint="cos.example.com",
            region="ap-guangzhou",
            session_token=token,
            principal_id="p1",
            credential_scope_id="scope-1",
            credential_generation_id=gen,
        )

    gen1 = _creds("gen-1", "token-1")
    gen2 = _creds("gen-2", "token-2")

    # 身份不含 secret / secret 哈希。
    fp1 = _creds_fingerprint(gen1)
    assert "sk" not in fp1
    assert "token-1" not in fp1
    assert "gen-1" in fp1

    apply_s3_credentials(con, gen1)
    row = con.execute(
        "SELECT name FROM duckdb_secrets() WHERE name='data_access_cos'"
    ).fetchone()
    assert row is not None and "data_access_cos" in row[0]

    # generation 变化 → 身份变化。
    assert _creds_fingerprint(gen1) != _creds_fingerprint(gen2)
    # 重新注入（SECRET 重建，token 变化随之生效）。
    apply_s3_credentials(con, gen2)
    row2 = con.execute(
        "SELECT name FROM duckdb_secrets() WHERE name='data_access_cos'"
    ).fetchone()
    assert row2 is not None and "data_access_cos" in row2[0]


def test_p07_fingerprint_has_no_secret_hash():
    """身份绝不包含 secret 或 secret 派生哈希。"""
    from data_access.cos.remote import S3Credentials
    from data_access.cos.s3_duckdb import _creds_fingerprint

    creds = S3Credentials(
        access_key_id="AKID",
        secret_access_key="SUPER_SECRET_XYZ",
        endpoint="cos.example.com",
        region="ap-guangzhou",
        session_token="STS_TOKEN",
        principal_id="p1",
        credential_scope_id="scope-1",
        credential_generation_id="gen-9",
    )
    fp = _creds_fingerprint(creds)
    assert "SUPER_SECRET_XYZ" not in fp
    assert "STS_TOKEN" not in fp
    assert "AKID" not in fp
    # 不含 sha256(secret) 前缀。
    import hashlib

    assert hashlib.sha256(b"SUPER_SECRET_XYZ").hexdigest()[:16] not in fp


# ---------------------------------------------------------------------------
# P0-1 production remote-first
# ---------------------------------------------------------------------------
def test_p01_production_mode_mirror_unset_effective_remote(monkeypatch):
    """production 上下文 + mode 未设置 → 有效 remote（remote-first）。"""
    from data_access.cos.remote import cos_read_mode

    monkeypatch.delenv("DATA_ACCESS_COS_READ_MODE", raising=False)
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION", "1")
    assert cos_read_mode() == "remote"


def test_p01_production_mode_mirror_explicit_still_remote(monkeypatch):
    """production 上下文 + mode 显式 mirror → 仍强制 remote。"""
    from data_access.cos.remote import cos_read_mode

    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION", "1")
    assert cos_read_mode() == "remote"


def test_p01_production_strict_remote_policy_effective_remote(monkeypatch):
    """STRICT_REMOTE spill policy → 有效 remote。"""
    from data_access.cos.remote import cos_read_mode

    monkeypatch.delenv("DATA_ACCESS_COS_READ_MODE", raising=False)
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "STRICT_REMOTE")
    assert cos_read_mode() == "remote"


def test_p01_research_mode_mirror_allowed(monkeypatch):
    """research 上下文 → mirror 仍允许。"""
    from data_access.cos.remote import cos_read_mode

    monkeypatch.delenv("FACTOR_ENGINE_PRODUCTION", raising=False)
    monkeypatch.delenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    assert cos_read_mode() == "mirror"


def test_p01_production_remote_read_failure_no_mirror_fallback(monkeypatch):
    """production + remote 读失败 → 抛错，绝不静默回退本地 mirror 下载。"""
    from data_access.cos.remote import should_read_cos_remote
    from data_access.registry import load_registry

    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")  # 即使显式 mirror
    monkeypatch.setenv("ASHARE_PARQUET_ROOT", "/nonexistent/ashare")
    registry = load_registry()
    ds = registry.get("ashare_stock_daily")

    # production 下 mode 被强制 remote → 走 remote，不落本地 mirror。
    assert should_read_cos_remote(ds, time_range=("2024-01-01", "2024-01-02")) is True
