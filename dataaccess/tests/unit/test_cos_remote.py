# -*- coding: utf-8 -*-
"""COS 远程直读路径与模式测试。"""
from __future__ import annotations

import os

import pytest

from data_access.cos.remote import (
    authorize_s3_path,
    build_remote_paths,
    cos_read_mode,
    cos_uri_to_s3_uri,
    local_mirror_complete_for_range,
    should_read_cos_remote,
)
from data_access.core.exceptions import ValidationError


def test_cos_uri_to_s3_uri():
    assert cos_uri_to_s3_uri("cos://qs-cold/foo/bar") == "s3://qs-cold/foo/bar"
    assert cos_uri_to_s3_uri("s3://qs-cold/foo") == "s3://qs-cold/foo"


def test_build_remote_daily_paths_with_time_range():
    paths = build_remote_paths(
        "ashare_stock_daily",
        time_range=("2024-01-01", "2024-01-02"),
    )
    assert len(paths) == 2
    assert paths[0].startswith("s3://")
    assert paths[0].endswith("2024-01-01.parquet")
    for path in paths:
        authorize_s3_path(path)


def test_build_remote_daily_glob_without_time_range():
    paths = build_remote_paths("ashare_stock_daily", time_range=None)
    assert len(paths) == 1
    assert "*.parquet" in paths[0]


def test_cos_read_mode_invalid(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "bogus")
    with pytest.raises(ValidationError, match="DATA_ACCESS_COS_READ_MODE"):
        cos_read_mode()


def test_should_read_cos_remote_modes(monkeypatch, tmp_path):
    from data_access.registry import load_registry

    monkeypatch.setenv("ASHARE_PARQUET_ROOT", str(tmp_path / "ashare"))
    registry = load_registry()
    ds = registry.get("ashare_stock_daily")

    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    assert should_read_cos_remote(ds, time_range=("2024-01-01", "2024-01-02")) is False

    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "remote")
    assert should_read_cos_remote(ds, time_range=("2024-01-01", "2024-01-02")) is True

    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "auto")
    assert (
        local_mirror_complete_for_range(
            "ashare_stock_daily",
            time_range=("2024-01-01", "2024-01-02"),
        )
        is False
    )
    assert should_read_cos_remote(ds, time_range=("2024-01-01", "2024-01-02")) is True


def test_resolve_s3_credentials_missing(monkeypatch):
    """env 与 coscli 配置都不存在时必须 fail closed。"""
    from data_access.cos.remote import resolve_s3_credentials

    monkeypatch.delenv("COS_SECRET_ID", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("DATA_ACCESS_COS_YAML", raising=False)
    monkeypatch.setenv("DATA_ACCESS_COS_YAML", "/nonexistent/cos.yaml")
    with pytest.raises(ValidationError, match="凭证"):
        resolve_s3_credentials()


def test_resolve_s3_credentials_missing_without_cos_yaml(monkeypatch):
    """缺少 env 且 cos.yaml 也不存在时仍须抛错（fail closed）。"""
    from data_access.cos.remote import resolve_s3_credentials

    monkeypatch.delenv("COS_SECRET_ID", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("DATA_ACCESS_COS_YAML", raising=False)
    monkeypatch.setenv("DATA_ACCESS_COS_YAML", "/nonexistent/cos.yaml")
    with pytest.raises(ValidationError, match="凭证"):
        resolve_s3_credentials()


def test_resolve_s3_credentials_falls_back_to_cos_yaml(monkeypatch, tmp_path):
    """research 显式 opt-in 时回退到 coscli 配置的 secretid/secretkey。

    R24 P0-S1 §3.4：读 ``~/.cos.yaml`` 必须 ``DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE=1``
    显式 opt-in（production/strict 永远禁止）。
    """
    from data_access.cos.remote import resolve_s3_credentials

    monkeypatch.delenv("COS_SECRET_ID", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.setenv("DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE", "1")
    cfg = tmp_path / "cos.yaml"
    cfg.write_text(
        "cos:\n  base:\n    secretid: 'TEST_ID'\n    secretkey: 'TEST_KEY'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ACCESS_COS_YAML", str(cfg))
    creds = resolve_s3_credentials()
    assert creds.access_key_id == "TEST_ID"
    assert creds.secret_access_key == "TEST_KEY"
    # endpoint 由 region 推断
    assert creds.endpoint.endswith(".myqcloud.com")


def test_resolve_s3_credentials_production_never_parses_cos_yaml(monkeypatch, tmp_path):
    """R24 P0-S1 / T-S03：production 下 ~/.cos.yaml 存在也绝不解析、绝不注入 env。

    即使机上有更宽的 base credential，DataAccess 也不能绕过 ``clean-cos-ro``
    的服务器权限分级自动读取。
    """
    from data_access.cos.remote import resolve_s3_credentials

    monkeypatch.delenv("COS_SECRET_ID", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    cfg = tmp_path / "cos.yaml"
    cfg.write_text(
        "cos:\n  base:\n    secretid: 'PROD_WIDE_ID'\n    secretkey: 'PROD_WIDE_KEY'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ACCESS_COS_YAML", str(cfg))
    # 显式 opt-in 也不能在 production 生效（allow_coscli_config_parse 恒 False）。
    monkeypatch.setenv("DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE", "1")
    with pytest.raises(ValidationError, match="凭证"):
        resolve_s3_credentials()
    # 绝不把 secret 写进 os.environ（T-S03）
    assert os.environ.get("COS_SECRET_ID") is None
    assert os.environ.get("COS_SECRET_KEY") is None
