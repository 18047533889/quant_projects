# -*- coding: utf-8 -*-
"""COS 远程直读路径与模式测试。"""
from __future__ import annotations

import pytest

from data_access.cos_remote import (
    authorize_s3_path,
    build_remote_paths,
    cos_read_mode,
    cos_uri_to_s3_uri,
    local_mirror_complete_for_range,
    should_read_cos_remote,
)
from data_access.exceptions import ValidationError


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
    from data_access.cos_remote import resolve_s3_credentials

    monkeypatch.delenv("COS_SECRET_ID", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    with pytest.raises(ValidationError, match="凭证"):
        resolve_s3_credentials()
