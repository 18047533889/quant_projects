# -*- coding: utf-8 -*-
"""get_object_store 工厂单测（UPSTREAM_FIX_PLAN 问题二）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.object_store import COSObjectStore, LocalObjectStore
from data_access.read.object_store_factory import get_object_store


def test_local_backend_with_root(tmp_path: Path):
    store = get_object_store(backend="local", root=tmp_path / "objs")
    assert isinstance(store, LocalObjectStore)
    assert (tmp_path / "objs").exists()


def test_local_backend_via_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATA_ACCESS_OBJECT_STORE_ROOT", str(tmp_path / "envroot"))
    store = get_object_store(backend="LOCAL")
    assert isinstance(store, LocalObjectStore)
    assert (tmp_path / "envroot").exists()


def test_local_backend_missing_root_fails_closed(monkeypatch):
    monkeypatch.delenv("DATA_ACCESS_OBJECT_STORE_ROOT", raising=False)
    with pytest.raises(ValidationError, match="root"):
        get_object_store(backend="local")


def test_cos_backend_requires_bucket():
    with pytest.raises(ValidationError, match="bucket"):
        get_object_store(backend="cos")


def test_cos_backend_builds_store():
    store = get_object_store(backend="cos", bucket="qs-cold", prefix="factor_pool/x")
    assert isinstance(store, COSObjectStore)
    assert store.bucket == "qs-cold"


def test_s3_alias_maps_to_cos_store():
    store = get_object_store(backend="s3", bucket="b")
    assert isinstance(store, COSObjectStore)


def test_unknown_backend_fails_closed():
    with pytest.raises(ValidationError, match="未知|backend|支持"):
        get_object_store(backend="oss")
