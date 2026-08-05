# -*- coding: utf-8 -*-
"""data_access HTTP 读数服务单元测试。"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from data_access.read.read_contract import (
    DataSnapshot,
    ReadLineage,
    ReadResult,
    ReadStats,
)
from data_access.service.app import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("data_access.service.app.API_KEY", "test-key")
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    return TestClient(app)


def _fake_read_result() -> ReadResult:
    table = pa.table(
        {
            "TradeDate": ["2024-01-02"],
            "Symbol": ["000001.SZ"],
            "Close": [10.5],
        }
    )
    snap = DataSnapshot(
        snapshot_id="snap123",
        dataset="ashare_stock_daily",
        registry_hash="reg",
        schema_hash="sch",
        file_manifest_hash="mf",
        files=(),
        created_at=datetime.now(timezone.utc),
    )
    return ReadResult(
        table=table,
        snapshot=snap,
        stats=ReadStats(rows=1, bytes=table.nbytes, elapsed_ms=12.5, paths=("/tmp/x.parquet",)),
        lineage=ReadLineage(dataset="ashare_stock_daily"),
    )


def _mock_store():
    store = MagicMock()
    ds = MagicMock()
    ds.access_mode = "published"
    ds.time_column = "TradeDate"
    ds.instrument_column = "Symbol"
    ds.kind = "static"
    ds.params_schema = {}
    store.registry.names.return_value = ["ashare_stock_daily"]
    store.registry.get.return_value = ds
    store.read_result.return_value = _fake_read_result()
    return store


def test_health_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_read_requires_api_key(client):
    resp = client.post(
        "/v1/read",
        json={
            "dataset": "ashare_stock_daily",
            "columns": ["Close"],
            "time_range": ["2024-01-01", "2024-01-31"],
        },
    )
    assert resp.status_code == 401


@patch("data_access.service.app.get_store", return_value=_mock_store())
def test_list_datasets(mock_get_store, client):
    resp = client.get("/v1/datasets", headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["name"] == "ashare_stock_daily"


@patch("data_access.service.app.get_store", return_value=_mock_store())
def test_read_parquet(mock_get_store, client):
    resp = client.post(
        "/v1/read",
        headers={"X-API-Key": "test-key"},
        json={
            "dataset": "ashare_stock_daily",
            "columns": ["TradeDate", "Symbol", "Close"],
            "time_range": ["2024-01-01", "2024-01-31"],
            "format": "parquet",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["X-Data-Snapshot-Id"] == "snap123"
    table = pa.parquet.read_table(io.BytesIO(resp.content))
    assert table.num_rows == 1
    mock_get_store.return_value.read_result.assert_called_once()


@patch("data_access.service.app.get_store", return_value=_mock_store())
def test_read_json(mock_get_store, client):
    resp = client.post(
        "/v1/read",
        headers={"X-API-Key": "test-key"},
        json={
            "dataset": "ashare_stock_daily",
            "columns": ["TradeDate", "Symbol", "Close"],
            "time_range": ["2024-01-01", "2024-01-31"],
            "format": "json",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["snapshot_id"] == "snap123"
    assert len(body["data"]) == 1


def test_production_requires_api_key_when_explicitly_empty(monkeypatch):
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    monkeypatch.setattr("data_access.service.app.API_KEY", "")
    monkeypatch.delenv("DATA_ACCESS_API_ALLOW_OPEN", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    local_app = create_app(
        ServiceSettings(api_key="", production_mode=True, allow_open=False)
    )
    client = TestClient(local_app)
    resp = client.get("/v1/datasets")
    assert resp.status_code == 503
    assert "DATA_ACCESS_API_KEY" in resp.json()["detail"]


def test_no_default_team_api_key(monkeypatch):
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    monkeypatch.setattr("data_access.service.app.API_KEY", "")
    monkeypatch.delenv("DATA_ACCESS_API_KEY", raising=False)
    monkeypatch.delenv("DATA_ACCESS_API_ALLOW_OPEN", raising=False)
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    settings = ServiceSettings.from_env()
    assert settings.api_key == ""
    local_app = create_app(settings)
    with patch("data_access.service.app.get_store", return_value=_mock_store()):
        client = TestClient(local_app)
        assert client.get("/v1/datasets").status_code == 200


def test_explicit_api_key_is_required_when_configured(monkeypatch):
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    monkeypatch.setattr("data_access.service.app.API_KEY", "")
    monkeypatch.setenv("DATA_ACCESS_API_KEY", "test-only-secret")
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    local_app = create_app(ServiceSettings.from_env())
    with patch("data_access.service.app.get_store", return_value=_mock_store()):
        client = TestClient(local_app)
        assert client.get("/v1/datasets").status_code == 401
        ok = client.get("/v1/datasets", headers={"X-API-Key": "test-only-secret"})
        assert ok.status_code == 200


def test_production_allow_open_escape(monkeypatch):
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    monkeypatch.setattr("data_access.service.app.API_KEY", "")
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    local_app = create_app(
        ServiceSettings(api_key="", production_mode=True, allow_open=True)
    )
    with patch("data_access.service.app.get_store", return_value=_mock_store()):
        client = TestClient(local_app)
        resp = client.get("/v1/datasets")
        assert resp.status_code == 200
