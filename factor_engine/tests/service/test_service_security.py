from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_service_requires_configured_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path))
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_API_KEY", "test-secret")
    from service.app import create_app

    client = TestClient(create_app())
    assert client.get("/factor-engine/operators").status_code == 401
    response = client.get(
        "/factor-engine/operators", headers={"X-API-Key": "test-secret"}
    )
    assert response.status_code == 200
    assert "field" in response.json()["operators"]


def test_service_rejects_config_path_escape(monkeypatch, tmp_path):
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path / "service"))
    monkeypatch.setenv("FACTOR_ENGINE_CONFIG_ROOT", str(tmp_path / "configs"))
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_API_KEY", "test-secret")
    from service.app import STORE, create_app

    client = TestClient(create_app())
    response = client.post(
        "/factor-engine/jobs/compute",
        headers={"X-API-Key": "test-secret"},
        json={"config_path": "../secret.yaml", "sync": True},
    )
    assert response.status_code == 200
    job = STORE.get(response.json()["run_id"])
    assert job is not None and job.status == "failed"
    assert "outside FACTOR_ENGINE_CONFIG_ROOT" in str(job.error)
    assert "traceback" not in job.to_public()["artifacts"]


def test_service_idempotency_and_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path))
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_API_KEY", "test-secret")
    from service import app as service_app

    monkeypatch.setattr(service_app, "STORE", service_app.JobStore(tmp_path))
    client = TestClient(service_app.create_app())
    payload = {"formula": "close", "idempotency_key": "same", "sync": True}
    headers = {"X-API-Key": "test-secret", "X-Request-Identity": "alice"}
    first = client.post("/factor-engine/research/compute", headers=headers, json=payload)
    second = client.post("/factor-engine/research/compute", headers=headers, json=payload)
    assert first.status_code == 200
    assert second.json()["run_id"] == first.json()["run_id"]
    assert second.json()["idempotent"] is True
    job = service_app.STORE.get(first.json()["run_id"])
    assert job is not None and job.requested_by == "alice"


def test_production_endpoint_rejects_research_only_formula(monkeypatch, tmp_path):
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path))
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_API_KEY", "test-secret")
    from service.app import create_app

    client = TestClient(create_app())
    response = client.post(
        "/factor-engine/production/compute",
        headers={"X-API-Key": "test-secret"},
        json={"formula": "shuffle(close, 1)"},
    )
    assert response.status_code == 422


def test_job_store_restores_manifests(tmp_path):
    from service.app import JobRecord, JobStore

    first = JobStore(tmp_path)
    first.create(JobRecord(run_id="restored", status="succeeded"))
    restored = JobStore(tmp_path).get("restored")
    assert restored is not None
    assert restored.status == "succeeded"


def test_validate_spec_uses_requested_surface():
    from service.app import validate_spec

    result = validate_spec({"formula": "rank(close)", "surface": "daily"})
    assert result["ok"] is True
    assert result["checked"]["dsl"]["ok"] is True
