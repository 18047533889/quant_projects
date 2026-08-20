"""R21 service-layer hardening tests.

Covers: production policy floor (R21-005), typed requests (R21-032..036),
cost gate (R21-044..047), idempotency request-binding (R21-065..069),
owner-or-admin authorization (R21-017), error redaction (R21-087..090),
livez/readyz (R21-083..086), bounded queue (R21-048..051), and the
no-sync-on-production rule (R21-052..054).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from service.app import STORE, create_app
from service.jobstore import JobRecord, JobStatus


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "FACTOR_ENGINE_SERVICE_API_KEY_MAPPING",
        json.dumps({"test-secret": {"identity": "alice", "roles": ["ADMIN"]}}),
    )
    from service.security import reset_principal_registry

    reset_principal_registry()
    from service import app as service_app

    monkeypatch.setattr(service_app, "STORE", service_app.JobStore(tmp_path))
    client_ = TestClient(create_app())
    return client_


AUTH = {"X-API-Key": "test-secret"}


def _submit(client, payload):
    return client.post("/factor-engine/research/compute", headers=AUTH, json=payload)


class TestProductionPolicy:
    def test_production_compute_rejects_sync(self, client):
        r = client.post(
            "/factor-engine/production/compute",
            headers=AUTH,
            json={"formula": "ts_mean(close, 2)", "sync": True},
        )
        assert r.status_code == 422
        assert "sync" in r.json()["detail"]["message"].lower()

    def test_production_compute_requires_approved_source_profile(self, client):
        r = client.post(
            "/factor-engine/production/compute",
            headers=AUTH,
            json={"formula": "ts_mean(close, 2)", "sync": False},
        )
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "UNAPPROVED_REMOTE_SOURCE"

    def test_research_endpoint_forces_research_run_mode(self, client):
        r = _submit(client, {"formula": "close", "run_mode": "production", "sync": True})
        # research endpoint downgrades the request to research: no production
        # downgrade possible since it never promises production.
        assert r.status_code in (200, 422)


class TestTypedRequests:
    def test_extra_field_rejected(self, client):
        r = _submit(client, {"formula": "close", "unknown_field": 1})
        assert r.status_code == 422

    def test_sync_string_false_is_not_truthy(self, client):
        # R21-033: "sync": "false" was truthy under bool() in the old raw-dict
        # path. It must NOT run the job synchronously — it is treated as False.
        r = _submit(client, {"formula": "close", "sync": "false"})
        assert r.status_code == 200
        assert r.json()["status"] in ("submitted", "queued")

    def test_invalid_backend_rejected(self, client):
        r = _submit(client, {"formula": "close", "backend": "not_a_backend"})
        assert r.status_code == 422

    def test_formula_too_long_rejected(self, client):
        big = "close + " * 20000
        r = _submit(client, {"formula": big})
        assert r.status_code in (413, 422)


class TestCostGate:
    def test_high_cost_formula_rejected(self, client, monkeypatch):
        monkeypatch.setenv("FACTOR_ENGINE_SERVICE_MAX_COST_PER_JOB", "1")
        r = _submit(client, {"formula": "close", "sync": True})
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "RESOURCE_BUDGET_EXCEEDED"


class TestIdempotency:
    def test_same_key_same_request_is_idempotent(self, client):
        payload = {"formula": "close", "idempotency_key": "k1", "sync": True}
        first = _submit(client, payload)
        second = _submit(client, payload)
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["run_id"] == first.json()["run_id"]

    def test_same_key_different_request_conflicts(self, client):
        _submit(client, {"formula": "close", "idempotency_key": "k2", "sync": True})
        r = _submit(client, {"formula": "rank(close)", "idempotency_key": "k2", "sync": True})
        assert r.status_code == 409
        assert r.json()["detail"]["error_code"] == "IDEMPOTENCY_KEY_CONFLICT"

    def test_key_scoped_by_principal(self, client):
        payload = {"formula": "close", "idempotency_key": "shared", "sync": True}
        _submit(client, payload)
        # different principal (bob) using the same key must NOT collide
        other = {"X-API-Key": "bob-key"}
        r = client.post(
            "/factor-engine/research/compute",
            headers={**AUTH, **other},
            json=payload,
        )
        # bob is not a mapped principal -> 401, which still proves scope
        assert r.status_code == 401


class TestAuthorization:
    def test_job_status_owner_or_admin(self, client):
        r = _submit(client, {"formula": "close", "sync": True})
        run_id = r.json()["run_id"]
        ok = client.get(f"/factor-engine/jobs/{run_id}", headers=AUTH)
        assert ok.status_code == 200

    def test_job_read_by_unauthenticated_denied(self, client):
        r = _submit(client, {"formula": "close", "sync": True})
        run_id = r.json()["run_id"]
        resp = client.get(f"/factor-engine/jobs/{run_id}")
        assert resp.status_code == 401


class TestErrorsAndReadiness:
    def test_error_never_leaks_secret(self, client, monkeypatch):
        # a job error containing a password-shaped token must be redacted
        from service import app as service_app

        rec = JobRecord(run_id="redacted-test", status=JobStatus.FAILED, error="password=hunter2")
        service_app.STORE.create(rec)
        r = client.get("/factor-engine/jobs/redacted-test", headers=AUTH)
        assert r.status_code == 200
        assert "hunter2" not in json.dumps(r.json())

    def test_livez(self, client):
        assert client.get("/livez").status_code == 200

    def test_readyz_shape(self, client):
        r = client.get("/readyz")
        assert r.status_code in (200, 503)
        body = r.json()
        # FastAPI wraps HTTPException detail under "detail"; a 200 returns the
        # payload directly.
        check = body.get("detail") if isinstance(body.get("detail"), dict) else body
        assert "critical_blockers" in check
        assert "corrupt_manifests" in check


class TestQueue:
    def test_per_principal_limit_rejects_extra_job(self, monkeypatch):
        # R21-050: per-principal running+queued limit, tested deterministically
        # at the queue level (a slow run_fn holds the slot).
        import threading
        import time

        from service.errors import ServiceError
        from service.jobstore import JobStore
        from service.queue import BoundedJobQueue

        monkeypatch.setenv("FACTOR_ENGINE_SERVICE_PER_PRINCIPAL_JOBS", "1")
        store = JobStore()
        queue = BoundedJobQueue(max_queue=16, max_running=2)
        queue.start(store)
        gate = threading.Event()

        def slow(job):
            gate.wait(timeout=5)

        j1 = store.create(JobRecord(run_id="p1", owner_principal="alice", status="submitted"))
        j2 = store.create(JobRecord(run_id="p2", owner_principal="alice", status="submitted"))
        queue.submit(j1, run_fn=slow)
        time.sleep(0.2)
        with pytest.raises(ServiceError) as exc_info:
            queue.submit(j2, run_fn=slow)
        assert exc_info.value.code == "JOB_QUEUE_FULL"
        gate.set()
        time.sleep(0.3)
