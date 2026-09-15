"""Default HTTP dispatch contracts only; not a real-data durable acceptance."""
import json
from types import SimpleNamespace as NS
import pytest
from factor_engine.service import app
from factor_engine.runtime.default_engine import ApprovedDeploymentProfile
from factor_engine.runtime.default_execution_policy import ExecutionPurpose, resolve_default_policy
from factor_engine.service.security import Principal
from factor_engine.runtime.default_engine import iter_parsed_factor_definitions
from factor_engine.runtime.finite_manifest import (
    FiniteFactorManifest, ManifestLimitExceeded, RejectedFactorDefinition,
)
from factor_engine.tests.runtime.test_r3_execution_purpose import _profile


@pytest.fixture
def configured(monkeypatch, tmp_path):
    deployment = ApprovedDeploymentProfile.from_mapping(_profile(tmp_path))
    monkeypatch.setattr("factor_engine.runtime.default_engine.load_deployment_profile", lambda: deployment)
    monkeypatch.setattr("factor_engine.runtime.operator_snapshot.get_cached_runtime_operator_snapshot",
                        lambda: {"catalog_digest": "catalog"})
    return deployment


def test_default_request_uses_approved_scope_and_no_user_performance_parameters(configured):
    execution, digest = app._validate_default_durable_request(
        {"factors": [{"name": "x", "formula": "close + open"}]},
        endpoint_policy=app.EndpointExecutionPolicy.PRODUCTION)
    assert execution["deployment_digest"] == configured.digest
    assert execution["execution_purpose"]["purpose"] == "research_compute"
    policy = resolve_default_policy(profile=configured.to_dict().get("execution"))
    assert execution["default_policy_id"] == policy.policy_id
    assert execution["default_policy_digest"] == policy.digest
    assert execution["timeout_seconds"] == policy.job_max_seconds
    assert execution["artifact_root"] == configured.to_dict()["artifact_root"]
    assert execution["profile_id"] == configured.to_dict()["profile_id"]
    assert execution["approval_id"] == configured.to_dict()["approval_id"]
    assert len(digest) == 64
    assert "backend" not in execution and "data_source" not in execution


@pytest.mark.parametrize("extra", [{"backend": "pandas"}, {"run_mode": "research"},
                                   {"workers": 999}, {"profile_path": "/unapproved"}])
def test_default_request_rejects_hidden_scope_or_tuning_overrides(configured, extra):
    with pytest.raises(app.ServiceError):
        app._validate_default_durable_request(
            {"factors": [{"name": "x", "formula": "close"}], **extra},
            endpoint_policy=app.EndpointExecutionPolicy.PRODUCTION)


def test_shared_definition_adapter_preserves_mixed_rejections_and_digests():
    entries = [
        {"name": "good", "formula": "close + open"},
        {"name": "syntax", "formula": "close +"},
        {"name": "unknown", "formula": "not_an_operator(close)"},
    ]

    parsed = tuple(iter_parsed_factor_definitions(entries))

    assert parsed[0].name == "good"
    assert isinstance(parsed[1], RejectedFactorDefinition)
    assert isinstance(parsed[2], RejectedFactorDefinition)
    assert parsed[1].error_code == "INVALID_DSL"
    assert parsed[2].error_code == "OPERATOR_UNKNOWN"
    for entry, rejected in zip(entries[1:], parsed[1:]):
        encoded = entry["formula"].encode("utf-8")
        assert rejected.definition_bytes == len(encoded)
        assert rejected.definition_digest == app.hashlib.sha256(encoded).hexdigest()


def test_rejected_definitions_are_native_manifest_rows_with_duplicate_and_quota(tmp_path):
    entries = [
        {"name": "syntax", "formula": "close +"},
        {"name": "unknown", "formula": "not_an_operator(close)"},
    ]
    parsed = tuple(iter_parsed_factor_definitions(entries))
    manifest = FiniteFactorManifest.ingest(parsed, tmp_path / "manifest.sqlite3")
    records = list(manifest.records(limit=10))
    manifest.close()

    assert [(record.name, record.valid, record.error_code) for record in records] == [
        ("syntax", False, "INVALID_DSL"),
        ("unknown", False, "OPERATOR_UNKNOWN"),
    ]
    assert [record.definition_digest for record in records] == [
        item.definition_digest for item in parsed
    ]

    duplicate = tuple(iter_parsed_factor_definitions([
        {"name": "same", "formula": "close +"},
        {"name": "same", "formula": "not_an_operator(close)"},
    ]))
    duplicate_manifest = FiniteFactorManifest.ingest(
        duplicate, tmp_path / "duplicates.sqlite3"
    )
    assert [record.error_code for record in duplicate_manifest.records(limit=10)] == [
        "DUPLICATE_FACTOR_NAME", "DUPLICATE_FACTOR_NAME",
    ]
    duplicate_manifest.close()

    with pytest.raises(ManifestLimitExceeded):
        FiniteFactorManifest.ingest(parsed, tmp_path / "quota.sqlite3", max_factors=1)


def test_default_dispatch_calls_same_facade_and_preserves_partial_receipt(configured, monkeypatch):
    execution, digest = app._validate_default_durable_request(
        {"factors": [{"name": "x", "formula": "close + open"}]},
        endpoint_policy=app.EndpointExecutionPolicy.PRODUCTION)
    seen = []
    receipt = {"counts": {"SUCCEEDED": 1, "REJECTED": 1}}
    class Engine:
        deployment = configured
        policy = resolve_default_policy(profile=configured.to_dict().get("execution"))
        execution_purpose = configured.execution_purpose
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def run_many(self, factors, **kwargs):
            seen.append((factors, kwargs))
            return receipt
    monkeypatch.setattr("factor_engine.runtime.default_engine.get_engine", Engine)
    job = NS(job_type="compute", endpoint_policy="production", run_id="unit", request_digest=digest)
    result = app._dispatch_execution(job, execution)
    assert seen[0][0][0].name == "x" and seen[0][0][0].surface == "all"
    assert set(seen[0][1]) == {"cancellation_token"}
    assert app._summarize_result(result, job) == {"mode": "default_durable", "receipt": receipt}
    execution["factors"][0]["formula"] = "close - open"
    with pytest.raises(app.ServiceError, match="changed"):
        app._dispatch_execution(job, execution)
    assert len(seen) == 1



def test_default_dispatch_rejects_resolved_policy_or_purpose_drift(configured, monkeypatch):
    execution, digest = app._validate_default_durable_request(
        {"factors": [{"name": "x", "formula": "close"}]},
        endpoint_policy=app.EndpointExecutionPolicy.PRODUCTION)

    class DriftedEngine:
        deployment = configured
        policy = resolve_default_policy(
            {"compute_max_seconds": 3599}, configured.to_dict().get("execution"))
        execution_purpose = configured.execution_purpose
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr("factor_engine.runtime.default_engine.get_engine", DriftedEngine)
    job = NS(job_type="compute", endpoint_policy="production", run_id="unit",
             request_digest=digest)
    with pytest.raises(app.ServiceError, match="policy changed"):
        app._dispatch_execution(job, execution)

    class PurposeDriftedEngine:
        deployment = configured
        policy = resolve_default_policy(profile=configured.to_dict().get("execution"))
        execution_purpose = ExecutionPurpose(purpose="production_compute")
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr("factor_engine.runtime.default_engine.get_engine", PurposeDriftedEngine)
    with pytest.raises(app.ServiceError, match="purpose changed"):
        app._dispatch_execution(job, execution)


def test_owner_scope_includes_tenant_and_project_with_explicit_admin_bypass():
    job = app.JobRecord(run_id="job", owner_principal="same", tenant="tenant-a",
                        project="project-a")
    owner = Principal(identity="same", tenant="tenant-a", project="project-a")
    other_tenant = Principal(identity="same", tenant="tenant-b", project="project-a")
    other_project = Principal(identity="same", tenant="tenant-a", project="project-b")
    admin = Principal(identity="admin", roles=("ADMIN",), tenant="tenant-b")

    assert app._principal_can_access_job(owner, job)
    assert not app._principal_can_access_job(other_tenant, job)
    assert not app._principal_can_access_job(other_project, job)
    assert app._principal_can_access_job(admin, job)


def test_aborted_receipt_is_preserved_only_from_approved_bounded_root(configured, tmp_path):
    execution, _ = app._validate_default_durable_request(
        {"factors": [{"name": "x", "formula": "close"}]},
        endpoint_policy=app.EndpointExecutionPolicy.PRODUCTION)
    approved_root = tmp_path / "artifacts"
    approved_root.mkdir(exist_ok=True)
    execution["artifact_root"] = str(approved_root)
    job = app.JobRecord(run_id="job", request={"execution": execution})
    receipt_path = approved_root / "run" / "receipt.json"
    run_identity = {
        "profile_id": execution["profile_id"],
        "approval_id": execution["approval_id"],
    }
    run_identity_digest = app.hashlib.sha256(json.dumps(
        run_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps({
        "schema_version": "factor_engine.artifact_receipt.v2",
        "status": "ABORTED",
        "counts": {"FAILED": 1},
        "deployment_digest": execution["deployment_digest"],
        "run_id": "run",
        "run_identity": run_identity,
        "run_identity_digest": run_identity_digest,
        "policy_digest": execution["default_policy_digest"],
        "execution_purpose": execution["execution_purpose"],
        "primary_error": "must not be copied",
    }))
    exc = RuntimeError("failed")
    exc.receipt_path = str(receipt_path)
    exc.service_job_run_id = job.run_id
    exc.durable_run_id = "run"
    exc.run_identity_digest = run_identity_digest

    app._preserve_default_failure_receipt(job, exc)

    assert job.result_summary["receipt"]["status"] == "ABORTED"
    assert job.result_summary["receipt"]["counts"] == {"FAILED": 1}
    assert "primary_error" not in job.result_summary["receipt"]
    wrong_run = app.JobRecord(run_id="wrong-run", request={"execution": execution})
    exc.service_job_run_id = wrong_run.run_id
    exc.durable_run_id = "different-durable-run"
    exc.receipt_path = str(receipt_path)
    app._preserve_default_failure_receipt(wrong_run, exc)
    assert wrong_run.result_summary == {}

    exc.durable_run_id = "run"


    outside = tmp_path / "receipt.json"
    outside.write_text(receipt_path.read_text())
    rejected = app.JobRecord(run_id="rejected", request={"execution": execution})
    exc.receipt_path = str(outside)
    app._preserve_default_failure_receipt(rejected, exc)
    assert rejected.result_summary == {}

    exc.service_job_run_id = rejected.run_id
    oversized = approved_root / "large" / "receipt.json"
    oversized.parent.mkdir()
    oversized.write_bytes(b" " * (app._MAX_FAILURE_RECEIPT_BYTES + 1))
    exc.receipt_path = str(oversized)
    app._preserve_default_failure_receipt(rejected, exc)
    assert rejected.result_summary == {}


def test_default_retry_is_new_attempt_with_same_bound_identity(configured, monkeypatch, tmp_path):
    execution, digest = app._validate_default_durable_request(
        {"factors": [{"name": "x", "formula": "close"}]},
        endpoint_policy=app.EndpointExecutionPolicy.PRODUCTION)
    store = app.JobStore(tmp_path / "jobs")

    class Queue:
        timeout_default = 1.0
        submitted = []
        def submit(self, job, *, run_fn):
            self.submitted.append((job, run_fn))

    queue = Queue()
    original = app.JobRecord(
        run_id="original",
        owner_principal="owner",
        tenant="tenant",
        project="project",
        job_type="compute",
        endpoint_policy="production",
        request={"execution": execution},
        request_digest=digest,
        execution_policy_digest=execution["default_policy_digest"],
        policy_id=execution["default_policy_id"],
        policy_digest=execution["default_policy_digest"],
        timeout_seconds=execution["timeout_seconds"],
        status=app.JobStatus.FAILED,
        deadline_at="2000-01-01T00:00:00+00:00",
    )
    store.create(original)
    monkeypatch.setattr(app, "STORE", store)
    monkeypatch.setattr(app, "QUEUE", queue)

    retry = app._retry_existing_job(original)

    assert retry.run_id != original.run_id
    assert retry.parent_run_id == original.run_id
    assert retry.root_operation_id == original.run_id
    assert retry.attempt == original.attempt + 1
    assert retry.attempt_id == original.attempt_id + 1
    assert retry.request == original.request
    assert retry.request_digest == original.request_digest
    assert retry.policy_digest == original.policy_digest
    assert retry.timeout_seconds == original.timeout_seconds
    assert retry.deadline_at != original.deadline_at
    assert queue.submitted[0][0] is retry
    app._drop_job_cancellation_token(retry.run_id)


def test_default_route_auth_and_missing_profile_fail_before_queue(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from factor_engine.service.security import reset_principal_registry

    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path))
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_API_KEY_MAPPING",
        json.dumps({"unit-key": {"identity": "unit", "roles": ["ADMIN"]}}))
    monkeypatch.delenv("FACTOR_ENGINE_V2_PROFILE", raising=False)
    reset_principal_registry()
    try:
        client = TestClient(app.create_app())
        payload = {"factors": [{"name": "x", "formula": "close"}]}
        rejected = client.post("/factor-engine/default/compute", json=payload)
        assert rejected.status_code in (401, 403)
        missing = client.post("/factor-engine/default/compute", json=payload,
                              headers={"X-API-Key": "unit-key"})
        assert missing.status_code == 503
        assert missing.json()["detail"]["code"] == "DEPLOYMENT_CONFIGURATION_REQUIRED"
    finally:
        reset_principal_registry()
